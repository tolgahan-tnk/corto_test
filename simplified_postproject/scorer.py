"""Per-object multi-metric scorer with two-pass alignment.

Correct scoring flow per frame
==============================
1. Load synthetic render + Phobos/Mars ID masks from CORTO output
2. Build combined ID mask: Mars=2 first, then Phobos=1 overwrites (camera-closer priority)
3. Center-crop synthetic to PDS dimensions (CORTO renders at 1024x1024)
4. **Bit equalization**: normalize both PDS and synthetic to [0, 1]
5. **Pass 1 — Global alignment**: phase cross-correlation on full image
   - Warp synthetic + mask onto PDS reference frame
   - Also apply same shift to RAW (un-normalized) synthetic for brightness
6. **Derive object regions** from the warped mask:
   - phobos = mask == 1, mars = mask == 2
7. **Brightness ratio**: aligned RAW pixels, pre-normalization (sun_scaler feedback)
8. **Pass 2 — Per-object masked refinement**:
   - Crop to symmetric bounding box of the object in both syn and pds
   - Phase correlation on the cropped pair (max 20% shift)
   - Apply sub-pixel warp to the synthetic crop
9. **Score** the aligned cropped pair with 6 metrics
10. **Debug save**: all stages when DEBUG_SAVE_FULL=True
11. **DN ratio** cross-object brightness score
12. **Combine** all per-object metrics → single scalar for CMA-ES
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from config import (
    ALIGN_MAX_SHIFT_FRAC,
    BRIGHTNESS_WEIGHT,
    DEBUG_DIR,
    DEBUG_SAVE_FULL,
    DN_RATIO_WEIGHT,
    EMD_BINS,
    FRAME_WEIGHTS,
    MASK_WEIGHTS,
    METRIC_WEIGHTS,
    MIN_OBJECT_FRACTION,
    MIN_PIXELS,
    PDS_IMAGES,
    SAVE_DEBUG_IMAGES,
)
from scipy.signal import convolve2d
from scipy.stats import wasserstein_distance
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize as _sp_minimize
from skimage.metrics import structural_similarity as _ssim_fn
from skimage.registration import phase_cross_correlation

logger = logging.getLogger(__name__)

# -- LPIPS model cache (loaded once per process) -----------------------------
_LPIPS_MODEL = None
_LPIPS_OK: bool | None = None  # None=untried, True=loaded, False=unavailable

class Scorer:
    """Multi-metric scorer with two-pass alignment and debug saving."""

    def __init__(self, pds_refs: list[np.ndarray]) -> None:
        self._pds = pds_refs
        self._eval_count = 0
        # Last eval metric breakdown (populated after each score_all)
        self.last_detail: dict = {}
        # Pass 2 alignment cache: (frame_idx, body) → history
        self._p2_shift_history: dict[tuple[int, str], list[tuple[int, int]]] = {}
        self._p2_best_M: dict[tuple[int, str], tuple[np.ndarray, bool]] = {}
        self._p2_locked: dict[tuple[int, str], tuple[np.ndarray, bool]] = {}

    def score_all(self, paths: dict) -> float:
        """Score all frames with PER-BODY GLOBAL AGGREGATION.

        Mars and Phobos contribute according to MASK_WEIGHTS regardless of how
        many frames they appear in. Per-frame body metric sums are pooled into
        per-body averages (FRAME_WEIGHTS as inner weights), then MASK_WEIGHTS
        body-weighted summed.
        """
        self._eval_count += 1
        frame_details: list[dict] = []
        all_brightness: list[dict[str, float]] = []
        all_dn_ratios: list[float] = []

        # body_metric_pool[body] = list of (frame_weight, body_weighted_metric_sum)
        body_metric_pool: dict[str, list[tuple[float, float]]] = {
            b: [] for b in MASK_WEIGHTS
        }

        for i in range(len(paths["img"])):
            syn_raw = self._load(paths["img"][i]).astype(np.float64)
            pds_raw = self._pds[i].astype(np.float64)

            # Combined ID mask from CORTO: 1=Phobos (mask_ID_1), 2=Mars (mask_ID_2).
            # Using true per-body ID masks instead of synthetic-derived regions
            # so Mars's shadowed/dark pixels are not dropped from scoring.
            # Order: Mars FIRST, then Phobos OVERWRITES — Phobos is physically
            # closer to the camera, so overlap pixels belong to Phobos.
            mask_phobos_raw = self._load(paths["mask"][i])
            if mask_phobos_raw.ndim == 3:
                mask_phobos_raw = mask_phobos_raw[..., 0]
            id_mask_raw = np.zeros(mask_phobos_raw.shape[:2], dtype=np.uint8)
            mars_mask_path = paths.get("mask_mars", [None] * len(paths["img"]))[i]
            if mars_mask_path and Path(mars_mask_path).exists():
                mask_mars_raw = self._load(mars_mask_path)
                if mask_mars_raw.ndim == 3:
                    mask_mars_raw = mask_mars_raw[..., 0]
                id_mask_raw[mask_mars_raw > 0] = 2
            else:
                logger.warning(
                    "  mask_ID_2 missing for frame %d (%s) -- Mars region empty",
                    i, mars_mask_path,
                )
            id_mask_raw[mask_phobos_raw > 0] = 1  # Phobos overwrites Mars at overlap

            # Extract PDS image name for debug labeling
            pds_name = Path(PDS_IMAGES[i]["file"]).stem if i < len(PDS_IMAGES) else f"frame{i}"
            is_calibrated = PDS_IMAGES[i].get("calibrated", False) if i < len(PDS_IMAGES) else False
            camera = PDS_IMAGES[i].get("camera", "hrsc") if i < len(PDS_IMAGES) else "hrsc"

            # Step 1: Center-crop synthetic + ID mask to PDS size
            syn, id_mask = self._center_crop(syn_raw, id_mask_raw, pds_raw.shape)

            # Step 2: Bit equalization — normalize both to [0, 1]
            pds_norm, syn_norm = self._bit_equalize(pds_raw, syn)

            # Step 3: Global alignment (pass 1) — SIFT+RANSAC on normalized
            # Returns 2x3 affine matrix M (translation + rotation + scale)
            syn_a, id_mask_a, M_global, pds_uint8, syn_uint8 = self._align_global(
                syn_norm, id_mask, pds_norm
            )

            # Step 3.5: Apply Pass 1 transform to RAW (un-normalized) synthetic
            # so brightness ratio compares aligned raw pixels, not normalized ones.
            H, W = pds_raw.shape[:2]
            if M_global is not None:
                syn_raw_aligned = cv2.warpAffine(
                    syn.astype(np.float32), M_global, (W, H),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                ).astype(np.float64)
            else:
                syn_raw_aligned = syn

            # Step 4: Derive object regions from ALIGNED ID mask (true geometry)
            regions = self._derive_regions(id_mask_a)

            # Step 4.5: Brightness ratio — uses ALIGNED regions on RAW pixels
            brightness = self._brightness_ratio(
                syn_raw_aligned, pds_raw, regions, pds_name,
                calibrated=is_calibrated, camera=camera,
            )
            all_brightness.append(brightness)

            # Full stage-by-stage debug save
            if DEBUG_SAVE_FULL:
                self._save_full_debug_stages(
                    i, pds_name, self._eval_count,
                    syn_raw_crop=syn, pds_raw=pds_raw,
                    id_mask_raw=id_mask,
                    pds_norm=pds_norm, syn_norm=syn_norm,
                    syn_aligned=syn_a, id_mask_aligned=id_mask_a,
                    syn_raw_aligned=syn_raw_aligned,
                    regions=regions,
                    pds_uint8=pds_uint8, syn_uint8=syn_uint8,
                    M_global=M_global,
                )

            # Step 5-7: Per-object scoring (per-frame combined score ignored;
            # only obj_scores + dn_ratio are aggregated globally)
            _, detail, dn_ratio = self._score_frame(
                syn_a, pds_norm, regions, frame_idx=i, pds_name=pds_name
            )

            # Pool per-body weighted metric sums for global aggregation
            fw = FRAME_WEIGHTS[i]
            for body, metrics in detail.items():
                if body not in body_metric_pool:
                    continue
                body_ws = 0.0
                for name, mw in METRIC_WEIGHTS.items():
                    val = metrics.get(name, 0.0)
                    if not np.isfinite(val):
                        val = 1.0
                    if name in ("ssim", "ncc"):
                        body_ws += mw * (1.0 - val)
                    else:
                        body_ws += mw * val
                body_metric_pool[body].append((fw, body_ws))

            all_dn_ratios.append(dn_ratio)
            frame_details.append(detail)

        # Per-body global average (FRAME_WEIGHTS as inner weights)
        body_avg: dict[str, float] = {}
        for body, samples in body_metric_pool.items():
            if not samples:
                continue
            w_sum_inner = sum(w for w, _ in samples)
            if w_sum_inner > 0:
                body_avg[body] = sum(w * v for w, v in samples) / w_sum_inner

        if not body_avg:
            score = 1e6  # No body scored anywhere
        else:
            # MASK_WEIGHTS body-weighted final: 50% Mars + 50% Phobos by default
            mw_sum = sum(MASK_WEIGHTS[b] for b in body_avg)
            score = sum(MASK_WEIGHTS[b] * body_avg[b] for b in body_avg) / mw_sum

            # DN ratio contribution (mean across frames where computed)
            valid_dn = [r for r in all_dn_ratios if np.isfinite(r)]
            mean_dn_ratio = float(np.mean(valid_dn)) if valid_dn else 1.0
            score += DN_RATIO_WEIGHT * (1.0 - mean_dn_ratio)

        # Brightness contribution (body-weighted across all frames)
        mean_brightness = self._aggregate_brightness(all_brightness)
        score += BRIGHTNESS_WEIGHT * mean_brightness

        # Log per-body summary
        for body, avg in body_avg.items():
            n = len(body_metric_pool[body])
            logger.info(
                "    body summary: %s — avg_metric=%.4f over %d frames",
                body, avg, n,
            )

        # CSV logging: per-metric means across frames+bodies (unchanged)
        all_metrics: dict[str, list[float]] = {}
        for fd in frame_details:
            for body_metrics in fd.values():
                if isinstance(body_metrics, dict):
                    for k, v in body_metrics.items():
                        if isinstance(v, (int, float)) and np.isfinite(v):
                            all_metrics.setdefault(k, []).append(v)
        self.last_detail = {
            k: float(np.mean(vs)) for k, vs in all_metrics.items()
        }
        self.last_detail["brightness"] = mean_brightness
        self.last_detail["score"] = score

        return score

    # -- CENTER CROP ----------------------------------------------------------

    @staticmethod
    def _center_crop(syn, mask, target_shape):
        """Crop synthetic and mask to target (PDS) dimensions."""
        sy, sx = syn.shape[:2]
        ty, tx = target_shape[:2]

        if (sy, sx) == (ty, tx):
            return syn, mask

        y0 = max(0, (sy - ty) // 2)
        x0 = max(0, (sx - tx) // 2)
        syn_c = syn[y0 : y0 + ty, x0 : x0 + tx]
        mask_c = mask[y0 : y0 + ty, x0 : x0 + tx]
        return syn_c, mask_c

    # -- BIT EQUALIZATION -----------------------------------------------------

    @staticmethod
    def _bit_equalize(pds, syn):
        """Normalize PDS and synthetic to [0, 1] for fair comparison.

        PDS may be int16 DN [0, 5000+] or float32 radiance [0, 0.13].
        Synthetic is typically uint16 [0, 65535].
        Normalizing both to [0, 1] removes scale bias from all metrics.
        """
        # PDS: use finite non-zero pixels for range
        pds_finite = pds[np.isfinite(pds)]
        pds_pos = pds_finite[pds_finite > 0]
        if pds_pos.size > 0:
            pds_min, pds_max = float(pds_pos.min()), float(pds_pos.max())
        else:
            pds_min, pds_max = 0.0, 1.0

        # Synthetic: use non-zero pixels for range
        syn_pos = syn[syn > 0]
        if syn_pos.size > 0:
            syn_min, syn_max = float(syn_pos.min()), float(syn_pos.max())
        else:
            syn_min, syn_max = 0.0, 1.0

        pds_range = pds_max - pds_min if pds_max > pds_min else 1.0
        syn_range = syn_max - syn_min if syn_max > syn_min else 1.0

        pds_norm = np.nan_to_num(pds, nan=0.0)
        pds_norm = np.clip((pds_norm - pds_min) / pds_range, 0.0, 1.0)
        # Restore background (originally 0 or NaN) to 0
        pds_norm[~np.isfinite(pds) | (pds <= 0)] = 0.0

        syn_norm = np.clip((syn - syn_min) / syn_range, 0.0, 1.0)
        syn_norm[syn <= 0] = 0.0

        return pds_norm, syn_norm

    # -- SIFT+RANSAC HELPERS ---------------------------------------------------

    @staticmethod
    def _normalize_to_uint8(img):
        """Convert [0,1] float image to uint8 [0,255] for SIFT.

        Uses percentile clipping (1-99%) for robust feature detection.
        """
        img = img.astype(np.float32)
        valid = img[np.isfinite(img) & (img > 0)]
        if valid.size < 10:
            return np.zeros(img.shape[:2], dtype=np.uint8)
        p1, p99 = np.percentile(valid, [1, 99])
        if p99 - p1 < 1e-8:
            return np.zeros(img.shape[:2], dtype=np.uint8)
        clipped = np.clip(img, p1, p99)
        normed = (clipped - p1) / (p99 - p1)
        return (normed * 255).astype(np.uint8)

    @staticmethod
    def _sift_ransac_transform(ref_uint8, mov_uint8, label="", mask_u8=None):
        """Estimate similarity transform (translation+rotation+uniform scale)
        from moving image to reference image using SIFT+FLANN+RANSAC.

        Args:
            ref_uint8: reference image (uint8, PDS stays fixed)
            mov_uint8: moving image (uint8, synthetic to be warped)
            label: logging label
            mask_u8: optional uint8 mask (255=valid) to limit keypoint detection
                     area. Images are NOT multiplied by the mask.

        Returns:
            M: 2x3 affine matrix (moving→reference) or None on failure
            info: dict with keypoints/matches/decomposed transform
        """
        sift = cv2.SIFT_create(
            nfeatures=5000, contrastThreshold=0.01,
            edgeThreshold=10, sigma=1.6,
        )
        kp_ref, des_ref = sift.detectAndCompute(ref_uint8, mask_u8)
        kp_mov, des_mov = sift.detectAndCompute(mov_uint8, mask_u8)

        info = {
            "ref_keypoints": len(kp_ref) if kp_ref else 0,
            "mov_keypoints": len(kp_mov) if kp_mov else 0,
            "good_matches": 0, "inliers": 0,
        }

        if des_ref is None or des_mov is None or len(kp_ref) < 4 or len(kp_mov) < 4:
            logger.warning("  %s SIFT: not enough keypoints (ref=%d, mov=%d)",
                           label, info["ref_keypoints"], info["mov_keypoints"])
            return None, info

        # FLANN KD-Tree matcher
        flann = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5), dict(checks=100)
        )
        knn_matches = flann.knnMatch(des_mov, des_ref, k=2)

        # Lowe ratio test
        good = [m for m, n in knn_matches if m.distance < 0.75 * n.distance]
        info["good_matches"] = len(good)

        if len(good) < 4:
            logger.warning("  %s SIFT: too few good matches (%d)", label, len(good))
            return None, info

        src_pts = np.float32([kp_mov[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, inlier_mask = cv2.estimateAffinePartial2D(
            src_pts, dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=5000,
            confidence=0.999,
            refineIters=20,
        )

        if M is None:
            logger.warning("  %s SIFT: RANSAC failed", label)
            return None, info

        info["inliers"] = int(inlier_mask.sum()) if inlier_mask is not None else 0
        info["inlier_ratio"] = info["inliers"] / max(1, len(good))

        # Decompose: M = [[a, -b, tx], [b, a, ty]]
        a, b = M[0, 0], M[1, 0]
        scale = float(np.sqrt(a * a + b * b))
        rotation_deg = float(np.degrees(np.arctan2(b, a)))
        tx, ty = float(M[0, 2]), float(M[1, 2])
        info["tx"] = tx
        info["ty"] = ty
        info["scale"] = scale
        info["rotation_deg"] = rotation_deg

        return M, info

    @staticmethod
    def _decompose_affine(M):
        """Decompose 2x3 partial affine matrix into components."""
        a, b = M[0, 0], M[1, 0]
        return {
            "tx": float(M[0, 2]),
            "ty": float(M[1, 2]),
            "scale": float(np.sqrt(a * a + b * b)),
            "rotation_deg": float(np.degrees(np.arctan2(b, a))),
        }

    # -- PASS 1: GLOBAL ALIGNMENT (SIFT+RANSAC) -------------------------------

    def _align_global(self, syn, mask, pds):
        """Global alignment via SIFT + RANSAC estimateAffinePartial2D.

        Estimates translation + rotation + uniform scale from the moving
        (synthetic, normalized) to the reference (PDS, normalized) frame.
        The same transform is later reapplied to RAW synthetic for brightness.

        Sanity limits:
            - |tx|, |ty| ≤ image_dim × ALIGN_MAX_SHIFT_FRAC (25%)
            - 0.9 ≤ scale ≤ 1.1
            - |rotation| ≤ 5°

        Returns:
            (syn_warped, mask_warped, M_2x3, pds_uint8, syn_uint8)
            or (syn, mask, None, None, None) if skipped.
        """
        H, W = pds.shape[:2]

        # Convert to uint8 for SIFT (percentile-clipped)
        pds_uint8 = self._normalize_to_uint8(pds)
        syn_uint8 = self._normalize_to_uint8(syn)

        M, info = self._sift_ransac_transform(pds_uint8, syn_uint8, label="Global")

        if M is None:
            logger.warning("  Global align: SIFT failed — skipping alignment")
            return syn, mask, None, pds_uint8, syn_uint8

        # Sanity checks
        tx, ty = info["tx"], info["ty"]
        scale = info["scale"]
        rot = info["rotation_deg"]
        max_shift_y = H * ALIGN_MAX_SHIFT_FRAC
        max_shift_x = W * ALIGN_MAX_SHIFT_FRAC

        reject = False
        if abs(tx) > max_shift_x or abs(ty) > max_shift_y:
            logger.warning(
                "  Global SIFT: shift (%.1f, %.1f) exceeds %.0f%% — rejected",
                tx, ty, ALIGN_MAX_SHIFT_FRAC * 100,
            )
            reject = True
        if not (0.9 <= scale <= 1.1):
            logger.warning(
                "  Global SIFT: scale=%.4f outside [0.9, 1.1] — rejected", scale,
            )
            reject = True
        if abs(rot) > 5.0:
            logger.warning(
                "  Global SIFT: rotation=%.2f° exceeds ±5° — rejected", rot,
            )
            reject = True

        if reject:
            return syn, mask, None, pds_uint8, syn_uint8

        # Warp synthetic and mask onto PDS frame using cv2
        syn_w = cv2.warpAffine(
            syn.astype(np.float32), M, (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        ).astype(np.float64)
        mask_w = cv2.warpAffine(
            mask, M, (W, H),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

        logger.info(
            "  Global SIFT: tx=%.2f ty=%.2f scale=%.4f rot=%.2f° "
            "(kp=%d/%d matches=%d inliers=%d)",
            tx, ty, scale, rot,
            info["ref_keypoints"], info["mov_keypoints"],
            info["good_matches"], info["inliers"],
        )
        return syn_w, mask_w, M, pds_uint8, syn_uint8

    # -- DERIVE OBJECT REGIONS ------------------------------------------------

    @staticmethod
    def _derive_regions(id_mask):
        """Derive per-object boolean regions from the combined CORTO ID mask.

        id_mask encodes true per-body geometry:
          - 1 = Phobos (mask_ID_1, CORTO pass_index=1)
          - 2 = Mars   (mask_ID_2, CORTO pass_index=2)

        Unlike the old synthetic-derived approach, this keeps Mars's shadowed
        and dark pixels in the region (they belong to Mars regardless of how
        bright the synthetic render makes them).
        """
        return {"phobos": id_mask == 1, "mars": id_mask == 2}

    # -- PER-FRAME SCORING ---------------------------------------------------

    def _score_frame(self, syn, pds, regions, frame_idx=0, pds_name=""):
        """Score a single frame with per-object metrics.

        Returns:
            (score, detail_dict) tuple.
        """
        obj_scores: dict[str, dict] = {}

        for body in MASK_WEIGHTS:
            if body not in regions:
                continue

            region = regions[body]
            n = int(region.sum())
            if n < MIN_PIXELS:
                logger.info(
                    "    %s: only %d pixels (< %d) -- skipping",
                    body, n, MIN_PIXELS,
                )
                continue

            frame_size = pds.shape[0] * pds.shape[1]
            if n / frame_size < MIN_OBJECT_FRACTION:
                logger.info(
                    "    %s: only %.2f%% of frame (< %.2f%%) -- too small, skipping",
                    body, 100 * n / frame_size, 100 * MIN_OBJECT_FRACTION,
                )
                continue

            # Skip if PDS has no signal in this region — object may be rendered
            # in the synthetic but lie outside the PDS camera FOV.
            pds_in_region = pds[region]
            pds_valid_in_region = int(
                np.sum(np.isfinite(pds_in_region) & (pds_in_region > 0))
            )
            if pds_valid_in_region < MIN_PIXELS:
                logger.info(
                    "    %s: PDS only %d valid pixels in region -- not in PDS FOV, skipping",
                    body, pds_valid_in_region,
                )
                continue

            # Pass 2: Per-object crop + best-of-3 alignment
            syn_crop, pds_crop, reg_crop, syn_valid = self._align_object_crop(
                syn, pds, region, body, frame_idx=frame_idx
            )

            # Overlap = region mask ∩ valid data.
            # After alignment, syn_crop is in the PDS (reference) frame,
            # so the SAME reg_crop applies to BOTH images.
            # Shadow pixels (zero DN) ARE included — correct shadow placement
            # improves the score. Only NaN (K-threshold) and warp edge fill
            # are excluded.
            overlap = reg_crop & np.isfinite(pds_crop) & syn_valid
            n_overlap = int(overlap.sum())
            if n_overlap < MIN_PIXELS:
                logger.info(
                    "    %s: only %d overlap pixels (< %d) -- skipping",
                    body, n_overlap, MIN_PIXELS,
                )
                continue

            logger.info(
                "    %s: region=%d, pds_finite=%d, syn_valid=%d, overlap=%d (%.0f%%)",
                body, int(reg_crop.sum()),
                int((np.isfinite(pds_crop) & reg_crop).sum()),
                int((syn_valid & reg_crop).sum()),
                n_overlap,
                100 * n_overlap / max(1, int(reg_crop.sum())),
            )

            # Extract scored pixels (overlap only)
            s_pds = pds_crop[overlap]
            s_syn = syn_crop[overlap]

            metrics = {
                "nrmse": self._nrmse(s_syn, s_pds),
                "ssim": self._ssim_region(syn_crop, pds_crop, overlap),
                "emd": self._emd(s_syn, s_pds),
                "ncc": self._ncc(s_syn, s_pds),
                "gmsd": self._gmsd_region(syn_crop, pds_crop, overlap),
                "lpips": self._lpips_region(syn_crop, pds_crop, overlap),
            }
            obj_scores[body] = metrics
            logger.info(
                "    %s (%d px): nrmse=%.4f ssim=%.4f emd=%.4f",
                body, n_overlap, metrics["nrmse"], metrics["ssim"], metrics["emd"],
            )

            # Save debug images (cropped aligned, overlap-masked)
            if SAVE_DEBUG_IMAGES:
                self._save_debug(
                    frame_idx, body, syn_crop, pds_crop, reg_crop,
                    self._eval_count, pds_name=pds_name,
                    syn_valid=syn_valid, overlap=overlap,
                )

        # Cross-object DN ratio score (on full aligned images)
        dn_ratio = self._dn_ratio_score(syn, pds, regions)

        return self._combine(obj_scores, dn_ratio), obj_scores, dn_ratio

    # -- PASS 2: BEST-OF-3 CANDIDATE ALIGNMENT ---------------------------------
    #
    # Candidates: SIFT+RANSAC (affine), ECC (translation), PCC+Pearson (translation)
    # Each is warped, evaluated (Pearson + SSIM), and gated.
    # Best accepted candidate wins; identity fallback if none pass.

    # Acceptance thresholds
    _P2_MIN_PEARSON_GAIN = 0.0
    _P2_MIN_SSIM_GAIN = 0.0
    _P2_MIN_VALID_RETENTION = 0.75
    _P2_MAX_SHIFT_FRAC = 0.20
    _P2_SIFT_MIN_INLIERS = 10

    @staticmethod
    def _quick_pearson(a, b, mask):
        """Fast masked Pearson correlation."""
        aa = a[mask].astype(np.float64)
        bb = b[mask].astype(np.float64)
        ok = np.isfinite(aa) & np.isfinite(bb)
        aa, bb = aa[ok], bb[ok]
        if aa.size < 10 or np.std(aa) < 1e-12 or np.std(bb) < 1e-12:
            return float("nan")
        return float(np.corrcoef(aa, bb)[0, 1])

    @staticmethod
    def _quick_ssim(a, b, mask):
        """Fast masked SSIM on bounding-box crop."""
        ys, xs = np.where(mask)
        if ys.size < 49:
            return float("nan")
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        ac = a[y0:y1, x0:x1].copy()
        bc = b[y0:y1, x0:x1].copy()
        mc = mask[y0:y1, x0:x1]
        ac[~mc] = 0.0
        bc[~mc] = 0.0
        if ac.shape[0] < 7 or ac.shape[1] < 7:
            return float("nan")
        dr = max(float(np.nanmax(ac) - np.nanmin(ac)),
                 float(np.nanmax(bc) - np.nanmin(bc)), 1e-6)
        return float(_ssim_fn(ac.astype(np.float64), bc.astype(np.float64),
                              data_range=dr))

    def _align_object_crop(self, syn, pds, region, body_name, frame_idx=0):
        """Crop to object bounding box, then refine with best-of-3 candidates.

        Candidates:
          A) SIFT+RANSAC partial affine (translation + rotation + scale)
          B) ECC translation (cv2.findTransformECC)
          C) PCC + Pearson sub-pixel refinement (translation)

        Each candidate is warped and evaluated with Pearson + SSIM.
        A candidate is accepted only if it improves both metrics and
        retains >=75% valid pixels. Best score wins; identity if none pass.

        Cache logic:
          - Accepted shifts are stored per (frame_idx, body_name) as
            integer-rounded (tx, ty) pixel keys.
          - On identity fallback, the most recent accepted M is reused.
          - After 5 repetitions of the same quantized shift, the transform
            is locked and alignment computation is skipped entirely.

        Returns:
            (syn_crop, pds_crop, reg_crop, syn_valid)
            reg_crop: region mask in reference frame (used for BOTH images)
            syn_valid: boolean mask of real syn pixels (not edge fill)
        """
        ys, xs = np.where(region)
        if ys.size == 0:
            ones = np.ones(syn.shape, dtype=bool)
            return syn, pds, region, ones

        cache_key = (frame_idx, body_name)

        # --- Symmetric BBox + Crop ---
        pad_y = max(10, int(0.1 * (ys.max() - ys.min())))
        pad_x = max(10, int(0.1 * (xs.max() - xs.min())))
        y0_raw, y1_raw = ys.min() - pad_y, ys.max() + pad_y + 1
        x0_raw, x1_raw = xs.min() - pad_x, xs.max() + pad_x + 1
        sym_pad_y = min(ys.min() - max(0, y0_raw),
                        min(pds.shape[0], y1_raw) - ys.max() - 1)
        sym_pad_x = min(xs.min() - max(0, x0_raw),
                        min(pds.shape[1], x1_raw) - xs.max() - 1)
        y0, y1 = ys.min() - sym_pad_y, ys.max() + sym_pad_y + 1
        x0, x1 = xs.min() - sym_pad_x, xs.max() + sym_pad_x + 1

        syn_crop = syn[y0:y1, x0:x1].copy()
        pds_crop = pds[y0:y1, x0:x1].copy()
        reg_crop = region[y0:y1, x0:x1].copy()
        cH, cW = syn_crop.shape[:2]

        # --- LOCKED: skip alignment entirely, reuse cached transform ---
        if cache_key in self._p2_locked:
            M_lock, inv_lock = self._p2_locked[cache_key]
            fl = cv2.INTER_LINEAR | (cv2.WARP_INVERSE_MAP if inv_lock else 0)
            syn_crop = cv2.warpAffine(
                syn_crop.astype(np.float32), M_lock.astype(np.float32),
                (cW, cH), flags=fl,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            ).astype(np.float64)
            ones_f = np.ones((cH, cW), dtype=np.float32)
            syn_valid = cv2.warpAffine(
                ones_f, M_lock.astype(np.float32), (cW, cH),
                flags=(cv2.INTER_NEAREST
                       | (cv2.WARP_INVERSE_MAP if inv_lock else 0)),
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            ) > 0.5
            logger.info("    %s Pass2 LOCKED: skipping alignment, reusing cached transform",
                        body_name)
            return syn_crop, pds_crop, reg_crop, syn_valid

        # --- Before-alignment metrics ---
        # Signal-filtered: same logic as standalone compute_metrics.
        # Alignment decision uses texture-rich signal pixels, not shadows.
        # (Final scoring overlap is separately shadow-inclusive.)
        _EPS = 1e-8
        eval_mask = (reg_crop
                     & np.isfinite(pds_crop) & (pds_crop > _EPS)
                     & np.isfinite(syn_crop) & (syn_crop > _EPS))
        n_before = int(eval_mask.sum())
        if n_before < 50:
            return syn_crop, pds_crop, reg_crop, np.ones(syn_crop.shape, bool)

        pearson_before = self._quick_pearson(pds_crop, syn_crop, eval_mask)
        ssim_before = self._quick_ssim(pds_crop, syn_crop, eval_mask)

        def _fin(v):
            return v if np.isfinite(v) else -1e9

        # --- Candidate helper: warp + evaluate + gate ---
        def _evaluate(M, name, inverse_map=False):
            flags = cv2.INTER_LINEAR
            if inverse_map:
                flags |= cv2.WARP_INVERSE_MAP
            warped = cv2.warpAffine(
                syn_crop.astype(np.float32), M.astype(np.float32), (cW, cH),
                flags=flags,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            ).astype(np.float64)
            ones = np.ones((cH, cW), dtype=np.float32)
            sv = cv2.warpAffine(
                ones, M.astype(np.float32), (cW, cH),
                flags=(cv2.INTER_NEAREST | (cv2.WARP_INVERSE_MAP if inverse_map else 0)),
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            ) > 0.5
            # Signal-filtered eval mask: same as standalone compute_metrics.
            # reg_crop & signal(pds) & signal(warped) & warp_valid
            em = (reg_crop
                  & np.isfinite(pds_crop) & (pds_crop > _EPS)
                  & np.isfinite(warped) & (warped > _EPS)
                  & sv)
            n_after = int(em.sum())
            retention = n_after / max(n_before, 1)
            p = self._quick_pearson(pds_crop, warped, em) if n_after >= 50 else float("nan")
            s = self._quick_ssim(pds_crop, warped, em) if n_after >= 50 else float("nan")
            pg = _fin(p) - _fin(pearson_before)
            sg = _fin(s) - _fin(ssim_before)
            accepted = (pg >= self._P2_MIN_PEARSON_GAIN
                        and sg >= self._P2_MIN_SSIM_GAIN
                        and retention >= self._P2_MIN_VALID_RETENTION
                        and np.isfinite(p) and np.isfinite(s))
            score = (3.0 * pg + 1.0 * sg + 0.05 * min(retention, 1.0)) if accepted else float("-inf")
            return {
                "name": name, "M": M, "inverse_map": inverse_map,
                "warped": warped, "syn_valid": sv,
                "pearson": p, "ssim": s, "pg": pg, "sg": sg,
                "retention": retention, "accepted": accepted, "score": score,
            }

        candidates = []
        max_dy, max_dx = cH * self._P2_MAX_SHIFT_FRAC, cW * self._P2_MAX_SHIFT_FRAC

        # --- Candidate A: SIFT+RANSAC ---
        try:
            pds_u8 = self._normalize_to_uint8(pds_crop)
            syn_u8 = self._normalize_to_uint8(syn_crop)
            mask_u8 = reg_crop.astype(np.uint8) * 255
            M_sift, si = self._sift_ransac_transform(
                pds_u8, syn_u8, label=body_name, mask_u8=mask_u8)
            if (M_sift is not None
                    and si.get("inliers", 0) >= self._P2_SIFT_MIN_INLIERS):
                sc = si.get("scale", 1.0)
                rot = si.get("rotation_deg", 0.0)
                tx, ty = si.get("tx", 0.0), si.get("ty", 0.0)
                if (abs(tx) <= max_dx and abs(ty) <= max_dy
                        and 0.9 <= sc <= 1.1 and abs(rot) <= 5.0):
                    candidates.append(_evaluate(M_sift, "SIFT"))
        except Exception:
            pass

        # --- Candidate B: ECC translation ---
        try:
            ref_ecc = self._normalize_to_uint8(pds_crop).astype(np.float32) / 255.0
            mov_ecc = self._normalize_to_uint8(syn_crop).astype(np.float32) / 255.0
            ecc_mask = reg_crop.astype(np.uint8) * 255
            warp_m = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-8)
            _, warp_m = cv2.findTransformECC(
                ref_ecc, mov_ecc, warp_m,
                motionType=cv2.MOTION_TRANSLATION,
                criteria=criteria, inputMask=ecc_mask, gaussFiltSize=5,
            )
            edx, edy = float(warp_m[0, 2]), float(warp_m[1, 2])
            if abs(edx) <= max_dx and abs(edy) <= max_dy:
                candidates.append(_evaluate(warp_m, "ECC", inverse_map=True))
        except cv2.error:
            pass

        # --- Candidate C: PCC + Pearson refinement ---
        try:
            # Use reg_crop as both masks — same as standalone make_pcc_candidate:
            #   phase_cross_correlation(ref01, mov01,
            #       reference_mask=region_mask, moving_mask=region_mask)
            # This ensures PCC shift estimation uses the object region,
            # not an arbitrary positivity mask.
            pds_safe = np.nan_to_num(pds_crop, nan=0.0)
            pcc_ref_mask = reg_crop & np.isfinite(pds_crop)
            pcc_mov_mask = reg_crop.copy()
            shift, _, _ = phase_cross_correlation(
                pds_safe, syn_crop,
                reference_mask=pcc_ref_mask, moving_mask=pcc_mov_mask,
                overlap_ratio=0.3,
            )
            pcc_dy, pcc_dx = float(shift[0]), float(shift[1])
            pcc_dy = float(np.clip(pcc_dy, -max_dy, max_dy))
            pcc_dx = float(np.clip(pcc_dx, -max_dx, max_dx))

            # Pearson sub-pixel refinement (Nelder-Mead)
            ref01 = self._normalize_to_uint8(pds_crop).astype(np.float32) / 255.0
            mov01 = self._normalize_to_uint8(syn_crop).astype(np.float32) / 255.0
            rough_mask = reg_crop.astype(bool)
            yy, xx = np.nonzero(rough_mask)
            if yy.size >= 1000:
                ref_samples = ref01[yy, xx].astype(np.float64)

                def _neg_pearson(s):
                    coords = np.vstack([yy - s[0], xx - s[1]])
                    ms = map_coordinates(mov01, coords, order=1,
                                         mode="constant", cval=np.nan,
                                         prefilter=False)
                    ok = np.isfinite(ms)
                    if ok.sum() < 500:
                        return 1.0
                    r, m = ref_samples[ok], ms[ok].astype(np.float64)
                    r = r - r.mean()
                    m = m - m.mean()
                    d = np.sqrt(np.sum(r * r) * np.sum(m * m))
                    return -float(np.sum(r * m) / d) if d > 0 else 1.0

                opt = _sp_minimize(
                    _neg_pearson, x0=[pcc_dy, pcc_dx],
                    method="Nelder-Mead",
                    options={"maxiter": 80, "xatol": 0.005, "fatol": 1e-6},
                )
                if opt.success:
                    pcc_dy, pcc_dx = float(opt.x[0]), float(opt.x[1])

            if abs(pcc_dy) >= 0.05 or abs(pcc_dx) >= 0.05:
                M_pcc = np.float32([[1, 0, pcc_dx], [0, 1, pcc_dy]])
                candidates.append(_evaluate(M_pcc, "PCC"))
        except Exception:
            pass

        # --- Select best accepted candidate ---
        accepted = [c for c in candidates if c["accepted"]]
        if accepted:
            best = max(accepted, key=lambda c: c["score"])
            syn_crop = best["warped"]
            syn_valid = best["syn_valid"]
            logger.info(
                "    %s Pass2 SELECTED: %s (score=%.4f, Pearson %.4f→%.4f, "
                "SSIM %.4f→%.4f, retention=%.0f%%)",
                body_name, best["name"], best["score"],
                pearson_before if np.isfinite(pearson_before) else -1,
                best["pearson"],
                ssim_before if np.isfinite(ssim_before) else -1,
                best["ssim"],
                best["retention"] * 100,
            )
            # Log rejected candidates
            for c in candidates:
                if not c["accepted"]:
                    logger.info("    %s %s rejected (pg=%.4f sg=%.4f ret=%.0f%%)",
                                body_name, c["name"], c["pg"], c["sg"],
                                c["retention"] * 100)

            # --- Cache accepted shift ---
            M_best = best["M"]
            inv_best = best.get("inverse_map", False)
            eff_tx = -float(M_best[0, 2]) if inv_best else float(M_best[0, 2])
            eff_ty = -float(M_best[1, 2]) if inv_best else float(M_best[1, 2])
            q_shift = (round(eff_tx), round(eff_ty))
            self._p2_shift_history.setdefault(cache_key, []).append(q_shift)
            self._p2_best_M[cache_key] = (M_best.copy(), inv_best)
            # Lock if any quantized shift repeated 5 times
            from collections import Counter
            counts = Counter(self._p2_shift_history[cache_key])
            for sk, cnt in counts.items():
                if cnt >= 5:
                    self._p2_locked[cache_key] = self._p2_best_M[cache_key]
                    logger.info(
                        "    %s Pass2 LOCKED shift (%d, %d) after %d repeats",
                        body_name, sk[0], sk[1], cnt,
                    )
                    break
        else:
            # Identity fallback — try cache from previous accepted eval
            if cache_key in self._p2_best_M:
                M_prev, inv_prev = self._p2_best_M[cache_key]
                flags_prev = cv2.INTER_LINEAR
                if inv_prev:
                    flags_prev |= cv2.WARP_INVERSE_MAP
                syn_crop = cv2.warpAffine(
                    syn_crop.astype(np.float32), M_prev.astype(np.float32),
                    (cW, cH), flags=flags_prev,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                ).astype(np.float64)
                ones_f = np.ones((cH, cW), dtype=np.float32)
                syn_valid = cv2.warpAffine(
                    ones_f, M_prev.astype(np.float32), (cW, cH),
                    flags=(cv2.INTER_NEAREST
                           | (cv2.WARP_INVERSE_MAP if inv_prev else 0)),
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                ) > 0.5
                logger.info(
                    "    %s Pass2: identity fallback → reusing previous accepted shift",
                    body_name,
                )
            else:
                syn_valid = np.ones(syn_crop.shape, dtype=bool)
                names = [c["name"] for c in candidates]
                logger.info(
                    "    %s Pass2: no candidate accepted (%s), no cache → identity",
                    body_name, ", ".join(names) if names else "none tried",
                )

        return syn_crop, pds_crop, reg_crop, syn_valid

    # -- METRICS --------------------------------------------------------------

    @staticmethod
    def _nrmse(a: np.ndarray, b: np.ndarray) -> float:
        """Normalized RMSE (by PDS range)."""
        b_clean = b[np.isfinite(b)]
        if b_clean.size == 0:
            return 1.0
        data_range = float(b_clean.max() - b_clean.min())
        if data_range <= 0:
            return 1.0
        mse = float(np.nanmean((a - b) ** 2))
        return float(np.sqrt(mse) / data_range)

    @staticmethod
    def _ssim_region(syn, pds, region) -> float:
        """SSIM on cropped region (NaN-safe)."""
        pds_clean = np.where(region, np.nan_to_num(pds, nan=0.0), 0.0)
        syn_clean = np.where(region, np.nan_to_num(syn, nan=0.0), 0.0)

        finite_pds = pds_clean[pds_clean > 0] if (pds_clean > 0).any() else pds_clean.ravel()
        finite_syn = syn_clean[syn_clean > 0] if (syn_clean > 0).any() else syn_clean.ravel()

        if finite_pds.size == 0 or finite_syn.size == 0:
            return 0.0

        dr = max(finite_pds.max(), finite_syn.max()) - min(finite_pds.min(), finite_syn.min())
        if dr <= 0:
            return 0.0

        try:
            val = float(_ssim_fn(pds_clean, syn_clean, data_range=float(dr)))
            return val if np.isfinite(val) else 0.0
        except (ValueError, RuntimeError):
            return 0.0

    @staticmethod
    def _emd(a: np.ndarray, b: np.ndarray) -> float:
        """Earth Mover's Distance with shared bin range."""
        a_clean = a[np.isfinite(a) & (a > 0)]
        b_clean = b[np.isfinite(b) & (b > 0)]
        if a_clean.size == 0 or b_clean.size == 0:
            return 1.0
        lo = min(a_clean.min(), b_clean.min())
        hi = max(a_clean.max(), b_clean.max())
        bins = np.linspace(lo, hi, EMD_BINS + 1)
        ha, _ = np.histogram(a_clean, bins=bins, density=True)
        hb, _ = np.histogram(b_clean, bins=bins, density=True)
        return float(wasserstein_distance(ha, hb))

    @staticmethod
    def _ncc(a: np.ndarray, b: np.ndarray) -> float:
        """Normalized cross-correlation (Pearson)."""
        mask = np.isfinite(a) & np.isfinite(b)
        a_m, b_m = a[mask], b[mask]
        if a_m.size < 10:
            return 0.0
        a_m = a_m - a_m.mean()
        b_m = b_m - b_m.mean()
        denom = float(np.sqrt(np.sum(a_m ** 2) * np.sum(b_m ** 2)))
        if denom <= 0:
            return 0.0
        return float(np.clip(np.sum(a_m * b_m) / denom, -1, 1))

    @staticmethod
    def _gmsd_region(syn, pds, region) -> float:
        """Gradient Magnitude Similarity Deviation."""
        pds_c = np.where(region, np.nan_to_num(pds, nan=0.0), 0.0)
        syn_c = np.where(region, syn, 0.0)
        hx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) / 3.0
        hy = hx.T
        try:
            gx_r = convolve2d(pds_c, hx, mode="same", boundary="symm")
            gy_r = convolve2d(pds_c, hy, mode="same", boundary="symm")
            gx_s = convolve2d(syn_c, hx, mode="same", boundary="symm")
            gy_s = convolve2d(syn_c, hy, mode="same", boundary="symm")
        except (ValueError, RuntimeError):
            return 0.0
        gm_r = np.sqrt(gx_r ** 2 + gy_r ** 2)
        gm_s = np.sqrt(gx_s ** 2 + gy_s ** 2)
        c = 0.0026
        gms = (2 * gm_r * gm_s + c) / (gm_r ** 2 + gm_s ** 2 + c)
        return float(np.std(gms))

    @staticmethod
    def _lpips_region(syn, pds, region) -> float:
        """Perceptual similarity (LPIPS, AlexNet). Lower = more similar.

        Lazy-loads the model on first call; returns 0.0 if lpips/torch
        is not installed.
        """
        global _LPIPS_MODEL, _LPIPS_OK
        if _LPIPS_OK is False:
            return 0.0
        try:
            import torch
            import lpips as _lpips_pkg

            if _LPIPS_MODEL is None:
                _LPIPS_MODEL = _lpips_pkg.LPIPS(net="alex", verbose=False)
                _LPIPS_MODEL.eval()
                _LPIPS_OK = True
                logger.info("LPIPS AlexNet model loaded")

            def _to_tensor(a):
                t = torch.from_numpy(
                    np.where(region, np.nan_to_num(a, nan=0.0), 0.0).astype(np.float32)
                ).unsqueeze(0).unsqueeze(0)
                t = t.repeat(1, 3, 1, 1)  # grayscale → fake RGB
                return t * 2.0 - 1.0       # [0,1] → [-1,1]

            with torch.no_grad():
                val = _LPIPS_MODEL(_to_tensor(pds), _to_tensor(syn))
            return float(val.item())

        except ImportError:
            _LPIPS_OK = False
            logger.info("lpips not installed — LPIPS metric disabled")
            return 0.0
        except Exception as exc:
            logger.warning("LPIPS computation failed: %s", exc)
            return 0.0

    # -- DN Ratio Score -------------------------------------------------------

    @staticmethod
    def _dn_ratio_score(syn, pds, regions) -> float:
        """Cross-object relative brightness score.

        Returns a value in [0, 1] where 1.0 = perfect ratio match.
        """
        phobos = regions.get("phobos", np.zeros_like(syn, dtype=bool))
        mars = regions.get("mars", np.zeros_like(syn, dtype=bool))

        if phobos.sum() < MIN_PIXELS or mars.sum() < MIN_PIXELS:
            return 1.0  # Cannot compute -> neutral

        p_real = float(np.nanmean(pds[phobos]))
        m_real = float(np.nanmean(pds[mars]))
        _p = syn[phobos]
        _m = syn[mars]
        p_syn = float(_p[_p > 0].mean()) if (_p > 0).any() else 0.0
        m_syn = float(_m[_m > 0].mean()) if (_m > 0).any() else 0.0

        if p_real <= 0 or p_syn <= 0:
            return 0.0

        r_real = m_real / p_real
        r_syn = m_syn / p_syn
        if max(r_real, r_syn) <= 0:
            return 0.0
        return min(r_real, r_syn) / max(r_real, r_syn)

    # -- Brightness Ratio (pre-normalization) ---------------------------------

    @staticmethod
    def _brightness_ratio(
        syn_raw: np.ndarray,
        pds_raw: np.ndarray,
        regions: dict[str, np.ndarray],
        pds_name: str = "",
        calibrated: bool = False,
        camera: str = "hrsc",
    ) -> dict[str, float]:
        """Per-object absolute brightness matching on RAW (pre-normalized) data.

        Computed BEFORE _bit_equalize so that sun_scaler has a visible effect.
        Both images must be in the same processing state:
          - PDS: k-thresholded (noise floor → NaN), NOT [0,1] normalized
          - Synthetic: DC-applied, NOT [0,1] normalized

        Per-camera CALIBRATED_SCALE:
          - HRSC SR2 (RDR Level-2, bit-normalized): scale = 1.0 (already in DN range)
          - OSIRIS (calibrated radiance W/m²/sr/nm): scale = 190_000 → pseudo-DN

        Returns dict mapping body name → 1 - exp(-|log(median_syn / median_pds)|).
        Score of 0.0 = perfect brightness match; approaches 1.0 as mismatch grows.
        """
        # Per-camera scale to bring PDS into rendered uint16 (~10k-30k) range.
        # OSIRIS derivation: ss=0.028 → median_syn=29386, median_pds=0.0554
        # → target equilibrium ss≈0.01 → CALIBRATED_SCALE ≈ 190_000.
        # HRSC: RDR DN values already in 1k-30k range, no scaling needed.
        CALIBRATED_SCALE_PER_CAMERA = {
            "hrsc":   1.0,
            "osiris": 190_000.0,
        }
        CALIBRATED_SCALE = CALIBRATED_SCALE_PER_CAMERA.get(camera, 1.0)

        scores: dict[str, float] = {}
        for body, region in regions.items():
            n_region = int(region.sum())
            if n_region < MIN_PIXELS:
                continue

            frame_size = region.shape[0] * region.shape[1]
            if n_region / frame_size < MIN_OBJECT_FRACTION:
                continue

            # Synthetic: positive rendered pixels within this object
            syn_obj = syn_raw[region]
            syn_pos = syn_obj[syn_obj > 0]
            if syn_pos.size < MIN_PIXELS:
                continue

            # PDS: finite positive pixels within this object
            pds_obj = pds_raw[region] if region.shape == pds_raw.shape else np.array([])
            if pds_obj.size == 0:
                continue
            pds_valid = pds_obj[np.isfinite(pds_obj) & (pds_obj > 0)]
            if pds_valid.size < MIN_PIXELS:
                continue

            med_syn = float(np.median(syn_pos))
            med_pds = float(np.median(pds_valid))

            # Scale calibrated radiance to pseudo-DN range
            if calibrated:
                med_pds *= CALIBRATED_SCALE

            if med_syn <= 0 or med_pds <= 0:
                continue

            ratio = med_syn / med_pds
            score = float(1.0 - np.exp(-abs(np.log(ratio))))
            scores[body] = score
            logger.info(
                "    Brightness %s [%s]: median_syn=%.1f, median_pds=%.1f%s, "
                "ratio=%.4f, |log|=%.4f",
                body, pds_name, med_syn, med_pds,
                " (x190k cal)" if calibrated else "",
                ratio, score,
            )

        return scores

    @staticmethod
    def _aggregate_brightness(
        all_brightness: list[dict[str, float]],
    ) -> float:
        """Aggregate per-frame per-object brightness scores.

        Returns a single scalar (lower=better) averaged over all bodies
        and frames, weighted by MASK_WEIGHTS.
        """
        body_scores: dict[str, list[float]] = {}
        for frame_b in all_brightness:
            for body, score in frame_b.items():
                body_scores.setdefault(body, []).append(score)

        if not body_scores:
            return 0.0  # No objects scored → neutral

        total, w_sum = 0.0, 0.0
        for body, scores in body_scores.items():
            w = MASK_WEIGHTS.get(body, 0.5)
            total += w * float(np.mean(scores))
            w_sum += w

        return total / w_sum if w_sum > 0 else 0.0

    # -- Combine Metrics ------------------------------------------------------

    @staticmethod
    def _combine(obj_scores: dict, dn_ratio: float) -> float:
        """Combine per-object metrics into a single scalar.

        Higher-is-better metrics (SSIM, NCC) are converted via (1 - val).
        All contributions are lower=better for CMA-ES minimization.
        Note: brightness_ratio is added in score_all(), not here.
        """
        if not obj_scores:
            return 1e6  # No body scored -> large penalty

        body_vals: dict[str, float] = {}
        for body, metrics in obj_scores.items():
            s = 0.0
            for name, w in METRIC_WEIGHTS.items():
                val = metrics.get(name, 0.0)
                # Guard against NaN/Inf metric values
                if not np.isfinite(val):
                    val = 1.0  # worst-case fallback
                if name in ("ssim", "ncc"):
                    s += w * (1.0 - val)  # higher -> lower
                else:
                    s += w * val          # already lower=better
            body_vals[body] = s

        # Body-weighted sum
        active_weights = {b: MASK_WEIGHTS[b] for b in body_vals}
        total_w = sum(active_weights.values())
        total = sum(active_weights[b] * body_vals[b] for b in body_vals)
        total /= total_w if total_w > 0 else 1.0

        # DN ratio contribution: score -> (1 - score) so lower=better
        total += DN_RATIO_WEIGHT * (1.0 - dn_ratio)

        return total

    # -- Debug Image Saving ---------------------------------------------------

    @staticmethod
    def _save_debug(frame_idx, body, syn_crop, pds_crop, region_crop,
                    eval_id, pds_name="", syn_valid=None, overlap=None):
        """Save cropped aligned masked images for visual inspection.

        Output structure (all images are object-cropped + aligned):
          debug_scoring/
            eval_0042/
              frame_00_H7982_0003_SR2_phobos_syn_aligned.tif
              frame_00_H7982_0003_SR2_phobos_real_aligned.tif
              frame_00_H7982_0003_SR2_phobos_syn_masked.tif
              frame_00_H7982_0003_SR2_phobos_real_masked.tif
              frame_00_H7982_0003_SR2_phobos_overlap.png       (NEW)
              frame_00_H7982_0003_SR2_phobos_syn_scored.tif    (NEW)
              frame_00_H7982_0003_SR2_phobos_pds_scored.tif    (NEW)
        """
        from PIL import Image

        out_dir = Path(DEBUG_DIR) / f"eval_{eval_id:04d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Include PDS image name in filename for traceability
        pds_tag = f"_{pds_name}" if pds_name else ""
        prefix = f"frame_{frame_idx:02d}{pds_tag}_{body}"

        # Full crop (aligned, with context around the object)
        Image.fromarray(syn_crop.astype(np.float32), mode="F").save(
            str(out_dir / f"{prefix}_syn_aligned.tif")
        )
        Image.fromarray(
            np.nan_to_num(pds_crop, nan=0.0).astype(np.float32), mode="F"
        ).save(str(out_dir / f"{prefix}_real_aligned.tif"))

        # Masked crop (only object pixels visible)
        syn_masked = np.where(region_crop, syn_crop, 0.0).astype(np.float32)
        Image.fromarray(syn_masked, mode="F").save(
            str(out_dir / f"{prefix}_syn_masked.tif")
        )

        pds_masked = np.where(
            region_crop, np.nan_to_num(pds_crop, nan=0.0), 0.0
        ).astype(np.float32)
        Image.fromarray(pds_masked, mode="F").save(
            str(out_dir / f"{prefix}_real_masked.tif")
        )

        # Overlap mask + scored pixels (exactly what gets scored)
        if overlap is not None:
            Image.fromarray((overlap.astype(np.uint8) * 255)).save(
                str(out_dir / f"{prefix}_overlap.png")
            )
            syn_scored = np.where(overlap, syn_crop, 0.0).astype(np.float32)
            Image.fromarray(syn_scored, mode="F").save(
                str(out_dir / f"{prefix}_syn_scored.tif")
            )
            pds_scored = np.where(
                overlap, np.nan_to_num(pds_crop, nan=0.0), 0.0
            ).astype(np.float32)
            Image.fromarray(pds_scored, mode="F").save(
                str(out_dir / f"{prefix}_pds_scored.tif")
            )

        logger.debug("  Saved debug: %s (crop %dx%d)", prefix,
                      syn_crop.shape[1], syn_crop.shape[0])

    @staticmethod
    def _save_full_debug_stages(
        frame_idx, pds_name, eval_id,
        syn_raw_crop, pds_raw, id_mask_raw,
        pds_norm, syn_norm,
        syn_aligned, id_mask_aligned,
        syn_raw_aligned, regions,
        pds_uint8=None, syn_uint8=None, M_global=None,
    ):
        """Save ALL intermediate pipeline stages for full debug tracing.

        Only runs when DEBUG_SAVE_FULL=True. Files are numbered by stage
        so they sort chronologically in file browsers.

        Output structure:
          debug_scoring/eval_0042/frame_00_H7982_0003_SR2/
            01_syn_raw_cropped.tif          ← after center-crop, raw DN
            02_pds_raw.tif                  ← PDS reference, raw DN (K-thresholded)
            03a_id_mask_cropped.png         ← combined mask BEFORE alignment (0/1/2)
            03b_pds_sift_input.png          ← SIFT input: PDS percentile-clipped uint8
            03c_syn_sift_input.png          ← SIFT input: SYN percentile-clipped uint8
            03d_transform.json             ← 2x3 affine M + decomposed (tx,ty,scale,rot)
            04_pds_norm.tif                 ← [0,1] normalized PDS
            05_syn_norm.tif                 ← [0,1] normalized syn
            06_syn_aligned_global.tif       ← Pass 1 output (normalized)
            07_id_mask_aligned_global.png   ← Pass 1 output mask
            08_syn_raw_aligned.tif          ← Pass 1 transform applied to RAW syn
            09_region_phobos.png            ← derived phobos bool mask (aligned)
            09_region_mars.png              ← derived mars bool mask (aligned)
        """
        import json as _json
        from PIL import Image

        pds_tag = f"_{pds_name}" if pds_name else ""
        out_dir = Path(DEBUG_DIR) / f"eval_{eval_id:04d}" / f"frame_{frame_idx:02d}{pds_tag}"
        out_dir.mkdir(parents=True, exist_ok=True)

        def _save_f32(arr, name):
            Image.fromarray(
                np.nan_to_num(arr, nan=0.0).astype(np.float32), mode="F"
            ).save(str(out_dir / name))

        def _save_mask(arr, name):
            # Scale uint8 mask values for visibility: 0→0, 1→127, 2→255
            if arr.max() <= 2:
                vis = (arr.astype(np.float32) * 127).clip(0, 255).astype(np.uint8)
            else:
                vis = arr.astype(np.uint8)
            Image.fromarray(vis).save(str(out_dir / name))

        def _save_bool(arr, name):
            Image.fromarray((arr.astype(np.uint8) * 255)).save(str(out_dir / name))

        # Stage 1: Raw cropped synthetic
        _save_f32(syn_raw_crop, "01_syn_raw_cropped.tif")

        # Stage 2: Raw PDS reference
        _save_f32(pds_raw, "02_pds_raw.tif")

        # Stage 3a: Combined ID mask (cropped, before alignment)
        _save_mask(id_mask_raw, "03a_id_mask_cropped.png")

        # Stage 3b-3c: SIFT input images (percentile-clipped uint8)
        if pds_uint8 is not None:
            Image.fromarray(pds_uint8).save(str(out_dir / "03b_pds_sift_input.png"))
        if syn_uint8 is not None:
            Image.fromarray(syn_uint8).save(str(out_dir / "03c_syn_sift_input.png"))

        # Stage 3d: Transform matrix and decomposition
        if M_global is not None:
            a, b = M_global[0, 0], M_global[1, 0]
            tinfo = {
                "affine_matrix": M_global.tolist(),
                "tx": float(M_global[0, 2]),
                "ty": float(M_global[1, 2]),
                "scale": float(np.sqrt(a * a + b * b)),
                "rotation_deg": float(np.degrees(np.arctan2(b, a))),
            }
            with open(str(out_dir / "03d_transform.json"), "w") as f:
                _json.dump(tinfo, f, indent=2)

        # Stage 4: Normalized PDS
        _save_f32(pds_norm, "04_pds_norm.tif")

        # Stage 5: Normalized synthetic
        _save_f32(syn_norm, "05_syn_norm.tif")

        # Stage 6: Pass 1 aligned synthetic (normalized)
        _save_f32(syn_aligned, "06_syn_aligned_global.tif")

        # Stage 7: Pass 1 aligned ID mask
        _save_mask(id_mask_aligned, "07_id_mask_aligned_global.png")

        # Stage 8: Pass 1 transform applied to RAW synthetic
        _save_f32(syn_raw_aligned, "08_syn_raw_aligned.tif")

        # Stage 9: Derived body regions (aligned)
        for body, region in regions.items():
            _save_bool(region, f"09_region_{body}.png")

        logger.debug(
            "  Full debug stages saved: eval_%04d/frame_%02d%s/ (13 files)",
            eval_id, frame_idx, pds_tag,
        )

    # -- Utilities ------------------------------------------------------------

    @staticmethod
    def _load(path: str) -> np.ndarray:
        """Load an image file as numpy array."""
        from PIL import Image
        return np.array(Image.open(path))
