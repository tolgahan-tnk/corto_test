"""PDS data reader and noise cleaning pipeline.

Pipeline steps (applied to raw PDS images before scoring):
  1. filter_hot_pixels  -> histogram/5-sigma outlier -> 8-neighbor mean replace
  2. filter_invalid     -> DN<=0 or radiance<=0 or NaN/Inf -> NaN
  3. preclip_percentile -> P1-P99 clip + [0,1] (only if NORMALIZE=True)
  4. fill_nan_8neighbor  -> fill NaNs from 8-neighbor mean (invalid neighbors excluded)
  5. k_sweep_threshold  -> shadow-mask-guided percentile threshold -> NaN (if K_SWEEP_ENABLED)

Independent of Blender. DC is applied to synthetic renders, not here.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from config import (
    HOT_PIXEL_GAP_LEN,
    HOT_PIXEL_SIGMA,
    K_FIXED,
    K_MAX_LIT_LOSS_PCT,
    K_SWEEP_ENABLED,
    K_SWEEP_RANGE,
    K_SWEEP_STEP,
    NORM_HIGH_P,
    NORM_LOW_P,
    NORMALIZE,
    PDS_IMAGES,
)

logger = logging.getLogger(__name__)


class PDSLoader:
    """Loads and cleans PDS reference images."""

    def load_all(
        self, optimal_k: dict[str, float] | None = None
    ) -> list[np.ndarray]:
        """Load all PDS images and apply the noise pipeline.

        Args:
            optimal_k: Optional dict mapping PDS stem -> threshold value.
                        If provided and K_SWEEP_ENABLED, Step 5 (K-threshold)
                        is applied after the standard 4-step pipeline.
        """
        refs: list[np.ndarray] = []
        for meta in PDS_IMAGES:
            raw = self._parse_pds(Path(meta["file"]), meta)
            cleaned = self._noise_pipeline(raw, meta)

            # Step 5: K-sweep threshold (if computed)
            if optimal_k is not None and K_SWEEP_ENABLED:
                stem = Path(meta["file"]).stem
                k_val = optimal_k.get(stem)
                if k_val is not None:
                    cleaned = self._apply_k_threshold(cleaned, k_val)
                    logger.info(
                        "  K-threshold applied: k=%.1f on %s", k_val, stem
                    )

            refs.append(cleaned)
            logger.info(
                "Loaded %s: shape=%s, range=[%.1f, %.1f]",
                meta["file"],
                cleaned.shape,
                float(np.nanmin(cleaned)),
                float(np.nanmax(cleaned)),
            )
        return refs

    # -- K-Sweep (Phase 1.5) --------------------------------------------------

    def find_all_optimal_k(
        self, shadow_mask_paths: list[str]
    ) -> dict[str, float]:
        """Run K-sweep for each PDS image to find optimal noise threshold.

        For each frame:
          1. Load raw PDS and run Steps 1-4 (noise pipeline)
          2. Load the corresponding shadow mask
          3. Sweep K from K_SWEEP_RANGE[0] to K_SWEEP_RANGE[1]
          4. At each K, threshold the image and check lit pixel retention
          5. Select the highest K where lit pixel loss < K_MAX_LIT_LOSS_PCT

        Args:
            shadow_mask_paths: List of shadow mask image paths (one per PDS frame).
                               CORTO convention: white (255) = lit, black (0) = shadow.

        Returns:
            Dict mapping PDS stem -> optimal K percentile value.
        """
        if not K_SWEEP_ENABLED:
            logger.info("K-sweep disabled, using K_FIXED: %s", K_FIXED)
            return dict(K_FIXED)

        optimal_k: dict[str, float] = {}

        for idx, meta in enumerate(PDS_IMAGES):
            stem = Path(meta["file"]).stem

            # Check for manual override
            if stem in K_FIXED:
                optimal_k[stem] = float(K_FIXED[stem])
                logger.info(
                    "K-sweep: %s -> FIXED k=%d (manual override)",
                    stem, K_FIXED[stem],
                )
                continue

            # Load raw PDS and run noise pipeline (Steps 1-4)
            raw = self._parse_pds(Path(meta["file"]), meta)
            cleaned = self._noise_pipeline(raw, meta)

            # Load shadow mask
            if idx >= len(shadow_mask_paths):
                logger.warning(
                    "K-sweep: no shadow mask for frame %d (%s), skipping",
                    idx, stem,
                )
                continue

            shadow_path = shadow_mask_paths[idx]
            shadow_mask = self._load_shadow_mask(shadow_path)

            if shadow_mask is None:
                logger.warning(
                    "K-sweep: shadow mask not found: %s, skipping", shadow_path
                )
                continue

            # Resize shadow mask to match PDS if needed
            if shadow_mask.shape != cleaned.shape:
                from PIL import Image
                shadow_pil = Image.fromarray(shadow_mask)
                shadow_pil = shadow_pil.resize(
                    (cleaned.shape[1], cleaned.shape[0]),
                    Image.NEAREST,
                )
                shadow_mask = np.array(shadow_pil)

            # Run sweep
            best_k = self._k_sweep_single(cleaned, shadow_mask, stem)
            optimal_k[stem] = best_k

        logger.info("K-sweep results: %s", optimal_k)
        return optimal_k

    def _k_sweep_single(
        self,
        cleaned: np.ndarray,
        shadow_mask: np.ndarray,
        stem: str,
    ) -> float:
        """Sweep K for a single PDS frame.

        Shadow mask convention (CORTO mask_ID_shadow_1):
          255 = lit (illuminated), 0 = shadow.
        We threshold the cleaned image at each percentile K and count
        how many lit pixels are zeroed out (lost).

        Returns:
            Optimal K percentile value.
        """
        # Derive lit/shadow regions from mask
        # CORTO: 255 = lit, 0 = shadow
        is_lit = shadow_mask > 128
        is_shadow = ~is_lit

        # Count valid lit pixels at baseline (no threshold)
        valid = np.isfinite(cleaned)
        lit_valid = valid & is_lit
        total_lit = int(lit_valid.sum())

        if total_lit == 0:
            logger.warning("K-sweep %s: no lit pixels found, using k=0", stem)
            return 0.0

        # Get valid pixel values for percentile computation
        valid_vals = cleaned[valid]
        if valid_vals.size == 0:
            return 0.0

        best_k = float(K_SWEEP_RANGE[0])
        k_lo, k_hi = K_SWEEP_RANGE

        for k in range(k_lo, k_hi + 1, K_SWEEP_STEP):
            threshold = float(np.percentile(valid_vals, k))

            # Count lit pixels that would be lost at this threshold
            would_be_zeroed = (cleaned < threshold) & lit_valid
            lit_loss_pct = 100.0 * would_be_zeroed.sum() / total_lit

            if lit_loss_pct <= K_MAX_LIT_LOSS_PCT:
                best_k = float(k)
            else:
                # First K that exceeds loss tolerance -> stop
                break

        logger.info(
            "K-sweep %s: optimal k=%.0f (percentile), "
            "threshold=%.1f, total_lit=%d",
            stem, best_k,
            float(np.percentile(valid_vals, best_k)),
            total_lit,
        )
        return best_k

    @staticmethod
    def _apply_k_threshold(
        img: np.ndarray, k_percentile: float
    ) -> np.ndarray:
        """Apply K-percentile threshold: pixels below threshold -> NaN.

        This replaces noise-floor pixels with NaN so they are excluded
        from scoring metrics.
        """
        valid = img[np.isfinite(img)]
        if valid.size == 0 or k_percentile <= 0:
            return img

        threshold = float(np.percentile(valid, k_percentile))
        result = img.copy()
        result[result < threshold] = np.nan
        n_zeroed = int((img < threshold).sum())
        logger.info(
            "  K-threshold: k=%.0f%% -> thr=%.1f, zeroed=%d pixels",
            k_percentile, threshold, n_zeroed,
        )
        return result

    @staticmethod
    def _load_shadow_mask(path: str) -> np.ndarray | None:
        """Load a shadow mask image as a 2D uint8 array."""
        from pathlib import Path as P
        if not P(path).exists():
            return None
        from PIL import Image
        img = Image.open(path).convert("L")
        return np.array(img)

    # -- Noise Pipeline -------------------------------------------------------

    def _noise_pipeline(self, raw: np.ndarray, _meta: dict) -> np.ndarray:
        """Full noise cleaning pipeline.

        Args:
            raw: Raw PDS image array.
            _meta: PDS metadata (reserved for future per-camera config).
        """

        # Step 1: Hot pixel filter
        cleaned, thr, n_hot = self._filter_hot_pixels(raw)
        logger.info("  Hot pixel: threshold=%.1f, replaced=%d", thr, n_hot)

        # Step 2: Invalid pixel -> NaN (DN<=0, radiance<=0, NaN, Inf)
        cleaned = self._filter_invalid(cleaned)

        # Step 3: Percentile clip + normalize (only if NORMALIZE=True)
        if NORMALIZE:
            cleaned = self._preclip_percentile(cleaned, NORM_LOW_P, NORM_HIGH_P)

        # Step 4: Fill NaN using 8-neighbor mean (invalid neighbors excluded)
        cleaned = self._fill_nan_8neighbor(cleaned)

        return cleaned

    # -- Step 1: Hot Pixel Filter ---------------------------------------------

    def _filter_hot_pixels(
        self, raw: np.ndarray
    ) -> tuple[np.ndarray, float, int]:
        """Bright outlier filter.

        HRSC (integer): histogram gap-based dynamic threshold.
        OSIRIS (float): mu + 5*sigma outlier rejection.
        Hot pixels are replaced with 8-neighbor mean of valid pixels.
        """
        if np.issubdtype(raw.dtype, np.floating):
            # Float path (OSIRIS): 5-sigma
            valid = raw[np.isfinite(raw) & (raw > 0)]
            if valid.size == 0:
                return raw.copy(), 0.0, 0
            mu = float(np.mean(valid))
            sigma = float(np.std(valid))
            thr = mu + HOT_PIXEL_SIGMA * sigma
            hot_mask = (raw > thr) | ~np.isfinite(raw)
            filtered = raw.copy()
            nbr = self._eight_neighbor_mean(raw, ~hot_mask)
            filtered[hot_mask] = nbr[hot_mask]
            return filtered, thr, int(hot_mask.sum())
        # Integer path (HRSC): histogram-based dynamic threshold
        thr = self._compute_dynamic_threshold(raw)
        hot_mask = raw > thr
        filtered = raw.copy()
        nbr = self._eight_neighbor_mean(raw, ~hot_mask)
        filtered[hot_mask] = (
            np.rint(nbr[hot_mask])
            .clip(max(0, int(raw.min())), int(raw.max()))
            .astype(raw.dtype)
        )
        return filtered, float(thr), int(hot_mask.sum())

    def _compute_dynamic_threshold(
        self, raw: np.ndarray, n_bins: int = 256, gap_len: int = 0
    ) -> float:
        """Find hot pixel threshold via histogram gap detection.

        Scans the histogram for the first gap of >= gap_len consecutive
        empty bins AFTER the main data body.  The main data body starts
        at the first bin containing > 1% of total pixels.  This prevents
        a few low-DN outliers from triggering a false gap at the start.
        """
        if gap_len == 0:
            gap_len = HOT_PIXEL_GAP_LEN

        hist, bin_edges = np.histogram(raw.ravel(), bins=n_bins)
        total = hist.sum()

        # Find where the main data body starts (first bin with > 1% of pixels)
        main_start = 0
        threshold_count = total * 0.01
        for i in range(len(hist)):
            if hist[i] > threshold_count:
                main_start = i
                break

        nonzero = hist > 0

        # Fill single isolated empty bins (noise)
        filled = nonzero.copy()
        for i in range(1, len(filled) - 1):
            if not filled[i] and filled[i - 1] and filled[i + 1]:
                filled[i] = True

        # Find first gap AFTER main data body
        false_run = (~filled).astype(int)
        in_gap = False
        gap_start_idx = 0
        for i in range(main_start, len(false_run)):
            if false_run[i] and not in_gap:
                gap_start_idx = i
                in_gap = True
            elif not false_run[i] and in_gap:
                if (i - gap_start_idx) >= gap_len:
                    return float(bin_edges[gap_start_idx])
                in_gap = False

        # No gap found: fallback to max value (no filtering)
        return float(bin_edges[-1])

    def _eight_neighbor_mean(
        self, img: np.ndarray, valid_mask: np.ndarray
    ) -> np.ndarray:
        """Compute 8-neighbor mean using only valid pixels.

        Uses padded shifted arrays for vectorized computation.
        For pixels with no valid neighbors, falls back to the
        mean of all 8 neighbors.
        """
        H, W = img.shape
        padded = np.pad(img.astype(np.float64), 1, mode="edge")
        padded_valid = np.pad(
            valid_mask, 1, mode="constant", constant_values=False
        )

        sum_valid = np.zeros((H, W), dtype=np.float64)
        cnt_valid = np.zeros((H, W), dtype=np.float64)
        sum_all = np.zeros((H, W), dtype=np.float64)

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                shifted = padded[1 + dy : 1 + dy + H, 1 + dx : 1 + dx + W]
                shifted_v = padded_valid[
                    1 + dy : 1 + dy + H, 1 + dx : 1 + dx + W
                ]
                sum_valid += np.where(shifted_v, shifted, 0.0)
                cnt_valid += shifted_v.astype(np.float64)
                sum_all += shifted

        has_valid = cnt_valid > 0
        mean = np.zeros((H, W), dtype=np.float64)
        mean[has_valid] = sum_valid[has_valid] / cnt_valid[has_valid]

        # Fallback for pixels with no valid neighbors
        mean[~has_valid] = sum_all[~has_valid] / 8.0

        return mean

    # -- Step 2: Invalid Pixel Filter -----------------------------------------

    def _filter_invalid(self, img: np.ndarray) -> np.ndarray:
        """Mark invalid pixels as NaN.

        Extended from filter_low_dn():
        - Old: DN<=1 (integer), radiance<=0 (float)
        - New: DN<=0 (negative+zero), radiance<=0, NaN, Inf
        """
        result = img.astype(np.float32)
        if np.issubdtype(img.dtype, np.floating):
            invalid = (result <= 0) | ~np.isfinite(result)
        else:
            invalid = img <= 0  # DN<=0: negative + zero
        result[invalid] = np.nan
        n = int(invalid.sum())
        if n > 0:
            logger.info("  Invalid pixel: %d pixels -> NaN", n)
        return result

    # -- Step 3: Percentile Clipping ------------------------------------------

    @staticmethod
    def _preclip_percentile(
        img: np.ndarray, low_p: float, high_p: float
    ) -> np.ndarray:
        """Clip to [P_low, P_high] and normalize to [0, 1].

        Only runs when NORMALIZE=True.
        """
        valid = img[np.isfinite(img)]
        if valid.size == 0:
            return img
        lo = float(np.percentile(valid, low_p))
        hi = float(np.percentile(valid, high_p))
        result = np.clip(img, lo, hi)
        result = (result - lo) / (hi - lo) if hi > lo else result * 0
        return result.astype(np.float32)

    # -- Step 4: NaN Fill (8-Neighbor) ----------------------------------------

    @staticmethod
    def _fill_nan_8neighbor(img: np.ndarray) -> np.ndarray:
        """Fill NaN pixels using 8-neighbor mean.

        Improved from fill_nan_cardinal():
        - 4-direction -> 8-direction (diagonals included)
        - Invalid neighbors (NaN, <=0, Inf) are excluded
        - Works on a static reference (not iterative)
        """
        original = img
        filled = img.copy()
        nan_mask = np.isnan(original)
        if not nan_mask.any():
            return filled

        # Valid pixel: finite AND > 0
        valid_mask = np.isfinite(original) & (original > 0)

        nan_indices = np.argwhere(nan_mask)
        filled_count = 0
        H, W = original.shape

        for r, c in nan_indices:
            neighbors: list[float] = []
            for dr, dc in [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1),
            ]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and valid_mask[nr, nc]:
                    neighbors.append(float(original[nr, nc]))
            if neighbors:
                filled[r, c] = np.mean(neighbors)
                filled_count += 1

        n_remaining = int(np.isnan(filled).sum())
        logger.info(
            "  NaN fill (8-nbr): %d/%d filled, %d remaining",
            filled_count,
            int(nan_mask.sum()),
            n_remaining,
        )
        return filled

    # -- PDS Parsing ----------------------------------------------------------

    @staticmethod
    def _parse_pds(path: Path, meta: dict) -> np.ndarray:
        """Parse a PDS IMG file into a numpy array.

        Reads the PDS label to find the image offset, then loads
        the raw binary data with the correct dtype and shape.
        Applies display direction corrections (fliplr/flipud) based
        on SAMPLE_DISPLAY_DIRECTION and LINE_DISPLAY_DIRECTION.
        """
        import re as _re

        lines = meta["lines"]
        samples = meta["line_samples"]
        bits = meta["sample_bits"]
        stype = meta["sample_type"]

        # Determine numpy dtype
        if stype == "MSB_INTEGER" and bits == 16:
            dtype = ">i2"  # big-endian int16
        elif stype == "PC_REAL" and bits == 32:
            dtype = "<f4"  # little-endian float32
        else:
            raise ValueError(f"Unknown sample format: {stype}/{bits}")

        with path.open("rb") as f:
            # Parse header for ^IMAGE pointer + display directions
            header = f.read(32000).decode("ascii", errors="ignore")
            img_ptr = 1
            rec_bytes = 0
            for line in header.splitlines():
                line_s = line.strip()
                if line_s.startswith("^IMAGE"):
                    img_ptr = int(line_s.split("=")[1].strip())
                elif line_s.startswith("RECORD_BYTES"):
                    rec_bytes = int(line_s.split("=")[1].strip())

            if rec_bytes == 0:
                rec_bytes = samples * (bits // 8)

            offset = (img_ptr - 1) * rec_bytes
            f.seek(offset)
            data = np.fromfile(f, dtype=dtype, count=lines * samples)

        if data.size < lines * samples:
            raise ValueError(
                f"Insufficient data: expected {lines * samples}, "
                f"got {data.size}"
            )

        data = data.reshape(lines, samples)

        # -- Display direction corrections ------------------------------------
        # SAMPLE_DISPLAY_DIRECTION=LEFT means columns are stored right-to-left
        # in the binary data -> flip horizontally to get correct orientation.
        # This is critical for Rosetta/OSIRIS images.
        sdd_m = _re.search(
            r"SAMPLE_DISPLAY_DIRECTION\s*=\s*(\S+)", header
        )
        ldd_m = _re.search(
            r"LINE_DISPLAY_DIRECTION\s*=\s*(\S+)", header
        )
        sample_dir = sdd_m.group(1).strip('"').upper() if sdd_m else "RIGHT"
        line_dir = ldd_m.group(1).strip('"').upper() if ldd_m else "DOWN"

        if sample_dir == "LEFT":
            data = np.fliplr(data)
            logger.info(
                "  [ORIENT] Flipped X (SAMPLE_DISPLAY_DIRECTION=LEFT): %s",
                path.name,
            )
        if line_dir == "UP":
            data = np.flipud(data)
            logger.info(
                "  [ORIENT] Flipped Y (LINE_DISPLAY_DIRECTION=UP): %s",
                path.name,
            )

        return data

