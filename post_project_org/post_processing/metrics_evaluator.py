#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metrics_evaluator.py
====================
Image quality metrics evaluation with k-sweep and optional cropping.

This module provides:
- SSIM, RMSE, and histogram correlation computation
- k-value sweep optimization (percentile normalization)
- Masked region detection for cropped comparison mode
- Per-IMG custom k-range support

Comparison Modes:
- uncropped: Full image comparison (default)
- cropped: Detect masked areas (+5% buffer), crop both images, then compare

Author: Post-Processing Pipeline
Date: 2025
"""

import numpy as np
import warnings
from pathlib import Path
from PIL import Image
from typing import Tuple, Optional, Dict, List
from collections import defaultdict
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import binary_dilation, gaussian_filter
from scipy.signal import convolve2d
from skimage.registration import phase_cross_correlation
from skimage.transform import warp, AffineTransform, EuclideanTransform
from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac


# Suppress warnings
warnings.filterwarnings("ignore")

# Cache for learned ORB alignment shifts per IMG stem
# When ORB succeeds, its EuclideanTransform is cached here.
# When ORB fails, the median of cached shifts is used instead of phase correlation.
_LEARNED_SHIFT_CACHE = {}  # {img_stem: [EuclideanTransform, ...]}


# ============================================================================
# PER-IMAGE K-RANGE CONFIGURATION
# ============================================================================

# Custom k-ranges for specific IMG files (based on Top-20 SSIM analysis)
# Format: "IMG_STEM": (k_min, k_max)
# If an IMG is not listed here, uses global K_DEFAULT_RANGE
PER_IMG_K_RANGES = {
    "H9463_0050_SR2": (92, 98),
    "H9517_0005_SR2": (79, 88),
    "HF425_0005_SR2": (75, 81),
    "HF102_0005_SR2": (81, 92),
    "HB992_0005_SR2": (79, 85),
    "H8193_0005_SR2": (91, 92),
}

# Default k-range for images not in PER_IMG_K_RANGES
K_DEFAULT_RANGE = (0, 100)

# k-step size for sweep
K_STEP = 1

# Learned k tracking (auto-freeze after repeated sweeps)
LEARNED_K_STATS: Dict[str, Dict[str, object]] = defaultdict(dict)
LEARNED_K_MIN_SAMPLES = 10



def get_k_range(img_stem: str) -> Tuple[int, int]:
    """
    Get k-value range for a specific IMG file.
    
    Args:
        img_stem: IMG filename stem (e.g., "H9463_0050_SR2")
        
    Returns:
        Tuple of (k_min, k_max)
        
    Example:
        >>> k_min, k_max = get_k_range("H9463_0050_SR2")
        >>> print(f"k range: {k_min}-{k_max}")
        k range: 92-98
    """
    k_min, k_max = PER_IMG_K_RANGES.get(img_stem, K_DEFAULT_RANGE)
    
    # Ensure valid range
    k_min = int(max(0, min(100, k_min)))
    k_max = int(max(0, min(100, k_max)))
    
    if k_min > k_max:
        k_min, k_max = k_max, k_min
    
    return k_min, k_max


# ============================================================================
# NORMALIZATION
# ============================================================================

def normalize_image(arr: np.ndarray,
                   low_p: float,
                   high_p: float = 100.0) -> Optional[np.ndarray]:
    """
    Normalize image to [0, 1] using k-threshold + min-max.
    
    Method:
    1. Compute the low_p-th percentile as threshold
    2. Set all values below threshold to 0
    3. Divide all values (zeros included) by max → [0, 1]
    
    This produces a result where the bottom k% of pixels become 0
    and the remaining values are linearly scaled to [0, 1].
    
    Args:
        arr: Input array (any dtype, typically float32)
        low_p: Lower percentile — bottom low_p% pixels are zeroed
        high_p: Kept for API compatibility (unused)
        
    Returns:
        Normalized float32 array in [0, 1], or None if normalization fails
    """
    try:
        # Exclude NaN values from percentile computation
        valid = arr[np.isfinite(arr)] if np.isnan(arr).any() else arr
        
        if valid.size == 0:
            return None
        
        threshold = float(np.percentile(valid, low_p))
        
        # Set values below threshold to 0
        result = arr.copy().astype(np.float32)
        result[result < threshold] = 0.0
        # Replace any remaining NaN with 0
        result = np.nan_to_num(result, nan=0.0)
        
        # Min-max normalize: since min is 0, this is just x / max
        max_val = float(result.max())
        if max_val <= 0:
            return None
        
        result = result / max_val
        return result.astype(np.float32)
        
    except Exception as e:
        print(f"[WARNING] Normalization failed: {e}")
        return None


# ============================================================================
# MASKED REGION DETECTION (FOR CROPPED MODE)
# ============================================================================

def detect_masked_regions(synthetic_norm: np.ndarray,
                         buffer_percent: float = 5.0,
                         mask_threshold: float = 0.01) -> np.ndarray:
    """
    Detect masked/invalid regions in synthetic image.
    
    This function identifies areas that are likely masked (very low values)
    and expands them by a buffer percentage to ensure clean cropping.
    
    Args:
        synthetic_norm: Normalized synthetic image [0, 1]
        buffer_percent: Percentage buffer around masked regions (default: 5%)
        mask_threshold: Threshold below which pixels are considered masked
        
    Returns:
        Boolean mask: True = valid region, False = masked/buffer region
        
    Example:
        >>> synth = np.array(Image.open("synthetic_norm.tif"), dtype=np.float32)
        >>> valid_mask = detect_masked_regions(synth, buffer_percent=5.0)
        >>> print(f"Valid pixels: {valid_mask.sum()}/{valid_mask.size}")
    """
    # Identify masked pixels (very low values)
    masked = synthetic_norm <= mask_threshold
    
    # Calculate buffer size in pixels
    H, W = synthetic_norm.shape
    buffer_pixels = int(max(H, W) * (buffer_percent / 100.0))
    
    # Dilate mask to add buffer
    if buffer_pixels > 0:
        # Create circular structuring element
        y, x = np.ogrid[-buffer_pixels:buffer_pixels+1, -buffer_pixels:buffer_pixels+1]
        structure = (x*x + y*y <= buffer_pixels*buffer_pixels)
        
        # Dilate the mask
        masked_buffered = binary_dilation(masked, structure=structure)
    else:
        masked_buffered = masked
    
    # Valid region is the inverse
    valid_mask = ~masked_buffered
    
    return valid_mask


def crop_to_valid_region(real_norm: np.ndarray,
                        synthetic_norm: np.ndarray,
                        valid_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop both images to the valid (non-masked) region.
    
    Finds the bounding box of the valid region and crops both images to it.
    
    Args:
        real_norm: Normalized real image [0, 1]
        synthetic_norm: Normalized synthetic image [0, 1]
        valid_mask: Boolean mask (True = valid)
        
    Returns:
        Tuple of (cropped_real, cropped_synthetic)
        
    Example:
        >>> valid_mask = detect_masked_regions(synth_norm)
        >>> real_crop, synth_crop = crop_to_valid_region(real_norm, synth_norm, valid_mask)
        >>> print(f"Original: {real_norm.shape}, Cropped: {real_crop.shape}")
        Original: (1024, 1024), Cropped: (950, 980)
    """
    # Find bounding box of valid region
    rows = np.any(valid_mask, axis=1)
    cols = np.any(valid_mask, axis=0)
    
    if not rows.any() or not cols.any():
        # No valid region found - return full images
        print("[WARNING] No valid region found, using full images")
        return real_norm, synthetic_norm
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Crop both images to same bounding box
    real_cropped = real_norm[rmin:rmax+1, cmin:cmax+1]
    synth_cropped = synthetic_norm[rmin:rmax+1, cmin:cmax+1]
    
    return real_cropped, synth_cropped


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def _ssim_at_scale(img1, img2, K1=0.01, K2=0.03, L=1.0):
    """Compute luminance and contrast-structure at a single scale."""
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu1 = gaussian_filter(img1, sigma=1.5)
    mu2 = gaussian_filter(img2, sigma=1.5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    sigma1_sq = np.maximum(gaussian_filter(img1**2, sigma=1.5) - mu1_sq, 0)
    sigma2_sq = np.maximum(gaussian_filter(img2**2, sigma=1.5) - mu2_sq, 0)
    sigma12 = gaussian_filter(img1 * img2, sigma=1.5) - mu1_mu2
    l = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    cs = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    return np.mean(l), np.mean(cs)


def compute_ms_ssim(img1, img2, levels=5):
    """Multi-Scale SSIM. Higher = better. Range [0, 1]."""
    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = min(levels, len(weights))
    mcs_vals = []
    for i in range(levels):
        l_val, cs_val = _ssim_at_scale(img1, img2)
        mcs_vals.append(cs_val)
        if i < levels - 1:
            img1 = img1[::2, ::2]
            img2 = img2[::2, ::2]
            if min(img1.shape) < 8:
                levels = i + 1
                break
    mcs_vals = np.array(mcs_vals[:levels])
    w = weights[:levels]
    w = w / w.sum()
    return float(np.prod(np.maximum(mcs_vals, 1e-10) ** w) * np.maximum(l_val, 1e-10))


def compute_gmsd(img1, img2, c=0.0026):
    """Gradient Magnitude Similarity Deviation. Lower = better."""
    hx = np.array([[1/3, 0, -1/3]] * 3)
    hy = hx.T
    gm1 = np.sqrt(convolve2d(img1, hx, mode='same', boundary='symm')**2 +
                  convolve2d(img1, hy, mode='same', boundary='symm')**2)
    gm2 = np.sqrt(convolve2d(img2, hx, mode='same', boundary='symm')**2 +
                  convolve2d(img2, hy, mode='same', boundary='symm')**2)
    gms = (2 * gm1 * gm2 + c) / (gm1**2 + gm2**2 + c)
    return float(np.std(gms))


def compute_ncc(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Normalized Cross-Correlation (NCC). Range [-1, 1], higher = better.

    Brightness-offset invariant: only measures structural/pattern similarity.
    Zero-means each image before correlating, then normalises by std products.
    NaN-safe: returns NaN if either image is constant.
    """
    a = img1.astype(np.float32).ravel()
    b = img2.astype(np.float32).ravel()
    a = a - a.mean()
    b = b - b.mean()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return float('nan')   # constant image → undefined
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---- LPIPS global model cache (loaded once, reused) ----
_LPIPS_MODEL = None
_LPIPS_AVAILABLE = None


def _get_lpips_model():
    """Lazy-load LPIPS AlexNet model. Returns None if lpips not installed."""
    global _LPIPS_MODEL, _LPIPS_AVAILABLE
    if _LPIPS_AVAILABLE is False:
        return None
    if _LPIPS_MODEL is not None:
        return _LPIPS_MODEL
    try:
        import lpips
        import torch
        # AlexNet: faster than VGG, acceptable accuracy
        _LPIPS_MODEL = lpips.LPIPS(net='alex', verbose=False)
        _LPIPS_MODEL.eval()
        _LPIPS_AVAILABLE = True
        print("[INFO] LPIPS AlexNet model loaded.")
    except ImportError:
        _LPIPS_AVAILABLE = False
        print("[INFO] lpips not installed — skipping LPIPS metric. (pip install lpips)")
    except Exception as e:
        _LPIPS_AVAILABLE = False
        print(f"[WARNING] LPIPS model load failed: {e}")
    return _LPIPS_MODEL


def compute_lpips(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Learned Perceptual Image Patch Similarity (LPIPS, AlexNet backbone).
    Lower = more perceptually similar. Typical range [0, 1].

    Inputs: float32 arrays [0,1], 2D grayscale.
    Requires: pip install lpips torch
    Falls back to NaN if lpips not available.
    """
    model = _get_lpips_model()
    if model is None:
        return float('nan')
    try:
        import torch
        # LPIPS expects (N, 3, H, W) in [-1, 1]
        def _to_lpips_tensor(arr):
            t = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # 1x1xHxW
            t = t.repeat(1, 3, 1, 1)   # grayscale → fake RGB
            t = t * 2.0 - 1.0          # [0,1] → [-1,1]
            return t
        with torch.no_grad():
            t1 = _to_lpips_tensor(img1)
            t2 = _to_lpips_tensor(img2)
            score = model(t1, t2)
        return float(score.item())
    except Exception as e:
        print(f"[WARNING] LPIPS computation failed: {e}")
        return float('nan')


def compute_metrics(real_01: np.ndarray,
                   synthetic_01: np.ndarray,
                   data_range: float = 1.0) -> Tuple[float, float, float, float, float, float, float]:
    """
    Compute image similarity metrics.

    Returns:
        Tuple of (ssim_score, rmse, hist_corr, ms_ssim, gmsd, ncc, lpips)
        - ssim, ms_ssim, ncc: higher is better (max 1)
        - rmse, gmsd, lpips: lower is better (min 0)
        - hist_corr: higher is better (max 1)
    """
    if real_01.shape != synthetic_01.shape:
        raise ValueError(f"Shape mismatch: {real_01.shape} vs {synthetic_01.shape}")

    if np.isnan(real_01).any():
        real_01 = np.nan_to_num(real_01, nan=0.0)
    if np.isnan(synthetic_01).any():
        synthetic_01 = np.nan_to_num(synthetic_01, nan=0.0)

    # 1. SSIM
    try:
        ssim_score = ssim(real_01, synthetic_01, data_range=data_range)
    except Exception as e:
        print(f"[WARNING] SSIM computation failed: {e}")
        ssim_score = np.nan

    # 2. RMSE
    try:
        rmse = float(np.sqrt(np.mean((real_01 - synthetic_01) ** 2)))
    except Exception as e:
        print(f"[WARNING] RMSE computation failed: {e}")
        rmse = np.nan

    # 3. Histogram Correlation
    try:
        h1, _ = np.histogram(real_01.ravel(), bins=256, range=(0.0, 1.0), density=True)
        h2, _ = np.histogram(synthetic_01.ravel(), bins=256, range=(0.0, 1.0), density=True)
        if np.allclose(h1, h1[0]) or np.allclose(h2, h2[0]):
            hist_corr = np.nan
        else:
            hist_corr = float(np.corrcoef(h1, h2)[0, 1])
    except Exception as e:
        print(f"[WARNING] Histogram correlation failed: {e}")
        hist_corr = np.nan

    # 4. MS-SSIM
    try:
        ms_ssim_score = compute_ms_ssim(real_01, synthetic_01)
    except Exception as e:
        print(f"[WARNING] MS-SSIM computation failed: {e}")
        ms_ssim_score = np.nan

    # 5. GMSD
    try:
        gmsd_score = compute_gmsd(real_01, synthetic_01)
    except Exception as e:
        print(f"[WARNING] GMSD computation failed: {e}")
        gmsd_score = np.nan

    # 6. NCC (Normalized Cross-Correlation)
    try:
        ncc_score = compute_ncc(real_01, synthetic_01)
    except Exception as e:
        print(f"[WARNING] NCC computation failed: {e}")
        ncc_score = np.nan

    # 7. LPIPS (perceptual, optional)
    try:
        lpips_score = compute_lpips(real_01, synthetic_01)
    except Exception as e:
        print(f"[WARNING] LPIPS computation failed: {e}")
        lpips_score = np.nan

    return ssim_score, rmse, hist_corr, ms_ssim_score, gmsd_score, ncc_score, lpips_score

def align_and_crop_with_buffer(
    imgA: np.ndarray,
    imgB: np.ndarray,
    buffer_pixels: Optional[int] = None,
    buffer_percent: Optional[float] = 5.0,
    mask_threshold: float = 0.02,
    upsample_factor: int = 100,
    interpolation_order: int = 1,
    buffer_ratio: Optional[float] = None,
    max_buffer_frac: float = 0.20,
) -> dict:
    """
    Align two images using phase cross-correlation and crop to the
    valid content overlap region (INWARD buffer).
    
    The buffer SHRINKS the overlap inward to exclude warp edge artifacts.
    max_buffer_frac (default 0.20 = 20%) caps the buffer so the crop
    cannot collapse below (1 - 2*max_buffer_frac) of each overlap dimension.
    
    Returns A_crop, B_crop, shift, bbox, and used buffer_px.
    """
    if imgA.ndim != 2 or imgB.ndim != 2:
        raise ValueError("Only single-channel (2D) images are supported.")

    A = imgA.astype("float32", copy=False)
    B = imgB.astype("float32", copy=False)
    H, W = A.shape

    # --- 1) Sub-pixel shift ---
    shift, error, diffphase = phase_cross_correlation(A, B, upsample_factor=upsample_factor)
    dy, dx = float(shift[0]), float(shift[1])
    print(f"  Phase correlation shift: dy={dy:.2f}, dx={dx:.2f} pixels")

    # --- 2) Warp B onto A's frame ---
    tform = AffineTransform(translation=(dx, dy))
    B_aligned = warp(
        B, inverse_map=tform.inverse, output_shape=A.shape,
        preserve_range=True, mode="constant", cval=np.nan, order=interpolation_order
    ).astype("float32")

    # --- 3) Content overlap mask ---
    finite_mask = np.isfinite(B_aligned)
    content_mask = finite_mask & (B_aligned > mask_threshold)

    if not np.any(content_mask):
        print("[WARNING] Content mask empty; falling back to finite overlap.")
        content_mask = finite_mask

    ys, xs = np.where(content_mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    overlap_h = y1 - y0
    overlap_w = x1 - x0
    print(f"  Content overlap: y=[{y0}:{y1}], x=[{x0}:{x1}] ({overlap_h}×{overlap_w} px)")

    # --- 4) Compute INWARD buffer ---
    if buffer_pixels is None:
        if buffer_ratio is not None:
            buffer_px = max(0, int(round(buffer_ratio * max(H, W))))
        elif buffer_percent is not None:
            buffer_px = max(0, int(round((buffer_percent / 100.0) * max(H, W))))
        else:
            buffer_px = 0
    else:
        buffer_px = int(max(0, buffer_pixels))

    # Cap buffer so crop doesn't collapse
    # max_buffer_frac=0.20 → each side can eat at most 20% of overlap dimension
    max_buf_y = max(1, int(overlap_h * max_buffer_frac))
    max_buf_x = max(1, int(overlap_w * max_buffer_frac))
    effective_buf = min(buffer_px, max_buf_y, max_buf_x)

    # Shrink INWARD from content overlap edges
    y0b = y0 + effective_buf
    y1b = y1 - effective_buf
    x0b = x0 + effective_buf
    x1b = x1 - effective_buf

    # Safety: if still collapsed, fall back to no buffer
    if y1b <= y0b or x1b <= x0b:
        print(f"  ⚠️ Inward buffer ({effective_buf}px) would collapse crop; using no buffer")
        y0b, y1b, x0b, x1b = y0, y1, x0, x1
        effective_buf = 0

    crop_h = y1b - y0b
    crop_w = x1b - x0b
    print(f"  Inward buffer: {effective_buf} px (requested {buffer_px}, max_frac={max_buffer_frac:.0%})")
    print(f"  Final crop: y=[{y0b}:{y1b}], x=[{x0b}:{x1b}] ({crop_h}×{crop_w} px)")

    # --- MINIMUM CROP GUARD ---
    MIN_CROP_PX = 100
    if crop_h < MIN_CROP_PX or crop_w < MIN_CROP_PX:
        # Phase correlation returned garbage — fallback to 80% center crop
        center_frac = 0.80
        margin_y = int(H * (1 - center_frac) / 2)
        margin_x = int(W * (1 - center_frac) / 2)
        y0b = margin_y
        y1b = H - margin_y
        x0b = margin_x
        x1b = W - margin_x
        crop_h = y1b - y0b
        crop_w = x1b - x0b
        print(f"  ⚠️ CROP TOO SMALL — fallback to 80% center crop: y=[{y0b}:{y1b}], x=[{x0b}:{x1b}] ({crop_h}×{crop_w} px)")

    # --- 5) Crop (both images guaranteed valid in this region) ---
    A_crop = A[y0b:y1b, x0b:x1b]
    B_crop = B_aligned[y0b:y1b, x0b:x1b]

    return {
        "A_crop": A_crop,
        "B_crop": B_crop,
        "shift": (dy, dx),
        "bbox": (y0b, y1b, x0b, x1b),
        "buffer_px": effective_buf,
    }



# ============================================================================
# K-SWEEP EVALUATION
# ============================================================================

def evaluate_k_sweep(real_raw: np.ndarray,
                     synthetic_norm: np.ndarray,
                     img_stem: str,
                     mode: str = "uncropped",
                     buffer_pixels: Optional[int] = None,
                     buffer_ratio: Optional[float] = None,
                     buffer_percent: float = 5.0,
                     verbose: bool = False,
                     alignment_strategy: str = "once",
                     prescan_step: int = 0) -> Dict:
    """
    Perform k-value sweep with alignment-based cropping.
    
    UPDATED: Now uses phase cross-correlation alignment instead of masked region detection.
    
    Args:
        real_raw: Raw real image (uint16 or float32)
        synthetic_norm: Normalized synthetic image [0, 1]
        img_stem: IMG filename stem
        mode: "uncropped" or "cropped"
            - uncropped: Direct comparison after normalization
            - cropped: Phase correlation alignment + overlap crop with buffer
        buffer_percent: Buffer percentage for overlap expansion (default: 5.0%)
        verbose: Verbose output
        
    Returns:
        Dictionary with evaluation results
    """
    if mode not in {"uncropped", "cropped"}:
        raise ValueError(f"Invalid mode: {mode}. Must be 'uncropped' or 'cropped'")

    k_min, k_max = get_k_range(img_stem)
    k_values = list(range(k_min, k_max + 1, K_STEP))

    print(f"\n{'='*80}")
    print(f"K-Sweep Evaluation: {img_stem}")
    print(f"Mode: {mode}")
    print(f"K-range: {k_min} to {k_max} (step {K_STEP})")
    if mode == "cropped":
        H, W = real_raw.shape
        est_buf_px = int(round((buffer_percent/100.0) * max(H, W)))
        print("Alignment method: Phase cross-correlation")
        print(f"Buffer: {buffer_percent}% ≈ {est_buf_px} px")
    print(f"{'='*80}")

    real_raw_f32 = real_raw.astype(np.float32)

    # ---- TEK KEZ HİZALAMA (alignment_strategy='once') ----
    synth_crop_ref = None
    crop_bbox = None
    if mode == "cropped" and alignment_strategy == "once":
        # k_ref seç: prescan varsa hızlıca, yoksa orta değer
        if prescan_step and prescan_step > 1:
            probe = list(range(k_min, k_max + 1, prescan_step))
            best_s, k_ref = -np.inf, (k_min + k_max) // 2
            for kk in probe:
                rn = normalize_image(real_raw_f32, low_p=float(kk), high_p=100.0)
                if rn is None:
                    continue
                try:
                    ar = align_and_crop_with_buffer(rn, synthetic_norm,
                                                    buffer_pixels=None,
                                                    buffer_percent=buffer_percent,
                                                    mask_threshold=0.02,
                                                    upsample_factor=50,
                                                    interpolation_order=1)
                    s_tmp, _, _, _, _, _, _ = compute_metrics(ar["A_crop"], ar["B_crop"])
                    if np.isfinite(s_tmp) and s_tmp > best_s:
                        best_s, k_ref = s_tmp, kk
                except:
                    pass
        else:
            k_ref = (k_min + k_max) // 2

        real_ref = normalize_image(real_raw_f32, low_p=float(k_ref), high_p=100.0)
        ar = align_and_crop_with_buffer(real_ref, synthetic_norm,
                                        buffer_pixels=None,
                                        buffer_percent=buffer_percent,
                                        mask_threshold=0.02,
                                        upsample_factor=50,
                                        interpolation_order=1)
        (dy, dx) = ar["shift"]
        (y0b, y1b, x0b, x1b) = ar["bbox"]

        # synth’i bir kez warpla ve sabit ROI hazırla
        tform = AffineTransform(translation=(dx, dy))
        synth_aligned = warp(synthetic_norm,
                             inverse_map=tform.inverse,
                             output_shape=real_raw.shape,
                             preserve_range=True,
                             mode="constant", cval=np.nan, order=1).astype("float32")
        synth_crop_ref = synth_aligned[y0b:y1b, x0b:x1b]
        crop_bbox = (y0b, y1b, x0b, x1b)

    all_results = []
    best_ssim, best_ssim_k = -np.inf, None
    best_rmse, best_rmse_k = np.inf, None
    best_combined, best_combined_k = np.inf, None
    cropped_shape = None

    for idx, k in enumerate(k_values, start=1):
        real_norm = normalize_image(real_raw_f32, low_p=float(k), high_p=100.0)
        if real_norm is None:
            if verbose: print(f"[k={k:3d}] Normalization failed")
            continue

        if mode == "cropped":
            if alignment_strategy == "once" and crop_bbox is not None:
                y0b, y1b, x0b, x1b = crop_bbox
                real_comp  = real_norm[y0b:y1b, x0b:x1b]
                synth_comp = synth_crop_ref
                cropped_shape = real_comp.shape
            else:
                ar = align_and_crop_with_buffer(real_norm, synthetic_norm,
                                                buffer_pixels=None,
                                                buffer_percent=buffer_percent,
                                                mask_threshold=0.02,
                                                upsample_factor=100,
                                                interpolation_order=1)
                real_comp, synth_comp = ar["A_crop"], ar["B_crop"]
                cropped_shape = real_comp.shape
        else:
            real_comp, synth_comp = real_norm, synthetic_norm

        s_val, r_val, h_val, ms_val, gm_val, ncc_val, lpips_val = compute_metrics(real_comp, synth_comp)
        # Combined skor: (1 - SSIM) + RMSE / 3
        c_val = (1.0 - s_val) + r_val / 3.0 if (np.isfinite(s_val) and np.isfinite(r_val)) else np.inf
        all_results.append({'k': k, 'ssim': s_val, 'rmse': r_val, 'hist_corr': h_val,
                            'ms_ssim': ms_val, 'gmsd': gm_val, 'ncc': ncc_val, 'lpips': lpips_val,
                            'combined': c_val})

        if np.isfinite(s_val) and s_val > best_ssim:
            best_ssim, best_ssim_k = s_val, k
        if np.isfinite(r_val) and r_val < best_rmse:
            best_rmse, best_rmse_k = r_val, k
        if np.isfinite(c_val) and c_val < best_combined:
            best_combined, best_combined_k = c_val, k

        if verbose or (idx % 10 == 0) or (idx == len(k_values)):
            pct = 100.0 * idx / len(k_values)
            print(f"[{pct:5.1f}%] k={k:3d} | SSIM={s_val:.4f} | RMSE={r_val:.4f} | Hist_Corr={h_val:.4f}", end="\r")

    print("\n\n" + "="*80)
    print("Results Summary:")
    print(f"  Best SSIM: {best_ssim:.6f} at k={best_ssim_k}")
    print(f"  Best RMSE: {best_rmse:.6f} at k={best_rmse_k}")
    if mode == "cropped":
        print(f"  Alignment: Phase cross-correlation (strategy={alignment_strategy})")
    print("="*80 + "\n")

    return {
        'img_stem': img_stem,
        'k_range': (k_min, k_max),
        'all_results': all_results,
        'best_ssim_k': best_ssim_k,
        'best_ssim_score': best_ssim,
        'best_rmse_k': best_rmse_k,
        'best_rmse_score': best_rmse,
        'best_combined_k': best_combined_k,
        'best_combined_score': best_combined,
        'mode': mode,
        'cropped_shape': cropped_shape,
        'alignment_method': 'phase_correlation' if mode == 'cropped' else 'none',
        'crop_bbox': crop_bbox if mode == 'cropped' else None,
        'buffer_pixels': None,
        'buffer_source': None
    }

def _record_learned_k(img_stem: str,
                      res: Dict,
                      learned_k_db: Optional[Dict[str, int]],
                      objective_type: str = 'ssim') -> None:
    """Record learned k based on the active objective type.

    Args:
        objective_type: 'ssim' | 'rmse' | 'nmrse' | 'combined'
    """
    stats = LEARNED_K_STATS[img_stem]
    stats['count'] = int(stats.get('count', 0)) + 1

    # Objective'e göre k ve skor seç
    if objective_type in ('rmse', 'nmrse'):
        chosen_k = int(res['best_rmse_k'])
        chosen_score = float(res.get('best_rmse_score', float('inf')))
        is_better = chosen_score < float(stats.get('best_score', float('inf')))
    elif objective_type == 'combined':
        chosen_k = int(res.get('best_combined_k', res['best_ssim_k']))
        chosen_score = float(res.get('best_combined_score', float('inf')))
        is_better = chosen_score < float(stats.get('best_score', float('inf')))
    else:  # ssim (default)
        chosen_k = int(res['best_ssim_k'])
        chosen_score = float(res.get('best_ssim_score', float('-inf')))
        is_better = chosen_score > float(stats.get('best_score', float('-inf')))

    history = stats.setdefault('history', [])
    history.append(chosen_k)
    if is_better:
        stats['best_score'] = chosen_score
        stats['best_k'] = chosen_k
    if (
        learned_k_db is not None
        and img_stem not in learned_k_db
        and stats.get('count', 0) >= LEARNED_K_MIN_SAMPLES
        and 'best_k' in stats
    ):
        learned_k_db[img_stem] = int(stats['best_k'])


def _evaluate_fixed_k(real_raw: np.ndarray,
                      synthetic_norm: np.ndarray,
                      img_stem: str,
                      mode: str,
                      buffer_pixels: Optional[int],
                      buffer_percent: float,
                      k_value: int) -> Dict:
    """ORB+RANSAC alignment evaluation: feature matching → robust warp → overlap crop → metrics.
    
    Uses ORB keypoint detection + RANSAC to find a robust EuclideanTransform
    (translation + rotation) between real and synthetic images. Falls back to
    phase correlation if ORB fails (too few features/matches).
    """
    real_raw_f32 = real_raw.astype(np.float32)
    real_norm = normalize_image(real_raw_f32, float(k_value))
    if real_norm is None:
        return {'error': 'Normalization failed', 'img_stem': img_stem}
    
    alignment_shift = None
    real_comp = real_norm
    synth_comp = synthetic_norm

    if mode == "cropped":
        MIN_OVERLAP_RATIO = 0.50  # Overlap must be ≥50% of original
        H, W = real_norm.shape
        full_area = H * W
        alignment_method = 'none'
        
        try:
            # ---- PRIMARY: ORB + RANSAC alignment ----
            orb_success = False
            orb_outlier_rejected = False
            orb_penalty = 0.0
            orb_overlap_failed = False
            
            cached = _LEARNED_SHIFT_CACHE.get(img_stem, [])
            shift_locked = len(cached) >= 10  # Lock after 10 successful alignments
            
            if shift_locked:
                # ===== LOCKED MODE: Use consensus median for alignment =====
                # Step 1: Compute locked shift (deterministic)
                locked_txs = [t.translation[0] for t in cached]
                locked_tys = [t.translation[1] for t in cached]
                locked_rots = [t.rotation for t in cached]
                med_tx = float(np.median(locked_txs))
                med_ty = float(np.median(locked_tys))
                med_rot = float(np.median(locked_rots))
                
                model_locked = EuclideanTransform(
                    rotation=med_rot,
                    translation=(med_tx, med_ty)
                )
                alignment_shift = (med_ty, med_tx)  # (dy, dx) convention
                
                # Step 2: Run ORB as HEALTH CHECK ONLY (not for alignment)
                orb_healthy = False
                try:
                    orb = ORB(n_keypoints=1000)
                    orb.detect_and_extract(real_norm)
                    kp_real = orb.keypoints
                    desc_real = orb.descriptors
                    orb.detect_and_extract(synthetic_norm)
                    kp_synth = orb.keypoints
                    desc_synth = orb.descriptors
                    
                    if desc_real is not None and desc_synth is not None and len(desc_real) >= 10 and len(desc_synth) >= 10:
                        matches = match_descriptors(desc_real, desc_synth, cross_check=True)
                        if len(matches) >= 10:
                            src = kp_synth[matches[:, 1]][:, ::-1]
                            dst = kp_real[matches[:, 0]][:, ::-1]
                            model_check, inliers = ransac(
                                (src, dst), EuclideanTransform,
                                min_samples=3, residual_threshold=2, max_trials=1000
                            )
                            n_inliers = int(inliers.sum()) if inliers is not None else 0
                            
                            if model_check is not None and n_inliers >= 5:
                                check_tx, check_ty = model_check.translation
                                dist = np.sqrt((check_tx - med_tx)**2 + (check_ty - med_ty)**2)
                                print(f"  ORB health check: {n_inliers} inliers, "
                                      f"shift=(tx={check_tx:.1f}, ty={check_ty:.1f}), "
                                      f"deviation={dist:.1f}px from locked")
                                if dist <= 20.0:
                                    orb_healthy = True
                                    # Cache fresh ORB result to keep median updated
                                    _LEARNED_SHIFT_CACHE[img_stem].append(model_check)
                                    _LEARNED_SHIFT_CACHE[img_stem] = _LEARNED_SHIFT_CACHE[img_stem][-20:]
                                else:
                                    print(f"  ⚠️ ORB outlier: deviation {dist:.1f}px > 20px threshold")
                            else:
                                print(f"  ⚠️ ORB health check: not enough inliers ({n_inliers})")
                        else:
                            print(f"  ⚠️ ORB health check: not enough matches ({len(matches)})")
                    else:
                        n_r = len(desc_real) if desc_real is not None else 0
                        n_s = len(desc_synth) if desc_synth is not None else 0
                        print(f"  ⚠️ ORB health check: not enough features (real={n_r}, synth={n_s})")
                except Exception as orb_err:
                    print(f"  ⚠️ ORB health check failed: {orb_err}")
                
                # Step 3: Choose alignment model based on health check
                if orb_healthy:
                    # ORB passed → use fresh ORB alignment for metrics & inspection
                    synth_aligned = warp(
                        synthetic_norm,
                        inverse_map=model_check.inverse,
                        output_shape=real_norm.shape,
                        preserve_range=True,
                        mode="constant",
                        cval=0.0,
                        order=1
                    ).astype("float32")
                    model_robust = model_check  # For mask warp
                    alignment_shift = (check_ty, check_tx)  # Update to fresh shift
                    alignment_method = 'locked_orb_fresh'
                    orb_penalty = 0.0
                    print(f"  ✅ Using fresh ORB alignment: "
                          f"rotation={np.degrees(model_check.rotation):.3f}°, "
                          f"tx={check_tx:.2f}, ty={check_ty:.2f}")
                else:
                    # ORB failed → fallback to locked median + penalty
                    synth_aligned = warp(
                        synthetic_norm,
                        inverse_map=model_locked.inverse,
                        output_shape=real_norm.shape,
                        preserve_range=True,
                        mode="constant",
                        cval=0.0,
                        order=1
                    ).astype("float32")
                    model_robust = model_locked  # For mask warp
                    orb_outlier_rejected = True
                    orb_penalty = 0.5  # Durum 1: locked mode health fail
                    alignment_method = 'locked_consensus'
                
                orb_success = True  # Use mask-based crop path
                print(f"  🔒 Locked shift (median of {len(cached)}): "
                      f"rotation={np.degrees(med_rot):.3f}°, tx={med_tx:.2f}, ty={med_ty:.2f}"
                      f"{' ✅ORB-aligned' if orb_healthy else ' ⚠️+penalty(0.5)'}")
                
            else:
                # ===== BUILDING MODE: Use ORB directly (cache < 10) =====
                try:
                    orb = ORB(n_keypoints=1000)
                    
                    orb.detect_and_extract(real_norm)
                    kp_real = orb.keypoints
                    desc_real = orb.descriptors
                    
                    orb.detect_and_extract(synthetic_norm)
                    kp_synth = orb.keypoints
                    desc_synth = orb.descriptors
                    
                    if desc_real is not None and desc_synth is not None and len(desc_real) >= 10 and len(desc_synth) >= 10:
                        matches = match_descriptors(desc_real, desc_synth, cross_check=True)
                        
                        if len(matches) >= 10:
                            src = kp_synth[matches[:, 1]][:, ::-1]
                            dst = kp_real[matches[:, 0]][:, ::-1]
                            
                            model_robust, inliers = ransac(
                                (src, dst), EuclideanTransform,
                                min_samples=3, residual_threshold=2, max_trials=1000
                            )
                            
                            n_inliers = int(inliers.sum()) if inliers is not None else 0
                            print(f"  ORB matches: {len(matches)}, inliers: {n_inliers}")
                            
                            if model_robust is not None and n_inliers >= 5:
                                rot_deg = float(np.degrees(model_robust.rotation))
                                tx, ty = model_robust.translation
                                alignment_shift = (ty, tx)
                                print(f"  ORB+RANSAC: rotation={rot_deg:.3f}°, tx={tx:.2f}, ty={ty:.2f}")
                                
                                # Consensus check (if enough cache)
                                if len(cached) >= 3:
                                    c_med_tx = float(np.median([t.translation[0] for t in cached]))
                                    c_med_ty = float(np.median([t.translation[1] for t in cached]))
                                    dist = np.sqrt((tx - c_med_tx)**2 + (ty - c_med_ty)**2)
                                    if dist > 20.0:
                                        print(f"  ⚠️ ORB outlier rejected: shift deviates {dist:.1f}px from consensus "
                                              f"(median tx={c_med_tx:.1f}, ty={c_med_ty:.1f})")
                                        orb_outlier_rejected = True
                                        orb_penalty = 1.0  # Durum 2: building deviation > 20px
                                    else:
                                        synth_aligned = warp(
                                            synthetic_norm,
                                            inverse_map=model_robust.inverse,
                                            output_shape=real_norm.shape,
                                            preserve_range=True,
                                            mode="constant",
                                            cval=0.0,
                                            order=1
                                        ).astype("float32")
                                        orb_success = True
                                        alignment_method = 'orb_ransac'
                                else:
                                    synth_aligned = warp(
                                        synthetic_norm,
                                        inverse_map=model_robust.inverse,
                                        output_shape=real_norm.shape,
                                        preserve_range=True,
                                        mode="constant",
                                        cval=0.0,
                                        order=1
                                    ).astype("float32")
                                    orb_success = True
                                    alignment_method = 'orb_ransac'
                            else:
                                print(f"  ⚠️ ORB+RANSAC: not enough inliers ({n_inliers}), falling back")
                        else:
                            print(f"  ⚠️ ORB: not enough matches ({len(matches)}), falling back")
                    else:
                        n_r = len(desc_real) if desc_real is not None else 0
                        n_s = len(desc_synth) if desc_synth is not None else 0
                        print(f"  ⚠️ ORB: not enough features (real={n_r}, synth={n_s}), falling back")
                except Exception as orb_err:
                    print(f"  ⚠️ ORB failed: {orb_err}, falling back")
                
                # Cache successful ORB shift
                if orb_success:
                    if img_stem not in _LEARNED_SHIFT_CACHE:
                        _LEARNED_SHIFT_CACHE[img_stem] = []
                    _LEARNED_SHIFT_CACHE[img_stem].append(model_robust)
                    _LEARNED_SHIFT_CACHE[img_stem] = _LEARNED_SHIFT_CACHE[img_stem][-20:]
                
                # Fallback: cached shift or phase correlation
                if not orb_success:
                    if cached:
                        cached_txs = [t.translation[0] for t in cached]
                        cached_tys = [t.translation[1] for t in cached]
                        cached_rots = [t.rotation for t in cached]
                        med_tx = float(np.median(cached_txs))
                        med_ty = float(np.median(cached_tys))
                        med_rot = float(np.median(cached_rots))
                        
                        model_cached = EuclideanTransform(
                            rotation=med_rot,
                            translation=(med_tx, med_ty)
                        )
                        alignment_shift = (med_ty, med_tx)
                        print(f"  📌 Using cached ORB shift (median of {len(cached)}): "
                              f"rotation={np.degrees(med_rot):.3f}°, tx={med_tx:.2f}, ty={med_ty:.2f}")
                        
                        synth_aligned = warp(
                            synthetic_norm,
                            inverse_map=model_cached.inverse,
                            output_shape=real_norm.shape,
                            preserve_range=True,
                            mode="constant",
                            cval=0.0,
                            order=1
                        ).astype("float32")
                        alignment_method = 'cached_orb_shift'
                        model_robust = model_cached
                        orb_success = True
                        orb_outlier_rejected = True
                        orb_penalty = 1.5  # Durum 3: building ORB fail, cached shift
                    else:
                        shift, error, diffphase = phase_cross_correlation(
                            real_norm, synthetic_norm, upsample_factor=100
                        )
                        dy, dx = float(shift[0]), float(shift[1])
                        alignment_shift = (dy, dx)
                        print(f"  Phase correlation fallback (no cache): dy={dy:.2f}, dx={dx:.2f} pixels")
                        
                        tform = AffineTransform(translation=(dx, dy))
                        synth_aligned = warp(
                            synthetic_norm,
                            inverse_map=tform.inverse,
                            output_shape=real_norm.shape,
                            preserve_range=True,
                            mode="constant",
                            cval=0.0,
                            order=1
                        ).astype("float32")
                        alignment_method = 'phase_correlation_fallback'
                        orb_outlier_rejected = True
                        orb_penalty = 2.0  # Durum 4: building ORB fail, no cache
            
            # ---- OVERLAP CROP (mask-based, works for rotation too) ----
            mask_ones = np.ones_like(synthetic_norm)
            if orb_success:
                aligned_mask = warp(
                    mask_ones, inverse_map=model_robust.inverse,
                    output_shape=real_norm.shape, preserve_range=True,
                    mode="constant", cval=0.0, order=0
                )
            else:
                aligned_mask = warp(
                    mask_ones, inverse_map=tform.inverse,
                    output_shape=real_norm.shape, preserve_range=True,
                    mode="constant", cval=0.0, order=0
                )
            
            # Find bounding box of valid overlap
            valid_rows = np.any(aligned_mask > 0.9, axis=1)
            valid_cols = np.any(aligned_mask > 0.9, axis=0)
            
            if not np.any(valid_rows) or not np.any(valid_cols):
                print(f"  [!] No overlap found after ORB alignment. Falling back to phase correlation.")
                # Fall through to phase correlation fallback below
                orb_overlap_failed = True
            else:
                y_start, y_end = int(np.where(valid_rows)[0][0]), int(np.where(valid_rows)[0][-1]) + 1
                x_start, x_end = int(np.where(valid_cols)[0][0]), int(np.where(valid_cols)[0][-1]) + 1
                crop_h, crop_w = y_end - y_start, x_end - x_start
                overlap_area = crop_h * crop_w
                overlap_ratio = overlap_area / full_area if full_area > 0 else 0.0
                orb_overlap_failed = False

                if overlap_ratio < MIN_OVERLAP_RATIO:
                    print(f"  [!] ORB overlap too small: {overlap_ratio:.1%} < {MIN_OVERLAP_RATIO:.0%}. Falling back to phase correlation.")
                    orb_overlap_failed = True

            # ---- PHASE CORRELATION FALLBACK when ORB alignment gives bad overlap ----
            if orb_overlap_failed:
                shift, error, diffphase = phase_cross_correlation(
                    real_norm, synthetic_norm, upsample_factor=100
                )
                dy, dx = float(shift[0]), float(shift[1])
                alignment_shift = (dy, dx)
                print(f"  Phase correlation rescue: dy={dy:.2f}, dx={dx:.2f} pixels")

                tform = AffineTransform(translation=(dx, dy))
                synth_aligned = warp(
                    synthetic_norm,
                    inverse_map=tform.inverse,
                    output_shape=real_norm.shape,
                    preserve_range=True,
                    mode="constant",
                    cval=0.0,
                    order=1
                ).astype("float32")
                alignment_method = 'phase_correlation_rescue'
                orb_outlier_rejected = True
                orb_penalty = 2.5  # Heaviest penalty: ORB overlap failed, rescued by phase corr

                # Recompute overlap with phase correlation result
                mask_ones_pc = np.ones_like(synthetic_norm)
                aligned_mask_pc = warp(
                    mask_ones_pc, inverse_map=tform.inverse,
                    output_shape=real_norm.shape, preserve_range=True,
                    mode="constant", cval=0.0, order=0
                )
                valid_rows_pc = np.any(aligned_mask_pc > 0.9, axis=1)
                valid_cols_pc = np.any(aligned_mask_pc > 0.9, axis=0)

                if not np.any(valid_rows_pc) or not np.any(valid_cols_pc):
                    # Even phase correlation failed — genuine penalty
                    print(f"  [!!] Phase correlation also has no overlap. Returning penalty.")
                    return {
                        'img_stem': img_stem, 'k_range': (k_value, k_value),
                        'all_results': [{'k': k_value, 'ssim': 0.0, 'rmse': 1.0,
                                         'hist_corr': 0.0, 'ms_ssim': 0.0, 'gmsd': 0.5,
                                         'ncc': float('nan'), 'lpips': float('nan')}],
                        'best_ssim_k': k_value, 'best_ssim_score': 0.0,
                        'best_rmse_k': k_value, 'best_rmse_score': 1.0,
                        'best_ms_ssim': 0.0, 'best_gmsd': 0.5,
                        'best_ncc': float('nan'), 'best_lpips': float('nan'),
                        'mode': mode, 'crop_bbox': None,
                        'alignment_method': 'penalty_no_overlap',
                        'aligned_real': None, 'aligned_synth': None,
                        'alignment_shift': alignment_shift, 'cropped_shape': (0, 0),
                        'orb_outlier_rejected': True, 'orb_penalty': 3.0,
                    }

                y_start = int(np.where(valid_rows_pc)[0][0])
                y_end   = int(np.where(valid_rows_pc)[0][-1]) + 1
                x_start = int(np.where(valid_cols_pc)[0][0])
                x_end   = int(np.where(valid_cols_pc)[0][-1]) + 1
                crop_h, crop_w = y_end - y_start, x_end - x_start
                overlap_ratio = (crop_h * crop_w) / full_area if full_area > 0 else 0.0

            real_comp = real_norm[y_start:y_end, x_start:x_end]
            synth_comp = synth_aligned[y_start:y_end, x_start:x_end]
            print(f"  Overlap crop: y=[{y_start}:{y_end}], x=[{x_start}:{x_end}] "
                  f"-> {real_comp.shape} ({overlap_ratio:.1%} of original)")
            
        except Exception as e:
            print(f"  [!] Alignment failed: {e}, using unaligned")
            synth_comp = synthetic_norm

    s_val, r_val, h_val, ms_val, gm_val, ncc_val, lpips_val = compute_metrics(real_comp, synth_comp)

    return {
        'img_stem': img_stem,
        'k_range': (k_value, k_value),
        'all_results': [{'k': k_value, 'ssim': s_val, 'rmse': r_val, 'hist_corr': h_val,
                         'ms_ssim': ms_val, 'gmsd': gm_val, 'ncc': ncc_val, 'lpips': lpips_val}],
        'best_ssim_k': k_value,
        'best_ssim_score': s_val,
        'best_rmse_k': k_value,
        'best_rmse_score': r_val,
        'best_ms_ssim': ms_val,
        'best_gmsd': gm_val,
        'best_ncc': ncc_val,
        'best_lpips': lpips_val,
        'mode': mode,
        'cropped_shape': real_comp.shape,
        'alignment_method': alignment_method if mode == 'cropped' else 'none',
        'buffer_pixels': buffer_pixels,
        'buffer_source': None,
        # Aligned images for inspection saving (avoid re-alignment)
        'aligned_real': real_comp,
        'aligned_synth': synth_comp,
        'alignment_shift': alignment_shift,
        'crop_bbox': None,
        'orb_outlier_rejected': orb_outlier_rejected if mode == 'cropped' else False,
        'orb_penalty': orb_penalty if mode == 'cropped' else 0.0,
    }



def finalize_learned_k(img_stem: str, learned_k_db: Dict[str, int]) -> None:
    """Force commit the best k seen so far for img_stem into learned_k_db."""
    if img_stem in LEARNED_K_STATS:
        stats = LEARNED_K_STATS[img_stem]
        if 'best_k' in stats:
            learned_k_db[img_stem] = int(stats['best_k'])

def evaluate_k(real_raw: np.ndarray,
               synthetic_norm: np.ndarray,
               img_stem: str,
               mode: str = "uncropped",
               buffer_pixels: Optional[int] = None,
               buffer_ratio: Optional[float] = None,
               buffer_percent: float = 5.0,
               verbose: bool = False,
               k_mode: str = "sweep",
               learned_k_db: Optional[Dict[str, int]] = None,
               guard_ssim: float = 0.70,
               fallback_delta: int = 5,
               fixed_bbox: Optional[Tuple[int, int, int, int]] = None,
               fixed_k: Optional[int] = None,
               objective_type: str = 'ssim') -> Dict:
    """
    Evaluate image pair with optional learned-k shortcut.

    If a fixed k has been stored for ``img_stem`` inside ``learned_k_db`` and
    ``k_mode`` is not forced to ``'sweep'``, that value is used directly without
    running a k-sweep. Otherwise a normal sweep is executed. After each sweep the
    best-SSIM k is recorded; once at least ``LEARNED_K_MIN_SAMPLES`` sweeps have
    been observed for the same IMG, the best-performing k is frozen inside
    ``learned_k_db`` so subsequent evaluations reuse it.
    
    NEW: If fixed_bbox is provided, skip full alignment and use the provided
    bounding box with minimal sub-pixel alignment.
    """
    
    # ============ FIXED BBOX MODE (Adaptive Crop) ============
    if fixed_bbox is not None and mode == "cropped":
        y0, y1, x0, x1 = fixed_bbox
        real_raw_f32 = real_raw.astype(np.float32)
        
        # 1. k değerini belirle
        # Öncelik: fixed_k > learned_k > sweep
        k_star = None
        
        if fixed_k is not None:
            k_star = int(fixed_k)
            if verbose: print(f"  🔒 Using fixed k={k_star} (CLI argument)")
            
        elif learned_k_db is not None and img_stem in learned_k_db:
            k_star = int(learned_k_db[img_stem])
            if verbose: print(f"  🧠 Using learned k={k_star}")
            
        if k_star is not None:
            # Tek bir k ile değerlendir
            real_norm = normalize_image(real_raw_f32, low_p=float(k_star), high_p=100.0)
            if real_norm is None:
                return {'error': 'Normalization failed', 'crop_bbox': fixed_bbox}
            
            # Crop with fixed bbox
            real_crop = real_norm[y0:y1, x0:x1]
            synth_crop = synthetic_norm[y0:y1, x0:x1]
            
            # --- CROP AREA GUARD ---
            # Eğer crop çok küçükse (örn. orijinalin %1'inden az), hatalı SSIM vermesin
            full_area = real_raw.shape[0] * real_raw.shape[1]
            crop_area = real_crop.shape[0] * real_crop.shape[1]
            if crop_area < (0.01 * full_area):
                if verbose: print(f"  ⚠️ Crop too small ({crop_area} px, <1% of full). Returning penalty.")
                return {
                    'img_stem': img_stem, 'k_range': (k_star, k_star),
                    'best_ssim_k': k_star, 'best_ssim_score': 0.0,
                    'best_rmse_k': k_star, 'best_rmse_score': 1.0,
                    'mode': mode, 'crop_bbox': fixed_bbox
                }
            
            # Minimal sub-pixel alignment on cropped region
            try:
                shift, _, _ = phase_cross_correlation(real_crop, synth_crop, upsample_factor=50)
                dy, dx = float(shift[0]), float(shift[1])
                
                # Check for catastrophic shift in minimal adjust
                # Eğer kilitli bbox içinde bile devasa kayma çıkıyorsa bir sorun vardır
                if abs(dy) > (y1-y0)*0.5 or abs(dx) > (x1-x0)*0.5:
                     if verbose: print(f"  ⚠️ Minimal align shift too large (dy={dy:.1f}, dx={dx:.1f}), ignoring.")
                     dy, dx = 0.0, 0.0

                if verbose:
                    print(f"  Minimal align on locked bbox: dy={dy:.2f}, dx={dx:.2f}")
                
                # Warp synth to align
                tform = AffineTransform(translation=(dx, dy))
                synth_aligned = warp(
                    synth_crop,
                    inverse_map=tform.inverse,
                    output_shape=real_crop.shape,
                    preserve_range=True,
                    mode="constant",
                    cval=0.0, # NaN yerine 0 (metric için)
                    order=1
                ).astype("float32")
                
            except Exception as e:
                if verbose:
                    print(f"  ⚠️ Minimal align failed: {e}, using unaligned")
                synth_aligned = synth_crop
            
            # Compute metrics
            s_val, r_val, h_val, ms_val, gm_val, ncc_val, lpips_val = compute_metrics(real_crop, synth_aligned)
            
            return {
                'img_stem': img_stem,
                'k_range': (k_star, k_star),
                'all_results': [{'k': k_star, 'ssim': s_val, 'rmse': r_val, 'hist_corr': h_val, 'ms_ssim': ms_val, 'gmsd': gm_val}],
                'best_ssim_k': k_star,
                'best_ssim_score': s_val,
                'best_rmse_k': k_star,
                'best_rmse_score': r_val,
                'mode': mode,
                'cropped_shape': real_crop.shape,
                'alignment_method': 'fixed_bbox_minimal_align',
                'crop_bbox': fixed_bbox,
                'buffer_pixels': None,
                'buffer_source': 'fixed',
                'aligned_real': real_crop,
                'aligned_synth': synth_aligned,
            }
        else:
            # Fixed bbox var ama K bilinmiyor → SWEEP yap (maalesef)
            # Ama fixed_bbox kullanarak sweep yapacağız, evaluate_k_sweep'i değil
            # Manuel sweep döngüsü
            if verbose: print("  🔒 Fixed bbox but no learned k. Running local sweep...")
            k_min, k_max = get_k_range(img_stem)
            k_values = list(range(k_min, k_max + 1, K_STEP))
            
            best_s, best_k = -1.0, (k_min + k_max)//2
            best_r, best_kr = 1.0, (k_min + k_max)//2
            best_c, best_ck = float('inf'), (k_min + k_max)//2
            
            # Synth crop fix (no alignment yet)
            synth_crop_base = synthetic_norm[y0:y1, x0:x1]
            
            for k in k_values:
                rn = normalize_image(real_raw_f32, low_p=float(k), high_p=100.0)
                if rn is None: continue
                rc = rn[y0:y1, x0:x1]
                
                s, r, h, ms, gm, ncc, lpips_v = compute_metrics(rc, synth_crop_base)
                c = (1.0 - s) + r / 3.0 if (np.isfinite(s) and np.isfinite(r)) else float('inf')
                if s > best_s: best_s, best_k = s, k
                if r < best_r: best_r, best_kr = r, k
                if c < best_c: best_c, best_ck = c, k
            
            # Record learned (objective-aware)
            fake_res = {
                'best_ssim_k': best_k, 'best_ssim_score': best_s,
                'best_rmse_k': best_kr, 'best_rmse_score': best_r,
                'best_combined_k': best_ck, 'best_combined_score': best_c,
            }
            _record_learned_k(img_stem, fake_res, learned_k_db, objective_type=objective_type)
            
            return {
                'img_stem': img_stem,
                'k_range': (k_min, k_max),
                'best_ssim_k': best_k,
                'best_ssim_score': best_s,
                'best_rmse_k': best_kr,
                'best_rmse_score': best_r,
                'best_combined_k': best_ck,
                'best_combined_score': best_c,
                'mode': mode,
                'crop_bbox': fixed_bbox,
                'alignment_method': 'fixed_bbox_sweep'
            }

    # =========================================================
    
    # K-Sweep Mode (fixed_k provided manually check)
    if fixed_k is not None:
         k_star = int(fixed_k)
         if verbose: print(f"  🔒 Using fixed k={k_star} (CLI argument, uncropped/sweep mode)")
         return _evaluate_fixed_k(real_raw, synthetic_norm, img_stem, mode, buffer_pixels, buffer_percent, k_star)

    if (
        learned_k_db is not None
        and img_stem in learned_k_db
        and k_mode != "sweep"
    ):
        k_star = int(learned_k_db[img_stem])
        try:
            return _evaluate_fixed_k(
                real_raw,
                synthetic_norm,
                img_stem,
                mode,
                buffer_pixels,
                buffer_percent,
                k_star
            )
        except Exception as exc:
            if verbose:
                print(f"[learned-k] fixed evaluation failed ({exc}), falling back to sweep.")
            learned_k_db.pop(img_stem, None)


    res = evaluate_k_sweep(
        real_raw,
        synthetic_norm,
        img_stem,
        mode,
        buffer_pixels,
        buffer_ratio,
        buffer_percent,
        verbose
    )
    
    _record_learned_k(img_stem, res, learned_k_db, objective_type=objective_type)
    return res

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def evaluate_image_pair(real_path: Path,
                        synthetic_path: Path,
                        img_stem: str,
                        mode: str = "uncropped",
                        buffer_pixels: Optional[int] = None,
                        buffer_ratio: Optional[float] = None,
                        buffer_percent: float = 5.0,
                        verbose: bool = False) -> Dict:
    """
    Evaluate a real-synthetic image pair with k-sweep.
    
    High-level convenience function that:
    1. Loads real raw image (expects uint16 TIFF)
    2. Loads synthetic normalized image (expects float32 TIFF in [0,1])
    3. Performs k-sweep evaluation
    
    Args:
        real_path: Path to real filtered image (uint16 TIFF)
        synthetic_path: Path to synthetic normalized image (float32 TIFF)
        img_stem: IMG filename stem
        mode: "uncropped" or "cropped"
        buffer_percent: Buffer for masked region detection
        verbose: Verbose output
        
    Returns:
        Results dictionary from evaluate_k_sweep()
        
    Example:
        >>> results = evaluate_image_pair(
        ...     real_path=Path("H9463_0050_SR2_filtered_avg.tif"),
        ...     synthetic_path=Path("template_001_shaped_matched_normalised.tiff"),
        ...     img_stem="H9463_0050_SR2",
        ...     mode="cropped"
        ... )
    """
    print(f"\nEvaluating pair:")
    print(f"  Real: {real_path.name}")
    print(f"  Synthetic: {synthetic_path.name}")
    
    # Load real image
    try:
        real_img = np.array(Image.open(real_path), dtype=np.uint16)
        print(f"  Real image shape: {real_img.shape}")
    except Exception as e:
        print(f"[ERROR] Failed to load real image: {e}")
        return {'error': f'Failed to load real image: {e}'}
    
    # Load synthetic image
    try:
        synth_img = np.array(Image.open(synthetic_path), dtype=np.float32)
        synth_img = np.clip(synth_img, 0.0, 1.0)  # Ensure [0,1] range
        print(f"  Synthetic image shape: {synth_img.shape}")
    except Exception as e:
        print(f"[ERROR] Failed to load synthetic image: {e}")
        return {'error': f'Failed to load synthetic image: {e}'}
    
    # Check shape compatibility
    if real_img.shape != synth_img.shape:
        print(f"[ERROR] Shape mismatch: real {real_img.shape} vs synthetic {synth_img.shape}")
        return {'error': f'Shape mismatch: {real_img.shape} vs {synth_img.shape}'}
    
    # Perform k-sweep
    results = evaluate_k(
        real_raw=filtered_img,
        synthetic_norm=synth_img,
        img_stem=img_stem,
        mode=mode,
        buffer_percent=buffer_percent,
        verbose=verbose
    )
    return results


def add_custom_k_range(img_stem: str, k_min: int, k_max: int):
    """
    Add or update custom k-range for an IMG file.
    
    Args:
        img_stem: IMG filename stem
        k_min: Minimum k value
        k_max: Maximum k value
    """
    global PER_IMG_K_RANGES
    k_min = int(max(0, min(100, k_min)))
    k_max = int(max(0, min(100, k_max)))
    if k_min > k_max:
        k_min, k_max = k_max, k_min
    PER_IMG_K_RANGES[img_stem] = (k_min, k_max)
    print(f"Added/Updated k-range for {img_stem}: [{k_min}, {k_max}]")


if __name__ == "__main__":
    # Example usage
    print("Metrics Evaluator Module")
    print("=" * 80)
    print("This module provides k-sweep evaluation with cropping support.")
    print("\nComparison Modes:")
    print("  - uncropped: Full image comparison (default)")
    print("  - cropped: Detect masked regions, crop both images, then compare")
    print("\nExample usage:")
    print("  from metrics_evaluator import evaluate_image_pair")
    print("  results = evaluate_image_pair(")
    print("      real_path=Path('real_filtered.tif'),")
    print("      synthetic_path=Path('synthetic_norm.tiff'),")
    print("      img_stem='H9463_0050_SR2',")
    print("      mode='cropped'")
    print("  )")
    print(f"  print(f\"Best SSIM: {{results['best_ssim_score']:.4f}} at k={{results['best_ssim_k']}}\")")