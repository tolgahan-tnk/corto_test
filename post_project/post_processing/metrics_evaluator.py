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
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import binary_dilation
from skimage.registration import phase_cross_correlation
from skimage.transform import warp, AffineTransform


# Suppress warnings
warnings.filterwarnings("ignore")


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
}

# Default k-range for images not in PER_IMG_K_RANGES
K_DEFAULT_RANGE = (0, 100)

# k-step size for sweep
K_STEP = 1


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
    Normalize image to [0, 1] using percentile clipping.
    
    Args:
        arr: Input array (any dtype)
        low_p: Lower percentile
        high_p: Upper percentile (default: 100.0)
        
    Returns:
        Normalized float32 array in [0, 1], or None if normalization fails
    """
    try:
        low, high = np.percentile(arr, [low_p, high_p])
        
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            return None
        
        norm = (arr - low) / (high - low)
        return np.clip(norm, 0.0, 1.0).astype(np.float32)
        
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

def compute_metrics(real_01: np.ndarray,
                   synthetic_01: np.ndarray,
                   data_range: float = 1.0) -> Tuple[float, float, float]:
    """
    Compute image similarity metrics.
    
    Calculates:
    - SSIM: Structural Similarity Index
    - RMSE: Root Mean Square Error (not normalized)
    - Histogram Correlation: Pearson correlation of histograms
    
    Args:
        real_01: Real image in [0, 1]
        synthetic_01: Synthetic image in [0, 1]
        data_range: Data range for SSIM (default: 1.0)
        
    Returns:
        Tuple of (ssim_score, rmse, hist_corr)
        
    Example:
        >>> s, r, h = compute_metrics(real_norm, synth_norm)
        >>> print(f"SSIM: {s:.4f}, RMSE: {r:.4f}, Hist_Corr: {h:.4f}")
        SSIM: 0.8523, RMSE: 0.0342, Hist_Corr: 0.9201
    """
    # Safety: shapes must match
    if real_01.shape != synthetic_01.shape:
        raise ValueError(f"Shape mismatch: {real_01.shape} vs {synthetic_01.shape}")
    
    # Replace NaNs if present
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
        
        # Guard against zero-variance histograms
        if np.allclose(h1, h1[0]) or np.allclose(h2, h2[0]):
            hist_corr = np.nan
        else:
            hist_corr = float(np.corrcoef(h1, h2)[0, 1])
    except Exception as e:
        print(f"[WARNING] Histogram correlation failed: {e}")
        hist_corr = np.nan
    
    return ssim_score, rmse, hist_corr

def align_and_crop_with_buffer(
    imgA: np.ndarray,
    imgB: np.ndarray,
    buffer_pixels: Optional[int] = None,
    buffer_percent: Optional[float] = 5.0,
    mask_threshold: float = 0.02,        # YENİ: içerik eşiği
    upsample_factor: int = 100,
    interpolation_order: int = 1,
    buffer_ratio: Optional[float] = None,
) -> dict:
    """
    Align two images using phase cross-correlation and crop to the *content* overlap.
    Cropping now uses: valid = isfinite(B_aligned) & (B_aligned > mask_threshold)
    Buffer can be given in pixels or as percent of max(H, W).
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

    # --- 2) Warp B to A ---
    tform = AffineTransform(translation=(dx, dy))
    B_aligned = warp(
        B, inverse_map=tform.inverse, output_shape=A.shape,
        preserve_range=True, mode="constant", cval=np.nan, order=interpolation_order
    ).astype("float32")

    # --- 3) İçerik maskesi (YENİ) ---
    finite_mask = np.isfinite(B_aligned)
    content_mask = finite_mask & (B_aligned > mask_threshold)

    if not np.any(content_mask):
        # Fallback: sadece finite overlap
        print("[WARNING] Content mask empty; falling back to finite overlap.")
        content_mask = finite_mask

    ys, xs = np.where(content_mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    print(f"  Content overlap (before buffer): y=[{y0}:{y1}], x=[{x0}:{x1}]")

    # --- 4) Buffer hesaplama ---
    if buffer_pixels is None:
        if buffer_ratio is not None:
            buffer_px = max(0, int(round(buffer_ratio * max(H, W))))
        elif buffer_percent is not None:
            buffer_px = max(0, int(round((buffer_percent / 100.0) * max(H, W))))
        else:
            buffer_px = 0
    else:
        buffer_px = int(max(0, buffer_pixels))

    # Dışa doğru genişlet, sınırları kırp
    y0b = max(0, y0 - buffer_px)
    x0b = max(0, x0 - buffer_px)
    y1b = min(H, y1 + buffer_px)
    x1b = min(W, x1 + buffer_px)

    if x1b <= x0b or y1b <= y0b:
        y0b, y1b, x0b, x1b = y0, y1, x0, x1

    print(f"  Buffer: {buffer_percent if buffer_pixels is None else 'px'} -> {buffer_px} px")
    print(f"  Buffered region: y=[{y0b}:{y1b}], x=[{x0b}:{x1b}]")
    print(f"  Final crop size: {y1b - y0b}×{x1b - x0b} pixels")

    # --- 5) Crop ---
    A_crop = A[y0b:y1b, x0b:x1b]
    B_crop = B_aligned[y0b:y1b, x0b:x1b]

    return {
        "A_crop": A_crop,
        "B_crop": B_crop,
        "shift": (dy, dx),
        "bbox": (y0b, y1b, x0b, x1b),
        "buffer_px": buffer_px,           # YENİ: raporlama için
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
                    s_tmp, _, _ = compute_metrics(ar["A_crop"], ar["B_crop"])
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

        s_val, r_val, h_val = compute_metrics(real_comp, synth_comp)
        all_results.append({'k': k, 'ssim': s_val, 'rmse': r_val, 'hist_corr': h_val})

        if np.isfinite(s_val) and s_val > best_ssim:
            best_ssim, best_ssim_k = s_val, k
        if np.isfinite(r_val) and r_val < best_rmse:
            best_rmse, best_rmse_k = r_val, k

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
        'mode': mode,
        'cropped_shape': cropped_shape,
        'alignment_method': 'phase_correlation' if mode == 'cropped' else 'none',
        'buffer_pixels': None,
        'buffer_source': None
    }

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
               fallback_delta: int = 5) -> Dict:
    """
    'Öğrenilmiş k' desteği:
    - k_mode='learned' ve learned_k_db[img_stem] varsa önce tek k denenir.
    - SSIM < guard_ssim olursa k_star±delta dar bant sweep'e düşer.
    - Yoksa normal sweep.
    - Başardığında learned_k_db[img_stem] güncellenir.
    """
    def _update_db(k_best: int):
        if learned_k_db is not None:
            learned_k_db[img_stem] = int(k_best)

    # 1) Öğrenilmiş k varsa hızlı kontrol
    if k_mode == "learned" and learned_k_db is not None and img_stem in learned_k_db:
        k_star = int(learned_k_db[img_stem])
        real_raw_f32 = real_raw.astype(np.float32)
        try:
            rn = normalize_image(real_raw_f32, float(k_star))
            if mode == "cropped":
                ar = align_and_crop_with_buffer(
                    rn, synthetic_norm,
                    buffer_pixels=None,
                    buffer_percent=buffer_percent,
                    mask_threshold=0.02,
                    upsample_factor=50,
                    interpolation_order=1
                )
                s_val, r_val, h_val = compute_metrics(ar["A_crop"], ar["B_crop"])
                if np.isfinite(s_val) and s_val >= guard_ssim:
                    _update_db(k_star)
                    return {
                        'img_stem': img_stem, 'k_range': (k_star, k_star),
                        'all_results': [{'k': k_star, 'ssim': s_val, 'rmse': r_val, 'hist_corr': h_val}],
                        'best_ssim_k': k_star, 'best_ssim_score': s_val,
                        'best_rmse_k': k_star, 'best_rmse_score': r_val,
                        'mode': mode, 'cropped_shape': ar["A_crop"].shape,
                        'alignment_method': 'phase_correlation'
                    }
            else:
                s_val, r_val, h_val = compute_metrics(rn, synthetic_norm)
                if np.isfinite(s_val) and s_val >= guard_ssim:
                    _update_db(k_star)
                    return {
                        'img_stem': img_stem, 'k_range': (k_star, k_star),
                        'all_results': [{'k': k_star, 'ssim': s_val, 'rmse': r_val, 'hist_corr': h_val}],
                        'best_ssim_k': k_star, 'best_ssim_score': s_val,
                        'best_rmse_k': k_star, 'best_rmse_score': r_val,
                        'mode': mode, 'cropped_shape': None,
                        'alignment_method': 'none'
                    }
        except Exception as e:
            if verbose:
                print(f"[learned-k] quick check failed, falling back to short sweep: {e}")

        # 2) Guard tutmadı → ±delta kısa sweep
        k_min = max(0, k_star - fallback_delta)
        k_max = min(100, k_star + fallback_delta)
        global K_DEFAULT_RANGE
        old_range = K_DEFAULT_RANGE
        K_DEFAULT_RANGE = (k_min, k_max)
        res = evaluate_k_sweep(
            real_raw, synthetic_norm, img_stem, mode,
            buffer_pixels, buffer_ratio, buffer_percent, verbose
        )
        K_DEFAULT_RANGE = old_range
        _update_db(int(res['best_ssim_k']))
        return res

    # 3) İlk kullanım ya da k_mode='sweep' → normal sweep
    res = evaluate_k_sweep(
        real_raw, synthetic_norm, img_stem, mode,
        buffer_pixels, buffer_ratio, buffer_percent, verbose
    )
    _update_db(int(res['best_ssim_k']))
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
        mode=eval_mode,
        buffer_percent=BUFFER_PERCENT,
        verbose=VERBOSE,
        k_mode="learned",            # "sweep" dersen eski davranış
        learned_k_db=LEARNED_K_DB,   # burada güncellenecek
        guard_ssim=0.70,             # gerekirse değiştir
        fallback_delta=5
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