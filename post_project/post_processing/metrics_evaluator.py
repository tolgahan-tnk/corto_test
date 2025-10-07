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


# ============================================================================
# K-SWEEP EVALUATION
# ============================================================================

def evaluate_k_sweep(real_raw: np.ndarray,
                    synthetic_norm: np.ndarray,
                    img_stem: str,
                    mode: str = "uncropped",
                    buffer_percent: float = 5.0,
                    verbose: bool = False) -> Dict:
    """
    Perform k-value sweep to find optimal normalization for real image.
    
    This function:
    1. Determines k-range for the IMG (custom or default)
    2. For each k value, normalizes real image with [k, 100] percentiles
    3. Optionally crops to valid regions (mode="cropped")
    4. Computes SSIM, RMSE, and histogram correlation
    5. Identifies best k for SSIM and RMSE
    
    Args:
        real_raw: Raw real image (uint16 or float32)
        synthetic_norm: Normalized synthetic image [0, 1]
        img_stem: IMG filename stem (for k-range lookup)
        mode: Comparison mode - "uncropped" or "cropped"
        buffer_percent: Buffer percentage for masked region detection (cropped mode)
        verbose: If True, prints progress updates
        
    Returns:
        Dictionary with:
        - 'k_range': (k_min, k_max) used
        - 'all_results': List of dicts with k, ssim, rmse, hist_corr for each k
        - 'best_ssim_k': k value with highest SSIM
        - 'best_ssim_score': Best SSIM score
        - 'best_rmse_k': k value with lowest RMSE
        - 'best_rmse_score': Best RMSE score
        - 'mode': Comparison mode used
        - 'cropped_shape': Shape after cropping (if mode="cropped")
        
    Example:
        >>> real_raw = np.array(Image.open("real_filtered.tif"), dtype=np.uint16)
        >>> synth_norm = np.array(Image.open("synthetic_norm.tif"), dtype=np.float32)
        >>> results = evaluate_k_sweep(
        ...     real_raw, synth_norm,
        ...     img_stem="H9463_0050_SR2",
        ...     mode="cropped"
        ... )
        >>> print(f"Best SSIM: {results['best_ssim_score']:.4f} at k={results['best_ssim_k']}")
    """
    if mode not in {"uncropped", "cropped"}:
        raise ValueError(f"Invalid mode: {mode}. Must be 'uncropped' or 'cropped'")
    
    # Get k-range for this IMG
    k_min, k_max = get_k_range(img_stem)
    k_values = list(range(k_min, k_max + 1, K_STEP))
    
    print(f"\n{'='*80}")
    print(f"K-Sweep Evaluation: {img_stem}")
    print(f"Mode: {mode}")
    print(f"K-range: {k_min} to {k_max} (step {K_STEP})")
    print(f"{'='*80}")
    
    # Convert raw to float32 if needed
    real_raw_f32 = real_raw.astype(np.float32)
    
    # For cropped mode, detect valid region from synthetic once
    if mode == "cropped":
        valid_mask = detect_masked_regions(synthetic_norm, buffer_percent=buffer_percent)
        valid_pixels = valid_mask.sum()
        total_pixels = valid_mask.size
        print(f"Valid region: {valid_pixels}/{total_pixels} pixels "
              f"({100.0 * valid_pixels / total_pixels:.1f}%)")
    
    # Storage for results
    all_results = []
    best_ssim = -np.inf
    best_ssim_k = None
    best_rmse = np.inf
    best_rmse_k = None
    
    # Progress tracking
    total_k = len(k_values)
    
    for idx, k in enumerate(k_values, start=1):
        # Normalize real image with this k value
        real_norm = normalize_image(real_raw_f32, low_p=float(k), high_p=100.0)
        
        if real_norm is None:
            if verbose:
                print(f"[k={k:3d}] Normalization failed")
            continue
        
        # Apply cropping if requested
        if mode == "cropped":
            real_comp, synth_comp = crop_to_valid_region(real_norm, synthetic_norm, valid_mask)
            cropped_shape = real_comp.shape
        else:
            real_comp = real_norm
            synth_comp = synthetic_norm
            cropped_shape = None
        
        # Compute metrics
        try:
            s_val, r_val, h_val = compute_metrics(real_comp, synth_comp)
        except Exception as e:
            if verbose:
                print(f"[k={k:3d}] Metrics failed: {e}")
            continue
        
        # Store result
        all_results.append({
            'k': k,
            'ssim': s_val,
            'rmse': r_val,
            'hist_corr': h_val
        })
        
        # Track best scores
        if np.isfinite(s_val) and s_val > best_ssim:
            best_ssim = s_val
            best_ssim_k = k
        
        if np.isfinite(r_val) and r_val < best_rmse:
            best_rmse = r_val
            best_rmse_k = k
        
        # Progress update
        if verbose or (idx % 10 == 0) or (idx == total_k):
            pct = 100.0 * idx / total_k
            print(f"[{pct:5.1f}%] k={k:3d} | SSIM={s_val:.4f} | RMSE={r_val:.4f} | "
                  f"Hist_Corr={h_val:.4f}", end="\r")
    
    print()  # Newline after progress
    
    # Summary
    print(f"\n{'='*80}")
    print(f"Results Summary:")
    print(f"  Best SSIM: {best_ssim:.6f} at k={best_ssim_k}")
    print(f"  Best RMSE: {best_rmse:.6f} at k={best_rmse_k}")
    print(f"{'='*80}\n")
    
    return {
        'img_stem': img_stem,
        'k_range': (k_min, k_max),
        'all_results': all_results,
        'best_ssim_k': best_ssim_k,
        'best_ssim_score': best_ssim,
        'best_rmse_k': best_rmse_k,
        'best_rmse_score': best_rmse,
        'mode': mode,
        'cropped_shape': cropped_shape
    }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def evaluate_image_pair(real_path: Path,
                       synthetic_path: Path,
                       img_stem: str,
                       mode: str = "uncropped",
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
    results = evaluate_k_sweep(
        real_raw=real_img,
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