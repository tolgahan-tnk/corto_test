#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pds_processor.py
================
PDS IMG file processing utilities for HRSC/SRC image data.

This module provides functions to:
- Read PDS IMG metadata (MINIMUM, MAXIMUM, MEAN, STANDARD_DEVIATION)
- Load 16-bit big-endian image data
- Apply hot pixel filtering using dynamic thresholding
- Normalize images using percentile ranges

Author: Post-Processing Pipeline
Date: 2025
"""

import re
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple, Optional, Dict


# ============================================================================
# PDS METADATA EXTRACTION
# ============================================================================

def extract_pds_metadata(pds_path: Path) -> Dict[str, float]:
    """
    Extract metadata from PDS IMG file header.
    
    Searches for MINIMUM, MAXIMUM, MEAN, and STANDARD_DEVIATION values
    in the PDS label (ASCII header).
    
    Args:
        pds_path: Path to PDS IMG file
        
    Returns:
        Dictionary with keys: 'min', 'max', 'mean', 'std'
        Values are None if not found in header
        
    Example:
        >>> metadata = extract_pds_metadata(Path("H9463_0050_SR2.IMG"))
        >>> print(metadata['mean'])
        125.3
    """
    metadata = {
        'min': None,
        'max': None,
        'mean': None,
        'std': None
    }
    
    try:
        with open(pds_path, 'r', errors='ignore') as f:
            for line in f:
                # Clean line endings
                ln = line.replace("<CR><LF>", "").rstrip("\n")
                
                # Extract metadata values
                if re.match(r"^\s*MINIMUM\s*=", ln):
                    metadata['min'] = float(re.search(r"=\s*([\d\.\+\-eE]+)", ln).group(1))
                elif re.match(r"^\s*MAXIMUM\s*=", ln):
                    metadata['max'] = float(re.search(r"=\s*([\d\.\+\-eE]+)", ln).group(1))
                elif re.match(r"^\s*MEAN\s*=", ln):
                    metadata['mean'] = float(re.search(r"=\s*([\d\.\+\-eE]+)", ln).group(1))
                elif re.match(r"^\s*STANDARD_DEVIATION\s*=", ln):
                    metadata['std'] = float(re.search(r"=\s*([\d\.\+\-eE]+)", ln).group(1))
                    
    except Exception as e:
        print(f"[WARNING] Metadata extraction failed for {pds_path.name}: {e}")
    
    return metadata


def load_pds_image_data(pds_path: Path) -> Tuple[Optional[np.ndarray], Dict[str, int]]:
    """
    Load image data from PDS IMG file using GDAL.
    
    Uses GDAL to read the PDS3 file, which correctly handles headers,
    byte ordering, NoData values, and signed int16 data type.
    
    Args:
        pds_path: Path to PDS IMG file
        
    Returns:
        Tuple of (image_array, image_info)
        - image_array: numpy array (H x W), dtype from GDAL (typically int16)
        - image_info: dict with 'lines', 'samples'
    """
    try:
        from osgeo import gdal
        import re as _re
        
        dataset = gdal.Open(str(pds_path), gdal.GA_ReadOnly)
        if dataset is None:
            raise RuntimeError(f"GDAL could not open {pds_path.name}")
        
        band = dataset.GetRasterBand(1)
        data = band.ReadAsArray()  # Returns proper signed int16
        
        lines, samples = data.shape
        image_info = {
            'lines': lines,
            'samples': samples,
        }
        
        # Log NoData info
        nodata = band.GetNoDataValue()
        if nodata is not None:
            print(f"  📋 GDAL NoData value: {nodata}")
        print(f"  📋 GDAL dtype: {data.dtype}, shape: {data.shape}, "
              f"min={data.min()}, max={data.max()}")
        
        dataset = None  # Close
        
        # Apply display direction corrections from PDS label
        content = open(pds_path, 'r', errors='ignore').read(32000)
        sdd_m = _re.search(r"SAMPLE_DISPLAY_DIRECTION\s*=\s*(\S+)", content)
        ldd_m = _re.search(r"LINE_DISPLAY_DIRECTION\s*=\s*(\S+)", content)
        sample_dir = sdd_m.group(1).strip('"').upper() if sdd_m else 'RIGHT'
        line_dir   = ldd_m.group(1).strip('"').upper() if ldd_m else 'DOWN'
        if sample_dir == 'LEFT':
            data = np.fliplr(data)
            print(f"  [ORIENT] Flipped X (GDAL: SAMPLE_DISPLAY_DIRECTION=LEFT)")
        if line_dir == 'UP':
            data = np.flipud(data)
            print(f"  [ORIENT] Flipped Y (GDAL: LINE_DISPLAY_DIRECTION=UP)")
        
        return data, image_info
        
    except ImportError:
        print("[WARNING] GDAL not available, falling back to custom parser")
        return _load_pds_custom(pds_path)
    except Exception as e:
        print(f"[ERROR] Failed to load image data from {pds_path.name}: {e}")
        return None, {}


def _pds_dtype(sample_type: str, sample_bits: int) -> str:
    """Map PDS SAMPLE_TYPE + SAMPLE_BITS to numpy dtype string."""
    st = sample_type.strip('"').upper()
    mapping = {
        ('MSB_INTEGER', 16):            '>i2',
        ('LSB_INTEGER', 16):            '<i2',
        ('MSB_UNSIGNED_INTEGER', 16):   '>u2',
        ('LSB_UNSIGNED_INTEGER', 16):   '<u2',
        ('PC_REAL', 32):                '<f4',   # Rosetta OSIRIS
        ('IEEE_REAL', 32):              '>f4',
        ('PC_REAL', 64):                '<f8',
        ('IEEE_REAL', 64):              '>f8',
        ('MSB_INTEGER', 8):             '>i1',
        ('UNSIGNED_INTEGER', 8):        'u1',
    }
    dtype = mapping.get((st, sample_bits), '>i2')  # HRSC default
    print(f"  [PDS] dtype: SAMPLE_TYPE={st}, BITS={sample_bits} -> numpy {dtype}")
    return dtype


def _load_pds_custom(pds_path: Path) -> Tuple[Optional[np.ndarray], Dict[str, int]]:
    """Fallback custom PDS parser when GDAL is not available."""
    try:
        import re as _re
        content = open(pds_path, 'r', errors='ignore').read()
        
        record_bytes = int(_re.search(r"RECORD_BYTES\s*=\s*(\d+)", content).group(1))
        image_record = int(_re.search(r"\^IMAGE\s*=\s*(\d+)", content).group(1))
        lines = int(_re.search(r"LINES\s*=\s*(\d+)", content).group(1))
        samples = int(_re.search(r"LINE_SAMPLES\s*=\s*(\d+)", content).group(1))
        
        # Detect SAMPLE_TYPE for correct dtype
        st_match = _re.search(r"SAMPLE_TYPE\s*=\s*(\S+)", content)
        sb_match = _re.search(r"SAMPLE_BITS\s*=\s*(\d+)", content)
        sample_type = st_match.group(1) if st_match else 'MSB_INTEGER'
        sample_bits = int(sb_match.group(1)) if sb_match else 16
        dtype = _pds_dtype(sample_type, sample_bits)
        
        # Detect display direction for orientation correction
        sdd_match = _re.search(r"SAMPLE_DISPLAY_DIRECTION\s*=\s*(\S+)", content)
        ldd_match = _re.search(r"LINE_DISPLAY_DIRECTION\s*=\s*(\S+)", content)
        sample_dir = sdd_match.group(1).strip('"').upper() if sdd_match else 'RIGHT'
        line_dir = ldd_match.group(1).strip('"').upper() if ldd_match else 'DOWN'
        
        image_info = {'lines': lines, 'samples': samples}
        
        with open(pds_path, 'rb') as f:
            f.seek((image_record - 1) * record_bytes)
            raw = np.fromfile(f, dtype=dtype, count=lines * samples)
        
        raw = raw.reshape((lines, samples))
        
        # Apply display direction corrections
        # SAMPLE_DISPLAY_DIRECTION=LEFT means columns stored right-to-left -> flip X
        if sample_dir == 'LEFT':
            raw = np.fliplr(raw)
            print(f"  [ORIENT] Flipped X (SAMPLE_DISPLAY_DIRECTION=LEFT)")
        # LINE_DISPLAY_DIRECTION=UP means rows stored bottom-to-top -> flip Y
        if line_dir == 'UP':
            raw = np.flipud(raw)
            print(f"  [ORIENT] Flipped Y (LINE_DISPLAY_DIRECTION=UP)")
        
        return raw, image_info
        
    except Exception as e:
        print(f"[ERROR] Fallback parser failed for {pds_path.name}: {e}")
        return None, {}


# ============================================================================
# HOT PIXEL FILTERING
# ============================================================================

def compute_dynamic_threshold(img: np.ndarray,
                              n_bins: int = 256,
                              factor: float = 1.5,
                              gap_len: int = 2) -> int:
    """
    Compute dynamic hot pixel threshold using histogram analysis.
    
    This function:
    1. Creates a histogram of DN values
    2. Finds the longest contiguous run of non-zero bins from DN=0
    3. Multiplies the ending DN of this run by 'factor' to get threshold
    4. Single-bin gaps are filled to improve continuity detection
    
    Args:
        img: Input image array (int16 or uint16)
        n_bins: Number of histogram bins (default: 256)
        factor: Multiplier for threshold (default: 1.5)
        gap_len: Minimum gap length to consider end of continuity (default: 2)
        
    Returns:
        Dynamic threshold value (int)
    """
    DN_MAX = int(img.max())
    DN_MIN = max(0, int(img.min()))  # Start histogram from 0 at minimum
    
    if DN_MAX <= DN_MIN:
        return DN_MAX
    
    # Create histogram with bins spanning the actual data range
    bins = np.linspace(DN_MIN, DN_MAX + 1, n_bins + 1)
    hist, edges = np.histogram(img, bins=bins)
    
    nonzero = hist > 0
    
    # Fill single-bin holes (True-False-True => True)
    filled = nonzero.copy()
    if len(nonzero) >= 3:
        holes = (~nonzero)
        single_hole = np.zeros_like(holes, dtype=bool)
        single_hole[1:-1] = holes[1:-1] & nonzero[:-2] & nonzero[2:]
        filled[1:-1] = np.where(single_hole[1:-1], True, filled[1:-1])
    
    # Find first gap of length >= gap_len
    false_mask = (~filled).astype(int)
    
    if false_mask.sum() == 0:
        # No gaps found - entire range is populated
        cont_end_dn = edges[-1]
    else:
        # Convolve to find consecutive False regions
        window = np.ones(gap_len, dtype=int)
        conv = np.convolve(false_mask, window, mode="valid")
        gap_starts = np.where(conv == gap_len)[0]
        
        if gap_starts.size > 0:
            # End of continuous region is at first long gap
            g0 = int(gap_starts[0])
            cont_end_dn = edges[g0]
        else:
            # No long gap found - use first False bin
            first_zero = np.where(~filled)[0]
            cont_end_dn = edges[first_zero[0]] if first_zero.size > 0 else edges[-1]
    
    # Apply factor and clip to valid range
    thr = int(min(DN_MAX, round(cont_end_dn * factor)))
    
    return thr


def eight_neighbor_mean(raw: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """
    Compute 8-neighbor mean for hot pixel replacement.
    
    For each pixel, computes the mean of its 8 neighbors. Preferentially
    uses only valid (non-hot) neighbors if available, otherwise uses all.
    
    Args:
        raw: Input image (int16 or uint16)
        valid_mask: Boolean mask (True = valid pixel, False = hot pixel)
        
    Returns:
        Array of mean values for each pixel position
        
    Note:
        - Uses zero-padding at image boundaries
        - Returns float64 array (should be rounded/clipped before use)
    """
    img = raw.astype(np.float64)
    vm = valid_mask.astype(np.float64)
    
    # Pad arrays to handle borders
    Pimg = np.pad(img, ((1, 1), (1, 1)), mode="constant", constant_values=0.0)
    Pvm = np.pad(vm, ((1, 1), (1, 1)), mode="constant", constant_values=0.0)
    
    # Extract 8 neighbors
    nbrs_img = [
        Pimg[:-2, :-2], Pimg[:-2, 1:-1], Pimg[:-2, 2:],
        Pimg[1:-1, :-2],                 Pimg[1:-1, 2:],
        Pimg[2:, :-2],  Pimg[2:, 1:-1],  Pimg[2:, 2:]
    ]
    nbrs_vm = [
        Pvm[:-2, :-2], Pvm[:-2, 1:-1], Pvm[:-2, 2:],
        Pvm[1:-1, :-2],                Pvm[1:-1, 2:],
        Pvm[2:, :-2],  Pvm[2:, 1:-1],  Pvm[2:, 2:]
    ]
    
    # Compute mean using only valid neighbors
    sum_valid = np.zeros_like(img)
    cnt_valid = np.zeros_like(img)
    
    for I, M in zip(nbrs_img, nbrs_vm):
        sum_valid += I * M
        cnt_valid += M
    
    has_valid = cnt_valid > 0
    mean_valid = np.zeros_like(img)
    mean_valid[has_valid] = sum_valid[has_valid] / cnt_valid[has_valid]
    
    # Fallback: use all neighbors if no valid ones exist
    sum_all = np.zeros_like(img)
    for I in nbrs_img:
        sum_all += I
    mean_all = sum_all / 8.0
    
    # Prefer valid-only mean, fall back to all-neighbor mean
    mean = mean_all.copy()
    mean[has_valid] = mean_valid[has_valid]
    
    return mean


def filter_hot_pixels(raw: np.ndarray,
                     n_bins: int = 256,
                     factor: float = 1.5,
                     gap_len: int = 2) -> Tuple[np.ndarray, int, int]:
    """
    Apply hot pixel filtering to raw image.
    
    For integer DN data (HRSC): histogram-based dynamic threshold.
    For float calibrated data (OSIRIS): 5-sigma outlier rejection.
    
    Args:
        raw: Input image (int16, uint16, or float32)
        n_bins: Histogram bins for threshold calculation (int data only)
        factor: Threshold multiplier (int data only)
        gap_len: Gap detection parameter (int data only)
        
    Returns:
        Tuple of (filtered_image, threshold_used, num_replaced_pixels)
    """
    # ── Float32 path (calibrated radiance, e.g. Rosetta OSIRIS) ──
    if np.issubdtype(raw.dtype, np.floating):
        valid = raw[np.isfinite(raw) & (raw > 0)]
        if valid.size == 0:
            return raw.copy(), 0.0, 0
        mu = float(np.mean(valid))
        sigma = float(np.std(valid))
        thr = mu + 5.0 * sigma  # 5-sigma outlier
        hot_mask = (raw > thr) | ~np.isfinite(raw)
        valid_mask = ~hot_mask
        nbr_mean = eight_neighbor_mean(raw, valid_mask)
        filtered = raw.copy()
        filtered[hot_mask] = nbr_mean[hot_mask].astype(raw.dtype)
        n_hot = int(hot_mask.sum())
        print(f"  [FLT] Float hot pixel filter: mu={mu:.6f}, sigma={sigma:.6f}, "
              f"thr={thr:.6f}, replaced={n_hot}")
        return filtered, thr, n_hot
    
    # ── Integer DN path (HRSC, original behavior) ──
    # Compute dynamic threshold
    thr = compute_dynamic_threshold(raw, n_bins=n_bins, factor=factor, gap_len=gap_len)
    
    # Identify hot pixels
    hot_mask = raw > thr
    valid_mask = raw <= thr
    
    # Compute replacement values using 8-neighbor mean
    nbr_mean = eight_neighbor_mean(raw, valid_mask)
    
    # Replace hot pixels
    filtered = raw.copy()
    dn_min = max(0, int(raw.min()))
    dn_max = int(raw.max())
    filtered_vals = np.rint(nbr_mean[hot_mask]).clip(dn_min, dn_max).astype(raw.dtype)
    filtered[hot_mask] = filtered_vals
    
    n_hot = int(hot_mask.sum())
    
    return filtered, thr, n_hot


# ============================================================================
# LOW-DN FILTERING & NaN FILLING
# ============================================================================

def filter_low_dn(img: np.ndarray, threshold: int = 1) -> np.ndarray:
    """
    Mark invalid pixels as NaN.
    
    For integer DN data: pixels with DN ≤ threshold → NaN.
    For float calibrated data: pixels ≤ 0 → NaN (negative radiance = invalid).
    
    Args:
        img: Input image (uint16, int16, or float32)
        threshold: DN values ≤ this are set to NaN (default: 1, int data only)
        
    Returns:
        float32 array with invalid pixels set to NaN
    """
    result = img.astype(np.float32)
    
    if np.issubdtype(img.dtype, np.floating):
        # Float calibrated data: negative/zero radiance → NaN
        low_mask = result <= 0
        label = "radiance ≤ 0"
    else:
        # Integer DN data (HRSC, original behavior)
        low_mask = img <= threshold
        label = f"DN ≤ {threshold}"
    
    result[low_mask] = np.nan
    n_low = int(low_mask.sum())
    if n_low > 0:
        print(f"  [FLT] filter_low_dn: {n_low} pixels with {label} -> NaN")
    return result


def fill_nan_cardinal(img: np.ndarray) -> np.ndarray:
    """
    Fill NaN pixels using 4-directional (cardinal) nearest-neighbor average.
    
    For each NaN pixel, searches left/right/up/down for the nearest
    non-NaN value in each direction, then replaces the NaN with the
    average of found neighbors.
    
    Args:
        img: float32 array with NaN values
        
    Returns:
        float32 array with NaN values filled
        
    Note:
        - If no valid neighbor is found in any direction, pixel stays NaN
        - Uses the original (static) data for lookups, not iteratively filled values
        - Typical PDS images have very few NaN pixels after DN≤1 filter (~1000)
          so the loop-based approach is fast enough
    """
    original = img  # Static reference for lookups (never modified)
    filled = img.copy()  # Output array (modified in-place)
    nan_mask = np.isnan(original)
    n_nan = int(nan_mask.sum())
    
    if n_nan == 0:
        return filled
    
    nan_indices = np.argwhere(nan_mask)
    filled_count = 0
    
    for r, c in nan_indices:
        neighbors = []
        
        # Search LEFT (same row, decreasing column)
        if c > 0:
            left_seg = original[r, :c][::-1]
            valid_idx = np.where(~np.isnan(left_seg))[0]
            if valid_idx.size > 0:
                neighbors.append(float(left_seg[valid_idx[0]]))
        
        # Search RIGHT (same row, increasing column)
        if c < original.shape[1] - 1:
            right_seg = original[r, c+1:]
            valid_idx = np.where(~np.isnan(right_seg))[0]
            if valid_idx.size > 0:
                neighbors.append(float(right_seg[valid_idx[0]]))
        
        # Search UP (same column, decreasing row)
        if r > 0:
            up_seg = original[:r, c][::-1]
            valid_idx = np.where(~np.isnan(up_seg))[0]
            if valid_idx.size > 0:
                neighbors.append(float(up_seg[valid_idx[0]]))
        
        # Search DOWN (same column, increasing row)
        if r < original.shape[0] - 1:
            down_seg = original[r+1:, c]
            valid_idx = np.where(~np.isnan(down_seg))[0]
            if valid_idx.size > 0:
                neighbors.append(float(down_seg[valid_idx[0]]))
        
        if neighbors:
            filled[r, c] = np.mean(neighbors)
            filled_count += 1
    
    remaining = int(np.isnan(filled).sum())
    print(f"  🔧 fill_nan_cardinal: filled {filled_count}/{n_nan} NaN pixels"
          f"{f', {remaining} remaining' if remaining > 0 else ''}")
    
    return filled


def preclip_percentile(img: np.ndarray,
                       low_p: float = 1.0,
                       high_p: float = 99.0) -> np.ndarray:
    """
    Clip outlier values using percentiles and normalize to [0, 1].
    
    Computes the low_p-th and high_p-th percentiles from valid (non-NaN)
    pixels, clips to that range, then linearly maps to [0, 1].
    NaN values are preserved.
    
    Args:
        img: float32 array (may contain NaN)
        low_p: Lower percentile for clipping (default: 2.0)
        high_p: Upper percentile for clipping (default: 98.0)
        
    Returns:
        float32 array in [0, 1] with NaN preserved
    """
    valid_mask = np.isfinite(img)
    valid_values = img[valid_mask]
    
    if valid_values.size == 0:
        return img.copy()
    
    p_low, p_high = np.percentile(valid_values, [low_p, high_p])
    
    if p_high <= p_low:
        print(f"  ⚠️ preclip_percentile: P{low_p}={p_low:.1f} >= P{high_p}={p_high:.1f}, skipping")
        return img.copy()
    
    result = img.copy()
    # Clip valid values to [p_low, p_high]
    result[valid_mask] = np.clip(valid_values, p_low, p_high)
    # Normalize to [0, 1]
    result[valid_mask] = (result[valid_mask] - p_low) / (p_high - p_low)
    
    print(f"  🔧 preclip_percentile: P{low_p}={p_low:.1f}, P{high_p}={p_high:.1f} → [0, 1]")
    
    return result


# ============================================================================
# NORMALIZATION
# ============================================================================

def normalize_percentile(img: np.ndarray,
                        low_p: float = 0.0,
                        high_p: float = 100.0) -> Optional[np.ndarray]:
    """
    Normalize image to [0, 1] using percentile clipping.
    
    Args:
        img: Input image (any dtype)
        low_p: Lower percentile (default: 0.0)
        high_p: Upper percentile (default: 100.0)
        
    Returns:
        Normalized float32 array in [0, 1], or None if normalization fails
        
    Example:
        >>> raw = np.array(Image.open("filtered.tif"), dtype=np.uint16)
        >>> norm = normalize_percentile(raw, low_p=2.0, high_p=98.0)
        >>> print(norm.min(), norm.max())  # Should be close to 0.0 and 1.0
        0.0 1.0
    """
    try:
        img_f32 = img.astype(np.float32)
        
        # Compute percentiles
        low, high = np.percentile(img_f32, [low_p, high_p])
        
        # Validate percentiles
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            print(f"[WARNING] Invalid percentiles: low={low}, high={high}")
            return None
        
        # Normalize and clip
        normalized = (img_f32 - low) / (high - low)
        normalized = np.clip(normalized, 0.0, 1.0)
        
        return normalized.astype(np.float32)
        
    except Exception as e:
        print(f"[ERROR] Normalization failed: {e}")
        return None


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def process_pds_image(pds_path: Path,
                     output_dir: Optional[Path] = None,
                     save_intermediate: bool = False) -> Tuple[Optional[np.ndarray], Dict]:
    """
    Complete PDS image processing pipeline.
    
    This convenience function:
    1. Extracts metadata
    2. Loads raw DN16 data
    3. Applies hot pixel filtering
    4. Optionally saves intermediate results
    
    Args:
        pds_path: Path to PDS IMG file
        output_dir: Directory for saving outputs (optional)
        save_intermediate: If True, saves raw and filtered TIFFs
        
    Returns:
        Tuple of (filtered_image, processing_info)
        - filtered_image: uint16 array after hot pixel filtering
        - processing_info: dict with metadata, threshold, num_replaced_pixels
        
    Example:
        >>> img, info = process_pds_image(
        ...     Path("H9463_0050_SR2.IMG"),
        ...     output_dir=Path("processed"),
        ...     save_intermediate=True
        ... )
        >>> print(f"Processed {info['pds_name']}: replaced {info['n_hot']} pixels")
    """
    print(f"\n{'='*80}")
    print(f"Processing: {pds_path.name}")
    print(f"{'='*80}")
    
    # Extract metadata
    metadata = extract_pds_metadata(pds_path)
    print(f"Metadata: MIN={metadata['min']}, MAX={metadata['max']}, "
          f"MEAN={metadata['mean']:.2f}, STD={metadata['std']:.2f}")
    
    # Load image data
    raw_img, img_info = load_pds_image_data(pds_path)
    
    if raw_img is None:
        return None, {'error': 'Failed to load image data'}
    
    print(f"Image size: {img_info['lines']} x {img_info['samples']}")
    
    # Apply hot pixel filtering
    filtered_img, threshold, n_hot = filter_hot_pixels(raw_img)
    print(f"Hot pixel filtering: threshold={threshold}, replaced={n_hot} pixels")
    
    # Save intermediate results if requested
    if save_intermediate and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = pds_path.stem
        
        # Save raw DN16
        raw_path = output_dir / f"{stem}_real_raw_DN16.tif"
        Image.fromarray(raw_img, mode='I;16').save(raw_path)
        print(f"Saved raw: {raw_path.name}")
        
        # Save filtered
        filtered_path = output_dir / f"{stem}_real_raw_DN16_filtered_avg.tif"
        Image.fromarray(filtered_img, mode='I;16').save(filtered_path)
        print(f"Saved filtered: {filtered_path.name}")
    
    # Compile processing info
    info = {
        'pds_name': pds_path.name,
        'pds_stem': pds_path.stem,
        'metadata': metadata,
        'image_info': img_info,
        'threshold': threshold,
        'n_hot': n_hot
    }
    
    return filtered_img, info


if __name__ == "__main__":
    # Example usage
    print("PDS Processor Module")
    print("=" * 80)
    print("This module provides utilities for processing PDS IMG files.")
    print("\nExample usage:")
    print("  from pds_processor import process_pds_image")
    print("  img, info = process_pds_image(Path('H9463_0050_SR2.IMG'))")