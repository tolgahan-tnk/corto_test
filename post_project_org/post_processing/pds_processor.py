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
    Load 16-bit big-endian image data from PDS IMG file.
    
    Reads the image dimensions and binary data offset from the PDS label,
    then loads the raw pixel values as a numpy array.
    
    Args:
        pds_path: Path to PDS IMG file
        
    Returns:
        Tuple of (image_array, image_info)
        - image_array: uint16 numpy array (H x W), or None if loading fails
        - image_info: dict with 'lines', 'samples', 'record_bytes', 'image_record'
        
    Example:
        >>> img, info = load_pds_image_data(Path("H9463_0050_SR2.IMG"))
        >>> print(img.shape)  # (1024, 1024)
        >>> print(info['lines'])  # 1024
    """
    try:
        # Read PDS label to get image parameters
        content = open(pds_path, 'r', errors='ignore').read()
        
        record_bytes = int(re.search(r"RECORD_BYTES\s*=\s*(\d+)", content).group(1))
        image_record = int(re.search(r"\^IMAGE\s*=\s*(\d+)", content).group(1))
        lines = int(re.search(r"LINES\s*=\s*(\d+)", content).group(1))
        samples = int(re.search(r"LINE_SAMPLES\s*=\s*(\d+)", content).group(1))
        
        image_info = {
            'lines': lines,
            'samples': samples,
            'record_bytes': record_bytes,
            'image_record': image_record
        }
        
        # Load 16-bit big-endian binary data
        with open(pds_path, 'rb') as f:
            # Seek to image data start (image_record is 1-indexed)
            f.seek((image_record - 1) * record_bytes)
            
            # Read as 16-bit big-endian signed integers
            raw_be = np.fromfile(f, dtype='>i2', count=lines * samples)
        
        # Convert to uint16 bit pattern (preserves DN values)
        raw_uint16 = raw_be.astype(np.uint16).reshape((lines, samples))
        
        return raw_uint16, image_info
        
    except Exception as e:
        print(f"[ERROR] Failed to load image data from {pds_path.name}: {e}")
        return None, {}


# ============================================================================
# HOT PIXEL FILTERING
# ============================================================================

def compute_dynamic_threshold(img_u16: np.ndarray,
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
        img_u16: Input uint16 image array
        n_bins: Number of histogram bins (default: 256)
        factor: Multiplier for threshold (default: 1.5)
        gap_len: Minimum gap length to consider end of continuity (default: 2)
        
    Returns:
        Dynamic threshold value (int)
        
    Example:
        >>> img = np.array(Image.open("real_raw.tif"), dtype=np.uint16)
        >>> thr = compute_dynamic_threshold(img)
        >>> print(f"Hot pixel threshold: {thr}")
        Hot pixel threshold: 58234
    """
    DN_MAX = 65535
    
    # Create histogram with fixed bins for DN16 range
    bins = np.linspace(0, DN_MAX + 1, n_bins + 1)
    hist, edges = np.histogram(img_u16, bins=bins)
    
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


def eight_neighbor_mean(raw_u16: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """
    Compute 8-neighbor mean for hot pixel replacement.
    
    For each pixel, computes the mean of its 8 neighbors. Preferentially
    uses only valid (non-hot) neighbors if available, otherwise uses all.
    
    Args:
        raw_u16: Input uint16 image
        valid_mask: Boolean mask (True = valid pixel, False = hot pixel)
        
    Returns:
        Array of mean values for each pixel position
        
    Note:
        - Uses zero-padding at image boundaries
        - Returns float64 array (should be rounded/clipped before use)
    """
    img = raw_u16.astype(np.float64)
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


def filter_hot_pixels(raw_u16: np.ndarray,
                     n_bins: int = 256,
                     factor: float = 1.5,
                     gap_len: int = 2) -> Tuple[np.ndarray, int, int]:
    """
    Apply hot pixel filtering to raw 16-bit image.
    
    This is the main filtering function that:
    1. Computes dynamic threshold from histogram
    2. Identifies hot pixels above threshold
    3. Replaces them with 8-neighbor mean
    
    Args:
        raw_u16: Input uint16 image
        n_bins: Histogram bins for threshold calculation
        factor: Threshold multiplier
        gap_len: Gap detection parameter
        
    Returns:
        Tuple of (filtered_image, threshold_used, num_replaced_pixels)
        
    Example:
        >>> raw = np.array(Image.open("real_raw.tif"), dtype=np.uint16)
        >>> filtered, thr, n_hot = filter_hot_pixels(raw)
        >>> print(f"Replaced {n_hot} hot pixels using threshold {thr}")
        Replaced 342 hot pixels using threshold 58234
    """
    # Compute dynamic threshold
    thr = compute_dynamic_threshold(raw_u16, n_bins=n_bins, factor=factor, gap_len=gap_len)
    
    # Identify hot pixels
    hot_mask = raw_u16 > thr
    valid_mask = raw_u16 <= thr
    
    # Compute replacement values using 8-neighbor mean
    nbr_mean = eight_neighbor_mean(raw_u16, valid_mask)
    
    # Replace hot pixels
    filtered = raw_u16.copy()
    filtered_vals = np.rint(nbr_mean[hot_mask]).clip(0, 65535).astype(np.uint16)
    filtered[hot_mask] = filtered_vals
    
    n_hot = int(hot_mask.sum())
    
    return filtered, thr, n_hot


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