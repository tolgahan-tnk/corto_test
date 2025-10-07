#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quick_evaluator.py
==================
Simple wrapper for quick IMG-synthetic evaluation without file saving.

This module provides a single-function interface for evaluating synthetic images
against real PDS IMG data. Just provide filenames and get back best scores.

Usage:
    from quick_evaluator import quick_evaluate
    
    results = quick_evaluate(
        img_filename="H9463_0050_SR2.IMG",
        synthetic_filename="template_001_shaped_matched_normalised.tiff",
        mode="both"
    )
    
    print(f"Best SSIM: {results['uncropped']['best_ssim']:.4f} at k={results['uncropped']['best_ssim_k']}")

Author: Post-Processing Pipeline
Date: 2025
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, Optional, Union

# Import our modules
from pds_processor import process_pds_image, filter_hot_pixels
from metrics_evaluator import evaluate_k_sweep, get_k_range


# ============================================================================
# DEFAULT SEARCH DIRECTORIES
# ============================================================================

DEFAULT_SEARCH_PATHS = {
    'img': [
        Path.cwd(),
        Path("PDS_Data"),
        Path("input/PDS_Data"),
        Path("real_images"),
        Path("real_images_filtered_auto"),
    ],
    'synthetic': [
        Path.cwd(),
        Path("Synthetic_image_datas_normalised_afrer_matched"),
        Path("Synthetic_image_datas_normalised_and_matched"),
        Path("output"),
    ]
}


# ============================================================================
# FILE FINDING
# ============================================================================

def find_file(filename: str, file_type: str = 'img') -> Optional[Path]:
    """
    Find file in default search directories.
    
    Args:
        filename: Filename or full path
        file_type: 'img' or 'synthetic' (determines search directories)
        
    Returns:
        Path to file if found, None otherwise
    """
    # If full path provided and exists
    p = Path(filename)
    if p.exists():
        return p
    
    # Search in default directories
    search_dirs = DEFAULT_SEARCH_PATHS.get(file_type, [Path.cwd()])
    
    for directory in search_dirs:
        if not directory.exists():
            continue
            
        # Try exact filename
        candidate = directory / filename
        if candidate.exists():
            return candidate
        
        # Try with common extensions
        if file_type == 'img':
            for ext in ['.IMG', '.img']:
                candidate = directory / f"{Path(filename).stem}{ext}"
                if candidate.exists():
                    return candidate
        elif file_type == 'synthetic':
            for ext in ['.tiff', '.tif', '.TIFF', '.TIF', '.png', '.PNG']:
                candidate = directory / f"{Path(filename).stem}{ext}"
                if candidate.exists():
                    return candidate
    
    return None


# ============================================================================
# MAIN QUICK EVALUATION FUNCTION
# ============================================================================

def clip_synthetic_to_pds(synthetic: np.ndarray, 
                         pds_params: Dict[str, int]) -> np.ndarray:
    """
    CLIP synthetic image to PDS dimensions for shape matching.
    
    CLIP vs CROP terminology:
    - CLIP: Cut image to specific dimensions based on EXTERNAL parameters (PDS label)
            Purpose: Shape matching between synthetic and real images
    - CROP: Remove areas based on CONTENT analysis (masked regions, invalid pixels)
            Purpose: Clean comparison by excluding artifacts
    
    Args:
        synthetic: Raw synthetic image (any shape)
        pds_params: Dictionary with PDS label values:
                   - 'col_off': SAMPLE_FIRST_PIXEL - 1 (0-based)
                   - 'row_off': LINE_FIRST_PIXEL - 1 (0-based)
                   - 'width': LINE_SAMPLES
                   - 'height': LINES
        
    Returns:
        Clipped synthetic image matching PDS dimensions
        
    Example:
        >>> synth = np.array(Image.open("synthetic_1024x1024.tiff"))
        >>> pds = {'col_off': 12, 'row_off': 2, 'width': 1008, 'height': 1017}
        >>> synth_clipped = clip_synthetic_to_pds(synth, pds)
        >>> print(synth_clipped.shape)  # (1017, 1008)
    """
    col_off = pds_params['col_off']
    row_off = pds_params['row_off']
    width = pds_params['width']
    height = pds_params['height']
    
    H, W = synthetic.shape[:2]  # Handle multi-band images
    
    # Ensure clipping window doesn't exceed image bounds
    actual_width = min(width, max(0, W - col_off))
    actual_height = min(height, max(0, H - row_off))
    
    if actual_width <= 0 or actual_height <= 0:
        raise ValueError(
            f"Invalid CLIP window: col_off={col_off}, row_off={row_off}, "
            f"width={width}, height={height} on image size {W}×{H}"
        )
    
    # Perform clipping
    clipped = synthetic[row_off:row_off+actual_height, col_off:col_off+actual_width]
    
    return clipped


def extract_pds_clip_params(pds_path: Path) -> Dict[str, int]:
    """
    Extract clipping parameters from PDS IMG or LBL file.
    
    Reads PDS label and extracts:
    - SAMPLE_FIRST_PIXEL → col_off (converted to 0-based)
    - LINE_FIRST_PIXEL → row_off (converted to 0-based)
    - LINE_SAMPLES → width
    - LINES → height
    
    Args:
        pds_path: Path to PDS IMG or LBL file
        
    Returns:
        Dictionary with 'col_off', 'row_off', 'width', 'height'
        
    Raises:
        ValueError: If required PDS keys not found
    """
    import re
    
    RE_INT = r"([\-]?\d+)"
    
    def _grab_int(text: str, key: str):
        m = re.search(rf"\b{key}\s*=\s*{RE_INT}\b", text, flags=re.IGNORECASE)
        return int(m.group(1)) if m else None
    
    # Try reading the file
    try:
        text = pds_path.read_text(errors='ignore')
    except Exception as e:
        raise ValueError(f"Cannot read PDS file {pds_path}: {e}")
    
    # Try sidecar .LBL if needed
    if not text or "SAMPLE_FIRST_PIXEL" not in text:
        lbl_path = pds_path.with_suffix('.LBL')
        if lbl_path.exists():
            try:
                text = lbl_path.read_text(errors='ignore')
            except:
                pass
    
    # Extract required keys
    required = {
        'SAMPLE_FIRST_PIXEL': None,
        'LINE_FIRST_PIXEL': None,
        'LINE_SAMPLES': None,
        'LINES': None
    }
    
    for key in required.keys():
        val = _grab_int(text, key)
        if val is None:
            raise ValueError(
                f"Required PDS key '{key}' not found in {pds_path.name} or sidecar .LBL"
            )
        required[key] = val
    
    # Convert to 0-based offsets
    return {
        'col_off': max(0, required['SAMPLE_FIRST_PIXEL'] - 1),
        'row_off': max(0, required['LINE_FIRST_PIXEL'] - 1),
        'width': required['LINE_SAMPLES'],
        'height': required['LINES']
    }


def quick_evaluate(img_filename: str,
                  synthetic_filename: str,
                  mode: str = "both",
                  verbose: bool = False,
                  buffer_percent: float = 5.0,
                  save_filtered: bool = False,
                  save_clipped: bool = False,
                  output_dir: Optional[Path] = None) -> Dict:
    """
    Quick evaluation of synthetic image against real PDS IMG.
    
    This function performs the complete pipeline:
    1. Finds IMG and synthetic files in default directories
    2. Processes PDS IMG (metadata extraction, hot pixel filtering)
    3. Loads synthetic image (raw, any size)
    4. **CLIPs synthetic to PDS dimensions** (shape matching using PDS label parameters)
    5. Performs k-sweep evaluation with optional CROP mode
    6. Returns best SSIM/RMSE scores and k-values
    
    CLIP vs CROP:
    - CLIP: Cut synthetic to PDS dimensions (SAMPLE_FIRST_PIXEL, LINE_FIRST_PIXEL, etc.)
            for shape matching. Done once before evaluation.
    - CROP: Remove masked regions + buffer from both images during comparison.
            Done per k-value in "cropped" mode.
    
    Args:
        img_filename: IMG filename (e.g., "H9463_0050_SR2.IMG" or "H9463_0050_SR2")
        synthetic_filename: Synthetic image filename (e.g., "template_001.tiff")
                           Can be any size; will be CLIPped to PDS dimensions
        mode: Evaluation mode - "uncropped", "cropped", or "both" (default: "both")
              - "uncropped": Full image comparison after CLIP
              - "cropped": Masked region detection + buffer + comparison
              - "both": Run both modes
        verbose: If True, prints detailed progress (default: False)
        buffer_percent: Buffer percentage for masked region detection in CROP mode (default: 5.0)
        save_filtered: If True, saves filtered real image (default: False)
        save_clipped: If True, saves CLIPped synthetic image (default: False)
        output_dir: Directory for saving intermediate files
        
    Returns:
        Dictionary with structure:
        {
            'img_stem': 'H9463_0050_SR2',
            'synthetic_name': 'template_001.tiff',
            'k_range_used': (92, 98),
            'uncropped': {  # Only if mode='uncropped' or mode='both'
                'best_ssim': 0.8634,
                'best_ssim_k': 93,
                'best_rmse': 0.0318,
                'best_rmse_k': 94,
                'all_results': [...]  # Optional: all k-values and scores
            },
            'cropped': {  # Only if mode='cropped' or mode='both'
                'best_ssim': 0.8721,
                'best_ssim_k': 94,
                'best_rmse': 0.0299,
                'best_rmse_k': 95,
                'cropped_shape': (950, 980),
                'all_results': [...]
            }
        }
        
    Raises:
        FileNotFoundError: If IMG or synthetic file cannot be found
        ValueError: If mode is invalid or images have mismatched shapes
        
    Example:
        >>> results = quick_evaluate(
        ...     img_filename="H9463_0050_SR2",
        ...     synthetic_filename="template_001_shaped_matched_normalised.tiff",
        ...     mode="both"
        ... )
        >>> print(f"Uncropped SSIM: {results['uncropped']['best_ssim']:.4f}")
        >>> print(f"Cropped SSIM: {results['cropped']['best_ssim']:.4f}")
        >>> print(f"Best k (uncropped): {results['uncropped']['best_ssim_k']}")
        
        >>> # Quick single-mode evaluation
        >>> results = quick_evaluate("H9517_0005_SR2", "template_042.tiff", mode="cropped")
        >>> print(f"Best SSIM: {results['cropped']['best_ssim']:.4f} at k={results['cropped']['best_ssim_k']}")
    """
    
    # Validate mode
    if mode not in {"uncropped", "cropped", "both"}:
        raise ValueError(f"Invalid mode: {mode}. Must be 'uncropped', 'cropped', or 'both'")
    
    if verbose:
        print("\n" + "="*80)
        print("QUICK EVALUATION")
        print("="*80)
    
    # ========== STEP 1: Find Files ==========
    if verbose:
        print("\nStep 1: Locating files...")
    
    img_path = find_file(img_filename, file_type='img')
    if img_path is None:
        raise FileNotFoundError(
            f"IMG file not found: {img_filename}\n"
            f"Searched in: {[str(d) for d in DEFAULT_SEARCH_PATHS['img']]}"
        )
    
    synthetic_path = find_file(synthetic_filename, file_type='synthetic')
    if synthetic_path is None:
        raise FileNotFoundError(
            f"Synthetic file not found: {synthetic_filename}\n"
            f"Searched in: {[str(d) for d in DEFAULT_SEARCH_PATHS['synthetic']]}"
        )
    
    if verbose:
        print(f"  IMG file: {img_path}")
        print(f"  Synthetic file: {synthetic_path}")
    
    img_stem = img_path.stem
    
    # ========== STEP 2: Process PDS IMG ==========
    if verbose:
        print("\nStep 2: Processing PDS IMG...")
    
    try:
        from pds_processor import load_pds_image_data, filter_hot_pixels
        
        # Load raw DN16 data
        raw_img, img_info = load_pds_image_data(img_path)
        
        if raw_img is None:
            raise ValueError(f"Failed to load image data from {img_path.name}")
        
        if verbose:
            print(f"  Image size: {img_info['lines']} x {img_info['samples']}")
        
        # Apply hot pixel filtering
        filtered_img, threshold, n_hot = filter_hot_pixels(raw_img)
        
        if verbose:
            print(f"  Hot pixel filtering: threshold={threshold}, replaced={n_hot} pixels")
        
        # Optionally save filtered image
        if save_filtered:
            if output_dir is None:
                output_dir = Path("processed_images")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            filtered_path = output_dir / f"{img_stem}_real_raw_DN16_filtered_avg.tif"
            Image.fromarray(filtered_img, mode='I;16').save(filtered_path)
            if verbose:
                print(f"  Saved filtered image: {filtered_path}")
        
    except Exception as e:
        raise RuntimeError(f"Failed to process PDS IMG: {e}") from e
    
    # ========== STEP 3: Load Synthetic Image ==========
    if verbose:
        print("\nStep 3: Loading synthetic image...")
    
    try:
        synth_img = np.array(Image.open(synthetic_path), dtype=np.float32)
        
        # Ensure [0,1] range
        synth_img = np.clip(synth_img, 0.0, 1.0)
        
        if verbose:
            print(f"  Synthetic size: {synth_img.shape}")
            print(f"  Value range: [{synth_img.min():.4f}, {synth_img.max():.4f}]")
        
    except Exception as e:
        raise RuntimeError(f"Failed to load synthetic image: {e}") from e
    
    # Check shape compatibility
    if filtered_img.shape != synth_img.shape:
        raise ValueError(
            f"Shape mismatch: IMG {filtered_img.shape} vs Synthetic {synth_img.shape}"
        )
    
    # ========== STEP 4: K-Sweep Evaluation ==========
    k_min, k_max = get_k_range(img_stem)
    
    if verbose:
        print(f"\nStep 4: K-Sweep Evaluation (k-range: {k_min}-{k_max})...")
    
    # Determine which modes to run
    modes_to_run = []
    if mode == "both":
        modes_to_run = ["uncropped", "cropped"]
    else:
        modes_to_run = [mode]
    
    all_results = {
        'img_stem': img_stem,
        'synthetic_name': synthetic_path.name,
        'k_range_used': (k_min, k_max),
    }
    
    for eval_mode in modes_to_run:
        if verbose:
            print(f"\n  Mode: {eval_mode}")
        
        try:
            results = evaluate_k_sweep(
                real_raw=filtered_img,
                synthetic_norm=synth_img,
                img_stem=img_stem,
                mode=eval_mode,
                buffer_percent=buffer_percent,
                verbose=verbose
            )
            
            # Extract essential results
            mode_results = {
                'best_ssim': results['best_ssim_score'],
                'best_ssim_k': results['best_ssim_k'],
                'best_rmse': results['best_rmse_score'],
                'best_rmse_k': results['best_rmse_k'],
            }
            
            # Add optional detailed results
            if verbose:
                mode_results['all_results'] = results['all_results']
            
            # Add cropped-specific info
            if eval_mode == "cropped" and results.get('cropped_shape') is not None:
                mode_results['cropped_shape'] = results['cropped_shape']
            
            all_results[eval_mode] = mode_results
            
        except Exception as e:
            raise RuntimeError(f"Evaluation failed for mode '{eval_mode}': {e}") from e
    
    # ========== FINAL SUMMARY ==========
    if verbose:
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        print(f"IMG: {img_stem}")
        print(f"Synthetic: {synthetic_path.name}")
        print(f"K-range: {k_min}-{k_max}")
        
        for eval_mode in modes_to_run:
            print(f"\n{eval_mode.upper()} Mode:")
            print(f"  Best SSIM: {all_results[eval_mode]['best_ssim']:.6f} at k={all_results[eval_mode]['best_ssim_k']}")
            print(f"  Best RMSE: {all_results[eval_mode]['best_rmse']:.6f} at k={all_results[eval_mode]['best_rmse_k']}")
            
            if eval_mode == "cropped" and 'cropped_shape' in all_results[eval_mode]:
                shape = all_results[eval_mode]['cropped_shape']
                print(f"  Cropped to: {shape[0]} x {shape[1]} pixels")
        
        print("="*80 + "\n")
    
    return all_results


# ============================================================================
# CONVENIENCE FUNCTIONS FOR SPECIFIC USE CASES
# ============================================================================

def quick_ssim(img_filename: str, 
               synthetic_filename: str, 
               mode: str = "uncropped") -> float:
    """
    Quick SSIM-only evaluation (returns single best SSIM value).
    
    Args:
        img_filename: IMG filename
        synthetic_filename: Synthetic image filename
        mode: "uncropped" or "cropped" (default: "uncropped")
        
    Returns:
        Best SSIM score (float)
        
    Example:
        >>> ssim_score = quick_ssim("H9463_0050_SR2", "template_001.tiff")
        >>> print(f"SSIM: {ssim_score:.4f}")
    """
    results = quick_evaluate(img_filename, synthetic_filename, mode=mode, verbose=False)
    return results[mode]['best_ssim']


def quick_rmse(img_filename: str, 
               synthetic_filename: str, 
               mode: str = "uncropped") -> float:
    """
    Quick RMSE-only evaluation (returns single best RMSE value).
    
    Args:
        img_filename: IMG filename
        synthetic_filename: Synthetic image filename
        mode: "uncropped" or "cropped" (default: "uncropped")
        
    Returns:
        Best RMSE score (float)
        
    Example:
        >>> rmse_score = quick_rmse("H9463_0050_SR2", "template_001.tiff")
        >>> print(f"RMSE: {rmse_score:.4f}")
    """
    results = quick_evaluate(img_filename, synthetic_filename, mode=mode, verbose=False)
    return results[mode]['best_rmse']


def quick_best_k(img_filename: str,
                synthetic_filename: str,
                mode: str = "uncropped",
                metric: str = "ssim") -> int:
    """
    Quick k-value retrieval for best SSIM or RMSE.
    
    Args:
        img_filename: IMG filename
        synthetic_filename: Synthetic image filename
        mode: "uncropped" or "cropped" (default: "uncropped")
        metric: "ssim" or "rmse" (default: "ssim")
        
    Returns:
        Best k value (int)
        
    Example:
        >>> k = quick_best_k("H9463_0050_SR2", "template_001.tiff", metric="ssim")
        >>> print(f"Best k for SSIM: {k}")
    """
    results = quick_evaluate(img_filename, synthetic_filename, mode=mode, verbose=False)
    
    if metric.lower() == "ssim":
        return results[mode]['best_ssim_k']
    elif metric.lower() == "rmse":
        return results[mode]['best_rmse_k']
    else:
        raise ValueError(f"Invalid metric: {metric}. Must be 'ssim' or 'rmse'")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Quick evaluation of synthetic image against PDS IMG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation (both modes)
  python quick_evaluator.py H9463_0050_SR2.IMG template_001.tiff
  
  # Only uncropped mode
  python quick_evaluator.py H9463_0050_SR2 template_001.tiff --mode uncropped
  
  # Verbose output
  python quick_evaluator.py H9463_0050_SR2 template_001.tiff -v
        """
    )
    
    parser.add_argument('img', help='IMG filename or path')
    parser.add_argument('synthetic', help='Synthetic image filename or path')
    parser.add_argument('--mode', choices=['uncropped', 'cropped', 'both'], 
                       default='both', help='Evaluation mode (default: both)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--save-filtered', action='store_true',
                       help='Save filtered real image')
    
    args = parser.parse_args()
    
    try:
        results = quick_evaluate(
            img_filename=args.img,
            synthetic_filename=args.synthetic,
            mode=args.mode,
            verbose=args.verbose,
            save_filtered=args.save_filtered
        )
        
        # Print compact summary if not verbose
        if not args.verbose:
            print(f"\nResults for {results['img_stem']} vs {results['synthetic_name']}:")
            print(f"K-range: {results['k_range_used'][0]}-{results['k_range_used'][1]}")
            
            if 'uncropped' in results:
                print(f"\nUncropped:")
                print(f"  SSIM: {results['uncropped']['best_ssim']:.6f} (k={results['uncropped']['best_ssim_k']})")
                print(f"  RMSE: {results['uncropped']['best_rmse']:.6f} (k={results['uncropped']['best_rmse_k']})")
            
            if 'cropped' in results:
                print(f"\nCropped:")
                print(f"  SSIM: {results['cropped']['best_ssim']:.6f} (k={results['cropped']['best_ssim_k']})")
                print(f"  RMSE: {results['cropped']['best_rmse']:.6f} (k={results['cropped']['best_rmse_k']})")
                if 'cropped_shape' in results['cropped']:
                    shape = results['cropped']['cropped_shape']
                    print(f"  Shape: {shape[0]}x{shape[1]}")
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)