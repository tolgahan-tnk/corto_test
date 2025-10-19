#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_runner.py
==================
Main pipeline for PDS IMG to synthetic image comparison.

This script orchestrates the complete workflow:
1. Process PDS IMG file (extract metadata, filter hot pixels)
2. Load synthetic normalized image
3. Perform k-sweep evaluation in both uncropped and cropped modes
4. Save detailed results to Excel/CSV

Usage:
    python pipeline_runner.py --img H9463_0050_SR2.IMG --synthetic template_001.tiff
    
    python pipeline_runner.py \
        --img H9463_0050_SR2.IMG \
        --synthetic template_001_shaped_matched_normalised.tiff \
        --mode both \
        --output results

Author: Post-Processing Pipeline
Date: 2025
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from PIL import Image

# Import our modules
from pds_processor import process_pds_image
from metrics_evaluator import evaluate_k_sweep, get_k_range


# ============================================================================
# CONFIGURATION
# ============================================================================

# Default directories
DEFAULT_IMG_DIR = Path("PDS_Data")
DEFAULT_SYNTHETIC_DIR = Path("Synthetic_image_datas_normalised_afrer_matched")
DEFAULT_OUTPUT_DIR = Path("evaluation_results")

# Processing parameters
SAVE_INTERMEDIATE = True  # Save filtered real images
BUFFER_PERCENT = 5.0      # Buffer for masked region detection (cropped mode)
VERBOSE = True            # Verbose output during k-sweep


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_file(filename: str, search_dirs: list[Path]) -> Path | None:
    """
    Find a file in multiple search directories.
    
    Args:
        filename: Filename to search for
        search_dirs: List of directories to search
        
    Returns:
        Path to file if found, None otherwise
    """
    for directory in search_dirs:
        candidate = directory / filename
        if candidate.exists():
            return candidate
    return None


def save_results_to_excel(results: dict, output_path: Path):
    """
    Save evaluation results to Excel file with multiple sheets.
    
    Creates two sheets:
    - all_sweeps: All k-values with SSIM, RMSE, histogram correlation
    - summary: Best k-values and scores for SSIM and RMSE
    
    Args:
        results: Results dictionary from evaluate_k_sweep()
        output_path: Path for output Excel file
    """
    # Sheet 1: All k-value results
    df_all = pd.DataFrame(results['all_results'])
    
    # Sheet 2: Summary
    summary_data = {
        'Metric': ['SSIM', 'RMSE'],
        'Best_k': [results['best_ssim_k'], results['best_rmse_k']],
        'Best_Score': [results['best_ssim_score'], results['best_rmse_score']]
    }
    df_summary = pd.DataFrame(summary_data)
    
    # Add metadata
    metadata_rows = [
        {'Metric': 'IMG_Stem', 'Best_k': results['img_stem'], 'Best_Score': ''},
        {'Metric': 'Comparison_Mode', 'Best_k': results['mode'], 'Best_Score': ''},
        {'Metric': 'K_Range', 'Best_k': f"{results['k_range'][0]}-{results['k_range'][1]}", 'Best_Score': ''},
    ]
    
    if results['mode'] == 'cropped' and results['cropped_shape'] is not None:
        metadata_rows.append({
            'Metric': 'Cropped_Shape',
            'Best_k': f"{results['cropped_shape'][0]}x{results['cropped_shape'][1]}",
            'Best_Score': ''
        })
    
    df_summary = pd.concat([pd.DataFrame(metadata_rows), df_summary], ignore_index=True)
    
    # Write to Excel with multiple sheets
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        df_all.to_excel(writer, sheet_name='all_sweeps', index=False)
        df_summary.to_excel(writer, sheet_name='summary', index=False)
        
        # Format summary sheet
        workbook = writer.book
        worksheet = writer.sheets['summary']
        
        # Add bold format for headers
        bold = workbook.add_format({'bold': True})
        worksheet.set_row(0, None, bold)
    
    print(f"Results saved to: {output_path}")


def save_results_to_csv(results: dict, output_path: Path):
    """
    Save evaluation results to CSV file.
    
    Saves the all_sweeps data to CSV format.
    
    Args:
        results: Results dictionary from evaluate_k_sweep()
        output_path: Path for output CSV file
    """
    df = pd.DataFrame(results['all_results'])
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_evaluation_pipeline(img_path: Path,
                           synthetic_path: Path,
                           output_dir: Path,
                           mode: str = "both",
                           save_format: str = "both") -> dict:
    """
    Run complete evaluation pipeline for IMG-synthetic pair.
    
    This function:
    1. Processes PDS IMG file (metadata, filtering)
    2. Loads synthetic normalized image
    3. Performs k-sweep evaluation
    4. Saves results in specified format(s)
    
    Args:
        img_path: Path to PDS IMG file
        synthetic_path: Path to synthetic normalized image
        output_dir: Directory for output files
        mode: Evaluation mode - "uncropped", "cropped", or "both"
        save_format: Output format - "excel", "csv", or "both"
        
    Returns:
        Dictionary with evaluation results
    """
    print("\n" + "="*80)
    print("EVALUATION PIPELINE")
    print("="*80)
    print(f"IMG File: {img_path.name}")
    print(f"Synthetic File: {synthetic_path.name}")
    print(f"Mode: {mode}")
    print(f"="*80 + "\n")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== STEP 1: Process PDS IMG ==========
    print("STEP 1: Processing PDS IMG file...")
    filtered_img, img_info = process_pds_image(
        pds_path=img_path,
        output_dir=output_dir / "processed_images",
        save_intermediate=SAVE_INTERMEDIATE
    )
    
    if filtered_img is None:
        print("[ERROR] Failed to process PDS IMG file")
        return {'error': 'PDS processing failed'}
    
    img_stem = img_info['pds_stem']
    
    # ========== STEP 2: Load Synthetic Image ==========
    print("\nSTEP 2: Loading synthetic image...")
    try:
        synth_img = np.array(Image.open(synthetic_path), dtype=np.float32)
        synth_img = np.clip(synth_img, 0.0, 1.0)
        print(f"Synthetic image loaded: {synth_img.shape}")
    except Exception as e:
        print(f"[ERROR] Failed to load synthetic image: {e}")
        return {'error': f'Failed to load synthetic: {e}'}
    
    # Check shape compatibility
    if filtered_img.shape != synth_img.shape:
        print(f"[ERROR] Shape mismatch: real {filtered_img.shape} vs synthetic {synth_img.shape}")
        return {'error': f'Shape mismatch: {filtered_img.shape} vs {synth_img.shape}'}
    
    # ========== STEP 3: K-Sweep Evaluation ==========
    all_results = {}
    
    modes_to_run = []
    if mode == "both":
        modes_to_run = ["uncropped", "cropped"]
    else:
        modes_to_run = [mode]
    
    for eval_mode in modes_to_run:
        print(f"\nSTEP 3: K-Sweep Evaluation (mode={eval_mode})...")
        
        results = evaluate_k_sweep(
            real_raw=filtered_img,
            synthetic_norm=synth_img,
            img_stem=img_stem,
            mode=eval_mode,
            buffer_percent=BUFFER_PERCENT,
            verbose=VERBOSE
        )
        
        all_results[eval_mode] = results
        
        # ========== STEP 4: Save Results ==========
        print(f"\nSTEP 4: Saving results (mode={eval_mode})...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        synth_stem = synthetic_path.stem
        
        base_filename = f"{img_stem}_{synth_stem}_{eval_mode}_{timestamp}"
        
        if save_format in ["excel", "both"]:
            excel_path = output_dir / f"{base_filename}.xlsx"
            save_results_to_excel(results, excel_path)
        
        if save_format in ["csv", "both"]:
            csv_path = output_dir / f"{base_filename}.csv"
            save_results_to_csv(results, csv_path)
    
    # ========== FINAL SUMMARY ==========
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    
    for eval_mode, results in all_results.items():
        print(f"\n{eval_mode.upper()} Mode Results:")
        print(f"  Best SSIM: {results['best_ssim_score']:.6f} at k={results['best_ssim_k']}")
        print(f"  Best RMSE: {results['best_rmse_score']:.6f} at k={results['best_rmse_k']}")
        
        if eval_mode == "cropped" and results['cropped_shape'] is not None:
            print(f"  Cropped to: {results['cropped_shape'][0]} x {results['cropped_shape'][1]} pixels")
    
    print("\n" + "="*80 + "\n")
    
    return all_results


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """
    Command-line interface for the evaluation pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic images against real PDS IMG data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with automatic file search
  python pipeline_runner.py --img H9463_0050_SR2.IMG --synthetic template_001.tiff
  
  # Specify full paths
  python pipeline_runner.py \\
      --img /path/to/H9463_0050_SR2.IMG \\
      --synthetic /path/to/template_001_shaped_matched_normalised.tiff
  
  # Only cropped mode
  python pipeline_runner.py \\
      --img H9463_0050_SR2.IMG \\
      --synthetic template_001.tiff \\
      --mode cropped
  
  # Custom output directory
  python pipeline_runner.py \\
      --img H9463_0050_SR2.IMG \\
      --synthetic template_001.tiff \\
      --output my_results
        """
    )
    
    parser.add_argument(
        '--img',
        required=True,
        help='PDS IMG filename or full path'
    )
    
    parser.add_argument(
        '--synthetic',
        required=True,
        help='Synthetic image filename or full path'
    )
    
    parser.add_argument(
        '--mode',
        choices=['uncropped', 'cropped', 'both'],
        default='both',
        help='Evaluation mode (default: both)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (default: evaluation_results/<IMG_stem>)'
    )
    
    parser.add_argument(
        '--save-format',
        choices=['excel', 'csv', 'both'],
        default='both',
        help='Output file format (default: both)'
    )
    
    parser.add_argument(
        '--img-dir',
        type=str,
        default=None,
        help='Directory to search for IMG file (default: PDS_Data)'
    )
    
    parser.add_argument(
        '--synthetic-dir',
        type=str,
        default=None,
        help='Directory to search for synthetic file (default: Synthetic_image_datas_normalised_afrer_matched)'
    )
    
    args = parser.parse_args()
    
    # Resolve IMG path
    img_path = Path(args.img)
    if not img_path.exists():
        # Try searching in default/specified directories
        search_dirs = [Path.cwd()]
        if args.img_dir:
            search_dirs.append(Path(args.img_dir))
        else:
            search_dirs.append(DEFAULT_IMG_DIR)
        
        img_path = find_file(args.img, search_dirs)
        if img_path is None:
            print(f"[ERROR] IMG file not found: {args.img}")
            print(f"Searched in: {[str(d) for d in search_dirs]}")
            sys.exit(1)
    
    # Resolve synthetic path
    synthetic_path = Path(args.synthetic)
    if not synthetic_path.exists():
        # Try searching in default/specified directories
        search_dirs = [Path.cwd()]
        if args.synthetic_dir:
            search_dirs.append(Path(args.synthetic_dir))
        else:
            search_dirs.append(DEFAULT_SYNTHETIC_DIR)
        
        synthetic_path = find_file(args.synthetic, search_dirs)
        if synthetic_path is None:
            print(f"[ERROR] Synthetic file not found: {args.synthetic}")
            print(f"Searched in: {[str(d) for d in search_dirs]}")
            sys.exit(1)
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        img_stem = img_path.stem
        output_dir = DEFAULT_OUTPUT_DIR / img_stem
    
    # Run pipeline
    results = run_evaluation_pipeline(
        img_path=img_path,
        synthetic_path=synthetic_path,
        output_dir=output_dir,
        mode=args.mode,
        save_format=args.save_format
    )
    
    if 'error' in results:
        print(f"[ERROR] Pipeline failed: {results['error']}")
        sys.exit(1)


if __name__ == "__main__":
    # Check if running with arguments
    if len(sys.argv) > 1:
        main()
    else:
        # Interactive mode for testing
        print("="*80)
        print("PIPELINE RUNNER - Interactive Mode")
        print("="*80)
        print("\nThis script compares PDS IMG files with synthetic images.")
        print("\nUsage:")
        print("  python pipeline_runner.py --img <IMG_FILE> --synthetic <SYNTHETIC_FILE>")
        print("\nFor help:")
        print("  python pipeline_runner.py --help")
        print("\nExample:")
        print("  python pipeline_runner.py \\")
        print("      --img H9463_0050_SR2.IMG \\")
        print("      --synthetic template_001_shaped_matched_normalised.tiff \\")
        print("      --mode both")
        print("\n" + "="*80)