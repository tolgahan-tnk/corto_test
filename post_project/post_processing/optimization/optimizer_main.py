#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimizer_main.py
=================
Main script for photometric parameter optimization.

Usage:
    # Automatic PDS scanning
    python optimizer_main.py --algorithm pso --pds-dir PDS_Data
    
    # Limit to first 3 images
    python optimizer_main.py --algorithm pso --pds-dir PDS_Data --max-images 3
    
    # Select specific images
    python optimizer_main.py --algorithm pso --pds-dir PDS_Data --img-pattern "H9463*"
"""

import argparse
import json
import sys
import openpyxl
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict

from optimization_helper import (
    OptimizationLogger,
    evaluate_params_with_rendering,
    print_params_table,
    PARAMETER_BOUNDS,
    with_eval_logging
)

from PSO import run_pso
from Genetic import run_genetic

# Add parent directory to path for phobos_data imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from phobos_data import CompactPDSProcessor, HAS_SPICE, SpiceDataProcessor


# ============================================================================
# PDS SCANNING AND METADATA EXTRACTION
# ============================================================================

def scan_pds_directory(pds_dir: Path,
                      max_images: int = None,
                      img_pattern: str = None) -> List[Dict]:
    """
    Automatically scan PDS_Data directory and extract metadata.
    
    This function:
    1. Uses CompactPDSProcessor to scan all IMG files
    2. Extracts UTC times from PDS labels
    3. Queries SPICE for solar distances
    4. Creates img_info_list automatically
    
    Args:
        pds_dir: Path to PDS_Data directory
        max_images: Maximum number of images to process (optional)
        img_pattern: Glob pattern to filter IMG files (optional, e.g., "H9463*")
        
    Returns:
        List of img_info dictionaries with:
        - filename: IMG filename
        - utc_time: UTC timestamp from PDS label
        - solar_distance_km: Solar distance from SPICE
        - pds_path: Full path to IMG file
        
    Example:
        >>> img_list = scan_pds_directory(Path("PDS_Data"), max_images=3)
        >>> print(f"Found {len(img_list)} images")
    """
    print("\n" + "="*80)
    print("SCANNING PDS DIRECTORY")
    print("="*80)
    print(f"Directory: {pds_dir}")
    
    if not pds_dir.exists():
        raise FileNotFoundError(f"PDS directory not found: {pds_dir}")
    
    # Use CompactPDSProcessor to scan directory
    print("\nStep 1: Parsing PDS IMG files...")
    processor = CompactPDSProcessor(pds_dir)
    df = processor.parse_dir()
    
    if df.empty:
        raise RuntimeError(f"No IMG files found in {pds_dir}")
    
    print(f"  Found {len(df)} IMG files")
    
    # Apply filters
    if img_pattern:
        print(f"\nStep 2: Filtering by pattern '{img_pattern}'...")
        import fnmatch
        mask = df['file_name'].apply(lambda x: fnmatch.fnmatch(x, img_pattern))
        df = df[mask].copy()
        print(f"  Matched {len(df)} files")
    
    if max_images is not None and len(df) > max_images:
        print(f"\nStep 3: Limiting to {max_images} images...")
        df = df.head(max_images).copy()
    
    # Filter out images without valid UTC times
    print("\nStep 4: Validating UTC times...")
    before_count = len(df)
    df = df[df['UTC_MEAN_TIME'].notna()].copy()
    after_count = len(df)
    
    if after_count < before_count:
        print(f"  ⚠️ Removed {before_count - after_count} images with invalid UTC times")
    
    if df.empty:
        raise RuntimeError("No IMG files with valid UTC timestamps found")
    
    print(f"  Valid images: {len(df)}")
    
    # Query SPICE for solar distances
    print("\nStep 5: Querying SPICE for solar distances...")
    
    if not HAS_SPICE:
        raise RuntimeError(
            "❌ SPICE is required for optimization but not available!\n"
            "Please install SPICE dependencies:\n"
            "  pip install spiceypy\n"
            "And ensure spice_data_processor.py is available."
        )
    
    sdp = SpiceDataProcessor()
    img_info_list = []
    
    for idx, row in df.iterrows():
        img_name = row['file_name']
        utc_time = row['UTC_MEAN_TIME']
        img_path = row['file_path']
        
        print(f"  [{idx+1}/{len(df)}] {img_name}: {utc_time}", end="")
        
        try:
            # Query SPICE for this UTC time
            spice_data = sdp.get_spice_data(utc_time)
            solar_distance_km = float(spice_data['distances']['sun_to_phobos'])
            
            img_info = {
                'filename': img_name,
                'utc_time': utc_time,
                'solar_distance_km': solar_distance_km,
                'pds_path': img_path
            }
            
            img_info_list.append(img_info)
            print(f" → {solar_distance_km:.0f} km ✓")
            
        except Exception as e:
            print(f" → SPICE query failed: {e}")
            print(f"     Skipping {img_name}")
            continue
    
    if not img_info_list:
        raise RuntimeError("No images could be processed successfully")
    
    # Summary
    print("\n" + "="*80)
    print("PDS SCAN COMPLETE")
    print("="*80)
    print(f"Total images ready for optimization: {len(img_info_list)}")
    print("\nImages:")
    for i, info in enumerate(img_info_list, 1):
        print(f"  {i}. {info['filename']}")
        print(f"     UTC: {info['utc_time']}")
        print(f"     Solar distance: {info['solar_distance_km']:.0f} km")
    print("="*80 + "\n")
    
    return img_info_list


# ============================================================================
# DEFAULT CONFIG
# ============================================================================

DEFAULT_CONFIG = {
    'objective_type': 'ssim',
    'evaluation_mode': 'cropped',
    'aggregation': 'mean',
    
    'algorithm': 'pso',
    
    'pso': {
        'n_particles': 30,
        'n_iterations': 100,
        'w_min': 0.4,      # LDIW: lineer azalan inertia (sona doğru 0.4)
        'w_max': 0.9,      # LDIW: başlangıçta 0.9
        'c1': 2.0,         # LDIW ile uyumlu hızlandırmalar
        'c2': 2.0,
        'v_max_factor': 0.13,
        'boundary_method': 'reflect',
        'convergence_threshold': 1e-6,
        'convergence_patience': 1000000,
        'seed': None
    },
    
    'genetic': {
        'population_size': 50,
        'n_generations': 100,
        'crossover_prob': 0.9,
        'mutation_prob': 0.1,
        'crossover_eta': 20.        0,
        'mutation_eta': 20.0,
        'tournament_size': 3,
        'elitism_count': 2,
        'boundary_method': 'clip',
        'convergence_threshold': 1e-6,
        'convergence_patience': 1000000000,
        'seed': None
    },
    
    'output_dir': 'optimization_results',
    'experiment_name': 'phobos_optimization',
    'temp_dir': 'optimization_temp',
    'save_plots': True,
    'verbose_eval': False
}


# ============================================================================
# OBJECTIVE FUNCTION FACTORY
# ============================================================================

def create_objective_function(config: Dict, img_info_list: List[Dict]):
    """Create objective function from configuration."""
    
    particle_counter = {'count': 0}
    learned_k_db: Dict[str, int] = {}
    
    def objective_function(params):
        particle_id = particle_counter['count']
        particle_counter['count'] += 1
        
        return evaluate_params_with_rendering(
            params=params,
            img_info_list=img_info_list,
            objective_type=config['objective_type'],
            mode=config['evaluation_mode'],
            aggregation=config['aggregation'],
            particle_id=particle_id,
            temp_dir=Path(config.get('temp_dir', 'optimization_temp')),
            verbose=config.get('verbose_eval', False),
            k_mode=config.get('k_mode', 'learned'),
            learned_k_db=learned_k_db,
            blender_persistent=config.get('blender_persistent', True),
            blender_batch_size=config.get('blender_batch_size', None),
        )
    
    # Wrapped objective with logging - img_info_list eklendi
    wrapped_objective = with_eval_logging(
        objective_fn=objective_function,
        pop_size=config[config['algorithm']].get('n_particles' if config['algorithm'] == 'pso' else 'population_size', 30),
        img_info_list=img_info_list,  # ← YENİ PARAMETRE
        csv_dir=Path(config.get('temp_dir', 'optimization_temp'))
    )
    
    return wrapped_objective


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Photometric Parameter Optimization with Automatic PDS Scanning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # Automatic scanning of all IMG files in PDS_Data
    python optimizer_main.py --algorithm pso --pds-dir PDS_Data
    
    # Limit to first 3 images
    python optimizer_main.py --algorithm pso --pds-dir PDS_Data --max-images 3
    
    # Filter by pattern
    python optimizer_main.py --algorithm pso --pds-dir PDS_Data --img-pattern "H9463*"
    
    # Genetic Algorithm with config file
    python optimizer_main.py --algorithm genetic --config my_config.json --pds-dir PDS_Data
    
    # Custom settings
    python optimizer_main.py \\
      --algorithm pso \\
      --pds-dir PDS_Data \\
      --max-images 5 \\
      --iterations 50 \\
      --population 20 \\
      --objective ssim \\
      --mode cropped
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--pds-dir',
        type=str,
        default='PDS_Data',
        help='PDS IMG directory to scan (default: PDS_Data)'
    )
    
    # Algorithm selection
    parser.add_argument(
        '--algorithm',
        choices=['pso', 'genetic', 'ga'],
        default='pso',
        help='Optimization algorithm (default: pso)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='JSON configuration file (optional)'
    )
    
    # Image selection
    parser.add_argument(
        '--max-images',
        type=int,
        help='Maximum number of images to process'
    )
    
    parser.add_argument(
        '--img-pattern',
        type=str,
        help='Glob pattern to filter IMG files (e.g., "H9463*", "HF*")'
    )
    
    # Objective function settings
    parser.add_argument(
        '--objective',
        choices=['ssim', 'rmse', 'nmrse', 'combined'],
        default='ssim',
        help='Objective function type (default: ssim)'
    )
    
    parser.add_argument(
        '--mode',
        choices=['cropped', 'uncropped'],
        default='cropped',
        help='Evaluation mode (default: cropped)'
    )
    
    parser.add_argument(
        '--aggregation',
        choices=['mean', 'median', 'max'],
        default='mean',
        help='Multi-image aggregation method (default: mean)'
    )
    
    # Output settings
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory'
    )
    
    parser.add_argument(
        '--name',
        type=str,
        help='Experiment name'
    )
    
    # Algorithm parameters
    parser.add_argument(
        '--iterations',
        type=int,
        help='Number of iterations/generations'
    )
    
    parser.add_argument(
        '--population',
        type=int,
        help='Population/swarm size'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose evaluation output'
    )
    
    # parser = argparse.ArgumentParser(...)
    parser.add_argument("--blender-batch-size", type=int, default=20,
                        help="Sahneyi her N renderda bir yeniden kur (persistent cache)")
    parser.add_argument("--k-mode", choices=["sweep","learned"], default="learned",
                        help="k araması: 'sweep' ya da 'learned' (tek k, guard+dar bant teyit)")

    args = parser.parse_args()
    
    # ========== Load Configuration ==========
    config = DEFAULT_CONFIG.copy()
    
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Config file not found: {args.config}")
            sys.exit(1)
        
        with open(config_path) as f:
            user_config = json.load(f)
        
        for key, value in user_config.items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
    
    # Override with command-line arguments
    if args.algorithm:
        config['algorithm'] = args.algorithm
    if args.objective:
        config['objective_type'] = args.objective
    if args.mode:
        config['evaluation_mode'] = args.mode
    if args.aggregation:
        config['aggregation'] = args.aggregation
    if args.output:
        config['output_dir'] = args.output
    if args.name:
        config['experiment_name'] = args.name
    if args.verbose:
        config['verbose_eval'] = True
    
    algorithm_key = 'pso' if args.algorithm == 'pso' else 'genetic'
    
    if args.iterations:
        if args.algorithm == 'pso':
            config['pso']['n_iterations'] = args.iterations
        else:
            config['genetic']['n_generations'] = args.iterations
    
    if args.population:
        if args.algorithm == 'pso':
            config['pso']['n_particles'] = args.population
        else:
            config['genetic']['population_size'] = args.population
    
    if args.seed:
        config[algorithm_key]['seed'] = args.seed

    # --- : CLI'dan gelen k arama modu config'e yaz ---
    config['k_mode'] = args.k_mode
    # YENİ: Blender kalıcılık/batch ayarları
    config['blender_persistent'] = True
    config['blender_batch_size'] = args.blender_batch_size
    # ========== Scan PDS Directory ==========
    try:
        img_info_list = scan_pds_directory(
            pds_dir=Path(args.pds_dir),
            max_images=args.max_images,
            img_pattern=args.img_pattern
        )
    except Exception as e:
        print(f"\n❌ Error scanning PDS directory: {e}")
        sys.exit(1)
    
    # ========== Initialize Logger ==========
    output_dir = Path(config['output_dir'])
    experiment_name = config['experiment_name']
    
    logger = OptimizationLogger(output_dir, experiment_name)
    logger.set_config(config)
    
    # ========== Print Configuration ==========
    logger.log("\n" + "="*80)
    logger.log("PHOTOMETRIC PARAMETER OPTIMIZATION")
    logger.log("="*80)
    
    logger.log(f"\nPDS Directory: {args.pds_dir}")
    logger.log(f"Images: {len(img_info_list)}")
    for img_info in img_info_list:
        logger.log(f"  - {img_info['filename']}")
        logger.log(f"    UTC: {img_info['utc_time']}")
        logger.log(f"    Solar distance: {img_info['solar_distance_km']:.0f} km")
    
    logger.log(f"\nObjective: {config['objective_type']}")
    logger.log(f"Mode: {config['evaluation_mode']}")
    logger.log(f"Aggregation: {config['aggregation']}")
    logger.log(f"Algorithm: {config['algorithm'].upper()}")
    
    logger.log("\nParameter Bounds:")
    for name, (lower, upper) in PARAMETER_BOUNDS.items():
        logger.log(f"  {name:<15}: [{lower:.4f}, {upper:.4f}]")
    
    # ========== Create Objective Function ==========
    objective_function = create_objective_function(config, img_info_list)
    
    # ========== Run Optimization ==========
    try:
        algorithm = config['algorithm'].lower()
        
        if algorithm == 'pso':
            logger.log("\nRunning Particle Swarm Optimization...")
            results = run_pso(
                objective_function=objective_function,
                config=config['pso'],
                logger=logger
            )
        
        elif algorithm in ['genetic', 'ga']:
            logger.log("\nRunning Genetic Algorithm...")
            results = run_genetic(
                objective_function=objective_function,
                config=config['genetic'],
                logger=logger
            )
        
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # ========== Save XLSX (YENİ) ==========
        if hasattr(objective_function, 'save_xlsx'):
            objective_function.save_xlsx()
        
        if hasattr(objective_function, 'csv_file') and objective_function.csv_file:
            objective_function.csv_file.close()
        
        # ========== Save Results ==========
        logger.log("\n" + "="*80)
        logger.log("FINAL RESULTS")
        logger.log("="*80)
        
        logger.log(f"\nBest Objective: {results['best_objective']:.6f}")
        logger.log("\nBest Parameters:")
        print_params_table(results['best_params'])
        
        logger.save_history()
        
        if config['save_plots']:
            logger.plot_convergence()
        
        # Save final results with IMG info
        final_results_file = output_dir / f"{experiment_name}_{logger.timestamp}_final.json"
        with open(final_results_file, 'w') as f:
            json.dump({
                'config': config,
                'images': img_info_list,
                'results': {
                    'best_objective': float(results['best_objective']),
                    'best_params': results['best_params'].tolist(),
                    'n_iterations': results.get('n_iterations', results.get('n_generations', 0))
                }
            }, f, indent=2)
        
        logger.log(f"\n✅ Optimization complete! Results saved to {output_dir}")
        
    except KeyboardInterrupt:
        logger.log("\n⚠️ Interrupted by user")
        logger.save_history()
        sys.exit(1)
    
    except Exception as e:
        logger.log(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        logger.save_history()
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("="*80)
        print("PHOTOMETRIC PARAMETER OPTIMIZATION")
        print("="*80)
        print("\nThis tool optimizes photometric parameters using PDS IMG files.")
        print("\nUsage:")
        print("  python optimizer_main.py --algorithm pso --pds-dir PDS_Data")
        print("\nFor help:")
        print("  python optimizer_main.py --help")
        print("\nExamples:")
        print("  # Process first 3 images")
        print("  python optimizer_main.py --pds-dir PDS_Data --max-images 3")
        print()
        print("  # Filter by pattern")
        print("  python optimizer_main.py --pds-dir PDS_Data --img-pattern 'H9463*'")
        print()
        print("  # Full optimization")
        print("  python optimizer_main.py \\")
        print("      --algorithm pso \\")
        print("      --pds-dir PDS_Data \\")
        print("      --max-images 5 \\")
        print("      --iterations 50 \\")
        print("      --population 20")
        print("="*80)
    else:
        main()