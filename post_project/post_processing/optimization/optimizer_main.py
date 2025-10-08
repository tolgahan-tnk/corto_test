#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimizer_main.py
=================
Main script for photometric parameter optimization.

Usage:
    python optimizer_main.py --algorithm pso --img-metadata img_metadata.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np

from optimization_helper import (
    OptimizationLogger,
    evaluate_params_with_rendering,
    print_params_table,
    PARAMETER_BOUNDS
)

from PSO import run_pso
from Genetic import run_genetic


# ============================================================================
# IMG METADATA LOADING
# ============================================================================

def load_img_metadata(metadata_file: Path) -> List[Dict]:
    """
    Load IMG metadata from JSON file.
    
    Expected format:
    {
      "images": [
        {
          "filename": "H9463_0050_SR2.IMG",
          "utc_time": "2019-06-07T10:46:42.8575Z",
          "solar_distance_km": 228000000.0,
          "pds_path": "PDS_Data/H9463_0050_SR2.IMG"
        },
        ...
      ]
    }
    """
    with open(metadata_file) as f:
        data = json.load(f)
    
    return data['images']


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
        'w': 0.7,
        'w_min': 0.4,
        'w_max': 0.9,
        'c1': 1.5,
        'c2': 1.5,
        'v_max_factor': 0.2,
        'boundary_method': 'clip',
        'convergence_threshold': 1e-6,
        'convergence_patience': 10,
        'seed': None
    },
    
    'genetic': {
        'population_size': 50,
        'n_generations': 100,
        'crossover_prob': 0.9,
        'mutation_prob': 0.1,
        'crossover_eta': 20.0,
        'mutation_eta': 20.0,
        'tournament_size': 3,
        'elitism_count': 2,
        'boundary_method': 'clip',
        'convergence_threshold': 1e-6,
        'convergence_patience': 10,
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
            verbose=config.get('verbose_eval', False)
        )
    
    return objective_function


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Photometric Parameter Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # PSO with metadata file
  python optimizer_main.py --algorithm pso --img-metadata img_metadata.json
  
  # Genetic Algorithm with config
  python optimizer_main.py --algorithm genetic --config my_config.json
  
  # Custom settings
  python optimizer_main.py --algorithm pso --img-metadata imgs.json --iterations 50 --population 20
        """
    )
    
    parser.add_argument('--algorithm', choices=['pso', 'genetic', 'ga'], 
                       default='pso', help='Optimization algorithm')
    parser.add_argument('--config', type=str, help='JSON configuration file')
    parser.add_argument('--img-metadata', type=str, required=True,
                       help='JSON file with IMG metadata')
    parser.add_argument('--objective', choices=['ssim', 'rmse', 'nmrse', 'combined'],
                       default='ssim')
    parser.add_argument('--mode', choices=['cropped', 'uncropped'], default='cropped')
    parser.add_argument('--aggregation', choices=['mean', 'median', 'max'], default='mean')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--name', type=str, help='Experiment name')
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--population', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--verbose', action='store_true', help='Verbose evaluation')
    
    args = parser.parse_args()
    
    # Load config
    config = DEFAULT_CONFIG.copy()
    
    if args.config:
        with open(args.config) as f:
            user_config = json.load(f)
        for key, value in user_config.items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
    
    # Override with args
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
    
    # Load IMG metadata
    metadata_file = Path(args.img_metadata)
    if not metadata_file.exists():
        print(f"Error: Metadata file not found: {args.img_metadata}")
        sys.exit(1)
    
    img_info_list = load_img_metadata(metadata_file)
    
    if not img_info_list:
        print("Error: No images in metadata file")
        sys.exit(1)
    
    # Initialize logger
    output_dir = Path(config['output_dir'])
    experiment_name = config['experiment_name']
    
    logger = OptimizationLogger(output_dir, experiment_name)
    logger.set_config(config)
    
    # Print config
    logger.log("\n" + "="*80)
    logger.log("PHOTOMETRIC PARAMETER OPTIMIZATION")
    logger.log("="*80)
    
    logger.log(f"\nImages: {len(img_info_list)}")
    for img_info in img_info_list:
        logger.log(f"  - {img_info['filename']}")
    
    logger.log(f"\nObjective: {config['objective_type']}")
    logger.log(f"Mode: {config['evaluation_mode']}")
    logger.log(f"Aggregation: {config['aggregation']}")
    logger.log(f"Algorithm: {config['algorithm'].upper()}")
    
    logger.log("\nParameter Bounds:")
    for name, (lower, upper) in PARAMETER_BOUNDS.items():
        logger.log(f"  {name:<15}: [{lower:.4f}, {upper:.4f}]")
    
    # Create objective function
    objective_function = create_objective_function(config, img_info_list)
    
    # Run optimization
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
        
        # Save results
        logger.log("\n" + "="*80)
        logger.log("FINAL RESULTS")
        logger.log("="*80)
        
        logger.log(f"\nBest Objective: {results['best_objective']:.6f}")
        logger.log("\nBest Parameters:")
        print_params_table(results['best_params'])
        
        logger.save_history()
        
        if config['save_plots']:
            logger.plot_convergence()
        
        # Save final results
        final_results_file = output_dir / f"{experiment_name}_{logger.timestamp}_final.json"
        with open(final_results_file, 'w') as f:
            json.dump({
                'config': config,
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
        print("\nUsage:")
        print("  python optimizer_main.py --algorithm pso --img-metadata <FILE>")
        print("\nFor help:")
        print("  python optimizer_main.py --help")
        print("="*80)
    else:
        main()
