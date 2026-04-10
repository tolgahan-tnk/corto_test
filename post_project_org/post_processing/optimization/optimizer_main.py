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
from typing import List, Dict, Optional

from optimization_helper import (
    N_PARAMS,
    OptimizationLogger,
    evaluate_params_with_rendering,
    print_params_table,
    PARAMETER_BOUNDS,
    with_eval_logging,
    reset_adaptive_crop_state,
    save_checkpoint,
    load_checkpoint,
    get_checkpoint_path,
    set_fixed_params,
    get_active_n_params
)

from PSO import run_pso
from Genetic import run_genetic
from Bayesian import run_bayesian
from CMAES import run_cmaes, run_ipop_cmaes, run_bipop_cmaes

# Add parent directory to path for phobos_data imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from phobos_data import CompactPDSProcessor, HAS_SPICE, SpiceDataProcessor
from mission_config import detect_mission, get_solar_distance_from_label, extract_pose_from_label


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
    
    # Query geometry per file (per-file mission detection for mixed directories)
    print("\nStep 5: Detecting mission and querying geometry...")

    # Lazy SPICE init: only created if any file needs it
    sdp = None

    img_info_list = []
    missions_seen = set()

    for idx, row in df.iterrows():
        img_name = row['file_name']
        utc_time = row['UTC_MEAN_TIME']
        img_path = row['file_path']

        # Per-file mission detection (supports mixed HRSC + OSIRIS directories)
        file_mission_cfg = detect_mission(Path(img_path))
        missions_seen.add(file_mission_cfg.mission_id)

        print(f"  [{idx+1}/{len(df)}] {img_name} [{file_mission_cfg.mission_id}]: {utc_time}", end="")

        try:
            if file_mission_cfg.use_spice:
                # ── HRSC path: SPICE query ──
                if sdp is None:
                    if not HAS_SPICE:
                        raise RuntimeError(
                            "SPICE is required for HRSC but not available!\n"
                            "Install: pip install spiceypy"
                        )
                    sdp = SpiceDataProcessor()
                spice_data = sdp.get_spice_data(utc_time)
                solar_distance_km = float(spice_data['distances']['sun_to_phobos'])
            else:
                # ── Label path: extract from PDS header ──
                solar_distance_km = get_solar_distance_from_label(Path(img_path))

            img_info = {
                'filename': img_name,
                'utc_time': utc_time,
                'solar_distance_km': solar_distance_km,
                'pds_path': img_path,
                'mission_config': file_mission_cfg,   # per-file config
            }

            img_info_list.append(img_info)
            print(f" → {solar_distance_km:.0f} km ✓")

        except Exception as e:
            print(f" → Query failed: {e}")
            print(f"     Skipping {img_name}")
            continue

    if not img_info_list:
        raise RuntimeError("No images could be processed successfully")

    # Summary
    missions_str = ", ".join(sorted(missions_seen))
    print("\n" + "="*80)
    print("PDS SCAN COMPLETE")
    print("="*80)
    print(f"Missions: {missions_str}")
    print(f"Total images ready for optimization: {len(img_info_list)}")
    print("\nImages:")
    for i, info in enumerate(img_info_list, 1):
        mission_id = info['mission_config'].mission_id
        print(f"  {i}. {info['filename']}  [{mission_id}]")
        print(f"     UTC: {info['utc_time']}")
        print(f"     Solar distance: {info['solar_distance_km']:.0f} km")
    print("="*80 + "\n")

    return img_info_list



# ============================================================================
# DEFAULT CONFIG
# ============================================================================

DEFAULT_CONFIG = {
    'objective_type': 'combined',
    'evaluation_mode': 'cropped',
    'aggregation': 'mean',
    
    'algorithm': 'pso',
    
    'pso': {
        'n_particles': 30,
        'n_iterations': 100,
        'w_min': 0.5,      # LDIW: lineer azalan inertia (sona doğru 0.4)
        'w_max': 0.8,      # LDIW: başlangıçta 0.9
        'c1': 2.0,         # LDIW ile uyumlu hızlandırmalar
        'c2': 2.0,
        'v_max_factor': 0.15,
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
        'crossover_eta': 20.0,        
        'mutation_eta': 20.0,
        'tournament_size': 3,
        'elitism_count': 2,
        'boundary_method': 'clip',
        'convergence_threshold': 1e-6,
        'convergence_patience': 1000000000,
        'seed': None
    },

    'bayesian': {
        'n_calls': 50,
        'n_initial_points': 10,
        'acq_func': 'gp_hedge',
        'xi': 0.01,
        'kappa': 1.96,
        'seed': None
    },

    'cmaes': {
        'sigma0': 0.3,
        'population_size': None,  # auto: 4 + floor(3*ln(n))
        'n_iterations': 500,
        'seed': None,
        'convergence_threshold': 1e-11,
        'convergence_patience': 1000000
    },
    
    'output_dir': 'optimization_results',
    'experiment_name': 'phobos_optimization',
    'temp_dir': str(Path(r"C:/CORTO/optimization_temp_combined_threshold_mars_018_2")), #'optimization_temp',
    'save_plots': True,
    'verbose_eval': False
}


# ============================================================================
# OBJECTIVE FUNCTION FACTORY
# ============================================================================

def create_objective_function(config: Dict, img_info_list: List[Dict], resume_state: Optional[dict] = None):
    """Create objective function from configuration.
    
    Args:
        config: Optimization configuration
        img_info_list: List of image info dictionaries
        resume_state: Optional checkpoint state for resuming
    """
    
    # Resume: start from previous particle count
    start_count = resume_state.get('iteration_count', 0) if resume_state else 0
    particle_counter = {'count': start_count}
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
            adaptive_crop_n=config.get('adaptive_crop_n', 0),  # YENİ: Adaptive crop
            # YENİ: Atmosphere & Displacement
            use_displacement=config.get('use_displacement', True),
            use_atmosphere=config.get('use_atmosphere', True),
            atm_color_mode=config.get('atm_color_mode', 'single'),
            dem_path=config.get('dem_path', None),
            fixed_k=config.get('fixed_k', None),
        )
    
    # Wrapped objective with logging - start_eval_idx eklendi
    # Bayesian tek tek değerlendirir (pop_size=1), PSO/Genetic popülasyon kullanır
    algorithm = config['algorithm'].lower()
    if algorithm in ['bayesian', 'bo']:
        pop_size = 1  # Bayesian: her eval bir "iterasyon"
    elif algorithm == 'pso':
        pop_size = config['pso'].get('n_particles', 30)
    elif algorithm == 'cmaes':
        # CMA-ES: auto popsize = 4 + floor(3*ln(n)), fallback 11 for 14D
        pop_size = config['cmaes'].get('population_size') or 11
    else:  # genetic
        pop_size = config['genetic'].get('population_size', 30)
    
    wrapped_objective = with_eval_logging(
        objective_fn=objective_function,
        pop_size=pop_size,
        img_info_list=img_info_list,
        csv_dir=Path(config.get('temp_dir', 'optimization_temp')),
        start_eval_idx=start_count  # Resume: logging da doğru iterasyondan başlasın
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
        choices=['pso', 'genetic', 'ga', 'bayesian', 'bo', 'cmaes'],
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
        choices=['ssim', 'rmse', 'nmrse', 'combined', 'combined-new'],
        default='ssim',
        help='Objective function type (default: ssim). combined-new uses MS-SSIM + NMRSE + GMSD'
    )
    
    parser.add_argument(
        '--mode',
        choices=['cropped', 'uncropped'],
        default='cropped',
        help='Evaluation mode (default: cropped)'
    )
    
    parser.add_argument(
        '--aggregation',
        choices=['mean', 'median', 'max', 'rss'],
        default='mean',
        help="Multi-image aggregation: 'mean','median','max', or 'rss' (root-sum-of-squares). "
             "Auto-set to 'rss' when >1 image and not explicitly overridden. (default: mean)"
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
    
    parser.add_argument("--blender-batch-size", type=int, default=100,
                        help="Sahneyi her N renderda bir yeniden kur (persistent cache)")
    parser.add_argument("--k-mode", choices=["sweep","learned"], default="learned",
                        help="k araması: 'sweep' ya da 'learned' (tek k, guard+dar bant teyit)")
    parser.add_argument("--adaptive-crop", type=int, default=10, metavar="N",
                        help="İlk N iterasyonda bbox öğren, sonra kilitle (0=devre dışı)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint dosyasından devam et (pickle)")
    parser.add_argument("--fixed-k", type=int, default=None,
                        help="K-percentile değerini sabitle (örn. 0). Otomatik k-sweep yerine bu değer kullanılır.")
    parser.add_argument("--checkpoint-interval", type=int, default=50,
                        help="Her N iterasyonda checkpoint kaydet (default: 50)")
    parser.add_argument("--restarts", type=int, default=0,
                        help="IPOP/BIPOP-CMA-ES restart sayisi (0=disabled, default: 0). "
                             "Her restart'ta population 2x buyur, sigma 2x artar. "
                             "Ornek: --restarts 2 -> Restart 0: lambda=11, Restart 1: lambda=22")
    parser.add_argument("--bipop", action="store_true",
                        help="BIPOP-CMA-ES kullan: buyuk pop (exploration) + kucuk pop (exploitation) "
                             "donusumlu restart stratejisi. --restarts ile birlikte kullanilir.")
    
    # Render-only mode
    parser.add_argument("--render-only", action="store_true",
                        help="Sadece --params ile verilen parametrelerle render yap, optimizasyon yapma")
    parser.add_argument("--params", type=str, default=None,
                        help="Render için parametreler (virgülle ayrılmış): base_gray,tex_mix,...")

    # ---- Atmosphere & Displacement ----
    parser.add_argument("--no-atmosphere", action="store_true",
                        help="Disable volumetric atmosphere sphere")
    parser.add_argument("--no-displacement", action="store_true",
                        help="Use bump mapping instead of MOLA DEM displacement")
    parser.add_argument("--dem-path", type=str, default=None,
                        help="Path to Mars DEM TIFF for displacement mapping")
    parser.add_argument("--atm-color-mode", choices=['single', 'rgb'], default='single',
                        help="Atmosphere color parameterization: 'single' (R only) or 'rgb' (default: single)")
    parser.add_argument("--fix", type=str, nargs='*', default=[],
                        help="Fix parameters at given values during optimization. "
                             "Format: param=value (e.g., --fix atm_beta0=3e-4 atm_scale_height=11.0)")

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
    
    if args.algorithm == 'pso':
        algorithm_key = 'pso'
    elif args.algorithm in ['genetic', 'ga']:
        algorithm_key = 'genetic'
    elif args.algorithm == 'cmaes':
        algorithm_key = 'cmaes'
    else:
        algorithm_key = 'bayesian'
    
    if args.iterations:
        if args.algorithm == 'pso':
            config['pso']['n_iterations'] = args.iterations
        elif args.algorithm in ['genetic', 'ga']:
            config['genetic']['n_generations'] = args.iterations
        elif args.algorithm in ['bayesian', 'bo']:
            config['bayesian']['n_calls'] = args.iterations
        elif args.algorithm == 'cmaes':
            config['cmaes']['n_iterations'] = args.iterations
    
    if args.population:
        if args.algorithm == 'pso':
            config['pso']['n_particles'] = args.population
        elif args.algorithm == 'cmaes':
            config['cmaes']['population_size'] = args.population
        else:
            config['genetic']['population_size'] = args.population
    
    if args.seed:
        config[algorithm_key]['seed'] = args.seed

    # --- CLI'dan gelen k arama modu config'e yaz ---
    config['k_mode'] = args.k_mode
    config['fixed_k'] = args.fixed_k
    # Blender kaliicilik/batch ayarlari
    config['blender_persistent'] = True
    config['blender_batch_size'] = args.blender_batch_size
    # Adaptive crop ayari
    config['adaptive_crop_n'] = args.adaptive_crop
    # Checkpoint ayarlari
    config['checkpoint_interval'] = args.checkpoint_interval
    config['resume_from'] = args.resume
    # IPOP/BIPOP-CMA-ES restart sayisi
    config['ipop_restarts'] = args.restarts
    config['use_bipop'] = getattr(args, 'bipop', False)

    # ---- Atmosphere & Displacement config ----
    config['use_atmosphere'] = not args.no_atmosphere
    config['use_displacement'] = not args.no_displacement
    config['atm_color_mode'] = args.atm_color_mode
    config['dem_path'] = args.dem_path

    # ---- Fixed parameters ----
    if args.fix:
        fixed_dict = {}
        for item in args.fix:
            if '=' not in item:
                print(f"❌ Invalid --fix format: '{item}'. Use param=value")
                sys.exit(1)
            k, v = item.split('=', 1)
            try:
                fixed_dict[k.strip()] = float(v.strip())
            except ValueError:
                print(f"❌ Cannot parse --fix value for '{k}': '{v}'")
                sys.exit(1)
        set_fixed_params(fixed_dict)

    # ---- If single-channel atm color mode, fix G/B channels ----
    if config['atm_color_mode'] == 'single' and config['use_atmosphere']:
        # In single mode, G and B are derived from R; fix them at defaults
        from optimization_helper import FIXED_PARAMS
        if 'atm_color_g' not in FIXED_PARAMS:
            FIXED_PARAMS['atm_color_g'] = 0.5  # Will be overridden by proportional math
        if 'atm_color_b' not in FIXED_PARAMS:
            FIXED_PARAMS['atm_color_b'] = 0.3
        print(f"  ℹ️ Single-channel atm color mode: fixed atm_color_g, atm_color_b")

    # ---- If atmosphere disabled, fix all atm params ----
    if not config['use_atmosphere']:
        from optimization_helper import FIXED_PARAMS
        atm_defaults = {
            'atm_beta0': 3e-4, 'atm_scale_height': 11.0, 'atm_anisotropy': 0.6,
            'atm_color_r': 0.8, 'atm_color_g': 0.5, 'atm_color_b': 0.3
        }
        for k, v in atm_defaults.items():
            if k not in FIXED_PARAMS:
                FIXED_PARAMS[k] = v
        print(f"  ℹ️ Atmosphere disabled: all atm params fixed at defaults")

    print(f"  📐 Active optimization dimensions: {get_active_n_params()}")
    # ========== Scan PDS Directory ==========
    try:
        img_info_list = scan_pds_directory(
            pds_dir=Path(args.pds_dir),
            max_images=args.max_images,
            img_pattern=args.img_pattern
        )
    except Exception as e:
        print(f"\nError scanning PDS directory: {e}")
        sys.exit(1)

    # ── Auto-select RSS aggregation when >1 image ──
    # Only activates when user did not explicitly pass --aggregation flag
    # (argparse default is 'mean', so check if user left it at default)
    _user_set_aggregation = args.aggregation != 'mean'
    if len(img_info_list) > 1 and not _user_set_aggregation:
        config['aggregation'] = 'rss'
        print(f"  [INFO] {len(img_info_list)} images detected "
              f"-> aggregation auto-set to 'rss' (root-sum-of-squares)")
    elif _user_set_aggregation:
        config['aggregation'] = args.aggregation
        print(f"  [INFO] Aggregation: {config['aggregation']} (from --aggregation flag)")
    
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
    
    # ========== Reset Adaptive Crop State (YENİ) ==========
    if config.get('adaptive_crop_n', 0) > 0:
        reset_adaptive_crop_state(config['adaptive_crop_n'])
        logger.log(f"\n🔄 Adaptive Crop enabled: will lock bbox after {config['adaptive_crop_n']} evaluations")
    
    # ========== Load Checkpoint (YENİ) ==========
    resume_state = None
    if config.get('resume_from'):
        checkpoint_path = Path(config['resume_from'])
        if checkpoint_path.exists():
            resume_state = load_checkpoint(checkpoint_path)
            logger.log(f"\n📂 Resuming from checkpoint: {checkpoint_path}")
            logger.log(f"   Previous iterations: {resume_state['iteration_count']}")
            logger.log(f"   Previous best: {resume_state['best_objective']:.6f}")
            
            # Checkpoint'taki algoritma mevcut algoritmayla eşleşmeli
            if resume_state['algorithm'] != config['algorithm'].lower():
                logger.log(f"⚠️ Warning: checkpoint algorithm ({resume_state['algorithm']}) differs from current ({config['algorithm']})")
        else:
            logger.log(f"⚠️ Checkpoint file not found: {checkpoint_path}")
    
    # ========== Create Objective Function ==========
    objective_function = create_objective_function(config, img_info_list, resume_state)
    
    # ========== RENDER-ONLY MODE ==========
    if args.render_only:
        if not args.params:
            logger.log("❌ --render-only requires --params argument!")
            logger.log("   Example: --params 0.338531,0.921311,0.855890,1.0,0.319419,1.754619,0.826627,0.011097,0.634067,209.046147")
            sys.exit(1)
        
        # Parse parameters
        try:
            param_values = [float(x.strip()) for x in args.params.split(',')]
            if len(param_values) != N_PARAMS:
                logger.log(f"❌ Expected {N_PARAMS} parameters, got {len(param_values)}")
                sys.exit(1)
            params = np.array(param_values)
        except ValueError as e:
            logger.log(f"❌ Could not parse parameters: {e}")
            sys.exit(1)
        
        logger.log("\n" + "="*60)
        logger.log("RENDER-ONLY MODE")
        logger.log("="*60)
        print_params_table(params)
        
        # Evaluate with these parameters (this does the render)
        logger.log("\nRendering with given parameters...")
        score = objective_function(params)
        
        logger.log(f"\n✅ Render complete!")
        logger.log(f"   Objective score: {score:.6f}")
        logger.log(f"   Output dir: {config['temp_dir']}")
        
        # Show where files are
        render_dir = Path(config['temp_dir']) / "particle_000"
        if render_dir.exists():
            for f in render_dir.glob("*.png"):
                logger.log(f"   📷 {f}")
        
        # Save .blend file for inspection
        try:
            from corto_renderer import save_debug_scene
            blend_path = save_debug_scene(render_dir, particle_id=0)
            if blend_path:
                logger.log(f"   📁 Blend file: {blend_path}")
        except Exception as e:
            logger.log(f"   ⚠️ Could not save .blend file: {e}")
        
        sys.exit(0)
    
    # ========== Run Optimization ==========
    try:
        algorithm = config['algorithm'].lower()
        
        if algorithm == 'pso':
            logger.log("\nRunning Particle Swarm Optimization...")
            # Checkpoint ayarlarını config'e ekle
            config['pso']['checkpoint_interval'] = config.get('checkpoint_interval', 50)
            config['pso']['temp_dir'] = str(config.get('temp_dir', 'optimization_temp'))
            
            # Resume state varsa optimizer_state'i al
            pso_resume = None
            if resume_state and resume_state.get('algorithm') == 'pso':
                pso_resume = resume_state.get('optimizer_state')
            
            results = run_pso(
                objective_function=objective_function,
                config=config['pso'],
                logger=logger,
                resume_state=pso_resume
            )
        
        elif algorithm in ['genetic', 'ga']:
            logger.log("\nRunning Genetic Algorithm...")
            # Checkpoint ayarlarını config'e ekle
            config['genetic']['checkpoint_interval'] = config.get('checkpoint_interval', 50)
            config['genetic']['temp_dir'] = str(config.get('temp_dir', 'optimization_temp'))
            
            # Resume state varsa optimizer_state'i al
            genetic_resume = None
            if resume_state and resume_state.get('algorithm') == 'genetic':
                genetic_resume = resume_state.get('optimizer_state')
            
            results = run_genetic(
                objective_function=objective_function,
                config=config['genetic'],
                logger=logger,
                resume_state=genetic_resume
            )

        elif algorithm in ['bayesian', 'bo']:
            logger.log("\nRunning Bayesian Optimization...")
            # Checkpoint ayarlarını config'e ekle
            config['bayesian']['checkpoint_interval'] = config.get('checkpoint_interval', 50)
            config['bayesian']['temp_dir'] = str(config.get('temp_dir', 'optimization_temp'))
            
            # Resume state varsa optimizer_state'i al
            bayesian_resume = None
            if resume_state and resume_state.get('algorithm') == 'bayesian':
                bayesian_resume = resume_state.get('optimizer_state')
            
            results = run_bayesian(
                objective_function=objective_function,
                config=config['bayesian'],
                logger=logger,
                resume_state=bayesian_resume
            )
        
        elif algorithm == 'cmaes':
            n_restarts = config.get('ipop_restarts', 0)
            if n_restarts > 0:
                logger.log(f"\nRunning IPOP-CMA-ES Optimization ({n_restarts} restarts)...")
            else:
                logger.log("\nRunning CMA-ES Optimization...")
            config['cmaes']['checkpoint_interval'] = config.get('checkpoint_interval', 50)
            config['cmaes']['temp_dir'] = str(config.get('temp_dir', 'optimization_temp'))

            cmaes_resume = None
            if resume_state and resume_state.get('algorithm') == 'cmaes':
                cmaes_resume = resume_state.get('optimizer_state')

            if n_restarts > 0:
                if config.get('use_bipop', False):
                    # -- BIPOP-CMA-ES --
                    logger.log(f"\nRunning BIPOP-CMA-ES Optimization ({n_restarts} restarts)...")
                    results = run_bipop_cmaes(
                        objective_function=objective_function,
                        config=config['cmaes'],
                        logger=logger,
                        n_restarts=n_restarts
                    )
                else:
                    # -- IPOP-CMA-ES --
                    results = run_ipop_cmaes(
                        objective_function=objective_function,
                        config=config['cmaes'],
                        logger=logger,
                        n_restarts=n_restarts
                    )
            else:
                # ── Standard CMA-ES (original behavior) ──
                results = run_cmaes(
                    objective_function=objective_function,
                    config=config['cmaes'],
                    logger=logger,
                    resume_state=cmaes_resume
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
        best_p = results['best_params']

        # Serialize img_info_list: strip MissionConfig (not JSON-serializable)
        def _serializable_img_info(info):
            mission_cfg = info.get('mission_config')
            return {
                'filename': info['filename'],
                'utc_time': info['utc_time'],
                'solar_distance_km': float(info['solar_distance_km']),
                'pds_path': str(info['pds_path']),
                'mission_id': mission_cfg.mission_id if mission_cfg else 'UNKNOWN',
                'use_spice': bool(mission_cfg.use_spice) if mission_cfg else False,
            }

        # Serialize config: strip any non-serializable values
        def _safe_config(cfg):
            safe = {}
            for k, v in cfg.items():
                try:
                    import json as _json
                    _json.dumps(v)
                    safe[k] = v
                except (TypeError, ValueError):
                    safe[k] = str(v)
            return safe

        with open(final_results_file, 'w') as f:
            json.dump({
                'config': _safe_config(config),
                'images': [_serializable_img_info(i) for i in img_info_list],
                'results': {
                    'best_objective': float(results['best_objective']),
                    'best_params': best_p.tolist() if best_p is not None else None,
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