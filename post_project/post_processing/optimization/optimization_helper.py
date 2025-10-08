#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimization_helper.py
======================
Helper functions for photometric parameter optimization.

Author: Optimization Pipeline
Date: 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable
import json
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import evaluation function from quick_evaluator
from quick_evaluator import quick_evaluate

# Forward declaration for corto_renderer (to avoid circular import)
render_synthetic_for_params = None


# ============================================================================
# PARAMETER DEFINITIONS
# ============================================================================

PARAMETER_BOUNDS = {
    'base_gray': (0.02, 0.60),
    'tex_mix': (0.0, 1.0),
    'oren_rough': (0.0, 1.0),
    'princ_rough': (0.0, 1.0),
    'shader_mix': (0.0, 1.0),
    'ior': (1.33, 1.79),
    'q_eff': (0.7, 1.4),
}

PARAMETER_NAMES = list(PARAMETER_BOUNDS.keys())
N_PARAMS = len(PARAMETER_NAMES)


# ============================================================================
# PARAMETER UTILITIES
# ============================================================================

def params_to_dict(params_array: np.ndarray) -> Dict[str, float]:
    """Convert parameter array to dictionary."""
    return {name: float(val) for name, val in zip(PARAMETER_NAMES, params_array)}


def dict_to_params(params_dict: Dict[str, float]) -> np.ndarray:
    """Convert parameter dictionary to array."""
    return np.array([params_dict[name] for name in PARAMETER_NAMES])


def clip_params(params: np.ndarray) -> np.ndarray:
    """Clip parameters to valid bounds."""
    clipped = params.copy()
    for i, name in enumerate(PARAMETER_NAMES):
        lower, upper = PARAMETER_BOUNDS[name]
        clipped[i] = np.clip(params[i], lower, upper)
    return clipped


def get_bounds_arrays() -> Tuple[np.ndarray, np.ndarray]:
    """Get lower and upper bounds as arrays."""
    lower = np.array([PARAMETER_BOUNDS[name][0] for name in PARAMETER_NAMES])
    upper = np.array([PARAMETER_BOUNDS[name][1] for name in PARAMETER_NAMES])
    return lower, upper


def random_params(n_samples: int = 1, seed: Optional[int] = None) -> np.ndarray:
    """Generate random parameter sets within bounds."""
    if seed is not None:
        np.random.seed(seed)
    
    lower, upper = get_bounds_arrays()
    return np.random.uniform(lower, upper, size=(n_samples, N_PARAMS))


def print_params_table(params: np.ndarray):
    """Print parameters in a formatted table."""
    params_dict = params_to_dict(params)
    
    print("\n" + "="*60)
    print("Parameter Values")
    print("="*60)
    print(f"{'Parameter':<20} {'Value':>15} {'Bounds':>20}")
    print("-"*60)
    
    for name in PARAMETER_NAMES:
        val = params_dict[name]
        lower, upper = PARAMETER_BOUNDS[name]
        print(f"{name:<20} {val:>15.6f} [{lower:.2f}, {upper:.2f}]")
    
    print("="*60 + "\n")


# ============================================================================
# OBJECTIVE FUNCTION WITH RENDERING
# ============================================================================

def evaluate_params_with_rendering(
    params: np.ndarray,
    img_info_list: List[Dict],
    objective_type: str = 'ssim',
    mode: str = 'cropped',
    aggregation: str = 'mean',
    particle_id: int = 0,
    temp_dir: Optional[Path] = None,
    verbose: bool = False
) -> float:
    """
    Evaluate parameters by:
    1. Rendering synthetic images with CORTO
    2. Comparing with real IMG files
    
    Args:
        params: Parameter array [7 values]
        img_info_list: List of IMG info dicts
        objective_type: 'ssim', 'rmse', 'nmrse', 'combined'
        mode: 'cropped' or 'uncropped'
        aggregation: 'mean', 'median', 'max'
        particle_id: Particle ID
        temp_dir: Temp directory
        verbose: Verbose output
        
    Returns:
        Objective value (lower is better)
    """
    global render_synthetic_for_params
    
    # Lazy import to avoid circular dependency
    if render_synthetic_for_params is None:
        from corto_renderer import render_synthetic_for_params as render_func
        render_synthetic_for_params = render_func
    
    try:
        params_clipped = clip_params(params)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating Particle {particle_id}")
            print(f"{'='*60}")
            print_params_table(params_clipped)
        
        # Temp dir
        if temp_dir is None:
            temp_dir = Path("optimization_temp")
        
        particle_temp_dir = temp_dir / f"particle_{particle_id:03d}"
        particle_temp_dir.mkdir(parents=True, exist_ok=True)
        
        # RENDER synthetic images
        if verbose:
            print(f"\nRendering {len(img_info_list)} synthetic images...")
        
        synthetic_files = render_synthetic_for_params(
            img_info_list=img_info_list,
            params=params_clipped,
            output_dir=particle_temp_dir,
            particle_id=particle_id
        )
        
        if None in synthetic_files:
            print(f"Error: Some renders failed for particle {particle_id}")
            return np.inf
        
        # EVALUATE each pair
        objectives = []
        
        for i, (img_info, synthetic_file) in enumerate(zip(img_info_list, synthetic_files)):
            if verbose:
                print(f"\nEvaluating pair {i+1}/{len(img_info_list)}:")
                print(f"  Real: {img_info['filename']}")
                print(f"  Synthetic: {synthetic_file.name}")
            
            results = quick_evaluate(
                img_filename=img_info['filename'],
                synthetic_filename=str(synthetic_file),
                mode=mode,
                verbose=verbose
            )
            
            if mode not in results:
                objectives.append(np.inf)
                continue
            
            if objective_type == 'ssim':
                obj_val = -results[mode]['best_ssim']
            elif objective_type == 'rmse':
                obj_val = results[mode]['best_rmse']
            elif objective_type == 'nmrse':
                obj_val = results[mode]['best_rmse'] / 1.0
            elif objective_type == 'combined':
                ssim = results[mode]['best_ssim']
                rmse = results[mode]['best_rmse']
                obj_val = (1.0 - ssim) + rmse
            else:
                raise ValueError(f"Unknown objective: {objective_type}")
            
            objectives.append(obj_val)
            
            if verbose:
                print(f"  Objective: {obj_val:.6f}")
        
        # AGGREGATE
        objectives = np.array(objectives)
        
        if aggregation == 'mean':
            final_obj = float(np.mean(objectives))
        elif aggregation == 'median':
            final_obj = float(np.median(objectives))
        elif aggregation == 'max':
            final_obj = float(np.max(objectives))
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        if verbose:
            print(f"\nFinal Objective: {final_obj:.6f}\n")
        
        return final_obj
        
    except Exception as e:
        print(f"Error evaluating particle {particle_id}: {e}")
        import traceback
        traceback.print_exc()
        return np.inf


# ============================================================================
# OPTIMIZATION LOGGER
# ============================================================================

class OptimizationLogger:
    """Logger for optimization progress."""
    
    def __init__(self, output_dir: Path, experiment_name: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.iteration_history = []
        self.best_history = []
        self.config = {}
        
        self.log_file = self.output_dir / f"{experiment_name}_{self.timestamp}.log"
        self.csv_file = self.output_dir / f"{experiment_name}_{self.timestamp}_history.csv"
        self.best_file = self.output_dir / f"{experiment_name}_{self.timestamp}_best.json"
        
        self.log(f"Optimization Logger initialized: {experiment_name}")
    
    def log(self, message: str):
        """Write message to log."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        
        with open(self.log_file, 'a') as f:
            f.write(log_msg + '\n')
    
    def set_config(self, config: Dict):
        """Store configuration."""
        self.config = config
        self.log(f"Configuration set")
    
    def log_iteration(self, iteration: int, best_objective: float, 
                     best_params: np.ndarray, **kwargs):
        """Log iteration results."""
        params_dict = params_to_dict(best_params)
        
        record = {
            'iteration': iteration,
            'best_objective': best_objective,
            **params_dict,
            **kwargs
        }
        
        self.iteration_history.append(record)
        self.log(f"Iteration {iteration}: Best = {best_objective:.6f}")
    
    def update_best(self, objective: float, params: np.ndarray, **kwargs):
        """Update best result."""
        params_dict = params_to_dict(params)
        
        best_result = {
            'objective': objective,
            'parameters': params_dict,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }
        
        self.best_history.append(best_result)
        
        with open(self.best_file, 'w') as f:
            json.dump({
                'config': self.config,
                'best_result': best_result,
                'history': self.best_history
            }, f, indent=2)
        
        self.log(f"New best! Objective: {objective:.6f}")
    
    def save_history(self):
        """Save history to CSV."""
        if not self.iteration_history:
            return
        
        df = pd.DataFrame(self.iteration_history)
        df.to_csv(self.csv_file, index=False)
        self.log(f"History saved to {self.csv_file}")
    
    def plot_convergence(self, save_path: Optional[Path] = None):
        """Plot convergence curve."""
        if not self.iteration_history:
            return
        
        df = pd.DataFrame(self.iteration_history)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['iteration'], df['best_objective'], 'b-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Best Objective Value')
        ax.set_title(f'Convergence: {self.experiment_name}')
        ax.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = self.output_dir / f"{self.experiment_name}_{self.timestamp}_convergence.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log(f"Convergence plot saved")


# ============================================================================
# CONSTRAINT HANDLING
# ============================================================================

def repair_params(params: np.ndarray, method: str = 'clip') -> np.ndarray:
    """Repair parameters that violate constraints."""
    if method == 'clip':
        return clip_params(params)
    
    elif method == 'reflect':
        lower, upper = get_bounds_arrays()
        repaired = params.copy()
        
        for i in range(N_PARAMS):
            while repaired[i] < lower[i]:
                repaired[i] = 2 * lower[i] - repaired[i]
            while repaired[i] > upper[i]:
                repaired[i] = 2 * upper[i] - repaired[i]
        
        return repaired
    
    elif method == 'wrap':
        lower, upper = get_bounds_arrays()
        range_size = upper - lower
        repaired = params.copy()
        
        for i in range(N_PARAMS):
            repaired[i] = lower[i] + np.mod(repaired[i] - lower[i], range_size[i])
        
        return repaired
    
    else:
        raise ValueError(f"Unknown repair method: {method}")


if __name__ == "__main__":
    print("Optimization Helper Module")
    print("="*60)
    
    # Test
    params = random_params(1)[0]
    print("\nRandom parameters:")
    print_params_table(params)
    
    # Test logger
    logger = OptimizationLogger(Path("test_output"), "test")
    logger.log_iteration(1, 0.5, params)
    logger.save_history()
    
    print("\n✅ Tests passed!")