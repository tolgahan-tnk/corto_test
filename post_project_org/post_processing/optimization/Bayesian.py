#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bayesian.py
===========
Bayesian Optimization for photometric parameter optimization.

Implements:
- Gaussian Process (GP) Surrogate Model
- Expected Improvement (EI) or LCB Acquisition Function
- Uses scikit-optimize (skopt) library

Author: Optimization Pipeline
Date: 2025
"""

import numpy as np
from typing import Dict, Callable, Optional, Tuple, List
from pathlib import Path
import sys
import warnings

from optimization_helper import (
    N_PARAMS,
    get_bounds_arrays,
    get_active_names,
    get_active_n_params,
    OptimizationLogger,
    print_params_table,
    PARAMETER_NAMES,
    save_checkpoint,
    get_checkpoint_path
)

# Try to import scikit-optimize
try:
    from skopt import Optimizer
    from skopt.space import Real
    from skopt.utils import use_named_args
    from skopt.learning import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False


# ============================================================================
# BAYESIAN OPTIMIZATION
# ============================================================================

class BayesianOptimizerWrapper:
    """
    Wrapper for scikit-optimize Bayesian Optimization.
    
    Uses Gaussian Processes to model the objective function and 
    choose the next points to evaluate based on an acquisition function.
    """
    
    def __init__(self,
                 objective_function: Callable,
                 n_calls: int = 50,
                 n_initial_points: int = 10,
                 acq_func: str = "EI",   # EI daha iyi exploration sağlar
                 xi: float = 0.05,       # Artırıldı: 0.01 -> 0.05 (daha fazla exploration)
                 kappa: float = 1.96,    # For LCB
                 noise_alpha: float = 0.005,  # GP noise for robustness
                 seed: Optional[int] = None,
                 logger: Optional[OptimizationLogger] = None,
                 checkpoint_interval: int = 50,
                 temp_dir: Optional[str] = None,
                 resume_state: Optional[dict] = None):
        """
        Initialize Bayesian Optimizer.
        
        Args:
            objective_function: Function to minimize f(params) -> float
            n_calls: Total number of evaluations
            n_initial_points: Number of random initial points
            acq_func: Acquisition function ('GP', 'EI', 'LCB', 'gp_hedge')
                      'GP' usually defaults to gp_hedge (portfolio)
            xi: Parameter for EI (improvement > xi) - higher = more exploration
            kappa: Parameter for LCB
            noise_alpha: Observation noise for GP (robustness against render noise)
            seed: Random seed
            logger: Optional logger
            checkpoint_interval: Save checkpoint every N iterations
            temp_dir: Directory for checkpoint files
            resume_state: Previous optimizer state to resume from
        """
        if not HAS_SKOPT:
            raise ImportError(
                "scikit-optimize is not installed! "
                "Please install it with: pip install scikit-optimize"
            )
            
        self.objective_function = objective_function
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
        self.seed = seed
        self.logger = logger
        self.checkpoint_interval = checkpoint_interval
        self.temp_dir = Path(temp_dir) if temp_dir else None
        
        # Get bounds (active params only)
        lower, upper = get_bounds_arrays()
        active_names = get_active_names()
        self.dimensions = [
            Real(l, u, name=name) 
            for l, u, name in zip(lower, upper, active_names)
        ]
        
        # Resume from checkpoint or initialize fresh
        if resume_state and 'skopt_optimizer' in resume_state:
            self.optimizer = resume_state['skopt_optimizer']
            self.best_history = resume_state.get('best_history', [])
            self.best_params = resume_state.get('best_params')
            self.best_objective = resume_state.get('best_objective', np.inf)
            self.start_iteration = resume_state.get('iteration_count', 0)
            self._log(f"  ✅ Resumed from iteration {self.start_iteration}")
        else:
            # Initialize custom GP with Matern kernel and noise
            # Matern kernel is more robust for photometric optimization
            kernel = Matern(
                length_scale=1.0,
                length_scale_bounds=(0.01, 10.0),
                nu=2.5  # Smoothness parameter
            )
            
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=noise_alpha,  # Observation noise (render gürültüsü için)
                normalize_y=True,
                noise="gaussian",
                n_restarts_optimizer=2
            )
            
            # Initialize skopt Optimizer with custom GP
            self.optimizer = Optimizer(
                dimensions=self.dimensions,
                base_estimator=gp,
                n_initial_points=n_initial_points,
                acq_func=acq_func,
                acq_optimizer="sampling",  # sampling works with custom kernels (lbfgs requires gradient_x)
                random_state=seed,
                acq_func_kwargs={"xi": xi, "kappa": kappa}
            )
            
            # History
            self.best_history = []
            self.best_params = None
            self.best_objective = np.inf
            self.start_iteration = 0
        
        self._log("Bayesian Optimization initialized (scikit-optimize)")
        self._log(f"  Total Calls: {n_calls}")
        self._log(f"  Initial Points: {n_initial_points}")
        self._log(f"  Acquisition: {acq_func}")
        self._log(f"  Checkpoint interval: {checkpoint_interval}")
    
    def _log(self, message: str):
        """Log message."""
        if self.logger:
            self.logger.log(message)
        else:
            print(message)
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Run Bayesian Optimization loop.
        
        Returns:
            Tuple of (best_params, best_objective)
        """
        self._log("\n" + "="*80)
        self._log("Starting Bayesian Optimization")
        self._log("="*80)
        
        for i in range(self.n_calls):
            # Ask for next point to evaluate
            next_x = self.optimizer.ask()
            
            # Evaluate (convert list to numpy array for our pipeline)
            val = self.objective_function(np.array(next_x))
            
            # WARMUP döneminde (val=inf) optimizer'a söyleme, best güncelleme
            if val == float('inf') or not np.isfinite(val):
                self._log(f"  ⏭️ Warmup call {i+1}: skipping GP update (val=inf)")
                self.best_history.append(self.best_objective)
                continue
            
            # Tell the result to the optimizer
            self.optimizer.tell(next_x, val)
            
            # Update best
            if val < self.best_objective:
                self.best_objective = val
                self.best_params = np.array(next_x)
                
                if self.logger:
                    self.logger.update_best(
                        objective=val,
                        params=self.best_params,
                        iteration=i+1,
                        particle_id=i
                    )
            
            # History
            self.best_history.append(self.best_objective)
            
            # Log progress
            if (i + 1) % 5 == 0 or i == 0:
                self._log(f"Call {i+1}/{self.n_calls}: "
                         f"Current = {val:.6f}, Best = {self.best_objective:.6f}")
                
                if self.logger and self.best_params is not None:
                    self.logger.log_iteration(
                        iteration=i+1,
                        best_objective=self.best_objective,
                        best_params=self.best_params,
                        diversity=0.0 # Diversity not typically tracked in BO same way
                    )
            
            # Checkpoint kaydet
            total_iteration = self.start_iteration + i + 1
            if self.checkpoint_interval > 0 and total_iteration % self.checkpoint_interval == 0:
                if self.temp_dir:
                    checkpoint_path = get_checkpoint_path(self.temp_dir, total_iteration)
                    optimizer_state = {
                        'skopt_optimizer': self.optimizer,
                        'best_history': self.best_history,
                        'best_params': self.best_params,
                        'best_objective': self.best_objective,
                        'iteration_count': total_iteration
                    }
                    save_checkpoint(
                        filepath=checkpoint_path,
                        algorithm='bayesian',
                        optimizer_state=optimizer_state,
                        best_params=self.best_params if self.best_params is not None else np.zeros(get_active_n_params()),
                        best_objective=self.best_objective,
                        iteration_count=total_iteration,
                        config={'n_calls': self.n_calls, 'n_initial_points': self.n_initial_points}
                    )
        
        # Final results
        self._log("\n" + "="*80)
        self._log("Bayesian Optimization Complete")
        self._log("="*80)
        self._log(f"Best objective: {self.best_objective:.6f}")
        
        print_params_table(self.best_params)
        
        return self.best_params, self.best_objective


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def run_bayesian(objective_function: Callable,
                 config: Optional[Dict] = None,
                 logger: Optional[OptimizationLogger] = None,
                 resume_state: Optional[dict] = None) -> Dict:
    """
    Run Bayesian Optimization with configuration.
    
    Args:
        objective_function: Function to minimize
        config: Configuration dictionary
        logger: Optional logger
        resume_state: Optional state to resume from
        
    Returns:
        Dictionary with results
    """
    # Default configuration
    default_config = {
        'n_calls': 50,
        'n_initial_points': 10,
        'acq_func': 'gp_hedge',
        'xi': 0.01,
        'kappa': 1.96,
        'seed': None,
        'checkpoint_interval': 50,
        'temp_dir': None
    }
    
    # Update with user config
    if config:
        default_config.update(config)
    
    # Create optimizer
    opt = BayesianOptimizerWrapper(
        objective_function=objective_function,
        logger=logger,
        resume_state=resume_state,
        **default_config
    )
    
    # Run optimization
    best_params, best_obj = opt.optimize()
    
    # Return results
    return {
        'best_params': best_params,
        'best_objective': best_obj,
        'best_history': opt.best_history,
        'n_iterations': len(opt.best_history),
        'config': default_config
    }


if __name__ == "__main__":
    # Test BO
    print("Testing Bayesian Optimization...")
    if not HAS_SKOPT:
        print("❌ scikit-optimize not found. Please install: pip install scikit-optimize")
        sys.exit(1)
        
    # Sphere function
    def sphere_function(x):
        return np.sum(x**2)
    
    results = run_bayesian(
        objective_function=sphere_function,
        config={
            'n_calls': 20,
            'n_initial_points': 5,
            'seed': 42
        }
    )
    
    print(f"\n✅ Test complete!")
    print(f"Best objective: {results['best_objective']:.6f}")
