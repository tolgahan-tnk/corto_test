#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PSO.py
======
Particle Swarm Optimization for photometric parameter optimization.

Based on Kennedy & Eberhart (1995) with modern improvements.

Author: Optimization Pipeline
Date: 2025
"""

import numpy as np
from typing import Dict, Callable, Optional, Tuple
from pathlib import Path

from optimization_helper import (
    N_PARAMS,
    get_bounds_arrays,
    get_active_n_params,
    clip_params,
    repair_params,
    OptimizationLogger,
    print_params_table,
    save_checkpoint,
    get_checkpoint_path
)


# ============================================================================
# PSO ALGORITHM
# ============================================================================

class ParticleSwarmOptimizer:
    """
    Particle Swarm Optimization for continuous optimization.
    
    Implements standard PSO with:
    - Inertia weight damping
    - Velocity clamping
    - Boundary handling
    - Convergence detection
    """
    
    def __init__(self,
                 objective_function: Callable,
                 n_particles: int = 30,
                 n_iterations: int = 100,
                 w: float = 0.7,
                 w_min: float = 0.4,
                 w_max: float = 0.9,
                 c1: float = 1.5,
                 c2: float = 1.5,
                 v_max_factor: float = 0.2,
                 boundary_method: str = 'clip',
                 convergence_threshold: float = 1e-6,
                 convergence_patience: int = 10,
                 seed: Optional[int] = None,
                 logger: Optional[OptimizationLogger] = None,
                 checkpoint_interval: int = 50,
                 temp_dir: Optional[str] = None,
                 resume_state: Optional[dict] = None):
        """
        Initialize PSO optimizer.
        
        Args:
            objective_function: Function to minimize f(params) -> float
            n_particles: Number of particles in swarm
            n_iterations: Maximum number of iterations
            w: Inertia weight (if constant), or starting weight (if damping)
            w_min: Minimum inertia weight (for damping)
            w_max: Maximum inertia weight (for damping)
            c1: Cognitive (personal best) coefficient
            c2: Social (global best) coefficient
            v_max_factor: Maximum velocity as fraction of search range
            boundary_method: How to handle boundary violations ('clip', 'reflect', 'wrap')
            convergence_threshold: Threshold for detecting convergence
            convergence_patience: Iterations without improvement before stopping
            seed: Random seed
            logger: Optional logger for progress tracking
            checkpoint_interval: Save checkpoint every N iterations
            temp_dir: Directory for checkpoint files
            resume_state: Previous optimizer state to resume from
        """
        self.objective_function = objective_function
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.checkpoint_interval = checkpoint_interval
        self.temp_dir = Path(temp_dir) if temp_dir else None
        
        # PSO parameters
        self.w = w
        self.w_min = w_min
        self.w_max = w_max
        self.c1 = c1
        self.c2 = c2
        self.v_max_factor = v_max_factor
        self.boundary_method = boundary_method
        
        # Convergence
        self.convergence_threshold = convergence_threshold
        self.convergence_patience = convergence_patience
        
        # Random seed
        if seed is not None:
            np.random.seed(seed)
        
        # Logger
        self.logger = logger
        
        # Bounds (active params only)
        self.lower, self.upper = get_bounds_arrays()
        self.n_dims = get_active_n_params()
        self.range = self.upper - self.lower
        self.v_max = v_max_factor * self.range
        
        # Resume from checkpoint or initialize fresh
        if resume_state:
            self.particles = resume_state.get('particles')
            self.velocities = resume_state.get('velocities')
            self.pbest = resume_state.get('pbest')
            self.pbest_obj = resume_state.get('pbest_obj')
            self.gbest = resume_state.get('gbest')
            self.gbest_obj = resume_state.get('gbest_obj', np.inf)
            self.gbest_history = resume_state.get('gbest_history', [])
            self.diversity_history = resume_state.get('diversity_history', [])
            self.start_iteration = resume_state.get('iteration_count', 0)
            self._log(f"  ✅ Resumed from iteration {self.start_iteration}")
        else:
            # Swarm state
            self.particles = None
            self.velocities = None
            self.pbest = None
            self.pbest_obj = None
            self.gbest = None
            self.gbest_obj = None
            
            # History
            self.gbest_history = []
            self.diversity_history = []
            self.start_iteration = 0
        
        self._log("PSO Optimizer initialized")
        self._log(f"  Particles: {n_particles}")
        self._log(f"  Iterations: {n_iterations}")
        self._log(f"  w: {w}, c1: {c1}, c2: {c2}")
        self._log(f"  Checkpoint interval: {checkpoint_interval}")
    
    def _log(self, message: str):
        """Log message."""
        if self.logger:
            self.logger.log(message)
        else:
            print(message)
    
    def initialize_swarm(self):
        """Initialize particle positions and velocities."""
        self._log("Initializing swarm...")
        
        # Random positions within bounds
        self.particles = np.random.uniform(
            self.lower, self.upper, 
            size=(self.n_particles, self.n_dims)
        )
        
        # Random velocities (small initial velocities)
        self.velocities = np.random.uniform(
            -self.v_max * 0.1, self.v_max * 0.1,
            size=(self.n_particles, self.n_dims)
        )
        
        # Initialize personal bests
        self.pbest = self.particles.copy()
        self.pbest_obj = np.full(self.n_particles, np.inf)
        
        # Initialize global best
        self.gbest = None
        self.gbest_obj = np.inf
        
        self._log(f"Swarm initialized with {self.n_particles} particles")
    
    def evaluate_swarm(self):
        """Evaluate all particles in the swarm."""
        for i in range(self.n_particles):
            # Evaluate particle
            obj_val = self.objective_function(self.particles[i])
            
            # Update personal best
            if obj_val < self.pbest_obj[i]:
                self.pbest[i] = self.particles[i].copy()
                self.pbest_obj[i] = obj_val
            
            # Update global best
            if obj_val < self.gbest_obj:
                self.gbest = self.particles[i].copy()
                self.gbest_obj = obj_val
                
                if self.logger:
                    self.logger.update_best(
                        objective=obj_val,
                        params=self.particles[i],
                        particle_id=i
                    )
    
    def update_velocities(self, iteration: int):
        """
        Update particle velocities.
        
        Args:
            iteration: Current iteration number
        """
        # Adaptive inertia weight (linearly decreasing)
        w_current = self.w_max - (self.w_max - self.w_min) * iteration / self.n_iterations
        
        # Random factors
        r1 = np.random.uniform(0, 1, size=(self.n_particles, self.n_dims))
        r2 = np.random.uniform(0, 1, size=(self.n_particles, self.n_dims))
        
        # Velocity update
        cognitive = self.c1 * r1 * (self.pbest - self.particles)
        social = self.c2 * r2 * (self.gbest - self.particles)
        
        self.velocities = w_current * self.velocities + cognitive + social
        
        # Velocity clamping
        self.velocities = np.clip(self.velocities, -self.v_max, self.v_max)
    
    def update_positions(self):
        """Update particle positions."""
        # Position update
        self.particles = self.particles + self.velocities
        
        # Boundary handling
        for i in range(self.n_particles):
            self.particles[i] = repair_params(self.particles[i], method=self.boundary_method)
    
    def compute_diversity(self) -> float:
        """
        Compute swarm diversity (average distance to centroid).
        
        Returns:
            Diversity measure
        """
        centroid = np.mean(self.particles, axis=0)
        distances = np.linalg.norm(self.particles - centroid, axis=1)
        return np.mean(distances)
    
    def check_convergence(self) -> bool:
        """
        Check if optimization has converged.
        
        Returns:
            True if converged, False otherwise
        """
        if len(self.gbest_history) < self.convergence_patience:
            return False
        
        # Check if best objective hasn't improved
        recent_best = self.gbest_history[-self.convergence_patience:]
        improvement = max(recent_best) - min(recent_best)
        
        return improvement < self.convergence_threshold
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Run PSO optimization.
        
        Returns:
            Tuple of (best_params, best_objective)
        """
        self._log("\n" + "="*80)
        self._log("Starting PSO Optimization")
        self._log("="*80)
        
        # Initialize only if not resuming
        if self.start_iteration == 0:
            self.initialize_swarm()
            
            # Initial evaluation
            self.evaluate_swarm()
            self.gbest_history.append(self.gbest_obj)
            self.diversity_history.append(self.compute_diversity())
            
            self._log(f"Initial best objective: {self.gbest_obj:.6f}")
        else:
            self._log(f"Resuming from iteration {self.start_iteration}")
            self._log(f"Current best: {self.gbest_obj:.6f}")
        
        # Main loop
        for iteration in range(1, self.n_iterations + 1):
            total_iteration = self.start_iteration + iteration
            
            # Update velocities and positions
            self.update_velocities(total_iteration)
            self.update_positions()
            
            # Evaluate new positions
            self.evaluate_swarm()
            
            # Track history
            self.gbest_history.append(self.gbest_obj)
            diversity = self.compute_diversity()
            self.diversity_history.append(diversity)
            
            # Log progress
            if iteration % 10 == 0 or iteration == 1:
                self._log(f"Iteration {total_iteration}/{self.start_iteration + self.n_iterations}: "
                         f"Best = {self.gbest_obj:.6f}, "
                         f"Diversity = {diversity:.6f}")
                
                if self.logger:
                    self.logger.log_iteration(
                        iteration=total_iteration,
                        best_objective=self.gbest_obj,
                        best_params=self.gbest,
                        diversity=diversity
                    )
            
            # Checkpoint kaydet
            if self.checkpoint_interval > 0 and total_iteration % self.checkpoint_interval == 0:
                if self.temp_dir:
                    checkpoint_path = get_checkpoint_path(self.temp_dir, total_iteration)
                    optimizer_state = {
                        'particles': self.particles,
                        'velocities': self.velocities,
                        'pbest': self.pbest,
                        'pbest_obj': self.pbest_obj,
                        'gbest': self.gbest,
                        'gbest_obj': self.gbest_obj,
                        'gbest_history': self.gbest_history,
                        'diversity_history': self.diversity_history,
                        'iteration_count': total_iteration
                    }
                    save_checkpoint(
                        filepath=checkpoint_path,
                        algorithm='pso',
                        optimizer_state=optimizer_state,
                        best_params=self.gbest,
                        best_objective=self.gbest_obj,
                        iteration_count=total_iteration,
                        config={'n_particles': self.n_particles, 'n_iterations': self.n_iterations}
                    )
            
            # Check convergence
            if self.check_convergence():
                self._log(f"\nConverged at iteration {total_iteration}")
                break
        
        # Final results
        self._log("\n" + "="*80)
        self._log("PSO Optimization Complete")
        self._log("="*80)
        self._log(f"Best objective: {self.gbest_obj:.6f}")
        self._log(f"Total iterations: {iteration}")
        
        print_params_table(self.gbest)
        
        return self.gbest, self.gbest_obj


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def run_pso(objective_function: Callable,
            config: Optional[Dict] = None,
            logger: Optional[OptimizationLogger] = None,
            resume_state: Optional[dict] = None) -> Dict:
    """
    Run PSO optimization with configuration.
    
    Args:
        objective_function: Function to minimize
        config: Configuration dictionary with PSO parameters
        logger: Optional logger
        resume_state: Optional state to resume from
        
    Returns:
        Dictionary with results
    """
    # Default configuration
    default_config = {
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
        'seed': None,
        'checkpoint_interval': 50,
        'temp_dir': None
    }
    
    # Update with user config
    if config:
        default_config.update(config)
    
    # Create optimizer
    pso = ParticleSwarmOptimizer(
        objective_function=objective_function,
        logger=logger,
        resume_state=resume_state,
        **default_config
    )
    
    # Run optimization
    best_params, best_obj = pso.optimize()
    
    # Return results
    return {
        'best_params': best_params,
        'best_objective': best_obj,
        'gbest_history': pso.gbest_history,
        'diversity_history': pso.diversity_history,
        'n_iterations': len(pso.gbest_history) - 1,
        'config': default_config
    }


if __name__ == "__main__":
    # Test PSO with simple function
    print("Testing PSO optimizer...")
    
    # Sphere function (minimum at origin)
    def sphere_function(x):
        return np.sum(x**2)
    
    # Run PSO
    results = run_pso(
        objective_function=sphere_function,
        config={
            'n_particles': 20,
            'n_iterations': 50,
            'seed': 42
        }
    )
    
    print(f"\n✅ Test complete!")
    print(f"Best objective: {results['best_objective']:.6f}")
    print(f"Best params: {results['best_params']}")