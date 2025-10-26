#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genetic.py
==========
Genetic Algorithm for photometric parameter optimization.

Implements:
- Real-valued encoding
- Tournament selection
- Simulated Binary Crossover (SBX)
- Polynomial mutation
- Elitism

Author: Optimization Pipeline
Date: 2025
"""

import numpy as np
from typing import Dict, Callable, Optional, Tuple, List
from pathlib import Path

from optimization_helper import (
    N_PARAMS,
    get_bounds_arrays,
    clip_params,
    repair_params,
    OptimizationLogger,
    print_params_table
)


# ============================================================================
# GENETIC ALGORITHM
# ============================================================================

class GeneticAlgorithm:
    """
    Genetic Algorithm for continuous optimization.
    
    Uses real-valued encoding with:
    - Tournament selection
    - Simulated Binary Crossover (SBX)
    - Polynomial mutation
    - Elitism
    """
    
    def __init__(self,
                 objective_function: Callable,
                 population_size: int = 50,
                 n_generations: int = 100,
                 crossover_prob: float = 0.9,
                 mutation_prob: float = 0.1,
                 crossover_eta: float = 20.0,
                 mutation_eta: float = 20.0,
                 tournament_size: int = 3,
                 elitism_count: int = 2,
                 boundary_method: str = 'clip',
                 convergence_threshold: float = 1e-6,
                 convergence_patience: int = 10,
                 seed: Optional[int] = None,
                 logger: Optional[OptimizationLogger] = None):
        """
        Initialize Genetic Algorithm.
        
        Args:
            objective_function: Function to minimize f(params) -> float
            population_size: Number of individuals in population (should be even)
            n_generations: Maximum number of generations
            crossover_prob: Probability of crossover
            mutation_prob: Probability of mutation per gene
            crossover_eta: Distribution index for SBX crossover (larger = more exploitation)
            mutation_eta: Distribution index for polynomial mutation
            tournament_size: Number of individuals in tournament selection
            elitism_count: Number of best individuals to preserve
            boundary_method: How to handle boundary violations
            convergence_threshold: Threshold for detecting convergence
            convergence_patience: Generations without improvement before stopping
            seed: Random seed
            logger: Optional logger
        """
        self.objective_function = objective_function
        self.population_size = population_size if population_size % 2 == 0 else population_size + 1
        self.n_generations = n_generations
        
        # GA parameters
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.crossover_eta = crossover_eta
        self.mutation_eta = mutation_eta
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        self.boundary_method = boundary_method
        
        # Convergence
        self.convergence_threshold = convergence_threshold
        self.convergence_patience = convergence_patience
        
        # Random seed
        if seed is not None:
            np.random.seed(seed)
        
        # Logger
        self.logger = logger
        
        # Bounds
        self.lower, self.upper = get_bounds_arrays()
        self.range = self.upper - self.lower
        
        # Population state
        self.population = None
        self.fitness = None
        self.best_individual = None
        self.best_fitness = None
        
        # History
        self.best_history = []
        self.diversity_history = []
        
        self._log("Genetic Algorithm initialized")
        self._log(f"  Population size: {self.population_size}")
        self._log(f"  Generations: {n_generations}")
        self._log(f"  Crossover prob: {crossover_prob}, eta: {crossover_eta}")
        self._log(f"  Mutation prob: {mutation_prob}, eta: {mutation_eta}")
    
    def _log(self, message: str):
        """Log message."""
        if self.logger:
            self.logger.log(message)
        else:
            print(message)
    
    def initialize_population(self):
        """Initialize random population."""
        self._log("Initializing population...")
        
        self.population = np.random.uniform(
            self.lower, self.upper,
            size=(self.population_size, N_PARAMS)
        )
        
        self.fitness = np.full(self.population_size, np.inf)
        self.best_individual = None
        self.best_fitness = np.inf
        
        self._log(f"Population initialized with {self.population_size} individuals")
    
    def evaluate_population(self):
        """Evaluate fitness of all individuals."""
        for i in range(self.population_size):
            self.fitness[i] = self.objective_function(self.population[i])
            
            # Update best
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_individual = self.population[i].copy()
                
                if self.logger:
                    self.logger.update_best(
                        objective=self.fitness[i],
                        params=self.population[i],
                        individual_id=i
                    )
    
    def tournament_selection(self, k: int = None) -> int:
        """
        Select individual using tournament selection.
        
        Args:
            k: Tournament size (uses self.tournament_size if None)
            
        Returns:
            Index of selected individual
        """
        if k is None:
            k = self.tournament_size
        
        # Random tournament
        tournament_idx = np.random.choice(self.population_size, size=k, replace=False)
        tournament_fitness = self.fitness[tournament_idx]
        
        # Select best from tournament
        winner_idx = tournament_idx[np.argmin(tournament_fitness)]
        return winner_idx
    
    def sbx_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulated Binary Crossover (SBX).
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Tuple of (offspring1, offspring2)
        """
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        if np.random.random() > self.crossover_prob:
            return offspring1, offspring2
        
        for i in range(N_PARAMS):
            if np.random.random() < 0.5:
                if abs(parent1[i] - parent2[i]) > 1e-9:
                    # SBX crossover
                    y1 = min(parent1[i], parent2[i])
                    y2 = max(parent1[i], parent2[i])
                    
                    u = np.random.random()
                    
                    if u <= 0.5:
                        beta = (2.0 * u) ** (1.0 / (self.crossover_eta + 1.0))
                    else:
                        beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (self.crossover_eta + 1.0))
                    
                    offspring1[i] = 0.5 * ((y1 + y2) - beta * (y2 - y1))
                    offspring2[i] = 0.5 * ((y1 + y2) + beta * (y2 - y1))
        
        return offspring1, offspring2
    
    def polynomial_mutation(self, individual: np.ndarray) -> np.ndarray:
        """
        Polynomial mutation.
        
        Args:
            individual: Individual to mutate
            
        Returns:
            Mutated individual
        """
        mutated = individual.copy()
        
        for i in range(N_PARAMS):
            if np.random.random() < self.mutation_prob:
                y = mutated[i]
                yl = self.lower[i]
                yu = self.upper[i]
                
                delta1 = (y - yl) / (yu - yl)
                delta2 = (yu - y) / (yu - yl)
                
                u = np.random.random()
                
                if u < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * u + (1.0 - 2.0 * u) * xy ** (self.mutation_eta + 1.0)
                    deltaq = val ** (1.0 / (self.mutation_eta + 1.0)) - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * xy ** (self.mutation_eta + 1.0)
                    deltaq = 1.0 - val ** (1.0 / (self.mutation_eta + 1.0))
                
                mutated[i] = y + deltaq * (yu - yl)
        
        return mutated
    
    def create_offspring(self) -> np.ndarray:
        """
        Create offspring population through selection, crossover, and mutation.
        
        Returns:
            Offspring population
        """
        offspring = []
        
        # Generate offspring (in pairs)
        for _ in range(self.population_size // 2):
            # Select parents
            parent1_idx = self.tournament_selection()
            parent2_idx = self.tournament_selection()
            
            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]
            
            # Crossover
            child1, child2 = self.sbx_crossover(parent1, parent2)
            
            # Mutation
            child1 = self.polynomial_mutation(child1)
            child2 = self.polynomial_mutation(child2)
            
            # Boundary handling
            child1 = repair_params(child1, method=self.boundary_method)
            child2 = repair_params(child2, method=self.boundary_method)
            
            offspring.append(child1)
            offspring.append(child2)
        
        return np.array(offspring)
    
    def compute_diversity(self) -> float:
        """
        Compute population diversity (average distance to centroid).
        
        Returns:
            Diversity measure
        """
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        return np.mean(distances)
    
    def check_convergence(self) -> bool:
        """
        Check if optimization has converged.
        
        Returns:
            True if converged, False otherwise
        """
        if len(self.best_history) < self.convergence_patience:
            return False
        
        # Check if best fitness hasn't improved
        recent_best = self.best_history[-self.convergence_patience:]
        improvement = max(recent_best) - min(recent_best)
        
        return improvement < self.convergence_threshold
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Run genetic algorithm optimization.
        
        Returns:
            Tuple of (best_params, best_fitness)
        """
        self._log("\n" + "="*80)
        self._log("Starting Genetic Algorithm Optimization")
        self._log("="*80)
        
        # Initialize
        self.initialize_population()
        
        # Initial evaluation
        self.evaluate_population()
        self.best_history.append(self.best_fitness)
        self.diversity_history.append(self.compute_diversity())
        
        self._log(f"Initial best fitness: {self.best_fitness:.6f}")
        
        # Main loop
        for generation in range(1, self.n_generations + 1):
            # Create offspring
            offspring_pop = self.create_offspring()
            
            # Evaluate offspring
            offspring_fitness = np.array([
                self.objective_function(ind) for ind in offspring_pop
            ])
            
            # Elitism: preserve best individuals
            if self.elitism_count > 0:
                # Get indices of best individuals in current population
                elite_idx = np.argsort(self.fitness)[:self.elitism_count]
                elite_pop = self.population[elite_idx]
                elite_fitness = self.fitness[elite_idx]
                
                # Combine elites with offspring
                combined_pop = np.vstack([elite_pop, offspring_pop[:-self.elitism_count]])
                combined_fitness = np.concatenate([elite_fitness, offspring_fitness[:-self.elitism_count]])
            else:
                combined_pop = offspring_pop
                combined_fitness = offspring_fitness
            
            # Update population
            self.population = combined_pop
            self.fitness = combined_fitness
            
            # Update best
            best_idx = np.argmin(self.fitness)
            if self.fitness[best_idx] < self.best_fitness:
                self.best_fitness = self.fitness[best_idx]
                self.best_individual = self.population[best_idx].copy()
                
                if self.logger:
                    self.logger.update_best(
                        objective=self.best_fitness,
                        params=self.best_individual,
                        generation=generation
                    )
            
            # Track history
            self.best_history.append(self.best_fitness)
            diversity = self.compute_diversity()
            self.diversity_history.append(diversity)
            
            # Log progress
            if generation % 10 == 0 or generation == 1:
                self._log(f"Generation {generation}/{self.n_generations}: "
                         f"Best = {self.best_fitness:.6f}, "
                         f"Diversity = {diversity:.6f}")
                
                if self.logger:
                    self.logger.log_iteration(
                        iteration=generation,
                        best_objective=self.best_fitness,
                        best_params=self.best_individual,
                        diversity=diversity
                    )
            
            # Check convergence
            if self.check_convergence():
                self._log(f"\nConverged at generation {generation}")
                break
        
        # Final results
        self._log("\n" + "="*80)
        self._log("Genetic Algorithm Optimization Complete")
        self._log("="*80)
        self._log(f"Best fitness: {self.best_fitness:.6f}")
        self._log(f"Total generations: {generation}")
        
        print_params_table(self.best_individual)
        
        return self.best_individual, self.best_fitness


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def run_genetic(objective_function: Callable,
                config: Optional[Dict] = None,
                logger: Optional[OptimizationLogger] = None) -> Dict:
    """
    Run Genetic Algorithm optimization with configuration.
    
    Args:
        objective_function: Function to minimize
        config: Configuration dictionary with GA parameters
        logger: Optional logger
        
    Returns:
        Dictionary with results
    """
    # Default configuration
    default_config = {
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
    }
    
    # Update with user config
    if config:
        default_config.update(config)
    
    # Create optimizer
    ga = GeneticAlgorithm(
        objective_function=objective_function,
        logger=logger,
        **default_config
    )
    
    # Run optimization
    best_params, best_fitness = ga.optimize()
    
    # Return results
    return {
        'best_params': best_params,
        'best_objective': best_fitness,
        'best_history': ga.best_history,
        'diversity_history': ga.diversity_history,
        'n_generations': len(ga.best_history) - 1,
        'config': default_config
    }


if __name__ == "__main__":
    # Test GA with simple function
    print("Testing Genetic Algorithm...")
    
    # Sphere function
    def sphere_function(x):
        return np.sum(x**2)
    
    # Run GA
    results = run_genetic(
        objective_function=sphere_function,
        config={
            'population_size': 30,
            'n_generations': 50,
            'seed': 42
        }
    )
    
    print(f"\n✅ Test complete!")
    print(f"Best fitness: {results['best_objective']:.6f}")
    print(f"Best params: {results['best_params']}")