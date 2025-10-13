#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimization package for photometric parameter optimization.

Modules:
- optimization_helper: Helper functions and utilities
- corto_renderer: CORTO rendering wrapper
- PSO: Particle Swarm Optimization
- Genetic: Genetic Algorithm
- optimizer_main: Main optimization script
"""

__version__ = "1.0.0"

from .optimization_helper import (
    PARAMETER_BOUNDS,
    PARAMETER_NAMES,
    N_PARAMS,
    OptimizationLogger,
    params_to_dict,
    dict_to_params,
    clip_params,
    get_bounds_arrays,
    evaluate_params_with_rendering
)

from .PSO import ParticleSwarmOptimizer, run_pso
from .Genetic import GeneticAlgorithm, run_genetic

__all__ = [
    'PARAMETER_BOUNDS',
    'PARAMETER_NAMES',
    'N_PARAMS',
    'OptimizationLogger',
    'params_to_dict',
    'dict_to_params',
    'clip_params',
    'get_bounds_arrays',
    'evaluate_params_with_rendering',
    'ParticleSwarmOptimizer',
    'run_pso',
    'GeneticAlgorithm',
    'run_genetic'
]