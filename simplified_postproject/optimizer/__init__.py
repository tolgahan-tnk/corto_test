"""Optimizer factory.

Usage:
    from optimizer import create_optimizer
    opt = create_optimizer("cmaes", n_params=24, ...)
"""
from __future__ import annotations


def create_optimizer(name: str, **kwargs):
    """Create an optimizer by name."""
    name = name.lower()
    if name == "cmaes":
        from .cmaes import CMAESOptimizer
        return CMAESOptimizer(**kwargs)
    if name == "pso":
        raise NotImplementedError("PSO not yet implemented")
    if name == "bayesian":
        raise NotImplementedError("Bayesian not yet implemented")
    if name == "de":
        raise NotImplementedError("DE not yet implemented")
    raise ValueError(f"Unknown optimizer: {name}")
