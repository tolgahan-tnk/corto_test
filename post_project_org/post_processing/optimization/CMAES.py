#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMAES.py
========
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for photometric
parameter optimization.

CMA-ES is the state-of-the-art algorithm for continuous, derivative-free
optimization in 5–50 dimensions.  It adapts a full covariance matrix to
learn correlations between parameters and converge quickly.

Reference:
    Hansen, N. (2016). "The CMA Evolution Strategy: A Tutorial."
    arXiv:1604.00772

Dependencies:
    pip install cma

Author: Optimization Pipeline
Date: 2026
"""

import sys
import numpy as np
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from optimization_helper import (
    get_bounds_arrays,
    get_active_n_params,
    OptimizationLogger,
    print_params_table,
    save_checkpoint,
    get_checkpoint_path
)

try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False


# ============================================================================
# CMA-ES ALGORITHM
# ============================================================================

class CMAESOptimizer:
    """CMA-ES optimizer for continuous optimization.

    Uses the pycma library (Hansen) with:
    - Automatic population sizing: lambda = 4 + floor(3 * ln(n))
    - Bound handling via CMA boundary transform
    - Adaptive step-size control (built-in)
    - Covariance matrix adaptation (built-in)
    - Checkpoint/resume support
    """

    def __init__(self,
                 objective_function: Callable,
                 sigma0: float = 0.3,
                 population_size: Optional[int] = None,
                 n_iterations: int = 500,
                 seed: Optional[int] = None,
                 convergence_threshold: float = 1e-11,
                 convergence_patience: int = 1000000,
                 logger: Optional[OptimizationLogger] = None,
                 checkpoint_interval: int = 50,
                 temp_dir: Optional[str] = None,
                 resume_state: Optional[dict] = None,
                 custom_x0: Optional[np.ndarray] = None):
        """
        Initialize CMA-ES optimizer.

        Args:
            objective_function: Function to minimize f(params) -> float
            sigma0: Initial step size as fraction of parameter range.
                    0.3 means ~30% of max bound range.
            population_size: Number of offspring per generation.
                    None = auto (4 + floor(3*ln(n))).  For 14D this gives 11.
            n_iterations: Maximum number of generations
            seed: Random seed
            convergence_threshold: Stop if function value change < threshold
            convergence_patience: Number of stagnant iterations before stop
            logger: Optional OptimizationLogger
            checkpoint_interval: Save checkpoint every N iterations
            temp_dir: Directory for checkpoint files
            resume_state: Optional state dict to resume from
        """
        if not HAS_CMA:
            raise ImportError(
                "❌ CMA-ES requires the 'cma' library.\n"
                "   Install with: pip install cma"
            )

        self.objective_function = objective_function
        self.sigma0 = sigma0
        self.population_size = population_size
        self.n_iterations = n_iterations
        self.seed = seed
        self.convergence_threshold = convergence_threshold
        self.convergence_patience = convergence_patience
        self.logger = logger
        self.checkpoint_interval = checkpoint_interval
        self.temp_dir = Path(temp_dir) if temp_dir else None
        self._custom_x0 = custom_x0

        # Parameter bounds
        self.lower, self.upper = get_bounds_arrays()
        self.n_dim = get_active_n_params()

        # History tracking
        self.best_history = []

        # Resume handling
        self.start_iteration = 0
        self._resume_state = resume_state
        if resume_state:
            self.start_iteration = resume_state.get('iteration_count', 0)
            self.best_history = resume_state.get('best_history', [])

    def _log(self, message: str):
        """Log message."""
        if self.logger:
            self.logger.log(message)
        else:
            print(message)

    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Run CMA-ES optimization.

        Returns:
            Tuple of (best_params, best_objective)
        """
        self._log("\n" + "="*80)
        self._log("Starting CMA-ES Optimization")
        self._log("="*80)
        self._log(f"  Dimensions       : {self.n_dim}")
        self._log(f"  Initial sigma    : {self.sigma0}")
        self._log(f"  Max iterations   : {self.n_iterations}")

        # --- Starting point: custom or center of bounds ---
        if self._custom_x0 is not None:
            x0 = np.clip(self._custom_x0, self.lower, self.upper)
            self._log(f"  Start point    : custom (provided)")
        else:
            x0 = (self.lower + self.upper) / 2.0

        # --- Compute absolute sigma from relative fraction ---
        bound_range = self.upper - self.lower
        sigma_abs = self.sigma0 * np.max(bound_range)

        # --- CMA-ES options ---
        opts = {
            'bounds': [self.lower.tolist(), self.upper.tolist()],
            'maxiter': self.n_iterations,
            'tolx': self.convergence_threshold,
            'tolfun': self.convergence_threshold,
            'verbose': -9,  # suppress CMA internal output
        }

        if self.population_size is not None:
            opts['popsize'] = self.population_size

        if self.seed is not None:
            opts['seed'] = self.seed

        # --- Create CMA-ES instance ---
        es = cma.CMAEvolutionStrategy(x0, sigma_abs, opts)

        actual_popsize = es.popsize
        self._log(f"  Population size  : {actual_popsize} (lambda)")
        self._log(f"  Sigma (absolute) : {sigma_abs:.4f}")
        self._log("")

        # --- Resume: inject previous best if available ---
        if self._resume_state:
            self._log(f"Resuming from iteration {self.start_iteration}")
            prev_best = self._resume_state.get('best_params')
            prev_obj = self._resume_state.get('best_objective', float('inf'))
            if prev_best is not None:
                self._log(f"Previous best objective: {prev_obj:.6f}")
                # Inject as mean of the distribution
                es.mean = np.array(prev_best, dtype=float)

        # --- Main optimization loop ---
        iteration = 0
        best_obj = float('inf')
        best_params = None

        while not es.stop():
            iteration += 1
            total_iteration = self.start_iteration + iteration

            # Ask for candidate solutions
            solutions = es.ask()

            # Evaluate each candidate
            fitnesses = []
            for sol in solutions:
                f = self.objective_function(np.array(sol))
                fitnesses.append(f)

            # Tell CMA-ES the results
            es.tell(solutions, fitnesses)

            # Track best
            gen_best_idx = np.argmin(fitnesses)
            gen_best_obj = fitnesses[gen_best_idx]
            if gen_best_obj < best_obj:
                best_obj = gen_best_obj
                best_params = np.array(solutions[gen_best_idx])

            self.best_history.append(best_obj)

            # Log progress
            if iteration % 10 == 0 or iteration == 1:
                sigma_cur = es.sigma
                self._log(
                    f"Generation {total_iteration}: "
                    f"Best = {best_obj:.6f}, "
                    f"Gen best = {gen_best_obj:.6f}, "
                    f"Sigma = {sigma_cur:.6f}"
                )

                if self.logger:
                    if best_params is not None:
                        self.logger.log_iteration(
                            iteration=total_iteration,
                            best_objective=best_obj,
                            best_params=best_params,
                            diversity=sigma_cur  # sigma as diversity proxy
                        )
                    else:
                        self._log(f"  ⚠️ Generation {total_iteration}: best_params not yet found, skipping log")

            # Checkpoint
            if (self.checkpoint_interval > 0 and
                    total_iteration % self.checkpoint_interval == 0):
                if self.temp_dir:
                    checkpoint_path = get_checkpoint_path(
                        self.temp_dir, total_iteration
                    )
                    optimizer_state = {
                        'mean': es.mean.tolist(),
                        'sigma': es.sigma,
                        'best_params': best_params.tolist() if best_params is not None else None,
                        'best_objective': best_obj,
                        'best_history': self.best_history,
                        'iteration_count': total_iteration,
                    }
                    save_checkpoint(
                        filepath=checkpoint_path,
                        algorithm='cmaes',
                        optimizer_state=optimizer_state,
                        best_params=best_params,
                        best_objective=best_obj,
                        iteration_count=total_iteration,
                        config={
                            'sigma0': self.sigma0,
                            'population_size': actual_popsize,
                            'n_iterations': self.n_iterations
                        }
                    )

            # Safety cap
            if iteration >= self.n_iterations:
                break

        # --- Final results ---
        # Use CMA-ES result if better than tracked best
        result = es.result
        if result.fbest < best_obj:
            best_obj = result.fbest
            best_params = np.array(result.xbest)

        self._log("\n" + "="*80)
        self._log("CMA-ES Optimization Complete")
        self._log("="*80)
        self._log(f"Best objective : {best_obj:.6f}")
        self._log(f"Total generations: {iteration}")
        self._log(f"Stop reason    : {es.stop()}")

        print_params_table(best_params)

        return best_params, best_obj


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def run_cmaes(objective_function: Callable,
              config: Optional[Dict] = None,
              logger: Optional[OptimizationLogger] = None,
              resume_state: Optional[dict] = None) -> Dict:
    """
    Run CMA-ES optimization with configuration.

    Args:
        objective_function: Function to minimize
        config: Configuration dictionary with CMA-ES parameters
        logger: Optional logger
        resume_state: Optional state to resume from

    Returns:
        Dictionary with results
    """
    # Default configuration
    default_config = {
        'sigma0': 0.3,
        'population_size': None,  # auto
        'n_iterations': 500,
        'seed': None,
        'convergence_threshold': 1e-11,
        'convergence_patience': 1000000,
        'checkpoint_interval': 50,
        'temp_dir': None
    }

    # Update with user config
    if config:
        default_config.update(config)

    # Create optimizer
    optimizer = CMAESOptimizer(
        objective_function=objective_function,
        logger=logger,
        resume_state=resume_state,
        **default_config
    )

    # Run optimization
    best_params, best_obj = optimizer.optimize()

    # Return results
    return {
        'best_params': best_params,
        'best_objective': best_obj,
        'best_history': optimizer.best_history,
        'n_iterations': len(optimizer.best_history),
        'config': default_config
    }


# ============================================================================
# IPOP-CMA-ES: Increasing Population Restart Strategy
# ============================================================================

def run_ipop_cmaes(objective_function: Callable,
                   config: Optional[Dict] = None,
                   logger: Optional[OptimizationLogger] = None,
                   n_restarts: int = 3) -> Dict:
    """
    IPOP-CMA-ES: Increasing Population CMA-ES with restarts.

    On each restart r (0-indexed):
      - Population size: lambda_r = lambda0 * 2^r
      - Initial sigma:   sigma_r  = sigma0  * 2^r  (wider search)
      - Iterations: kept constant (same budget per restart)

    The global best across all restarts is returned.

    Reference:
        Auger & Hansen (2005), "A restart CMA evolution strategy with
        increasing population size", IEEE CEC.

    Args:
        objective_function: Function to minimize
        config: CMA-ES config dict (same keys as run_cmaes)
        logger: Optional logger
        n_restarts: Number of restarts (0 = single run, identical to run_cmaes)

    Returns:
        Dict with best_params, best_objective, all_histories
    """
    if n_restarts == 0:
        # Degenerate case: single run
        return run_cmaes(objective_function, config, logger)

    # Resolve defaults
    base_config = {
        'sigma0': 0.3,
        'population_size': None,  # auto
        'n_iterations': 300,
        'seed': None,
        'convergence_threshold': 1e-11,
        'convergence_patience': 1000000,
        'checkpoint_interval': 50,
        'temp_dir': None
    }
    if config:
        base_config.update(config)

    # Resolve base lambda (population size)
    # CMA-ES auto: 4 + floor(3 * ln(n_dim)), for 14D ~= 11
    from optimization_helper import get_active_n_params
    import math
    n_dim = get_active_n_params()
    lambda0 = base_config['population_size'] or (4 + int(3 * math.log(n_dim)))
    sigma0  = base_config['sigma0']

    global_best_params = None
    global_best_obj    = float('inf')
    all_histories      = []

    for restart in range(n_restarts):
        popsize = lambda0 * (2 ** restart)
        sigma_r = sigma0  * (2.0 ** restart)

        restart_config = base_config.copy()
        restart_config['population_size'] = popsize
        restart_config['sigma0'] = sigma_r

        if logger:
            logger.log(
                f"\n{'='*60}\n"
                f"[IPOP] Restart {restart + 1}/{n_restarts}: "
                f"lambda={popsize}, sigma0={sigma_r:.4f}\n"
                f"{'='*60}"
            )
        else:
            print(f"\n[IPOP] Restart {restart + 1}/{n_restarts}: "
                  f"lambda={popsize}, sigma0={sigma_r:.4f}")

        result = run_cmaes(objective_function, restart_config, logger)

        all_histories.append(result.get('best_history', []))

        if result['best_objective'] < global_best_obj:
            global_best_obj    = result['best_objective']
            global_best_params = result['best_params']
            if logger:
                logger.log(f"  [IPOP] New global best: {global_best_obj:.6f}")

    if logger:
        logger.log(f"\n[IPOP] Completed {n_restarts} restarts. "
                   f"Global best: {global_best_obj:.6f}")

    return {
        'best_params':    global_best_params,
        'best_objective': global_best_obj,
        'best_history':   all_histories[-1] if all_histories else [],
        'all_histories':  all_histories,
        'n_restarts':     n_restarts,
        'config':         base_config
    }


# ============================================================================
# BIPOP-CMA-ES: Budget-balanced Exploration + Exploitation
# ============================================================================

def run_bipop_cmaes(objective_function: Callable,
                    config: Optional[Dict] = None,
                    logger: Optional[OptimizationLogger] = None,
                    n_restarts: int = 6) -> Dict:
    """
    BIPOP-CMA-ES: Bi-Population CMA-ES with restarts.

    Alternates between two regimes based on evaluation budget balance:
      - LARGE-POP (exploration): doubled population, larger sigma, random x0
      - SMALL-POP (exploitation): halved population, tiny sigma, x0 near best

    The regime with fewer used evaluations runs next.

    Reference:
        Hansen (2009), "Benchmarking a BI-Population CMA-ES on the
        BBOB-2009 Function Testbed", GECCO Companion.

    Args:
        objective_function: Function to minimize
        config: CMA-ES config dict (same keys as run_cmaes)
        logger: Optional logger
        n_restarts: Total number of restarts (large + small combined)

    Returns:
        Dict with best_params, best_objective, all_histories
    """
    if n_restarts == 0:
        return run_cmaes(objective_function, config, logger)

    import math

    # Resolve defaults
    base_config = {
        'sigma0': 0.3,
        'population_size': None,
        'n_iterations': 300,
        'seed': None,
        'convergence_threshold': 1e-11,
        'convergence_patience': 1000000,
        'checkpoint_interval': 50,
        'temp_dir': None
    }
    if config:
        base_config.update(config)

    from optimization_helper import get_active_n_params
    n_dim = get_active_n_params()
    lambda0 = base_config['population_size'] or (4 + int(3 * math.log(n_dim)))
    sigma0 = base_config['sigma0']
    lower, upper = get_bounds_arrays()
    bound_range = upper - lower

    rng = np.random.default_rng(base_config.get('seed'))

    global_best_params = None
    global_best_obj = float('inf')
    all_histories = []

    budget_large = 0   # total evals spent in large-pop restarts
    budget_small = 0   # total evals spent in small-pop restarts
    large_count = 0    # how many large restarts done so far

    def _log(msg):
        if logger:
            logger.log(msg)
        else:
            print(msg)

    for restart in range(n_restarts):
        # ── Regime selection: run the regime that has fewer evals ──
        if budget_large <= budget_small:
            # ---------- LARGE POPULATION (exploration) ----------
            regime = 'LARGE'
            popsize = lambda0 * (2 ** large_count)
            sigma_r = sigma0 * (2.0 ** large_count)
            # Random starting point via Latin Hypercube-like sampling
            x0 = lower + rng.uniform(0.1, 0.9, size=n_dim) * bound_range
            large_count += 1

            _log(
                f"\n{'='*60}\n"
                f"[BIPOP-LARGE] Restart {restart + 1}/{n_restarts}: "
                f"lambda={popsize}, sigma0={sigma_r:.4f}, x0=random\n"
                f"  budget_large={budget_large}, budget_small={budget_small}\n"
                f"{'='*60}"
            )
        else:
            # ---------- SMALL POPULATION (exploitation) ----------
            regime = 'SMALL'
            # Small lambda: roughly lambda0/2, with some randomness
            u1 = rng.uniform(0.0, 1.0)
            popsize = max(3, int(lambda0 * (0.5 * (2.0 ** u1))))
            # Small sigma: sigma0 * 10^(-2*U[0,1])
            u2 = rng.uniform(0.0, 1.0)
            sigma_r = sigma0 * (10.0 ** (-2.0 * u2))
            # Start near global best with small perturbation
            if global_best_params is not None:
                perturbation = rng.normal(0, 1, size=n_dim) * sigma_r * bound_range * 0.1
                x0 = np.clip(global_best_params + perturbation, lower, upper)
            else:
                x0 = lower + rng.uniform(0.1, 0.9, size=n_dim) * bound_range

            _log(
                f"\n{'='*60}\n"
                f"[BIPOP-SMALL] Restart {restart + 1}/{n_restarts}: "
                f"lambda={popsize}, sigma0={sigma_r:.6f}, x0=best+perturb\n"
                f"  budget_large={budget_large}, budget_small={budget_small}\n"
                f"  global_best_obj={global_best_obj:.6f}\n"
                f"{'='*60}"
            )

        # Build restart config
        restart_config = base_config.copy()
        restart_config['population_size'] = popsize
        restart_config['sigma0'] = sigma_r
        restart_config['custom_x0'] = x0

        # Run CMA-ES with custom x0
        optimizer = CMAESOptimizer(
            objective_function=objective_function,
            logger=logger,
            custom_x0=x0,
            **{k: v for k, v in restart_config.items() if k != 'custom_x0'}
        )
        best_params, best_obj = optimizer.optimize()

        # Track budget: popsize * generations_run
        evals_used = len(optimizer.best_history) * popsize
        if regime == 'LARGE':
            budget_large += evals_used
        else:
            budget_small += evals_used

        all_histories.append(optimizer.best_history)

        # Update global best
        if best_obj < global_best_obj:
            global_best_obj = best_obj
            global_best_params = np.array(best_params)
            _log(f"  [BIPOP] ** New global best: {global_best_obj:.6f} **")

        _log(f"  [BIPOP] Restart {restart + 1} done: "
             f"evals={evals_used}, best_this_run={best_obj:.6f}, "
             f"global_best={global_best_obj:.6f}")

    _log(
        f"\n{'='*60}\n"
        f"[BIPOP] Completed {n_restarts} restarts. "
        f"Global best: {global_best_obj:.6f}\n"
        f"  budget_large={budget_large}, budget_small={budget_small}\n"
        f"{'='*60}"
    )

    return {
        'best_params': global_best_params,
        'best_objective': global_best_obj,
        'best_history': all_histories[-1] if all_histories else [],
        'all_histories': all_histories,
        'n_restarts': n_restarts,
        'budget_large': budget_large,
        'budget_small': budget_small,
        'config': base_config
    }


if __name__ == "__main__":
    # Test CMA-ES with simple function
    print("Testing CMA-ES optimizer...")

    if not HAS_CMA:
        print("❌ cma not installed. Run: pip install cma")
        sys.exit(1)

    # Sphere function (minimum at origin)
    def sphere_function(x):
        return float(np.sum(x**2))

    # Run CMA-ES
    results = run_cmaes(
        objective_function=sphere_function,
        config={
            'sigma0': 0.5,
            'n_iterations': 100,
            'seed': 42
        }
    )

    print(f"\n✅ Test complete!")
    print(f"Best objective: {results['best_objective']:.10f}")
    print(f"Best params norm: {np.linalg.norm(results['best_params']):.10f}")
    print(f"Generations: {results['n_iterations']}")
