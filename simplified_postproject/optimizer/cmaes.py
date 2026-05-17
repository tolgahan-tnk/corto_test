"""CMA-ES optimizer wrapper using the `cma` package."""
from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from .base import BaseOptimizer

logger = logging.getLogger(__name__)


class CMAESOptimizer(BaseOptimizer):
    """CMA-ES optimizer via pycma."""

    def __init__(
        self,
        bounds_lo: list[float],
        bounds_hi: list[float],
        x0: list[float],
        sigma0: float = 0.05,
        popsize: int = 8,
        maxiter: int = 150,
        **kwargs,
    ):
        super().__init__()
        import cma

        # Per-coordinate initial std: sigma0 x parameter range.
        # Without this, a single scalar sigma0 applies the same absolute step
        # to all parameters regardless of their scale — small-range params
        # (dc_*, sun_scaler_osiris) get wildly oversampled and blow up renders.
        # With CMA_stds, each param is perturbed proportionally to its own range.
        stds = [
            sigma0 * (hi - lo)
            for lo, hi in zip(bounds_lo, bounds_hi)
        ]

        self.es = cma.CMAEvolutionStrategy(
            x0,
            1.0,  # global sigma0 = 1.0; per-coordinate scale lives in CMA_stds
            {
                "popsize": popsize,
                "maxiter": maxiter,
                "bounds": [bounds_lo, bounds_hi],
                "CMA_stds": stds,
                "verb_disp": 1,
                "verbose": -1,
            },
        )
        logger.info(
            "CMA-ES initialized: %d params, sigma0=%.4f (per-coord CMA_stds), "
            "pop=%d, maxiter=%d",
            len(x0), sigma0, popsize, maxiter,
        )

    def optimize(
        self,
        eval_fn: Callable,
        callback: Callable | None = None,
    ) -> tuple:
        """Run CMA-ES optimization loop.

        Each generation:
          1. Ask for `popsize` candidate solutions
          2. Evaluate each candidate
          3. Tell CMA-ES the scores
          4. Update best and call callback
        """
        gen = 0
        while not self.es.stop():
            gen += 1
            X = self.es.ask()

            scores = []
            for x in X:
                s = eval_fn(x)
                scores.append(s)

            self.es.tell(X, scores)

            # Track best
            best_idx = int(np.argmin(scores))
            if scores[best_idx] < self._best_score:
                self._best_score = scores[best_idx]
                self._best_x = list(X[best_idx])

            # Log
            logger.info(
                "Gen %d: best=%.6f, mean=%.6f, sigma=%.4f",
                gen,
                self._best_score,
                float(np.mean(scores)),
                self.es.sigma,
            )

            # History
            self.history.append(
                {
                    "gen": gen,
                    "best": self._best_score,
                    "mean": float(np.mean(scores)),
                    "sigma": float(self.es.sigma),
                }
            )

            # Callback
            if callback is not None:
                callback(gen, self._best_score, self._best_x)

        logger.info(
            "CMA-ES stopped after %d generations. Best score: %.6f",
            gen, self._best_score,
        )
        return self._best_x, self._best_score
