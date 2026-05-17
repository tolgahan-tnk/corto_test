"""Abstract base class for optimizers."""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable

logger = logging.getLogger(__name__)


class BaseOptimizer(ABC):
    """Base optimizer interface."""

    def __init__(self):
        self.history: list[dict] = []
        self._best_score: float = float("inf")
        self._best_x = None

    @abstractmethod
    def optimize(
        self,
        eval_fn: Callable,
        callback: Callable | None = None,
    ) -> tuple:
        """Run the optimization loop.

        Args:
            eval_fn: Evaluation function mapping parameter vector -> scalar score.
            callback: Optional callback(generation, best_score, best_x).

        Returns:
            (best_x, best_score) tuple.
        """
        ...

    def save_history(self, path: str) -> None:
        """Save optimization history to JSON."""
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info("History saved to %s (%d entries)", path, len(self.history))
