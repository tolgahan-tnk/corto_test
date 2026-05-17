"""Entry point for photometric shader optimization.

Run with:
    blender --background --python run_optimize.py
"""
from __future__ import annotations

import sys as _sys
if _sys.stdout and hasattr(_sys.stdout, "reconfigure"):
    try:
        _sys.stdout.reconfigure(errors="replace")
        _sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass

import json
import logging
import sys
import time
from pathlib import Path as _Path

import numpy as np

# Ensure our package is importable
sys.path.insert(0, str(_Path(__file__).resolve().parent))

from config import (
    _GROUP_MAP,
    ALGO_SETTINGS,
    ALGORITHM,
    ALL_X0,
    BOUNDS_HI,
    BOUNDS_LO,
    CHECKPOINT_EVERY,
    COEFF_PATH,
    K_SWEEP_ENABLED,
    LOG_FILE,
    N_FRAMES,
    OPT_INDICES,
    PARAMS,
    X0,
)
from csv_logger import EvalCSVLogger
from optimizer import create_optimizer
from pds_loader import PDSLoader
from scene import SceneManager
from scorer import Scorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("run_optimize.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# -- Helper functions --------------------------------------------------------

def reduced_to_full(x_reduced: list[float]) -> list[float]:
    """Map optimizer's reduced parameter vector to full 24-param vector.

    Clips each value to its defined bounds.
    """
    x_full = list(ALL_X0)
    for i, oi in enumerate(OPT_INDICES):
        x_full[oi] = float(np.clip(x_reduced[i], PARAMS[oi].lo, PARAMS[oi].hi))
    return x_full


def save_coefficients(x_full: list[float]) -> None:
    """Save parameter values to coefficients.json grouped by component."""
    coeff: dict[str, dict] = {}
    for grp in set(_GROUP_MAP.values()):
        coeff[grp] = {}

    for i, p in enumerate(PARAMS):
        group_name = _GROUP_MAP[p.group]
        coeff[group_name][p.json_key] = x_full[i]

    COEFF_PATH.parent.mkdir(parents=True, exist_ok=True)
    with COEFF_PATH.open("w") as f:
        json.dump(coeff, f, indent=2)
    logger.info("Coefficients saved to %s", COEFF_PATH)


# -- Main --------------------------------------------------------------------

def main():
    t0 = time.time()

    # 1. Build Blender scene
    logger.info("=== Phase 1: Building scene ===")
    scene_mgr = SceneManager()
    scene_mgr.build()

    # 2. Load PDS reference images
    logger.info("=== Phase 1: Loading PDS references ===")
    loader = PDSLoader()
    pds_refs = loader.load_all()
    logger.info("Loaded %d PDS references", len(pds_refs))

    # 2.5 K-Sweep: find optimal noise thresholds using shadow masks
    optimal_k = None
    if K_SWEEP_ENABLED:
        logger.info("=== Phase 1.5: K-Sweep (shadow mask thresholding) ===")
        # Render once at x0 to obtain shadow masks
        x0_full = list(ALL_X0)
        scene_mgr.update_all(x0_full)
        init_paths = scene_mgr.render_all()
        shadow_masks = init_paths.get("shadow", [])
        if shadow_masks:
            optimal_k = loader.find_all_optimal_k(shadow_masks)
            # Reload PDS with K-thresholds applied
            pds_refs = loader.load_all(optimal_k=optimal_k)
            logger.info("PDS references reloaded with K-thresholds")
        else:
            logger.warning("No shadow masks from initial render, skipping K-sweep")

    # 3. Create scorer
    scorer = Scorer(pds_refs)

    # 3.5 CSV logger
    csv_log = EvalCSVLogger("optimization_output", PARAMS, n_frames=N_FRAMES)

    # 4. Evaluation function
    eval_count = 0

    def evaluate(x_reduced):
        nonlocal eval_count
        eval_count += 1

        x_full = reduced_to_full(x_reduced)
        scene_mgr.update_all(x_full)
        paths = scene_mgr.render_all()
        score = scorer.score_all(paths)

        # Determine generation/particle from eval_count and popsize
        popsize = ALGO_SETTINGS[ALGORITHM].get("popsize", 8)
        gen = (eval_count - 1) // popsize
        pid = (eval_count - 1) % popsize
        csv_log.log(eval_count, gen, pid, score, scorer.last_detail, x_full,
                    render_info=scene_mgr.last_render_info)

        logger.info("Eval #%d: score=%.6f", eval_count, score)
        return score

    # 5. Checkpoint callback
    def on_progress(gen, best_score, best_x):
        if gen % CHECKPOINT_EVERY == 0:
            save_coefficients(reduced_to_full(best_x))
            logger.info(
                "Checkpoint at gen %d: best=%.6f", gen, best_score
            )

    # 6. Create optimizer
    logger.info("=== Phase 2: Starting optimization ===")
    settings = ALGO_SETTINGS[ALGORITHM]
    opt = create_optimizer(
        ALGORITHM,
        bounds_lo=BOUNDS_LO,
        bounds_hi=BOUNDS_HI,
        x0=X0,
        **settings,
    )

    # 7. Run optimization
    best_x, best_score = opt.optimize(evaluate, callback=on_progress)

    # 8. Final save
    logger.info("=== Phase 3: Finalization ===")
    best_x_full = reduced_to_full(best_x)
    save_coefficients(best_x_full)
    opt.save_history(LOG_FILE)
    csv_log.close()

    # Re-render with best params so cortopy holds the best state, then save .blend
    logger.info("Final render with best parameters (score=%.6f) ...", best_score)
    scene_mgr.update_all(best_x_full)
    scene_mgr.render_all()
    scene_mgr.save_debug_blend("optimization_output", "best_final")

    elapsed = time.time() - t0
    logger.info(
        "Optimization complete. Best score: %.6f, Evals: %d, Time: %.1fs",
        best_score, eval_count, elapsed,
    )


if __name__ == "__main__":
    main()
