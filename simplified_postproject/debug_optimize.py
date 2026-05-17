"""Quick integration test: 4 PDS images, 3 CMA-ES generations.

Uses the debug PDS directory with all 4 images (3 HRSC + 1 OSIRIS).
Saves debug scoring images for visual inspection.

Usage (from project root):
    python simplified_postproject/debug_optimize.py
"""
from __future__ import annotations

# Windows encoding guard (cp1254 can't print emoji from old modules)
import sys as _sys
if _sys.stdout and hasattr(_sys.stdout, "reconfigure"):
    try:
        _sys.stdout.reconfigure(errors="replace")
        _sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass

import logging
import sys
import time
from pathlib import Path

import numpy as np

# Setup path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# -- Override config for debug run -------------------------------------------
import config

PDS_DEBUG_DIR = Path(__file__).resolve().parent.parent / "PDS_debug"

config.PDS_IMAGES = [
    {
        "file": str(PDS_DEBUG_DIR / "H7982_0003_SR2.IMG"),
        "camera": "hrsc",
        "calibrated": False,
        "exposure_ms": 16.128,
        "lines": 1017,
        "line_samples": 1008,
        "sample_type": "MSB_INTEGER",
        "sample_bits": 16,
    },
    {
        "file": str(PDS_DEBUG_DIR / "H7982_0006_SR2.IMG"),
        "camera": "hrsc",
        "calibrated": False,
        "exposure_ms": 16.128,
        "lines": 1017,
        "line_samples": 1008,
        "sample_type": "MSB_INTEGER",
        "sample_bits": 16,
    },
    {
        "file": str(PDS_DEBUG_DIR / "HB992_0005_SR2.IMG"),
        "camera": "hrsc",
        "calibrated": False,
        "exposure_ms": 20.16,
        "lines": 1018,
        "line_samples": 1007,
        "sample_type": "MSB_INTEGER",
        "sample_bits": 16,
    },
    {
        "file": str(PDS_DEBUG_DIR / "N20070224T184008217ID4CF82.IMG"),
        "camera": "osiris",
        "calibrated": True,
        "exposure_ms": 30.0,
        "lines": 1024,
        "line_samples": 1024,
        "sample_type": "PC_REAL",
        "sample_bits": 32,
    },
]

config.N_FRAMES = len(config.PDS_IMAGES)
config.FRAME_WEIGHTS = [1.0] * config.N_FRAMES
config.SAVE_DEBUG_IMAGES = True
config.DEBUG_DIR = Path("debug_opt_test")
config.RENDER_SAMPLES = 64  # Low quality for speed 
config.CHECKPOINT_EVERY = 1

# Quick test: 2 gen x 2 pop = 4 evals to verify fixes
config.ALGO_SETTINGS["cmaes"] = {
    "sigma0": 0.10,
    "popsize": 2,
    "maxiter": 1,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# -- Import modules after config override ------------------------------------
from pds_loader import PDSLoader
from scene import SceneManager
from scorer import Scorer
from optimizer import create_optimizer
from config import (
    ALL_X0, BOUNDS_LO, BOUNDS_HI, X0, OPT_INDICES,
    PARAMS, COEFF_PATH, _GROUP_MAP, LOG_FILE, CHECKPOINT_EVERY,
    K_SWEEP_ENABLED, N_FRAMES,
)
from csv_logger import EvalCSVLogger
import json


def reduced_to_full(x_reduced):
    x_full = list(ALL_X0)
    for i, oi in enumerate(OPT_INDICES):
        x_full[oi] = float(np.clip(x_reduced[i], PARAMS[oi].lo, PARAMS[oi].hi))
    return x_full


def save_coefficients(x_full):
    coeff = {}
    for grp in set(_GROUP_MAP.values()):
        coeff[grp] = {}
    for i, p in enumerate(PARAMS):
        coeff[_GROUP_MAP[p.group]][p.json_key] = x_full[i]
    COEFF_PATH.parent.mkdir(parents=True, exist_ok=True)
    with COEFF_PATH.open("w") as f:
        json.dump(coeff, f, indent=2)


def main():
    t0 = time.time()

    # 1. Load PDS references (no Blender needed)
    logger.info("=" * 60)
    logger.info("PHASE 1: Loading PDS references")
    logger.info("=" * 60)
    loader = PDSLoader()
    pds_refs = loader.load_all()
    logger.info("Loaded %d PDS references", len(pds_refs))

    # 2. Build Blender scene
    logger.info("=" * 60)
    logger.info("PHASE 2: Building Blender scene")
    logger.info("=" * 60)
    scene_mgr = SceneManager()
    scene_mgr.build()

    # 2.5 K-Sweep: find optimal noise thresholds using shadow masks
    optimal_k = None
    if K_SWEEP_ENABLED:
        logger.info("=" * 60)
        logger.info("PHASE 1.5: K-Sweep (shadow mask thresholding)")
        logger.info("=" * 60)
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

    # 3. Scorer
    scorer = Scorer(pds_refs)

    # 3.5 CSV logger
    csv_log = EvalCSVLogger("debug_opt_output", PARAMS, n_frames=N_FRAMES)

    # 4. Evaluate
    eval_count = 0

    def evaluate(x_reduced):
        nonlocal eval_count
        eval_count += 1
        x_full = reduced_to_full(x_reduced)
        logger.info("--- Eval #%d ---", eval_count)
        logger.info("  Params[0:6] (phobos): %s",
                     [f"{v:.4f}" for v in x_full[:6]])
        logger.info("  Params[20:24] (sun/dc): %s",
                     [f"{v:.6f}" for v in x_full[20:24]])

        scene_mgr.update_all(x_full)
        paths = scene_mgr.render_all()
        score = scorer.score_all(paths)

        # CSV logging
        popsize = config.ALGO_SETTINGS["cmaes"].get("popsize", 2)
        gen = (eval_count - 1) // popsize
        pid = (eval_count - 1) % popsize
        csv_log.log(eval_count, gen, pid, score, scorer.last_detail, x_full,
                    render_info=scene_mgr.last_render_info)

        logger.info("  Score: %.6f", score)
        return score

    def on_progress(gen, best_score, best_x):
        if gen % CHECKPOINT_EVERY == 0:
            save_coefficients(reduced_to_full(best_x))
            logger.info("Checkpoint gen %d: best=%.6f", gen, best_score)

    # 5. Optimizer
    logger.info("=" * 60)
    logger.info("PHASE 3: Running CMA-ES (3 gen x 4 pop = 12 evals, 4 images each)")
    logger.info("=" * 60)

    settings = config.ALGO_SETTINGS["cmaes"]
    opt = create_optimizer(
        "cmaes",
        bounds_lo=BOUNDS_LO,
        bounds_hi=BOUNDS_HI,
        x0=X0,
        **settings,
    )

    best_x, best_score = opt.optimize(evaluate, callback=on_progress)

    # 6. Final
    best_x_full = reduced_to_full(best_x)
    save_coefficients(best_x_full)
    opt.save_history(LOG_FILE)
    csv_log.close()

    logger.info("Final render with best parameters (score=%.6f) ...", best_score)
    scene_mgr.update_all(best_x_full)
    scene_mgr.render_all()
    scene_mgr.save_debug_blend("debug_opt_test", "debug_final")

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("DONE. Best score: %.6f, Evals: %d, Time: %.1fs",
                best_score, eval_count, elapsed)
    logger.info("=" * 60)

    # Print debug output locations
    debug_dir = Path("debug_opt_test")
    if debug_dir.exists():
        for d in sorted(debug_dir.iterdir()):
            if d.is_dir():
                n = len(list(d.iterdir()))
                logger.info("  Debug: %s (%d files)", d.name, n)


if __name__ == "__main__":
    main()
