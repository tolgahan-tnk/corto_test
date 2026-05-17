"""Integration test using the EXISTING CORTORenderer for rendering.

Instead of reinventing the Blender scene setup, this uses the proven
CORTORenderer from the old pipeline for the render step, while using
the new PDS loader and scorer for evaluation.

Usage:
    python debug_optimize_v2.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

# -- Path setup ---------------------------------------------------------------
PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(PROJ_ROOT))
sys.path.insert(0, str(PROJ_ROOT / "post_project_org" / "post_processing"))
sys.path.insert(0, str(PROJ_ROOT / "post_project_org" / "post_processing" / "optimization"))

# -- Override config for debug run -------------------------------------------
import config

PDS_DEBUG_DIR = PROJ_ROOT / "PDS_debug"

config.PDS_IMAGES = [
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
config.DEBUG_DIR = Path("debug_opt_test_v2")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# -- Use existing optimization pipeline for rendering -----------------------
from pds_loader import PDSLoader
from scorer import Scorer

# Import the OLD renderer and helpers
from corto_renderer import get_renderer
from optimization_helper import (
    PARAMETER_BOUNDS,
    clip_params,
    params_to_dict,
)
from pds_processor import (
    load_pds_image_data,
)
from phobos_data import (
    CompactPDSProcessor,
    get_spice_data_for_time,
    HAS_SPICE,
    SpiceDataProcessor,
)
from mission_config import (
    detect_mission,
    get_solar_distance_from_label,
    extract_pose_from_label,
)


def main():
    t0 = time.time()

    # =========================================================================
    # PHASE 1: Load PDS references with our new noise pipeline
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PHASE 1: Loading PDS references (new pipeline)")
    logger.info("=" * 60)
    loader = PDSLoader()
    pds_refs = loader.load_all()
    logger.info("Loaded %d PDS references", len(pds_refs))

    # =========================================================================
    # PHASE 2: Prepare image metadata for the old renderer
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PHASE 2: Preparing image metadata")
    logger.info("=" * 60)

    # Build img_info_list compatible with old CORTORenderer
    img_info_list = []
    for meta in config.PDS_IMAGES:
        pds_path = Path(meta["file"])
        mission_cfg = detect_mission(pds_path)
        logger.info("  %s -> mission=%s", pds_path.name, mission_cfg.get("name", "?"))

        # Get geometry from PDS label
        label_data = extract_pose_from_label(pds_path, mission_cfg)
        solar_dist = get_solar_distance_from_label(pds_path)

        img_info = {
            "pds_file": str(pds_path),
            "mission": mission_cfg,
            "utc_time": label_data.get("utc_time", "2007-02-24T18:40:08Z"),
            "solar_distance_km": solar_dist,
            "label_data": label_data,
            "camera": meta["camera"],
        }
        img_info_list.append(img_info)
        logger.info("    solar_dist=%.0f km", solar_dist)

    # =========================================================================
    # PHASE 3: Initialize renderer
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PHASE 3: Building scene (old CORTORenderer)")
    logger.info("=" * 60)

    renderer = get_renderer(
        persistent=True,
        batch_size=None,
        use_displacement=False,  # Faster for debug
        use_atmosphere=True,
        atm_color_mode="rgb",
    )

    # =========================================================================
    # PHASE 4: Quick evaluation with default params
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PHASE 4: Evaluation with default parameters")
    logger.info("=" * 60)

    from config import ALL_X0, PARAMS

    scorer = Scorer(pds_refs)

    # Use default x0 params
    x = np.array(ALL_X0, dtype=float)
    params_dict = {}
    for i, p in enumerate(PARAMS):
        params_dict[p.json_key] = x[i]

    logger.info("  Testing render with %d images...", len(img_info_list))

    # Render each image using old pipeline
    render_paths = {"img": [], "mask": []}
    for idx, img_info in enumerate(img_info_list):
        logger.info("  Rendering frame %d: %s", idx, Path(img_info["pds_file"]).name)
        try:
            result = renderer.render_for_image(
                img_info=img_info,
                params_dict=params_dict,
                particle_id=0,
                q_eff=1.0,
            )
            render_paths["img"].append(result["rendered_path"])
            render_paths["mask"].append(result["mask_path"])
            logger.info("    Rendered: %s", result["rendered_path"])
        except Exception as e:
            logger.error("    Render FAILED: %s", e)
            import traceback
            traceback.print_exc()
            return

    # Score
    score = scorer.score_all(render_paths)
    logger.info("\n--- Score with default params: %.6f ---", score)

    # =========================================================================
    # PHASE 5: Report
    # =========================================================================
    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("DONE in %.1fs", elapsed)
    logger.info("=" * 60)

    # List debug outputs
    debug_dir = config.DEBUG_DIR
    if debug_dir.exists():
        for d in sorted(debug_dir.iterdir()):
            if d.is_dir():
                n = len(list(d.iterdir()))
                logger.info("  Debug: %s (%d files)", d.name, n)


if __name__ == "__main__":
    main()
