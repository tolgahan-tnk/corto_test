"""Debug test for PDS loader + scorer pipeline (no Blender required).

Tests:
  1. PDS parsing (HRSC int16 + OSIRIS float32)
  2. Noise pipeline (hot pixel, invalid, NaN fill)
  3. Scorer metrics on synthetic = noisy copy of real (sanity check)
  4. Debug image saving

Usage:
    python debug_pipeline.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

# -- Setup path ---------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Override config before importing modules
import config

# Point to debug PDS directory
PDS_DEBUG_DIR = Path(__file__).resolve().parent.parent / "PDS_debug"

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
config.DEBUG_DIR = Path("debug_scoring_test")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    t0 = time.time()

    # =========================================================================
    # TEST 1: PDS Loader
    # =========================================================================
    logger.info("=" * 60)
    logger.info("TEST 1: PDS Loader + Noise Pipeline")
    logger.info("=" * 60)

    from pds_loader import PDSLoader
    loader = PDSLoader()
    pds_refs = loader.load_all()

    logger.info("\n--- PDS Loading Summary ---")
    for i, (ref, meta) in enumerate(zip(pds_refs, config.PDS_IMAGES)):
        finite = ref[np.isfinite(ref)]
        nan_count = int(np.isnan(ref).sum())
        logger.info(
            "  Frame %d [%s]: shape=%s, dtype=%s, "
            "range=[%.2f, %.2f], mean=%.2f, nan=%d",
            i, meta["camera"], ref.shape, ref.dtype,
            float(np.nanmin(ref)), float(np.nanmax(ref)),
            float(np.nanmean(ref)), nan_count,
        )

    # =========================================================================
    # TEST 2: Scorer Metrics (synthetic = real + noise)
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Scorer Metrics (synthetic = noisy real)")
    logger.info("=" * 60)

    from scorer import Scorer

    # Create fake synthetic images: real + gaussian noise + shift
    # Also create fake masks (top half = phobos, bottom = mars)
    fake_paths = {"img": [], "mask": []}
    out_dir = Path("debug_scoring_test") / "fake_renders"
    out_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    for i, ref in enumerate(pds_refs):
        # Create synthetic: add noise + small brightness offset
        clean = np.nan_to_num(ref, nan=0.0).astype(np.float64)
        noise = np.random.default_rng(42 + i).normal(0, clean.std() * 0.05, clean.shape)
        synthetic = np.clip(clean + noise, 0, clean.max()).astype(np.uint16)

        # Save as PNG (uint16)
        img_path = str(out_dir / f"{i:06d}.png")
        Image.fromarray(synthetic).save(img_path)
        fake_paths["img"].append(img_path)

        # Create mask: simple circle in center = phobos (ID=1)
        H, W = ref.shape
        yy, xx = np.mgrid[:H, :W]
        cy, cx = H // 2, W // 2
        r = min(H, W) // 6
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[(yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2] = 255  # phobos

        mask_path = str(out_dir / f"{i:06d}_mask.png")
        Image.fromarray(mask).save(mask_path)
        fake_paths["mask"].append(mask_path)

    scorer = Scorer(pds_refs)
    score = scorer.score_all(fake_paths)
    logger.info("\n--- Overall Score: %.6f ---", score)

    # =========================================================================
    # TEST 3: Verify debug images were saved
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Debug Image Output")
    logger.info("=" * 60)

    debug_dir = Path("debug_scoring_test")
    if debug_dir.exists():
        for eval_dir in sorted(debug_dir.iterdir()):
            if eval_dir.is_dir() and eval_dir.name.startswith("eval_"):
                files = sorted(eval_dir.iterdir())
                logger.info("  %s: %d files", eval_dir.name, len(files))
                for f in files[:6]:
                    size_kb = f.stat().st_size / 1024
                    logger.info("    %s (%.1f KB)", f.name, size_kb)
                if len(files) > 6:
                    logger.info("    ... and %d more", len(files) - 6)

    elapsed = time.time() - t0
    logger.info("\n--- All tests completed in %.1fs ---", elapsed)


if __name__ == "__main__":
    main()
