#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
render_atm_comparison.py
========================
Render best optimization parameters twice:
  1) WITH atmosphere  (optimized params as-is)
  2) WITHOUT atmosphere (atm_beta0=0, making the volume sphere transparent)

Saves full-resolution normalized float32 TIFFs + amplified difference.

Best params from: eval_history_20260310_011835.csv  (iteration 66, obj=0.5274)

Usage (from corto_test directory):
    python post_project_org/post_processing/optimization/render_atm_comparison.py
"""

import sys
import os
import numpy as np
from pathlib import Path

# ---- paths ----
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent  # corto_test/
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from optimization_helper import PARAMETER_NAMES

# Output
OUT_BASE = Path(r"C:\CORTO\atm_comparison_render_v2")

# ============================================================================
# BEST PARAMETERS  (iteration 66, objective = 0.5274)
# ============================================================================
BEST_PARAMS = {
    # Phobos (fixed during optimization)
    'base_gray':      0.15,
    'tex_mix':        0.30,
    'oren_rough':     0.45,
    'princ_rough':    0.60,
    'shader_mix':     0.25,
    'ior':            1.40,
    'q_eff':          1.0,
    # Threshold
    'threshold_value': 0.0105876707078343,
    # Mars
    'mars_base_gray':   0.0628641070644747,
    'mars_tex_mix':     0.3491819710064218,
    'mars_oren_rough':  0.8022097256411574,
    'mars_princ_rough': 0.7852099775364657,
    'mars_shader_mix':  0.6339145233544331,
    'mars_ior':         2.9978350010978585,
    'mars_albedo_mul':  2.3827105280498344,
    # Atmosphere (optimized)
    'atm_beta0':         0.0099901336185325,
    'atm_scale_height':  58.50965434446451,
    'atm_anisotropy':   -0.0311904103667496,
    'atm_color_r':       0.9022428739068736,
    'atm_color_g':       0.431679036829217,
    'atm_color_b':       0.5992257098630418,
}

# Build arrays
params_with_atm = np.array([BEST_PARAMS[k] for k in PARAMETER_NAMES])

# "No atmosphere" = zero out scattering so the volume sphere is transparent
params_no_atm = params_with_atm.copy()
for i, name in enumerate(PARAMETER_NAMES):
    if name == 'atm_beta0':
        params_no_atm[i] = 0.0          # zero scattering → transparent
    elif name == 'atm_scale_height':
        params_no_atm[i] = 0.0          # zero scale height
    elif name == 'atm_anisotropy':
        params_no_atm[i] = 0.0          # isotropic (neutral)
    elif name.startswith('atm_color_'):
        params_no_atm[i] = 0.0          # no color contribution


# ============================================================================
# PDS scan — find the single image used
# ============================================================================
sys.path.append(str(SCRIPT_DIR.parent.parent))  # post_project_org
from phobos_data import CompactPDSProcessor, SpiceDataProcessor

def get_img_info_list():
    pds_dir = PROJECT_ROOT / "PDS_Data"
    processor = CompactPDSProcessor(pds_dir)
    df = processor.parse_dir()
    if df.empty:
        raise RuntimeError(f"No IMG files found in {pds_dir}")
    sdp = SpiceDataProcessor()
    results = []
    for _, row in df.iterrows():
        try:
            spice_data = sdp.get_spice_data(row['UTC_MEAN_TIME'])
            results.append({
                'filename':  row['file_name'],
                'utc_time':  row['UTC_MEAN_TIME'],
                'solar_distance_km': float(spice_data['distances']['sun_to_phobos']),
                'pds_path':  row['file_path'],
            })
            print(f"  Found IMG: {row['file_name']}")
        except Exception as e:
            print(f"  ⚠️ Skipping {row['file_name']}: {e}")
    if not results:
        raise RuntimeError("No IMG files could be processed with SPICE data")
    return results


# ============================================================================
# RENDER
# ============================================================================
def render_once(img_info_list, params, output_dir, label):
    """Render with the given params to output_dir. Returns rendered file path."""
    # IMPORTANT: reset the global renderer so a fresh scene is built
    import corto_renderer as cr
    cr._GLOBAL_RENDERER = None

    from corto_renderer import render_synthetic_for_params

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"  RENDERING: {label}")
    print(f"  Output: {output_dir}")
    print(f"{'='*80}\n")

    rendered = render_synthetic_for_params(
        img_info_list=img_info_list,
        params=params,
        output_dir=output_dir,
        particle_id=0,
        persistent=False,
        batch_size=None,
        use_displacement=True,
        use_atmosphere=True,       # always True — we zero params to disable
        atm_color_mode='rgb',
        dem_path=None,
    )

    for r in rendered:
        if r and Path(r).exists():
            print(f"  ✅ {r}")
        else:
            print(f"  ❌ Render failed")

    return rendered


# ============================================================================
# NORMALIZE + SAVE
# ============================================================================
def normalize_and_save(png_path, out_tiff_path):
    """Load 16-bit PNG, normalize to [0,1] float32, save as TIFF."""
    from PIL import Image
    import tifffile

    img = np.array(Image.open(png_path), dtype=np.float32)
    lo, hi = img.min(), img.max()
    if hi > lo:
        norm = (img - lo) / (hi - lo)
    else:
        norm = np.zeros_like(img)

    tifffile.imwrite(str(out_tiff_path), norm.astype(np.float32))
    print(f"  Saved normalized TIFF: {out_tiff_path}  (min={lo:.1f}, max={hi:.1f})")
    return norm


# ============================================================================
# MAIN
# ============================================================================
def main():
    from PIL import Image

    OUT_BASE.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("ATMOSPHERE COMPARISON RENDER v2")
    print("  Method: zero atm_beta0/scale_height/anisotropy/color for 'no atmosphere'")
    print("="*80)

    img_info_list = get_img_info_list()
    print(f"Target image: {img_info_list[0]['filename']}")

    # ---- Render 1: WITH atmosphere ----
    r1 = render_once(
        img_info_list, params_with_atm,
        OUT_BASE / "with_atmosphere",
        "WITH ATMOSPHERE (optimized params)"
    )

    # ---- Render 2: WITHOUT atmosphere (zeroed params) ----
    r2 = render_once(
        img_info_list, params_no_atm,
        OUT_BASE / "without_atmosphere",
        "WITHOUT ATMOSPHERE (atm_beta0=0, all atm zeroed)"
    )

    # ---- Post-process: normalize + diff for each rendered image ----
    try:
        import tifffile
    except ImportError:
        os.system(f"{sys.executable} -m pip install tifffile")
        import tifffile

    with_dir = OUT_BASE / "with_atmosphere"
    without_dir = OUT_BASE / "without_atmosphere"

    # Find all rendered PNGs in with_atmosphere dir
    with_pngs = sorted(with_dir.glob("particle*.png"))
    if not with_pngs:
        print("❌ No rendered PNGs found in with_atmosphere!")
        return

    for png_with in with_pngs:
        stem = png_with.stem  # e.g. particle000_img00_H7982_0006_SR2
        png_without = without_dir / png_with.name

        if not png_without.exists():
            print(f"  ⚠️ No matching without-atmosphere render for {png_with.name}")
            continue

        print(f"\n--- Processing: {stem} ---")

        norm_with = normalize_and_save(
            png_with, OUT_BASE / f"{stem}_with_atm_normalized.tiff"
        )
        norm_without = normalize_and_save(
            png_without, OUT_BASE / f"{stem}_without_atm_normalized.tiff"
        )

        # Difference (absolute)
        diff = np.abs(norm_with - norm_without)
        tifffile.imwrite(str(OUT_BASE / f"{stem}_diff_normalized.tiff"), diff.astype(np.float32))

        # Amplified 8-bit difference
        if diff.max() > 0:
            diff_vis = (diff / diff.max() * 255).clip(0, 255).astype(np.uint8)
        else:
            diff_vis = np.zeros_like(diff, dtype=np.uint8)
        Image.fromarray(diff_vis).save(str(OUT_BASE / f"{stem}_diff_amplified_8bit.png"))

        # 8-bit viewable normalized images
        Image.fromarray((norm_with * 255).clip(0, 255).astype(np.uint8)).save(
            str(OUT_BASE / f"{stem}_with_atm_8bit.png")
        )
        Image.fromarray((norm_without * 255).clip(0, 255).astype(np.uint8)).save(
            str(OUT_BASE / f"{stem}_without_atm_8bit.png")
        )

        # Stats
        a_raw = np.array(Image.open(png_with), dtype=np.float32)
        b_raw = np.array(Image.open(png_without), dtype=np.float32)
        raw_diff = np.abs(a_raw - b_raw)

        print(f"  WITH ATM:     min={a_raw.min():.0f}, max={a_raw.max():.0f}, mean={a_raw.mean():.2f}")
        print(f"  WITHOUT ATM:  min={b_raw.min():.0f}, max={b_raw.max():.0f}, mean={b_raw.mean():.2f}")
        print(f"  RAW DIFF:     max={raw_diff.max():.0f} DN, mean={raw_diff.mean():.2f} DN")
        print(f"  NORM DIFF:    max={diff.max():.6f}, mean={diff.mean():.6f}")
        print(f"  Pixels diff:  {np.count_nonzero(raw_diff)} / {raw_diff.size}")

    print(f"\n{'='*80}")
    print(f"  Output dir: {OUT_BASE}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
