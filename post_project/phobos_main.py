# -*- coding: utf-8 -*-
"""
phobos_main.py
Phobos Photometric Template Generator - Main Execution Module

Sorumluluklar:
- Ana execution loop
- PDS ve SPICE koordinasyonu
- Render yönetimi
- Sonuç toplama ve kaydetme
- CLI interface
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import bpy
from pathlib import Path
from datetime import datetime
from PIL import Image

sys.path.append(os.getcwd())
import cortopy as corto

# Import our modules
from phobos_data import (
    CompactPDSProcessor,
    calculate_sun_strength,
    build_param_grid,
    get_camera_config,
    get_spice_data_for_time,
    write_dynamic_scene_files,
    HAS_SPICE,
    SPICE_IMPORT_ERROR,
    AU_KM
)

from phobos_scene import (
    force_nvidia_gpu,
    save_raw_16bit_png,
    build_scene_and_materials,
    set_phobos_params,
    add_asset_paths
)

if HAS_SPICE:
    from spice_data_processor import SpiceDataProcessor


# ============================================================================
# MAIN RUNNER
# ============================================================================
def run_pds_simulations(
    pds_dir,
    n_templates=27,
    max_images=None,
    output_tag="hybrid",
    sun_blender_scaler=3.90232e-1,
    param_ranges=None
):
    """
    Ana simülasyon runner:
    - PDS parsing
    - SPICE entegrasyonu (opsiyonel)
    - Parametre sweep
    - Rendering
    
    Features:
    - SPICE optional with fallback
    - All CORTO compositing branches preserved
    - 16-bit PNG handling
    - Robust Excel/CSV logging
    - Tutorial-faithful Phobos shading
    """
    
    force_nvidia_gpu()
    corto.Utils.clean_scene()

    # Tutorial inputs
    scenario_name = "S07_Mars_Phobos_Deimos"
    scene_name = "scene_mmx.json"
    geometry_name = "geometry_mmx.json"
    body_name = [
        "g_phobos_036m_spc_0000n00000_v002.obj",
        "Mars_65k.obj",
        "g_deimos_162m_spc_0000n00000_v001.obj",
    ]

    # Output dizini
    out_dir = Path("output") / scenario_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Log dosyaları
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    xlsx_path = out_dir / f"phobos_results_{output_tag}_{stamp}.xlsx"
    csv_path = out_dir / f"phobos_results_{output_tag}_{stamp}.csv"

    # 1. PDS parse
    print("\n1. Processing PDS database...")
    pds = CompactPDSProcessor(pds_dir)
    df = pds.parse_dir()
    if max_images is not None:
        df = df.head(int(max_images)).copy()

    # 2. Parameter grid
    print("\n2. Building parameter grid...")
    combos = build_param_grid(n_templates, ranges=param_ranges)
    print(f" Generated {len(combos)} parameter combinations")

    # 3. SPICE processor (REQUIRED - NO FALLBACK)
    if not HAS_SPICE:
        raise RuntimeError(
            "❌ SPICE is REQUIRED for this simulation but not available!\n"
            f"Import error: {SPICE_IMPORT_ERROR}\n"
            "Please install SPICE dependencies:\n"
            "  pip install spiceypy\n"
            "And ensure spice_data_processor.py is available."
        )
    
    print(" ✅ SPICE processor available")
    sdp = SpiceDataProcessor()

    # Configure rendering
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_depth = '16'
    bpy.context.scene.view_settings.view_transform = 'Raw'
    bpy.context.scene.render.image_settings.compression = 0

    rows = []
    
    print("\n3. Starting template generation...")
    
    # ====================== IMAGE LOOP ======================
    for idx, row in df.reset_index(drop=True).iterrows():
        corto.Utils.clean_scene()
        
        img_name = row.get('file_name', f'IMG_{idx:06d}.IMG')
        utc_mid = row.get('UTC_MEAN_TIME')
        
        # Validate UTC time
        if not isinstance(utc_mid, str) or not utc_mid:
            raise ValueError(
                f"❌ Missing or invalid UTC_MEAN_TIME for image {img_name}\n"
                f"Got: {utc_mid} (type: {type(utc_mid)})\n"
                "All images must have valid UTC timestamps for SPICE queries."
            )

        print(f"\n{'='*80}")
        print(f"Image {idx+1}/{len(df)}: {img_name}")
        print(f"UTC: {utc_mid}")
        print(f"{'='*80}")

        # Query SPICE data (REQUIRED - will raise exception if fails)
        spice_data = get_spice_data_for_time(sdp, utc_mid)
        
        # Extract solar distance
        sd_km = spice_data.get('distances', {}).get('sun_to_phobos')
        if sd_km is None:
            raise ValueError(f"Missing 'sun_to_phobos' distance in SPICE data for {utc_mid}")
        solar_distance_km = float(sd_km)
        
        # Calculate base sun energy
        base_energy = calculate_sun_strength(
            solar_distance_km, q_eff=1.0, sun_blender_scaler=sun_blender_scaler
        )
        
        # Write dynamic scene files
        scene_filename, geom_filename = write_dynamic_scene_files(
            out_dir, spice_data, get_camera_config(), base_energy, idx
        )
        
        # Create State with dynamic SPICE data
        State_img = corto.State(
            scene=scene_filename,
            geometry=geom_filename,
            body=body_name,
            scenario=scenario_name
        )
        add_asset_paths(State_img)
        
        print(f" ✅ Using SPICE positioning")
        print(f" Solar distance: {solar_distance_km:.0f} km")
        print(f" Base sun energy: {base_energy:.4f}")

        # Build scene
        ENV, cam, sun, bodies, mat_nodes, tree = build_scene_and_materials(State_img, body_name)

        # Position objects (always use SPICE data, index 0)
        try:
            ENV.PositionAll(State_img, index=0)
        except Exception as e:
            raise RuntimeError(f"❌ Failed to position objects using SPICE data: {e}") from e

        # Ensure camera is set
        cam_obj = cam.toBlender()
        if bpy.context.scene.camera is None:
            bpy.context.scene.camera = cam_obj or bpy.data.objects.get(cam.name)
        corto.Camera.select_camera(cam.name)
        
        for o in list(bpy.data.objects):
            if o.type == 'CAMERA' and o.name != cam.name:
                bpy.data.objects.remove(o, do_unlink=True)

        # ====================== TEMPLATE LOOP ======================
        for t_idx, params in enumerate(combos):
            print(f"\n Template {t_idx + 1}/{len(combos)}")

            # Apply Phobos parameters
            set_phobos_params(mat_nodes, params)

            # Calculate sun energy with q_eff
            q_eff = float(params.get('q_eff', 1.0))
            sun_energy = calculate_sun_strength(
                solar_distance_km,
                q_eff=q_eff,
                sun_blender_scaler=sun_blender_scaler
            )

            try:
                sun.set_energy(sun_energy)
            except Exception as e:
                raise RuntimeError(f"❌ Failed to set sun energy: {e}") from e

            # Frame ID (unique across all images and templates)
            n_t = len(combos)
            output_index = idx * n_t + t_idx

            # Re-position with SPICE data (always index 0 for SPICE scenes)
            try:
                ENV.PositionAll(State_img, index=0)
            except Exception as e:
                raise RuntimeError(f"❌ Failed to re-position objects: {e}") from e

            bpy.context.scene.frame_current = output_index

            # Render
            try:
                ENV.RenderOne(cam, State_img, index=output_index, depth_flag=True)
                status = 'SUCCESS'

                # Post-process with 16-bit PNG handling
                img_path = Path(State_img.path["output_path"]) / "img" / f"{output_index:06d}.png"
                if img_path.exists():
                    try:
                        rendered_16bit = np.array(Image.open(str(img_path)))
                        if len(rendered_16bit.shape) == 3:
                            rendered_16bit = rendered_16bit[:, :, 0]
                        save_raw_16bit_png(rendered_16bit, img_path)
                    except Exception as e:
                        print(f" Warning: 16-bit PNG save failed: {e}")

            except Exception as e:
                print(f" Error: Render failed: {e}")
                status = f'FAILED: {e}'

            # Record result
            rec = {
                'img_index': idx,
                'img_filename': img_name,
                'utc_mean_time': utc_mid,
                'template_index': t_idx,
                'frame_id': output_index,
                'status': status,
                'sun_energy': float(sun_energy),
                'solar_distance_km': float(solar_distance_km),
                **{k: float(v) for k, v in params.items()},
                'output_root': str(State_img.path['output_path']),
            }
            rows.append(rec)

        # Save blend file per image
        try:
            safe_name = img_name.replace('.IMG', '').replace(' ', '_')
            corto.Utils.save_blend(State_img, f'phobos_{safe_name}')
        except Exception as e:
            print(f" Warning: Blend save failed: {e}")

    # ====================== SAVE RESULTS ======================
    print(f"\n{'='*80}")
    print("4. Saving results...")
    print(f"{'='*80}")
    
    res_df = pd.DataFrame(rows)
    
    try:
        res_df.to_excel(xlsx_path, index=False)
        print(f"Excel saved: {xlsx_path}")
    except Exception as e:
        print(f"Excel save failed: {e}")

    try:
        res_df.to_csv(csv_path, index=False)
        print(f"CSV saved: {csv_path}")
    except Exception as e:
        print(f"CSV save failed: {e}")

    print(f"\nDONE!")
    print(f"Total templates generated: {len(rows)}")
    print(f"Output directory: {out_dir}")
    print(f"Compositing branches: img/, depth/, slopes/, mask_ID_*/")

    return res_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phobos Photometric Template Generator - Modular Hybrid Version"
    )
    parser.add_argument("--pds_dir", default="PDS_Data", help="PDS IMG directory")
    parser.add_argument("--n_templates", type=int, default=128, help="Templates per image")
    parser.add_argument("--max_images", type=int, default=None, help="Max images to process")
    parser.add_argument("--sun_scaler", type=float, default=3.90232e-1, help="Sun Blender scaler")
    parser.add_argument("--output_tag", default="hybrid", help="Output file tag")
    
    args = parser.parse_args()

    results = run_pds_simulations(
        pds_dir=args.pds_dir,
        n_templates=args.n_templates,
        max_images=args.max_images,
        sun_blender_scaler=args.sun_scaler,
        output_tag=args.output_tag
    )