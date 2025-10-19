# -*- coding: utf-8 -*-
"""
Phobos Photometric Template Generator - Hybrid Version
Combines tutorial fidelity + robust features + SPICE optional
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import cv2
from PIL import Image

import bpy
sys.path.append(os.getcwd())

import cortopy as corto

# Optional SPICE
# Optional SPICE (verbose + fail-fast)
SPICE_IMPORT_ERROR = None
try:
    from spice_data_processor import SpiceDataProcessor
    HAS_SPICE = True
except Exception as e:
    SpiceDataProcessor = None
    HAS_SPICE = False
    SPICE_IMPORT_ERROR = e
    print("❌ SPICE import failed. No fallback will be used.")
    import traceback; traceback.print_exc()



AU_KM = 149_597_870.7

# ============================================================================
# GPU FORCING (from alternatif kod)
# ============================================================================
def force_nvidia_gpu():
    try:
        cyc = bpy.context.preferences.addons['cycles'].preferences
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.device = 'GPU'

        prefer_order = ['OPTIX', 'CUDA']
        nvidia_keywords = ['nvidia', 'geforce', 'rtx', 'quadro', 'tesla']
        enabled_any = False
        chosen_backend = None

        def _refresh_devices():
            try:
                cyc.get_devices()
            except Exception:
                try:
                    cyc.refresh_devices()
                except Exception:
                    pass

        for backend in prefer_order:
            cyc.compute_device_type = backend
            _refresh_devices()
            devices = list(cyc.devices)

            gpu_candidates = []
            for d in devices:
                if d.type == backend:
                    name_lc = d.name.lower()
                    is_nvidia = any(k in name_lc for k in nvidia_keywords)
                    d.use = bool(is_nvidia)
                    if d.use:
                        gpu_candidates.append(d)
                else:
                    d.use = False

            if gpu_candidates:
                enabled_any = True
                chosen_backend = backend
                break

        if not enabled_any:
            for backend in prefer_order:
                cyc.compute_device_type = backend
                _refresh_devices()
                devices = list(cyc.devices)
                first_gpu = next((d for d in devices if d.type == backend), None)
                if first_gpu:
                    for d in devices:
                        d.use = (d is first_gpu)
                    enabled_any = True
                    chosen_backend = backend
                    break

        print(f"GPU backend: {chosen_backend}" if enabled_any else "No GPU enabled")

    except Exception as e:
        print(f"GPU selection failed: {e}")


# ============================================================================
# PDS PROCESSOR (from alternatif kod - simple & effective)
# ============================================================================
class CompactPDSProcessor:
    def __init__(self, pds_data_path):
        self.pds_data_path = Path(pds_data_path)
        import re
        self._key_val = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=\s*(.*)$")

    def parse_dir(self):
        records = []
        img_dir = self.pds_data_path
        print(f"Processing IMG files in: {img_dir}")
        for img_file in img_dir.rglob("*.IMG"):
            try:
                label = {}
                with open(img_file, "r", encoding="utf-8", errors="ignore") as fh:
                    for i, raw in enumerate(fh):
                        if i > 50000:
                            break
                        line = raw.strip().replace("<CR><LF>", "")
                        if line.upper().startswith("END"):
                            break
                        m = self._key_val.match(line)
                        if m:
                            key, val = m.groups()
                            label[key] = val.strip().strip('"').strip("'")
                label.update({"file_path": str(img_file), "file_name": img_file.name})
                records.append(label)
                print(f"  Processed: {img_file.name}")
            except Exception as e:
                print(f"  Error: {img_file.name}: {e}")
        
        if not records:
            raise RuntimeError("No IMG files found!")
        
        df = pd.DataFrame(records)
        
        # Time handling
        for col in ["START_TIME", "STOP_TIME"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_localize(None)
        
        if {"START_TIME", "STOP_TIME"} <= set(df.columns):
            df["DURATION_SECONDS"] = (df["STOP_TIME"] - df["START_TIME"]).dt.total_seconds()
            df["MEAN_TIME"] = df["START_TIME"] + (df["STOP_TIME"] - df["START_TIME"]) / 2
            df["UTC_MEAN_TIME"] = df["MEAN_TIME"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-2] + "Z"
        else:
            df["DURATION_SECONDS"] = pd.NA
            df["MEAN_TIME"] = pd.NaT
            df["UTC_MEAN_TIME"] = pd.NA
        
        df["STATUS"] = "UNKNOWN"
        if "START_TIME" in df and "STOP_TIME" in df:
            df["STATUS"] = (df["START_TIME"].notna() & df["STOP_TIME"].notna()).map(
                {True: "SUCCESS", False: "MISSING_TIME_DATA"}
            )
        
        return df


# ============================================================================
# SUN ENERGY CALCULATION (from alternatif kod)
# ============================================================================
def calculate_sun_strength(solar_distance_km, q_eff=1.0, sun_blender_scaler=1.0):
    """Compute Blender Sun energy from solar distance and q_eff."""
    try:
        dist_au = float(solar_distance_km) / AU_KM
        if dist_au <= 0:
            dist_au = 1.0
    except Exception:
        dist_au = 1.0
    
    W_1AU = 427.815
    irradiance = W_1AU / (dist_au ** 2)
    effective = irradiance * float(q_eff)
    return float(sun_blender_scaler) * effective


# ============================================================================
# 16-BIT PNG HANDLING (from ilk kod - önemli!)
# ============================================================================
def save_raw_16bit_png(image_data, output_path):
    """Save image as raw 16-bit PNG preserving precision"""
    try:
        if image_data.dtype == np.uint16:
            raw_uint16 = image_data
        elif image_data.dtype == np.uint8:
            raw_uint16 = (image_data.astype(np.uint16) << 8)
        else:
            if image_data.max() <= 1.0:
                raw_uint16 = (image_data * 65535).astype(np.uint16)
            else:
                raw_uint16 = np.clip(image_data, 0, 65535).astype(np.uint16)
        
        Image.fromarray(raw_uint16, mode='I;16').save(str(output_path))
        return True
    except Exception as e:
        print(f"   16-bit PNG save failed: {e}")
        return False


# ============================================================================
# PHOBOS ADVANCED SHADING (from alternatif kod - tutorial sadık)
# ============================================================================
# ============================================================================
# PHOBOS ADVANCED SHADING (CORRECTED)
# ============================================================================
def build_scene_and_materials(State, body_names):
    """Build scene with advanced Phobos shading + compositing branches
    
    Args:
        State: CORTO State object
        body_names: List of body filenames [phobos.obj, mars.obj, deimos.obj]
    """
    
    # Setup bodies
    cam = corto.Camera('WFOV_Camera', State.properties_cam)
    sun = corto.Sun('Sun', State.properties_sun)

    name_1, _ = os.path.splitext(body_names[0])
    name_2, _ = os.path.splitext(body_names[1])
    name_3, _ = os.path.splitext(body_names[2])

    body_1 = corto.Body(name_1, State.properties_body_1)
    body_2 = corto.Body(name_2, State.properties_body_2)
    body_3 = corto.Body(name_3, State.properties_body_3)

    rendering_engine = corto.Rendering(State.properties_rendering)
    ENV = corto.Environment(cam, [body_1, body_2, body_3], sun, rendering_engine)

    # Advanced Phobos material
    phobos_material = corto.Shading.create_new_material('Phobos_Advanced_Hybrid')

    # Shader models
    oren_node = corto.Shading.diffuse_BSDF(phobos_material, location=(200, 200))
    principled_node = corto.Shading.principled_BSDF(phobos_material, location=(200, -200))

    # Grayscale sources
    albedo_texture_node = corto.Shading.texture_node(
        phobos_material, State.path["albedo_path_1"], location=(-600, 100)
    )
    try:
        bpy.data.images[os.path.basename(State.path["albedo_path_1"])].colorspace_settings.name = 'Non-Color'
    except Exception:
        pass

    uv_map_node = corto.Shading.uv_map(phobos_material, location=(-800, 100))
    rgb_to_bw_node = corto.Shading.create_node("ShaderNodeRGBToBW", phobos_material, location=(-300, 100))
    value_node = corto.Shading.create_node("ShaderNodeValue", phobos_material, location=(-300, -100))
    mix_value_node = corto.Shading.create_node("ShaderNodeMix", phobos_material, location=(0, 0))
    mix_value_node.data_type = 'FLOAT'

    # Final mix
    mix_shader_node = corto.Shading.mix_node(phobos_material, location=(500, 0))
    material_output_node = corto.Shading.material_output(phobos_material, location=(800, 0))

    # Default parameters
    value_node.outputs["Value"].default_value = 0.15
    mix_value_node.inputs["Factor"].default_value = 0.8
    oren_node.inputs["Roughness"].default_value = 0.9
    principled_node.inputs["Roughness"].default_value = 0.85

    # Başlangıç IOR değeri (dielectric varsayılanı)
    if 'IOR' in principled_node.inputs:
        principled_node.inputs['IOR'].default_value = 1.50

    # Fiziksele müdahale etmeme: 0.50
    for spec_name in ("Specular IOR Level", "Specular"):
        if spec_name in principled_node.inputs:
            principled_node.inputs[spec_name].default_value = 0.50
            break

    # (Opsiyonel güvenlik: malzeme türü gereği)
    for slot in ('Metallic', 'Clearcoat', 'Sheen', 'Coat Weight'):
        if slot in principled_node.inputs:
            principled_node.inputs[slot].default_value = 0.0

    mix_shader_node.inputs[0].default_value = 0.1

    # Link nodes
    corto.Shading.link_nodes(phobos_material, uv_map_node.outputs["UV"], albedo_texture_node.inputs["Vector"])
    corto.Shading.link_nodes(phobos_material, albedo_texture_node.outputs["Color"], rgb_to_bw_node.inputs["Color"])
    corto.Shading.link_nodes(phobos_material, rgb_to_bw_node.outputs["Val"], mix_value_node.inputs["A"])
    corto.Shading.link_nodes(phobos_material, value_node.outputs["Value"], mix_value_node.inputs["B"])

    final_gray = mix_value_node.outputs["Result"]
    corto.Shading.link_nodes(phobos_material, final_gray, oren_node.inputs["Color"])
    corto.Shading.link_nodes(phobos_material, final_gray, principled_node.inputs["Base Color"])
    corto.Shading.link_nodes(phobos_material, oren_node.outputs["BSDF"], mix_shader_node.inputs[1])
    corto.Shading.link_nodes(phobos_material, principled_node.outputs["BSDF"], mix_shader_node.inputs[2])
    corto.Shading.link_nodes(phobos_material, mix_shader_node.outputs["Shader"], material_output_node.inputs["Surface"])

    # Assign to Phobos
    corto.Shading.load_uv_data(body_1, State, 1)
    corto.Shading.assign_material_to_object(phobos_material, body_1)

    # Standard for Mars/Deimos
    material_2 = corto.Shading.create_new_material('Mars_Standard')
    material_3 = corto.Shading.create_new_material('Deimos_Standard')
    corto.Shading.create_branch_albedo_mix(material_2, State, 2)
    corto.Shading.create_branch_albedo_mix(material_3, State, 3)
    corto.Shading.load_uv_data(body_2, State, 2)
    corto.Shading.assign_material_to_object(material_2, body_2)
    corto.Shading.load_uv_data(body_3, State, 3)
    corto.Shading.assign_material_to_object(material_3, body_3)

    # COMPOSITING - ALL BRANCHES PRESERVED
    tree = corto.Compositing.create_compositing()
    render_node = corto.Compositing.rendering_node(tree, (0, 0))
    corto.Compositing.create_img_denoise_branch(tree, render_node)
    corto.Compositing.create_depth_branch(tree, render_node)
    corto.Compositing.create_slopes_branch(tree, render_node, State)
    corto.Compositing.create_maskID_branch(tree, render_node, State)
    
    try:
        num = "######"  # 6 hane; frame 6 haneden büyükse yine de doğru yazılır
        for node in tree.nodes:
            if getattr(node, "bl_idname", "") == "CompositorNodeOutputFile":
                for slot in node.file_slots:
                    slot.path = num  # ör. img/000034.png, slopes/000034.png...
    except Exception as e:
        print(f"   Warning: compositor numbering not set: {e}")

    # Scales
    body_1.set_scale(np.array([1, 1, 1]))
    body_2.set_scale(np.array([1e3, 1e3, 1e3]))
    body_3.set_scale(np.array([1, 1, 1]))

    # Pack refs
    mat_nodes = {
        'value_node': value_node,
        'mix_value_node': mix_value_node,
        'oren_node': oren_node,
        'principled_node': principled_node,
        'mix_shader_node': mix_shader_node,
    }

    return ENV, cam, sun, (body_1, body_2, body_3), mat_nodes, tree


# ============================================================================
# PARAMETER HANDLING (from alternatif kod)
# ============================================================================
def set_phobos_params(mat_nodes, params):
    """Apply parameters to Phobos material nodes (science mode).
    - IOR taranır
    - Specular IOR Level fiziksel olsun diye 0.5'te sabitlenir
    """
    if 'base_gray' in params:
        mat_nodes['value_node'].outputs['Value'].default_value = float(params['base_gray'])
    if 'tex_mix' in params:
        mat_nodes['mix_value_node'].inputs['Factor'].default_value = float(params['tex_mix'])
    if 'oren_rough' in params:
        mat_nodes['oren_node'].inputs['Roughness'].default_value = float(params['oren_rough'])
    if 'princ_rough' in params:
        mat_nodes['principled_node'].inputs['Roughness'].default_value = float(params['princ_rough'])

    # ——— NEW: IOR'u tara
    if 'ior' in params and 'IOR' in mat_nodes['principled_node'].inputs:
        mat_nodes['principled_node'].inputs['IOR'].default_value = float(params['ior'])

    # ——— CHG: Specular IOR Level'i daima 0.5'e kilitle (no adjustment)
    for spec_name in ('Specular IOR Level', 'Specular'):
        if spec_name in mat_nodes['principled_node'].inputs:
            mat_nodes['principled_node'].inputs[spec_name].default_value = 0.5
            break

    if 'shader_mix' in params:
        mat_nodes['mix_shader_node'].inputs[0].default_value = float(params['shader_mix'])

    # (Opsiyonel, bilimsel uygunluk için sabitleme)
    for slot in ('Metallic', 'Clearcoat', 'Sheen', 'Coat Weight'):
        if slot in mat_nodes['principled_node'].inputs:
            mat_nodes['principled_node'].inputs[slot].default_value = 0.0



def build_param_grid(n_templates, ranges=None):
    """Create parameter grid including q_eff (and IOR instead of specular)."""
    default_ranges = {
        'base_gray':   (0.02, 0.60),
        'tex_mix':     (0.0, 1.0),
        'oren_rough':  (0.0, 1.0),
        'princ_rough': (0.0, 1.0),
        'shader_mix':  (0.0, 1.0),
        # NEW: IOR taraması (dielectric Fresnel'den türetilmiş)
        'ior':         (1.33, 1.79),
        # Mevcut güneş ölçeklemesi korunuyorsa:
        'q_eff':       (0.7, 1.4),
    }
    if ranges:
        default_ranges.update(ranges)

    keys = list(default_ranges.keys())
    P = len(keys)
    steps = max(1, int(round(n_templates ** (1.0 / P))))

    grids = {}
    for k in keys:
        if steps == 1:
            grids[k] = [(default_ranges[k][0] + default_ranges[k][1]) / 2]
        else:
            grids[k] = np.linspace(default_ranges[k][0], default_ranges[k][1], steps).tolist()

    combos = []
    import itertools
    for vals in itertools.product(*(grids[k] for k in keys)):
        combos.append(dict(zip(keys, vals)))
    return combos



# ============================================================================
# SPICE INTEGRATION (OPTIONAL - from alternatif kod)
# ============================================================================
def _get_camera_cfg_with_fallback():
    """Get camera config in CORTO-compatible format"""
    if HAS_SPICE:
        try:
            sdp_tmp = SpiceDataProcessor()
            spice_cfg = sdp_tmp.get_hrsc_camera_config('SRC')
            # Extract only CORTO-compatible keys
            return {
                'fov': float(spice_cfg.get('fov', 0.54)),
                'res_x': int(spice_cfg.get('res_x', 1024)),
                'res_y': int(spice_cfg.get('res_y', 1024)),
                'film_exposure': float(spice_cfg.get('film_exposure', 1.0)),
                'sensor': str(spice_cfg.get('sensor', 'BW')),
                'clip_start': float(spice_cfg.get('clip_start', 0.1)),
                'clip_end': float(spice_cfg.get('clip_end', 1.0e8)),
                'bit_encoding': str(spice_cfg.get('bit_encoding', '16')),
                'viewtransform': str(spice_cfg.get('viewtransform', 'Standard')),
                'K': spice_cfg.get('K', [[1222.0, 0, 512.0], [0, 1222.0, 512.0], [0, 0, 1]]),
            }
        except Exception as e:
            print(f"Warning: SPICE camera config failed: {e}")
    
    # Fallback
    return {
        'fov': 0.54,
        'res_x': 1024,
        'res_y': 1024,
        'film_exposure': 1.0,
        'sensor': 'BW',
        'clip_start': 0.1,
        'clip_end': 1.0e8,
        'bit_encoding': '16',
        'viewtransform': 'Standard',
        'K': [[1222.0, 0, 512.0], [0, 1222.0, 512.0], [0, 0, 1]],
    }

def convert_to_python(obj):
    """Recursively convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_python(item) for item in obj)
    else:
        return obj
def _add_asset_paths(st):
    """Input/UV/albedo varlık yollarını tek noktadan ekler."""
    base = st.path["input_path"]
    st.add_path('albedo_path_1', os.path.join(base, 'body', 'albedo', 'Phobos grayscale.jpg'))
    st.add_path('uv_data_path_1', os.path.join(base, 'body', 'uv data', 'g_phobos_036m_spc_0000n00000_v002.json'))
    st.add_path('albedo_path_2', os.path.join(base, 'body', 'albedo', 'mars_1k_color.jpg'))
    st.add_path('uv_data_path_2', os.path.join(base, 'body', 'uv data', 'Mars_65k.json'))
    st.add_path('albedo_path_3', os.path.join(base, 'body', 'albedo', 'Deimos grayscale.jpg'))
    st.add_path('uv_data_path_3', os.path.join(base, 'body', 'uv data', 'g_deimos_162m_spc_0000n00000_v001.json'))

def _write_dynamic_scene_files(base_output_dir, spice_data, camera_cfg, sun_energy, idx):
    """
    SPICE verisinden senaryoya uygun dinamik geometry/scene JSON'larını yazar.
    - Dosyalar CORTO'nun senaryo yapısına yazılır: input/S07_Mars_Phobos_Deimos/{scene,geometry}
    - camera_settings.K string olarak "np.array([...], dtype=float)" formatında yazılır
      (cortopy/_State.import_scene() 'eval' ettiği için şart).
    - Geriye SADECE dosya adlarını (filename) döndürür.
    """
    from pathlib import Path
    import json
    import numpy as np

    # Senaryo klasörleri
    scenario_dir = Path("input/S07_Mars_Phobos_Deimos")
    scene_dir = scenario_dir / "scene"
    geom_dir  = scenario_dir / "geometry"
    scene_dir.mkdir(parents=True, exist_ok=True)
    geom_dir.mkdir(parents=True, exist_ok=True)

    # --- GEOMETRY ------------------------------------------------------------
    geometry_data = {
        "sun": {
            "position":   [spice_data["sun"]["position"]],
            "orientation":[spice_data["sun"]["quaternion"]],
        },
        "camera": {
            "position":   [spice_data["hrsc"]["position"]],
            "orientation":[spice_data["hrsc"]["quaternion"]],
        },
        "body_1": {
            "position":   [spice_data["phobos"]["position"]],
            "orientation":[spice_data["phobos"]["quaternion"]],
        },
        "body_2": {
            "position":   [spice_data["mars"]["position"]],
            "orientation":[spice_data["mars"]["quaternion"]],
        },
        "body_3": {
            "position":   [spice_data["deimos"]["position"]],
            "orientation":[spice_data["deimos"]["quaternion"]],
        },
    }

    # --- CAMERA SETTINGS (tipler garanti, K string) -------------------------
    # K liste/np array gelebilir → listeye çevirip stringe göm
    K_list = camera_cfg.get('K', [[1222.0, 0.0, 512.0],
                                  [0.0, 1222.0, 512.0],
                                  [0.0, 0.0, 1.0]])
    K_list = np.array(K_list, dtype=float).tolist()
    K_str  = "np.array(" + json.dumps(K_list) + ", dtype=float)"

    cam_settings = {
        'fov':           float(camera_cfg.get('fov', 0.54)),
        'res_x':         int(camera_cfg.get('res_x', 1024)),
        'res_y':         int(camera_cfg.get('res_y', 1024)),
        'film_exposure': float(camera_cfg.get('film_exposure', 1.0)),
        'sensor':        str(camera_cfg.get('sensor', 'BW')),
        'clip_start':    float(camera_cfg.get('clip_start', 0.1)),
        'clip_end':      float(camera_cfg.get('clip_end', 1.0e8)),
        'bit_encoding':  str(camera_cfg.get('bit_encoding', '16')),
        'viewtransform': str(camera_cfg.get('viewtransform', 'Standard')),
        'K':             K_str,  # <<< ÖNEMLİ: string yaz
    }

    # --- SCENE ---------------------------------------------------------------
    scene_config = {
        "camera_settings":   cam_settings,
        "sun_settings":      {"angle": 0.00927, "energy": float(sun_energy)},
        "body_settings_1":   {"pass_index": 1, "diffuse_bounces": 4},
        "body_settings_2":   {"pass_index": 2, "diffuse_bounces": 4},
        "body_settings_3":   {"pass_index": 3, "diffuse_bounces": 4},
        "rendering_settings":{"engine": "CYCLES", "device": "GPU", "samples": 256, "preview_samples": 16},
    }

    # --- Dosya adları / yazım ------------------------------------------------
    geom_filename  = f"geometry_dynamic_{idx:06d}.json"
    scene_filename = f"scene_dynamic_{idx:06d}.json"
    geom_path  = geom_dir  / geom_filename
    scene_path = scene_dir / scene_filename

    # NumPy → Python dönüşümü (K zaten string)
    def _to_py(o):
        import numpy as _np
        if isinstance(o, _np.integer):   return int(o)
        if isinstance(o, _np.floating):  return float(o)
        if isinstance(o, _np.ndarray):   return o.tolist()
        if isinstance(o, dict):          return {k: _to_py(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [_to_py(x) for x in o]
        return o

    geometry_data_safe = _to_py(geometry_data)
    scene_config_safe  = _to_py(scene_config)  # K string olduğu için aynen kalır

    geom_path.write_text(json.dumps(geometry_data_safe, indent=2))
    scene_path.write_text(json.dumps(scene_config_safe,  indent=2))

    print(f"   Written: {scene_path}")
    print(f"   Written: {geom_path}")

    # Sadece dosya adlarını döndür (State bu adları senaryo içinde bulur)
    return scene_filename, geom_filename


def _get_spice_data_for_time(sdp, utc_time):
    """Get SPICE data with quaternion normalization"""
    try:
        sd = sdp.get_spice_data(utc_time)
        for key in ("sun", "hrsc", "phobos", "mars", "deimos"):
            q = np.asarray(sd.get(key, {}).get("quaternion", []), dtype=float)
            n = np.linalg.norm(q)
            if n > 0:
                sd[key]["quaternion"] = (q / n).tolist()
        return sd
    except Exception as e:
        print(f"SPICE query failed: {e}")
        return None


# ============================================================================
# MAIN RUNNER
# ============================================================================
def run_pds_simulations(pds_dir, n_templates=27, max_images=None, 
                       output_tag="hybrid", sun_blender_scaler=3.90232e-1, 
                       param_ranges=None):
    """
    Main runner: PDS parsing + SPICE (optional) + parameter sweep + rendering
    
    Features:
    - SPICE optional with fallback
    - All CORTO compositing branches preserved
    - 16-bit PNG handling
    - Robust Excel/CSV logging
    - Tutorial-faithful Phobos shading
    """
    
    force_nvidia_gpu()

    #ROOT = Path(__file__).resolve().parent
    #os.chdir(ROOT)
    #sys.path.insert(0, str(ROOT))
    #print(f"Working directory: {os.getcwd()}")

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

    
    # Çıktı dizini (hem Excel/CSV hem de dinamik sahne yazımı için kullanıyoruz)
    out_dir = Path("output") / scenario_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Log dosyaları
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    xlsx_path = out_dir / f"phobos_results_{output_tag}_{stamp}.xlsx"
    csv_path = out_dir / f"phobos_results_{output_tag}_{stamp}.csv"

    # PDS parse
    print("\n1. Processing PDS database...")
    pds = CompactPDSProcessor(pds_dir)
    df = pds.parse_dir()
    
    if max_images is not None:
        df = df.head(int(max_images)).copy()

    # Parameter grid
    print("\n2. Building parameter grid...")
    combos = build_param_grid(n_templates, ranges=param_ranges)
    print(f"   Generated {len(combos)} parameter combinations")

    # SPICE processor (optional)
    sdp = SpiceDataProcessor() if HAS_SPICE else None
    if sdp:
        print("   SPICE processor available")
    else:
        print("   Using scenario positioning (SPICE not available)")

    # Configure rendering
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_depth = '16'
    bpy.context.scene.view_settings.view_transform = 'Raw'
    bpy.context.scene.render.image_settings.compression = 0

    rows = []

    print("\n3. Starting template generation...")
    for idx, row in df.reset_index(drop=True).iterrows():
        corto.Utils.clean_scene()
        img_name = row.get('file_name', f'IMG_{idx:06d}.IMG')
        utc_mid = row.get('UTC_MEAN_TIME')
        is_spice = (sdp is not None and isinstance(utc_mid, str) and utc_mid)

        print(f"\n{'='*80}")
        print(f"Image {idx+1}/{len(df)}: {img_name}")
        print(f"UTC: {utc_mid}")
        print(f"{'='*80}")

        # Per-IMG State (SPICE varsa dinamik, yoksa default)
        State_img = None
        solar_distance_km = AU_KM
        
        if is_spice:
            spice_data = _get_spice_data_for_time(sdp, utc_mid)
            if spice_data is not None:
                sd_km = spice_data.get('distances', {}).get('sun_to_phobos', AU_KM)
                solar_distance_km = float(sd_km) if sd_km else AU_KM
                
                base_energy = calculate_sun_strength(solar_distance_km, q_eff=1.0, 
                                                    sun_blender_scaler=sun_blender_scaler)
                
                scene_filename, geom_filename = _write_dynamic_scene_files(
                    out_dir, spice_data, _get_camera_cfg_with_fallback(), base_energy, idx
                )
                State_img = corto.State(scene=scene_filename,
                                        geometry=geom_filename,
                                        body=body_name,
                                        scenario=scenario_name)
                _add_asset_paths(State_img)                
                
                print(f"   Using SPICE positioning")
                print(f"   Solar distance: {solar_distance_km:.0f} km")


        if State_img is None:
            # SPICE yok/başarısız → default sahne ile tek State
            State_img = corto.State(scene=scene_name,
                                    geometry=geometry_name,
                                    body=body_name,
                                    scenario=scenario_name)
            _add_asset_paths(State_img)
            print(f"   Using scenario positioning")

        # Build scene
        ENV, cam, sun, bodies, mat_nodes, tree = build_scene_and_materials(State_img, body_name)
        # Position objects
        try:
            ENV.PositionAll(State_img, index=0 if is_spice else idx)
        except Exception as e:
            print(f"   Warning: Positioning failed: {e}")

        # Ensure camera is set
        cam_obj = cam.toBlender()
        if bpy.context.scene.camera is None:
            bpy.context.scene.camera = cam_obj or bpy.data.objects.get(cam.name)
        corto.Camera.select_camera(cam.name)
        for o in list(bpy.data.objects):
            if o.type == 'CAMERA' and o.name != cam.name:
                bpy.data.objects.remove(o, do_unlink=True)

        # Template loop
        for t_idx, params in enumerate(combos):
            print(f"\n   Template {t_idx + 1}/{len(combos)}")
            
            # Apply Phobos parameters
            set_phobos_params(mat_nodes, params)

            # Calculate sun energy with q_eff
            q_eff = float(params.get('q_eff', 1.0))
            sun_energy = calculate_sun_strength(
                solar_distance_km if is_spice else AU_KM,
                q_eff=q_eff,
                sun_blender_scaler=sun_blender_scaler
            )
            
            try:
                sun.set_energy(sun_energy)
            except Exception as e:
                print(f"   Warning: Sun energy update failed: {e}")

            # Frame ID
            n_t = len(combos)  # toplam template sayısı
            output_index = idx * n_t + t_idx   # görüntüler arası benzersiz & artan

            # Pozisyon indeksi: SPICE sahnesi tek örnek olduğundan 0 kalabilir
            pos_index = 0 if is_spice else idx
            try:
                ENV.PositionAll(State_img, index=pos_index)
            except Exception as e:
                print(f"   Warning: Positioning failed: {e}")

            # (opsiyonel) sahne frame'i eşitlemek istersen:
            bpy.context.scene.frame_current = output_index

            # Render: dosya adını belirleyen parametre bu!
            try:
                ENV.RenderOne(
                    cam, State_img,
                    index=output_index,
                    depth_flag=True
                )
                status = 'SUCCESS'
                
                # OPTIONAL: Post-process with 16-bit PNG handling
                img_path = Path(State_img.path["output_path"]) / "img" / f"{output_index:06d}.png"
                if img_path.exists():
                    try:
                        rendered_16bit = np.array(Image.open(str(img_path)))
                        if len(rendered_16bit.shape) == 3:
                            rendered_16bit = rendered_16bit[:,:,0]
                        save_raw_16bit_png(rendered_16bit, img_path)
                    except Exception as e:
                        print(f"   Warning: 16-bit PNG save failed: {e}")
                
            except Exception as e:
                print(f"   Error: Render failed: {e}")
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
            print(f"   Warning: Blend save failed: {e}")

    # Save results
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Phobos Photometric Template Generator - Hybrid")
    parser.add_argument("--pds_dir", default="PDS_Data", help="PDS IMG directory")
    parser.add_argument("--n_templates", type=int, default=27, help="Templates per image")
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