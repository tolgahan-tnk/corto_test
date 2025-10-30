# -*- coding: utf-8 -*-
"""
phobos_data.py
Phobos Photometric Template Generator - Data Management Module

Sorumluluklar:
- PDS veri işleme
- SPICE entegrasyonu
- Parametre grid oluşturma
- Güneş enerjisi hesaplamaları
- Yardımcı fonksiyonlar
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# SPICE (Optional)
SPICE_IMPORT_ERROR = None
HAS_SPICE = False

try:
    from spice_data_processor import SpiceDataProcessor
    HAS_SPICE = True
except Exception as e:
    SpiceDataProcessor = None
    SPICE_IMPORT_ERROR = str(e)
    print(f"❌ SPICE import failed: {e}")
    print("   This will cause errors when running simulations.")
    print("   Install with: pip install spiceypy")

# Constants
AU_KM = 149_597_870.7


# ============================================================================
# PDS PROCESSOR
# ============================================================================
class CompactPDSProcessor:
    """PDS IMG dosyalarını işler ve DataFrame'e dönüştürür"""
    
    def __init__(self, pds_data_path):
        self.pds_data_path = Path(pds_data_path)
        import re
        self._key_val = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=\s*(.*)$")

    def parse_dir(self):
        """IMG dosyalarını parse eder"""
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
                print(f" Processed: {img_file.name}")
            except Exception as e:
                print(f" Error: {img_file.name}: {e}")

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
# SUN ENERGY CALCULATION
# ============================================================================
def calculate_sun_strength(solar_distance_km, q_eff=1.0, sun_blender_scaler=1.0):
    """Güneş uzaklığından Blender Sun energy değerini hesaplar"""
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
# PARAMETER GRID BUILDER
# ============================================================================
def build_param_grid(n_templates, ranges=None):
    """Fotometrik parametre grid'i oluşturur (IOR dahil)"""
    default_ranges = {
        'base_gray': (0.02, 0.60),
        'tex_mix': (0.0, 1.0),
        'oren_rough': (0.0, 1.0),
        'princ_rough': (0.0, 1.0),
        'shader_mix': (0.0, 1.0),
        'ior': (1.33, 1.79),  # IOR taraması
        'q_eff': (0.7, 1.4),
        'threshold_value': (0.0, 0.15),
        'mars_rough': (0.0, 1.0),
        'mars_albedo_mul': (0.0, 2.0),
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
# SPICE HELPER FUNCTIONS
# ============================================================================
def get_camera_config():
    """
    SPICE'dan kamera konfigürasyonu alır.
    SPICE yoksa veya başarısız olursa hata fırlatır (NO FALLBACK).
    """
    if not HAS_SPICE:
        raise RuntimeError(
            "❌ SPICE is required but not available!\n"
            f"Import error: {SPICE_IMPORT_ERROR}\n"
            "Please install SPICE dependencies or check spice_data_processor.py"
        )
    
    try:
        sdp_tmp = SpiceDataProcessor()
        spice_cfg = sdp_tmp.get_hrsc_camera_config('SRC')
        
        # Validate that we got real data
        if not spice_cfg or 'fov' not in spice_cfg:
            raise ValueError("SPICE returned empty or invalid camera configuration")
        
        return {
            'fov': float(spice_cfg.get('fov')),
            'res_x': int(spice_cfg.get('res_x')),
            'res_y': int(spice_cfg.get('res_y')),
            'film_exposure': float(spice_cfg.get('film_exposure')),
            'sensor': str(spice_cfg.get('sensor')),
            'clip_start': float(spice_cfg.get('clip_start')),
            'clip_end': float(spice_cfg.get('clip_end')),
            'bit_encoding': str(spice_cfg.get('bit_encoding')),
            'viewtransform': str(spice_cfg.get('viewtransform')),
            'K': spice_cfg.get('K'),
        }
    except Exception as e:
        raise RuntimeError(
            f"❌ Failed to get SPICE camera configuration: {e}\n"
            "Camera data must be available in SPICE kernels."
        ) from e


def get_spice_data_for_time(sdp, utc_time):
    """
    SPICE verisi alır ve quaternion normalize eder.
    Başarısız olursa hata fırlatır (NO FALLBACK).
    """
    if sdp is None:
        raise RuntimeError(
            "❌ SPICE processor is None! Cannot query SPICE data.\n"
            "Ensure SPICE is properly initialized."
        )
    
    try:
        sd = sdp.get_spice_data(utc_time)
        
        if sd is None:
            raise ValueError(f"SPICE returned None for UTC time: {utc_time}")
        
        # Normalize quaternions
        for key in ("sun", "hrsc", "phobos", "mars", "deimos"):
            if key not in sd:
                raise KeyError(f"Missing '{key}' in SPICE data for time {utc_time}")
            
            quaternion_data = sd.get(key, {}).get("quaternion")
            if quaternion_data is None:
                raise ValueError(f"Missing quaternion for '{key}' in SPICE data")
            
            q = np.asarray(quaternion_data, dtype=float)
            n = np.linalg.norm(q)
            
            if n == 0 or not np.isfinite(n):
                raise ValueError(f"Invalid quaternion for '{key}': {quaternion_data}")
            
            sd[key]["quaternion"] = (q / n).tolist()
        
        return sd
        
    except Exception as e:
        raise RuntimeError(
            f"❌ Failed to query SPICE data for UTC time '{utc_time}': {e}\n"
            "Ensure SPICE kernels are loaded and time is within coverage."
        ) from e


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def convert_to_python(obj):
    """NumPy tiplerini Python native tiplerine çevirir (JSON serileştirme için)"""
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


def write_dynamic_scene_files(base_output_dir, spice_data, camera_cfg, sun_energy, idx):
    """
    SPICE verisinden dinamik geometry/scene JSON dosyalarını yazar.
    CORTO'nun beklediği formatta dosya oluşturur.
    
    Returns:
        tuple: (scene_filename, geometry_filename)
    """
    from pathlib import Path
    import json
    import numpy as np

    scenario_dir = Path("input/S07_Mars_Phobos_Deimos")
    scene_dir = scenario_dir / "scene"
    geom_dir = scenario_dir / "geometry"
    scene_dir.mkdir(parents=True, exist_ok=True)
    geom_dir.mkdir(parents=True, exist_ok=True)

    # Geometry data
    geometry_data = {
        "sun": {
            "position": [spice_data["sun"]["position"]],
            "orientation": [spice_data["sun"]["quaternion"]],
        },
        "camera": {
            "position": [spice_data["hrsc"]["position"]],
            "orientation": [spice_data["hrsc"]["quaternion"]],
        },
        "body_1": {
            "position": [spice_data["phobos"]["position"]],
            "orientation": [spice_data["phobos"]["quaternion"]],
        },
        "body_2": {
            "position": [spice_data["mars"]["position"]],
            "orientation": [spice_data["mars"]["quaternion"]],
        },
        "body_3": {
            "position": [spice_data["deimos"]["position"]],
            "orientation": [spice_data["deimos"]["quaternion"]],
        },
    }

    # Camera settings (K'yı string olarak yaz)
    K_list = camera_cfg.get('K', [[1222.0, 0.0, 512.0], [0.0, 1222.0, 512.0], [0.0, 0.0, 1.0]])
    K_list = np.array(K_list, dtype=float).tolist()
    K_str = "np.array(" + json.dumps(K_list) + ", dtype=float)"

    cam_settings = {
        'fov': float(camera_cfg.get('fov', 0.54)),
        'res_x': int(camera_cfg.get('res_x', 1024)),
        'res_y': int(camera_cfg.get('res_y', 1024)),
        'film_exposure': float(camera_cfg.get('film_exposure', 1.0)),
        'sensor': str(camera_cfg.get('sensor', 'BW')),
        'clip_start': float(camera_cfg.get('clip_start', 0.1)),
        'clip_end': float(camera_cfg.get('clip_end', 1.0e8)),
        'bit_encoding': str(camera_cfg.get('bit_encoding', '16')),
        'viewtransform': str(camera_cfg.get('viewtransform', 'Standard')),
        'K': K_str,
    }

    # Scene config
    scene_config = {
        "camera_settings": cam_settings,
        "sun_settings": {"angle": 0.00927, "energy": float(sun_energy)},
        "body_settings_1": {"pass_index": 1, "diffuse_bounces": 4},
        "body_settings_2": {"pass_index": 2, "diffuse_bounces": 4},
        "body_settings_3": {"pass_index": 3, "diffuse_bounces": 4},
        "rendering_settings": {"engine": "CYCLES", "device": "GPU", "samples": 256, "preview_samples": 16},
    }

    # Dosya adları
    geom_filename = f"geometry_dynamic_{idx:06d}.json"
    scene_filename = f"scene_dynamic_{idx:06d}.json"
    geom_path = geom_dir / geom_filename
    scene_path = scene_dir / scene_filename

    # NumPy → Python dönüşümü
    geometry_data_safe = convert_to_python(geometry_data)
    scene_config_safe = convert_to_python(scene_config)

    geom_path.write_text(json.dumps(geometry_data_safe, indent=2))
    scene_path.write_text(json.dumps(scene_config_safe, indent=2))

    print(f" Written: {scene_path}")
    print(f" Written: {geom_path}")

    return scene_filename, geom_filename