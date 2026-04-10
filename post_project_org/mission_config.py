#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mission_config.py
=================
Multi-mission PDS support: detection, configuration, and label-based pose extraction.

Supports:
    - HRSC (Mars Express) — via SPICE kernels
    - OSIRIS NAC/WAC (Rosetta) — via PDS label embedded pose data
    - Generic / Unknown — auto-probe from PDS label

Architecture:
    detect_mission(pds_path) → MissionConfig
    MissionConfig.camera_config → dict for CORTO scene JSON
    extract_pose_from_label(pds_path, cfg) → SPICE-compatible dict

Author: Multi-mission pipeline
Date: 2026
"""

import re
import math
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime


# ============================================================================
# QUATERNION HELPERS (standalone, no SPICE dependency)
# ============================================================================

def _q_mult(q1, q2):
    """Hamilton quaternion product [w,x,y,z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _q_conj(q):
    """Quaternion conjugate [w,-x,-y,-z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _q_normalize(q):
    n = np.linalg.norm(q)
    return q / n if n > 0 else q


# ============================================================================
# MISSION CONFIG DATACLASS
# ============================================================================

@dataclass
class MissionConfig:
    """Configuration for a specific spacecraft/instrument combination."""
    mission_id: str                         # 'HRSC', 'OSIRIS_NAC', 'GENERIC'
    use_spice: bool                         # True=SPICE kernels, False=PDS label pose
    camera_config: Optional[Dict]           # Dict for CORTO scene JSON (None=from SPICE)
    body_files: List[str]                   # OBJ files [phobos, mars, deimos]
    solar_distance_key: str                 # Key in spice_data['distances']
    camera_roll_correction: float = 0.0     # Degrees CW around boresight


# ============================================================================
# PREDEFINED MISSION CONFIGURATIONS
# ============================================================================

# Default body files (shared)
_DEFAULT_BODIES = [
    "g_phobos_144m_spc_0000n00000_v002.obj",
    "Mars_65k_km.obj",
    "g_deimos_162m_spc_0000n00000_v001.obj",
]

HRSC_CONFIG = MissionConfig(
    mission_id='HRSC',
    use_spice=True,
    camera_config=None,   # Obtained from SPICE at runtime
    body_files=_DEFAULT_BODIES,
    solar_distance_key='sun_to_phobos',
    camera_roll_correction=0.0,
)

# OSIRIS NAC camera config calculated from PDS label physical parameters:
#   CCD:            2048×2048 px, 13.5 μm pixel pitch
#   Binning:        2×2 → 1024×1024 effective
#   Sensor size:    2048 × 13.5μm = 27.648 mm (physical, pre-binning)
#   Focal length V: 712.4 mm   → FOV_V = 2*atan(27.648 / (2*712.4)) = 2.2226°
#   Focal length H: 721.1 mm   → FOV_H = 2*atan(27.648 / (2*721.1)) = 2.1963°
#   K matrix fx:    712.4 / (0.0135*2) = 26385.2 px (unbinned)
#                   → binned (2×2): 26385.2 / 2 ≈ NO — K uses physical focal/pixel
#                   fx = f_mm / (pixel_pitch_mm) = 712.4 / 0.0135 = 52770.4 (sensor px)
#                   For 1024 binned image: fx_binned = 712.4 / (0.027) = 26385.2
#   CORTO uses FOV mode (not K for projection), so FOV accuracy is critical.
_OSIRIS_FOV_DEG = 2.0 * math.degrees(math.atan(27.648 / (2.0 * 712.4)))  # 2.2226°
_OSIRIS_FX = 712.4 / (13.5e-3 * 2)  # Binned pixel = 27μm → fx = 26385.2

OSIRIS_NAC_CONFIG = MissionConfig(
    mission_id='OSIRIS_NAC',
    use_spice=False,
    camera_config={
        'fov': round(_OSIRIS_FOV_DEG, 6),     # Scientifically precise FOV
        'res_x': 1024,
        'res_y': 1024,
        'film_exposure': 0.039,
        'sensor': 'BW',
        'clip_start': 0.1,
        'clip_end': 1e7,
        'bit_encoding': '16',
        'viewtransform': 'Standard',
        'focal_length_mm': 712.4,
        'pixel_size_microns': [13.5, 13.5],
        'K': [
            [_OSIRIS_FX, 0, 512.0],
            [0, _OSIRIS_FX, 512.0],
            [0, 0, 1],
        ],
    },
    body_files=_DEFAULT_BODIES,
    solar_distance_key='sun_to_target',
    camera_roll_correction=90.0,    # 90° CW roll correction
)


# ============================================================================
# PDS LABEL PARSER (lightweight, no GDAL/SPICE dependency)
# ============================================================================

def _parse_pds_label(pds_path: Path, max_bytes: int = 32000) -> Dict[str, str]:
    """Parse PDS3 label key=value pairs from binary IMG file."""
    with open(pds_path, 'rb') as f:
        raw = f.read(max_bytes)
    text = raw.decode('latin-1')

    result = {}
    for m in re.finditer(r'^\s*(\w+)\s*=\s*(.+?)(?:\r?\n)', text, re.MULTILINE):
        key = m.group(1).strip()
        val = m.group(2).strip().strip('"')
        result[key] = val
    return result


def _grab_float(text: str, key: str) -> Optional[float]:
    m = re.search(rf'{key}\s*=\s*([\-\+]?[\d\.eE\+\-]+)', text)
    return float(m.group(1)) if m else None


def _grab_vector(text: str, key: str) -> Optional[np.ndarray]:
    m = re.search(rf'{key}\s*=\s*\(([^)]+)\)', text)
    if not m:
        return None
    parts = m.group(1).split(',')
    vals = []
    for p in parts:
        num = re.search(r'([\-\+]?[\d\.eE\+\-]+)', p.strip())
        if num:
            vals.append(float(num.group(1)))
    return np.array(vals[:3]) if len(vals) >= 3 else None


def _grab_group_quaternion(text: str, group_name: str) -> Optional[np.ndarray]:
    pattern = rf'GROUP\s*=\s*{group_name}(.*?)END_GROUP\s*=\s*{group_name}'
    m = re.search(pattern, text, re.DOTALL)
    if not m:
        return None
    block = m.group(1)
    qm = re.search(r'ORIGIN_ROTATION_QUATERNION\s*=\s*\(([^)]+)\)', block)
    if not qm:
        return None
    parts = qm.group(1).split(',')
    return np.array([float(p.strip()) for p in parts[:4]])


# ============================================================================
# MISSION DETECTION
# ============================================================================

def detect_mission(pds_path: Path) -> MissionConfig:
    """
    Auto-detect spacecraft/instrument from PDS label.

    Decision tree:
        1. INSTRUMENT_ID contains 'HRSC' or 'SRC' → HRSC
        2. INSTRUMENT_ID contains 'OSINAC'/'OSIWAC' or MISSION_NAME contains 'ROSETTA' → OSIRIS
        3. Otherwise → GENERIC (build config from label)

    Args:
        pds_path: Path to PDS IMG file

    Returns:
        MissionConfig for detected mission
    """
    label = _parse_pds_label(pds_path)

    instrument = label.get('INSTRUMENT_ID', '').upper()
    mission = label.get('MISSION_NAME', '').upper()
    target = label.get('TARGET_NAME', '').upper()

    # ── HRSC (Mars Express) ──
    if 'HRSC' in instrument or 'SRC' in instrument:
        print(f"  🔍 Mission detected: HRSC (Mars Express)")
        return HRSC_CONFIG

    # ── OSIRIS (Rosetta) ──
    if 'OSINAC' in instrument or 'OSIWAC' in instrument or 'ROSETTA' in mission:
        is_nac = 'OSINAC' in instrument
        print(f"  🔍 Mission detected: OSIRIS {'NAC' if is_nac else 'WAC'} (Rosetta)")
        if is_nac:
            return OSIRIS_NAC_CONFIG
        else:
            # WAC has different FOV — build from label
            return _build_generic_config(pds_path, label, base_id='OSIRIS_WAC')

    # ── GENERIC / UNKNOWN ──
    print(f"  🔍 Mission: UNKNOWN (instrument={instrument}, mission={mission})")
    print(f"     Attempting generic config from PDS label...")
    return _build_generic_config(pds_path, label)


def _build_generic_config(pds_path: Path, label: Dict, base_id: str = 'GENERIC') -> MissionConfig:
    """Build MissionConfig from PDS label fields for unknown missions."""
    with open(pds_path, 'rb') as f:
        text = f.read(32000).decode('latin-1')

    # Try extracting camera parameters from label
    fov_elev = _grab_float(text, 'ELEVATION_FOV')
    fov_azim = _grab_float(text, 'AZIMUTH_FOV')
    lines = _grab_float(text, r'\bLINES')
    samples = _grab_float(text, 'LINE_SAMPLES')

    if lines is None or samples is None:
        raise ValueError(f"Cannot build generic config: LINES/LINE_SAMPLES missing in {pds_path.name}")

    res_x = int(samples)
    res_y = int(lines)

    # FOV: prefer label value, fallback to HRSC default
    if fov_elev is not None:
        fov = fov_elev
    elif fov_azim is not None:
        fov = fov_azim
    else:
        print(f"  ⚠️ No FOV in label — using 2.0° default")
        fov = 2.0

    # K matrix: try to compute from focal length + pixel size
    focal_v = _grab_float(text, r'VERTICAL_FOCAL_LENGTH|FOCAL_LENGTH')
    pixel_w = _grab_float(text, 'DETECTOR_PIXEL_WIDTH')
    if focal_v and pixel_w:
        fx = (focal_v * 1000) / (pixel_w * 1e-3)  # m→mm, μm→mm
    else:
        fx = res_x / (2.0 * math.tan(math.radians(fov / 2.0)))

    # Check for embedded pose data
    has_pose = _grab_vector(text, 'SC_TARGET_POSITION_VECTOR') is not None

    print(f"  📐 Generic config: FOV={fov}°, res={res_x}×{res_y}, use_spice={not has_pose}")

    return MissionConfig(
        mission_id=base_id,
        use_spice=not has_pose,
        camera_config={
            'fov': fov,
            'res_x': res_x,
            'res_y': res_y,
            'film_exposure': 0.039,
            'sensor': 'BW',
            'clip_start': 0.1,
            'clip_end': 1e7,
            'bit_encoding': '16',
            'viewtransform': 'Standard',
            'K': [[fx, 0, res_x / 2], [0, fx, res_y / 2], [0, 0, 1]],
        },
        body_files=_DEFAULT_BODIES,
        solar_distance_key='sun_to_target',
        camera_roll_correction=0.0,
    )


# ============================================================================
# SOLAR DISTANCE FROM LABEL
# ============================================================================

def get_solar_distance_from_label(pds_path: Path) -> float:
    """Extract sun→target distance from PDS label (km)."""
    with open(pds_path, 'rb') as f:
        text = f.read(32000).decode('latin-1')

    # Try direct field
    dist = _grab_float(text, 'SPACECRAFT_SOLAR_DISTANCE')
    if dist is not None:
        return dist

    # Compute from vectors
    sc_to_sun = _grab_vector(text, 'SC_SUN_POSITION_VECTOR')
    sc_to_target = _grab_vector(text, 'SC_TARGET_POSITION_VECTOR')
    if sc_to_sun is not None and sc_to_target is not None:
        return float(np.linalg.norm(sc_to_sun - sc_to_target))

    raise ValueError(f"Cannot determine solar distance from {pds_path.name}")


# ============================================================================
# POSE EXTRACTION FROM PDS LABEL (SPICE-compatible dict)
# ============================================================================

def _compute_mars_quaternion_iau(utc_str: str) -> np.ndarray:
    """
    Compute IAU_MARS → J2000 quaternion using IAU 2015 Mars rotation model.
    Equivalent to SPICE pxform('IAU_MARS', 'J2000', et).

    Parameters from IAU 2015 Report:
        α₀ = 317.68143° − 0.1061°T
        δ₀ = 52.88650° − 0.0609°T
        W  = 176.630° + 350.89198226°d
    """
    dt = datetime.strptime(utc_str.split('.')[0], "%Y-%m-%dT%H:%M:%S")
    j2000_epoch = datetime(2000, 1, 1, 12, 0, 0)
    d = (dt - j2000_epoch).total_seconds() / 86400.0
    T = d / 36525.0

    alpha0 = np.radians(317.68143 - 0.1061 * T)
    delta0 = np.radians(52.88650 - 0.0609 * T)
    W = np.radians(176.630 + 350.89198226 * d)

    ca, sa = np.cos(alpha0), np.sin(alpha0)
    cd, sd = np.cos(delta0), np.sin(delta0)
    cw, sw = np.cos(W), np.sin(W)

    # Rotation matrix: IAU_MARS body-fixed → J2000 inertial
    R = np.array([
        [-sa*cw - ca*sd*sw,  sa*sw - ca*sd*cw,  ca*cd],
        [ ca*cw - sa*sd*sw, -ca*sw - sa*sd*cw,  sa*cd],
        [ cd*sw,             cd*cw,              sd   ]
    ])

    # Matrix → quaternion (Shepperd's method)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z])
    if q[0] < 0:
        q *= -1
    return q / np.linalg.norm(q)


def _calculate_sun_alignment_quaternion(pos_sun: np.ndarray) -> np.ndarray:
    """
    Compute sun lamp quaternion for Blender (observer→Sun direction).
    EXACT COPY of SpiceDataProcessor._calculate_sun_alignment_quaternion().
    """
    norm = np.linalg.norm(pos_sun)
    if norm == 0:
        return np.array([1.0, 0.0, 0.0, 0.0])

    unit_to_sun = pos_sun / norm
    light_direction = -unit_to_sun  # Sun → Observer

    start_vec = np.array([0.0, 0.0, -1.0])  # Blender lamp default
    target_vec = light_direction

    dot = np.dot(start_vec, target_vec)

    if dot < -0.999999:
        cross = np.cross(np.array([1.0, 0.0, 0.0]), start_vec)
        if np.linalg.norm(cross) < 0.000001:
            cross = np.cross(np.array([0.0, 1.0, 0.0]), start_vec)
        axis = cross / np.linalg.norm(cross)
        return np.array([0.0, axis[0], axis[1], axis[2]])

    cross = np.cross(start_vec, target_vec)
    w_comp = math.sqrt((1.0 + dot) * 2.0)

    q = np.array([w_comp / 2.0, cross[0] / w_comp, cross[1] / w_comp, cross[2] / w_comp])
    return q / np.linalg.norm(q)


def _calculate_camera_blender_quaternion(q_cam_to_j2000: np.ndarray,
                                          roll_correction_deg: float = 0.0) -> np.ndarray:
    """
    Convert camera→J2000 quaternion to Blender convention.
    Same logic as SpiceDataProcessor._calculate_hrsc_blender_quaternion()
    with optional roll correction.

    Steps:
        1. Apply 180° X flip (camera +Z boresight → Blender -Z)
        2. Apply CW roll correction around boresight (if nonzero)
        3. Normalize, ensure w > 0
    """
    Q_flip = np.array([0.0, 1.0, 0.0, 0.0])  # 180° about X
    Q_blender = _q_mult(q_cam_to_j2000, Q_flip)

    if roll_correction_deg != 0.0:
        angle = math.radians(roll_correction_deg)
        Q_roll = np.array([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])
        Q_blender = _q_mult(Q_blender, Q_roll)

    if Q_blender[0] < 0:
        Q_blender *= -1

    return _q_normalize(Q_blender)


def extract_pose_from_label(pds_path: Path, mission_cfg: MissionConfig) -> Dict:
    """
    Extract SPICE-compatible pose dict from PDS label (no SPICE required).

    Returns dict with same structure as SpiceDataProcessor.get_spice_data():
        {
            'phobos': {'position': [...], 'quaternion': [...]},
            'mars':   {'position': [...], 'quaternion': [...]},
            'deimos': {'position': [...], 'quaternion': [...]},
            'sun':    {'position': [...], 'quaternion': [...]},
            'hrsc':   {'position': [0,0,0], 'quaternion': [...]},
            'distances': {'sun_to_phobos': ..., 'sun_to_target': ..., ...}
        }
    """
    with open(pds_path, 'rb') as f:
        text = f.read(32000).decode('latin-1')

    # ── Positions (observer→target vectors in J2000, km) ──
    sc_to_sun = _grab_vector(text, 'SC_SUN_POSITION_VECTOR')
    sc_to_target = _grab_vector(text, 'SC_TARGET_POSITION_VECTOR')

    if sc_to_sun is None or sc_to_target is None:
        raise ValueError(f"Missing SC_SUN/TARGET_POSITION_VECTOR in {pds_path.name}")

    # ── Camera quaternion ──
    # PDS label gives J2000→SC quaternion; SPICE gives Camera→J2000
    # So we need: conj(J2000→NAC) = NAC→J2000 (= what SPICE pxform returns)
    q_j2000_to_sc = _grab_group_quaternion(text, 'SC_COORDINATE_SYSTEM')
    q_sc_to_cam = _grab_group_quaternion(text, 'CAMERA_COORDINATE_SYSTEM')

    if q_j2000_to_sc is None:
        raise ValueError(f"Missing SC_COORDINATE_SYSTEM quaternion in {pds_path.name}")

    if q_sc_to_cam is not None:
        q_j2000_to_cam = _q_mult(q_sc_to_cam, q_j2000_to_sc)
    else:
        q_j2000_to_cam = q_j2000_to_sc

    # Conjugate: J2000→CAM → CAM→J2000 (matches SPICE pxform direction)
    q_cam_to_j2000 = _q_conj(_q_normalize(q_j2000_to_cam))

    # Apply Blender flip + mission-specific roll correction
    q_cam_blender = _calculate_camera_blender_quaternion(
        q_cam_to_j2000,
        roll_correction_deg=mission_cfg.camera_roll_correction
    )

    # ── Sun lamp quaternion ──
    q_sun = _calculate_sun_alignment_quaternion(sc_to_sun)

    # ── Mars quaternion (IAU model) ──
    utc_str = None
    m = re.search(r'START_TIME\s*=\s*([\d\-T:\.]+)', text)
    if m:
        utc_str = m.group(1)
    if utc_str is None:
        raise ValueError(f"Missing START_TIME in {pds_path.name}")

    q_mars = _compute_mars_quaternion_iau(utc_str)

    # ── Placeholder bodies (Phobos far, Deimos very far) ──
    phobos_pos = sc_to_target + np.array([50000.0, 0.0, 0.0])
    deimos_pos = np.array([1.0e9, 0.0, 0.0])
    q_identity = np.array([1.0, 0.0, 0.0, 0.0])

    # ── Distances ──
    sun_to_target_km = float(np.linalg.norm(sc_to_sun - sc_to_target))
    sun_to_sc_km = float(np.linalg.norm(sc_to_sun))

    # Build SPICE-compatible result dict
    result = {
        'phobos': {'position': phobos_pos.tolist(), 'quaternion': q_identity.tolist()},
        'mars':   {'position': sc_to_target.tolist(), 'quaternion': q_mars.tolist()},
        'deimos': {'position': deimos_pos.tolist(), 'quaternion': q_identity.tolist()},
        'sun':    {'position': sc_to_sun.tolist(), 'quaternion': q_sun.tolist()},
        'hrsc':   {'position': [0.0, 0.0, 0.0], 'quaternion': q_cam_blender.tolist()},
        'distances': {
            'sun_to_phobos': sun_to_target_km,   # HRSC uyumluluk (same key)
            'sun_to_target': sun_to_target_km,
            'sun_to_mars': sun_to_target_km,
            'sun_to_hrsc': sun_to_sc_km,
        },
    }

    return result
