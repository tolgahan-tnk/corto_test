"""Optimization configuration.

Defines all parameters, bounds, noise pipeline settings,
multi-metric scoring weights, and algorithm hyperparameters.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple


class ParamDef(NamedTuple):
    name: str
    group: str
    lo: float
    hi: float
    x0: float
    optimize: bool
    json_key: str


# -- Paths -----------------------------------------------------------------
SCENARIO = "S07_Mars_Phobos_Deimos"
INPUT_DIR = Path("input") / SCENARIO
COEFF_PATH = INPUT_DIR / "scene" / "coefficients.json"

BODY_NAMES = [
    "g_phobos_036m_spc_0000n00000_v002.obj",
    "Mars_65k_km.obj",
    "g_deimos_162m_spc_0000n00000_v001.obj",
]

# -- Per-Image PDS Metadata -------------------------------------------------
PDS_IMAGES = [
    {
        "file": "PDS_Data_Mars/H7982_0006_SR2.IMG",
        "camera": "hrsc",
        "calibrated": True,
        "exposure_ms": 16.128,
        "lines": 1017,
        "line_samples": 1008,
        "sample_type": "MSB_INTEGER",
        "sample_bits": 16,
    },
    {
        "file": "PDS_Data_Mars/H7982_0003_SR2.IMG",
        "camera": "hrsc",
        "calibrated": True,
        "exposure_ms": 16.128,
        "lines": 1017,
        "line_samples": 1008,
        "sample_type": "MSB_INTEGER",
        "sample_bits": 16,
    },
    {
        "file": "PDS_Combined/HF490_0004_SR2.IMG",
        "camera": "hrsc",
        "calibrated": True,
        "exposure_ms": 5.04,
        "lines": 1018,
        "line_samples": 1007,
        "sample_type": "MSB_INTEGER",
        "sample_bits": 16,
    },
    {
        "file": "PDS_Combined/N20070224T184008217ID4CF82.IMG",
        "camera": "osiris",
        "calibrated": True,
        "exposure_ms": 30.0,
        "lines": 1024,
        "line_samples": 1024,
        "sample_type": "PC_REAL",
        "sample_bits": 32,
    },
    {
        "file": "PDS_Rosetta/N20070224T215318200ID30F82.IMG",
        "camera": "osiris",
        "calibrated": True,
        "exposure_ms": 30.0,
        "lines": 1024,
        "line_samples": 1024,
        "sample_type": "PC_REAL",
        "sample_bits": 32,
    },
]

# -- Atmosphere toggle ------------------------------------------------------
# Set False to disable Mars atmosphere creation entirely (skips volume render).
# Also excludes atm_* parameters from optimization (their x0 values remain in
# ALL_X0 but are not optimized).
USE_ATMOSPHERE = False

# -- Parameters (24) --------------------------------------------------------
PARAMS: list[ParamDef] = [
    # Phobos (0-5)
    ParamDef("base_gray",        "phobos", 0.089, 0.281, 0.185,  True, "base_gray"),
    ParamDef("tex_mix",          "phobos", 0.0,   1.0,   0.825,  True, "tex_mix"),
    ParamDef("oren_rough",       "phobos", 0.5,   1.0,   0.954,  True, "oren_rough"),
    ParamDef("princ_rough",      "phobos", 0.2,   1.0,   0.497,  True, "princ_rough"),
    ParamDef("shader_mix",       "phobos", 0.0,   1.0,   0.553,  True, "shader_mix"),
    ParamDef("ior",              "phobos", 1.0,   2.5,   1.331,  True, "ior"),
    # Mars (6-13)
    ParamDef("mars_base_gray",   "mars",   0.036, 0.307, 0.171,  True, "base_gray"),
    ParamDef("mars_tex_mix",     "mars",   0.0,   0.5,   0.056,  True, "tex_mix"),
    ParamDef("mars_oren_rough",  "mars",   0.5,   1.0,   0.999,  True, "oren_rough"),
    ParamDef("mars_princ_rough", "mars",   0.2,   1.0,   0.740,  True, "princ_rough"),
    ParamDef("mars_shader_mix",  "mars",   0.0,   1.0,   0.080,  True, "shader_mix"),
    ParamDef("mars_ior",         "mars",   1.2,   3.5,   2.383,  True, "ior"),
    ParamDef("mars_albedo_mul",  "mars",   2.60,  2.60,   2.60,  False, "albedo_mul"),
    ParamDef("mars_contrast",    "mars",   0.0,   2.0,   0.176,  True, "contrast"),
    # Atmosphere (14-19) — optimize field tied to USE_ATMOSPHERE
    ParamDef("atm_beta0",        "atm",    1e-8,  1e-2,  0.00062, USE_ATMOSPHERE, "beta0"),
    ParamDef("atm_scale_height", "atm",    0.0, 120.0, 60.06,    USE_ATMOSPHERE, "scale_height"),
    ParamDef("atm_anisotropy",   "atm",   -1.0,   1.0,  -0.183,   USE_ATMOSPHERE, "anisotropy"),
    ParamDef("atm_color_r",      "atm",    0.1,   3.0,   0.746,   USE_ATMOSPHERE, "color_r"),
    ParamDef("atm_color_g",      "atm",    0.1,   3.0,   0.620,   USE_ATMOSPHERE, "color_g"),
    ParamDef("atm_color_b",      "atm",    0.1,   3.0,   0.370,   USE_ATMOSPHERE, "color_b"),
    # Per-camera (20-23)
    # HRSC does NOT saturate even at ss=0.25 (raw render verified) -> bound 1.0 safe.
    # OSIRIS saturates around ss~0.05-0.06; equilibrium is ~0.01-0.02. Bound 0.1
    # keeps the per-coord CMA_std (0.2 x 0.099 ~ 0.02) proportional to x0 so the
    # first generation no longer blows past the uint16 ceiling.
    ParamDef("sun_scaler_hrsc",  "render", 0.001, 0.1,  0.002,     True, "sun_scaler_hrsc"),
    ParamDef("sun_scaler_osiris","render", 0.001, 0.1,  0.015,    True, "sun_scaler_osiris"),
    ParamDef("dc_hrsc",          "render", 0.0,  0.01, 0.000127, True, "dc_hrsc"),
    ParamDef("dc_osiris",        "render", 0.0,  0.01, 0.000050, True, "dc_osiris"),
]

# -- Noise Pipeline ----------------------------------------------------------
HOT_PIXEL_SIGMA = 5.0       # Sigma multiplier for float data (OSIRIS) 
HOT_PIXEL_GAP_LEN = 2       # Min consecutive empty bins for integer gap detection

# -- Normalization -----------------------------------------------------------
NORMALIZE = False            # True: percentile clip + [0,1] | False: raw DN
NORM_LOW_P = 1.0             # Lower percentile (if NORMALIZE=True)
NORM_HIGH_P = 99.0           # Upper percentile

# -- K-Sweep (Shadow Mask Based Noise Threshold) ----------------------------
K_SWEEP_ENABLED = True            # True: auto-find k, False: use K_FIXED
K_SWEEP_RANGE = (0, 99)           # Percentile sweep range
K_SWEEP_STEP = 1                  # Percentile step size
K_MAX_LIT_LOSS_PCT = 5.0          # Max acceptable lit pixel loss (%)
K_FIXED: dict[str, int] = {}      # Manual override: {"H7982_0003_SR2": 5}
                                   # Empty -> auto sweep for all

# -- DC Application ----------------------------------------------------------
DC_IN_BLENDER = False        # True: compositing node | False: post-process

# -- Scoring -----------------------------------------------------------------
SCORING_MODE = "masked"
MASK_WEIGHTS = {"phobos": 0.5, "mars": 0.5}
MIN_PIXELS = 100
MIN_OBJECT_FRACTION = 0.005  # Object must cover ≥ 0.5% of frame to be scored
ALIGN_MAX_SHIFT_FRAC = 0.25  # Max 25% pixel shift during global alignment
                             # (pose/FOV uncertainty between synthetic & PDS)

# Per-object metrics (all normalized to lower=better)
METRIC_WEIGHTS = {
    "nrmse": 0.20,   # Pixel-level error (lower=better)
    "ssim":  0.20,   # Structural similarity (higher->1-val)
    "emd":   0.10,   # Earth Mover's Distance — DN position+shape (lower=better)
    "ncc":   0.10,   # Normalized cross-correlation (higher->1-val)
    "gmsd":  0.20,   # Gradient similarity (lower=better)
    "lpips": 0.20,   # Perceptual similarity (lower=better)
}
DN_RATIO_WEIGHT = 0.05       # Cross-object relative brightness score
BRIGHTNESS_WEIGHT = 0.10     # Per-object absolute brightness (pre-normalization)
EXPOSURE_REF_MS = 16.128     # Reference exposure in ms (for Blender film_exposure scaling)
EMD_BINS = 256               # Histogram bin count for EMD

# Debug image saving (True -> saves masked aligned images per eval) 
SAVE_DEBUG_IMAGES = True
DEBUG_DIR = Path("debug_scoring")

# Full stage-by-stage debug (True -> saves ALL intermediate states per eval:
# raw crops, normalized, Pass1 aligned full-frame, Pass2 crops, overlap masks).
# Heavy disk usage (~75 files/eval). Use only when debugging pipeline issues.
DEBUG_SAVE_FULL = False

# -- Algorithm ---------------------------------------------------------------
ALGORITHM = "cmaes"
ALGO_SETTINGS = {
    # 12h budget @ ~80s/eval = 540 evals, n=17 boyut   
    # CMA-ES: Hansen 2016 default popsize=4+⌊3·ln(n)⌋=12, sigma0=0.2 (1/5 range)  
    "cmaes":    {"sigma0": 0.20, "popsize": 12, "maxiter": 240},
    # PSO: Engelbrecht 2007: swarm ~1.5×n; LDIW (Shi & Eberhart 1999); constriction (Clerc & Kennedy 2002) 
    "pso":      {"n_particles": 18, "maxiter": 30,
                 "w_min": 0.4, "w_max": 0.9, "c1": 2.05, "c2": 2.05},
    "bayesian": {"n_initial": 10, "n_iter": 50, "acq": "ei"},
    "de":       {"popsize": 15, "maxiter": 200, "strategy": "best1bin"},
}

# -- Render ------------------------------------------------------------------
N_FRAMES = len(PDS_IMAGES)
RENDER_SAMPLES = 64
FRAME_WEIGHTS = [1.0] * N_FRAMES

# -- Output ------------------------------------------------------------------
LOG_FILE = "optimization_log.json"
CHECKPOINT_EVERY = 5

# -- Derived -----------------------------------------------------------------
_GROUP_MAP = {
    "phobos": "phobos",
    "mars": "mars",
    "atm": "atmosphere",
    "render": "render",
}
OPT_INDICES = [i for i, p in enumerate(PARAMS) if p.optimize]
N_OPTIMIZED = len(OPT_INDICES)
BOUNDS_LO = [PARAMS[i].lo for i in OPT_INDICES]
BOUNDS_HI = [PARAMS[i].hi for i in OPT_INDICES]
X0 = [PARAMS[i].x0 for i in OPT_INDICES]
ALL_X0 = [p.x0 for p in PARAMS]
CAM_SUN_SCALER = {"hrsc": 20, "osiris": 21}
CAM_DC = {"hrsc": 22, "osiris": 23}
