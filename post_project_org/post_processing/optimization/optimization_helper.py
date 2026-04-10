#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimization_helper.py
======================
Helper functions for photometric parameter optimization.

Author: Optimization Pipeline
Date: 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable, Union
import json
import matplotlib.pyplot as plt
import openpyxl 
import sys
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import evaluation function from quick_evaluator


# Forward declaration for corto_renderer (to avoid circular import)
render_synthetic_for_params = None
# ============================================================================
# LIGHTWEIGHT METRIC STASH (for per-eval logging)
# ============================================================================
_LAST_EVAL_METRICS: dict = {
    "nmrse": float("nan"), "ssim": float("nan"),
    "ms_ssim": float("nan"), "gmsd": float("nan"),
    "hist_corr": float("nan"), "orb_penalty": 0.0,
    "ncc": float("nan"), "lpips": float("nan")
}

def set_last_eval_metrics(nmrse: float, ssim: float,
                          ms_ssim: float = float("nan"),
                          gmsd: float = float("nan"),
                          hist_corr: float = float("nan"),
                          orb_penalty: float = 0.0,
                          ncc: float = float("nan"),
                          lpips: float = float("nan")):
    _LAST_EVAL_METRICS["nmrse"] = float(nmrse)
    _LAST_EVAL_METRICS["ssim"] = float(ssim)
    _LAST_EVAL_METRICS["ms_ssim"] = float(ms_ssim)
    _LAST_EVAL_METRICS["gmsd"] = float(gmsd)
    _LAST_EVAL_METRICS["hist_corr"] = float(hist_corr)
    _LAST_EVAL_METRICS["orb_penalty"] = float(orb_penalty)
    _LAST_EVAL_METRICS["ncc"] = float(ncc)
    _LAST_EVAL_METRICS["lpips"] = float(lpips)

def get_last_eval_metrics():
    return dict(_LAST_EVAL_METRICS)

# ============================================================================
# ADAPTIVE CROP STATE - İlk N iterasyonda bbox öğren, sonra kilitle
# Per-image: her IMG kendi bbox geçmişine ve kilitli bbox'ına sahiptir.
# ============================================================================
_ADAPTIVE_CROP_STATE = {
    'per_image': {},        # {img_stem: {'history': [], 'locked_bbox': None}}
    'eval_count': 0,        # Toplam eval sayacı (warmup N'i için)
    'n': 0,                 # Kaç iterasyondan sonra kilitlenecek (0=devre dışı)
    'warmup_complete': False  # Warmup tamamlandı mı?
}

def _find_median_bbox(history):
    """Bbox geçmişinden median koordinatlar hesapla."""
    if not history:
        return None
    y0s = [b[0] for b in history]
    y1s = [b[1] for b in history]
    x0s = [b[2] for b in history]
    x1s = [b[3] for b in history]
    return (
        int(np.median(y0s)),
        int(np.median(y1s)),
        int(np.median(x0s)),
        int(np.median(x1s))
    )

def reset_adaptive_crop_state(n: int = 0):
    """Adaptive crop state'i sıfırla."""
    global _ADAPTIVE_CROP_STATE
    _ADAPTIVE_CROP_STATE = {
        'per_image': {},
        'eval_count': 0,
        'n': n,
        'warmup_complete': False
    }

def get_adaptive_crop_state():
    """Mevcut adaptive crop state'ini döndür."""
    return dict(_ADAPTIVE_CROP_STATE)

def is_in_warmup() -> bool:
    """Warmup döneminde miyiz? (bbox henüz kilitlenmemiş)"""
    if _ADAPTIVE_CROP_STATE['n'] == 0:
        return False  # Adaptive crop devre dışı
    return not _ADAPTIVE_CROP_STATE['warmup_complete']

# ============================================================================
# PARAMETER DEFINITIONS
# ============================================================================

PARAMETER_BOUNDS = {
    'base_gray': (0.02, 0.60),
    'tex_mix': (0.0, 1.0),
    'oren_rough': (0.0, 1.0),
    'princ_rough': (0.0, 1.0),
    'shader_mix': (0.0, 1.0),
    'ior': (1.33, 1.79),
    'q_eff': (0.7, 1.4),
    'threshold_value': (0.0, 0.1),
    'mars_base_gray': (0.02, 0.60),
    'mars_tex_mix': (0.0, 1.0),
    'mars_oren_rough': (0.0, 1.0),
    'mars_princ_rough': (0.0, 1.0),
    'mars_shader_mix': (0.0, 1.0),
    'mars_ior': (0.5, 5.0),
    'mars_albedo_mul': (0.2, 5.0),
    # Atmosphere parameters
    'atm_beta0': (1e-8, 1e-2),
    'atm_scale_height': (0.0, 120.0),   
    'atm_anisotropy': (-1.0, 1.0),
    'atm_color_r': (0.3, 1.0),
    'atm_color_g': (0.2, 0.8),
    'atm_color_b': (0.1, 0.6),
}

PARAMETER_NAMES = list(PARAMETER_BOUNDS.keys())
N_PARAMS = len(PARAMETER_NAMES)


# ============================================================================
# FIXED-VARIABLE MECHANISM
# ============================================================================

# Parameters in this dict are fixed at the given values and excluded from
# the optimizer search space.  Set via set_fixed_params().
FIXED_PARAMS: Dict[str, float] = {}


def set_fixed_params(fixed: Dict[str, float]):
    """Fix specific params at given values — they won't be optimized.

    Example:
        set_fixed_params({'atm_beta0': 3e-4, 'atm_scale_height': 11.0})
    """
    global FIXED_PARAMS
    FIXED_PARAMS = dict(fixed)
    print(f"  🔒 Fixed params ({len(FIXED_PARAMS)}): {FIXED_PARAMS}")


def get_active_bounds() -> Dict[str, tuple]:
    """Return bounds only for non-fixed (optimized) params."""
    return {k: v for k, v in PARAMETER_BOUNDS.items() if k not in FIXED_PARAMS}


def get_active_names() -> List[str]:
    """Return names of optimized (non-fixed) params."""
    return [k for k in PARAMETER_NAMES if k not in FIXED_PARAMS]


def get_active_n_params() -> int:
    """Return number of optimized (non-fixed) params."""
    return len(get_active_names())


def active_to_full(active_array: np.ndarray) -> np.ndarray:
    """Map reduced-dimension optimizer vector back to full param array.

    Fixed values are injected at their canonical positions.
    """
    full = np.zeros(N_PARAMS)
    active_names = get_active_names()
    active_idx = 0
    for i, name in enumerate(PARAMETER_NAMES):
        if name in FIXED_PARAMS:
            full[i] = FIXED_PARAMS[name]
        else:
            full[i] = active_array[active_idx]
            active_idx += 1
    return full


# ============================================================================
# PARAMETER UTILITIES
# ============================================================================

def params_to_dict(params_array: np.ndarray) -> Dict[str, float]:
    """Convert parameter array to dictionary."""
    return {name: float(val) for name, val in zip(PARAMETER_NAMES, params_array)}


def dict_to_params(params_dict: Dict[str, float]) -> np.ndarray:
    """Convert parameter dictionary to array."""
    return np.array([params_dict[name] for name in PARAMETER_NAMES])


def clip_params(params: np.ndarray) -> np.ndarray:
    """Clip parameters to valid bounds."""
    clipped = params.copy()
    for i, name in enumerate(PARAMETER_NAMES):
        lower, upper = PARAMETER_BOUNDS[name]
        clipped[i] = np.clip(params[i], lower, upper)
    return clipped


def get_bounds_arrays() -> Tuple[np.ndarray, np.ndarray]:
    """Get lower and upper bounds as arrays (active params only)."""
    active = get_active_bounds()
    lower = np.array([v[0] for v in active.values()])
    upper = np.array([v[1] for v in active.values()])
    return lower, upper


def get_full_bounds_arrays() -> Tuple[np.ndarray, np.ndarray]:
    """Get lower and upper bounds for ALL params (including fixed)."""
    lower = np.array([PARAMETER_BOUNDS[name][0] for name in PARAMETER_NAMES])
    upper = np.array([PARAMETER_BOUNDS[name][1] for name in PARAMETER_NAMES])
    return lower, upper


def random_params(n_samples: int = 1, seed: Optional[int] = None) -> np.ndarray:
    """Generate random parameter sets within bounds (active params only)."""
    if seed is not None:
        np.random.seed(seed)
    
    lower, upper = get_bounds_arrays()
    n_active = get_active_n_params()
    return np.random.uniform(lower, upper, size=(n_samples, n_active))


def print_params_table(params: np.ndarray):
    """Print parameters in a formatted table.
    
    Accepts either active-dim or full-dim arrays.
    Fixed params are marked with 🔒.
    """
    if params is None:
        print("\n  ⚠️ No valid parameters found (all iterations were warmup?)\n")
        return
    n_active = get_active_n_params()
    if len(params) == n_active and n_active < N_PARAMS:
        full_params = active_to_full(params)
    else:
        full_params = params
    
    params_dict = params_to_dict(full_params)
    
    print("\n" + "="*60)
    print("Parameter Values")
    print("="*60)
    print(f"{'Parameter':<20} {'Value':>15} {'Bounds':>20}")
    print("-"*60)
    
    for name in PARAMETER_NAMES:
        val = params_dict[name]
        lower, upper = PARAMETER_BOUNDS[name]
        marker = " 🔒" if name in FIXED_PARAMS else ""
        print(f"{name:<20} {val:>15.6f} [{lower:.2f}, {upper:.2f}]{marker}")
    
    print("="*60 + "\n")


# ============================================================================
# OBJECTIVE FUNCTION WITH RENDERING
# ============================================================================

def evaluate_params_with_rendering(params,
                                   img_info_list,
                                   objective_type,
                                   mode,
                                   aggregation,
                                   particle_id,
                                   temp_dir,
                                   verbose=False,
                                   # YENİ: k araması için opsiyonlar
                                   k_mode: str = 'learned',
                                   learned_k_db: Optional[Dict[str, int]] = None,
                                   # YENİ: Blender kalıcılık/batch
                                   blender_persistent: bool = True,
                                   blender_batch_size: Optional[int] = None,
                                   # YENİ: Adaptive crop
                                   adaptive_crop_n: int = 0,
                                   # YENİ: Atmosphere & Displacement
                                   use_displacement: bool = True,
                                   use_atmosphere: bool = True,
                                   atm_color_mode: str = 'single',
                                   dem_path: Optional[str] = None,
                                   fixed_k: Optional[int] = None):
    """
    Evaluate parameters by:
    1. Rendering synthetic images with CORTO
    2. Comparing with real IMG files
    
    Args:
        params: Parameter array [10 values]
        img_info_list: List of IMG info dicts
        objective_type: 'ssim', 'rmse', 'nmrse', 'combined'
        mode: 'cropped' or 'uncropped'
        aggregation: 'mean', 'median', 'max'
        particle_id: Particle ID
        temp_dir: Temp directory
        verbose: Verbose output
        
    Returns:
        Objective value (lower is better)
    """
    global render_synthetic_for_params
    
    # ============ IMPORT'LAR BURAYA ============
    from PIL import Image
    import numpy as np
    import shutil
    import json
    from pathlib import Path
    from pds_processor import load_pds_image_data, filter_hot_pixels, filter_low_dn, fill_nan_cardinal, preclip_percentile
    from metrics_evaluator import normalize_image, compute_metrics, evaluate_k
    from quick_evaluator import extract_pds_clip_params
    # ==========================================
    
    # Lazy import to avoid circular dependency
    if render_synthetic_for_params is None:
        from corto_renderer import render_synthetic_for_params as render_func
        render_synthetic_for_params = render_func
    
    try:
        params_clipped = clip_params(params)
        params_dict = params_to_dict(params_clipped)
        threshold_value = float(params_dict.get('threshold_value', 0.0))        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating Particle {particle_id}")
            print(f"{'='*60}")
            print_params_table(params_clipped)
        
        # Temp dir
        if temp_dir is None:
            temp_dir = Path("optimization_temp")
        # Mutlak yol kullan
        temp_dir = Path(temp_dir).resolve()
        particle_temp_dir = temp_dir / f"particle_{particle_id:03d}"
        particle_temp_dir.mkdir(parents=True, exist_ok=True)
        
        # RENDER synthetic images
        if verbose:
            print(f"\nRendering {len(img_info_list)} synthetic images...")
        
        synthetic_files = render_synthetic_for_params(
            img_info_list=img_info_list,
            params=params_clipped,
            output_dir=particle_temp_dir,
            particle_id=particle_id,
            # YENİ: kalıcılık/batch parametreleri aşağıya iletiliyor
            persistent=blender_persistent,
            batch_size=blender_batch_size,
            # YENİ: Atmosphere & Displacement
            use_displacement=use_displacement,
            use_atmosphere=use_atmosphere,
            atm_color_mode=atm_color_mode,
            dem_path=dem_path
        )
        
        if None in synthetic_files:
            print(f"Error: Some renders failed for particle {particle_id}")
            return np.inf
        
        # ============ CLIP SYNTHETIC TO PDS DIMENSIONS ============
        # Her görüntü için kendi PDS clip parametreleri kullanılır.
        if 'pds_path' in img_info_list[0]:
            if verbose:
                print(f"\nClipping synthetic images to PDS dimensions...")
            
            clipped_files = []
            for synth_file, img_info_clip in zip(synthetic_files, img_info_list):
                # Her görüntü için kendi PDS clip parametreleri
                pds_params = extract_pds_clip_params(Path(img_info_clip['pds_path']))
                
                if verbose:
                    print(f"  PDS Clip [{img_info_clip['filename']}]:")
                    print(f"    col_off={pds_params['col_off']}, row_off={pds_params['row_off']}")
                    print(f"    width={pds_params['width']}, height={pds_params['height']}")
                
                # Load as uint16 PNG (Blender output)
                synth_img = np.array(Image.open(synth_file))
                
                # Normalize to [0, 1] BEFORE clipping
                if synth_img.dtype == np.uint16:
                    synth_img = synth_img.astype(np.float32) / 65535.0
                elif synth_img.dtype == np.uint8:
                    synth_img = synth_img.astype(np.float32) / 255.0
                else:
                    synth_img = synth_img.astype(np.float32)
                
                # Clip to [0, 1] range
                synth_img = np.clip(synth_img, 0.0, 1.0)
                
                # Clip to PDS dimensions
                col_off = pds_params['col_off']
                row_off = pds_params['row_off']
                width = pds_params['width']
                height = pds_params['height']
                
                synth_clipped = synth_img[row_off:row_off+height, col_off:col_off+width]


                # Step 6: threshold low DN values to zero
                synth_clipped = np.where(
                    synth_clipped < threshold_value,
                    np.float32(0.0),
                    synth_clipped
                )

                # Step 7: re-normalize positive DN values to [0, 1]
                positive_mask = synth_clipped > 0.0
                if np.any(positive_mask):
                    positive_vals = synth_clipped[positive_mask]
                    min_val = float(positive_vals.min())
                    max_val = float(positive_vals.max())
                    if max_val > min_val:
                        synth_clipped = synth_clipped.copy()
                        synth_clipped[positive_mask] = (
                            (positive_vals - min_val) / (max_val - min_val)
                        )
                    else:
                        synth_clipped = synth_clipped.copy()
                        synth_clipped[positive_mask] = 1.0
                else:
                    synth_clipped = np.zeros_like(synth_clipped, dtype=np.float32)

                synth_clipped = np.asarray(synth_clipped, dtype=np.float32)


                
                # Save as float32 TIFF in [0,1] range
                clipped_path = synth_file.parent / f"{synth_file.stem}_clipped.tiff"
                Image.fromarray(synth_clipped, mode='F').save(clipped_path, format='TIFF')
                clipped_files.append(clipped_path)
                
                if verbose:
                    print(f"  Clipped {synth_file.name}: {synth_img.shape} → {synth_clipped.shape}")
                    print(f"    Range: [{synth_clipped.min():.4f}, {synth_clipped.max():.4f}]")
            
            synthetic_files = clipped_files
        # =============================================================
        
        # EVALUATE each pair
        objectives = []
        results_list = []  # Store results for later inspection
        is_warmup_iteration = False  # Warmup flag — döngü sonunda kontrol edilecek
        
        for i, (img_info, synthetic_file) in enumerate(zip(img_info_list, synthetic_files)):
            if verbose:
                print(f"\nEvaluating pair {i+1}/{len(img_info_list)}:")
                print(f"  Real: {img_info['filename']}")
                print(f"  Synthetic: {synthetic_file.name}")
            
            # --- metrics_evaluator.evaluate_k kullan ---
            # Real görüntüyü oku ve hot-pixel filtresi uygula
            raw_img, _ = load_pds_image_data(Path(img_info['pds_path']))
            filtered_img, _, _ = filter_hot_pixels(raw_img)
            # DN ≤ 1 → NaN (dead pixels)
            filtered_img = filter_low_dn(filtered_img, threshold=1)
            # P1-P99 clip + [0,1] normalize (NaN korunur)
            filtered_img = preclip_percentile(filtered_img, low_p=1.0, high_p=99.0)
            # 4-yönlü NaN doldurma ([0,1] data üzerinde)
            filtered_img = fill_nan_cardinal(filtered_img)
            
            # NOTE: real image from load_pds_image_data is already LINES×LINE_SAMPLES.
            # LINE_FIRST_PIXEL/SAMPLE_FIRST_PIXEL are sensor frame offsets, not within-data offsets.
            # Synthetic is also clipped to same PDS dimensions earlier (line 387-393).
            # Both images should be the same shape here — no additional clipping needed.
            
            # Save filled version for visual inspection
            _filled_path = synthetic_file.parent / f"{Path(img_info['filename']).stem}_real_filled.tif"
            try:
                from PIL import Image as _PILImage
                # filtered_img is float32 [0,1] — save as 32-bit TIFF
                _PILImage.fromarray(filtered_img, mode='F').save(str(_filled_path))
                print(f"  💾 Saved filled real image: {_filled_path.name}")
            except Exception as _e:
                print(f"  ⚠️ Could not save filled image: {_e}")
            # Synthetic'i float32 olarak yükle (zaten [0,1] kaydedildi)
            synth_img = np.array(Image.open(synthetic_file), dtype=np.float32)
            synth_img = np.clip(synth_img, 0.0, 1.0)

            img_stem = Path(img_info['filename']).stem
            
            # ============ ADAPTIVE CROP LOGIC (PER-IMAGE) ============
            global _ADAPTIVE_CROP_STATE
            fixed_bbox = None
            
            # Fixed K value (passed as argument)
            fixed_k_val = fixed_k
            
            # Adaptive crop aktifse ve mode='cropped' ise
            if adaptive_crop_n > 0 and mode == 'cropped':
                # State'i başlat (ilk kez çağrılıyorsa)
                if _ADAPTIVE_CROP_STATE['n'] == 0:
                    _ADAPTIVE_CROP_STATE['n'] = adaptive_crop_n
                
                # Per-image state al veya oluştur
                img_crop_state = _ADAPTIVE_CROP_STATE['per_image'].setdefault(
                    img_stem, {'history': [], 'locked_bbox': None}
                )
                
                # Bu IMG'nin kilitli bbox'ı varsa kullan
                if img_crop_state['locked_bbox'] is not None:
                    fixed_bbox = img_crop_state['locked_bbox']
                    if verbose or _ADAPTIVE_CROP_STATE['eval_count'] == adaptive_crop_n:
                        print(f"  🔒 Using locked bbox for {img_stem}: {fixed_bbox}")
            
            eval_res = evaluate_k(
                real_raw=filtered_img,
                synthetic_norm=synth_img,
                img_stem=img_stem,
                mode=mode,                      # "cropped" | "uncropped"
                buffer_percent=5.0,
                verbose=verbose,
                k_mode=k_mode,
                learned_k_db=learned_k_db,
                fixed_bbox=fixed_bbox,           # Bu IMG'ye özgü kilitli bbox
                fixed_k=fixed_k_val,             # Manuel k değeri
                objective_type=objective_type,    # Objective-aware k seçimi
            )
            
            # Bbox'ı kaydet (henüz kilitlenmemişse) — PER-IMAGE
            if adaptive_crop_n > 0 and mode == 'cropped' and not _ADAPTIVE_CROP_STATE['warmup_complete']:
                img_crop_state = _ADAPTIVE_CROP_STATE['per_image'].setdefault(
                    img_stem, {'history': [], 'locked_bbox': None}
                )
                # eval_res'ten bbox al (cropped modda döner)
                if 'crop_bbox' in eval_res and eval_res['crop_bbox'] is not None:
                    bbox = eval_res['crop_bbox']
                    bbox_h = bbox[1] - bbox[0]  # y1 - y0
                    bbox_w = bbox[3] - bbox[2]  # x1 - x0
                    if bbox_h >= 100 and bbox_w >= 100:
                        img_crop_state['history'].append(bbox)
                        print(f"  🔄 WARMUP [{img_stem}]: Learning bbox {bbox} (sample #{len(img_crop_state['history'])})")
                    else:
                        print(f"  ⚠️ WARMUP [{img_stem}]: Skipping garbage bbox {bbox} ({bbox_h}×{bbox_w} px)")
                
                # Eval count'u sadece son görüntüde artır (her tam döngü = 1 eval)
                if i == len(img_info_list) - 1:
                    _ADAPTIVE_CROP_STATE['eval_count'] += 1
                    print(f"  🔄 WARMUP eval {_ADAPTIVE_CROP_STATE['eval_count']}/{adaptive_crop_n}")
                    
                    # N'e ulaştıysak kilitle — tüm IMG'ler için
                    if _ADAPTIVE_CROP_STATE['eval_count'] >= adaptive_crop_n:
                        for stem, state in _ADAPTIVE_CROP_STATE['per_image'].items():
                            if len(state['history']) > 0:
                                state['locked_bbox'] = _find_median_bbox(state['history'])
                                print(f"  🔒 Bbox locked for {stem}: {state['locked_bbox']} (from {len(state['history'])} samples)")
                            else:
                                # All warmup bboxes were garbage — use None (no fixed bbox, center-crop fallback in evaluator)
                                state['locked_bbox'] = None
                                print(f"  ⚠️ No valid warmup bboxes for {stem} — will use dynamic crop with center fallback")
                        _ADAPTIVE_CROP_STATE['warmup_complete'] = True
                        
                        # Tüm IMG'ler için learned K finalize
                        if learned_k_db is not None:
                            from metrics_evaluator import finalize_learned_k
                            for info in img_info_list:
                                stem = Path(info['filename']).stem
                                finalize_learned_k(stem, learned_k_db)
                                print(f"  🧠 Learned k finalized for {stem}: {learned_k_db.get(stem, 'Not found')}")

                        print(f"\n  🔒 WARMUP COMPLETE!")
                        print(f"     ⚠️ Warmup scores will NOT be used for optimization!\n")
                
                # Warmup döneminde bu IMG'nin objective'ini sayma
                is_warmup_iteration = True
                continue  # Diğer görüntülere devam et
            # =============================================
            
            results_list.append(eval_res)  # Save for inspection

            # Calculate objective
            if objective_type == 'ssim':
                obj_val = -float(eval_res['best_ssim_score'])  # minimize
            elif objective_type == 'rmse':
                obj_val = float(eval_res['best_rmse_score'])
            elif objective_type == 'nmrse':
                obj_val = float(eval_res['best_rmse_score']) / 1.0
            elif objective_type == 'combined':
                ssim_best = float(eval_res['best_ssim_score'])
                rmse_best = float(eval_res['best_rmse_score'])
                obj_val = (1.0 - ssim_best) + rmse_best*(1/3) #weigh is 1/3 for rmse
            elif objective_type == 'combined-new':
                ms_ssim_best  = float(eval_res.get('best_ms_ssim', 0.0))
                rmse_best     = float(eval_res['best_rmse_score'])
                gmsd_best     = float(eval_res.get('best_gmsd', 0.15))
                ncc_best      = float(eval_res.get('best_ncc', 0.0))
                lpips_best    = float(eval_res.get('best_lpips', 0.5))
                # Weights:
                #   MS-SSIM: multi-scale structural (1.0)
                #   NMRSE:   pixel-level diff       (0.33)
                #   GMSD:    gradient texture        (0.33)
                #   NCC:     pattern (brightness-invariant) (-0.10 because higher=better)
                #   LPIPS:   perceptual              (+0.10 because lower=better)
                ncc_contrib  = 0.0 if not np.isfinite(ncc_best)  else -0.30 * ncc_best
                lpips_contrib = 0.0 if not np.isfinite(lpips_best) else  0.5 * lpips_best
                obj_val = ((1.0 - ms_ssim_best)
                           + rmse_best * (1/3)
                           + gmsd_best * (1/3)
                           + ncc_contrib
                           + lpips_contrib)
            else:
                raise ValueError(f"Unknown objective: {objective_type}")
            
            # ---- ORB PENALTY (tiered) ----
            orb_penalty = eval_res.get('orb_penalty', 0.0)
            if orb_penalty > 0:
                obj_val += orb_penalty
                print(f"  🚫 ORB penalty +{orb_penalty:.1f}: obj {obj_val - orb_penalty:.4f} → {obj_val:.4f}")
            
            objectives.append(obj_val)
            
            if verbose:
                print(f"  Objective: {obj_val:.6f}")
        
        # ============ WARMUP CHECK ============
        # Warmup döneminde tüm görüntüler değerlendirildi ama skorlar kullanılmayacak
        if is_warmup_iteration:
            return float('inf')
        # =======================================
        
        # AGGREGATE
        objectives = np.array(objectives)
        
        if aggregation == 'mean':
            final_obj = float(np.mean(objectives))
        elif aggregation == 'median':
            final_obj = float(np.median(objectives))
        elif aggregation == 'max':
            final_obj = float(np.max(objectives))
        elif aggregation == 'rss':
            # Root-sum-of-squares: sqrt(sum(obj_i^2))
            # 1 image: sqrt(x^2) = x  (identical to mean)
            # N images: L2-norm, larger errors penalized more than mean
            final_obj = float(np.sqrt(np.sum(objectives ** 2)))
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        if verbose:
            print(f"\nFinal Objective: {final_obj:.6f}\n")
        # ---- Aggregate metrics for logging wrapper (stash globally) ----
        try:
            # results_list elemanları evaluate_k çıktısı
            ssim_vals = [float(r.get('best_ssim_score', np.nan)) for r in results_list]
            rmse_vals = [float(r.get('best_rmse_score', np.nan)) for r in results_list]

            # SSIM ve NMRSE (RMSE, veri aralığı [0,1] olduğundan 1.0'a bölmek yeterli)
            ssim_mean  = float(np.nanmean(np.array(ssim_vals))) if len(ssim_vals) else float('nan')
            nmrse_mean = float(np.nanmean(np.array(rmse_vals))) if len(rmse_vals) else float('nan')

            # Additional metrics
            ms_ssim_vals = [float(r.get('best_ms_ssim', np.nan)) for r in results_list]
            gmsd_vals = [float(r.get('best_gmsd', np.nan)) for r in results_list]
            # hist_corr: from all_results[0] if available
            hist_corr_vals = []
            for r in results_list:
                ar = r.get('all_results', [])
                if ar and 'hist_corr' in ar[0]:
                    hist_corr_vals.append(float(ar[0]['hist_corr']))
                else:
                    hist_corr_vals.append(float('nan'))
            ncc_vals  = [float(r.get('best_ncc',   np.nan)) for r in results_list]
            lpips_vals = [float(r.get('best_lpips', np.nan)) for r in results_list]
            orb_penalty_vals = [float(r.get('orb_penalty', 0.0)) for r in results_list]

            ms_ssim_mean  = float(np.nanmean(np.array(ms_ssim_vals)))  if len(ms_ssim_vals)  else float('nan')
            gmsd_mean     = float(np.nanmean(np.array(gmsd_vals)))     if len(gmsd_vals)     else float('nan')
            hist_corr_mean = float(np.nanmean(np.array(hist_corr_vals))) if len(hist_corr_vals) else float('nan')
            ncc_mean      = float(np.nanmean(np.array(ncc_vals)))      if len(ncc_vals)      else float('nan')
            lpips_mean    = float(np.nanmean(np.array(lpips_vals)))    if len(lpips_vals)    else float('nan')
            orb_penalty_mean = float(np.mean(orb_penalty_vals)) if len(orb_penalty_vals) else 0.0

            set_last_eval_metrics(
                nmrse=nmrse_mean, ssim=ssim_mean,
                ms_ssim=ms_ssim_mean, gmsd=gmsd_mean,
                hist_corr=hist_corr_mean, orb_penalty=orb_penalty_mean,
                ncc=ncc_mean, lpips=lpips_mean
            )
        except Exception:
            # Logging hiçbir zaman objective akışını bozmasın
            pass        
        
        
        # ============ SAVE BEST K IMAGES FOR INSPECTION ============
        try:
            from metrics_evaluator import align_and_crop_with_buffer
            
            # Create inspection directory
            inspection_dir = particle_temp_dir / "best_k_inspection"
            inspection_dir.mkdir(parents=True, exist_ok=True)
            
            # For each IMG, save best k images
            for i, (img_info, synthetic_file, result_dict) in enumerate(zip(img_info_list, synthetic_files, results_list)):
                # result_dict artık evaluate_k çıktısı
                
                img_stem = Path(img_info['filename']).stem
                
                # ===== RELOAD SYNTHETIC (already PDS-clipped) =====
                try:
                    # synthetic_file IS _clipped.tiff (PDS clip already applied at line 387-393)
                    # NO second clip needed — loading as-is
                    synth_clipped = np.array(Image.open(synthetic_file), dtype=np.float32)
                    synth_clipped = np.clip(synth_clipped, 0.0, 1.0)
                    
                    print(f"  Synthetic clipped shape: {synth_clipped.shape}")
                    
                except Exception as e:
                    print(f"  ⚠️ Warning: Could not reload synthetic: {e}")
                    continue
                
                # ===== BEST SSIM K =====
                best_ssim_k = result_dict.get('best_ssim_k')
                if best_ssim_k is not None:
                    try:
                        # Use pre-aligned images from evaluate_k (same alignment as SSIM computation)
                        aligned_real = result_dict.get('aligned_real')
                        aligned_synth = result_dict.get('aligned_synth')
                        
                        if aligned_real is not None and aligned_synth is not None:
                            real_norm_ssim = aligned_real
                            synth_for_ssim = aligned_synth
                            
                            # Save alignment info
                            if mode == "cropped":
                                a_shift = result_dict.get('alignment_shift')
                                a_bbox = result_dict.get('crop_bbox')
                                alignment_info = {
                                    'k': int(best_ssim_k),
                                    'metric': 'SSIM',
                                    'shift_dy': float(a_shift[0]) if a_shift is not None else 0.0,
                                    'shift_dx': float(a_shift[1]) if a_shift is not None else 0.0,
                                    'bbox': [int(x) for x in a_bbox] if a_bbox is not None else None,
                                    'cropped_shape': list(real_norm_ssim.shape),
                                    'alignment_method': result_dict.get('alignment_method', 'unknown'),
                                    'note': 'Same alignment used for SSIM computation'
                                }
                                
                                alignment_path = inspection_dir / f"{img_stem}_alignment_k{best_ssim_k:02d}_bestSSIM.json"
                                with open(alignment_path, 'w') as f:
                                    json.dump(alignment_info, f, indent=2)
                                
                                print(f"  ✅ Saved alignment info: {alignment_path.name}")
                        else:
                            # Fallback: re-load and re-align (legacy path for sweep mode)
                            raw_img, _ = load_pds_image_data(Path(img_info['pds_path']))
                            filtered_img, _, _ = filter_hot_pixels(raw_img)
                            filtered_img = filter_low_dn(filtered_img, threshold=1)
                            filtered_img = preclip_percentile(filtered_img, low_p=1.0, high_p=99.0)
                            real_norm_ssim = fill_nan_cardinal(filtered_img)
                            
                            # NOTE: load_pds_image_data already returns LINES×LINE_SAMPLES.
                            # No additional PDS offset clipping needed.
                            
                            print(f"  Real (k={best_ssim_k}) clipped shape: {real_norm_ssim.shape}")
                            
                            if mode == "cropped":
                                alignment_result = align_and_crop_with_buffer(
                                    imgA=real_norm_ssim,
                                    imgB=synth_clipped,
                                    buffer_ratio=0.05,
                                    upsample_factor=100,
                                    interpolation_order=1
                                )
                                real_norm_ssim = alignment_result['A_crop']
                                synth_for_ssim = alignment_result['B_crop']
                                
                                alignment_info = {
                                    'k': int(best_ssim_k),
                                    'metric': 'SSIM',
                                    'shift_dy': float(alignment_result['shift'][0]),
                                    'shift_dx': float(alignment_result['shift'][1]),
                                    'bbox': [int(x) for x in alignment_result['bbox']],
                                    'cropped_shape': list(real_norm_ssim.shape),
                                    'note': 'Re-aligned (fallback path)'
                                }
                                
                                alignment_path = inspection_dir / f"{img_stem}_alignment_k{best_ssim_k:02d}_bestSSIM.json"
                                with open(alignment_path, 'w') as f:
                                    json.dump(alignment_info, f, indent=2)
                                
                                print(f"  ✅ Saved alignment info: {alignment_path.name}")
                            else:
                                synth_for_ssim = synth_clipped
                        
                        # Save CLIPPED+(ALIGNED+CROPPED) real
                        real_ssim_path = inspection_dir / f"{img_stem}_real_norm_k{best_ssim_k:02d}_bestSSIM_{mode}.tif"
                        Image.fromarray(real_norm_ssim, mode='F').save(real_ssim_path, format='TIFF')
                        
                        # Save corresponding synthetic
                        synth_ssim_path = inspection_dir / f"{img_stem}_synthetic_k{best_ssim_k:02d}_bestSSIM_{mode}.tif"
                        Image.fromarray(synth_for_ssim, mode='F').save(synth_ssim_path, format='TIFF')
                        
                        print(f"  ✅ Saved best SSIM pair (k={best_ssim_k}, mode={mode}):")
                        print(f"     Real: {real_ssim_path.name} (shape: {real_norm_ssim.shape})")
                        print(f"     Synth: {synth_ssim_path.name} (shape: {synth_for_ssim.shape})")
                        if mode == "cropped" and 'alignment_info' in dir():
                            print(f"     Shift: dy={alignment_info['shift_dy']:.2f}, dx={alignment_info['shift_dx']:.2f}")
                            
                    except Exception as e:
                        print(f"  ⚠️ Warning: Could not save SSIM images: {e}")
                        import traceback
                        traceback.print_exc()
                
                # ===== BEST RMSE K =====
                best_rmse_k = result_dict.get('best_rmse_k')
                if best_rmse_k is not None and best_rmse_k != best_ssim_k:
                    try:
                        # Load and process real image with Colab pipeline
                        raw_img, _ = load_pds_image_data(Path(img_info['pds_path']))
                        filtered_img, _, _ = filter_hot_pixels(raw_img)
                        filtered_img = filter_low_dn(filtered_img, threshold=1)
                        filtered_img = preclip_percentile(filtered_img, low_p=1.0, high_p=99.0)
                        real_norm_rmse = fill_nan_cardinal(filtered_img)
                        
                        # Clip to PDS dimensions
                        if 'pds_path' in img_info:
                            real_norm_rmse = real_norm_rmse[
                                pds_params['row_off']:pds_params['row_off']+pds_params['height'],
                                pds_params['col_off']:pds_params['col_off']+pds_params['width']
                            ]
                        
                        print(f"  Real (k={best_rmse_k}) clipped shape: {real_norm_rmse.shape}")
                        
                        # Check shape compatibility
                        if real_norm_rmse.shape != synth_clipped.shape:
                            print(f"  ⚠️ Shape mismatch: real {real_norm_rmse.shape} vs synth {synth_clipped.shape}")
                            continue
                        
                        # Apply ALIGNMENT & CROPPING if in cropped mode
                        if mode == "cropped":
                            alignment_result = align_and_crop_with_buffer(
                                imgA=real_norm_rmse,              # ← DOĞRU: RMSE normalizasyonu
                                imgB=synth_clipped,
                                buffer_pixels=50,
                                upsample_factor=100,
                                interpolation_order=1
                            )
                            real_norm_rmse = alignment_result['A_crop']   # ← DOĞRU: RMSE değişkenine yaz
                            synth_for_rmse = alignment_result['B_crop']   # ← DOĞRU: RMSE sentetiği
                            # Save alignment info
                            alignment_info = {
                                'k': int(best_rmse_k),
                                'metric': 'RMSE',
                                'shift_dy': float(alignment_result['shift'][0]),
                                'shift_dx': float(alignment_result['shift'][1]),
                                'bbox': [int(x) for x in alignment_result['bbox']],
                                'cropped_shape': list(real_norm_rmse.shape)  # ← DOĞRU şekil
                            }
                            alignment_path = inspection_dir / f"{img_stem}_alignment_k{best_rmse_k:02d}_bestRMSE.json"
                            with open(alignment_path, 'w') as f:
                                json.dump(alignment_info, f, indent=2)
                        else:
                            synth_for_rmse = synth_clipped

                        # Save CLIPPED+(ALIGNED+CROPPED) real
                        real_rmse_path = inspection_dir / f"{img_stem}_real_norm_k{best_rmse_k:02d}_bestRMSE_{mode}.tif"
                        Image.fromarray(real_norm_rmse, mode='F').save(real_rmse_path, format='TIFF')

                        # Save corresponding synthetic
                        synth_rmse_path = inspection_dir / f"{img_stem}_synthetic_k{best_rmse_k:02d}_bestRMSE_{mode}.tif"
                        Image.fromarray(synth_for_rmse, mode='F').save(synth_rmse_path, format='TIFF')
                        
                        print(f"  ✅ Saved best RMSE pair (k={best_rmse_k}, mode={mode}):")
                        print(f"     Real: {real_rmse_path.name} (shape: {real_norm_rmse.shape})")
                        print(f"     Synth: {synth_rmse_path.name} (shape: {synth_for_rmse.shape})")
                        if mode == "cropped":
                            print(f"     Shift: dy={alignment_info['shift_dy']:.2f}, dx={alignment_info['shift_dx']:.2f}")
                            
                    except Exception as e:
                        print(f"  ⚠️ Warning: Could not save RMSE images: {e}")
                        import traceback
                        traceback.print_exc()
                
                # ===== COPY SYNTHETIC CLIPPED (FOR REFERENCE) =====
                try:
                    synth_copy_path = inspection_dir / f"{img_stem}_synthetic_clipped.tif"
                    Image.fromarray(synth_clipped, mode='F').save(synth_copy_path, format='TIFF')
                    print(f"  ✅ Saved synthetic clipped: {synth_copy_path.name}")
                except Exception as e:
                    print(f"  ⚠️ Warning: Could not copy synthetic: {e}")
                
                # ===== SAVE METADATA =====
                try:
                    metadata = {
                        'particle_id': particle_id,
                        'img_filename': img_info['filename'],
                        'best_ssim_k': int(best_ssim_k) if best_ssim_k is not None else None,
                        'best_ssim_score': float(result_dict.get('best_ssim_score', np.nan)),
                        'best_rmse_k': int(best_rmse_k) if best_rmse_k is not None else None,
                        'best_rmse_score': float(result_dict.get('best_rmse_score', np.nan)),
                        'mode': mode,
                        'objective_type': objective_type,
                        'final_objective': float(final_obj)
                    }
                    
                    metadata_path = inspection_dir / f"{img_stem}_metadata.json"
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    print(f"  ✅ Saved metadata: {metadata_path.name}")
                except Exception as e:
                    print(f"  ⚠️ Warning: Could not save metadata: {e}")
            
            print(f"\n✅ Best k inspection images saved to: {inspection_dir}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not save best k images: {e}")
            import traceback
            traceback.print_exc()
        # ===========================================================
        # ============ CLEANUP TO SAVE RAM (YENİ) ============
        try:
            # Synthetic PNG'leri sil (evaluation sonrası gerek yok)
            for synth_file in synthetic_files:
                if synth_file and Path(synth_file).exists():
                    Path(synth_file).unlink()
            
            # Clipped synthetic'leri de sil
            for synth_file in synthetic_files:
                clipped = Path(synth_file).parent / f"{Path(synth_file).stem}_clipped.tiff"
                if clipped.exists():
                    clipped.unlink()
                    
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")
        # ===================================================
        
        return final_obj

        
    except Exception as e:
        print(f"Error evaluating particle {particle_id}: {e}")
        import traceback
        traceback.print_exc()
        return np.inf
# ============================================================================
# OBJECTIVE WRAPPER: ITER/PARTICLE LOG + OPTIONAL CSV
# ============================================================================
def with_eval_logging(objective_fn: Callable,
                      pop_size: int,
                      img_info_list: list,
                      csv_dir: Optional[Union[Path, str]] = None,
                      start_eval_idx: int = 0):
    """
    objective_fn: mevcut objective (params -> float)
    pop_size    : her iterasyondaki parçacık sayısı (swarm/population)
    csv_dir     : CSV yazılacak klasör (None ise sadece konsol)
    start_eval_idx: Resume için başlangıç indeksi (checkpoint'tan)

    Kullanım:
        obj_wrapped = with_eval_logging(objective, pop_size, csv_dir="optimization_temp")
        result = optimizer.minimize(obj_wrapped, ...)
    """
    eval_idx = start_eval_idx
    csv_f = None
    xlsx_path = None
    records = []  # XLSX için kayıtlar
    
    if csv_dir:
        csv_dir = Path(csv_dir)
        csv_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = csv_dir / f"eval_history_{stamp}.csv"
        xlsx_path = csv_dir / f"eval_history_{stamp}.xlsx"
        
        csv_f = open(csv_path, "w", buffering=1)
        # Dynamic header from PARAMETER_NAMES
        param_cols = ",".join(PARAMETER_NAMES)
        header = (f"particle_no,iteration,particle,objective,"
                  f"nmrse,ssim,ms_ssim,gmsd,ncc,lpips,hist_corr,orb_penalty,"
                  f"{param_cols},img_files")
        print(header, file=csv_f)
    
    # IMG dosya adlarını birleştir
    img_names = "; ".join([info['filename'] for info in img_info_list])

    def wrapped(x):
        nonlocal eval_idx, csv_f, records
        ps = max(1, int(pop_size))
        it  = eval_idx // ps
        pid = eval_idx %  ps

        if pid == 0:
            print(f"\n{'='*100}")
            print(f"=== Iteration {it} ===")
            print(f"{'='*100}")

        # Map active-dim optimizer vector → full-dim param array
        x_full = active_to_full(np.asarray(x))
        val = objective_fn(x_full)

        # WARMUP döneminde (val=inf) loglama yapma
        if val == float('inf'):
            print(f"  ⏭️ WARMUP iteration - skipping score logging")
            eval_idx += 1
            return val

        try:
            m = get_last_eval_metrics()
            nmrse     = float(m.get("nmrse",     float("nan")))
            ssim      = float(m.get("ssim",      float("nan")))
            ms_ssim   = float(m.get("ms_ssim",   float("nan")))
            gmsd      = float(m.get("gmsd",      float("nan")))
            hist_corr = float(m.get("hist_corr", float("nan")))
            orb_pen   = float(m.get("orb_penalty", 0.0))
            ncc_m     = float(m.get("ncc",   float("nan")))
            lpips_m   = float(m.get("lpips", float("nan")))
        except Exception:
            nmrse, ssim = float("nan"), float("nan")
            ms_ssim, gmsd, hist_corr, orb_pen = float("nan"), float("nan"), float("nan"), 0.0
            ncc_m, lpips_m = float("nan"), float("nan")

        # Parametreleri al (full-dim)
        params_dict = params_to_dict(x_full)

        # Global particle number (monotonically increasing)
        particle_no = eval_idx

        # Konsol çıktısı - dynamic
        print(f"[it {it:03d} | p{pid:03d}] obj={val:+.6f}  NMRSE={nmrse:.6f}  SSIM={ssim:.6f}  NCC={ncc_m:.4f}  LPIPS={lpips_m:.4f}")
        param_str = ", ".join(f"{k}={v:.4f}" for k, v in params_dict.items())
        print(f"  Params: {param_str}")
        print(f"  IMG Files: {img_names}")

        # CSV kaydı
        if csv_f:
            param_vals = ",".join(str(params_dict[k]) for k in PARAMETER_NAMES)
            csv_line = (f"{particle_no},{it},{pid},{val},"
                        f"{nmrse},{ssim},{ms_ssim},{gmsd},{ncc_m},{lpips_m},"
                        f"{hist_corr},{orb_pen},{param_vals},\"{img_names}\"")
            print(csv_line, file=csv_f)

            # XLSX için kayıt
            record = {
                'particle_no': particle_no,
                'iteration': it,
                'particle': pid,
                'objective': val,
                'nmrse': nmrse,
                'ssim': ssim,
                'ms_ssim': ms_ssim,
                'gmsd': gmsd,
                'ncc': ncc_m,
                'lpips': lpips_m,
                'hist_corr': hist_corr,
                'orb_penalty': orb_pen,
                **params_dict,
                'img_files': img_names
            }
            records.append(record)
        
        eval_idx += 1
        return val
    
    # XLSX kaydetme fonksiyonu
    def save_xlsx():
        if xlsx_path and records:
            try:
                df = pd.DataFrame(records)
                df.to_excel(xlsx_path, index=False, engine='openpyxl')
                print(f"\n✅ XLSX saved: {xlsx_path}")
            except Exception as e:
                print(f"⚠️ XLSX save failed: {e}")
    
    # Cleanup için wrapper'a attribute ekle
    wrapped.save_xlsx = save_xlsx
    wrapped.csv_file = csv_f
    
    return wrapped

# ============================================================================
# OPTIMIZATION LOGGER
# ============================================================================

class OptimizationLogger:
    """Logger for optimization progress."""
    
    def __init__(self, output_dir: Path, experiment_name: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.iteration_history = []
        self.best_history = []
        self.config = {}
        
        self.log_file = self.output_dir / f"{experiment_name}_{self.timestamp}.log"
        self.csv_file = self.output_dir / f"{experiment_name}_{self.timestamp}_history.csv"
        self.best_file = self.output_dir / f"{experiment_name}_{self.timestamp}_best.json"
        
        self.log(f"Optimization Logger initialized: {experiment_name}")
    
    def log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        # ---- FIX: UTF-8 + errors='replace' ile Windows cp1254 sorunu çözülür
        with open(self.log_file, 'a', encoding='utf-8', errors='replace') as f:
            f.write(log_msg + '\n')
    
    
    def set_config(self, config: Dict):
        """Store configuration."""
        self.config = config
        self.log(f"Configuration set")
    
    def log_iteration(self, iteration: int, best_objective: float, 
                     best_params: np.ndarray, **kwargs):
        """Log iteration results."""
        params_dict = params_to_dict(best_params)
        
        record = {
            'iteration': iteration,
            'best_objective': best_objective,
            **params_dict,
            **kwargs
        }
        
        self.iteration_history.append(record)
        self.log(f"Iteration {iteration}: Best = {best_objective:.6f}")
    
    def update_best(self, objective: float, params: np.ndarray, **kwargs):
        """Update best result."""
        params_dict = params_to_dict(params)
        
        best_result = {
            'objective': objective,
            'parameters': params_dict,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }
        
        self.best_history.append(best_result)
        
        with open(self.best_file, 'w') as f:
            json.dump({
                'config': self.config,
                'best_result': best_result,
                'history': self.best_history
            }, f, indent=2)
        
        self.log(f"New best! Objective: {objective:.6f}")

        particle_id = kwargs.get('particle_id')
        if particle_id is not None:
            try:
                from corto_renderer import save_debug_scene

                debug_dir = self.output_dir / "debug"
                debug_path = save_debug_scene(debug_dir, int(particle_id))
                if debug_path is not None:
                    self.log(f"Debug blend saved to {debug_path}")
            except Exception as exc:
                self.log(f"Warning: Failed to save debug blend: {exc}")
    
    def save_history(self):
        """Save history to CSV."""
        if not self.iteration_history:
            return
        
        df = pd.DataFrame(self.iteration_history)
        df.to_csv(self.csv_file, index=False)
        self.log(f"History saved to {self.csv_file}")
    
    def plot_convergence(self, save_path: Optional[Path] = None):
        """Plot convergence curve."""
        if not self.iteration_history:
            return
        
        df = pd.DataFrame(self.iteration_history)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['iteration'], df['best_objective'], 'b-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Best Objective Value')
        ax.set_title(f'Convergence: {self.experiment_name}')
        ax.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = self.output_dir / f"{self.experiment_name}_{self.timestamp}_convergence.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log(f"Convergence plot saved")


# ============================================================================
# CONSTRAINT HANDLING
# ============================================================================

def repair_params(params: np.ndarray, method: str = 'clip') -> np.ndarray:
    """Repair parameters that violate constraints."""
    if method == 'clip':
        return clip_params(params)
    
    elif method == 'reflect':
        lower, upper = get_bounds_arrays()
        repaired = params.copy()
        
        for i in range(N_PARAMS):
            while repaired[i] < lower[i]:
                repaired[i] = 2 * lower[i] - repaired[i]
            while repaired[i] > upper[i]:
                repaired[i] = 2 * upper[i] - repaired[i]
        
        return repaired
    
    elif method == 'wrap':
        lower, upper = get_bounds_arrays()
        range_size = upper - lower
        repaired = params.copy()
        
        for i in range(N_PARAMS):
            repaired[i] = lower[i] + np.mod(repaired[i] - lower[i], range_size[i])
        
        return repaired
    
    else:
        raise ValueError(f"Unknown repair method: {method}")


# ============================================================================
# CHECKPOINT MANAGER - Optimizer state'i kaydet ve yükle
# ============================================================================

import pickle

def save_checkpoint(filepath: Union[str, Path],
                   algorithm: str,
                   optimizer_state: dict,
                   best_params: np.ndarray,
                   best_objective: float,
                   iteration_count: int,
                   config: dict,
                   learned_k_db: Optional[dict] = None):
    """
    Optimizer state'ini dosyaya kaydet.
    
    Args:
        filepath: Checkpoint dosya yolu
        algorithm: 'pso', 'genetic', veya 'bayesian'
        optimizer_state: Algoritma-specific state (particles, population, skopt optimizer)
        best_params: En iyi parametreler
        best_objective: En iyi objective değeri
        iteration_count: Tamamlanan iterasyon sayısı
        config: Optimizer config
        learned_k_db: Learned k values database
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'algorithm': algorithm,
        'optimizer_state': optimizer_state,
        'best_params': best_params,
        'best_objective': best_objective,
        'iteration_count': iteration_count,
        'config': config,
        'learned_k_db': learned_k_db,
        'adaptive_crop_state': get_adaptive_crop_state(),
        'timestamp': datetime.now().isoformat(),
        'version': '1.0'
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"  💾 Checkpoint saved: {filepath}")
    print(f"     Iteration: {iteration_count}, Best: {best_objective:.6f}")


def load_checkpoint(filepath: Union[str, Path]) -> dict:
    """
    Checkpoint dosyasından state'i yükle.
    
    Args:
        filepath: Checkpoint dosya yolu
        
    Returns:
        Checkpoint dictionary
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        checkpoint = pickle.load(f)
    
    print(f"  📂 Checkpoint loaded: {filepath}")
    print(f"     Algorithm: {checkpoint['algorithm']}")
    print(f"     Iteration: {checkpoint['iteration_count']}")
    print(f"     Best: {checkpoint['best_objective']:.6f}")
    print(f"     Saved at: {checkpoint['timestamp']}")
    
    # Restore adaptive crop state
    if checkpoint.get('adaptive_crop_state'):
        global _ADAPTIVE_CROP_STATE
        _ADAPTIVE_CROP_STATE.update(checkpoint['adaptive_crop_state'])
        print(f"     Adaptive crop state restored")
    
    return checkpoint


def get_checkpoint_path(temp_dir: Path, iteration: int) -> Path:
    """Checkpoint dosya yolunu oluştur."""
    checkpoint_dir = temp_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return checkpoint_dir / f"checkpoint_iter{iteration}_{timestamp}.pkl"


def find_latest_checkpoint(temp_dir: Path) -> Optional[Path]:
    """En son checkpoint dosyasını bul."""
    checkpoint_dir = temp_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob("checkpoint_*.pkl"))
    if not checkpoints:
        return None
    
    # En son değiştirileni bul
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return latest


if __name__ == "__main__":
    print("Optimization Helper Module")
    print("="*60)
    
    # Test
    params = random_params(1)[0]
    print("\nRandom parameters:")
    print_params_table(params)
    
    # Test logger
    logger = OptimizationLogger(Path("test_output"), "test")
    logger.log_iteration(1, 0.5, params)
    logger.save_history()
    
    print("\n✅ Tests passed!")