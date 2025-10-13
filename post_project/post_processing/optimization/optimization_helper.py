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
_LAST_EVAL_METRICS = {"nmrse": float("nan"), "ssim": float("nan")}

def set_last_eval_metrics(nmrse: float, ssim: float):
    _LAST_EVAL_METRICS["nmrse"] = float(nmrse)
    _LAST_EVAL_METRICS["ssim"] = float(ssim)

def get_last_eval_metrics():
    return dict(_LAST_EVAL_METRICS)

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
}

PARAMETER_NAMES = list(PARAMETER_BOUNDS.keys())
N_PARAMS = len(PARAMETER_NAMES)


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
    """Get lower and upper bounds as arrays."""
    lower = np.array([PARAMETER_BOUNDS[name][0] for name in PARAMETER_NAMES])
    upper = np.array([PARAMETER_BOUNDS[name][1] for name in PARAMETER_NAMES])
    return lower, upper


def random_params(n_samples: int = 1, seed: Optional[int] = None) -> np.ndarray:
    """Generate random parameter sets within bounds."""
    if seed is not None:
        np.random.seed(seed)
    
    lower, upper = get_bounds_arrays()
    return np.random.uniform(lower, upper, size=(n_samples, N_PARAMS))


def print_params_table(params: np.ndarray):
    """Print parameters in a formatted table."""
    params_dict = params_to_dict(params)
    
    print("\n" + "="*60)
    print("Parameter Values")
    print("="*60)
    print(f"{'Parameter':<20} {'Value':>15} {'Bounds':>20}")
    print("-"*60)
    
    for name in PARAMETER_NAMES:
        val = params_dict[name]
        lower, upper = PARAMETER_BOUNDS[name]
        print(f"{name:<20} {val:>15.6f} [{lower:.2f}, {upper:.2f}]")
    
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
                                   blender_batch_size: Optional[int] = None):
    """
    Evaluate parameters by:
    1. Rendering synthetic images with CORTO
    2. Comparing with real IMG files
    
    Args:
        params: Parameter array [7 values]
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
    from pds_processor import load_pds_image_data, filter_hot_pixels
    from metrics_evaluator import normalize_image, compute_metrics, evaluate_k
    from quick_evaluator import extract_pds_clip_params
    # ==========================================
    
    # Lazy import to avoid circular dependency
    if render_synthetic_for_params is None:
        from corto_renderer import render_synthetic_for_params as render_func
        render_synthetic_for_params = render_func
    
    try:
        params_clipped = clip_params(params)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating Particle {particle_id}")
            print(f"{'='*60}")
            print_params_table(params_clipped)
        
        # Temp dir
        if temp_dir is None:
            temp_dir = Path("optimization_temp")
        
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
            batch_size=blender_batch_size
        )
        
        if None in synthetic_files:
            print(f"Error: Some renders failed for particle {particle_id}")
            return np.inf
        
        # ============ CLIP SYNTHETIC TO PDS DIMENSIONS ============
        if 'pds_path' in img_info_list[0]:
            if verbose:
                print(f"\nClipping synthetic images to PDS dimensions...")
            
            pds_params = extract_pds_clip_params(Path(img_info_list[0]['pds_path']))
            
            if verbose:
                print(f"  PDS Clip Parameters:")
                print(f"    col_off={pds_params['col_off']}, row_off={pds_params['row_off']}")
                print(f"    width={pds_params['width']}, height={pds_params['height']}")
            
            clipped_files = []
            for synth_file in synthetic_files:
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
        
        for i, (img_info, synthetic_file) in enumerate(zip(img_info_list, synthetic_files)):
            if verbose:
                print(f"\nEvaluating pair {i+1}/{len(img_info_list)}:")
                print(f"  Real: {img_info['filename']}")
                print(f"  Synthetic: {synthetic_file.name}")
            
            # --- YENİ: metrics_evaluator.evaluate_k kullan ---
            # Real (filtered) ve synthetic (float32 [0,1]) burada dosyadan okunarak
            # evaluate_k içinde normalize/align/crop yapılır.
            # Not: quick_evaluate yerine in-RAM learned_k_db ile hızlanır.
            # Real görüntüyü oku ve hot-pixel filtresi uygula
            raw_img, _ = load_pds_image_data(Path(img_info['pds_path']))
            filtered_img, _, _ = filter_hot_pixels(raw_img)
            # Synthetic'i float32 olarak yükle (zaten [0,1] kaydedildi)
            synth_img = np.array(Image.open(synthetic_file), dtype=np.float32)
            synth_img = np.clip(synth_img, 0.0, 1.0)

            img_stem = Path(img_info['filename']).stem
            eval_res = evaluate_k(
                real_raw=filtered_img,
                synthetic_norm=synth_img,
                img_stem=img_stem,
                mode=mode,                      # "cropped" | "uncropped"
                buffer_percent=5.0,
                verbose=verbose,
                k_mode=k_mode,
                learned_k_db=learned_k_db
            )
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
                obj_val = (1.0 - ssim_best) + rmse_best
            else:
                raise ValueError(f"Unknown objective: {objective_type}")
            
            objectives.append(obj_val)
            
            if verbose:
                print(f"  Objective: {obj_val:.6f}")
        
        # AGGREGATE
        objectives = np.array(objectives)
        
        if aggregation == 'mean':
            final_obj = float(np.mean(objectives))
        elif aggregation == 'median':
            final_obj = float(np.median(objectives))
        elif aggregation == 'max':
            final_obj = float(np.max(objectives))
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

            set_last_eval_metrics(nmrse=nmrse_mean, ssim=ssim_mean)
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
                
                # ===== RELOAD SYNTHETIC AND CLIP TO PDS DIMENSIONS =====
                try:
                    # Load synthetic (already normalized [0,1])
                    synth_raw = np.array(Image.open(synthetic_file), dtype=np.float32)
                    synth_raw = np.clip(synth_raw, 0.0, 1.0)
                    
                    # Clip to PDS dimensions
                    if 'pds_path' in img_info:
                        pds_params = extract_pds_clip_params(Path(img_info['pds_path']))
                        synth_clipped = synth_raw[
                            pds_params['row_off']:pds_params['row_off']+pds_params['height'],
                            pds_params['col_off']:pds_params['col_off']+pds_params['width']
                        ]
                    else:
                        synth_clipped = synth_raw
                    
                    print(f"  Synthetic clipped shape: {synth_clipped.shape}")
                    
                except Exception as e:
                    print(f"  ⚠️ Warning: Could not reload synthetic: {e}")
                    continue
                
                # ===== BEST SSIM K =====
                best_ssim_k = result_dict.get('best_ssim_k')
                if best_ssim_k is not None:
                    try:
                        # Load and normalize real at best SSIM k
                        raw_img, _ = load_pds_image_data(Path(img_info['pds_path']))
                        filtered_img, _, _ = filter_hot_pixels(raw_img)
                        real_norm_ssim = normalize_image(filtered_img, low_p=float(best_ssim_k), high_p=100.0)
                        
                        # Clip to PDS dimensions
                        if 'pds_path' in img_info:
                            real_norm_ssim = real_norm_ssim[
                                pds_params['row_off']:pds_params['row_off']+pds_params['height'],
                                pds_params['col_off']:pds_params['col_off']+pds_params['width']
                            ]
                        
                        print(f"  Real (k={best_ssim_k}) clipped shape: {real_norm_ssim.shape}")
                        
                        # Check shape compatibility
                        if real_norm_ssim.shape != synth_clipped.shape:
                            print(f"  ⚠️ Shape mismatch: real {real_norm_ssim.shape} vs synth {synth_clipped.shape}")
                            continue
                        
                        # Apply ALIGNMENT & CROPPING if in cropped mode
                        if mode == "cropped":
                            alignment_result = align_and_crop_with_buffer(
                                imgA=real_norm_ssim,
                                imgB=synth_clipped,
                                buffer_ratio=0.05,            # ← A patch’i ile artık geçerli
                                upsample_factor=100,
                                interpolation_order=1
                            )
                            real_norm_ssim = alignment_result['A_crop']
                            synth_for_ssim = alignment_result['B_crop']
                            
                            # Save alignment info
                            alignment_info = {
                                'k': int(best_ssim_k),
                                'metric': 'SSIM',
                                'shift_dy': float(alignment_result['shift'][0]),
                                'shift_dx': float(alignment_result['shift'][1]),
                                'bbox': [int(x) for x in alignment_result['bbox']],
                                'cropped_shape': list(real_norm_ssim.shape)
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
                        if mode == "cropped":
                            print(f"     Shift: dy={alignment_info['shift_dy']:.2f}, dx={alignment_info['shift_dx']:.2f}")
                            
                    except Exception as e:
                        print(f"  ⚠️ Warning: Could not save SSIM images: {e}")
                        import traceback
                        traceback.print_exc()
                
                # ===== BEST RMSE K =====
                best_rmse_k = result_dict.get('best_rmse_k')
                if best_rmse_k is not None and best_rmse_k != best_ssim_k:
                    try:
                        # Load and normalize real at best RMSE k
                        raw_img, _ = load_pds_image_data(Path(img_info['pds_path']))
                        filtered_img, _, _ = filter_hot_pixels(raw_img)
                        real_norm_rmse = normalize_image(filtered_img, low_p=float(best_rmse_k), high_p=100.0)
                        
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
                      csv_dir: Optional[Union[Path, str]] = None):
    """
    objective_fn: mevcut objective (params -> float)
    pop_size    : her iterasyondaki parçacık sayısı (swarm/population)
    csv_dir     : CSV yazılacak klasör (None ise sadece konsol)

    Kullanım:
        obj_wrapped = with_eval_logging(objective, pop_size, csv_dir="optimization_temp")
        result = optimizer.minimize(obj_wrapped, ...)
    """
    eval_idx = 0
    csv_f = None
    if csv_dir:
        csv_dir = Path(csv_dir)
        csv_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = csv_dir / f"eval_history_{stamp}.csv"
        csv_f = open(csv_path, "w", buffering=1)
        print("iteration,particle,objective,nmrse,ssim", file=csv_f)

    def wrapped(x):
        nonlocal eval_idx, csv_f
        ps = max(1, int(pop_size))
        it  = eval_idx // ps
        pid = eval_idx %  ps

        if pid == 0:
            print(f"\n=== Iteration {it} ===")

        val = objective_fn(x)

        try:
            m = get_last_eval_metrics()
            nmrse = float(m.get("nmrse", float("nan")))
            ssim  = float(m.get("ssim",  float("nan")))
        except Exception:
            nmrse, ssim = float("nan"), float("nan")

        print(f"[it {it:03d} | p{pid:03d}] obj={val:+.6f}  NMRSE={nmrse:.6f}  SSIM={ssim:.6f}")
        if csv_f:
            print(f"{it},{pid},{val},{nmrse},{ssim}", file=csv_f)
        eval_idx += 1
        return val

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
        """Write message to log."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        
        with open(self.log_file, 'a') as f:
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