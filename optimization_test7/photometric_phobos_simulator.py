""""
INDEX.png ve proccesed_index.png'lerin düzgün gözüktüğü başka kod
photometric_phobos_simulator.py
Compact Integrated Fixed Photometric Phobos Simulator
All fixes integrated into the main file - no separate imports needed
"""

import sys, os, json, numpy as np, cv2, time, pickle, re, requests, pandas as pd
from pathlib import Path
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from scipy.signal import correlate2d
import argparse 
import math
import bpy

#######SAME RANDOM VALUESSS##########
np.random.seed(42)   
  
# Add current directory to Python path
sys.path.append(os.getcwd())

# Import other modules
try:
    from spice_data_processor import SpiceDataProcessor
    from corto_post_processor import CORTOPostProcessor
    import cortopy as corto
except ImportError as e:
    print(f"Error: Required modules not found. Details: {e}")
    sys.exit(1)

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.floating): return float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, dict): return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list): return [convert_numpy_types(item) for item in obj]
    return obj

AU_KM = 149_597_870.7          # km
SOLAR_CONSTANT_WM2 = 1361.0    # W/m² at 1 AU



class CompactCORTOValidator:
    """Compact CORTO validation with complete Figure 15 pipeline"""
    
    def __init__(self, post_processor):
        self.post_processor = post_processor
        self.validation_results = []
        self._pds_cache = {}

    def _parse_pds_load_image(self, pds_file_path):
        """Combined PDS parsing and image loading"""
        if str(pds_file_path) in self._pds_cache:
            return self._pds_cache[str(pds_file_path)]
        
        try:
            # Parse label
            label = {}
            key_val = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=\s*(.*)$")
            with open(pds_file_path, "r", encoding="utf-8", errors="ignore") as fh:
                for line_num, raw in enumerate(fh):
                    if line_num > 50000: break
                    line = raw.strip().replace("<CR><LF>", "")
                    if line.upper().startswith("END"): break
                    m = key_val.match(line)
                    if m:
                        key, val = m.groups()
                        label[key] = val.strip().strip('"').strip("'")
            
            # Extract parameters and load image
            lines = int(label.get('LINES', 0))
            line_samples = int(label.get('LINE_SAMPLES', 0))
            sample_bits = int(label.get('SAMPLE_BITS', 16))
            sample_type = label.get('SAMPLE_TYPE', 'MSB_INTEGER')
            header_bytes = int(label.get('^IMAGE', '1').split()[0]) - 1
            
            # Determine dtype
            if sample_bits == 16:
                dtype = ('>u2' if 'UNSIGNED' in sample_type else '>i2') if 'MSB' in sample_type else ('<u2' if 'UNSIGNED' in sample_type else '<i2')
            else:
                dtype = '>u1' if 'MSB' in sample_type else '<u1'
            
            # Read and convert
            with open(pds_file_path, 'rb') as f:
                f.seek(header_bytes)
                data = f.read(lines * line_samples * (sample_bits // 8))
            
            image_array = np.frombuffer(data, dtype=dtype).reshape(lines, line_samples)
            
            # Normalize
            if sample_bits == 16:
                img_min, img_max = image_array.min(), image_array.max()
                if img_max > img_min:
                    image_array = ((image_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    image_array = image_array.astype(np.uint8)
            
            self._pds_cache[str(pds_file_path)] = image_array
            return image_array
            
        except Exception as e:
            print(f"❌ Error loading PDS image {pds_file_path}: {e}")
            return None
    def _find_available_mask(self, img_path):
        """Mevcut mask dosyalarından uygun olanı bul"""
        img_path = Path(img_path)
        stem = img_path.stem.replace("_L2", "").replace("processed_", "")
        
        # Mask dosyası arama sırası
        mask_candidates = [
            img_path.parent.parent / "mask_ID_1" / f"{stem}.png",
            img_path.parent.parent / "mask_ID_shadow_1" / f"{stem}.png",
            img_path.parent / "mask_ID_1" / f"{stem}.png", 
            img_path.parent / "mask_ID_shadow_1" / f"{stem}.png",
            # Index-based arama (000000, 000001, etc.)
            img_path.parent.parent / "mask_ID_1" / f"{str(img_path.stem).zfill(6)}.png",
            img_path.parent.parent / "mask_ID_shadow_1" / f"{str(img_path.stem).zfill(6)}.png",
        ]
        
        for candidate in mask_candidates:
            if candidate.exists():
                print(f"   🎯 Found mask: {candidate}")
                return str(candidate)
        
        print(f"   ⚠️ No mask found for {stem}")
        return None

    def _get_slopes_enhancement(self, img_path):
        """Slopes dosyasından normal bilgisi al"""
        img_path = Path(img_path)
        stem = img_path.stem.replace("_L2", "").replace("processed_", "")
        
        slopes_candidates = [
            img_path.parent.parent / "slopes" / f"{stem}.png",
            img_path.parent.parent / "slopes" / f"{str(img_path.stem).zfill(6)}.png",
        ]
        
        for slopes_path in slopes_candidates:
            if slopes_path.exists():
                slopes_img = cv2.imread(str(slopes_path), cv2.IMREAD_GRAYSCALE)
                if slopes_img is not None:
                    print(f"   📐 Using slopes: {slopes_path}")
                    return slopes_img
        return None

    def _apply_slopes_enhancement(self, processed_img, slopes_img):
        """Slopes bilgisini kullanarak enhancement yap"""
        if slopes_img is None:
            return processed_img
        
        if slopes_img.shape != processed_img.shape:
            slopes_img = cv2.resize(slopes_img, (processed_img.shape[1], processed_img.shape[0]))
        
        # Slopes bilgisini edge enhancement için kullan
        enhanced = cv2.addWeighted(processed_img, 0.85, slopes_img, 0.15, 0)
        return enhanced.astype(processed_img.dtype)
        
    def _find_mask_and_calculate_correlation(self, real_img, synthetic_img, img_path):
        """Combined mask finding and correlation calculation"""
        try:
            # Find mask
            img_path = Path(img_path)
            stem = img_path.stem
            mask_candidates = [
                img_path.parent.parent / "mask_ID_1" / f"{stem}.png",
                img_path.parent.parent / "mask_ID_shadow_1" / f"{stem}.png",
                img_path.parent.parent / "label" / "IDmask" / "Mask_1" / f"{stem}.png",
                img_path.parent.parent / "label" / "IDmask" / "Mask_1_shadow" / f"{stem}.png",
            ]
            
            mask_path = None
            for candidate in mask_candidates:
                if candidate.exists():
                    mask_path = candidate
                    break
            
            if not mask_path:
                return 0, False
            
            # Calculate correlation with mask
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_img is None: return 0, False
            
            if mask_img.shape != real_img.shape:
                mask_img = cv2.resize(mask_img, (real_img.shape[1], real_img.shape[0]))
            
            mask_binary = (mask_img > 0).astype(np.uint8)
            real_masked = real_img * mask_binary
            synthetic_masked = synthetic_img * mask_binary
            
            correlation = cv2.matchTemplate(real_masked, synthetic_masked, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(correlation)
            
            return max_val, True
            
        except Exception:
            return 0, False

    def _robust_ssim_calculation(self, img1, img2):
        """Robust SSIM calculation with error handling"""
        try:
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
            img1_norm, img2_norm = img1.astype(np.float64), img2.astype(np.float64)
            img1_range, img2_range = img1_norm.max() - img1_norm.min(), img2_norm.max() - img2_norm.min()
            
            if img1_range == 0 or img2_range == 0: return 0.0
            
            data_range = max(img1_range, img2_range)
            ssim_value = ssim(img1_norm, img2_norm, data_range=data_range, multichannel=False, 
                            gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
            
            return float(ssim_value) if not (np.isnan(ssim_value) or np.isinf(ssim_value)) else 0.0
            
        except Exception:
            return 0.0

    # 🔧 DÜZELTME 1: Global Fotometrik Scale
    # CompactCORTOValidator sınıfında bu metodu değiştirin:

    def _apply_corto_pipeline_and_template_matching(self, real_img, synthetic_imgs, synthetic_img_paths):
        """Combined CORTO pipeline with MASK-AWARE processing"""
        labels = {'CoB': [real_img.shape[1]//2, real_img.shape[0]//2], 'range': 1000.0, 'phase_angle': 0.0}
        
        # ✅ DÜZELTİLDİ: Adaptive resolution
        optimal_size = min(max(real_img.shape[:2]), 256)  # En büyük boyut 256'ya sınırla
        optimal_size = max(optimal_size, 64)  # En küçük boyut 64 olsun
        
        gamma_value = 0.45
        max_physical_value = 2**14 - 1
        scale_factor = 255.0 / max_physical_value
        
        real_normalized = np.clip(real_img * scale_factor, 0, 255).astype(np.uint8) if real_img.dtype != np.uint8 else real_img
        real_gamma_corrected = np.power(real_normalized / 255.0, gamma_value) * 255.0
        real_rgb = cv2.cvtColor(real_gamma_corrected.astype(np.uint8), cv2.COLOR_GRAY2RGB) if len(real_gamma_corrected.shape) == 2 else real_gamma_corrected
        
        # ✅ DÜZELTİLDİ: İlk synthetic image'dan mask path al
        mask_path = self._find_available_mask(synthetic_img_paths[0]) if synthetic_img_paths else None
        
        try:
            real_processed, _ = self.post_processor.process_image_label_pair_no_stretch(
                real_rgb, labels.copy(), target_size=optimal_size, mask_path=mask_path)
        except (AttributeError, TypeError):
            # Fallback: eski API
            real_processed, _ = self.post_processor.process_image_label_pair(real_rgb, labels.copy())
            # Manuel resize optimal boyuta
            if real_processed.shape[0] != optimal_size:
                real_processed = cv2.resize(real_processed, (optimal_size, optimal_size))
        except Exception:
            real_processed = cv2.resize(real_normalized, (optimal_size, optimal_size))
        
        real_processed = cv2.cvtColor(real_processed.astype(np.uint8), cv2.COLOR_RGB2GRAY) if len(real_processed.shape) == 3 else real_processed
        
        template_results = []
        brightness_differences = []
        
        for i, (synthetic_img, img_path) in enumerate(zip(synthetic_imgs, synthetic_img_paths)):
            synthetic_normalized = np.clip(synthetic_img * scale_factor, 0, 255).astype(np.uint8) if synthetic_img.dtype != np.uint8 else synthetic_img
            synthetic_gamma_corrected = np.power(synthetic_normalized / 255.0, gamma_value) * 255.0
            synthetic_rgb = cv2.cvtColor(synthetic_gamma_corrected.astype(np.uint8), cv2.COLOR_GRAY2RGB) if len(synthetic_gamma_corrected.shape) == 2 else synthetic_gamma_corrected

            # ✅ DÜZELTİLDİ: Her synthetic image için kendi mask'ını bul
            synthetic_mask_path = self._find_available_mask(img_path)
            
            try:
                synthetic_processed, _ = self.post_processor.process_image_label_pair_no_stretch(
                    synthetic_rgb, labels.copy(), target_size=optimal_size, mask_path=synthetic_mask_path)
            except (AttributeError, TypeError):
                synthetic_processed, _ = self.post_processor.process_image_label_pair(synthetic_rgb, labels.copy())
                if synthetic_processed.shape[0] != optimal_size:
                    synthetic_processed = cv2.resize(synthetic_processed, (optimal_size, optimal_size))
            except Exception:
                synthetic_processed = cv2.resize(synthetic_normalized, (optimal_size, optimal_size))
            
            synthetic_processed = cv2.cvtColor(synthetic_processed.astype(np.uint8), cv2.COLOR_RGB2GRAY) if len(synthetic_processed.shape) == 3 else synthetic_processed
            
            # ✅ DÜZELTİLDİ: Slopes enhancement uygula
            slopes_img = self._get_slopes_enhancement(img_path)
            if slopes_img is not None:
                synthetic_processed = self._apply_slopes_enhancement(synthetic_processed, slopes_img)
            
            # Brightness analysis
            real_mean_brightness = np.mean(real_processed)
            synthetic_mean_brightness = np.mean(synthetic_processed)
            brightness_diff_absolute = real_mean_brightness - synthetic_mean_brightness
            brightness_diff_relative = brightness_diff_absolute / real_mean_brightness if real_mean_brightness > 0 else 0
            brightness_differences.append({
                'real_mean': float(real_mean_brightness),
                'synthetic_mean': float(synthetic_mean_brightness),
                'absolute_diff': float(brightness_diff_absolute),
                'relative_diff': float(brightness_diff_relative)
            })
            
            # Sub-pixel alignment
            try:
                synthetic_aligned = self._subpixel_alignment(real_processed, synthetic_processed)
            except:
                synthetic_aligned = synthetic_processed
            
            # Soft-edge masking
            try:
                mask_soft = self._create_soft_edge_mask(real_processed, synthetic_aligned)
                real_masked = real_processed * mask_soft
                synthetic_masked = synthetic_aligned * mask_soft
            except:
                real_masked = real_processed
                synthetic_masked = synthetic_aligned
            
            # Template matching
            correlation = cv2.matchTemplate(real_masked, synthetic_masked, cv2.TM_CCOEFF_NORMED)
            _, ncc, _, max_loc = cv2.minMaxLoc(correlation)
            
            # Enhanced mask correlation
            mask_correlation, mask_used = self._find_mask_and_calculate_correlation(real_masked, synthetic_masked, img_path)
            ncc = max(ncc, mask_correlation)
            
            # NRMSE calculation
            mse = np.mean((real_masked.astype(np.float64) - synthetic_masked.astype(np.float64)) ** 2)
            d = real_masked.size
            nrmse = np.sqrt(mse) / np.sqrt(d)
            
            template_results.append({
                'index': i, 'real_processed': real_masked, 'synthetic_processed': synthetic_masked,
                'ncc': float(ncc), 'nrmse': float(nrmse), 'crop_location': max_loc, 'mask_used': mask_used,
                'brightness_analysis': brightness_differences[i],
                'slopes_used': slopes_img is not None
            })
        
        return template_results
    # 🔧 YENİ HELPER METODLAR:

    def _subpixel_alignment(self, reference_img, target_img):
        """Sub-pixel accurate image alignment using phase correlation"""
        try:
            # Phase correlation for sub-pixel accuracy
            ref_float = reference_img.astype(np.float32)
            target_float = target_img.astype(np.float32)
            shift, response = cv2.phaseCorrelate(ref_float, target_float)
            
            # Apply sub-pixel shift
            from scipy import ndimage
            aligned_img = ndimage.shift(target_img, shift, order=1, mode='nearest')
            return aligned_img.astype(np.uint8)
        except:
            return target_img

    def _create_soft_edge_mask(self, real_img, synthetic_img):
        """Create soft-edge mask to reduce aliasing penalties"""
        try:
            # Create binary mask from real image
            _, binary_mask = cv2.threshold(real_img, 1, 1, cv2.THRESH_BINARY)
            
            # Apply Gaussian smoothing for soft edges
            kernel = cv2.getGaussianKernel(5, 1.0)
            mask_soft = cv2.filter2D(binary_mask.astype(np.float32), -1, kernel @ kernel.T)
            
            return mask_soft
        except:
            return np.ones_like(real_img, dtype=np.float32)

    def _generate_and_apply_noise_combinations(self, best_templates):
        """Generate noise combinations and apply to best templates"""
        # CORTO Table 1 noise parameters
        gaussian_means, gaussian_vars = [0.01, 0.09, 0.17, 0.25], [1e-5, 1e-4, 1e-3]
        blur_values, brightness_values = [0.6, 0.8, 1.0, 1.2], [1.00, 1.17, 1.33, 1.50]
        
        noise_combinations = []
        for g_mean in gaussian_means:
            for g_var in gaussian_vars:
                for blur in blur_values:
                    for brightness in brightness_values:
                        noise_combinations.append({'gaussian_mean': g_mean, 'gaussian_var': g_var, 'blur': blur, 'brightness': brightness})
        
        # Apply noise and calculate SSIM
        noisy_results = []
        J = min(len(noise_combinations), 72)  # Limit for performance
        
        for template_result in best_templates:
            for j, noise_params in enumerate(noise_combinations[:J]):
                try:
                    # Apply noise
                    result = template_result['synthetic_processed'].copy().astype(np.float32)
                    noise = np.random.normal(noise_params['gaussian_mean'], np.sqrt(noise_params['gaussian_var']), result.shape)
                    result += noise
                    
                    kernel_size = int(noise_params['blur'] * 2) * 2 + 1
                    result = cv2.GaussianBlur(result, (kernel_size, kernel_size), noise_params['blur'])
                    result *= noise_params['brightness']
                    noisy_synthetic = np.clip(result, 0, 255).astype(np.uint8)
                    
                    ssim_score = self._robust_ssim_calculation(template_result['real_processed'], noisy_synthetic)
                    
                    noisy_results.append({
                        'template_index': template_result.get('index', 0),
                        'synthetic_index': template_result.get('synthetic_index', 0),
                        'noise_params': noise_params, 'ssim': ssim_score,
                        'nrmse': template_result['nrmse'], 'ncc': template_result['ncc']
                    })
                except Exception:
                    continue
        
        return noisy_results

    def _calculate_final_scores(self, best_result):
        """Calculate composite and status scores"""
        ssim_val = max(0.0, min(1.0, best_result.get('ssim', 0.0))) if not (np.isnan(best_result.get('ssim', 0.0)) or np.isinf(best_result.get('ssim', 0.0))) else 0.0
        ncc_val = max(-1.0, min(1.0, best_result.get('ncc', 0.0))) if not (np.isnan(best_result.get('ncc', 0.0)) or np.isinf(best_result.get('ncc', 0.0))) else 0.0
        nrmse_val = max(0.0, min(2.0, best_result.get('nrmse', 1.0))) if not (np.isnan(best_result.get('nrmse', 1.0)) or np.isinf(best_result.get('nrmse', 1.0))) else 1.0
        
        composite = (ssim_val + ncc_val + (1 - min(nrmse_val, 1.0))) / 3
        status = 'EXCELLENT' if composite > 0.8 else 'GOOD' if composite > 0.7 else 'MODERATE' if composite > 0.5 else 'POOR'
        
        return float(composite), status

    def validate_with_complete_pipeline(self, pds_img_path, synthetic_img_paths, utc_time, img_filename):
        """Complete CORTO Figure 15 validation pipeline - COMPACT VERSION"""
        try:
            print(f"\n🔍 Starting Enhanced CORTO Validation Pipeline...")
            print(f"   Validating {img_filename} with complete CORTO Figure 15 methodology...")
            
            # 1. Load PDS image
            real_img = self._parse_pds_load_image(pds_img_path)
            if real_img is None:
                print(f"   ❌ Failed to load PDS image")
                return None
            
            # 2. Load synthetic images
            synthetic_imgs = []
            valid_paths = []
            for img_path in synthetic_img_paths:
                if Path(img_path).exists():
                    synthetic_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if synthetic_img is not None:
                        synthetic_imgs.append(synthetic_img)
                        valid_paths.append(img_path)
            
            if not synthetic_imgs:
                print(f"   ❌ No valid synthetic images found")
                return None
            
            print(f"   📸 Processing {len(synthetic_imgs)} synthetic image(s)")
            
            # 3. Apply CORTO pipeline and template matching
            template_results = self._apply_corto_pipeline_and_template_matching(real_img, synthetic_imgs, valid_paths)
            
            # 4. Select M best templates
            M = min(len(template_results), 5)
            best_M = sorted(template_results, key=lambda x: x['nrmse'])[:M]
            print(f"   🎯 Selected {len(best_M)} best templates")
            
            # 5. Apply noise and calculate SSIM
            noisy_results = self._generate_and_apply_noise_combinations(best_M)
            
            # 6. Select L final candidates
            L = min(len(noisy_results), 10)
            best_L = sorted(noisy_results, key=lambda x: x['ssim'], reverse=True)[:L]
            print(f"   🏆 Selected {len(best_L)} final candidates")
            
            # 7. Calculate final results
            best_result = best_L[0] if best_L else (template_results[0] if template_results else {'ssim': 0, 'ncc': 0, 'nrmse': 1})
            if not best_L and template_results:
                best_result.update({'ssim': 0})
            
            composite_score, validation_status = self._calculate_final_scores(best_result)
            
            validation_result = {
                'utc_time': utc_time, 'img_filename': img_filename, 'pds_img_path': str(pds_img_path),
                'num_synthetic_imgs': len(synthetic_imgs), 'validation_pipeline': 'CORTO_Figure_15_Complete_COMPACT',
                'template_matching_results': len(template_results), 'M_best_selected': len(best_M),
                'J_noise_combinations': min(len(self._generate_and_apply_noise_combinations([])) if best_M else 0, 20),
                'L_final_candidates': len(best_L), 'best_ncc': best_result.get('ncc', 0),
                'best_nrmse': best_result.get('nrmse', 1), 'best_ssim': best_result.get('ssim', 0),
                'composite_score': composite_score, 'validation_status': validation_status,
                'timestamp': datetime.now().isoformat(),
                'mask_alignment': any(r.get('mask_used', False) for r in template_results),
                'detailed_results': {'template_results': template_results[:3], 'best_M_results': best_M[:3], 'final_L_results': best_L[:3]}
            }
            
            self.validation_results.append(validation_result)
            print(f"   ✅ Validation completed - Status: {validation_status}, Score: {composite_score:.4f}")
            
            return validation_result
            
        except Exception as e:
            print(f"❌ Error in enhanced validation for {img_filename}: {e}")
            return None

    def get_validation_summary(self):
        """Get summary of all validations - COMPACT VERSION"""
        if not self.validation_results:
            return {'total_validations': 0, 'successful_validations': 0, 'success_rate': 0.0, 'average_composite_score': 0.0, 'average_ssim': 0.0, 'pds_processing_enabled': True, 'transformation_alignment': 'FIXED', 'mask_alignment_available': False}
        
        total = len(self.validation_results)
        successful = len([r for r in self.validation_results if r.get('validation_status', 'POOR') != 'POOR'])
        
        valid_composite_scores = [r.get('composite_score', 0) for r in self.validation_results if not (np.isnan(r.get('composite_score', 0)) or np.isinf(r.get('composite_score', 0)))]
        valid_ssim_scores = [r.get('best_ssim', 0) for r in self.validation_results if not (np.isnan(r.get('best_ssim', 0)) or np.isinf(r.get('best_ssim', 0)))]
        
        return {
            'total_validations': total, 'successful_validations': successful, 'success_rate': successful / total if total > 0 else 0.0,
            'average_composite_score': float(np.mean(valid_composite_scores)) if valid_composite_scores else 0.0,
            'average_ssim': float(np.mean(valid_ssim_scores)) if valid_ssim_scores else 0.0,
            'pds_processing_enabled': True, 'transformation_alignment': 'FIXED',
            'mask_alignment_available': any(r.get('mask_alignment', False) for r in self.validation_results)
        }

class CompactPDSProcessor:
    """Compact PDS processor"""
    
    def __init__(self, pds_data_path="PDS_Data"):
        self.pds_data_path = Path(pds_data_path)
        self._key_val = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=\s*(.*)$")

    def parse_and_process_directory(self, img_directory_path):
        """Combined parsing and processing of IMG directory"""
        records = []
        img_dir = Path(img_directory_path)
        
        print(f"Processing IMG files in: {img_dir}")
        
        for img_file in img_dir.rglob("*.IMG"):
            try:
                # Parse label
                label = {}
                with open(img_file, "r", encoding="utf-8", errors="ignore") as fh:
                    for i, raw in enumerate(fh):
                        if i > 50000: break
                        line = raw.strip().replace("<CR><LF>", "")
                        if line.upper().startswith("END"): break
                        m = self._key_val.match(line)
                        if m:
                            key, val = m.groups()
                            label[key] = val.strip().strip('"').strip("'")
                
                label.update({"file_path": str(img_file), "file_name": img_file.name})
                records.append(label)
                print(f"Processed: {img_file.name}")
            except Exception as e:
                print(f"Error processing {img_file.name}: {e}")
                
        if not records:
            raise RuntimeError("No IMG files found or processed!")
            
        df = pd.DataFrame(records)
        
        # Process time columns
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
            
        df["STATUS"] = (df["START_TIME"].notna() & df["STOP_TIME"].notna()).map({True: "SUCCESS", False: "MISSING_TIME_DATA"})
        
        return df

class CompactPhotometricSimulator:
    """Compact Photometric Phobos Simulator"""
    
    def __init__(self, config_path=None, camera_type='SRC', pds_data_path=None, 
                model="radiance", output_scale="DN", 
                global_gain_factor=1.0, gamma_correction=0.45):
        """Initialize simulator with photometric enhancement parameters
        
        Args:
            global_gain_factor: Factor to apply to sun energy instead of albedo (default 0.18)
            gamma_correction: Gamma value for camera response simulation (default 0.45 for HRSC-SRC)
        """
        self.config = self._load_or_create_config(config_path)
        self.camera_type = camera_type
        self.pds_data_path = pds_data_path or self.config.get('pds_data_path', 'PDS_Data')
        self.model = model
        self.output_scale = output_scale.upper()   # "DN" veya "ABS"
        
        
        # Initialize components
        self.spice_processor = SpiceDataProcessor(base_path=self.config['spice_data_path'])
        self.pds_processor = CompactPDSProcessor(self.pds_data_path)
        self.post_processor = CORTOPostProcessor(target_size=128)
        self.validator = CompactCORTOValidator(self.post_processor)
        self.global_gain_factor = global_gain_factor
        self.gamma_correction = gamma_correction
    
        self.scenario_name = "S07_Mars_Phobos_Deimos"
        
        # Get camera and photometric configuration
        self.camera_config = self._get_camera_config()
        #self.photometric_params = self._get_photometric_params() deleted for exposure time usage  
        
        print(f"✅ Compact simulator initialized with FIXED validation")
        print(f"✅ Photometric simulator initialized with:")
        print(f"   Global gain factor: {self.global_gain_factor}")
        print(f"   Gamma correction: {self.gamma_correction}")

        self._calculate_albedo_factors()

    def _calculate_albedo_factors(self):
        """Calculate albedo factors from actual texture files - called once in init"""
        import cv2
        import numpy as np
        
        # Texture paths (relative to input directory)
        base_input = Path(self.config.get('input_path', 'input/S07_Mars_Phobos_Deimos'))
        texture_paths = {
            1: base_input / 'body' / 'albedo' / 'Phobos grayscale.jpg',
            2: base_input / 'body' / 'albedo' / 'mars_1k_color.jpg', 
            3: base_input / 'body' / 'albedo' / 'Deimos grayscale.jpg'
        }
        
        # Target albedos for each body (literature values) - ✅ EKLENDİ
        target_albedos = {
            1: 0.068,   # Phobos: geometric albedo
            2: 0.14,   # Mars: average geometric albedo #for HRSC SRC channel
            3: 0.06    # Deimos: geometric albedo
        }
        
        # Calculate texture averages and albedo factors
        self.texture_averages = {}
        self.albedo_factors = {}
        
        print("\n🎨 Analyzing texture files for albedo calibration:")
        
        for body_id, tex_path in texture_paths.items():
            try:
                # Load texture (handle different bit depths)
                tex = cv2.imread(str(tex_path), cv2.IMREAD_UNCHANGED)
                if tex is None:
                    print(f"   ⚠️  Body {body_id}: Texture not found at {tex_path}")
                    print(f"   ⚠️  Using fallback average: 0.216")
                    self.texture_averages[body_id] = 0.216  # Fallback
                    self.albedo_factors[body_id] = target_albedos[body_id] / 0.216
                    continue
                
                # Convert to grayscale if needed
                if tex.ndim == 3:
                    tex = cv2.cvtColor(tex, cv2.COLOR_BGR2GRAY)
                
                tex = tex.astype(np.float32)
                
                # Calculate relative average (0-1 range)
                max_val = tex.max()
                if max_val == 0:
                    print(f"   ❌ Body {body_id}: Empty texture detected")
                    self.texture_averages[body_id] = 0.216  # Fallback
                    self.albedo_factors[body_id] = target_albedos[body_id] / 0.216
                    continue
                
                texture_avg_relative = tex.mean() / tex.max()
                albedo_factor = 0.07 / texture_avg_relative
                albedo_factor = target_albedos[body_id] / texture_avg_relative
                
                # Store results
                self.texture_averages[body_id] = texture_avg_relative
                self.albedo_factors[body_id] = albedo_factor
                
                print(f"   📊 Body {body_id} ({tex_path.name}):")
                print(f"      Texture shape: {tex.shape}")
                print(f"      Max pixel value: {max_val:.0f}")
                print(f"      Texture average (relative): {texture_avg_relative:.4f}")
                print(f"      Target albedo: {target_albedos[body_id]:.3f}")
                print(f"      ✅ Albedo factor: {albedo_factor:.4f}")
                
            except Exception as e:
                print(f"   ❌ Error processing Body {body_id} texture: {e}")
                # Fallback values
                self.texture_averages[body_id] = 0.216
                self.albedo_factors[body_id] = target_albedos[body_id] / 0.216
                print(f"   ⚠️  Using fallback factor: {self.albedo_factors[body_id]:.4f}")
        
        print(f"\n✅ Albedo factor calculation complete:")
        for body_id in [1, 2, 3]:
            print(f"   Body {body_id}: factor = {self.albedo_factors[body_id]:.4f}")

    # 1)  EKLENTİ  ——  LUT kazancı   (sınıf içinde herhangi bir yere)
    # ================================================================
    @staticmethod
    def gain_L2_eff(exp_ms      : float,
                    buffer_mode : str   = "BUFFER_8",
                    release_id  : str   = "0148",
                    T_degC      : float = 0.0) -> float:
        """
        SRC Level‑2 etkin LUT kazancı  g_eff  [e⁻ / DN8]
        """
        base     = 2.0e6 if buffer_mode == "BUFFER_8" else 0.25e6
        gamma    = 0.45
        exp_ref  = 84.672                          # ms
        rel_corr = {"0146":0.93,"0147":0.97,"0148":1.00,
                    "0149":0.93,"0150":1.08}.get(release_id, 1.0)
        scale_T  = 1.0 + 0.01*T_degC              # +1 % / °C
        return base*(exp_ref/exp_ms)**gamma*rel_corr*scale_T
        # ================================================================


    

    def compute_sun_strength_photon(
            self,                        # <-- sınıf içi çağrı için
            solar_distance_km : float,
            exposure_time_s   : float,
            *,
            buffer_mode       = "BUFFER_8",
            release_id        = "0148",
            instrument_temp   = 0.0,
            lambda_eff        = 675e-9,   # m
            pixel_pitch       = 7e-6,     # m
            qe                = 0.07,     # e⁻/photon 0.07 #effectşv
            gain_e_per_dn_L0  = 40.0,     # e⁻/DN14
            level             = "L2",     # 'L0' | 'L2'
            return_mode       = "radiance_rel"  # varsay.
    ):
        AU_KM, SOLC = 1.495978707e8, 1361.0
        H,C,PI      = 6.62607015e-34, 2.99792458e8, math.pi

        # ---- Güneş akısı -------------------------------------------------
        irr   = SOLC / (solar_distance_km / AU_KM)**2          # W m⁻²
        E_ph  = H*C / lambda_eff
        N_ph  = irr/E_ph * exposure_time_s                     # photon m⁻²
        e_min = N_ph * (pixel_pitch**2) * qe                   # e⁻

        # --------------------------- Level‑0 ------------------------------
        if level == "L0":
            if return_mode == "electron":
                return e_min
            dn14 = e_min / gain_e_per_dn_L0
            if return_mode in ("dn", "radiance_rel"):
                return dn14
            K0 = gain_e_per_dn_L0*E_ph/((pixel_pitch**2)*exposure_time_s*qe)
            return dn14 * K0                                   # radiance_abs

        # --------------------------- Level‑2 ------------------------------
        g_eff  = self.gain_L2_eff(exposure_time_s*1e3,
                                buffer_mode, release_id, instrument_temp)
        dn_rel = e_min / g_eff

        if return_mode in ("dn", "radiance_rel"):
            return dn_rel
        if return_mode == "electron":
            return e_min

        K = g_eff*E_ph / ((pixel_pitch**2)*exposure_time_s*qe)
        return dn_rel * K                                      # radiance_abs




    def _load_or_create_config(self, config_path):
        """Load or create default configuration"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        base_dir = Path(os.getcwd())
        return {
            'input_path': str(base_dir / "input" / "S07_Mars_Phobos_Deimos"),
            'output_path': str(base_dir / 'output' / 'enhanced_photometric_validation'),
            'spice_data_path': str(base_dir / 'spice_kernels'),
            'pds_data_path': str(base_dir / 'PDS_Data'),
            'real_images_path': str(base_dir / 'real_hrsc_images'),
            'body_files': ['g_phobos_036m_spc_0000n00000_v002.obj', 'Mars_65k.obj', 'g_deimos_162m_spc_0000n00000_v001.obj'],
            'scene_file': 'scene_mmx.json',
            'geometry_file': 'geometry_mmx.json'
        }

    def _get_camera_config(self):
        """Get camera configuration with fallback"""
        try:
            return self.spice_processor.get_hrsc_camera_config(self.camera_type)
        except Exception:
            if self.camera_type == 'SRC':
                # ✅ DÜZELTME: exposure_time_s yerine 1.0 default değer
                return {
                    'fov': 0.54, 
                    'res_x': 1024, 
                    'res_y': 1024, 
                    'film_exposure': 1.0,  # ✅ DÜZELTİLDİ: exposure_time_s tanımsızdı
                    'sensor': 'BW', 
                    'clip_start': 0.1, 
                    'clip_end': 100000000.0, 
                    'bit_encoding': '16', 
                    'viewtransform': 'Standard', 
                    'K': [[1222.0, 0, 512.0], [0, 1222.0, 512.0], [0, 0, 1]]
                }
            else:
                return {
                    'fov': 11.9, 
                    'res_x': 5184, 
                    'res_y': 1, 
                    'film_exposure': 1.0,  # ✅ DÜZELTİLDİ: exposure_time_s tanımsızdı
                    'sensor': 'BW', 
                    'clip_start': 0.1, 
                    'clip_end': 10000.0, 
                    'bit_encoding': '16', 
                    'viewtransform': 'Standard', 
                    'K': [[2500.0, 0, 2592.0], [0, 2500.0, 0.5], [0, 0, 1]]
                }
    def _get_photometric_params(self,
                                exposure_time_s: float,
                                solar_distance_km: float | None = None,
                                global_gain_factor: float = 1.0):  # Reference: CORTO validation paper
        """
        Tek kaynaktan iki ölçek:
        • sun_strength_abs  – W m⁻² sr⁻¹
        • dn_rel            – 0‑255 DN₈
        self.output_scale göre sadece biri sun_strength olarak dönüyor.
        """
        solar_distance_km = solar_distance_km or AU_KM
        solar_distance_au = solar_distance_km / AU_KM
        irradiance = SOLAR_CONSTANT_WM2 / solar_distance_au**2        # W m⁻²

        # 1) Mutlak radyans
        L_abs = self.compute_sun_strength_photon(
                    solar_distance_km, exposure_time_s,
                    return_mode="radiance_abs")

        # 2) Aynı koşulların DN (0‑255) karşılığı
        dn_rel = self.compute_sun_strength_photon(
                    solar_distance_km, exposure_time_s,
                    return_mode="dn")

        K = L_abs / dn_rel        # ≈ 2.0 – 2.3   abs→DN dönüştürücü

        # 3) Hangisi çıktı olacak?
        if self.output_scale == "ABS":  # ✅ Doğru girinti
            sun_strength = L_abs * global_gain_factor
        else:  # "DN" 
            sun_strength = dn_rel * global_gain_factor
            
        print(f"      Global gain factor: {global_gain_factor}")  # Debug info

        # 4) DEBUG
        scale_txt = "W m⁻² sr⁻¹" if self.output_scale == "ABS" else "DN₈"
        print("   🌞 Solar const model:")
        print(f"      Distance       : {solar_distance_km/1e6:7.3f} Mkm "
            f"({solar_distance_au:4.3f} AU)")
        print(f"      Irradiance     : {irradiance:6.1f} W m⁻²")
        print(f"      Exposure time  : {exposure_time_s:.4f} s")
        print(f"      Sun strength   : {sun_strength:.3f}  {scale_txt}")
        if self.output_scale == "ABS":
            print(f"      DN (rel)       : {dn_rel:.2f} DN₈   →  K = {K:.3f}")

        return {
            "sun_strength"      : sun_strength,   # ölçek seçimine göre
            "sun_strength_abs"  : L_abs,          # daima döner (isterseniz kullanın)
            "dn_rel"            : dn_rel,
            "K_factor"          : K,
            "irradiance_Wm2"    : irradiance,
            "exposure_time"     : exposure_time_s,
            "solar_distance_km" : solar_distance_km,
            "solar_distance_au" : solar_distance_au,
        }

    def setup_scene_and_environment(self, utc_time, index, exposure_time_s):
        """scene + environment (foton modeli entegre) - UPDATED WITH BODY-SPECIFIC ROUGHNESS"""
        print(f"\nSetting up scene for: {utc_time}")

        # ------------------------------------------------------------------
        # 1) SPICE verisi ve Güneş uzaklığı
        # ------------------------------------------------------------------
        corto.Utils.clean_scene()
        try:
            spice_data = self.spice_processor.get_spice_data(utc_time)
            solar_distance_km = spice_data.get("distances", {}).get("sun_to_phobos", 0)
        except Exception:
            # ✅ DÜZELTME: Düzgün fallback dict
            spice_data = {
                "sun": {"position": [0, 0, -149597870.7]},
                "hrsc": {"position": [0, 0, 0], "quaternion": [0, 0, 0, 1]},
                "phobos": {"position": [0, 0, 9376]}, 
                "mars": {"position": [0, 0, 0]},
                "deimos": {"position": [0, 0, 23463.2]}
            }
            solar_distance_km = 149_597_870.7       # 1 AU
        
        # ✅ DÜZELTME: Bu kısım exception bloğunun DIŞINDA olmalı
        # ------------------------------------------------------------------
        # 2) Fotometrik parametreleri hesapla
        # ------------------------------------------------------------------
        photometric_params = self._get_photometric_params(
            exposure_time_s=exposure_time_s,
            solar_distance_km=solar_distance_km,
            global_gain_factor=self.global_gain_factor
        )
        sun_energy = photometric_params["sun_strength"]

        # ------------------------------------------------------------------
        # 3) ✅ YENİ: Literatür-temelli roughness değerleri
        # ------------------------------------------------------------------
        body_roughness_values = self._get_literature_based_roughness()
        print(f"   🎨 Literature-based roughness values: {body_roughness_values}")

        # ------------------------------------------------------------------
        # 4) Scene / geometry JSON'larını yaz
        # ------------------------------------------------------------------
        output_dir = Path(self.config["output_path"]); output_dir.mkdir(parents=True, exist_ok=True)

        geometry_data = {
            "sun":   {"position": [convert_numpy_types(spice_data["sun"]["position"])]},
            "camera":{"position": [convert_numpy_types(spice_data["hrsc"]["position"])],
                    "orientation":[convert_numpy_types(spice_data["hrsc"]["quaternion"])]},
            "body_1":{"position": [convert_numpy_types(spice_data["phobos"]["position"])],
                    "orientation":[convert_numpy_types(spice_data["phobos"]["quaternion"])]},
            "body_2":{"position": [convert_numpy_types(spice_data["mars"]["position"])],
                    "orientation":[convert_numpy_types(spice_data["mars"]["quaternion"])]},
            "body_3":{"position": [convert_numpy_types(spice_data["deimos"]["position"])],
                    "orientation":[convert_numpy_types(spice_data["deimos"]["quaternion"])]},
        }

        # kameranın shutter değeri – sadece radiance modelinde kullan
        shutter_val = exposure_time_s if self.model == "radiance" else 1.0

        scene_config = {
            "camera_settings": {
                **{k: (float(v) if k in ["fov","film_exposure","clip_start","clip_end"]
                    else int(v) if k in ["res_x","res_y"]
                    else str(v))
                for k,v in self.camera_config.items()},
                "K": self.camera_config.get("K", [[1222,0,512],[0,1222,512],[0,0,1]]),
                "shutter": shutter_val
            },
            "sun_settings": {
                "angle": 0.00927,
                "energy": sun_energy           # ► yeni foton/irradiance değeri
            },
            "body_settings_1": {"pass_index":1,"diffuse_bounces":4},
            "body_settings_2": {"pass_index":2,"diffuse_bounces":4},
            "body_settings_3": {"pass_index":3,"diffuse_bounces":4},
            "rendering_settings": {"engine":"CYCLES","device":"CPU","samples":256,"preview_samples":16},
        }

        # json kaydet
        geom_path  = output_dir / "geometry_dynamic.json"
        scene_path = output_dir / "scene_src.json"
        geom_path.write_text(json.dumps(geometry_data, indent=4))
        scene_path.write_text(json.dumps(convert_numpy_types(scene_config), indent=4))

        # ------------------------------------------------------------------
        # 5) CORTO State & Environment
        # ------------------------------------------------------------------
        state = corto.State(scene=str(scene_path), geometry=str(geom_path),
                            body=self.config["body_files"], scenario=self.scenario_name)
        self._add_paths_to_state(state)

        cam_props = {k: (float(v) if k in ["fov","film_exposure","clip_start","clip_end"]
                        else int(v) if k in ["res_x","res_y"]
                        else str(v))
                    for k,v in self.camera_config.items()}
        cam_props["K"] = self.camera_config.get("K", [[1222,0,512],[0,1222,512],[0,0,1]])
        cam = corto.Camera(f"HRSC_{self.camera_type}_Camera", cam_props)

        sun = corto.Sun("Sun", {"angle":0.00927, "energy":sun_energy})

        bodies = []
        for i, body_name in enumerate([Path(b).stem for b in self.config["body_files"]], 1):
            bodies.append(corto.Body(body_name, {"pass_index":i, "diffuse_bounces":4}))

        rendering = corto.Rendering({"engine":"CYCLES","device":"CPU","samples":256,"preview_samples":16})
        env       = corto.Environment(cam, bodies, sun, rendering)

        # ✅ Light leakage reduction (diffuse_bounces=4 korunur)
        if hasattr(bpy.context.scene, 'cycles'):
            bpy.context.scene.cycles.glossy_bounces = 1        # Reduce specular bounces only
            bpy.context.scene.cycles.transmission_bounces = 0  # No transmission  
            bpy.context.scene.cycles.volume_bounces = 0        # No volume scattering
            # diffuse_bounces = 4 remains unchanged as requested
            
            # Set world background to pure black (no ambient light)
            if bpy.data.worlds.get("World"):
                world = bpy.data.worlds["World"] 
                if world.node_tree and world.node_tree.nodes.get('Background'):
                    world.node_tree.nodes['Background'].inputs[1].default_value = 0  # World strength = 0

        # ------------------------------------------------------------------
        # 6) Malzeme + kompoziting (YENİ: body-specific roughness)
        # ------------------------------------------------------------------
        materials = []
        for i, body in enumerate(bodies, 1):
            mat = corto.Shading.create_new_material(f"photometric_material_{i}")
            if hasattr(corto.Shading, "create_branch_albedo_mix"):
                corto.Shading.create_branch_albedo_mix(mat, state, i)
            if hasattr(corto.Shading, "load_uv_data"):
                corto.Shading.load_uv_data(body, state, i)
            corto.Shading.assign_material_to_object(mat, body)
            materials.append(mat)

        # ✅ YENİ: Body-specific roughness ile material setup
        self._setup_material_with_body_specific_roughness(
            state, materials, bodies, body_roughness_values, self.albedo_factors
        )

        tree = self._setup_compositing(state)

        return state, env, cam, bodies, sun, materials, tree, spice_data

    def _get_literature_based_roughness(self):
        """Literatür-temelli roughness değerlerini hesapla"""
        # ✅ Literatür referansları:
        # Phobos: Hapke roughness ~24°, Phase coeff. β=0.020±0.001, Porosity %87
        # Mars: HRSC photometry - daha düşük roughness, higher SSA
        # Deimos: Phobos'tan "smoother surface", lower phase coefficient
        
        body_roughness = {
            1: 0.85,  # Phobos: Çok pürüzlü asteroidal regolith 
                     # Reference: "very rough surface", "extensive grooves", high porosity
            2: 0.45,  # Mars: Orta düzey pürüzlülük, toz tabakaları
                     # Reference: HRSC photometry - "lower roughness than Phobos"
            3: 0.75   # Deimos: Pürüzlü ama Phobos'tan düz
                     # Reference: "smoother than Phobos", sediment-filled craters
        }
        
        # Ek fotometrik özellikler (isteğe bağlı)
        body_photometric_properties = {
            1: {  # Phobos
                'anisotropic': 0.1,     # Groove yapısından dolayı hafif anisotropic
                'sheen': 0.05,          # Dust layer için minimal sheen
                'description': 'Highly porous asteroidal regolith with grooves'
            },
            2: {  # Mars  
                'anisotropic': 0.0,     # İzotropic yüzey
                'sheen': 0.0,           # Sheen yok
                'description': 'Variable dust/rock surface'
            },
            3: {  # Deimos
                'anisotropic': 0.05,    # Minimal anisotropy
                'sheen': 0.03,          # Hafif dust sheen
                'description': 'Regolith with sediment-filled craters'
            }
        }
        
        return {
            'roughness': body_roughness,
            'properties': body_photometric_properties
        }

    def _setup_material_with_body_specific_roughness(self, state, materials, bodies, roughness_data, albedo_factors=None):
        """CORTOPY/MONET uyumlu, literatür-temelli material setup - BLENDER COMPATIBLE"""
        
        roughness_values = roughness_data['roughness']
        photometric_props = roughness_data['properties']
        
        for i, (material, body) in enumerate(zip(materials, bodies), 1):
            if hasattr(corto.Shading, "create_branch_albedo_mix"):
                # ✅ ÖNCE CORTO'nun tree'sini oluştur (CORTOPY uyumluluğu)
                corto.Shading.create_branch_albedo_mix(material, state, i)
                
                if material.node_tree:
                    nodes = material.node_tree.nodes
                    links = material.node_tree.links
                    
                    # Literatür-temelli değerler
                    body_roughness = roughness_values.get(i, 0.7)
                    albedo_factor = albedo_factors.get(i, 0.127) if albedo_factors else 0.127
                    props = photometric_props.get(i, {})
                    
                    print(f"\n   🧹 LITERATURE-BASED MATERIAL SETUP for Body {i}")
                    print(f"   📊 Roughness: {body_roughness:.3f} - {props.get('description', 'Unknown')}")
                    print(f"   🔍 Original tree has {len(nodes)} nodes")
                    
                    # ✅ STEP 1: Texture nodeları ve image'ları kaydet
                    texture_images = {}
                    for node in nodes:
                        if node.type == 'TEX_IMAGE' and node.image:
                            texture_images[node.name] = node.image
                            node.image.colorspace_settings.name = 'Non-Color'
                    
                    print(f"   💾 Preserved {len(texture_images)} CORTOPY texture images")
                    
                    # ✅ STEP 2: Output node'u kaydet
                    output_node = None
                    for node in nodes:
                        if node.type == 'OUTPUT_MATERIAL':
                            output_node = node
                            break
                    
                    # ✅ STEP 3: Output HARİÇ tüm nodeları temizle
                    for node in list(nodes):
                        if node.type != 'OUTPUT_MATERIAL':
                            nodes.remove(node)
                    
                    print(f"   🧹 Cleared all nodes except output (CORTOPY compatibility)")
                    
                    # ✅ STEP 4: YENİ, CORTOPY/MONET uyumlu node tree oluştur
                    if texture_images:
                        # Texture node oluştur
                        texture_node = nodes.new(type='ShaderNodeTexImage')
                        texture_node.location = (-400, 0)
                        texture_node.name = f"CORTOPY_Texture_Body_{i}"
                        
                        # CORTOPY texture'ını ata
                        first_image = list(texture_images.values())[0]
                        texture_node.image = first_image
                        texture_node.image.colorspace_settings.name = 'Non-Color'
                        
                        # Albedo multiply node
                        multiply_node = nodes.new(type='ShaderNodeMath')
                        multiply_node.operation = 'MULTIPLY'
                        multiply_node.location = (-200, 0)
                        multiply_node.inputs[1].default_value = albedo_factor
                        multiply_node.name = f"CORTOPY_Albedo_Scale_Body_{i}"
                        
                        # ✅ Principled BSDF (literatür-temelli roughness)
                        bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
                        bsdf_node.location = (0, 0)
                        bsdf_node.name = f"Literature_BSDF_Body_{i}"
                        
                        # ✅ LITERATÜR-TEMELLİ AYARLAR:
                        bsdf_node.inputs['Roughness'].default_value = body_roughness
                        
                        # ✅ BLENDER VERSION COMPATIBLE: Güvenli input assignment
                        # İsteğe bağlı fotometrik özellikler - güvenli şekilde
                        if 'anisotropic' in props and props['anisotropic'] > 0:
                            try:
                                bsdf_node.inputs['Anisotropic'].default_value = props['anisotropic']
                                print(f"   🔧 Applied anisotropic: {props['anisotropic']:.3f}")
                            except KeyError:
                                print(f"   ⚠️ Anisotropic input not available in this Blender version")
                        
                        if 'sheen' in props and props['sheen'] > 0:
                            try:
                                # ✅ Farklı Blender versiyonları için alternatif isimler
                                if 'Sheen' in bsdf_node.inputs:
                                    bsdf_node.inputs['Sheen'].default_value = props['sheen']
                                    print(f"   ✨ Applied sheen: {props['sheen']:.3f}")
                                elif 'Sheen Weight' in bsdf_node.inputs:  # Blender 4.0+
                                    bsdf_node.inputs['Sheen Weight'].default_value = props['sheen']
                                    print(f"   ✨ Applied sheen weight: {props['sheen']:.3f}")
                                else:
                                    print(f"   ⚠️ Sheen input not available in this Blender version")
                            except (KeyError, TypeError):
                                print(f"   ⚠️ Sheen input not available in this Blender version")
                        
                        # Bağlantılar: Texture -> Multiply -> BSDF -> Output
                        links.new(texture_node.outputs['Color'], multiply_node.inputs[0])
                        links.new(multiply_node.outputs['Value'], bsdf_node.inputs['Base Color'])
                        if output_node:
                            links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])
                        
                        print(f"   ✅ CORTOPY-compatible material: Texture -> Albedo({albedo_factor:.4f}) -> BSDF(R={body_roughness:.3f}) -> Output")
                    
                    else:
                        # Texture yoksa pure diffuse
                        bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
                        bsdf_node.location = (0, 0)
                        bsdf_node.name = f"Literature_Pure_BSDF_Body_{i}"
                        bsdf_node.inputs['Base Color'].default_value = (albedo_factor, albedo_factor, albedo_factor, 1.0)
                        bsdf_node.inputs['Roughness'].default_value = body_roughness
                        
                        # ✅ BLENDER VERSION COMPATIBLE: Güvenli fotometrik özellikler
                        if 'anisotropic' in props:
                            try:
                                bsdf_node.inputs['Anisotropic'].default_value = props['anisotropic']
                            except KeyError:
                                pass  # Bu Blender versiyonunda yok
                        
                        if 'sheen' in props:
                            try:
                                if 'Sheen' in bsdf_node.inputs:
                                    bsdf_node.inputs['Sheen'].default_value = props['sheen']
                                elif 'Sheen Weight' in bsdf_node.inputs:
                                    bsdf_node.inputs['Sheen Weight'].default_value = props['sheen']
                            except (KeyError, TypeError):
                                pass  # Bu Blender versiyonunda yok
                        
                        if output_node:
                            links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])
                        
                        print(f"   ✅ Pure diffuse CORTOPY material: Albedo={albedo_factor:.4f}, Roughness={body_roughness:.3f}")
                    
                else:
                    print(f"   ❌ Body {i}: No node tree found - CORTOPY compatibility issue")
            
            else:
                print(f"   ⚠️ Body {i}: CORTOPY create_branch_albedo_mix not available - using fallback")
                self._fallback_material_setup(material, i, roughness_values.get(i, 0.7), albedo_factors)

    def _fallback_material_setup(self, material, body_id, roughness, albedo_factors):
        """CORTOPY olmadan material setup (MONET uyumluluğu)"""
        if not material.node_tree:
            material.use_nodes = True
        
        nodes = material.node_tree.nodes
        nodes.clear()
        
        # Basit material tree
        output_node = nodes.new(type='ShaderNodeOutputMaterial')
        output_node.location = (200, 0)
        
        bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled') 
        bsdf_node.location = (0, 0)
        bsdf_node.inputs['Roughness'].default_value = roughness
        
        albedo_factor = albedo_factors.get(body_id, 0.127) if albedo_factors else 0.127
        bsdf_node.inputs['Base Color'].default_value = (albedo_factor, albedo_factor, albedo_factor, 1.0)
        
        material.node_tree.links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])
        
        print(f"   🔧 Fallback material Body {body_id}: R={roughness:.3f}, A={albedo_factor:.3f}")

    def _add_paths_to_state(self, state):
        """Add required paths to state"""
        state.add_path('albedo_path_1', os.path.join(state.path["input_path"], 'body', 'albedo', 'Phobos grayscale.jpg'))
        state.add_path('albedo_path_2', os.path.join(state.path["input_path"], 'body', 'albedo', 'mars_1k_color.jpg'))
        state.add_path('albedo_path_3', os.path.join(state.path["input_path"], 'body', 'albedo', 'Deimos grayscale.jpg'))
        state.add_path('uv_data_path_1', os.path.join(state.path["input_path"], 'body', 'uv data', 'g_phobos_036m_spc_0000n00000_v002.json'))
        state.add_path('uv_data_path_2', os.path.join(state.path["input_path"], 'body', 'uv data', 'Mars_65k.json'))
        state.add_path('uv_data_path_3', os.path.join(state.path["input_path"], 'body', 'uv data', 'g_deimos_162m_spc_0000n00000_v001.json'))

    def _setup_compositing(self, state):
        """Setup compositing for 16-bit EXR output (gamma applied in post-processing)
        Reference: HRSC L2 processing - linear render -> Python gamma conversion"""
        tree = corto.Compositing.create_compositing()
        render_node = corto.Compositing.rendering_node(tree, (0, 0))
        
        # Remove gamma node - will be applied in Python post-processing for correct L2 simulation
        # Create simple denoised output to EXR
        denoise_node = corto.Compositing.denoise_node(tree, (400, 0))
        composite_node = corto.Compositing.composite_node(tree, (600, 0))
        
        # Simple chain: render -> denoise -> composite (no gamma in Blender)
        tree.links.new(render_node.outputs["Image"], denoise_node.inputs["Image"]) 
        tree.links.new(denoise_node.outputs["Image"], composite_node.inputs["Image"])
        
        # Other branches (unchanged)
        corto.Compositing.create_depth_branch(tree, render_node)
        corto.Compositing.create_slopes_branch(tree, render_node, state)
        corto.Compositing.create_maskID_branch(tree, render_node, state)
        
        return tree
    def _apply_l2_lut_processing(self, linear_img_path, output_path):
        """Apply HRSC L2 LUT processing with correct gamma and electronic offset
        Reference: HRSC Level-2 processing pipeline simulation"""
        try:
            # Read 16-bit PNG linear data using imageio (reliable)
            import imageio.v3 as iio
            
            # Load 16-bit linear PNG 
            linear_img = iio.imread(str(linear_img_path))
            
            # Handle different image formats
            if len(linear_img.shape) == 3:
                # RGB image - take first channel
                dn_linear = linear_img[:,:,0].astype(np.float32)
            else:
                # Grayscale image
                dn_linear = linear_img.astype(np.float32)
            
            # Apply L2 LUT processing steps
            # 1) Normalize to 0-1 range
            max_val = dn_linear.max()
            if max_val > 0:
                dn_normalized = dn_linear / max_val
            else:
                dn_normalized = dn_linear
            
            # 2) Apply gamma correction (L2 characteristic)
            gamma_value = self.gamma_correction  # Use instance variable
            dn_gamma = np.power(dn_normalized, gamma_value)
            
            # 3) Scale to 8-bit range
            dn8_pre_offset = (dn_gamma * 255).astype(np.uint8)
            
            # 4) Apply electronic offset (HRSC characteristic)
            electronic_offset = 8  # DN units, reference: HRSC calibration documentation
            dn8_final = np.clip(dn8_pre_offset.astype(np.int16) - electronic_offset, 0, 255).astype(np.uint8)
            
            # Save processed image
            cv2.imwrite(str(output_path), dn8_final)
            
            print(f"   📐 L2 LUT processing applied:")
            print(f"      Linear max value: {max_val:.1f}")
            print(f"      Gamma correction: {gamma_value}")
            print(f"      Electronic offset: -{electronic_offset} DN")
            print(f"      Output range: {dn8_final.min()}-{dn8_final.max()} DN")
            
            return dn8_final
            
        except Exception as e:
            print(f"   ❌ L2 LUT processing failed: {e}")
            # Emergency fallback - simple 8-bit conversion
            try:
                import imageio.v3 as iio
                img = iio.imread(str(linear_img_path))
                if len(img.shape) == 3:
                    img = img[:,:,0]
                img_8bit = (img / img.max() * 255).astype(np.uint8) if img.max() > 0 else img.astype(np.uint8)
                cv2.imwrite(str(output_path), img_8bit)
                return img_8bit
            except:
                print(f"   ❌ Fallback conversion also failed")
                return None
            
        except ImportError:
            print("   ⚠️ OpenEXR not available, using fallback method")
            # Fallback using imageio
            import imageio
            dn_linear = imageio.v3.imread(str(exr_img_path))[:,:,0] 
            dn8 = 255 * np.power(dn_linear / dn_linear.max(), self.gamma_correction)
            dn8_final = np.clip(dn8 - 5, 0, 255).astype(np.uint8)
            cv2.imwrite(str(output_path), dn8_final)
            return dn8_final
            
        except Exception as e:
            print(f"   ❌ L2 LUT processing failed: {e}")
            return None
    def _setup_linear_render_settings(self, cam):
        """Configure camera and render settings for 16-bit PNG linear output
        Reference: Linear space rendering for accurate L2 LUT simulation"""
        # Set 16-bit PNG output format using CORTO camera configuration
        cam.toBlender()  # Apply camera settings to Blender
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.image_settings.color_depth = '16'  # ✅ Valid option
        bpy.context.scene.view_settings.view_transform = 'Raw'      # Linear output
        bpy.context.scene.render.image_settings.compression = 15   # PNG compression


    def run_single_simulation_and_validation(self, utc_time, real_img_path, img_filename, index, exposure_time_s):
        """Combined simulation and validation with L2 LUT processing - UPDATED WITH LITERATURE ROUGHNESS"""
        try:
            print(f"\n🔬 Starting simulation with literature-based roughness values...")
            
            # Setup and render (artık yeni roughness sistemi kullanıyor)
            state, env, cam, bodies, sun, materials, tree, spice_data = self.setup_scene_and_environment(utc_time, index, exposure_time_s)
            
            # ✅ YENİ: Material quality control
            print(f"   🎨 Material setup verification:")
            for i, material in enumerate(materials, 1):
                if material.node_tree:
                    bsdf_nodes = [n for n in material.node_tree.nodes if n.type == 'BSDF_PRINCIPLED']
                    if bsdf_nodes:
                        roughness_val = bsdf_nodes[0].inputs['Roughness'].default_value
                        print(f"      Body {i}: Verified roughness = {roughness_val:.3f}")
                    else:
                        print(f"      Body {i}: ⚠️ No Principled BSDF found")
                else:
                    print(f"      Body {i}: ⚠️ No node tree")
            
            # Configure for linear PNG output
            self._setup_linear_render_settings(cam)
            
            # Scale bodies
            bodies[0].set_scale(np.array([1.0, 1.0, 1.0]))      # Phobos
            bodies[1].set_scale(np.array([0.01, 0.01, 0.01]))  # Mars  bodies[1].set_scale(np.array([1000.0, 1000.0, 1000.0]))  # Mars  
            bodies[2].set_scale(np.array([1.0, 1.0, 1.0]))      # Deimos

            # Position and render to 16-bit PNG
            env.PositionAll(state, index=0)
            env.RenderOne(cam, state, index=index, depth_flag=True)

            # Get 16-bit PNG and apply L2 LUT processing  
            linear_img_path = Path(state.path["output_path"]) / "img" / f"{str(index).zfill(6)}.png" 
            synthetic_img_path = Path(state.path["output_path"]) / "img" / f"{str(index).zfill(6)}_L2.png"
            processed_img_path = None
            
            if linear_img_path.exists():
                # Apply L2 LUT processing to convert PNG -> L2 PNG
                l2_processed_img = self._apply_l2_lut_processing(linear_img_path, synthetic_img_path)
                
                if l2_processed_img is not None:
                    # Additional CORTO post-processing if needed
                    labels = {'CoB': [l2_processed_img.shape[1]//2, l2_processed_img.shape[0]//2], 
                            'range': float(np.linalg.norm(spice_data["phobos"]["position"])), 'phase_angle': 0.0}
                    processed_img, _ = self.post_processor.process_image_label_pair(
                        cv2.cvtColor(l2_processed_img, cv2.COLOR_GRAY2RGB), labels)
                    processed_img_path = synthetic_img_path.parent / f"processed_{synthetic_img_path.name}"
                    cv2.imwrite(str(processed_img_path), processed_img.astype(np.uint8))
            
            # Validation with L2-processed images
            validation_result = None
            if Path(real_img_path).exists() and synthetic_img_path.exists():
                synthetic_img_paths = [str(synthetic_img_path)]
                if processed_img_path and processed_img_path.exists():
                    synthetic_img_paths.append(str(processed_img_path))
                
                validation_result = self.validator.validate_with_complete_pipeline(
                    real_img_path, synthetic_img_paths, utc_time, img_filename)
                
                if validation_result:
                    validation_result['exposure_time_s'] = exposure_time_s
                    validation_result['l2_processing_applied'] = True
                    validation_result['literature_roughness_applied'] = True  # ✅ YENİ FIELD
            
            # Save blend file
            corto.Utils.save_blend(state, f'simulation_{index}_{img_filename.replace(".IMG", "")}')
            
            return {
                'index': index, 
                'utc_time': utc_time, 
                'img_filename': img_filename, 
                'real_img_path': real_img_path, 
                'synthetic_img_path': str(synthetic_img_path), 
                'processed_img_path': str(processed_img_path) if processed_img_path else None, 
                'spice_data': convert_numpy_types(spice_data), 
                'validation_result': validation_result, 
                'status': 'SUCCESS', 
                'literature_roughness_applied': True
            }
            
        except Exception as e:
            print(f"Error in L2 simulation with literature roughness: {e}")
            return {
                'index': index, 
                'utc_time': utc_time, 
                'img_filename': img_filename, 
                'status': 'FAILED', 
                'error': str(e), 
                'literature_roughness_applied': False
            }
            
    def run_complete_pipeline(self, pds_data_path=None, max_simulations=None):
        """Complete pipeline: process PDS -> run simulations -> save results"""
        try:
            # Process PDS database
            print("\n1. Processing PDS database...")
            img_database = self.pds_processor.parse_and_process_directory(pds_data_path or self.pds_data_path)
            
            # Save database
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(self.config['output_path'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                img_database.to_excel(output_dir / f"pds_img_database_{timestamp}.xlsx", index=False)
            except Exception:
                img_database.to_csv(output_dir / f"pds_img_database_{timestamp}.csv", index=False)
            
            # Run simulations
            print("\n2. Running simulations with validation...")
            valid_entries = img_database[img_database['STATUS'] == 'SUCCESS'].copy()
            if max_simulations:
                valid_entries = valid_entries.head(max_simulations)
            
            simulation_results = []
            for idx, row in valid_entries.iterrows():
                utc_time, img_filename, real_img_path, exposure_time = row['UTC_MEAN_TIME'], row['file_name'], row['file_path'], row['EXPOSURE_DURATION']
                print(f"\nProcessing {idx+1}/{len(valid_entries)}: {img_filename}")

                val_ms = float(exposure_time[:-4].strip())   # son 4 karakter  "<ms>"  düşer
                exposure_time_s  = val_ms / 1000.0                     # milisaniyeden saniyeye
                print("exposure duration (s):", exposure_time_s)
                result = self.run_single_simulation_and_validation(utc_time, real_img_path, img_filename, idx, exposure_time_s)
                if result:
                    simulation_results.append(result)
                
            # Save results
            print("\n3. Saving results...")
            results_path = output_dir / f"simulation_results_{timestamp}.json"
            with open(results_path, 'w') as f:
                json.dump(convert_numpy_types(simulation_results), f, indent=4)

            validation_summary = self.validator.get_validation_summary()
            with open(output_dir / f"validation_summary_{timestamp}.json", 'w') as f:
                json.dump(convert_numpy_types(validation_summary), f, indent=4)

            # Create detailed Excel report
            detailed_results = []
            for result in simulation_results:
                if result.get('validation_result'):
                    val = result['validation_result']
                    detailed_results.append({
                        'Index': result.get('index', ''),
                        'File_Name': result.get('img_filename', ''),
                        'UTC_Time': result.get('utc_time', ''),
                        'Exposure_Time_s': val.get('exposure_time_s', ''),  # Bu field eklenmeli
                        'Real_IMG_Path': result.get('real_img_path', ''),
                        'Synthetic_IMG_Path': result.get('synthetic_img_path', ''),
                        'SSIM_Score': val.get('best_ssim', 0),
                        'NCC_Score': val.get('best_ncc', 0),
                        'NRMSE_Score': val.get('best_nrmse', 1),
                        'Composite_Score': val.get('composite_score', 0),
                        'Validation_Status': val.get('validation_status', 'UNKNOWN'),
                        'Num_Synthetic_Images': val.get('num_synthetic_imgs', 0),
                        'Template_Matching_Results': val.get('template_matching_results', 0),
                        'M_Best_Selected': val.get('M_best_selected', 0),
                        'L_Final_Candidates': val.get('L_final_candidates', 0),
                        'Mask_Alignment': val.get('mask_alignment', False),
                        'Pipeline_Type': val.get('validation_pipeline', ''),
                        'Timestamp': val.get('timestamp', '')
                    })

            if detailed_results:
                try:
                    detailed_df = pd.DataFrame(detailed_results)
                    excel_path = output_dir / f"detailed_validation_scores_{timestamp}.xlsx"
                    detailed_df.to_excel(excel_path, index=False)
                    print(f"   📋 Detailed scores saved to: {excel_path}")
                except Exception as e:
                    print(f"   ⚠️ Excel save failed: {e}")
                    # CSV fallback
                    csv_path = output_dir / f"detailed_validation_scores_{timestamp}.csv"
                    detailed_df.to_csv(csv_path, index=False)
                    print(f"   📋 Detailed scores saved to CSV: {csv_path}")
            
            # Print summary
            print(f"\n📊 COMPACT VALIDATION SUMMARY:")
            print(f"   🔢 Total validations: {validation_summary['total_validations']}")
            print(f"   ✅ Successful validations: {validation_summary['successful_validations']}")
            print(f"   📈 Success rate: {validation_summary['success_rate']:.2%}")
            print(f"   📊 Average composite score: {validation_summary['average_composite_score']:.4f}")
            print(f"   🎯 Average SSIM: {validation_summary['average_ssim']:.4f}")
            
            return simulation_results, validation_summary
            
        except Exception as e:
            print(f"❌ Error in pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None, None

def main(pds_data_path=None, max_simulations=None, camera_type='SRC'):
    """Main function with compact simulator"""
    print("="*80)
    print("Compact Photometric Phobos Simulator with FIXED CORTO Validation")
    print("="*80)
    
    simulator = CompactPhotometricSimulator(
        camera_type   = args.camera_type,
        pds_data_path = args.pds_data_path,
        model         = args.model,          # ← self.model burada set edilir
        output_scale  = args.output_scale    # ← yeni parametre
    )
    
    simulation_results, validation_summary = simulator.run_complete_pipeline(pds_data_path, max_simulations)
    
    if validation_summary:
        print("\n✅ Compact simulation pipeline completed successfully!")
    else:
        print("\n❌ Pipeline failed!")

if __name__ == "__main__":
    # -------------------- CLI argümanları --------------------
    parser = argparse.ArgumentParser(
        description="Compact Photometric Phobos Simulator")

    parser.add_argument("--pds_data_path", default="/home/tt_mmx/corto/PDS_Data",
                        help="PDS IMG dizini")
    parser.add_argument("--max_simulations", type=int, default=5,
                        help="Çalıştırılacak maksimum IMG sayısı")
    parser.add_argument("--camera_type", default="SRC",
                        choices=["SRC", "NADIR", "COLOR"],
                        help="HRSC kamera kanalı / sensör tipi")
    parser.add_argument("--model", default="radiance",
                        choices=["radiance", "electron"],
                        help="Sun‑strength modeli: 'radiance' (irradiance/π) veya "
                            "'electron' (e- per pixel)")
    parser.add_argument("--output_scale", default="DN",
        choices=["DN", "ABS"],
        help="Render görüntüsünü Level‑2 DN8 (0‑255) veya "
            "mutlak radyans (W m⁻² sr⁻¹) olarak kaydet.")

    args = parser.parse_args()
    print(f"Starting simulator with:\n  PDS path  : {args.pds_data_path}"
        f"\n  Max sims  : {args.max_simulations}"
        f"\n  Camera    : {args.camera_type}"
        f"\n  Model     : {args.model}"
        f"\n  Output    : {args.output_scale}")

    simulator = CompactPhotometricSimulator(
        camera_type   = args.camera_type,
        pds_data_path = args.pds_data_path,
        model         = args.model          # ← self.model burada set edilir
    )

    simulation_results, validation_summary = simulator.run_complete_pipeline(
        pds_data_path  = args.pds_data_path,
        max_simulations= args.max_simulations
    )

    if validation_summary:
        print("\n✅ Pipeline completed successfully!")
    else:
        print("\n❌ Pipeline failed!")
