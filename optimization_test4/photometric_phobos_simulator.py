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

    def _apply_corto_pipeline_and_template_matching(self, real_img, synthetic_imgs, synthetic_img_paths):
        """Combined CORTO pipeline application and template matching"""
        # Default labels
        labels = {'CoB': [real_img.shape[1]//2, real_img.shape[0]//2], 'range': 1000.0, 'phase_angle': 0.0}
        
        # Process real image
        real_normalized = real_img if real_img.dtype == np.uint8 else ((real_img - real_img.min()) / (real_img.max() - real_img.min()) * 255.0).astype(np.uint8) if real_img.max() > real_img.min() else np.zeros_like(real_img, dtype=np.uint8)
        real_rgb = cv2.cvtColor(real_normalized, cv2.COLOR_GRAY2RGB) if len(real_normalized.shape) == 2 else real_normalized
        
        try:
            real_processed, _ = self.post_processor.process_image_label_pair(real_rgb, labels.copy())
        except Exception:
            real_processed = cv2.resize(real_normalized, (128, 128))
        
        real_processed = cv2.cvtColor(real_processed.astype(np.uint8), cv2.COLOR_RGB2GRAY) if len(real_processed.shape) == 3 else real_processed
        
        # Process synthetic images and perform template matching
        template_results = []
        for i, (synthetic_img, img_path) in enumerate(zip(synthetic_imgs, synthetic_img_paths)):
            # Process synthetic
            synthetic_normalized = synthetic_img if synthetic_img.dtype == np.uint8 else (synthetic_img / 255.0 * 255.0).astype(np.uint8)
            synthetic_rgb = cv2.cvtColor(synthetic_normalized, cv2.COLOR_GRAY2RGB) if len(synthetic_normalized.shape) == 2 else synthetic_normalized
            
            try:
                synthetic_processed, _ = self.post_processor.process_image_label_pair(synthetic_rgb, labels.copy())
            except Exception:
                synthetic_processed = cv2.resize(synthetic_normalized, (128, 128))
            
            synthetic_processed = cv2.cvtColor(synthetic_processed.astype(np.uint8), cv2.COLOR_RGB2GRAY) if len(synthetic_processed.shape) == 3 else synthetic_processed
            
            # Template matching
            correlation = cv2.matchTemplate(real_processed, synthetic_processed, cv2.TM_CCOEFF_NORMED)
            _, ncc, _, max_loc = cv2.minMaxLoc(correlation)
            
            # Enhanced mask correlation
            mask_correlation, mask_used = self._find_mask_and_calculate_correlation(real_processed, synthetic_processed, img_path)
            ncc = max(ncc, mask_correlation)
            
            # NRMSE
            mse = np.mean((real_processed.astype(np.float64) - synthetic_processed.astype(np.float64)) ** 2)
            img_range = real_processed.max() - real_processed.min()
            nrmse = np.sqrt(mse) / img_range if img_range > 0 else 0
            
            template_results.append({
                'index': i, 'real_processed': real_processed, 'synthetic_processed': synthetic_processed,
                'ncc': float(ncc), 'nrmse': float(nrmse), 'crop_location': max_loc, 'mask_used': mask_used
            })
        
        return template_results

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
        J = min(len(noise_combinations), 20)  # Limit for performance
        
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
    
    def __init__(self, config_path=None, camera_type='SRC', pds_data_path=None, model="radiance"):
        self.config = self._load_or_create_config(config_path)
        self.camera_type = camera_type
        self.pds_data_path = pds_data_path or self.config.get('pds_data_path', 'PDS_Data')
        self.model = model    
        
        # Initialize components
        self.spice_processor = SpiceDataProcessor(base_path=self.config['spice_data_path'])
        self.pds_processor = CompactPDSProcessor(self.pds_data_path)
        self.post_processor = CORTOPostProcessor(target_size=128)
        self.validator = CompactCORTOValidator(self.post_processor)
        
        self.scenario_name = "S07_Mars_Phobos_Deimos"
        
        # Get camera and photometric configuration
        self.camera_config = self._get_camera_config()
        #self.photometric_params = self._get_photometric_params() deleted for exposure time usage  
        
        print(f"✅ Compact simulator initialized with FIXED validation")

    @staticmethod
    def compute_sun_strength_photon(solar_distance_km,
                                    exposure_time_s,
                                    lambda_eff=675e-9,      # m
                                    pixel_area=(7e-6)**2,    # m²
                                    QE=0.30,
                                    return_mode="radiance"): # 'radiance' or 'electron'

        AU_KM          = 149_597_870.7
        SOLAR_CONSTANT = 1361.0                # W m^-2 @ 1 AU
        PI             = 3.141592653589793
        PLANCK_H       = 6.626_070_15e-34
        LIGHT_C        = 2.997_924_58e8

        distance_AU = solar_distance_km / AU_KM
        irradiance  = SOLAR_CONSTANT / distance_AU**2        # W m^-2

        radiance = (irradiance / PI) *0.20                           # W m^-2 sr^-1

        if return_mode == "radiance":
            return radiance

        # photon → electron
        E_photon   = PLANCK_H * LIGHT_C / lambda_eff
        photon_flux = irradiance / E_photon                  # photon m^-2 s^-1
        photons     = photon_flux * exposure_time_s          # photon m^-2
        electrons   = photons * pixel_area * QE              # e- per pixel

        return electrons

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
            'body_files': ['g_phobos_287m_spc_0000n00000_v002.obj', 'Mars_65k.obj', 'g_deimos_162m_spc_0000n00000_v001.obj'],
            'scene_file': 'scene_mmx.json',
            'geometry_file': 'geometry_mmx.json'
        }

    def _get_camera_config(self):
        """Get camera configuration with fallback"""
        try:
            return self.spice_processor.get_hrsc_camera_config(self.camera_type)
        except Exception:
            if self.camera_type == 'SRC':
                return {'fov': 0.54, 'res_x': 1024, 'res_y': 1024, 'film_exposure': exposure_time_s, 'sensor': 'BW', 'clip_start': 0.1, 'clip_end': 100000000.0, 'bit_encoding': '16', 'viewtransform': 'Standard', 'K': [[1222.0, 0, 512.0], [0, 1222.0, 512.0], [0, 0, 1]]}
            else:
                return {'fov': 11.9, 'res_x': 5184, 'res_y': 1, 'film_exposure': exposure_time_s, 'sensor': 'BW', 'clip_start': 0.1, 'clip_end': 10000.0, 'bit_encoding': '16', 'viewtransform': 'Standard', 'K': [[2500.0, 0, 2592.0], [0, 2500.0, 0.5], [0, 0, 1]]}

    def _get_photometric_params(self,
                                exposure_time_s: float = 1.0,
                                solar_distance_km: float | None = None,
                                model: str = "radiance"):
        """
        Photometric parameters based on physical solar constant.
        """
        # ------------------------------------------------------------
        # 1) Güneş–Phobos mesafesi
        # ------------------------------------------------------------
        solar_distance_km = solar_distance_km or AU_KM          # varsayılan 1 AU
        solar_distance_au = solar_distance_km / AU_KM           # ➊  ← HATA buradaydı
        irradiance        = SOLAR_CONSTANT_WM2 / solar_distance_au**2  # ➋

        # ------------------------------------------------------------
        # 2) Güç/radyans (veya elektron) hesabı
        # ------------------------------------------------------------
        sun_strength = self.compute_sun_strength_photon(
            solar_distance_km,
            exposure_time_s,
            return_mode=model
        )

        # ------------------------------------------------------------
        # 3) DEBUG çıktısı
        # ------------------------------------------------------------
        print("   🌞 Solar const model:")
        print(f"      Distance: {solar_distance_km/1e6:7.3f} Mkm "
            f"({solar_distance_au:4.3f} AU)")
        print(f"      Irradiance @Phobos: {irradiance:6.1f} W/m²")
        print(f"      Exposure time: {exposure_time_s:.4f} s")
        print(f"      Sun strength ({model}): {sun_strength:.3e}")

        # ------------------------------------------------------------
        # 4) Fonksiyon çıktısı
        # ------------------------------------------------------------
        return {
            "sun_strength"       : sun_strength,
            "model"              : model,
            "phobos_albedo"      : 0.068,
            "mars_albedo"        : 0.170,
            "deimos_albedo"      : 0.068,
            "sensor_aging_factor": 0.95,
            "brdf_model"         : "principled",
            "brdf_roughness"     : 0.5,
            "gamma_correction"   : 1.0,
            "exposure_time"      : exposure_time_s,
            "solar_distance_km"  : solar_distance_km,
            "solar_distance_au"  : solar_distance_au,
            "irradiance_Wm2"     : irradiance,
        }


    def setup_scene_and_environment(self, utc_time, index, exposure_time_s):
        """scene + environment (foton modeli entegre)"""
        print(f"\nSetting up scene for: {utc_time}")

        # ------------------------------------------------------------------
        # 1) SPICE verisi ve Güneş uzaklığı
        # ------------------------------------------------------------------
        corto.Utils.clean_scene()
        try:
            spice_data = self.spice_processor.get_spice_data(utc_time)
            solar_distance_km = spice_data.get("distances", {}).get("sun_to_phobos", 0)
        except Exception:
            spice_data = {...}                      #  ↞ sizin fallback dict’iniz
            solar_distance_km = 149_597_870.7       # 1 AU

        # ------------------------------------------------------------------
        # 2) Fotometrik parametreleri hesapla
        #    (self.model = "radiance" veya "electron")
        # ------------------------------------------------------------------
        photometric_params = self._get_photometric_params(
            exposure_time_s=exposure_time_s,
            solar_distance_km=solar_distance_km,
            model=self.model
        )
        sun_energy = float(photometric_params["sun_strength"])

        # ------------------------------------------------------------------
        # 3) Scene / geometry JSON’larını yaz
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
        # 4) CORTO State & Environment
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

        # ------------------------------------------------------------------
        # 5) Malzeme + kompoziting
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

        tree = self._setup_compositing(state)

        return state, env, cam, bodies, sun, materials, tree, spice_data


    def _add_paths_to_state(self, state):
        """Add required paths to state"""
        state.add_path('albedo_path_1', os.path.join(state.path["input_path"], 'body', 'albedo', 'Phobos grayscale.jpg'))
        state.add_path('albedo_path_2', os.path.join(state.path["input_path"], 'body', 'albedo', 'mars_1k_color.jpg'))
        state.add_path('albedo_path_3', os.path.join(state.path["input_path"], 'body', 'albedo', 'Deimos grayscale.jpg'))
        state.add_path('uv_data_path_1', os.path.join(state.path["input_path"], 'body', 'uv data', 'g_phobos_287m_spc_0000n00000_v002.json'))
        state.add_path('uv_data_path_2', os.path.join(state.path["input_path"], 'body', 'uv data', 'Mars_65k.json'))
        state.add_path('uv_data_path_3', os.path.join(state.path["input_path"], 'body', 'uv data', 'g_deimos_162m_spc_0000n00000_v001.json'))

    def _setup_compositing(self, state):
        """Setup compositing with mask ID support"""
        tree = corto.Compositing.create_compositing()
        render_node = corto.Compositing.rendering_node(tree, (0, 0))
        corto.Compositing.create_img_denoise_branch(tree, render_node)
        corto.Compositing.create_depth_branch(tree, render_node)
        corto.Compositing.create_slopes_branch(tree, render_node, state)
        corto.Compositing.create_maskID_branch(tree, render_node, state)
        return tree

    def run_single_simulation_and_validation(self, utc_time, real_img_path, img_filename, index, exposure_time_s):
        """Combined simulation and validation"""
        try:
            # Setup and render
            state, env, cam, bodies, sun, materials, tree, spice_data = self.setup_scene_and_environment(utc_time, index, exposure_time_s)
            
            # Scale bodies
            bodies[0].set_scale(np.array([1.0, 1.0, 1.0]))
            bodies[1].set_scale(np.array([1000.0, 1000.0, 1000.0]))
            bodies[2].set_scale(np.array([1.0, 1.0, 1.0]))
            
            # Position and render
            env.PositionAll(state, index=0)
            env.RenderOne(cam, state, index=index, depth_flag=True)
            
            # Get and process synthetic image
            synthetic_img_path = Path(state.path["output_path"]) / "img" / f"{str(index).zfill(6)}.png"
            processed_img_path = None
            
            if synthetic_img_path.exists():
                synthetic_img = cv2.imread(str(synthetic_img_path))
                if synthetic_img is not None:
                    labels = {'CoB': [synthetic_img.shape[1]//2, synthetic_img.shape[0]//2], 'range': float(np.linalg.norm(spice_data["phobos"]["position"])), 'phase_angle': 0.0}
                    processed_img, _ = self.post_processor.process_image_label_pair(synthetic_img, labels)
                    processed_img_path = synthetic_img_path.parent / f"processed_{synthetic_img_path.name}"
                    cv2.imwrite(str(processed_img_path), processed_img.astype(np.uint8))
            
            # Validate
            # Validate
            validation_result = None
            if Path(real_img_path).exists() and synthetic_img_path.exists():
                synthetic_img_paths = [str(synthetic_img_path)]
                if processed_img_path and processed_img_path.exists():
                    synthetic_img_paths.append(str(processed_img_path))
                
                validation_result = self.validator.validate_with_complete_pipeline(real_img_path, synthetic_img_paths, utc_time, img_filename)
                
                # Exposure time bilgisini validation_result'a ekle
                if validation_result:
                    validation_result['exposure_time_s'] = exposure_time_s
            
            # Save blend file
            corto.Utils.save_blend(state, f'simulation_{index}_{img_filename.replace(".IMG", "")}')
            
            return {'index': index, 'utc_time': utc_time, 'img_filename': img_filename, 'real_img_path': real_img_path, 'synthetic_img_path': str(synthetic_img_path), 'processed_img_path': str(processed_img_path) if processed_img_path else None, 'spice_data': convert_numpy_types(spice_data), 'validation_result': validation_result, 'status': 'SUCCESS'}
            
        except Exception as e:
            print(f"Error in simulation: {e}")
            return {'index': index, 'utc_time': utc_time, 'img_filename': img_filename, 'status': 'FAILED', 'error': str(e)}

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
    
    simulator = CompactPhotometricSimulator(camera_type=args.camera_type, pds_data_path=args.pds_data_path, model=args.model)
    
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

    args = parser.parse_args()            # ← artık 'args' var

    print(f"Starting simulator with:\n  PDS path  : {args.pds_data_path}"
          f"\n  Max sims  : {args.max_simulations}"
          f"\n  Camera    : {args.camera_type}"
          f"\n  Model     : {args.model}")

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
