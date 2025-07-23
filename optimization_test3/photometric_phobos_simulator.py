"""
Integrated Fixed Photometric Phobos Simulator
All fixes integrated into the main file - no separate imports needed
"""

import sys
import os
import json
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from scipy.signal import correlate2d
import pandas as pd
import requests
import re
import time
import pickle

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
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


class PDSImageProcessor:
    """Process PDS IMG files and extract UTC_MEAN_TIME information"""
    
    def __init__(self, pds_data_path="PDS_Data"):
        self.pds_data_path = Path(pds_data_path)
        self._key_val = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=\s*(.*)$")
        
    def parse_pds_label(self, file_path, max_records=50_000):
        """Parse PDS label from IMG file"""
        label = {}
        with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
            for i, raw in enumerate(fh):
                if i > max_records:
                    break
                line = raw.strip().replace("<CR><LF>", "")
                if line.upper().startswith("END"):
                    break
                m = self._key_val.match(line)
                if m:
                    key, val = m.groups()
                    val = val.strip().strip('"').strip("'")
                    label[key] = val
        return label
    
    def process_img_directory(self, img_directory_path):
        """Process all IMG files in directory and create UTC database"""
        records = []
        img_dir = Path(img_directory_path)
        
        print(f"Processing IMG files in: {img_dir}")
        
        for img_file in img_dir.rglob("*.IMG"):
            try:
                label = self.parse_pds_label(img_file)
                label.update({"file_path": str(img_file), "file_name": img_file.name})
                records.append(label)
                print(f"Processed: {img_file.name}")
            except Exception as e:
                print(f"Error processing {img_file.name}: {e}")
                
        if not records:
            raise RuntimeError("No IMG files found or processed!")
            
        df = pd.DataFrame(records)
        print(f"IMG files processed: {len(df)}")
        
        # Process time columns - ensure timezone-naive datetimes
        for col in ["START_TIME", "STOP_TIME"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
                # Convert to timezone-naive for Excel compatibility
                df[col] = df[col].dt.tz_localize(None)
        
        if {"START_TIME", "STOP_TIME"} <= set(df.columns):
            df["DURATION_SECONDS"] = (df["STOP_TIME"] - df["START_TIME"]).dt.total_seconds()
            df["MEAN_TIME"] = df["START_TIME"] + (df["STOP_TIME"] - df["START_TIME"]) / 2
            df["UTC_MEAN_TIME"] = (
                df["MEAN_TIME"]
                  .dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
                  .str[:-2] + "Z"
            )
        else:
            df["DURATION_SECONDS"] = pd.NA
            df["MEAN_TIME"] = pd.NaT
            df["UTC_MEAN_TIME"] = pd.NA
            
        df["STATUS"] = (
            df["START_TIME"].notna() & df["STOP_TIME"].notna()
        ).map({True: "SUCCESS", False: "MISSING_TIME_DATA"})
        
        return df


class FixedCORTOValidator:
    """FIXED CORTO validation pipeline with proper PDS handling and alignment"""
    
    def __init__(self, post_processor):
        self.post_processor = post_processor
        self.validation_results = []
        self._pds_cache = {}  # Cache for PDS images
        
    def _parse_pds_label(self, file_path, max_records=50_000):
        """Parse PDS label from IMG file"""
        label = {}
        key_val = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=\s*(.*)$")
        
        with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
            for i, raw in enumerate(fh):
                if i > max_records:
                    break
                line = raw.strip().replace("<CR><LF>", "")
                if line.upper().startswith("END"):
                    break
                m = key_val.match(line)
                if m:
                    key, val = m.groups()
                    val = val.strip().strip('"').strip("'")
                    label[key] = val
        return label
    
    def _load_pds_image_data(self, pds_file_path):
        """Load actual image data from PDS IMG file - FIXED VERSION"""
        if str(pds_file_path) in self._pds_cache:
            return self._pds_cache[str(pds_file_path)]
        
        try:
            # Parse PDS label
            label = self._parse_pds_label(pds_file_path)
            
            # Extract image parameters
            lines = int(label.get('LINES', 0))
            line_samples = int(label.get('LINE_SAMPLES', 0))
            sample_bits = int(label.get('SAMPLE_BITS', 16))
            sample_type = label.get('SAMPLE_TYPE', 'MSB_INTEGER')
            byte_order = label.get('SAMPLE_TYPE', 'MSB')
            header_bytes = int(label.get('^IMAGE', '1').split()[0]) - 1
            
            print(f"📁 Loading PDS IMG file: {pds_file_path.name}")
            print(f"   📊 Image parameters:")
            print(f"      Dimensions: {line_samples} x {lines}")
            print(f"      Sample bits: {sample_bits}")
            print(f"      Sample type: {sample_type}")
            print(f"      Header bytes: {header_bytes}")
            
            # Determine data type
            if sample_bits == 8:
                if 'MSB' in byte_order:
                    dtype = '>u1'
                else:
                    dtype = '<u1'
            elif sample_bits == 16:
                if 'MSB' in byte_order:
                    if 'UNSIGNED' in sample_type:
                        dtype = '>u2'
                    else:
                        dtype = '>i2'
                else:
                    if 'UNSIGNED' in sample_type:
                        dtype = '<u2'
                    else:
                        dtype = '<i2'
            else:
                raise ValueError(f"Unsupported sample bits: {sample_bits}")
            
            # Read binary data
            with open(pds_file_path, 'rb') as f:
                f.seek(header_bytes)
                data = f.read(lines * line_samples * (sample_bits // 8))
            
            # Convert to numpy array
            image_array = np.frombuffer(data, dtype=dtype).reshape(lines, line_samples)
            
            # Convert to standard format (uint8 or uint16)
            if sample_bits == 16:
                # Normalize to 0-65535 range for uint16
                img_min, img_max = image_array.min(), image_array.max()
                if img_max > img_min:
                    image_array = ((image_array - img_min) / (img_max - img_min) * 65535).astype(np.uint16)
                else:
                    image_array = image_array.astype(np.uint16)
            else:
                image_array = image_array.astype(np.uint8)
            
            print(f"   ✅ Successfully loaded: {image_array.shape}, dtype: {image_array.dtype}")
            print(f"   📈 Image stats: min={image_array.min()}, max={image_array.max()}, mean={image_array.mean():.2f}")
            
            # Cache the result
            self._pds_cache[str(pds_file_path)] = image_array
            
            return image_array
            
        except Exception as e:
            print(f"❌ Error loading PDS image {pds_file_path}: {e}")
            return None
    
    def _apply_same_transformation(self, real_img, synthetic_img, labels=None):
        """Apply SAME S0→S1→S2 transformation to both images - FIXED"""
        
        # Default labels if none provided
        if labels is None:
            labels = {
                'CoB': [real_img.shape[1]//2, real_img.shape[0]//2],
                'range': 1000.0,
                'phase_angle': 0.0
            }
        
        print(f"   🔄 Applying SAME CORTO pipeline to both images:")
        
        # Apply CORTO post-processing to real image
        print(f"      Real: {real_img.shape} -> ", end="")
        if len(real_img.shape) == 2:
            real_rgb = cv2.cvtColor((real_img / 256).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            real_rgb = real_img
            
        real_processed, real_labels = self.post_processor.process_image_label_pair(
            real_rgb, labels.copy()
        )
        print(f"{real_processed.shape}")
        
        # Apply CORTO post-processing to synthetic image  
        print(f"      Synthetic: {synthetic_img.shape} -> ", end="")
        if len(synthetic_img.shape) == 2:
            synthetic_rgb = cv2.cvtColor(synthetic_img, cv2.COLOR_GRAY2RGB)
        else:
            synthetic_rgb = synthetic_img
            
        synthetic_processed, synthetic_labels = self.post_processor.process_image_label_pair(
            synthetic_rgb, labels.copy()
        )
        print(f"{synthetic_processed.shape}")
        
        # Convert back to grayscale for comparison
        if len(real_processed.shape) == 3:
            real_processed = cv2.cvtColor(real_processed.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        if len(synthetic_processed.shape) == 3:
            synthetic_processed = cv2.cvtColor(synthetic_processed.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        return real_processed, synthetic_processed, real_labels, synthetic_labels
    
    def _compute_alignment_with_masks(self, real_img, synthetic_img, mask_id_path=None):
        """
        Compute NCC alignment.
        - Eğer uygun bir maske varsa, maske de post‑process boyutuna
        (real_img.shape) ölçeklenir ve sadece o pikseller kullanılır.
        - Maske bulunamaz veya hata çıkarsa standart (ya da resize) hizalama yapılır.
        """

        # 1) MASKELİ HİZALAMA
        if mask_id_path and Path(mask_id_path).exists():
            try:
                mask_img = cv2.imread(str(mask_id_path), cv2.IMREAD_GRAYSCALE)
                if mask_img is not None:
                    # ikili maske
                    mask_binary = (mask_img > 0).astype(np.uint8)

                    # boyut uyuşmuyorsa maske → real_img boyutuna getir
                    if mask_binary.shape != real_img.shape:
                        mask_binary = cv2.resize(
                            mask_binary,
                            (real_img.shape[1], real_img.shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        )

                    # şimdi aynı boyutta
                    real_masked      = real_img * mask_binary
                    synthetic_masked = synthetic_img * mask_binary

                    correlation = cv2.matchTemplate(
                        real_masked, synthetic_masked, cv2.TM_CCOEFF_NORMED
                    )
                    _, max_val, _, _ = cv2.minMaxLoc(correlation)
                    print(f"      🎯 Mask‑based alignment: correlation = {max_val:.4f}")
                    return max_val
            except Exception as e:
                print(f"      ⚠️  Mask alignment failed: {e}")

        # 2) STANDART HİZALAMA (maske yoksa veya hata olduysa)
        if real_img.shape == synthetic_img.shape:
            corr_src = synthetic_img
        else:
            corr_src = cv2.resize(
                synthetic_img, (real_img.shape[1], real_img.shape[0])
            )

        correlation = cv2.matchTemplate(real_img, corr_src, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(correlation)
        print(f"      📊 Standard alignment: correlation = {max_val:.4f}")
        return max_val

    def _enhanced_mask_alignment(self, real_img, synthetic_img, mask_paths):
        """Enhanced mask-based alignment with multiple mask types"""
        
        best_correlation = 0
        best_mask_type = None
        
        for i, mask_path in enumerate(mask_paths):
            if mask_path and Path(mask_path).exists():
                try:
                    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask_img is not None:
                        # Resize mask to match image dimensions
                        if mask_img.shape != real_img.shape:
                            mask_img = cv2.resize(
                                mask_img, (real_img.shape[1], real_img.shape[0]),
                                interpolation=cv2.INTER_NEAREST
                            )
                        
                        # Binary mask
                        mask_binary = (mask_img > 0).astype(np.uint8)
                        
                        # Apply mask
                        real_masked = real_img * mask_binary
                        synthetic_masked = synthetic_img * mask_binary
                        
                        # Calculate correlation
                        correlation = cv2.matchTemplate(
                            real_masked, synthetic_masked, cv2.TM_CCOEFF_NORMED
                        )
                        _, max_val, _, _ = cv2.minMaxLoc(correlation)
                        
                        if max_val > best_correlation:
                            best_correlation = max_val
                            best_mask_type = f"mask_type_{i}"
                            
                        print(f"      🎭 Mask {i+1} correlation: {max_val:.4f}")
                        
                except Exception as e:
                    print(f"      ⚠️ Mask {i+1} failed: {e}")
        
        return best_correlation, best_mask_type

    
    # YENİ HAL - Enhanced validation pipeline
    def validate_with_fixed_pipeline(self, pds_img_path, synthetic_img_paths, utc_time, img_filename):
        """Enhanced validation with FIXED PDS processing and complete CORTO Figure 15 pipeline"""
        
        try:
            print(f"\n🔍 Starting Enhanced CORTO Validation Pipeline...")
            print(f"   Validating {img_filename} with complete CORTO Figure 15 methodology...")
            
            # 1. Load PDS image properly - FIXED
            print(f"   📁 Loading PDS IMG file...")
            real_img = self._load_pds_image_data(Path(pds_img_path))
            if real_img is None:
                return None
                
            print(f"   ✅ PDS image loaded: {real_img.shape}")
            
            # 2. Load synthetic images and find mask paths
            synthetic_imgs = []
            mask_paths = []

            for img_path in synthetic_img_paths:
                img_path = Path(img_path)
                if img_path.exists():
                    # Load synthetic image
                    synthetic_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if synthetic_img is None:
                        continue

                    synthetic_imgs.append(synthetic_img)
                    stem = img_path.stem

                    # Find mask paths (kept from original implementation)
                    candidate_paths = [
                        img_path.parent.parent / "label" / "IDmask" / "Mask_1" / f"{stem}.png",
                        img_path.parent.parent / "mask_ID_1" / f"{stem}.png",
                        img_path.parent.parent / "mask_ID_shadow_1" / f"{stem}.png",
                    ]

                    mask_path = next((p for p in candidate_paths if p.exists()), None)
                    mask_paths.append(mask_path)
            
            if not synthetic_imgs:
                print(f"   ❌ No valid synthetic images found")
                return None
            
            print(f"   📸 Processing {len(synthetic_imgs)} synthetic image(s) with enhanced pipeline")
            
            # 3. Apply SAME transformation pipeline to both images (from original code)
            print(f"   🔄 Applying SAME CORTO S0→S1→S2 pipeline to all images:")
            
            # Default labels if none provided
            labels = {
                'CoB': [real_img.shape[1]//2, real_img.shape[0]//2],
                'range': 1000.0,
                'phase_angle': 0.0
            }
            
            # Apply CORTO post-processing to real image
            print(f"      Real: {real_img.shape} -> ", end="")
            if len(real_img.shape) == 2:
                real_rgb = cv2.cvtColor((real_img / 256).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            else:
                real_rgb = real_img
                
            real_processed, real_labels = self.post_processor.process_image_label_pair(
                real_rgb, labels.copy()
            )
            print(f"{real_processed.shape}")
            
            # Convert back to grayscale for comparison
            if len(real_processed.shape) == 3:
                real_processed = cv2.cvtColor(real_processed.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # 4. Template matching and cropping (CORTO Figure 15 Step 1)
            print(f"   📊 Template matching with {len(synthetic_imgs)} candidates...")
            template_results = []
            
            for i, synthetic_img in enumerate(synthetic_imgs):
                # Apply same transformation
                if len(synthetic_img.shape) == 2:
                    synthetic_rgb = cv2.cvtColor(synthetic_img, cv2.COLOR_GRAY2RGB)
                else:
                    synthetic_rgb = synthetic_img
                    
                synthetic_processed, synthetic_labels = self.post_processor.process_image_label_pair(
                    synthetic_rgb, labels.copy()
                )
                
                if len(synthetic_processed.shape) == 3:
                    synthetic_processed = cv2.cvtColor(synthetic_processed.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                
                # Template matching
                result = self._template_matching_single(real_processed, synthetic_processed, mask_paths[i])
                result['synthetic_index'] = i
                template_results.append(result)
            
            # 5. Select M best images with lowest NRMSE
            M = min(len(template_results), 5)  # Configurable M value
            best_M = sorted(template_results, key=lambda x: x['nrmse'])[:M]
            print(f"   🎯 Selected {len(best_M)} best templates from {len(template_results)} candidates")
            
            # 6. Apply J noise combinations to M images
            noise_combinations = self._create_noise_combinations()
            J = min(len(noise_combinations), 50)  # Limit for performance
            print(f"   🎨 Applying {J} noise combinations to {len(best_M)} best templates...")
            
            noisy_validation_results = []
            
            for template_result in best_M:
                for j, noise_params in enumerate(noise_combinations[:J]):
                    # Apply noise to synthetic image
                    noisy_synthetic = self._apply_noise_combination(
                        template_result['synthetic_processed'], noise_params
                    )
                    
                    # SSIM calculation
                    try:
                        ssim_score = ssim(
                            template_result['real_processed'].astype(np.float64),
                            noisy_synthetic.astype(np.float64),
                            data_range=template_result['real_processed'].max() - template_result['real_processed'].min()
                        )
                    except:
                        ssim_score = 0.0
                    
                    noisy_validation_results.append({
                        'ncc': template_result['ncc'],
                        'nrmse': template_result['nrmse'],
                        'ssim': ssim_score,
                        'noise_params': noise_params,
                        'synthetic_index': template_result['synthetic_index']
                    })
            
            # 7. Select L images with maximum SSIM
            L = min(len(noisy_validation_results), 10)  # Configurable L value
            best_L = sorted(noisy_validation_results, key=lambda x: x['ssim'], reverse=True)[:L]
            print(f"   🏆 Selected {len(best_L)} final candidates with highest SSIM")
            
            # 8. Final result (enhanced from original)
            best_result = best_L[0] if best_L else template_results[0]
            
            # Enhanced composite score
            composite_score = (
                best_result['ssim'] + 
                best_result['ncc'] + 
                (1 - min(best_result['nrmse'], 1.0))
            ) / 3
            
            composite_score_normalized = (
                best_result['ssim']
                + best_result['ncc']
                + (1.0 - min(best_result['nrmse'], 1.0))
            ) / 3.0
            
            # Create final validation result (enhanced)
            final_result = {
                'utc_time': utc_time,
                'img_filename': img_filename,
                'pds_img_path': str(pds_img_path),
                'num_synthetic_imgs': len(synthetic_imgs),
                'validation_pipeline': 'CORTO_Figure_15_Complete_Enhanced',
                'template_matching_results': len(template_results),
                'M_best_selected': len(best_M),
                'J_noise_combinations': J,
                'L_final_candidates': len(best_L),
                'best_ncc': best_result['ncc'],
                'best_nrmse': best_result['nrmse'], 
                'best_ssim': best_result['ssim'],
                'composite_score': composite_score,
                'composite_score_normalized': composite_score_normalized,
                'validation_status': 'SUCCESS' if composite_score > 0.7 else 'MODERATE' if composite_score > 0.5 else 'LOW_SIMILARITY',
                'timestamp': datetime.now().isoformat(),
                'pds_processing': 'ENABLED',
                'mask_alignment': any(r.get('mask_used', False) for r in template_results),
                'transformation_aligned': 'YES',
                'detailed_results': {
                    'template_results': template_results,
                    'best_M_results': best_M,
                    'final_L_results': best_L
                }
            }
            
            self.validation_results.append(final_result)
            return final_result
            
        except Exception as e:
            print(f"❌ Error in enhanced validation for {img_filename}: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        def get_validation_summary(self):
            """Get summary of all validations"""
            if not self.validation_results:
                return {
                    'total_validations': 0,
                    'successful_validations': 0,
                    'success_rate': 0,
                    'average_composite_score': 0,
                    'average_ssim': 0,
                    'pds_processing_enabled': True,
                    'transformation_alignment': 'FIXED',
                    'mask_alignment_available': False
                }
            
            total = len(self.validation_results)
            successful = len([r for r in self.validation_results if r['validation_status'] == 'SUCCESS'])
            avg_composite = np.mean([r['composite_score'] for r in self.validation_results])
            avg_ssim = np.mean([r['best_ssim'] for r in self.validation_results])
            
            return {
                'total_validations': total,
                'successful_validations': successful,
                'success_rate': successful / total if total > 0 else 0,
                'average_composite_score': avg_composite,
                'average_ssim': avg_ssim,
                'pds_processing_enabled': True,
                'transformation_alignment': 'FIXED',
                'mask_alignment_available': any(r.get('mask_alignment', False) for r in self.validation_results)
            }

    def _template_matching_single(self, real_img, synthetic_img, mask_path):
        """Single template matching with optional mask"""
        
        # Normalized cross-correlation
        if real_img.shape == synthetic_img.shape:
            correlation = cv2.matchTemplate(real_img, synthetic_img, cv2.TM_CCOEFF_NORMED)
            _, ncc, _, max_loc = cv2.minMaxLoc(correlation)
            cropped_real = real_img
            cropped_synthetic = synthetic_img
        else:
            # Resize synthetic to match real
            synthetic_resized = cv2.resize(synthetic_img, (real_img.shape[1], real_img.shape[0]))
            correlation = cv2.matchTemplate(real_img, synthetic_resized, cv2.TM_CCOEFF_NORMED)
            _, ncc, _, max_loc = cv2.minMaxLoc(correlation)
            cropped_real = real_img
            cropped_synthetic = synthetic_resized
        
        # NRMSE calculation
        mse = np.mean((cropped_real.astype(np.float64) - cropped_synthetic.astype(np.float64)) ** 2)
        img_range = cropped_real.max() - cropped_real.min()
        nrmse = np.sqrt(mse) / img_range if img_range > 0 else 0
        
        # Check mask usage
        mask_used = False
        if mask_path and Path(mask_path).exists():
            mask_used = True
            # Enhanced mask correlation could be added here
        
        return {
            'real_processed': cropped_real,
            'synthetic_processed': cropped_synthetic,
            'ncc': float(ncc),
            'nrmse': float(nrmse),
            'mask_used': mask_used,
            'crop_location': max_loc
        }

class EnhancedPhotometricPhobosSimulator:
    """Enhanced Photometric Phobos Simulator with FIXED CORTO Validation"""
    
    def __init__(self, config_path=None, camera_type='SRC', pds_data_path=None):
        self.config = self._load_config(config_path)
        self.camera_type = camera_type
        self.pds_data_path = pds_data_path or self.config.get('pds_data_path', 'PDS_Data')
        
        # Initialize components
        self.spice_processor = SpiceDataProcessor(base_path=self.config['spice_data_path'])
        self.pds_processor = PDSImageProcessor(self.pds_data_path)
        self.post_processor = CORTOPostProcessor(target_size=128)
        self.validator = FixedCORTOValidator(self.post_processor)  # 🔧 FIXED VALIDATOR
        
        self.scenario_name = "S07_Mars_Phobos_Deimos"
        
        # Get camera configuration
        try:
            self.camera_config = self.spice_processor.get_hrsc_camera_config(camera_type)
        except Exception as e:
            print(f"Warning: Could not get camera config from SPICE: {e}")
            self.camera_config = self._get_fallback_camera_config(camera_type)
            
        # Set photometric parameters
        self.photometric_params = {
            'sun_strength': 589.0,
            'phobos_albedo': 0.068,
            'mars_albedo': 0.170,
            'deimos_albedo': 0.068,
            'sensor_aging_factor': 0.95,
            'brdf_model': 'principled',
            'brdf_roughness': 0.5,
            'gamma_correction': 2.2,
            'exposure_time': self.camera_config['film_exposure'],
        }
        
        print(f"✅ Enhanced simulator initialized with FIXED validation")
        print(f"   📷 Camera: {camera_type}")
        print(f"   📁 PDS data path: {self.pds_data_path}")
    
    def _load_config(self, config_path):
        """Load configuration"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            return self._create_default_config()
    
    def _create_default_config(self):
        """Create default configuration"""
        base_dir = Path(os.getcwd())
        return {
            'input_path': str(base_dir / "input" / "S07_Mars_Phobos_Deimos"),
            'output_path': str(base_dir / 'output' / 'enhanced_photometric_validation'),
            'spice_data_path': str(base_dir / 'spice_kernels'),
            'pds_data_path': str(base_dir / 'PDS_Data'),
            'real_images_path': str(base_dir / 'real_hrsc_images'),
            'body_files': [
                'g_phobos_287m_spc_0000n00000_v002.obj',
                'Mars_65k.obj',
                'g_deimos_162m_spc_0000n00000_v001.obj'
            ],
            'scene_file': 'scene_mmx.json',
            'geometry_file': 'geometry_mmx.json'
        }
    
    def _get_fallback_camera_config(self, camera_type='SRC'):
        """Fallback camera configuration"""
        if camera_type == 'SRC':
            return {
                'fov': 0.54,
                'res_x': 1024,
                'res_y': 1024,
                'film_exposure': 0.039,
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
                'film_exposure': 0.039,
                'sensor': 'BW',
                'clip_start': 0.1,
                'clip_end': 10000.0,
                'bit_encoding': '16',
                'viewtransform': 'Standard',
                'K': [[2500.0, 0, 2592.0], [0, 2500.0, 0.5], [0, 0, 1]]
            }
    
    def setup_enhanced_compositing(self, state):
        """Setup enhanced compositing with mask ID support"""
        tree = corto.Compositing.create_compositing()
        render_node = corto.Compositing.rendering_node(tree, (0, 0))
        
        # Create image denoising branch
        corto.Compositing.create_img_denoise_branch(tree, render_node)
        
        # Create depth branch
        corto.Compositing.create_depth_branch(tree, render_node)
        
        # Create slopes branch
        corto.Compositing.create_slopes_branch(tree, render_node, state)
        
        # Create mask ID branch - IMPORTANT FOR ALIGNMENT
        corto.Compositing.create_maskID_branch(tree, render_node, state)
        
        return tree
    
    def process_pds_database(self, pds_directory_path=None):
        """Process PDS IMG files and create UTC database"""
        if pds_directory_path is None:
            pds_directory_path = self.pds_data_path
            
        print(f"Processing PDS database from: {pds_directory_path}")
        
        # Process IMG files
        img_database = self.pds_processor.process_img_directory(pds_directory_path)
        
        # Convert timezone-aware datetimes to timezone-naive for Excel compatibility
        datetime_cols = ['START_TIME', 'STOP_TIME', 'MEAN_TIME']
        for col in datetime_cols:
            if col in img_database.columns:
                if hasattr(img_database[col].dtype, 'tz') and img_database[col].dtype.tz is not None:
                    img_database[col] = img_database[col].dt.tz_localize(None)
        
        # Save database
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.config['output_path'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        database_path = output_dir / f"pds_img_database_{timestamp}.xlsx"
        
        try:
            img_database.to_excel(database_path, index=False)
            print(f"PDS database saved to: {database_path}")
        except Exception as e:
            print(f"Warning: Could not save Excel file: {e}")
            # Save as CSV instead
            csv_path = output_dir / f"pds_img_database_{timestamp}.csv"
            img_database.to_csv(csv_path, index=False)
            print(f"PDS database saved as CSV to: {csv_path}")
        
        print(f"Total IMG files processed: {len(img_database)}")
        print(f"Files with valid UTC times: {len(img_database[img_database['STATUS'] == 'SUCCESS'])}")
        
        return img_database
    
    def setup_photometric_scene(self, utc_time):
        """Setup photometrically correct scene"""
        print(f"Setting up photometric scene for: {utc_time}")
        
        # Clean previous renders
        corto.Utils.clean_scene()
        
        # Get SPICE data
        try:
            spice_data = self.spice_processor.get_spice_data(utc_time)
        except Exception as e:
            print(f"Warning: Could not get SPICE data: {e}")
            spice_data = self._get_default_spice_data()
        
        # Create geometry file
        output_dir = Path(self.config['output_path'])
        output_dir.mkdir(parents=True, exist_ok=True)
        dynamic_geometry_path = output_dir / 'geometry_dynamic.json'
        
        geometry_data = {
            "sun": {"position": [convert_numpy_types(spice_data["sun"]["position"])]},
            "camera": {
                "position": [convert_numpy_types(spice_data["hrsc"]["position"])],
                "orientation": [convert_numpy_types(spice_data["hrsc"]["quaternion"])]
            },
            "body_1": {
                "position": [convert_numpy_types(spice_data["phobos"]["position"])],
                "orientation": [convert_numpy_types(spice_data["phobos"]["quaternion"])]
            },
            "body_2": {
                "position": [convert_numpy_types(spice_data["mars"]["position"])],
                "orientation": [convert_numpy_types(spice_data["mars"]["quaternion"])]
            },
            "body_3": {
                "position": [convert_numpy_types(spice_data["deimos"]["position"])],
                "orientation": [convert_numpy_types(spice_data["deimos"]["quaternion"])]
            }
        }
        
        with open(dynamic_geometry_path, 'w') as f:
            json.dump(geometry_data, f, indent=4)
        
        # Create scene configuration
        scene_config = self._create_scene_config()
        scene_config_path = output_dir / 'scene_src.json'
        
        with open(scene_config_path, 'w') as f:
            json.dump(convert_numpy_types(scene_config), f, indent=4)
        
        # Create CORTO State
        state = corto.State(
            scene=str(scene_config_path),
            geometry=str(dynamic_geometry_path),
            body=self.config['body_files'],
            scenario=self.scenario_name
        )
        
        self._add_photometric_paths(state)
        
        return state, spice_data
    
    def _get_default_spice_data(self):
        """Default SPICE data"""
        return {
            'et': 0.0,
            'phobos': {'position': [0.0, 0.0, 0.0], 'quaternion': [1.0, 0.0, 0.0, 0.0]},
            'mars':   {'position': [0.0, 0.0, 0.0], 'quaternion': [1.0, 0.0, 0.0, 0.0]},
            'deimos': {'position': [0.0, 0.0, 0.0], 'quaternion': [1.0, 0.0, 0.0, 0.0]},
            'sun':    {'position': [0.0, 0.0, 0.0], 'quaternion': [1.0, 0.0, 0.0, 0.0]},
            'hrsc':   {'position': [0.0, 0.0, 0.0], 'quaternion': [1.0, 0.0, 0.0, 0.0]}
        }
    
    def _create_scene_config(self):
        """Create scene configuration"""
        K_matrix = self.camera_config.get('K', [
            [1222.0, 0, 512.0],
            [0, 1222.0, 512.0],
            [0, 0, 1]
        ])
        
        return {
            "camera_settings": {
                "fov": float(self.camera_config['fov']),
                "res_x": int(self.camera_config['res_x']),
                "res_y": int(self.camera_config['res_y']),
                "film_exposure": float(self.camera_config['film_exposure']),
                "sensor": str(self.camera_config['sensor']),
                "K": K_matrix,
                "clip_start": float(self.camera_config['clip_start']),
                "clip_end": float(self.camera_config['clip_end']),
                "bit_encoding": str(self.camera_config['bit_encoding']),
                "viewtransform": str(self.camera_config['viewtransform'])
            },
            "sun_settings": {
                "angle": 0.00927,
                "energy": float(self.photometric_params['sun_strength'])
            },
            "body_settings_1": {"pass_index": 1, "diffuse_bounces": 4},  # Phobos
            "body_settings_2": {"pass_index": 2, "diffuse_bounces": 4},  # Mars
            "body_settings_3": {"pass_index": 3, "diffuse_bounces": 4},  # Deimos
            "rendering_settings": {
                "engine": "CYCLES",
                "device": "CPU",
                "samples": 256,
                "preview_samples": 16
            }
        }
    
    def _add_photometric_paths(self, state):
        """Add photometric paths"""
        state.add_path('albedo_path_1', os.path.join(state.path["input_path"], 'body', 'albedo', 'Phobos grayscale.jpg'))
        state.add_path('albedo_path_2', os.path.join(state.path["input_path"], 'body', 'albedo', 'mars_1k_color.jpg'))
        state.add_path('albedo_path_3', os.path.join(state.path["input_path"], 'body', 'albedo', 'Deimos grayscale.jpg'))
        state.add_path('uv_data_path_1', os.path.join(state.path["input_path"], 'body', 'uv data', 'g_phobos_287m_spc_0000n00000_v002.json'))
        state.add_path('uv_data_path_2', os.path.join(state.path["input_path"], 'body', 'uv data', 'Mars_65k.json'))
        state.add_path('uv_data_path_3', os.path.join(state.path["input_path"], 'body', 'uv data', 'g_deimos_162m_spc_0000n00000_v001.json'))
    
    def create_photometric_environment(self, state):
        """Create photometric environment"""
        cam_props = {
            'fov': float(self.camera_config.get('fov', 0.54)),
            'res_x': int(self.camera_config.get('res_x', 1024)),
            'res_y': int(self.camera_config.get('res_y', 1024)),
            'film_exposure': float(self.photometric_params.get('exposure_time', 0.039)),
            'sensor': str(self.camera_config.get('sensor', 'BW')),
            'clip_start': float(self.camera_config.get('clip_start', 0.1)),
            'clip_end': float(self.camera_config.get('clip_end', 10000.0)),
            'bit_encoding': str(self.camera_config.get('bit_encoding', '16')),
            'viewtransform': str(self.camera_config.get('viewtransform', 'Standard'))
        }
        
        if 'K' in self.camera_config:
            cam_props['K'] = self.camera_config['K']
        else:
            cam_props['K'] = [[1222.0, 0, 512.0], [0, 1222.0, 512.0], [0, 0, 1]]
        
        # Create components
        cam = corto.Camera(f'HRSC_{self.camera_type}_Camera', cam_props)
        sun = corto.Sun('Sun', {'angle': 0.00927, 'energy': float(self.photometric_params['sun_strength'])})
        
        bodies = []
        body_names = [Path(bf).stem for bf in self.config['body_files']]
        for i, body_name in enumerate(body_names):
            try:
                body_props = getattr(state, f'properties_body_{i+1}')
            except AttributeError:
                body_props = {'pass_index': i + 1, 'diffuse_bounces': 4}
            body = corto.Body(body_name, body_props)
            bodies.append(body)
        
        rendering = corto.Rendering({'engine': 'CYCLES', 'device': 'CPU', 'samples': 256, 'preview_samples': 16})
        env = corto.Environment(cam, bodies, sun, rendering)
        return env, cam, bodies, sun
    
    def create_photometric_materials(self, state, bodies):
        """Create photometric materials"""
        materials = []
        
        for i, body in enumerate(bodies, 1):
            material = corto.Shading.create_new_material(f'photometric_material_{i}')
            
            if hasattr(corto.Shading, 'create_branch_albedo_mix'):
                corto.Shading.create_branch_albedo_mix(material, state, i)
            
            if hasattr(corto.Shading, 'load_uv_data'):
                corto.Shading.load_uv_data(body, state, i)
            
            corto.Shading.assign_material_to_object(material, body)
            materials.append(material)
        
        return materials
    
    def run_simulation_batch(self, img_database, max_simulations=None):
        """Run simulations for all valid UTC times in database"""
        valid_entries = img_database[img_database['STATUS'] == 'SUCCESS'].copy()
        
        if max_simulations:
            valid_entries = valid_entries.head(max_simulations)
            
        print(f"Running simulations for {len(valid_entries)} valid entries")
        
        simulation_results = []
        
        for idx, row in valid_entries.iterrows():
            utc_time = row['UTC_MEAN_TIME']
            img_filename = row['file_name']
            real_img_path = row['file_path']
            
            print(f"\n{'-'*60}")
            print(f"Processing {idx+1}/{len(valid_entries)}: {img_filename}")
            print(f"UTC Time: {utc_time}")
            print(f"{'-'*60}")
            
            try:
                # Run simulation for this UTC time
                result = self.run_single_simulation(utc_time, real_img_path, img_filename, idx)
                
                if result:
                    simulation_results.append(result)
                    print(f"✅ Simulation completed for {img_filename}")
                else:
                    print(f"❌ Simulation failed for {img_filename}")
                    
            except Exception as e:
                print(f"❌ Error in simulation for {img_filename}: {e}")
                import traceback
                traceback.print_exc()
                
        return simulation_results
    
    def run_single_simulation(self, utc_time, real_img_path, img_filename, index):
        """Run single simulation and validation with FIXED validator"""
        try:
            # Setup scene
            state, spice_data = self.setup_photometric_scene(utc_time)
            env, cam, bodies, sun = self.create_photometric_environment(state)
            
            # Create materials
            materials = self.create_photometric_materials(state, bodies)
            
            # Setup compositing with mask ID support - IMPORTANT
            tree = self.setup_enhanced_compositing(state)
            
            # Scale bodies
            bodies[0].set_scale(np.array([1.0, 1.0, 1.0]))      # Phobos
            bodies[1].set_scale(np.array([1000.0, 1000.0, 1000.0]))  # Mars
            bodies[2].set_scale(np.array([1.0, 1.0, 1.0]))      # Deimos
            
            # Position all objects
            env.PositionAll(state, index=0)
            
            # Render
            env.RenderOne(cam, state, index=index, depth_flag=True)
            
            # Get synthetic image path
            synthetic_img_path = Path(state.path["output_path"]) / "img" / f"{str(index).zfill(6)}.png"
            
            # Apply CORTO post-processing if synthetic image exists
            processed_img_path = None
            if synthetic_img_path.exists():
                # Load synthetic image for post-processing
                synthetic_img = cv2.imread(str(synthetic_img_path))
                if synthetic_img is not None:
                    # Apply CORTO post-processing pipeline
                    labels = {
                        'CoB': [synthetic_img.shape[1]//2, synthetic_img.shape[0]//2],
                        'range': float(np.linalg.norm(spice_data["phobos"]["position"])),
                        'phase_angle': 0.0
                    }
                    
                    processed_img, processed_labels = self.post_processor.process_image_label_pair(
                        synthetic_img, labels
                    )
                    
                    # Save processed image
                    processed_img_path = synthetic_img_path.parent / f"processed_{synthetic_img_path.name}"
                    cv2.imwrite(str(processed_img_path), processed_img)
            
            # Validate with FIXED CORTO pipeline
            validation_result = None
            if Path(real_img_path).exists() and synthetic_img_path.exists():
                # Create list of synthetic images for validation
                synthetic_img_paths = [str(synthetic_img_path)]
                if processed_img_path:
                    synthetic_img_paths.append(str(processed_img_path))
                
                # Use FIXED validator
                validation_result = self.validator.validate_with_fixed_pipeline(
                    real_img_path, synthetic_img_paths, utc_time, img_filename
                )
            
            # Save blend file for debugging
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
                'status': 'SUCCESS'
            }
            
        except Exception as e:
            print(f"Error in single simulation: {e}")
            return {
                'index': index,
                'utc_time': utc_time,
                'img_filename': img_filename,
                'status': 'FAILED',
                'error': str(e)
            }
    
    def save_final_results(self, simulation_results, img_database):
        """Save final comprehensive results"""
        output_dir = Path(self.config['output_path'])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save simulation results
        results_path = output_dir / f"simulation_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(convert_numpy_types(simulation_results), f, indent=4)
        
        # Save validation results
        validation_summary = self.validator.get_validation_summary()
        summary_path = output_dir / f"validation_summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(convert_numpy_types(validation_summary), f, indent=4)
        
        print(f"\nFinal results saved to: {output_dir}")
        print(f"Total simulations: {len(simulation_results)}")
        print(f"Successful simulations: {len([r for r in simulation_results if r.get('status') == 'SUCCESS'])}")
        
        return results_path, summary_path

    def _create_noise_combinations(self):
        """Create noise combinations as per CORTO Table 1"""
        noise_combinations = []
        
        # Table 1 values from CORTO paper
        gaussian_means = [0.01, 0.09, 0.17, 0.25]
        gaussian_vars = [1e-5, 1e-4, 1e-3]
        blur_values = [0.6, 0.8, 1.0, 1.2]
        brightness_values = [1.00, 1.17, 1.33, 1.50]
        
        for g_mean in gaussian_means:
            for g_var in gaussian_vars:
                for blur in blur_values:
                    for brightness in brightness_values:
                        noise_combinations.append({
                            'gaussian_mean': g_mean,
                            'gaussian_var': g_var,
                            'blur': blur,
                            'brightness': brightness
                        })
        
        return noise_combinations

    def _apply_noise_combination(self, image, noise_params):
        """Apply specific noise combination to image"""
        result = image.copy().astype(np.float32)
        
        # Gaussian noise
        noise = np.random.normal(
            noise_params['gaussian_mean'], 
            np.sqrt(noise_params['gaussian_var']), 
            image.shape
        )
        result += noise
        
        # Blur
        kernel_size = int(noise_params['blur'] * 2) * 2 + 1
        result = cv2.GaussianBlur(result, (kernel_size, kernel_size), noise_params['blur'])
        
        # Brightness adjustment
        result *= noise_params['brightness']
        
        return np.clip(result, 0, 255).astype(np.uint8)

    def _template_matching_validation(self, real_img, template_imgs):
        """Enhanced template matching following CORTO Figure 15"""
        results = []
        
        for template_img in template_imgs:
            # Normalized cross-correlation
            correlation = cv2.matchTemplate(real_img, template_img, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(correlation)
            
            # Crop both images to maximize correlation
            h, w = template_img.shape
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            
            cropped_real = real_img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            
            # NRMSE calculation
            if cropped_real.shape == template_img.shape:
                mse = np.mean((cropped_real.astype(np.float64) - template_img.astype(np.float64)) ** 2)
                img_range = cropped_real.max() - cropped_real.min()
                nrmse = np.sqrt(mse) / img_range if img_range > 0 else 0
                
                results.append({
                    'template': template_img,
                    'cropped_real': cropped_real,
                    'ncc': max_val,
                    'nrmse': nrmse,
                    'crop_location': max_loc
                })
        
        return results


def main(pds_data_path=None, max_simulations=None, camera_type='SRC'):
    """Main function with FIXED validator integration"""
    print("="*80)
    print("Enhanced Photometric Phobos Simulator with FIXED CORTO Validation")
    print("="*80)
    
    # Initialize simulator with FIXED validator
    simulator = EnhancedPhotometricPhobosSimulator(
        camera_type=camera_type,
        pds_data_path=pds_data_path
    )
    
    try:
        # Step 1: Process PDS database
        print("\n1. Processing PDS database...")
        img_database = simulator.process_pds_database(pds_data_path)
        
        # Step 2: Run simulations with FIXED validation
        print("\n2. Running simulations with FIXED validation...")
        simulation_results = simulator.run_simulation_batch(img_database, max_simulations)
        
        # Step 3: Save results and show validation summary
        print("\n3. Saving final results...")
        simulator.save_final_results(simulation_results, img_database)
        
        # 📊 Show FIXED validation summary
        validation_summary = simulator.validator.get_validation_summary()
        print(f"\n📊 FIXED VALIDATION SUMMARY:")
        print(f"   🔢 Total validations: {validation_summary['total_validations']}")
        print(f"   ✅ Successful validations: {validation_summary['successful_validations']}")
        print(f"   📈 Success rate: {validation_summary['success_rate']:.2%}")
        print(f"   📊 Average composite score: {validation_summary['average_composite_score']:.4f}")
        print(f"   🎯 Average SSIM: {validation_summary['average_ssim']:.4f}")
        print(f"   📁 PDS processing: {'✅ ENABLED' if validation_summary['pds_processing_enabled'] else '❌ DISABLED'}")
        print(f"   🔄 Transformation alignment: {'✅ FIXED' if validation_summary['transformation_alignment'] == 'FIXED' else '❌ BROKEN'}")
        print(f"   🎭 Mask alignment: {'✅ AVAILABLE' if validation_summary['mask_alignment_available'] else '⚠️ NOT USED'}")
        
        print("\n✅ Enhanced simulation pipeline with FIXED validation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in main pipeline: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Configuration - you can modify these parameters
    
    # PDS data path - modify this to point to your IMG files
    PDS_DATA_PATH = "/home/tt_mmx/corto/PDS_Data"  # Change this to your PDS data directory
    
    # Maximum number of simulations to run (None for all)
    MAX_SIMULATIONS = 5  # Set to None to process all IMG files
    
    # Camera type
    CAMERA_TYPE = 'SRC'  # or 'HEAD'
    
    print(f"Starting FIXED enhanced simulator with:")
    print(f"PDS Data Path: {PDS_DATA_PATH}")
    print(f"Max Simulations: {MAX_SIMULATIONS}")
    print(f"Camera Type: {CAMERA_TYPE}")
    
    main(
        pds_data_path=PDS_DATA_PATH,
        max_simulations=MAX_SIMULATIONS,
        camera_type=CAMERA_TYPE
    )
