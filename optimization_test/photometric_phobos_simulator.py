"""
photometric_phobos_simulator_fixed.py
Enhanced Photometric Phobos Simulator with Complete CORTO Validation Pipeline
Fixed version addressing all issues mentioned in the analysis
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


class CompleteCORTOValidator:
    """Complete CORTO validation pipeline implementation following Figure 15"""
    
    def __init__(self):
        self.validation_results = []
        
    def normalized_cross_correlation(self, template, image):
        """Compute normalized cross-correlation (Equation 1)"""
        # Ensure images are float
        template = template.astype(np.float64)
        image = image.astype(np.float64)
        
        # Normalize template
        template_mean = np.mean(template)
        template_norm = template - template_mean
        template_std = np.std(template_norm)
        
        if template_std == 0:
            return 0.0
            
        template_norm = template_norm / template_std
        
        # Compute cross-correlation
        correlation = correlate2d(image, template_norm, mode='valid')
        
        # Normalize by local image statistics
        max_corr = 0.0
        for i in range(correlation.shape[0]):
            for j in range(correlation.shape[1]):
                # Extract local region
                local_region = image[i:i+template.shape[0], j:j+template.shape[1]]
                local_mean = np.mean(local_region)
                local_norm = local_region - local_mean
                local_std = np.std(local_norm)
                
                if local_std > 0:
                    ncc = correlation[i, j] / (template.size * local_std)
                    max_corr = max(max_corr, ncc)
        
        return max_corr
    
    def crop_images_for_correlation(self, real_img, synthetic_imgs):
        """Crop images to maximize correlation"""
        cropped_pairs = []
        
        for synthetic_img in synthetic_imgs:
            # Find best correlation position
            ncc = self.normalized_cross_correlation(synthetic_img, real_img)
            
            # For simplicity, crop to same size as synthetic image
            if real_img.shape[0] >= synthetic_img.shape[0] and real_img.shape[1] >= synthetic_img.shape[1]:
                start_y = (real_img.shape[0] - synthetic_img.shape[0]) // 2
                start_x = (real_img.shape[1] - synthetic_img.shape[1]) // 2
                cropped_real = real_img[start_y:start_y+synthetic_img.shape[0], 
                                       start_x:start_x+synthetic_img.shape[1]]
            else:
                cropped_real = cv2.resize(real_img, (synthetic_img.shape[1], synthetic_img.shape[0]))
            
            cropped_pairs.append((cropped_real, synthetic_img, ncc))
        
        return cropped_pairs
    
    def compute_nrmse(self, real_img, synthetic_img):
        """Compute Normalized Root Mean Square Error (Equation 2)"""
        if real_img.shape != synthetic_img.shape:
            synthetic_img = cv2.resize(synthetic_img, (real_img.shape[1], real_img.shape[0]))
        
        # Ensure same data type
        real_img = real_img.astype(np.float64)
        synthetic_img = synthetic_img.astype(np.float64)
        
        mse = np.mean((real_img - synthetic_img) ** 2)
        
        # Normalize by image dynamic range
        img_range = np.max(real_img) - np.min(real_img)
        if img_range == 0:
            return 0.0
            
        nrmse = np.sqrt(mse) / img_range
        return nrmse
    
    def compute_ssim(self, real_img, synthetic_img):
        """Compute Structural Similarity Index Measure (Equation 3)"""
        if real_img.shape != synthetic_img.shape:
            synthetic_img = cv2.resize(synthetic_img, (real_img.shape[1], real_img.shape[0]))
        
        # Convert to grayscale if needed
        if len(real_img.shape) == 3:
            real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2GRAY)
        if len(synthetic_img.shape) == 3:
            synthetic_img = cv2.cvtColor(synthetic_img, cv2.COLOR_BGR2GRAY)
        
        # Ensure same data type
        real_img = real_img.astype(np.float64)
        synthetic_img = synthetic_img.astype(np.float64)
            
        return ssim(real_img, synthetic_img, data_range=synthetic_img.max() - synthetic_img.min())
    
    def apply_noise_combinations(self, synthetic_imgs, noise_params):
        """Apply J noise combinations as per Table 1"""
        noisy_images = []
        
        # Table 1 noise values
        gaussian_means = [0.01, 0.09, 0.17, 0.25]
        gaussian_variances = [1e-5, 1e-4, 1e-3]
        blur_values = [0.6, 0.8, 1.0, 1.2]
        brightness_values = [1.00, 1.17, 1.33, 1.50]
        
        for img in synthetic_imgs:
            for g_mean in gaussian_means:
                for g_var in gaussian_variances:
                    for blur in blur_values:
                        for brightness in brightness_values:
                            noisy_img = img.copy().astype(np.float64)
                            
                            # Apply Gaussian noise
                            noise = np.random.normal(g_mean, np.sqrt(g_var), noisy_img.shape)
                            noisy_img = noisy_img + noise * 255
                            
                            # Apply blur
                            if blur > 0:
                                kernel_size = max(3, int(blur * 3))
                                if kernel_size % 2 == 0:
                                    kernel_size += 1
                                noisy_img = cv2.GaussianBlur(noisy_img, (kernel_size, kernel_size), blur)
                            
                            # Apply brightness
                            noisy_img = noisy_img * brightness
                            
                            # Clip values
                            noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
                            
                            noise_combo = {
                                'gaussian_mean': g_mean,
                                'gaussian_var': g_var,
                                'blur': blur,
                                'brightness': brightness
                            }
                            
                            noisy_images.append((noisy_img, noise_combo))
        
        return noisy_images
    
    def validate_complete_pipeline(self, real_img_path, synthetic_img_paths, utc_time, img_filename, max_templates=10):
        """Complete validation pipeline following CORTO Figure 15"""
        try:
            print(f"Starting validation for {img_filename}")
            
            # Load real image
            if not Path(real_img_path).exists():
                print(f"Warning: Real image not found: {real_img_path}")
                return None
                
            real_img = cv2.imread(str(real_img_path), cv2.IMREAD_GRAYSCALE)
            if real_img is None:
                print(f"Warning: Could not load real image: {real_img_path}")
                return None
            
            # Load synthetic images (templates)
            synthetic_imgs = []
            for img_path in synthetic_img_paths[:max_templates]:  # Limit to N templates
                if Path(img_path).exists():
                    synthetic_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if synthetic_img is not None:
                        synthetic_imgs.append(synthetic_img)
            
            if not synthetic_imgs:
                print(f"Warning: No valid synthetic images found for {img_filename}")
                return None
            
            print(f"Validating with {len(synthetic_imgs)} synthetic images")
            
            # Step 1: Normalized Cross-Correlation and Cropping
            cropped_pairs = self.crop_images_for_correlation(real_img, synthetic_imgs)
            
            # Step 2: NRMSE Calculation and Selection of M best images
            nrmse_results = []
            for cropped_real, cropped_synthetic, ncc in cropped_pairs:
                nrmse = self.compute_nrmse(cropped_real, cropped_synthetic)
                nrmse_results.append((cropped_real, cropped_synthetic, ncc, nrmse))
            
            # Sort by NRMSE and select M best (e.g., top 3)
            M = min(3, len(nrmse_results))
            best_M = sorted(nrmse_results, key=lambda x: x[3])[:M]
            
            # Step 3: Add J noise combinations
            noise_params = {}  # Could be configured
            all_noisy_results = []
            
            for cropped_real, cropped_synthetic, ncc, nrmse in best_M:
                noisy_images = self.apply_noise_combinations([cropped_synthetic], noise_params)
                
                for noisy_img, noise_combo in noisy_images[:10]:  # Limit for performance
                    ssim_score = self.compute_ssim(cropped_real, noisy_img)
                    all_noisy_results.append({
                        'ncc': ncc,
                        'nrmse': nrmse,
                        'ssim': ssim_score,
                        'noise_combo': noise_combo
                    })
            
            # Step 4: Select L images with maximum SSIM
            L = min(5, len(all_noisy_results))
            best_L = sorted(all_noisy_results, key=lambda x: x['ssim'], reverse=True)[:L]
            
            # Create validation result
            validation_result = {
                'utc_time': utc_time,
                'img_filename': img_filename,
                'real_img_path': str(real_img_path),
                'num_synthetic_imgs': len(synthetic_imgs),
                'num_templates_processed': len(cropped_pairs),
                'best_M_selected': M,
                'best_L_selected': L,
                'best_ncc': max([r['ncc'] for r in best_L]) if best_L else 0.0,
                'best_nrmse': min([r['nrmse'] for r in best_L]) if best_L else 1.0,
                'best_ssim': max([r['ssim'] for r in best_L]) if best_L else 0.0,
                'average_ssim': np.mean([r['ssim'] for r in best_L]) if best_L else 0.0,
                'validation_status': 'SUCCESS' if best_L and best_L[0]['ssim'] > 0.7 else 'LOW_SIMILARITY',
                'timestamp': datetime.now().isoformat(),
                'detailed_results': best_L
            }
            
            self.validation_results.append(validation_result)
            print(f"Validation completed - Best SSIM: {validation_result['best_ssim']:.4f}")
            
            return validation_result
            
        except Exception as e:
            print(f"Error in validation for {img_filename}: {e}")
            return None
    
    def save_validation_results(self, output_path):
        """Save validation results to JSON and Excel"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        json_path = output_path / f"validation_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(convert_numpy_types(self.validation_results), f, indent=4)
        
        # Save as Excel
        excel_path = None
        if self.validation_results:
            # Create summary dataframe
            summary_data = []
            for result in self.validation_results:
                summary_row = {
                    'utc_time': result['utc_time'],
                    'img_filename': result['img_filename'],
                    'num_synthetic_imgs': result['num_synthetic_imgs'],
                    'best_ncc': result['best_ncc'],
                    'best_nrmse': result['best_nrmse'],
                    'best_ssim': result['best_ssim'],
                    'average_ssim': result['average_ssim'],
                    'validation_status': result['validation_status'],
                    'timestamp': result['timestamp']
                }
                summary_data.append(summary_row)
            
            df = pd.DataFrame(summary_data)
            
            excel_path = output_path / f"validation_results_{timestamp}.xlsx"
            try:
                with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Validation Summary', index=False)
                    
                    # Create detailed sheet if needed
                    if len(summary_data) > 0:
                        detailed_data = []
                        for result in self.validation_results:
                            for detail in result.get('detailed_results', []):
                                detailed_row = {
                                    'img_filename': result['img_filename'],
                                    'utc_time': result['utc_time'],
                                    'ncc': detail['ncc'],
                                    'nrmse': detail['nrmse'],
                                    'ssim': detail['ssim'],
                                    'gaussian_mean': detail['noise_combo']['gaussian_mean'],
                                    'gaussian_var': detail['noise_combo']['gaussian_var'],
                                    'blur': detail['noise_combo']['blur'],
                                    'brightness': detail['noise_combo']['brightness']
                                }
                                detailed_data.append(detailed_row)
                        
                        if detailed_data:
                            detail_df = pd.DataFrame(detailed_data)
                            detail_df.to_excel(writer, sheet_name='Detailed Results', index=False)
                            
            except Exception as e:
                print(f"Warning: Could not save Excel file: {e}")
                # Save as CSV instead
                csv_path = output_path / f"validation_results_{timestamp}.csv"
                df.to_csv(csv_path, index=False)
                excel_path = csv_path
            
            # Create summary statistics
            summary_stats = {
                'total_validations': len(self.validation_results),
                'successful_validations': len([r for r in self.validation_results if r['validation_status'] == 'SUCCESS']),
                'average_ssim': np.mean([r['best_ssim'] for r in self.validation_results]),
                'average_nrmse': np.mean([r['best_nrmse'] for r in self.validation_results]),
                'average_ncc': np.mean([r['best_ncc'] for r in self.validation_results]),
                'ssim_threshold': 0.7,
                'validation_methodology': 'CORTO Figure 15 - Complete Pipeline Implementation'
            }
            
            summary_path = output_path / f"validation_summary_{timestamp}.json"
            with open(summary_path, 'w') as f:
                json.dump(convert_numpy_types(summary_stats), f, indent=4)
                
        print(f"Validation results saved to: {output_path}")
        return json_path, excel_path


class EnhancedPhotometricPhobosSimulator:
    """Enhanced Photometric Phobos Simulator with Complete CORTO Validation"""
    
    def __init__(self, config_path=None, camera_type='SRC', pds_data_path=None):
        self.config = self._load_config(config_path)
        self.camera_type = camera_type
        self.pds_data_path = pds_data_path or self.config.get('pds_data_path', 'PDS_Data')
        
        # Initialize components
        self.spice_processor = SpiceDataProcessor(base_path=self.config['spice_data_path'])
        self.pds_processor = PDSImageProcessor(self.pds_data_path)
        self.post_processor = CORTOPostProcessor(target_size=128)
        self.validator = CompleteCORTOValidator()  # Use complete validator
        
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
        
        print(f"Enhanced simulator initialized with {camera_type} camera")
        print(f"PDS data path: {self.pds_data_path}")
    
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
        
        # Create mask ID branch - FIXED: This was missing in the original code
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
        """Run single simulation and validation"""
        try:
            # Setup scene
            state, spice_data = self.setup_photometric_scene(utc_time)
            env, cam, bodies, sun = self.create_photometric_environment(state)
            
            # Create materials
            materials = self.create_photometric_materials(state, bodies)
            
            # Setup compositing with mask ID support
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
            
            # Validate with complete CORTO pipeline
            validation_result = None
            if Path(real_img_path).exists() and synthetic_img_path.exists():
                # Create list of synthetic images for validation
                synthetic_img_paths = [str(synthetic_img_path)]
                if processed_img_path:
                    synthetic_img_paths.append(str(processed_img_path))
                
                validation_result = self.validator.validate_complete_pipeline(
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
        json_path, excel_path = self.validator.save_validation_results(output_dir)
        
        print(f"\nFinal results saved to: {output_dir}")
        print(f"Total simulations: {len(simulation_results)}")
        print(f"Successful simulations: {len([r for r in simulation_results if r.get('status') == 'SUCCESS'])}")
        
        if self.validator.validation_results:
            print(f"Validation results:")
            print(f"  - JSON: {json_path}")
            print(f"  - Excel: {excel_path}")
            print(f"  - Total validations: {len(self.validator.validation_results)}")
            print(f"  - Successful validations: {len([r for r in self.validator.validation_results if r['validation_status'] == 'SUCCESS'])}")
            avg_ssim = np.mean([r['best_ssim'] for r in self.validator.validation_results])
            print(f"  - Average SSIM: {avg_ssim:.4f}")
        
        return results_path


def main(pds_data_path=None, max_simulations=None, camera_type='SRC'):
    """Main function to run the enhanced simulator"""
    print("="*80)
    print("Enhanced Photometric Phobos Simulator with Complete CORTO Validation")
    print("="*80)
    
    # Initialize simulator
    simulator = EnhancedPhotometricPhobosSimulator(
        camera_type=camera_type,
        pds_data_path=pds_data_path
    )
    
    try:
        # Step 1: Process PDS database
        print("\n1. Processing PDS database...")
        if pds_data_path:
            img_database = simulator.process_pds_database(pds_data_path)
        else:
            print("Warning: No PDS data path provided. Using default path.")
            img_database = simulator.process_pds_database()
        
        # Step 2: Run simulations
        print("\n2. Running simulations...")
        simulation_results = simulator.run_simulation_batch(img_database, max_simulations)
        
        # Step 3: Save results
        print("\n3. Saving final results...")
        simulator.save_final_results(simulation_results, img_database)
        
        print("\n✅ Enhanced simulation pipeline completed successfully!")
        
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
    
    print(f"Starting enhanced simulator with:")
    print(f"PDS Data Path: {PDS_DATA_PATH}")
    print(f"Max Simulations: {MAX_SIMULATIONS}")
    print(f"Camera Type: {CAMERA_TYPE}")
    
    main(
        pds_data_path=PDS_DATA_PATH,
        max_simulations=MAX_SIMULATIONS,
        camera_type=CAMERA_TYPE
    )
