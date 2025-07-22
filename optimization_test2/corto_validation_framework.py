"""
Enhanced Photometric Phobos Simulator with CORTO Validation Framework
Integrates PDS_reader, post-processing, and validation according to CORTO paper
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
import re

# Import existing modules
sys.path.append(os.getcwd())
try:
    from spice_data_processor import SpiceDataProcessor
    from corto_post_processor import CORTOPostProcessor
    import cortopy as corto
except ImportError as e:
    print(f"Error: Required modules not found. Detail: {e}")
    sys.exit(1)


class CORTOValidationFramework:
    """CORTO Validation Framework implementing paper's methodology"""
    
    def __init__(self, config_path=None, camera_type='SRC'):
        self.config = self._load_config(config_path)
        self.camera_type = camera_type
        
        # Initialize processors
        self.spice_processor = SpiceDataProcessor(base_path=self.config['spice_data_path'])
        self.post_processor = CORTOPostProcessor(target_size=128)
        
        # CORTO validation parameters (from paper)
        self.validation_params = {
            'ncc_threshold': 0.8,      # Normalized Cross-Correlation
            'nrmse_threshold': 0.05,   # Normalized RMSE  
            'ssim_threshold': 0.98,    # Structural Similarity
            'noise_combinations': 192, # J noise combinations (Table 1)
            'top_m_images': 5,         # M images with lowest NRMSE
            'final_l_images': 3        # L images with highest SSIM
        }
        
        # Noise parameters from CORTO Table 1
        self.noise_params = {
            'gaussian_mean': [0.01, 0.09, 0.17, 0.25],
            'gaussian_variance': [1e-5, 1e-4, 1e-3],
            'blur': [0.6, 0.8, 1.0, 1.2],
            'brightness': [1.00, 1.17, 1.33, 1.50]
        }
        
        # Results storage
        self.validation_results = []
        self.utc_database = None

    def _load_config(self, config_path):
        """Load configuration with enhanced validation paths"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = self._create_default_config()
        
        # Add validation specific paths
        config.update({
            'pds_images_path': config.get('pds_images_path', './PDS_Data'),
            'validation_output_path': config.get('validation_output_path', './validation_results'),
            'real_images_path': config.get('real_images_path', './real_hrsc_images')
        })
        
        return config

    def _create_default_config(self):
        """Enhanced default configuration"""
        base_dir = Path(os.getcwd())
        return {
            'input_path': str(base_dir / "input" / "S07_Mars_Phobos_Deimos"),
            'output_path': str(base_dir / 'output' / 'photometric_validation'),
            'spice_data_path': str(base_dir / 'spice_kernels'),
            'pds_images_path': str(base_dir / 'PDS_Data'),
            'validation_output_path': str(base_dir / 'validation_results'),
            'real_images_path': str(base_dir / 'real_hrsc_images'),
            'body_files': [
                'g_phobos_287m_spc_0000n00000_v002.obj',
                'Mars_65k.obj', 
                'g_deimos_162m_spc_0000n00000_v001.obj'
            ],
            'scene_file': 'scene_mmx.json',
            'geometry_file': 'geometry_mmx.json'
        }

    def extract_pds_database(self, pds_path=None):
        """
        Extract UTC time database from PDS IMG files
        Based on PDS_reader.py methodology
        """
        if pds_path is None:
            pds_path = self.config['pds_images_path']
        
        print(f"Scanning PDS directory: {pds_path}")
        
        records = []
        pds_path = Path(pds_path)
        
        # Walk through directory and find .IMG files
        for img_file in pds_path.rglob("*.IMG"):
            try:
                label_data = self._parse_pds_label(img_file)
                if 'START_TIME' in label_data and 'STOP_TIME' in label_data:
                    # Calculate mean time (CORTO requirement)
                    start_time = pd.to_datetime(label_data['START_TIME'], errors='coerce')
                    stop_time = pd.to_datetime(label_data['STOP_TIME'], errors='coerce')
                    
                    if pd.notna(start_time) and pd.notna(stop_time):
                        mean_time = start_time + (stop_time - start_time) / 2
                        utc_mean_time = mean_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-2] + "Z"
                        
                        record = {
                            'file_path': str(img_file),
                            'file_name': img_file.name,
                            'START_TIME': start_time,
                            'STOP_TIME': stop_time,
                            'MEAN_TIME': mean_time,
                            'UTC_MEAN_TIME': utc_mean_time,
                            'DURATION_SECONDS': (stop_time - start_time).total_seconds()
                        }
                        records.append(record)
                        
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                continue
        
        # Create DataFrame
        self.utc_database = pd.DataFrame(records)
        
        # Fix timezone issues for Excel compatibility
        if len(self.utc_database) > 0:
            for col in ['START_TIME', 'STOP_TIME', 'MEAN_TIME']:
                if col in self.utc_database.columns:
                    self.utc_database[col] = pd.to_datetime(self.utc_database[col]).dt.tz_localize(None)
        
        # Save database
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        db_path = Path(self.config['validation_output_path']) / f'UTC_database_{timestamp}.xlsx'
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with pd.ExcelWriter(db_path, engine="openpyxl") as writer:
            self.utc_database.to_excel(writer, sheet_name="UTC_Database", index=False)
            
            # Summary sheet
            summary = {
                'Metric': ['Total IMG Files', 'Valid Time Records', 'Time Range Start', 'Time Range End'],
                'Value': [
                    len(self.utc_database),
                    len(self.utc_database[self.utc_database['UTC_MEAN_TIME'].notna()]),
                    str(self.utc_database['START_TIME'].min()) if len(self.utc_database) > 0 else 'N/A',
                    str(self.utc_database['STOP_TIME'].max()) if len(self.utc_database) > 0 else 'N/A'
                ]
            }
            pd.DataFrame(summary).to_excel(writer, sheet_name="Summary", index=False)
        
        print(f"UTC Database saved: {db_path}")
        print(f"Total records: {len(self.utc_database)}")
        
        return self.utc_database

    def _parse_pds_label(self, file_path):
        """Parse PDS label from IMG file (from PDS_reader.py)"""
        label = {}
        key_val_pattern = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=\s*(.*)$")
        
        with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
            for i, raw_line in enumerate(fh):
                if i > 50000:  # Safety limit
                    break
                line = raw_line.strip().replace("<CR><LF>", "")
                if line.upper().startswith("END"):
                    break
                match = key_val_pattern.match(line)
                if match:
                    key, val = match.groups()
                    val = val.strip().strip('"').strip("'")
                    label[key] = val
        
        return label

    def run_photometric_simulation_batch(self):
        """
        Run photometric simulations for all UTC times in database
        """
        if self.utc_database is None:
            print("Error: UTC database not loaded. Run extract_pds_database() first.")
            return
        
        results = []
        total_records = len(self.utc_database)
        
        print(f"Starting batch simulation for {total_records} records...")
        
        for index, record in self.utc_database.iterrows():
            utc_time = record['UTC_MEAN_TIME']
            img_file = record['file_name']
            
            print(f"Processing {index+1}/{total_records}: {img_file} at {utc_time}")
            
            try:
                # Run simulation using existing photometric simulator logic
                state, spice_data = self._run_single_simulation(utc_time, index)
                
                if state is not None:
                    # Store simulation results
                    sim_result = {
                        'index': index,
                        'img_file': img_file,
                        'utc_time': utc_time,
                        'file_path': record['file_path'],
                        'simulation_success': True,
                        'synthetic_image_path': self._get_synthetic_image_path(index),
                        'spice_data': spice_data
                    }
                    results.append(sim_result)
                    print(f"✅ Simulation successful for {img_file}")
                else:
                    print(f"❌ Simulation failed for {img_file}")
                    
            except Exception as e:
                print(f"❌ Error simulating {img_file}: {e}")
                continue
        
        self.simulation_results = results
        print(f"Batch simulation completed. {len(results)} successful simulations.")
        
        return results

    def _run_single_simulation(self, utc_time, index):
        """Run single photometric simulation (adapted from existing code)"""
        try:
            # Setup scene using existing logic
            state, spice_data = self._setup_photometric_scene(utc_time)
            env, cam, bodies, sun = self._create_photometric_environment(state)
            materials = self._create_photometric_materials(state, bodies)
            tree = self._setup_photometric_compositing(state)
            
            # Scale bodies
            bodies[0].set_scale(np.array([1.0, 1.0, 1.0]))      # Phobos
            bodies[1].set_scale(np.array([1000.0, 1000.0, 1000.0]))  # Mars
            bodies[2].set_scale(np.array([1.0, 1.0, 1.0]))      # Deimos
            
            # Position and render
            env.PositionAll(state, index=0)
            env.RenderOne(cam, state, index=index, depth_flag=True)
            
            return state, spice_data
            
        except Exception as e:
            print(f"Error in single simulation: {e}")
            return None, None

    def run_corto_validation(self):
        """
        Run CORTO validation framework on simulation results
        Following CORTO paper Section 3.1 methodology
        """
        if not hasattr(self, 'simulation_results'):
            print("Error: No simulation results available. Run simulation batch first.")
            return
        
        validation_results = []
        
        for sim_result in self.simulation_results:
            print(f"Validating: {sim_result['img_file']}")
            
            try:
                # Load real and synthetic images
                real_img_path = sim_result['file_path']
                synthetic_img_path = sim_result['synthetic_image_path']
                
                real_image = self._load_pds_image(real_img_path)
                synthetic_image = self._load_synthetic_image(synthetic_img_path)
                
                if real_image is None or synthetic_image is None:
                    print(f"❌ Could not load images for {sim_result['img_file']}")
                    continue
                
                # Apply CORTO validation pipeline
                validation_result = self._apply_corto_validation_pipeline(
                    real_image, synthetic_image, sim_result
                )
                
                validation_results.append(validation_result)
                print(f"✅ Validation completed for {sim_result['img_file']}")
                
            except Exception as e:
                print(f"❌ Validation error for {sim_result['img_file']}: {e}")
                continue
        
        # Save validation results
        self._save_validation_results(validation_results)
        
        return validation_results

    def _apply_corto_validation_pipeline(self, real_image, synthetic_image, sim_result):
        """
        Apply CORTO validation pipeline (Section 3.1)
        Steps: NCC → Crop → NRMSE → Noise → SSIM
        """
        # Step 1: Normalized Cross-Correlation
        ncc_score = self._calculate_normalized_cross_correlation(real_image, synthetic_image)
        
        # Step 2: Crop images based on correlation
        real_cropped, synthetic_cropped = self._crop_images_for_correlation(
            real_image, synthetic_image
        )
        
        # Step 3: Calculate NRMSE
        nrmse_score = self._calculate_nrmse(real_cropped, synthetic_cropped)
        
        # Step 4: Apply noise combinations (from Table 1)
        noisy_images = self._apply_noise_combinations(synthetic_cropped)
        
        # Step 5: Calculate SSIM for each noisy image
        best_ssim_score = 0
        best_noise_params = None
        
        for noise_params, noisy_image in noisy_images:
            ssim_score = self._calculate_ssim(real_cropped, noisy_image)
            if ssim_score > best_ssim_score:
                best_ssim_score = ssim_score
                best_noise_params = noise_params
        
        # Apply post-processing pipeline (Figure 12)
        processed_real, processed_labels_real = self.post_processor.process_image_label_pair(
            real_cropped, self._extract_labels_from_real_image(real_cropped)
        )
        
        processed_synthetic, processed_labels_synthetic = self.post_processor.process_image_label_pair(
            synthetic_cropped, self._extract_labels_from_synthetic_image(synthetic_cropped, sim_result)
        )
        
        # Final validation scores
        validation_result = {
            'img_file': sim_result['img_file'],
            'utc_time': sim_result['utc_time'],
            'index': sim_result['index'],
            'ncc_score': float(ncc_score),
            'nrmse_score': float(nrmse_score),
            'ssim_score': float(best_ssim_score),
            'best_noise_params': best_noise_params,
            'validation_passed': self._check_validation_thresholds(ncc_score, nrmse_score, best_ssim_score),
            'processed_labels_difference': self._compare_labels(processed_labels_real, processed_labels_synthetic)
        }
        
        return validation_result

    def _calculate_normalized_cross_correlation(self, img1, img2):
        """Calculate normalized cross-correlation (CORTO Equation 1)"""
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Convert to grayscale if needed
        if len(img1.shape) == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if len(img2.shape) == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Normalize images
        img1_norm = (img1 - np.mean(img1)) / np.std(img1)
        img2_norm = (img2 - np.mean(img2)) / np.std(img2)
        
        # Cross-correlation
        correlation = correlate2d(img1_norm, img2_norm, mode='valid')
        max_correlation = np.max(correlation)
        
        return max_correlation

    def _calculate_nrmse(self, img1, img2):
        """Calculate Normalized RMSE (CORTO Equation 2)"""
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        rmse = np.sqrt(mse)
        
        # Normalize by image size
        nrmse = rmse / np.sqrt(img1.shape[0] * img1.shape[1])
        
        return nrmse

    def _calculate_ssim(self, img1, img2):
        """Calculate SSIM (CORTO Equation 3)"""
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Convert to grayscale if needed
        if len(img1.shape) == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if len(img2.shape) == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        ssim_score = ssim(img1, img2, data_range=img1.max() - img1.min())
        return ssim_score

    def _apply_noise_combinations(self, image):
        """Apply noise combinations from CORTO Table 1"""
        noisy_images = []
        
        # Generate combinations from Table 1 parameters
        for g_mean in self.noise_params['gaussian_mean']:
            for g_var in self.noise_params['gaussian_variance']:
                for blur_val in self.noise_params['blur']:
                    for brightness in self.noise_params['brightness']:
                        noise_params = {
                            'gaussian_mean': g_mean,
                            'gaussian_variance': g_var,
                            'blur': blur_val,
                            'brightness': brightness
                        }
                        
                        noisy_img = self._apply_noise(image, noise_params)
                        noisy_images.append((noise_params, noisy_img))
        
        return noisy_images

    def _apply_noise(self, image, noise_params):
        """Apply noise according to CORTO methodology"""
        noisy_img = image.copy().astype(float)
        
        # Gaussian noise
        noise = np.random.normal(
            noise_params['gaussian_mean'],
            np.sqrt(noise_params['gaussian_variance']),
            noisy_img.shape
        )
        noisy_img += noise
        
        # Blur
        ksize = int(noise_params['blur'] * 2) * 2 + 1
        noisy_img = cv2.GaussianBlur(noisy_img, (ksize, ksize), noise_params['blur'])
        
        # Brightness
        noisy_img *= noise_params['brightness']
        
        # Clip values
        noisy_img = np.clip(noisy_img, 0, 255)
        
        return noisy_img.astype(np.uint8)

    def _check_validation_thresholds(self, ncc, nrmse, ssim):
        """Check if validation meets CORTO thresholds"""
        return (ncc >= self.validation_params['ncc_threshold'] and
                nrmse <= self.validation_params['nrmse_threshold'] and
                ssim >= self.validation_params['ssim_threshold'])

    def _save_validation_results(self, validation_results):
        """Save comprehensive validation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(self.config['validation_output_path'])
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save as Excel with multiple sheets
        excel_path = output_path / f'CORTO_validation_results_{timestamp}.xlsx'
        
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            # Main results
            df_results = pd.DataFrame(validation_results)
            df_results.to_excel(writer, sheet_name="Validation_Results", index=False)
            
            # Summary statistics
            passed_count = sum(1 for r in validation_results if r['validation_passed'])
            summary = {
                'Metric': [
                    'Total Validations',
                    'Passed Validations', 
                    'Success Rate (%)',
                    'Average NCC',
                    'Average NRMSE',
                    'Average SSIM'
                ],
                'Value': [
                    len(validation_results),
                    passed_count,
                    (passed_count / len(validation_results) * 100) if validation_results else 0,
                    np.mean([r['ncc_score'] for r in validation_results]) if validation_results else 0,
                    np.mean([r['nrmse_score'] for r in validation_results]) if validation_results else 0,
                    np.mean([r['ssim_score'] for r in validation_results]) if validation_results else 0
                ]
            }
            pd.DataFrame(summary).to_excel(writer, sheet_name="Summary", index=False)
        
        print(f"Validation results saved: {excel_path}")
        
        # Save JSON for programmatic access
        json_path = output_path / f'CORTO_validation_results_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(validation_results, f, indent=4, default=str)
        
        return excel_path

    # Helper methods for image loading and processing
    def _load_pds_image(self, pds_path):
        """Load PDS IMG file (simplified - may need enhancement)"""
        try:
            # This is a placeholder - actual PDS image loading may require specialized libraries
            # For now, assume we can extract the image data somehow
            return cv2.imread(str(pds_path), cv2.IMREAD_GRAYSCALE)
        except:
            return None

    def _load_synthetic_image(self, synthetic_path):
        """Load synthetic image from simulation"""
        try:
            return cv2.imread(str(synthetic_path), cv2.IMREAD_GRAYSCALE)
        except:
            return None

    def _get_synthetic_image_path(self, index):
        """Get path for synthetic image"""
        return Path(self.config['output_path']) / 'img' / f'{index:06d}.png'

    def _crop_images_for_correlation(self, img1, img2):
        """Crop images to maximize correlation"""
        # Simplified cropping - can be enhanced with more sophisticated methods
        min_h = min(img1.shape[0], img2.shape[0])
        min_w = min(img1.shape[1], img2.shape[1])
        
        crop_h = min_h // 2
        crop_w = min_w // 2
        
        cropped1 = img1[crop_h:crop_h+crop_h, crop_w:crop_w+crop_w]
        cropped2 = img2[crop_h:crop_h+crop_h, crop_w:crop_w+crop_w]
        
        return cropped1, cropped2

    def _extract_labels_from_real_image(self, real_image):
        """Extract labels from real image (placeholder)"""
        return {'CoB': [real_image.shape[1]//2, real_image.shape[0]//2]}

    def _extract_labels_from_synthetic_image(self, synthetic_image, sim_result):
        """Extract labels from synthetic image"""
        return {'CoB': [synthetic_image.shape[1]//2, synthetic_image.shape[0]//2]}

    def _compare_labels(self, labels1, labels2):
        """Compare processed labels"""
        if 'CoB' in labels1 and 'CoB' in labels2:
            diff = np.linalg.norm(np.array(labels1['CoB']) - np.array(labels2['CoB']))
            return float(diff)
        return 0.0

    # Implemented methods based on photometric_phobos_simulator.py
    def _setup_photometric_scene(self, utc_time):
        """Setup photometric scene using SPICE data"""
        print(f"Setting up photometric scene for: {utc_time}")
        
        try:
            # Clean previous renders
            corto.Utils.clean_scene()
            
            # Get SPICE data for the given UTC time
            try:
                spice_data = self.spice_processor.get_spice_data(utc_time)
            except Exception as e:
                print(f"Warning: Could not get SPICE data: {e}")
                # Use default data if SPICE fails
                spice_data = self._get_default_spice_data()
            
            # Setup output paths
            output_dir = Path(self.config['output_path'])
            output_dir.mkdir(parents=True, exist_ok=True)
            dynamic_geometry_path = output_dir / 'geometry_dynamic.json'
            
            # Convert to CORTO geometry format
            geometry_data = {
                "sun": {
                    "position": [self._convert_numpy_types(spice_data["sun"]["position"])]
                },
                "camera": {
                    "position": [self._convert_numpy_types(spice_data["hrsc"]["position"])],
                    "orientation": [self._convert_numpy_types(spice_data["hrsc"]["quaternion"])]
                },
                "body_1": {
                    "position": [self._convert_numpy_types(spice_data["phobos"]["position"])],
                    "orientation": [self._convert_numpy_types(spice_data["phobos"]["quaternion"])]
                },
                "body_2": {
                    "position": [self._convert_numpy_types(spice_data["mars"]["position"])],
                    "orientation": [self._convert_numpy_types(spice_data["mars"]["quaternion"])]
                },
                "body_3": {
                    "position": [self._convert_numpy_types(spice_data["deimos"]["position"])],
                    "orientation": [self._convert_numpy_types(spice_data["deimos"]["quaternion"])]
                }
            }
            
            with open(dynamic_geometry_path, 'w') as f:
                json.dump(geometry_data, f, indent=4)
            
            # Create scene configuration
            scene_config = self._create_scene_config()
            scene_config_path = output_dir / 'scene_src.json'
            
            with open(scene_config_path, 'w') as f:
                json.dump(self._convert_numpy_types(scene_config), f, indent=4)
                
            # Create CORTO State object
            scenario_name = "S07_Mars_Phobos_Deimos"
            state = corto.State(
                scene=str(scene_config_path),
                geometry=str(dynamic_geometry_path),
                body=self.config['body_files'],
                scenario=scenario_name
            )
            
            # Add photometric paths
            self._add_photometric_paths(state)
            
            return state, spice_data
            
        except Exception as e:
            print(f"Error in setup_photometric_scene: {e}")
            return None, None

    def _create_photometric_environment(self, state):
        """Create photometrically calibrated environment"""
        try:
            # Get camera configuration
            camera_config = self.spice_processor.get_hrsc_camera_config(self.camera_type)
            
            # Type-safe camera properties
            cam_props = {
                'fov': float(camera_config.get('fov', 0.54)),
                'res_x': int(camera_config.get('res_x', 1024)),
                'res_y': int(camera_config.get('res_y', 1024)),
                'film_exposure': float(camera_config.get('film_exposure', 0.039)),
                'sensor': str(camera_config.get('sensor', 'BW')),
                'clip_start': float(camera_config.get('clip_start', 0.1)),
                'clip_end': float(camera_config.get('clip_end', 10000.0)),
                'bit_encoding': str(camera_config.get('bit_encoding', '16')),
                'viewtransform': str(camera_config.get('viewtransform', 'Standard')),
                'K': camera_config.get('K', [[1222.0, 0, 512.0], [0, 1222.0, 512.0], [0, 0, 1]])
            }
            
            # Create CORTO components
            cam = corto.Camera(f'HRSC_{self.camera_type}_Camera', cam_props)
            sun = corto.Sun('Sun', {'angle': 0.00927, 'energy': 589.0})
            
            # Create bodies
            bodies = []
            body_names = [Path(bf).stem for bf in self.config['body_files']]
            for i, body_name in enumerate(body_names):
                try:
                    body_props = getattr(state, f'properties_body_{i+1}')
                except AttributeError:
                    body_props = {'pass_index': i + 1, 'diffuse_bounces': 4}
                body = corto.Body(body_name, body_props)
                bodies.append(body)
            
            rendering = corto.Rendering({
                'engine': 'CYCLES', 
                'device': 'CPU', 
                'samples': 256, 
                'preview_samples': 16
            })
            
            env = corto.Environment(cam, bodies, sun, rendering)
            return env, cam, bodies, sun
            
        except Exception as e:
            print(f"Error in create_photometric_environment: {e}")
            return None, None, None, None
        
    def _create_photometric_materials(self, state, bodies):
        """Create photometrically accurate materials"""
        try:
            materials = []
            albedo_values = [0.068, 0.170, 0.068]  # Phobos, Mars, Deimos
            
            for i, body in enumerate(bodies, 1):
                material = corto.Shading.create_new_material(f'photometric_material_{i}')
                
                # Create principled BSDF with appropriate albedo
                corto.Shading.create_simple_principled_BSDF(
                    material, 
                    PBSDF_color_RGB=np.array([albedo_values[i-1]] * 3),
                    PBSDF_roughness=0.5
                )
                
                # Assign material to body
                corto.Shading.assign_material_to_object(material, body)
                materials.append(material)
            
            return materials
            
        except Exception as e:
            print(f"Error in create_photometric_materials: {e}")
            return []

    def _setup_photometric_compositing(self, state):
        """Setup compositing for photometric accuracy"""
        try:
            tree = corto.Compositing.create_compositing()
            render_node = corto.Compositing.rendering_node(tree, (0, 0))
            
            # Create branches for validation
            corto.Compositing.create_img_denoise_branch(tree, render_node)
            corto.Compositing.create_depth_branch(tree, render_node)
            
            return tree
            
        except Exception as e:
            print(f"Error in setup_photometric_compositing: {e}")
            return None

    def _get_default_spice_data(self):
        """Default SPICE data if actual SPICE processing fails"""
        return {
            'et': 0.0,
            'phobos': {'position': [0.0, 0.0, 0.0], 'quaternion': [1.0, 0.0, 0.0, 0.0]},
            'mars':   {'position': [0.0, 0.0, 0.0], 'quaternion': [1.0, 0.0, 0.0, 0.0]},
            'deimos': {'position': [0.0, 0.0, 0.0], 'quaternion': [1.0, 0.0, 0.0, 0.0]},
            'sun':    {'position': [0.0, 0.0, 0.0], 'quaternion': [1.0, 0.0, 0.0, 0.0]},
            'hrsc':   {'position': [0.0, 0.0, 0.0], 'quaternion': [1.0, 0.0, 0.0, 0.0]}
        }

    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        return obj

    def _create_scene_config(self):
        """Create scene configuration with camera parameters"""
        camera_config = self.spice_processor.get_hrsc_camera_config(self.camera_type)
        
        return {
            "camera_settings": {
                "fov": float(camera_config.get('fov', 0.54)),
                "res_x": int(camera_config.get('res_x', 1024)),
                "res_y": int(camera_config.get('res_y', 1024)),
                "film_exposure": float(camera_config.get('film_exposure', 0.039)),
                "sensor": str(camera_config.get('sensor', 'BW')),
                "K": camera_config.get('K', [[1222.0, 0, 512.0], [0, 1222.0, 512.0], [0, 0, 1]]),
                "clip_start": float(camera_config.get('clip_start', 0.1)),
                "clip_end": float(camera_config.get('clip_end', 10000.0)),
                "bit_encoding": str(camera_config.get('bit_encoding', '16')),
                "viewtransform": str(camera_config.get('viewtransform', 'Standard'))
            },
            "sun_settings": {
                "angle": 0.00927,
                "energy": 589.0
            },
            "body_settings_1": {"pass_index": 1, "diffuse_bounces": 4},
            "body_settings_2": {"pass_index": 2, "diffuse_bounces": 4},
            "body_settings_3": {"pass_index": 3, "diffuse_bounces": 4},
            "rendering_settings": {
                "engine": "CYCLES",
                "device": "CPU", 
                "samples": 256,
                "preview_samples": 16
            }
        }

    def _add_photometric_paths(self, state):
        """Add photometric maps paths to State object"""
        try:
            # Add paths if they exist
            input_path = Path(self.config['input_path'])
            
            albedo_paths = [
                'body/albedo/Phobos grayscale.jpg',
                'body/albedo/mars_1k_color.jpg', 
                'body/albedo/Deimos grayscale.jpg'
            ]
            
            uv_paths = [
                'body/uv data/g_phobos_287m_spc_0000n00000_v002.json',
                'body/uv data/Mars_65k.json',
                'body/uv data/g_deimos_162m_spc_0000n00000_v001.json'
            ]
            
            for i, (albedo, uv) in enumerate(zip(albedo_paths, uv_paths), 1):
                albedo_full = input_path / albedo
                uv_full = input_path / uv
                
                if albedo_full.exists():
                    state.add_path(f'albedo_path_{i}', str(albedo_full))
                if uv_full.exists():
                    state.add_path(f'uv_data_path_{i}', str(uv_full))
                    
        except Exception as e:
            print(f"Warning: Could not add photometric paths: {e}")


# Main execution function
def main():
    """Main execution with complete CORTO validation workflow"""
    print("🚀 Starting CORTO Validation Framework")
    
    # Initialize framework
    framework = CORTOValidationFramework(camera_type='SRC')
    
    # Step 1: Extract UTC database from PDS files
    print("\n📊 Extracting UTC database from PDS files...")
    utc_db = framework.extract_pds_database()
    
    if utc_db is None or len(utc_db) == 0:
        print("❌ No valid PDS records found. Exiting.")
        return
    
    # Step 2: Run batch photometric simulations
    print(f"\n🎬 Running photometric simulations for {len(utc_db)} records...")
    simulation_results = framework.run_photometric_simulation_batch()
    
    if not simulation_results:
        print("❌ No successful simulations. Exiting.")
        return
    
    # Step 3: Run CORTO validation framework
    print(f"\n✅ Running CORTO validation on {len(simulation_results)} simulations...")
    validation_results = framework.run_corto_validation()
    
    # Step 4: Generate summary report
    print(f"\n📋 Validation completed!")
    if validation_results:
        passed = sum(1 for r in validation_results if r['validation_passed'])
        print(f"Results: {passed}/{len(validation_results)} validations passed")
        print(f"Success rate: {passed/len(validation_results)*100:.1f}%")
    
    print("🎉 CORTO Validation Framework completed successfully!")


if __name__ == "__main__":
    main()