"""
photometric_phobos_simulater.py (FIXED)
Photometrically Accurate Phobos Simulation with Dynamic SPICE Data
Based on S07_Mars_Phobos_Deimos.py with SPICE integration
and user-provided file structure.
"""

"""
photometric_phobos_simulator.py (UPDATED WITH SRC CAMERA SUPPORT)
Photometrically Accurate Phobos Simulation with Dynamic SPICE Data and HRSC SRC Camera
"""

"""
photometric_phobos_simulator.py (UPDATED WITH SRC CAMERA SUPPORT)
Photometrically Accurate Phobos Simulation with Dynamic SPICE Data and HRSC SRC Camera
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

# Add current directory to Python path
sys.path.append(os.getcwd())

# Import other modules
try:
    from spice_data_processor import SpiceDataProcessor
    import cortopy as corto
except ImportError as e:
    print(f"Hata: Gerekli modüller bulunamadı. Detay: {e}")
    sys.exit(1)


class PhotometricPhobosSimulator:
    """Photometrically accurate Phobos simulation with SPICE integration and SRC camera support."""

    def __init__(self, config_path=None, camera_type='SRC'):
        self.config = self._load_config(config_path)
        self.camera_type = camera_type
        
        # Initialize SPICE processor with enhanced capabilities
        self.spice_processor = SpiceDataProcessor(base_path=self.config['spice_data_path'])
        
        self.scenario_name = "S07_Mars_Phobos_Deimos"

        # Get camera configuration from IK kernel
        self.camera_config = self.spice_processor.get_hrsc_camera_config(camera_type)
        print(f"Using {camera_type} camera configuration:")
        print(f"  FOV: {self.camera_config['fov']:.3f}°")
        print(f"  Resolution: {self.camera_config['res_x']}x{self.camera_config['res_y']}")
        print(f"  Focal Length: {self.camera_config['focal_length_mm']:.1f} mm")
        print(f"  IFOV: {self.camera_config['ifov_rad']*206265:.3f} arcsec/pixel")

        # Photometric parameters optimized for SRC camera
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

        # Noise model parameters
        self.noise_params = {
            'gaussian_mean': 0.0,
            'gaussian_variance': 0.001,
            'blur_kernel_size': 1.0,
            'dead_pixel_probability': 0.0001,
            'radiation_line_probability': 0.001,
        }

        # Validation parameters
        self.validation_params = {
            'ssim_threshold': 0.98,
            'rmse_threshold': 0.05,
            'ncc_threshold': 0.8,
        }

    def _load_config(self, config_path):
        """Load configuration from JSON file or create default."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            print("Uyarı: config.json bulunamadı. Varsayılan yapılandırma oluşturuluyor.")
            return self._create_default_config()

    def _create_default_config(self):
        """Create default configuration based on user's file structure."""
        base_dir = Path(os.getcwd())

        return {
            'input_path': str(base_dir / "input" / "S07_Mars_Phobos_Deimos"),
            'output_path': str(base_dir / 'output' / 'photometric_validation'),
            'spice_data_path': str(base_dir / 'spice_kernels'),
            'real_images_path': str(base_dir / 'real_hrsc_images'),

            'body_files': [
                'g_phobos_287m_spc_0000n00000_v002.obj',
                'Mars_65k.obj',
                'g_deimos_162m_spc_0000n00000_v001.obj'
            ],
            
            'scene_file': 'scene_mmx.json',
            'geometry_file': 'geometry_mmx.json'
        }

    def setup_photometric_scene(self, utc_time):
        """Set up photometrically correct scene with SRC camera parameters."""
        print(f"Fotometrik sahne kuruluyor: {utc_time}")
        
        # Clean previous renders
        corto.Utils.clean_scene()
        
        # Get SPICE data for the given UTC time
        spice_data = self.spice_processor.get_spice_data(utc_time)
        
        # Write SPICE data to CORTO-compatible geometry file
        output_dir = Path(self.config['output_path'])
        output_dir.mkdir(parents=True, exist_ok=True)
        dynamic_geometry_path = output_dir / 'geometry_dynamic.json'
        
        # Convert to CORTO geometry format
        geometry_data = {
            "sun": {
                "position": [spice_data["sun"]["position"]]
            },
            "camera": {
                "position": [spice_data["hrsc"]["position"]],
                "orientation": [spice_data["hrsc"]["quaternion"]]
            },
            "body_1": {
                "position": [spice_data["phobos"]["position"]],
                "orientation": [spice_data["phobos"]["quaternion"]]
            },
            "body_2": {
                "position": [spice_data["mars"]["position"]],
                "orientation": [spice_data["mars"]["quaternion"]]
            },
            "body_3": {
                "position": [spice_data["deimos"]["position"]],
                "orientation": [spice_data["deimos"]["quaternion"]]
            }
        }
        
        with open(dynamic_geometry_path, 'w') as f:
            json.dump(geometry_data, f, indent=4)
        
        # Create updated scene configuration with SRC camera parameters
        scene_config = self._create_scene_config()
        scene_config_path = output_dir / 'scene_src.json'
        
        with open(scene_config_path, 'w') as f:
            json.dump(scene_config, f, indent=4)
            
        # Create CORTO State object
        state = corto.State(
            scene=str(scene_config_path),
            geometry=str(dynamic_geometry_path),
            body=self.config['body_files'],
            scenario=self.scenario_name
        )
        
        # Add photometric paths
        self._add_photometric_paths(state)
        
        return state, spice_data

    def _create_scene_config(self):
        """Create scene configuration with SRC camera parameters."""
        scene_config = {
            "camera_settings": {
                "fov": self.camera_config['fov'],
                "res_x": self.camera_config['res_x'],
                "res_y": self.camera_config['res_y'],
                "film_exposure": self.camera_config['film_exposure'],
                "sensor": self.camera_config['sensor'],
                "K": self.camera_config['K'],
                "clip_start": self.camera_config['clip_start'],
                "clip_end": self.camera_config['clip_end'],
                "bit_encoding": self.camera_config['bit_encoding'],
                "viewtransform": self.camera_config['viewtransform']
            },
            "sun_settings": {
                "angle": 0.00927,  # Sun angular size as seen from Mars
                "energy": self.photometric_params['sun_strength']
            },
            "body_settings_1": {  # Phobos
                "pass_index": 1,
                "diffuse_bounces": 4
            },
            "body_settings_2": {  # Mars
                "pass_index": 2,
                "diffuse_bounces": 4
            },
            "body_settings_3": {  # Deimos
                "pass_index": 3,
                "diffuse_bounces": 4
            },
            "rendering_settings": {
                "engine": "CYCLES",
                "device": "CPU",
                "samples": 256,
                "preview_samples": 16
            }
        }
        return scene_config

    def _add_photometric_paths(self, state):
        """Add photometric maps paths to State object."""
        # Albedo maps
        state.add_path('albedo_path_1', os.path.join(state.path["input_path"], 'body', 'albedo', 'Phobos grayscale.jpg'))
        state.add_path('albedo_path_2', os.path.join(state.path["input_path"], 'body', 'albedo', 'mars_1k_color.jpg'))
        state.add_path('albedo_path_3', os.path.join(state.path["input_path"], 'body', 'albedo', 'Deimos grayscale.jpg'))

        # UV Data Files
        state.add_path('uv_data_path_1', os.path.join(state.path["input_path"], 'body', 'uv data', 'g_phobos_287m_spc_0000n00000_v002.json'))
        state.add_path('uv_data_path_2', os.path.join(state.path["input_path"], 'body', 'uv data', 'Mars_65k.json'))
        state.add_path('uv_data_path_3', os.path.join(state.path["input_path"], 'body', 'uv data', 'g_deimos_162m_spc_0000n00000_v001.json'))

    def create_photometric_environment(self, state):
        """Create photometrically calibrated environment with SRC camera."""
        # Camera settings (now using SRC parameters)
        cam_props = state.properties_cam.copy()
        cam_props['film_exposure'] = self.photometric_params['exposure_time']
        cam = corto.Camera(f'HRSC_{self.camera_type}_Camera', cam_props)
        
        # Sun settings
        sun_props = state.properties_sun.copy()
        sun_props['energy'] = self.photometric_params['sun_strength']
        sun = corto.Sun('Sun', sun_props)
        
        # Create bodies
        bodies = []
        body_names = [Path(bf).stem for bf in self.config['body_files']]
        
        for i, (body_name, body_file) in enumerate(zip(body_names, self.config['body_files'])):
            body_props = getattr(state, f'properties_body_{i+1}')
            body = corto.Body(body_name, body_props)
            bodies.append(body)
        
        # Rendering settings optimized for SRC camera
        rendering_props = state.properties_rendering.copy()
        rendering_props['samples'] = 256  # High quality for photometric accuracy
        rendering = corto.Rendering(rendering_props)
        
        env = corto.Environment(cam, bodies, sun, rendering)
        return env, cam, bodies, sun
        
    def create_photometric_materials(self, state, bodies):
        """Create photometrically accurate materials"""
        materials = []

        for i, body in enumerate(bodies, 1):
            material = corto.Shading.create_new_material(f'photometric_material_{i}')

            # Set albedo based on photometric parameters
            albedo_key = ['phobos_albedo', 'mars_albedo', 'deimos_albedo'][i-1]
            albedo_value = self.photometric_params[albedo_key]

            # Create photometric shading
            if self.photometric_params['brdf_model'] == 'oren_nayar':
                settings = {
                    'albedo': {
                        'roughness': self.photometric_params['brdf_roughness'],
                        'colorspace_name': 'Linear CIE-XYZ D65'
                    }
                }
                corto.Shading.oren(material, state, settings, i)
            else:
                # Use principled BSDF as default
                corto.Shading.create_branch_albedo_mix(material, state, i)

            # Load UV data and assign material
            corto.Shading.load_uv_data(body, state, i)
            corto.Shading.assign_material_to_object(material, body)

            materials.append(material)

        return materials

    def setup_photometric_compositing(self, state):
        """Setup compositing for photometric accuracy"""
        tree = corto.Compositing.create_compositing()
        render_node = corto.Compositing.rendering_node(tree, (0, 0))

        # Create branches for validation
        corto.Compositing.create_img_denoise_branch(tree, render_node)
        corto.Compositing.create_depth_branch(tree, render_node)
        corto.Compositing.create_slopes_branch(tree, render_node, state)

        # Add gamma correction for photometric accuracy
        gamma_node = corto.Compositing.gamma_node(tree, (600, 0))
        gamma_node.inputs[1].default_value = 1.0 / self.photometric_params['gamma_correction']

        return tree

    def apply_noise_model(self, image_array, noise_combination=None):
        """Apply CORTO noise model to synthetic image"""
        if noise_combination is None:
            noise_combination = [0.01, 1e-4, 1.0, 1.17]

        gaussian_mean, gaussian_var, blur_factor, brightness = noise_combination

        # Convert to numpy array if needed
        if isinstance(image_array, list):
            image_array = np.array(image_array)

        # Reshape if needed (Blender pixel format)
        if len(image_array.shape) == 1:
            h, w = int(np.sqrt(len(image_array) // 4)), int(np.sqrt(len(image_array) // 4))
            image_array = image_array.reshape(h, w, 4)

        # Extract RGB channels
        image_rgb = image_array[:, :, :3]

        # 1. Generic blur
        blur_kernel = int(blur_factor * 3)
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        blurred = cv2.GaussianBlur(image_rgb, (blur_kernel, blur_kernel), 0)

        # 2. Gaussian noise
        noise = np.random.normal(gaussian_mean, gaussian_var, image_rgb.shape)
        noisy = np.clip(blurred + noise, 0, 1)

        # 3. Brightness adjustment
        bright = np.clip(noisy * brightness, 0, 1)

        # 4. Gamma correction
        gamma_corrected = np.power(bright, 1.0 / self.photometric_params['gamma_correction'])

        # 5. Sensor effects
        sensor_affected = self._apply_sensor_effects(gamma_corrected)

        return sensor_affected

    def _apply_sensor_effects(self, image):
        """Apply sensor degradation effects"""
        h, w = image.shape[:2]

        # Dead pixels
        dead_pixels = np.random.random((h, w)) < self.noise_params['dead_pixel_probability']
        image[dead_pixels] = 0

        # Radiation lines
        if np.random.random() < self.noise_params['radiation_line_probability']:
            line_y = np.random.randint(0, h)
            image[line_y, :] = 1.0

        # Sensor aging
        aged = image * self.photometric_params['sensor_aging_factor']

        return np.clip(aged, 0, 1)

    def validate_photometric_accuracy(self, synthetic_image, real_image_path):
        """Validate photometric accuracy using CORTO validation pipeline"""
        # Load real image
        real_image = cv2.imread(real_image_path, cv2.IMREAD_GRAYSCALE)
        if real_image is None:
            print(f"Warning: Could not load real image {real_image_path}")
            return None

        # Ensure same data type
        real_image = real_image.astype(np.float32) / 255.0
        synthetic_image = synthetic_image.astype(np.float32)

        # 1. Normalized Cross Correlation (NCC)
        ncc = correlate2d(real_image, synthetic_image, mode='valid')
        max_ncc = np.max(ncc)

        if max_ncc < self.validation_params['ncc_threshold']:
            print(f"Warning: Low NCC correlation: {max_ncc:.3f}")

        # 2. Crop images to same size
        min_h = min(real_image.shape[0], synthetic_image.shape[0])
        min_w = min(real_image.shape[1], synthetic_image.shape[1])

        real_cropped = real_image[:min_h, :min_w]
        synthetic_cropped = synthetic_image[:min_h, :min_w]

        # 3. Calculate SSIM
        ssim_score = ssim(real_cropped, synthetic_cropped, data_range=1.0)

        # 4. Calculate RMSE
        rmse = np.sqrt(np.mean((real_cropped - synthetic_cropped) ** 2))

        # 5. Calculate normalized RMSE
        nrmse = rmse / (np.max(real_cropped) - np.min(real_cropped))

        validation_results = {
            'ncc': max_ncc,
            'ssim': ssim_score,
            'rmse': rmse,
            'nrmse': nrmse,
            'passed': (ssim_score > self.validation_params['ssim_threshold'] and
                       nrmse < self.validation_params['rmse_threshold'])
        }

        return validation_results

    def run_photometric_simulation(self, utc_time, real_image_path=None):
        """Run complete photometric simulation with SRC camera"""
        print(f"Starting photometric simulation for {utc_time}")

        # Setup scene
        state, spice_data = self.setup_photometric_scene(utc_time)
        env, cam, bodies, sun = self.create_photometric_environment(state)

        # Create materials
        materials = self.create_photometric_materials(state, bodies)

        # Setup compositing
        tree = self.setup_photometric_compositing(state)

        # Scale bodies appropriately for SRC camera viewing
        bodies[0].set_scale(np.array([1.0, 1.0, 1.0]))      # Phobos
        bodies[1].set_scale(np.array([1000.0, 1000.0, 1000.0]))  # Mars
        bodies[2].set_scale(np.array([1.0, 1.0, 1.0]))      # Deimos

        # Position all objects
        env.PositionAll(state, index=0)

        # Render
        env.RenderOne(cam, state, index=0, depth_flag=True)

        # Validate if real image provided
        if real_image_path:
            print("Validation with real image would be performed here")

        # Save results
        corto.Utils.save_blend(state, 'initial_debug')

        return state, spice_data

    def create_validation_report(self, validation_results_list):
        """Create comprehensive validation report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'camera_type': self.camera_type,
            'camera_config': self.camera_config,
            'photometric_parameters': self.photometric_params,
            'validation_results': validation_results_list,
            'summary': {
                'total_images': len(validation_results_list),
                'passed_validation': sum(1 for r in validation_results_list if r and r['passed']),
                'average_ssim': np.mean([r['ssim'] for r in validation_results_list if r]),
                'average_rmse': np.mean([r['rmse'] for r in validation_results_list if r])
            }
        }

        # Save report
        report_path = os.path.join(self.config['output_path'], 'validation_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)

        return report

    def save_camera_report(self):
        """Save detailed camera configuration report"""
        report_path = os.path.join(self.config['output_path'], f'camera_config_{self.camera_type}.json')
        
        # Get additional camera parameters from IK kernel
        camera_params = self.spice_processor._get_camera_parameters(f'HRSC_{self.camera_type}')
        
        full_report = {
            'camera_type': self.camera_type,
            'timestamp': datetime.now().isoformat(),
            'corto_config': self.camera_config,
            'ik_parameters': camera_params,
            'photometric_params': self.photometric_params
        }
        
        # Convert numpy types to JSON-serializable types
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
            
        full_report_safe = convert_numpy_types(full_report)
        
        with open(report_path, 'w') as f:
            json.dump(full_report_safe, f, indent=4)
        
        print(f"Camera configuration report saved to {report_path}")

# Usage example
if __name__ == "__main__":
    # Test with both SRC and main HRSC camera
    camera_types = ['SRC', 'HEAD']
    
    # Example UTC times from HRSC observations
    utc_times = [
        "2019-06-07T10:46:42.8575Z",
        "2018-08-02T08:48:03.6855Z"
    ]

    for camera_type in camera_types:
        print(f"\n{'='*60}")
        print(f"Testing with {camera_type} camera")
        print(f"{'='*60}")
        
        simulator = PhotometricPhobosSimulator(camera_type=camera_type)
        
        # Save camera configuration report
        simulator.save_camera_report()

        for utc_time in utc_times:
            try:
                state, spice_data = simulator.run_photometric_simulation(utc_time)
                print(f"✅ Completed simulation for {utc_time} with {camera_type} camera")
            except Exception as e:
                print(f"❌ Error in simulation for {utc_time}: {e}")
