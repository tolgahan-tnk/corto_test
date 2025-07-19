"""
photometric_phobos_simulator.py (FIXED VERSION)
Photometrically Accurate Phobos Simulation with Dynamic SPICE Data and HRSC SRC Camera
Fixed JSON serialization and camera matrix issues
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


class PhotometricPhobosSimulator:
    """Photometrically accurate Phobos simulation with SPICE integration and SRC camera support."""

    def __init__(self, config_path=None, camera_type='SRC'):
        self.config = self._load_config(config_path)
        self.camera_type = camera_type
        
        # Initialize SPICE processor with enhanced capabilities
        self.spice_processor = SpiceDataProcessor(base_path=self.config['spice_data_path'])
        
        self.scenario_name = "S07_Mars_Phobos_Deimos"

        # Get camera configuration from IK kernel with fallback
        try:
            self.camera_config = self.spice_processor.get_hrsc_camera_config(camera_type)
        except Exception as e:
            print(f"Warning: Could not get camera config from SPICE: {e}")
            self.camera_config = self._get_fallback_camera_config(camera_type)
            
        print(f"Using {camera_type} camera configuration:")
        print(f"  FOV: {self.camera_config['fov']:.3f}°")
        print(f"  Resolution: {self.camera_config['res_x']}x{self.camera_config['res_y']}")
        print(f"  Focal Length: {self.camera_config.get('focal_length_mm', 'N/A')} mm")
        if 'ifov_rad' in self.camera_config:
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

    def _get_fallback_camera_config(self, camera_type='SRC'):
        """Fallback camera configuration if SPICE data is not available"""
        if camera_type == 'SRC':
            return {
                'fov': 0.54,  # degrees
                'res_x': 1024,
                'res_y': 1024,
                'film_exposure': 0.039,
                'sensor': 'BW',
                'clip_start': 0.1,
                'clip_end': 100000000.0,
                'bit_encoding': '16',
                'viewtransform': 'Standard',
                'focal_length_mm': 984.76,
                'f_ratio': 9.2,
                'pixel_size_microns': [9.0, 9.0],
                'ifov_rad': 0.00000914,
                'K': [
                    [1222.0, 0, 512.0],  # fx, 0, cx
                    [0, 1222.0, 512.0],  # 0, fy, cy  
                    [0, 0, 1]            # 0, 0, 1
                ]
            }
        else:  # HEAD camera
            return {
                'fov': 11.9,  # degrees
                'res_x': 5184,
                'res_y': 1,
                'film_exposure': 0.039,
                'sensor': 'BW',
                'clip_start': 0.1,
                'clip_end': 10000.0,
                'bit_encoding': '16',
                'viewtransform': 'Standard',
                'focal_length_mm': 175.0,
                'f_ratio': 5.6,
                'pixel_size_microns': [7.0, 7.0],
                'ifov_rad': 0.000040,
                'K': [
                    [2500.0, 0, 2592.0],  # fx, 0, cx
                    [0, 2500.0, 0.5],     # 0, fy, cy  
                    [0, 0, 1]             # 0, 0, 1
                ]
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
        try:
            spice_data = self.spice_processor.get_spice_data(utc_time)
        except Exception as e:
            print(f"Warning: Could not get SPICE data: {e}")
            # Use default data if SPICE fails
            spice_data = self._get_default_spice_data()
        
        # Write SPICE data to CORTO-compatible geometry file
        output_dir = Path(self.config['output_path'])
        output_dir.mkdir(parents=True, exist_ok=True)
        dynamic_geometry_path = output_dir / 'geometry_dynamic.json'
        
        # Convert to CORTO geometry format with proper type conversion
        geometry_data = {
            "sun": {
                "position": [convert_numpy_types(spice_data["sun"]["position"])]
            },
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
        
        # Create updated scene configuration with SRC camera parameters
        scene_config = self._create_scene_config()
        scene_config_path = output_dir / 'scene_src.json'
        
        # Convert scene config to ensure JSON compatibility
        scene_config_safe = convert_numpy_types(scene_config)
        
        with open(scene_config_path, 'w') as f:
            json.dump(scene_config_safe, f, indent=4)
            
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

    def _create_scene_config(self):
        """Create scene configuration with SRC camera parameters."""
        # Ensure K matrix is properly formatted
        K_matrix = self.camera_config.get('K', [
            [1222.0, 0, 512.0],
            [0, 1222.0, 512.0],
            [0, 0, 1]
        ])
        
        scene_config = {
            "camera_settings": {
                "fov": float(self.camera_config['fov']),
                "res_x": int(self.camera_config['res_x']),
                "res_y": int(self.camera_config['res_y']),
                "film_exposure": float(self.camera_config['film_exposure']),
                "sensor": str(self.camera_config['sensor']),
                "K": K_matrix,  # This ensures 'K' is always present
                "clip_start": float(self.camera_config['clip_start']),
                "clip_end": float(self.camera_config['clip_end']),
                "bit_encoding": str(self.camera_config['bit_encoding']),
                "viewtransform": str(self.camera_config['viewtransform'])
            },
            "sun_settings": {
                "angle": 0.00927,  # Sun angular size as seen from Mars
                "energy": float(self.photometric_params['sun_strength'])
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
        
        # Type-safe camera properties (converts numpy types to Python types)
        cam_props = {
            'fov': float(self.camera_config.get('fov', 0.54)),
            'res_x': int(self.camera_config.get('res_x', 1024)),          # Python int
            'res_y': int(self.camera_config.get('res_y', 1024)),          # Python int  
            'film_exposure': float(self.photometric_params.get('exposure_time', 0.039)),
            'sensor': str(self.camera_config.get('sensor', 'BW')),
            'clip_start': float(self.camera_config.get('clip_start', 0.1)),
            'clip_end': float(self.camera_config.get('clip_end', 10000.0)),
            'bit_encoding': str(self.camera_config.get('bit_encoding', '16')),
            'viewtransform': str(self.camera_config.get('viewtransform', 'Standard'))
        }
        
        # Handle K matrix with camera-specific fallbacks
        if 'K' in self.camera_config and self.camera_config['K'] is not None:
            cam_props['K'] = self.camera_config['K']
        else:
            if self.camera_type == 'SRC':
                cam_props['K'] = [[1222.0, 0, 512.0], [0, 1222.0, 512.0], [0, 0, 1]]
            elif self.camera_type == 'HEAD':
                cam_props['K'] = [[2500.0, 0, 2592.0], [0, 2500.0, 0.5], [0, 0, 1]]
        
        # Create CORTO components
        cam = corto.Camera(f'HRSC_{self.camera_type}_Camera', cam_props)
        sun = corto.Sun('Sun', {'angle': 0.00927, 'energy': float(self.photometric_params['sun_strength'])})
        
        # Create bodies with fallback properties
        bodies = []
        body_names = [Path(bf).stem for bf in self.config['body_files']]
        for i, (body_name, body_file) in enumerate(zip(body_names, self.config['body_files'])):
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

    def run_photometric_simulation(self, utc_time, real_image_path=None, index=0): 
        """Run complete photometric simulation with SRC camera"""
        print(f"Starting photometric simulation for {utc_time}")

        try:
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
            env.RenderOne(cam, state, index=index, depth_flag=True)

            # Validate if real image provided
            if real_image_path:
                print("Validation with real image would be performed here")

            # Save results
            corto.Utils.save_blend(state, f'photometric_debug_for{utc_time}_{camera_type}')

            return state, spice_data

        except Exception as e:
            print(f"❌ Error in simulation for {utc_time}: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def save_camera_report(self):
        """Save detailed camera configuration report"""
        report_path = os.path.join(self.config['output_path'], f'camera_config_{self.camera_type}.json')
        
        full_report = {
            'camera_type': self.camera_type,
            'timestamp': datetime.now().isoformat(),
            'corto_config': self.camera_config,
            'photometric_params': self.photometric_params
        }
        
        # Convert numpy types to JSON-serializable types
        full_report_safe = convert_numpy_types(full_report)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(full_report_safe, f, indent=4)
        
        print(f"Camera configuration report saved to {report_path}")


# Usage example
if __name__ == "__main__":
    # Test with both SRC and main HRSC camera
    camera_types = ['SRC']
    
    # Example UTC times from HRSC observations
    utc_times = [
        "2018-08-02T08:47:45.7005Z",
        "2019-06-05T10:39:54.6265Z",
   ]

    for camera_type in camera_types:
        print(f"\n{'='*60}")
        print(f"Testing with {camera_type} camera")
        print(f"{'='*60}")
        
        try:
            simulator = PhotometricPhobosSimulator(camera_type=camera_type)
            
            # Save camera configuration report
            simulator.save_camera_report()

            for index, utc_time in enumerate(utc_times):
                try:
                    result = simulator.run_photometric_simulation(utc_time, index=index)
                    if result[0] is not None:
                        print(f"✅ Completed simulation for {utc_time} with {camera_type} camera")
                    else:
                        print(f"❌ Failed simulation for {utc_time} with {camera_type} camera")
                except Exception as e:
                    print(f"❌ Error in simulation for {utc_time}: {e}")
                    
        except Exception as e:
            print(f"❌ Error initializing {camera_type} camera simulator: {e}")
            import traceback
            traceback.print_exc()