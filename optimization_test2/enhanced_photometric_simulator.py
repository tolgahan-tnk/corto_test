'''
enhanced_photometric_simulator.py
Enhanced photometric simulator with distance-based calculations
and complete CORTO pipeline integration
'''

import sys
import os
import json
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
import logging

# Import enhanced modules
from enhanced_corto_post_processor import EnhancedCORTOPostProcessor
from complete_validation_pipeline import CompleteValidationPipeline
from enhanced_noise_modeling import EnhancedNoiseModeling, NoiseParameters

# Add current directory to Python path
sys.path.append(os.getcwd())

try:
    from spice_data_processor import SpiceDataProcessor
    import cortopy as corto
except ImportError as e:
    print(f"Error: Required modules not found. Details: {e}")
    sys.exit(1)

class EnhancedPhotometricSimulator:
    """
    Enhanced photometric simulator with complete CORTO pipeline integration
    
    Features:
    - Distance-based photometric calculations
    - Complete Figure 12 post-processing
    - Complete Figure 15 validation
    - Enhanced noise modeling (Figure 8)
    - Systematic template generation
    """
    
    def __init__(self, config_path=None, camera_type='SRC'):
        self.config = self._load_config(config_path)
        self.camera_type = camera_type
        
        # Initialize enhanced components
        self.spice_processor = SpiceDataProcessor(base_path=self.config['spice_data_path'])
        self.post_processor = EnhancedCORTOPostProcessor(target_size=128, enable_domain_randomization=True)
        self.validation_pipeline = CompleteValidationPipeline(self.post_processor)
        self.noise_modeling = EnhancedNoiseModeling()
        
        self.scenario_name = "S07_Mars_Phobos_Deimos"
        
        # Get camera configuration
        try:
            self.camera_config = self.spice_processor.get_hrsc_camera_config(camera_type)
        except Exception as e:
            print(f"Warning: Could not get camera config from SPICE: {e}")
            self.camera_config = self._get_fallback_camera_config(camera_type)
        
        # Enhanced photometric parameters with distance calculations
        self.photometric_params = {
            'sun_base_intensity': 589.0,  # Base solar intensity
            'distance_falloff_enabled': True,  # Enable distance-based calculations
            'phobos_albedo': 0.068,
            'mars_albedo': 0.170,
            'deimos_albedo': 0.068,
            'sensor_aging_factor': 0.95,
            'brdf_model': 'principled',
            'brdf_roughness': 0.5,
            'gamma_correction': 2.2,
            'exposure_time': self.camera_config['film_exposure'],
            'distance_scaling_law': 'inverse_square'  # or 'linear'
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"✅ Enhanced photometric simulator initialized")
        self.logger.info(f"   📷 Camera: {camera_type}")
        self.logger.info(f"   🔄 Post-processing: Enhanced Figure 12")
        self.logger.info(f"   ✅ Validation: Complete Figure 15")
        self.logger.info(f"   🎛️ Noise modeling: 8-step Figure 8")

    def _load_config(self, config_path):
        """Load enhanced configuration"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            return self._create_enhanced_config()

    def _create_enhanced_config(self):
        """Create enhanced configuration with validation paths"""
        base_dir = Path(os.getcwd())
        return {
            'input_path': str(base_dir / "input" / "S07_Mars_Phobos_Deimos"),
            'output_path': str(base_dir / 'output' / 'enhanced_photometric_validation'),
            'spice_data_path': str(base_dir / 'spice_kernels'),
            'pds_data_path': str(base_dir / 'PDS_Data'),
            'real_images_path': str(base_dir / 'real_hrsc_images'),
            'template_output_path': str(base_dir / 'output' / 'templates'),
            'body_files': [
                'g_phobos_287m_spc_0000n00000_v002.obj',
                'Mars_65k.obj', 
                'g_deimos_162m_spc_0000n00000_v001.obj'
            ],
            'scene_file': 'scene_mmx.json',
            'geometry_file': 'geometry_mmx.json'
        }

    def _get_fallback_camera_config(self, camera_type='SRC'):
        """Enhanced fallback camera configuration"""
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

    def calculate_distance_based_intensity(self, sun_position: np.ndarray, 
                                         body_position: np.ndarray) -> float:
        """Calculate distance-based photometric intensity"""
        
        if not self.photometric_params['distance_falloff_enabled']:
            return self.photometric_params['sun_base_intensity']
        
        # Calculate distance from Sun to body
        distance = np.linalg.norm(sun_position - body_position)
        
        # Reference distance (AU in km)
        au_km = 149597870.7  # 1 AU in kilometers
        
        if self.photometric_params['distance_scaling_law'] == 'inverse_square':
            # Inverse square law: I = I0 * (r0/r)^2
            intensity_factor = (au_km / distance) ** 2
        elif self.photometric_params['distance_scaling_law'] == 'linear':
            # Linear falloff: I = I0 * (r0/r)
            intensity_factor = au_km / distance
        else:
            intensity_factor = 1.0
        
        # Apply base intensity with distance scaling
        final_intensity = self.photometric_params['sun_base_intensity'] * intensity_factor
        
        # Clamp to reasonable values
        final_intensity = np.clip(final_intensity, 0.1, 10000.0)
        
        self.logger.debug(f"Distance-based intensity: {distance:.0f} km -> {final_intensity:.2f}")
        
        return float(final_intensity)

    def generate_systematic_templates(self, utc_time: str, base_index: int = 0) -> List[str]:
        """Generate systematic template variations for validation"""
        
        self.logger.info(f"Generating systematic templates for validation")
        
        # Create template output directory
        template_dir = Path(self.config['template_output_path'])
        template_dir.mkdir(parents=True, exist_ok=True)
        
        template_paths = []
        
        # Template generation variations
        template_configs = [
            {'sun_intensity_mult': 0.5, 'albedo_mult': 0.8, 'noise_level': 'low'},
            {'sun_intensity_mult': 0.75, 'albedo_mult': 0.9, 'noise_level': 'low'},
            {'sun_intensity_mult': 1.0, 'albedo_mult': 1.0, 'noise_level': 'none'},
            {'sun_intensity_mult': 1.25, 'albedo_mult': 1.1, 'noise_level': 'medium'},
            {'sun_intensity_mult': 1.5, 'albedo_mult': 1.2, 'noise_level': 'high'},
        ]
        
        for i, template_config in enumerate(template_configs):
            try:
                # Setup scene with variations
                state, spice_data = self.setup_photometric_scene_with_variations(
                    utc_time, template_config
                )
                
                env, cam, bodies, sun = self.create_photometric_environment(state)
                materials = self.create_photometric_materials(state, bodies)
                
                # Setup compositing
                tree = self.setup_enhanced_compositing(state)
                
                # Scale bodies
                bodies[0].set_scale(np.array([1.0, 1.0, 1.0]))      # Phobos
                bodies[1].set_scale(np.array([1000.0, 1000.0, 1000.0]))  # Mars
                bodies[2].set_scale(np.array([1.0, 1.0, 1.0]))      # Deimos
                
                # Position all objects
                env.PositionAll(state, index=0)
                
                # Render template
                template_index = base_index * 10 + i
                env.RenderOne(cam, state, index=template_index, depth_flag=True)
                
                # Get template path
                template_path = Path(state.path["output_path"]) / "img" / f"{str(template_index).zfill(6)}.png"
                if template_path.exists():
                    template_paths.append(str(template_path))
                
                self.logger.info(f"Generated template {i+1}/{len(template_configs)}")
                
            except Exception as e:
                self.logger.error(f"Error generating template {i}: {e}")
                continue
        
        self.logger.info(f"Generated {len(template_paths)} systematic templates")
        return template_paths

    def setup_photometric_scene_with_variations(self, utc_time: str, 
                                               variations: Dict) -> Tuple:
        """Setup photometric scene with systematic variations"""
        
        # Clean previous renders
        corto.Utils.clean_scene()
        
        # Get SPICE data
        try:
            spice_data = self.spice_processor.get_spice_data(utc_time)
        except Exception as e:
            self.logger.warning(f"Could not get SPICE data: {e}")
            spice_data = self._get_default_spice_data()
        
        # Calculate distance-based intensity with variations
        sun_intensity = self.calculate_distance_based_intensity(
            np.array(spice_data["sun"]["position"]),
            np.array(spice_data["phobos"]["position"])
        )
        sun_intensity *= variations.get('sun_intensity_mult', 1.0)
        
        # Create enhanced scene configuration
        scene_config = self._create_enhanced_scene_config(sun_intensity, variations)
        
        # Create geometry file with SPICE data
        output_dir = Path(self.config['output_path'])
        output_dir.mkdir(parents=True, exist_ok=True)
        dynamic_geometry_path = output_dir / 'geometry_dynamic.json'
        
        geometry_data = {
            "sun": {"position": [spice_data["sun"]["position"]]},
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
        
        # Save scene configuration
        scene_config_path = output_dir / 'scene_enhanced.json'
        with open(scene_config_path, 'w') as f:
            json.dump(scene_config, f, indent=4)
        
        # Create CORTO State
        state = corto.State(
            scene=str(scene_config_path),
            geometry=str(dynamic_geometry_path),
            body=self.config['body_files'],
            scenario=self.scenario_name
        )
        
        self._add_photometric_paths(state)
        
        return state, spice_data

    def _create_enhanced_scene_config(self, sun_intensity: float, variations: Dict):
        """Create enhanced scene configuration with variations"""
        
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
                "energy": float(sun_intensity)  # Distance-based intensity
            },
            "body_settings_1": {"pass_index": 1, "diffuse_bounces": 4},  # Phobos
            "body_settings_2": {"pass_index": 2, "diffuse_bounces": 4},  # Mars
            "body_settings_3": {"pass_index": 3, "diffuse_bounces": 4},  # Deimos
            "rendering_settings": {
                "engine": "CYCLES",
                "device": "CPU",
                "samples": 256,
                "preview_samples": 16
            },
            "variations": variations  # Store variations for reference
        }

    def setup_enhanced_compositing(self, state):
        """Setup enhanced compositing with full mask support"""
        tree = corto.Compositing.create_compositing()
        render_node = corto.Compositing.rendering_node(tree, (0, 0))
        
        # Create enhanced compositing branches
        corto.Compositing.create_img_denoise_branch(tree, render_node)
        corto.Compositing.create_depth_branch(tree, render_node)
        corto.Compositing.create_slopes_branch(tree, render_node, state)
        corto.Compositing.create_maskID_branch(tree, render_node, state)
        
        return tree

    def create_photometric_environment(self, state):
        """Create enhanced photometric environment"""
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
        sun = corto.Sun('Sun', {'angle': 0.00927, 'energy': float(self.photometric_params['sun_base_intensity'])})
        
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
        """Create enhanced photometric materials"""
        materials = []
        
        for i, body in enumerate(bodies, 1):
            material = corto.Shading.create_new_material(f'enhanced_material_{i}')
            
            if hasattr(corto.Shading, 'create_branch_albedo_mix'):
                corto.Shading.create_branch_albedo_mix(material, state, i)
            
            if hasattr(corto.Shading, 'load_uv_data'):
                corto.Shading.load_uv_data(body, state, i)
            
            corto.Shading.assign_material_to_object(material, body)
            materials.append(material)
        
        return materials

    def _add_photometric_paths(self, state):
        """Add enhanced photometric paths"""
        state.add_path('albedo_path_1', os.path.join(state.path["input_path"], 'body', 'albedo', 'Phobos grayscale.jpg'))
        state.add_path('albedo_path_2', os.path.join(state.path["input_path"], 'body', 'albedo', 'mars_1k_color.jpg'))
        state.add_path('albedo_path_3', os.path.join(state.path["input_path"], 'body', 'albedo', 'Deimos grayscale.jpg'))
        state.add_path('uv_data_path_1', os.path.join(state.path["input_path"], 'body', 'uv data', 'g_phobos_287m_spc_0000n00000_v002.json'))
        state.add_path('uv_data_path_2', os.path.join(state.path["input_path"], 'body', 'uv data', 'Mars_65k.json'))
        state.add_path('uv_data_path_3', os.path.join(state.path["input_path"], 'body', 'uv data', 'g_deimos_162m_spc_0000n00000_v001.json'))

    def _get_default_spice_data(self):
        """Default SPICE data fallback"""
        return {
            'et': 0.0,
            'phobos': {'position': [0.0, 0.0, 0.0], 'quaternion': [1.0, 0.0, 0.0, 0.0]},
            'mars':   {'position': [0.0, 0.0, 0.0], 'quaternion': [1.0, 0.0, 0.0, 0.0]},
            'deimos': {'position': [0.0, 0.0, 0.0], 'quaternion': [1.0, 0.0, 0.0, 0.0]},
            'sun':    {'position': [0.0, 0.0, 0.0], 'quaternion': [1.0, 0.0, 0.0, 0.0]},
            'hrsc':   {'position': [0.0, 0.0, 0.0], 'quaternion': [1.0, 0.0, 0.0, 0.0]}
        }

    def run_enhanced_simulation_with_complete_validation(self, 
                                                       utc_time: str,
                                                       real_img_path: str,
                                                       img_filename: str,
                                                       index: int) -> Dict:
        """Run enhanced simulation with complete CORTO validation pipeline"""
        
        try:
            self.logger.info(f"Starting enhanced simulation with complete validation")
            
            # Step 1: Generate systematic templates
            template_paths = self.generate_systematic_templates(utc_time, index)
            
            if not template_paths:
                self.logger.error("No templates generated for validation")
                return {'status': 'FAILED', 'error': 'No templates generated'}
            
            # Step 2: Setup main scene
            state, spice_data = self.setup_photometric_scene_with_variations(
                utc_time, {'sun_intensity_mult': 1.0, 'albedo_mult': 1.0, 'noise_level': 'none'}
            )
            
            env, cam, bodies, sun = self.create_photometric_environment(state)
            materials = self.create_photometric_materials(state, bodies)
            tree = self.setup_enhanced_compositing(state)
            
            # Scale bodies
            bodies[0].set_scale(np.array([1.0, 1.0, 1.0]))      # Phobos
            bodies[1].set_scale(np.array([1000.0, 1000.0, 1000.0]))  # Mars
            bodies[2].set_scale(np.array([1.0, 1.0, 1.0]))      # Deimos
            
            # Position and render main scene
            env.PositionAll(state, index=0)
            env.RenderOne(cam, state, index=index, depth_flag=True)
            
            # Step 3: Apply enhanced post-processing
            synthetic_img_path = Path(state.path["output_path"]) / "img" / f"{str(index).zfill(6)}.png"
            
            if not synthetic_img_path.exists():
                self.logger.error("Main synthetic image not generated")
                return {'status': 'FAILED', 'error': 'Main synthetic image not generated'}
            
            # Load and process synthetic image
            synthetic_img = cv2.imread(str(synthetic_img_path))
            if synthetic_img is not None:
                # Apply enhanced post-processing
                labels = {
                    'CoB': [synthetic_img.shape[1]//2, synthetic_img.shape[0]//2],
                    'range': float(np.linalg.norm(spice_data["phobos"]["position"])),
                    'phase_angle': 0.0
                }
                
                processed_img, processed_labels, processing_params = \
                    self.post_processor.process_image_label_pair_enhanced(
                        synthetic_img, labels, domain_randomization_mode='systematic'
                    )
                
                # Save processed image
                processed_img_path = synthetic_img_path.parent / f"processed_{synthetic_img_path.name}"
                cv2.imwrite(str(processed_img_path), processed_img)
                
                # Apply noise modeling
                noise_params = NoiseParameters(
                    generic_blur_sigma=0.8,
                    motion_blur_length=3,
                    noise_variance=0.005,
                    gamma_value=2.2,
                    dead_pixel_probability=0.001
                )
                
                noise_variations = self.noise_modeling.create_systematic_noise_variations(
                    processed_img.astype(np.float32) / 255.0, noise_params, n_variations=3
                )
                
                # Save noise variations
                for i, noisy_img in enumerate(noise_variations):
                    noisy
