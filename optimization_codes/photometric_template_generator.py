    """
    photometric_template_generator.py
    Photometric template generator with Hapke BRDF parameter exploration for Mars, Phobos, and Deimos
    """

    import sys
    import os
    import json
    import numpy as np
    import cv2
    import time
    import pickle
    import pandas as pd
    import bpy
    from pathlib import Path
    from datetime import datetime
    from PIL import Image

    # Add current directory to Python path
    sys.path.append(os.getcwd())

    # Import required modules
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

    AU_KM = 149_597_870.7  # km

    # Hapke BRDF OSL Shader Code - Not used anymore, using native Blender nodes instead
    # HAPKE_OSL_SHADER = """..."""

    class PhotometricTemplateGenerator:
        """Photometric template generator with Hapke BRDF parameter exploration"""
        
        def __init__(self, config_path=None, camera_type='SRC', pds_data_path=None,
                    Sun_Blender_scaler=1.0, N_templates=27,
                    mars_w=0.4, mars_theta_bar=20.0,
                    deimos_w=0.3, deimos_theta_bar=20.0):
            """Initialize template generator
            
            Args:
                Sun_Blender_scaler: Scaling factor for sun strength in Blender
                N_templates: Number of templates to generate per scene
                mars_w: Fixed single scattering albedo for Mars
                mars_theta_bar: Fixed roughness angle for Mars
                deimos_w: Fixed single scattering albedo for Deimos
                deimos_theta_bar: Fixed roughness angle for Deimos
            """
            self.config = self._load_or_create_config(config_path)
            self.camera_type = camera_type
            self.pds_data_path = pds_data_path or self.config.get('pds_data_path', 'PDS_Data')
            self.Sun_Blender_scaler = Sun_Blender_scaler
            self.N_templates = N_templates

            # -------------------------------------------------------------
            #  Fixed Hapke BRDF values: 2 = Mars, 3 = Deimos
            #  (Phobos = 1 still in parameter exploration)
            # -------------------------------------------------------------
            self.static_body_brdf = {
                2: {'w': mars_w, 'theta_bar': mars_theta_bar},
                3: {'w': deimos_w, 'theta_bar': deimos_theta_bar},
            }
            
            # Initialize components
            self.spice_processor = SpiceDataProcessor(base_path=self.config['spice_data_path'])
            self.pds_processor = CompactPDSProcessor(self.pds_data_path)
            self.post_processor = CORTOPostProcessor(target_size=192)
            
            self.scenario_name = "S07_Mars_Phobos_Deimos"
            
            # Get camera configuration
            self.camera_config = self._get_camera_config()
            
            # Calculate albedo factors for each body
            self._calculate_albedo_factors()
            
            # Setup Hapke BRDF parameter ranges
            self._setup_hapke_ranges()
            
            # Note: We're using native Blender nodes instead of OSL
            # self._write_hapke_osl_shader()
            
            # Setup output tracking
            self.template_results = []
            self.output_excel_path = None
            
            print(f"✅ Photometric Template Generator initialized with Hapke BRDF")
            print(f"   Sun Blender scaler: {self.Sun_Blender_scaler}")
            print(f"   Templates per scene: {self.N_templates}")

        # def _write_hapke_osl_shader(self):
        #     """Write Hapke OSL shader to file - Not used anymore"""
        #     shader_path = Path(self.config.get('output_path', 'output')) / 'hapke_brdf.osl'
        #     shader_path.parent.mkdir(parents=True, exist_ok=True)
        #     
        #     with open(shader_path, 'w') as f:
        #         f.write(HAPKE_OSL_SHADER)
        #     
        #     self.hapke_shader_path = shader_path
        #     print(f"   ✅ Hapke OSL shader written to: {shader_path}")

        def _calculate_albedo_factors(self):
            """Calculate albedo factors from actual texture files"""
            # Texture paths
            base_input = Path(self.config.get('input_path', 'input/S07_Mars_Phobos_Deimos'))
            texture_paths = {
                1: base_input / 'body' / 'albedo' / 'Phobos_Viking_Mosaic_40ppd_DLRcontrol.tif',
                2: base_input / 'body' / 'albedo' / 'Mars_grayscale.tif', 
                3: base_input / 'body' / 'albedo' / 'Deimos grayscale.jpg'
            }
            
            # Target albedos for each body
            target_albedos = {
                1: 0.0812,   # Phobos: from Fornasier et al. 2024
                2: 0.2034,    # Mars: average value
                3: 0.0684     # Deimos: from literature
            }
            
            # Calculate texture averages and albedo factors
            self.texture_averages = {}
            self.albedo_factors = {}
            
            print("\n🎨 Analyzing texture files for albedo calibration:")
            
            for body_id, tex_path in texture_paths.items():
                try:
                    tex = cv2.imread(str(tex_path), cv2.IMREAD_UNCHANGED)
                    if tex is None:
                        print(f"   ⚠️  Body {body_id}: Texture not found, using fallback")
                        self.texture_averages[body_id] = 0.216
                        self.albedo_factors[body_id] = target_albedos[body_id] / 0.216
                        continue
                    
                    if tex.ndim == 3:
                        tex = cv2.cvtColor(tex, cv2.COLOR_BGR2GRAY)
                    
                    tex = tex.astype(np.float32)
                    
                    # Calculate mean relative value
                    max_dn = tex.max()
                    if max_dn == 0:
                        texture_avg_relative = 0.216
                    else:
                        mask = (tex > 0) & (tex < 0.98 * max_dn)
                        texture_avg_relative = tex[mask].mean() / max_dn if np.any(mask) else 0.216
                    
                    albedo_factor = target_albedos[body_id] / texture_avg_relative
                    
                    self.texture_averages[body_id] = texture_avg_relative
                    self.albedo_factors[body_id] = albedo_factor
                    
                    print(f"   📊 Body {body_id} ({tex_path.name}):")
                    print(f"      Texture average: {texture_avg_relative:.4f}")
                    print(f"      Target albedo: {target_albedos[body_id]:.3f}")
                    print(f"      ✅ Albedo factor: {albedo_factor:.4f}")
                    
                except Exception as e:
                    print(f"   ❌ Error processing Body {body_id}: {e}")
                    self.texture_averages[body_id] = 0.216
                    self.albedo_factors[body_id] = target_albedos[body_id] / 0.216

        def _setup_hapke_ranges(self):
            """Setup Hapke BRDF parameter ranges based on N_templates"""
            # Variable Hapke parameter ranges for Phobos
            # --------- G E N İ Ş   A R A M A   U Z A Y I  (b & h_S dâhil) ---------
            #   Kaynak: Fornasier-2024, Simonelli-1998  (+ güvenli tampon)
            #   q_eff: HRSC-SRC radyometrik belirsizlik ±15 %  (Gwinner, ISPRS 2023)
            self.hapke_ranges = {
                'w':         {'min': 0.04,  'max': 0.20},   # biraz daha geniş
                'b':         {'min': -0.60, 'max': 0.60},   # asimetri
                'theta_bar': {'min':  5.0,  'max': 35.0},   # makro pürüz
                'B_S0':      {'min': 0.2,   'max': 3.0},    # opp. surge genliği
                'h_S':       {'min': 0.01,  'max': 0.15},   # opp. genişliği
                'q_eff':     {'min': 0.85,  'max': 2.05},   # ±15 %
            }

            # Sabit tek parametre
            self.hapke_fixed = {'c': 0.5}                   # back-scatter payı
            
            # Calculate number of steps per parameter
            n_params = len(self.hapke_ranges)
            self.steps_per_param = int(np.round(self.N_templates ** (1.0 / n_params)))
            actual_templates = self.steps_per_param ** n_params
            
            print(f"\n📊 Hapke BRDF Parameter Setup:")
            print(f"   Requested templates: {self.N_templates}")
            print(f"   Variable parameters: {n_params}")
            print(f"   Steps per parameter: {self.steps_per_param}")
            print(f"   Actual templates: {actual_templates}")
            print(f"   Fixed parameter:  c={self.hapke_fixed['c']}")
            
            # Generate parameter values
            self.param_values = {}
            for param, range_dict in self.hapke_ranges.items():
                if self.steps_per_param == 1:
                    # Single value: use midpoint
                    values = [(range_dict['min'] + range_dict['max']) / 2.0]
                else:
                    # Multiple values: linspace including endpoints
                    values = np.linspace(range_dict['min'], range_dict['max'], self.steps_per_param).tolist()
                
                self.param_values[param] = values
                print(f"   {param}: {[f'{v:.3f}' for v in values]}")

        def _generate_parameter_combinations(self):
            """Generate all parameter combinations for templates"""
            import itertools
            
            # Get all parameter names and values
            param_names = list(self.param_values.keys())
            param_value_lists = [self.param_values[name] for name in param_names]
            
            # Generate all combinations
            combinations = []
            for combo in itertools.product(*param_value_lists):
                param_dict = dict(zip(param_names, combo))
                combinations.append(param_dict)
            
            return combinations

        def _load_or_create_config(self, config_path):
            """Load or create default configuration"""
            if config_path and Path(config_path).exists():
                with open(config_path, 'r') as f:
                    return json.load(f)
            
            base_dir = Path(os.getcwd())
            return {
                'input_path': str(base_dir / "input" / "S07_Mars_Phobos_Deimos"),
                'output_path': str(base_dir / 'output' / 'photometric_templates'),
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
                    return {
                        'fov': 0.54, 
                        'res_x': 1024, 
                        'res_y': 1024, 
                        'film_exposure': 1.0,
                        'sensor': 'BW', 
                        'clip_start': 0.1, 
                        'clip_end': 100000000.0, 
                        'bit_encoding': '16', 
                        'viewtransform': 'Standard', 
                        'K': [[1222.0, 0, 512.0], [0, 1222.0, 512.0], [0, 0, 1]]
                    }

        def _calculate_sun_strength(self, solar_distance_km, q_eff=1.0):
            """Calculate sun strength
            
            Args:
                solar_distance_km: Distance from sun to Phobos in km
                q_eff: Quantum efficiency factor (default 1.0)
                
            Returns:
                sun_strength: Sun strength for Blender scene
            """
            # Convert distance to AU
            Phobos_Solar_Dist_AU = solar_distance_km / AU_KM
            
            # Calculate irradiance at given distance
            W_at_1_AU = 427.815  # W/m² between 475-725 nm at 1 AU
            W_at_given_AU = W_at_1_AU / (Phobos_Solar_Dist_AU**2)
            
            # Apply quantum efficiency factor
            W_effective = W_at_given_AU * q_eff
            
            # Calculate sun strength for scene
            Sun_strength_in_scene = self.Sun_Blender_scaler * W_effective
            
            return Sun_strength_in_scene

        def _setup_material_with_hapke_params(self, materials, bodies, hapke_params, albedo_factors):
            """Setup materials with Hapke-like approximation using Blender nodes"""
            
            for i, (material, body) in enumerate(zip(materials, bodies), 1):
                if material.node_tree:
                    nodes = material.node_tree.nodes
                    links = material.node_tree.links
                    
                    # Store texture images
                    texture_images = {}
                    for node in nodes:
                        if node.type == 'TEX_IMAGE' and node.image:
                            texture_images[node.name] = node.image
                            node.image.colorspace_settings.name = 'Non-Color'
                    
                    # Clear all nodes
                    nodes.clear()
                    
                    # Create output node
                    output_node = nodes.new(type='ShaderNodeOutputMaterial')
                    output_node.location = (800, 0)
                    
                    # Create texture node if we have textures
                    texture_node = None
                    if texture_images:
                        texture_node = nodes.new(type='ShaderNodeTexImage')
                        texture_node.location = (-600, 0)
                        texture_node.name = f"CORTOPY_Texture_Body_{i}"
                        
                        first_image = list(texture_images.values())[0]
                        texture_node.image = first_image
                        texture_node.image.colorspace_settings.name = 'Non-Color'
                    
                    # Albedo multiply node
                    albedo_multiply = nodes.new(type='ShaderNodeMath')
                    albedo_multiply.operation = 'MULTIPLY'
                    albedo_multiply.location = (-400, 0)
                    albedo_multiply.inputs[1].default_value = albedo_factors.get(i, 0.127)
                    
                    # Get Hapke parameters for this body
                    body_params = self.static_body_brdf.get(i, {})
                    w_val = body_params.get('w', hapke_params.get('w', 0.3))
                    theta_bar_val = body_params.get('theta_bar', hapke_params.get('theta_bar', 20.0))
                    b_val    = hapke_params.get('b',   0.0)
                    B_S0_val = hapke_params.get('B_S0', 1.0)
                    h_S_val  = hapke_params.get('h_S', 0.06)
                    
                    # Single scattering albedo multiply
                    w_multiply = nodes.new(type='ShaderNodeMath')
                    w_multiply.operation = 'MULTIPLY'
                    w_multiply.location = (-200, 0)
                    w_multiply.inputs[1].default_value = w_val
                    
                    # Create Mix Shader for combining diffuse and glossy
                    mix_shader = nodes.new(type='ShaderNodeMixShader')
                    mix_shader.location = (400, 0)
                    
                    # Diffuse BSDF (main component)
                    diffuse_bsdf = nodes.new(type='ShaderNodeBsdfDiffuse')
                    diffuse_bsdf.location = (0, 100)
                    
                    # Glossy BSDF (for backscatter approximation)
                    glossy_bsdf = nodes.new(type='ShaderNodeBsdfGlossy')
                    glossy_bsdf.location = (0, -100)
                    
                    # Set roughness based on theta_bar
                    roughness_val = theta_bar_val / 90.0  # Normalize to 0-1
                    glossy_bsdf.inputs['Roughness'].default_value = roughness_val
                    
                    # Mix factor based on B_S0 (backscatter strength)
                    # ---------- Opp. surge + asimetri karışım faktörü ----------
                    base_fac = min(B_S0_val * 0.1, 0.5)
                    # h_S genişliği -> daha geniş tepe  =>   küçük efektif fac
                    base_fac *= np.exp(-h_S_val * 8.0)
                    # b asimetri -> ileri saçılma (+) fac'i azaltır, geri (-) arttırır
                    base_fac *= (1.0 - 0.5 * b_val)

                    # Bir Value düğümü üzerinden bağla  (node eklenmiş olsun)
                    fac_val_node = nodes.new(type='ShaderNodeValue')
                    fac_val_node.location = (200, -250)
                    fac_val_node.label = "B_S0·h_S·b"
                    fac_val_node.outputs[0].default_value = base_fac
                    links.new(fac_val_node.outputs['Value'], mix_shader.inputs['Fac'])
                    # Connect nodes
                    if texture_node:
                        links.new(texture_node.outputs['Color'], albedo_multiply.inputs[0])
                    else:
                        albedo_multiply.inputs[0].default_value = 1.0
                        
                    links.new(albedo_multiply.outputs['Value'], w_multiply.inputs[0])
                    
                    # Connect to both BSDFs
                    links.new(w_multiply.outputs['Value'], diffuse_bsdf.inputs['Color'])
                    links.new(w_multiply.outputs['Value'], glossy_bsdf.inputs['Color'])
                    
                    # Connect BSDFs to mix shader
                    links.new(diffuse_bsdf.outputs['BSDF'], mix_shader.inputs[1])
                    links.new(glossy_bsdf.outputs['BSDF'], mix_shader.inputs[2])
                    
                    # Connect to output
                    links.new(mix_shader.outputs['Shader'], output_node.inputs['Surface'])
                    
                    print(f"   ✅ Body {i} Hapke-like material: "
                        f"w={w_val:.3f}, b={b_val:+.2f}, θ̅={theta_bar_val:.1f}°, "
                        f"B_S0={B_S0_val:.2f}, h_S={h_S_val:.3f}")
                        
        def _add_paths_to_state(self, state):
            """Add required paths to state"""
            state.add_path('albedo_path_1', os.path.join(state.path["input_path"], 'body', 'albedo', 'Phobos_Viking_Mosaic_40ppd_DLRcontrol.tif'))
            state.add_path('albedo_path_2', os.path.join(state.path["input_path"], 'body', 'albedo', 'Mars_grayscale.tif'))
            state.add_path('albedo_path_3', os.path.join(state.path["input_path"], 'body', 'albedo', 'Deimos grayscale.jpg'))
            state.add_path('uv_data_path_1', os.path.join(state.path["input_path"], 'body', 'uv data', 'g_phobos_036m_spc_0000n00000_v002.json'))
            state.add_path('uv_data_path_2', os.path.join(state.path["input_path"], 'body', 'uv data', 'Mars_65k.json'))
            state.add_path('uv_data_path_3', os.path.join(state.path["input_path"], 'body', 'uv data', 'g_deimos_162m_spc_0000n00000_v001.json'))

        def _setup_compositing(self, state):
            """Setup compositing for 16-bit PNG output with all required outputs"""
            tree = corto.Compositing.create_compositing()
            render_node = corto.Compositing.rendering_node(tree, (0, 0))
            
            # Create simple denoised output to PNG
            denoise_node = corto.Compositing.denoise_node(tree, (400, 0))
            composite_node = corto.Compositing.composite_node(tree, (600, 0))
            
            # Simple chain: render -> denoise -> composite
            tree.links.new(render_node.outputs["Image"], denoise_node.inputs["Image"]) 
            tree.links.new(denoise_node.outputs["Image"], composite_node.inputs["Image"])
            
            # Other branches
            corto.Compositing.create_depth_branch(tree, render_node)
            corto.Compositing.create_slopes_branch(tree, render_node, state)
            corto.Compositing.create_maskID_branch(tree, render_node, state)
            
            return tree

        def setup_scene_base(self, utc_time, spice_data=None):
            """Setup base scene without rendering - for template generation"""
            print(f"\n🏗️ Setting up base scene for: {utc_time}")

            # Clean scene
            corto.Utils.clean_scene()
            
            # Get SPICE data
            if spice_data is None:
                try:
                    spice_data = self.spice_processor.get_spice_data(utc_time)
                    solar_distance_km = spice_data.get("distances", {}).get("sun_to_phobos", AU_KM)
                except Exception:
                    # Fallback SPICE data
                    spice_data = self._get_fallback_spice_data()
                    solar_distance_km = AU_KM
            else:
                solar_distance_km = spice_data.get("distances", {}).get("sun_to_phobos", AU_KM)
                # Normalize quaternions
                for body in ("sun", "hrsc", "phobos", "mars", "deimos"):
                    q = np.array(spice_data.get(body, {}).get("quaternion", []), dtype=float)
                    norm = np.linalg.norm(q)
                    if norm > 0:
                        spice_data[body]["quaternion"] = (q / norm).tolist()
            
            # Create output directory structure
            output_dir = Path(self.config["output_path"])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (output_dir / "img").mkdir(exist_ok=True)
            (output_dir / "depth").mkdir(exist_ok=True)
            (output_dir / "slopes").mkdir(exist_ok=True)
            (output_dir / "mask_ID_1").mkdir(exist_ok=True)
            (output_dir / "mask_ID_shadow_1").mkdir(exist_ok=True)
            (output_dir / "blend").mkdir(exist_ok=True)

            # Create geometry and scene data
            geometry_data = {
                "sun": {"position": [convert_numpy_types(spice_data["sun"]["position"])],
                        "orientation": [convert_numpy_types(spice_data["sun"]["quaternion"])]},
                "camera": {"position": [convert_numpy_types(spice_data["hrsc"]["position"])],
                        "orientation": [convert_numpy_types(spice_data["hrsc"]["quaternion"])]},
                "body_1": {"position": [convert_numpy_types(spice_data["phobos"]["position"])],
                        "orientation": [convert_numpy_types(spice_data["phobos"]["quaternion"])]},
                "body_2": {"position": [convert_numpy_types(spice_data["mars"]["position"])],
                        "orientation": [convert_numpy_types(spice_data["mars"]["quaternion"])]},
                "body_3": {"position": [convert_numpy_types(spice_data["deimos"]["position"])],
                        "orientation": [convert_numpy_types(spice_data["deimos"]["quaternion"])]},
            }

            # Default sun energy (will be modified per template)
            base_sun_energy = self._calculate_sun_strength(solar_distance_km, q_eff=1.0)

            scene_config = {
                "camera_settings": {
                    **{k: (float(v) if k in ["fov","film_exposure","clip_start","clip_end"]
                        else int(v) if k in ["res_x","res_y"]
                        else str(v))
                    for k,v in self.camera_config.items()},
                    "K": self.camera_config.get("K", [[1222,0,512],[0,1222,512],[0,0,1]])
                },
                "sun_settings": {
                    "angle": 0.00927,
                    "energy": base_sun_energy
                },
                "body_settings_1": {"pass_index": 1, "diffuse_bounces": 4},
                "body_settings_2": {"pass_index": 2, "diffuse_bounces": 4},
                "body_settings_3": {"pass_index": 3, "diffuse_bounces": 4},
                "rendering_settings": {"engine": "CYCLES", "device": "GPU", "samples": 256, "preview_samples": 16},
            }

            # Save JSON files
            geom_path = output_dir / "geometry_dynamic.json"
            scene_path = output_dir / "scene_src.json"
            geom_path.write_text(json.dumps(geometry_data, indent=4))
            scene_path.write_text(json.dumps(convert_numpy_types(scene_config), indent=4))

            # Create CORTO state and environment
            state = corto.State(scene=str(scene_path), geometry=str(geom_path),
                            body=self.config["body_files"], scenario=self.scenario_name)
            self._add_paths_to_state(state)

            # Create camera
            cam_props = {k: (float(v) if k in ["fov","film_exposure","clip_start","clip_end"]
                            else int(v) if k in ["res_x","res_y"]
                            else str(v))
                        for k,v in self.camera_config.items()}
            cam_props["K"] = self.camera_config.get("K", [[1222,0,512],[0,1222,512],[0,0,1]])
            cam = corto.Camera(f"HRSC_{self.camera_type}_Camera", cam_props)

            # Create sun
            sun = corto.Sun("Sun", {"angle": 0.00927, "energy": base_sun_energy})

            # Create bodies
            bodies = []
            for i, body_name in enumerate([Path(b).stem for b in self.config["body_files"]], 1):
                bodies.append(corto.Body(body_name, {"pass_index": i, "diffuse_bounces": 4}))

            # Create rendering engine
            rendering = corto.Rendering({"engine": "CYCLES", "device": "GPU", "samples": 256, "preview_samples": 16})
            
            # Setup environment
            env = corto.Environment(cam, bodies, sun, rendering)

            # Setup materials
            materials = []
            for i, body in enumerate(bodies, 1):
                mat = corto.Shading.create_new_material(f"hapke_material_{i}")
                if hasattr(corto.Shading, "create_branch_albedo_mix"):
                    corto.Shading.create_branch_albedo_mix(mat, state, i)
                if hasattr(corto.Shading, "load_uv_data"):
                    corto.Shading.load_uv_data(body, state, i)
                corto.Shading.assign_material_to_object(mat, body)
                materials.append(mat)

            # Setup compositing
            tree = self._setup_compositing(state)

            # Set scales
            bodies[0].set_scale(np.array([1.0, 1.0, 1.0]))      # Phobos
            bodies[1].set_scale(np.array([1000.0, 1000.0, 1000.0]))  # Mars  
            bodies[2].set_scale(np.array([1.0, 1.0, 1.0]))      # Deimos

            # Position all objects once
            env.PositionAll(state, index=0)
            
            # Override sun orientation with SPICE data
            if 'sun' in spice_data and 'quaternion' in spice_data['sun']:
                sun_orientation = spice_data['sun']['quaternion']
                sun.SUN_Blender.rotation_mode = 'QUATERNION'
                sun.SUN_Blender.rotation_quaternion = sun_orientation

            return state, env, cam, bodies, sun, materials, tree, spice_data, solar_distance_km

        def _get_fallback_spice_data(self):
            """Get fallback SPICE data"""
            return {
                "sun": {"position": [0, 0, -149597870.7], "quaternion": [0.81961521, -0.43493541, -0.37291031, 0.0]},
                "hrsc": {"position": [0, 0, 0], "quaternion": [0.13266801, -0.13622879, -0.89072904, -0.41284706]},
                "phobos": {"position": [0, 0, 9376], "quaternion": [0.78270147, 0.03431313, 0.31611934, -0.53504167]}, 
                "mars": {"position": [0, 0, 0], "quaternion": [0.68522767, -0.00762764, 0.31824602, -0.65507582]},
                "deimos": {"position": [0, 0, 23463.2], "quaternion": [0.86976492, 0.29825867, 0.11086453, 0.37717343]},
                "distances": {"sun_to_phobos": AU_KM}
            }

        def _save_raw_16bit_png(self, image_data, output_path):
            """Save image data as raw 16-bit PNG preserving full precision"""
            try:
                # Ensure we have uint16 data
                if image_data.dtype == np.uint16:
                    raw_uint16 = image_data
                elif image_data.dtype == np.uint8:
                    raw_uint16 = (image_data.astype(np.uint16) << 8)
                else:
                    if image_data.max() <= 1.0:
                        raw_uint16 = (image_data * 65535).astype(np.uint16)
                    else:
                        raw_uint16 = np.clip(image_data, 0, 65535).astype(np.uint16)
                
                # Save using PIL's 16-bit mode
                Image.fromarray(raw_uint16, mode='I;16').save(str(output_path))
                return True
                
            except Exception as e:
                print(f"   ❌ Error saving raw 16-bit PNG: {e}")
                return False

        def render_template_batch(self, state, env, cam, bodies, sun, materials, tree, spice_data, 
                                solar_distance_km, img_index, img_filename):
            """Render all templates for a single scene"""
            print(f"\n🎬 Rendering templates for image {img_index}: {img_filename}")
            
            # Configure for 16-bit PNG output
            cam_obj = cam.toBlender()                    # create / update the Blender camera
            # Blender refuses to render if scene.camera is None
            if bpy.context.scene.camera is None:
                bpy.context.scene.camera = cam_obj or bpy.data.objects.get(cam.name)
            bpy.context.scene.render.image_settings.file_format = 'PNG'
            bpy.context.scene.render.image_settings.color_depth = '16'
            bpy.context.scene.view_settings.view_transform = 'Raw'
            bpy.context.scene.render.image_settings.compression = 0

            # Generate parameter combinations
            param_combinations = self._generate_parameter_combinations()
            
            # Initialize Excel data for this batch
            batch_results = []
            
            for template_idx, hapke_params in enumerate(param_combinations):
                print(f"\n📸 Template {template_idx + 1}/{len(param_combinations)}")
                print(f"   Hapke params: {', '.join(f'{k}={v:.3f}' for k,v in hapke_params.items())}")
                
                try:
                    # Update sun energy based on q_eff
                    sun_energy = self._calculate_sun_strength(
                                    solar_distance_km, hapke_params['q_eff'])
                    sun.set_energy(sun_energy)
                    
                    # Update materials with Hapke parameters
                    self._setup_material_with_hapke_params(materials, bodies, hapke_params, self.albedo_factors)
                    
                    # Define output paths
                    template_name = f"img_{img_index:06d}_template_{template_idx:04d}"
                    linear_img_path = Path(state.path["output_path"]) / "img" / f"{template_name}.png"
                    processed_img_path = Path(state.path["output_path"]) / "img" / f"processed_{template_name}.png"
                    
                    # Render
                    bpy.context.scene.render.filepath = str(linear_img_path)
                    bpy.context.scene.frame_current = img_index * 1000 + template_idx  # Unique frame number
                    bpy.ops.render.render(write_still=True)

                    # Process rendered image
                    if linear_img_path.exists():
                        rendered_img_16bit = np.array(Image.open(str(linear_img_path)))
                        
                        if len(rendered_img_16bit.shape) == 3:
                            rendered_img_16bit = rendered_img_16bit[:,:,0]
                        
                        # Ensure proper 16-bit format
                        if rendered_img_16bit.dtype != np.uint16:
                            if rendered_img_16bit.max() <= 1.0:
                                rendered_img_16bit = (rendered_img_16bit * 65535).astype(np.uint16)
                            else:
                                rendered_img_16bit = rendered_img_16bit.astype(np.uint16)
                        
                        # Save original as raw 16-bit PNG
                        self._save_raw_16bit_png(rendered_img_16bit, linear_img_path)
                        
                        # CORTO processing
                        labels = {
                            'CoB': [rendered_img_16bit.shape[1]//2, rendered_img_16bit.shape[0]//2], 
                            'range': float(np.linalg.norm(spice_data["phobos"]["position"])), 
                            'phase_angle': 0.0
                        }
                        
                        # Convert to RGB for CORTO processing
                        if len(rendered_img_16bit.shape) == 2:
                            rendered_rgb_16bit = np.stack([rendered_img_16bit] * 3, axis=-1)
                        else:
                            rendered_rgb_16bit = rendered_img_16bit
                        
                        # Convert to uint8 for CORTO
                        rendered_uint8_for_corto = (rendered_rgb_16bit / 256).astype(np.uint8)
                        
                        processed_img, _ = self.post_processor.process_image_label_pair_no_stretch(
                            rendered_uint8_for_corto, labels, target_size=192, mask_path=None)
                        
                        if len(processed_img.shape) == 3:
                            processed_gray = cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY)
                        else:
                            processed_gray = processed_img
                        
                        # Scale back to 16-bit
                        processed_16bit = (processed_gray.astype(np.float32) * 256).astype(np.uint16)
                        
                        # Save processed version
                        self._save_raw_16bit_png(processed_16bit, processed_img_path)
                    
                    # Record results with Hapke parameters
                    result = {
                        'img_index': img_index,
                        'img_filename': img_filename,
                        'template_index': template_idx,
                        'template_name': template_name,
                        'w': hapke_params['w'],
                        'b':        hapke_params['b'],
                        'theta_bar':hapke_params['theta_bar'],
                        'B_S0': hapke_params['B_S0'],
                        'h_S':   hapke_params['h_S'],
                        'q_eff': hapke_params['q_eff'],
                        'sun_energy': sun_energy,
                        'linear_img_path': str(linear_img_path),
                        'processed_img_path': str(processed_img_path),
                        'render_status': 'SUCCESS'
                    }
                    
                    batch_results.append(result)
                    self.template_results.append(result)
                    
                    # Save to Excel after each template (in case of crash)
                    self._save_results_to_excel()
                    
                    print(f"   ✅ Template rendered successfully")
                    
                except Exception as e:
                    print(f"   ❌ Error rendering template: {e}")
                    result = {
                        'img_index': img_index,
                        'img_filename': img_filename,
                        'template_index': template_idx,
                        'template_name': f"FAILED_{template_idx}",
                        'w': hapke_params.get('w', 0),
                        'b':        hapke_params.get('b', 0),
                        'theta_bar':hapke_params.get('theta_bar', 0),
                        'B_S0': hapke_params.get('B_S0', 0),
                        'h_S':   hapke_params.get('h_S', 0),
                        'q_eff': hapke_params.get('q_eff', 0),
                        'sun_energy': 0,
                        'linear_img_path': 'FAILED',
                        'processed_img_path': 'FAILED',
                        'render_status': 'FAILED'
                    }
                    batch_results.append(result)
                    self.template_results.append(result)

            # Save blend file for this scene
            corto.Utils.save_blend(state, f'template_batch_{img_index}_{img_filename.replace(".IMG", "")}')
            
            return batch_results

        def _save_results_to_excel(self):
            """Save current results to Excel file"""
            if not self.template_results:
                return
            
            df = pd.DataFrame(self.template_results)
            
            if self.output_excel_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = Path(self.config['output_path'])
                self.output_excel_path = output_dir / f"hapke_template_results_{timestamp}.xlsx"
            
            try:
                df.to_excel(self.output_excel_path, index=False)
                print(f"   💾 Results saved to: {self.output_excel_path}")
            except Exception as e:
                print(f"   ⚠️ Could not save Excel: {e}")
                # Try CSV as backup
                csv_path = self.output_excel_path.with_suffix('.csv')
                df.to_csv(csv_path, index=False)
                print(f"   💾 Results saved to CSV: {csv_path}")

        def run_pipeline_with_pds(self, pds_data_path, max_simulations=None):
            """Run pipeline using PDS data"""
            print("\n1. Processing PDS database...")
            img_database = self.pds_processor.parse_and_process_directory(pds_data_path)
            
            # Save database
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(self.config['output_path'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                img_database.to_excel(output_dir / f"pds_img_database_{timestamp}.xlsx", index=False)
            except Exception:
                img_database.to_csv(output_dir / f"pds_img_database_{timestamp}.csv", index=False)
            
            # Run simulations
            print("\n2. Running Hapke template generation...")
            valid_entries = img_database[img_database['STATUS'] == 'SUCCESS'].copy()
            if max_simulations:
                valid_entries = valid_entries.head(max_simulations)
            
            print(f"Processing {len(valid_entries)} images with {self.N_templates} templates each")
            print(f"Total templates to generate: {len(valid_entries) * self.N_templates}")
            
            for idx, row in valid_entries.iterrows():
                utc_time = row['UTC_MEAN_TIME']
                img_filename = row['file_name']
                
                print(f"\n{'='*80}")
                print(f"Processing image {idx+1}/{len(valid_entries)}: {img_filename}")
                print(f"{'='*80}")
                
                # Setup base scene
                state, env, cam, bodies, sun, materials, tree, spice_data, solar_distance_km = \
                    self.setup_scene_base(utc_time)
                
                # Render all templates for this scene
                batch_results = self.render_template_batch(
                    state, env, cam, bodies, sun, materials, tree, spice_data, 
                    solar_distance_km, idx, img_filename
                )
                
                print(f"\n✅ Completed {len(batch_results)} templates for {img_filename}")
            
            # Final summary
            print(f"\n{'='*80}")
            print(f"🎯 PIPELINE COMPLETE - Hapke BRDF Templates")
            print(f"Total templates generated: {len(self.template_results)}")
            print(f"Results saved to: {self.output_excel_path}")
            print(f"{'='*80}")
            
            return self.template_results


    class CompactPDSProcessor:
        """Compact PDS processor"""
        
        def __init__(self, pds_data_path="PDS_Data"):
            self.pds_data_path = Path(pds_data_path)
            import re
            self._key_val = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=\s*(.*)$")

        def parse_and_process_directory(self, img_directory_path):
            """Parse and process IMG directory"""
            records = []
            img_dir = Path(img_directory_path)
            
            print(f"Processing IMG files in: {img_dir}")
            
            for img_file in img_dir.rglob("*.IMG"):
                try:
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


    def main(pds_data_path=None, max_simulations=None, camera_type='SRC',
            Sun_Blender_scaler=3.90232e-1, N_templates=27,
            mars_w=0.4, mars_theta_bar=20.0,
            deimos_w=0.3, deimos_theta_bar=20.0):
        """Main function to run photometric template generator with Hapke BRDF
        
        Args:
            pds_data_path: Path to PDS IMG files
            max_simulations: Maximum number of images to process
            camera_type: Camera type ('SRC')
            Sun_Blender_scaler: Scaling factor for sun strength
            N_templates: Number of templates to generate per image
            mars_w: Fixed single scattering albedo for Mars
            mars_theta_bar: Fixed roughness angle for Mars
            deimos_w: Fixed single scattering albedo for Deimos
            deimos_theta_bar: Fixed roughness angle for Deimos
        """
        print("="*80)
        print("Photometric Template Generator with Hapke BRDF")
        print("For Mars, Phobos, and Deimos")
        print("="*80)
        
        generator = PhotometricTemplateGenerator(
            camera_type=camera_type,
            pds_data_path=pds_data_path,
            Sun_Blender_scaler=Sun_Blender_scaler,
            N_templates=N_templates,
            mars_w=mars_w, mars_theta_bar=mars_theta_bar,
            deimos_w=deimos_w, deimos_theta_bar=deimos_theta_bar
        )
        
        if pds_data_path:
            print(f"Using PDS data from: {pds_data_path}")
            template_results = generator.run_pipeline_with_pds(pds_data_path, max_simulations)
        else:
            print("No PDS data path provided")
            return None
        
        return template_results


    if __name__ == "__main__":
        import argparse
        
        PDS_path = '/home/tt_mmx/corto/PDS_Data'
        
        parser = argparse.ArgumentParser(description="Photometric Template Generator with Hapke BRDF")
        parser.add_argument("--pds_data_path", default=PDS_path, help="PDS IMG directory path")
        parser.add_argument("--max_simulations", type=int, default=5, help="Maximum number of images to process")
        parser.add_argument("--camera_type", default="SRC", choices=["SRC"], help="HRSC camera type")
        parser.add_argument("--sun_scaler", type=float, default=3.90232e-1, help="Sun Blender scaler")
        parser.add_argument("--n_templates", type=int, default=2500, help="Number of templates per image")
        # Fixed Mars / Deimos Hapke values
        parser.add_argument("--mars_w", type=float, default=0.4, help="Fixed single scattering albedo for Mars")
        parser.add_argument("--mars_theta_bar", type=float, default=20.0, help="Fixed roughness angle for Mars (degrees)")
        parser.add_argument("--deimos_w", type=float, default=0.3, help="Fixed single scattering albedo for Deimos")
        parser.add_argument("--deimos_theta_bar", type=float, default=20.0, help="Fixed roughness angle for Deimos (degrees)")
        
        args = parser.parse_args()
        
        # Real scale value is 3.90232e-4 but multiplied by 1000 for PNG integer storage
        # The real radiance value is RADIANCE = DN * 10**-3
        
        template_results = main(
            pds_data_path=args.pds_data_path,
            max_simulations=args.max_simulations,
            camera_type=args.camera_type,
            Sun_Blender_scaler=args.sun_scaler,
            N_templates=args.n_templates,
            mars_w=args.mars_w, mars_theta_bar=args.mars_theta_bar,
            deimos_w=args.deimos_w, deimos_theta_bar=args.deimos_theta_bar
        )
