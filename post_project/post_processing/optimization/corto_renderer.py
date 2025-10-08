#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
corto_renderer.py
=================
CORTO rendering wrapper for optimization.
"""

import sys
import os
import numpy as np
import bpy
from pathlib import Path
from typing import Dict, List, Optional
import tempfile
import shutil

# Add paths
sys.path.append(os.getcwd())
import cortopy as corto

# Add post_project to path
post_project_path = Path(__file__).parent.parent
sys.path.append(str(post_project_path))

from phobos_scene import (
    force_nvidia_gpu,
    build_scene_and_materials,
    set_phobos_params,
    add_asset_paths
)

from phobos_data import (
    calculate_sun_strength,
    get_camera_config,
    get_spice_data_for_time,
    write_dynamic_scene_files,
    HAS_SPICE,
    SpiceDataProcessor
)


# ============================================================================
# SPICE DATA CACHE
# ============================================================================

class SPICEDataCache:
    """Cache SPICE data for multiple IMG files to avoid redundant queries."""
    
    def __init__(self):
        self.cache = {}
        self.sdp = None
        
        if HAS_SPICE:
            self.sdp = SpiceDataProcessor()
    
    def get_spice_data(self, utc_time: str) -> Dict:
        """Get SPICE data with caching."""
        if utc_time not in self.cache:
            if self.sdp is None:
                raise RuntimeError("SPICE not available!")
            
            self.cache[utc_time] = get_spice_data_for_time(self.sdp, utc_time)
        
        return self.cache[utc_time]


# ============================================================================
# CORTO RENDERER CLASS
# ============================================================================

class CORTORenderer:
    """
    Wrapper for CORTO rendering with photometric parameters.
    """
    
    def __init__(self,
                 scenario_name: str = "S07_Mars_Phobos_Deimos",
                 body_files: List[str] = None,
                 temp_dir: Optional[Path] = None,
                 cleanup: bool = True):
        """
        Initialize CORTO renderer.
        
        Args:
            scenario_name: CORTO scenario name
            body_files: List of body OBJ files [phobos.obj, mars.obj, deimos.obj]
            temp_dir: Temporary directory for intermediate files
            cleanup: Whether to clean up temp files after rendering
        """
        self.scenario_name = scenario_name
        
        if body_files is None:
            self.body_files = [
                "g_phobos_036m_spc_0000n00000_v002.obj",
                "Mars_65k.obj",
                "g_deimos_162m_spc_0000n00000_v001.obj",
            ]
        else:
            self.body_files = body_files
        
        # Temp directory
        if temp_dir is None:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="corto_opt_"))
        else:
            self.temp_dir = Path(temp_dir)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.cleanup = cleanup
        
        # SPICE cache
        self.spice_cache = SPICEDataCache()
        
        # Camera config (load once)
        self.camera_config = get_camera_config()
        
        # Sun scaler (from phobos_main.py)
        self.sun_blender_scaler = 3.90232e-1
        
        print(f"CORTO Renderer initialized")
        print(f"  Scenario: {scenario_name}")
        print(f"  Temp dir: {self.temp_dir}")
        print(f"  Cleanup: {cleanup}")
    
    def render_for_image(self,
                         img_info: Dict,
                         params: np.ndarray,
                         output_path: Path) -> bool:
        """
        Render synthetic image for a single real IMG file.
        
        Args:
            img_info: Dictionary with IMG metadata:
                - 'filename': IMG filename
                - 'utc_time': UTC timestamp
                - 'solar_distance_km': Distance to sun
            params: Photometric parameter array [7 values]
            output_path: Where to save rendered PNG
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clean scene
            corto.Utils.clean_scene()
            force_nvidia_gpu()
            
            # Get SPICE data
            spice_data = self.spice_cache.get_spice_data(img_info['utc_time'])
            
            # Calculate sun energy
            solar_distance_km = spice_data['distances']['sun_to_phobos']
            base_energy = calculate_sun_strength(
                solar_distance_km,
                q_eff=1.0,
                sun_blender_scaler=self.sun_blender_scaler
            )
            
            # Create dynamic scene files
            scene_filename, geom_filename = write_dynamic_scene_files(
                self.temp_dir,
                spice_data,
                self.camera_config,
                base_energy,
                0  # Single frame
            )
            
            # Create CORTO State
            state = corto.State(
                scene=scene_filename,
                geometry=geom_filename,
                body=self.body_files,
                scenario=self.scenario_name
            )
            
            # Add asset paths
            add_asset_paths(state)
            
            # Build scene and materials
            ENV, cam, sun, bodies, mat_nodes, tree = build_scene_and_materials(
                state, self.body_files
            )
            
            # **APPLY PHOTOMETRIC PARAMETERS**
            params_dict = self._params_to_dict(params)
            set_phobos_params(mat_nodes, params_dict)
            
            # Calculate sun energy with q_eff from parameters
            q_eff = float(params_dict['q_eff'])
            sun_energy = calculate_sun_strength(
                solar_distance_km,
                q_eff=q_eff,
                sun_blender_scaler=self.sun_blender_scaler
            )
            sun.set_energy(sun_energy)
            
            # Position objects
            ENV.PositionAll(state, index=0)
            
            # Select camera
            corto.Camera.select_camera(cam.name)
            
            # Set output path
            bpy.context.scene.render.filepath = str(output_path.with_suffix(''))
            
            # Configure for 16-bit PNG
            bpy.context.scene.render.image_settings.file_format = 'PNG'
            bpy.context.scene.render.image_settings.color_depth = '16'
            bpy.context.scene.render.image_settings.compression = 0
            bpy.context.scene.view_settings.view_transform = 'Raw'
            
            # Render
            bpy.ops.render.render(write_still=True)
            
            # Blender adds .png extension
            rendered_file = output_path.with_suffix('.png')
            
            if not rendered_file.exists():
                print(f"Error: Rendered file not found: {rendered_file}")
                return False
            
            print(f"✓ Rendered: {rendered_file.name}")
            return True
            
        except Exception as e:
            print(f"Error rendering for {img_info['filename']}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _params_to_dict(self, params: np.ndarray) -> Dict[str, float]:
        """Convert parameter array to dictionary."""
        param_names = ['base_gray', 'tex_mix', 'oren_rough', 'princ_rough', 
                      'shader_mix', 'ior', 'q_eff']
        return {name: float(val) for name, val in zip(param_names, params)}
    
    def cleanup_temp(self):
        """Clean up temporary files."""
        if self.cleanup and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                print(f"Cleaned up temp dir: {self.temp_dir}")
            except Exception as e:
                print(f"Warning: Could not clean temp dir: {e}")
    
    def __del__(self):
        """Destructor - clean up on deletion."""
        self.cleanup_temp()


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def render_synthetic_for_params(
    img_info_list: List[Dict],
    params: np.ndarray,
    output_dir: Path,
    particle_id: int
) -> List[Path]:
    """
    Render synthetic images for multiple real IMG files with given parameters.
    
    Args:
        img_info_list: List of IMG info dicts (filename, utc_time, etc.)
        params: Photometric parameters [7 values]
        output_dir: Directory for output files
        particle_id: Particle/individual ID for naming
        
    Returns:
        List of paths to rendered PNG files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    renderer = CORTORenderer()
    rendered_files = []
    
    for i, img_info in enumerate(img_info_list):
        # Output filename
        img_stem = Path(img_info['filename']).stem
        output_file = output_dir / f"particle{particle_id:03d}_img{i:02d}_{img_stem}.png"
        
        # Render
        success = renderer.render_for_image(img_info, params, output_file)
        
        if success:
            rendered_files.append(output_file)
        else:
            rendered_files.append(None)
    
    renderer.cleanup_temp()
    
    return rendered_files


if __name__ == "__main__":
    # Test rendering
    print("Testing CORTO Renderer...")
    
    # Example IMG info
    img_info = {
        'filename': 'H9463_0050_SR2.IMG',
        'utc_time': '2019-06-07T10:46:42.8575Z',
        'solar_distance_km': 228000000.0
    }
    
    # Example parameters
    params = np.array([0.15, 0.30, 0.45, 0.60, 0.25, 1.40, 0.85])
    
    # Render
    renderer = CORTORenderer()
    output_path = Path("test_output/test_render.png")
    success = renderer.render_for_image(img_info, params, output_path)
    
    print(f"Render {'successful' if success else 'failed'}!")