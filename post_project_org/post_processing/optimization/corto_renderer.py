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

def _purge_orphan_data(max_iterations: int = 5) -> None:
    """Release Blender datablocks left behind after scene cleanup."""
    try:
        # Multiple passes may be needed because purging can reveal new orphaned
        # datablocks on each iteration (e.g., meshes -> materials -> images).
        for _ in range(max_iterations):
            # ``orphans_purge`` returns {'FINISHED'} when something was freed.
            if bpy.ops.outliner.orphans_purge.poll():
                result = bpy.ops.outliner.orphans_purge(
                    do_local_ids=True, do_linked_ids=True, do_recursive=True
                )
                if "FINISHED" not in result:
                    break
            else:
                break
        # The render result can also hang around and keep textures alive.
        render_result = bpy.data.images.get("Render Result")
        if render_result is not None:
            bpy.data.images.remove(render_result)
    except Exception:
        # Cleanup must never interrupt rendering; fall back silently.
        pass

# Add post_project to path
post_project_path = Path(__file__).parent.parent.parent
sys.path.append(str(post_project_path))

from phobos_scene import (
    force_nvidia_gpu,
    build_scene_and_materials,
    set_phobos_params,
    set_atmosphere_params,
    set_mars_params,
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

from mission_config import extract_pose_from_label


# ============================================================================
# SPICE DATA CACHE
# ============================================================================

class SPICEDataCache:
    """Cache SPICE/label data for multiple IMG files to avoid redundant queries."""
    
    def __init__(self):
        self.cache = {}
        self.sdp = None
        
        if HAS_SPICE:
            self.sdp = SpiceDataProcessor()
    
    def get_spice_data(self, utc_time: str) -> Dict:
        """Get SPICE data with caching (HRSC path)."""
        if utc_time not in self.cache:
            if self.sdp is None:
                raise RuntimeError("SPICE not available!")
            
            self.cache[utc_time] = get_spice_data_for_time(self.sdp, utc_time)
        
        return self.cache[utc_time]
    
    def get_label_data(self, pds_path: str, mission_cfg) -> Dict:
        """Get pose data from PDS label with caching (OSIRIS/generic path)."""
        cache_key = str(pds_path)
        if cache_key not in self.cache:
            self.cache[cache_key] = extract_pose_from_label(
                Path(pds_path), mission_cfg
            )
        return self.cache[cache_key]


# ============================================================================
# CORTO RENDERER CLASS
# ============================================================================

# ---- Modül-yerel tekil renderer (tüm parçacıklar boyunca reuse) ----
_GLOBAL_RENDERER = None

def get_renderer(persistent: bool = True, batch_size: Optional[int] = None, **kwargs):
    global _GLOBAL_RENDERER
    if _GLOBAL_RENDERER is None:
        _GLOBAL_RENDERER = CORTORenderer(
            persistent=persistent, batch_size=batch_size, **kwargs
        )
    else:
        # batch_size değişmişse güncelle
        _GLOBAL_RENDERER.batch_size = batch_size
        _GLOBAL_RENDERER.persistent = persistent
    return _GLOBAL_RENDERER


class CORTORenderer:
    """
    Wrapper for CORTO rendering with photometric parameters.
    """
    
    def __init__(self,
                 scenario_name: str = "S07_Mars_Phobos_Deimos",
                 body_files: List[str] = None,
                 temp_dir: Optional[Path] = None,
                 cleanup: bool = True,
                 persistent: bool = True,
                 batch_size: Optional[int] = None,
                 use_displacement: bool = True,
                 use_atmosphere: bool = True,
                 atm_color_mode: str = 'single',
                 dem_path: Optional[str] = None):
        """
        Initialize CORTO renderer.
        
        Args:
            scenario_name: CORTO scenario name
            body_files: List of body OBJ files [phobos.obj, mars.obj, deimos.obj]
            temp_dir: Temporary directory for intermediate files
            cleanup: Whether to clean up temp files after rendering
            persistent: Keep scene between renders
            batch_size: Rebuild scene after this many renders
            use_displacement: Use MOLA DEM displacement (vs bump mapping)
            use_atmosphere: Create volumetric atmosphere sphere
            atm_color_mode: 'single' or 'rgb' (atmosphere color parameterization)
            dem_path: Optional path to Mars DEM TIFF for displacement
        """
        self.scenario_name = scenario_name
        self.body_files = body_files or [
            "g_phobos_144m_spc_0000n00000_v002.obj",
            "Mars_65k_km.obj",
            "g_deimos_162m_spc_0000n00000_v001.obj",
        ]
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.mkdtemp(prefix="corto_opt_"))
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.cleanup = cleanup
        self.persistent = persistent
        self.batch_size = batch_size
        self.use_displacement = use_displacement
        self.use_atmosphere = use_atmosphere
        self.atm_color_mode = atm_color_mode
        self.dem_path = dem_path

        # Kalıcı sahne önbelleği
        self._built = False
        self._renders_since_build = 0
        self._cache = {
            'ENV': None, 'cam': None, 'sun': None,
            'bodies': None, 'mat_nodes': None, 'tree': None
        }

        # SPICE cache & camera
        self.spice_cache = SPICEDataCache()
        self.mission_config = None  # Set per-image from img_info
        self.camera_config = get_camera_config()  # Default: HRSC from SPICE
        self.sun_blender_scaler = 3.90232e-1

        # PERSIST ve BATCH kontrolü
        self.persistent = persistent
        self.batch_size = batch_size
        self._prepared = False
        self._batch_count = 0
        self._ENV = None; self._cam = None; self._sun = None
        self._mat_nodes = None; self._State = None

        print(f"CORTO Renderer initialized")
        print(f"  Scenario: {scenario_name}")
        print(f"  Temp dir: {self.temp_dir}")
        print(f"  Cleanup: {cleanup}")
        print(f"  Persistent: {self.persistent}, Batch size: {self.batch_size}")
        print(f"  Displacement: {self.use_displacement}, Atmosphere: {self.use_atmosphere}")
        print(f"  Atm color mode: {self.atm_color_mode}")

    def _build_or_reuse_scene(self, spice_data: Dict):
        """Sahneyi ilk kez kur veya batch sınırı geldiyse yeniden kur; aksi halde sadece güncelle."""
        need_rebuild = (not self._prepared) or (self.batch_size and self._batch_count >= self.batch_size) or (not self.persistent)

        # Güneş enerjisi (base)
        solar_distance_km = float(spice_data['distances']['sun_to_phobos'])
        base_energy = calculate_sun_strength(solar_distance_km, q_eff=1.0, sun_blender_scaler=self.sun_blender_scaler)

        if need_rebuild:
            # ============ BLENDER DATA CLEANUP (YENİ) ============
            if self._prepared:  # İlk seferinde cleanup yapma
                print(f"  🧹 Cleaning Blender data before rebuild (batch #{self._batch_count})...")
                try:
                    # 1. Materials cleanup
                    for mat in list(bpy.data.materials):
                        if mat.users == 0:
                            bpy.data.materials.remove(mat)
                    
                    # 2. Textures/Images cleanup
                    for img in list(bpy.data.images):
                        if img.users == 0:
                            bpy.data.images.remove(img)
                    
                    # 3. Meshes cleanup (eğer body'ler yeniden import ediliyorsa)
                    for mesh in list(bpy.data.meshes):
                        if mesh.users == 0:
                            bpy.data.meshes.remove(mesh)
                    
                    # 4. Node trees cleanup
                    for node_tree in list(bpy.data.node_groups):
                        if node_tree.users == 0:
                            bpy.data.node_groups.remove(node_tree)
                    
                    # 5. Force garbage collection
                    import gc
                    gc.collect()
                    
                    print(f"  ✅ Cleanup complete")
                    
                except Exception as e:
                    print(f"  ⚠️ Cleanup warning: {e}")
            # ===================================================
            
            # Sahneyi temizle ve GPU'yu seç
            corto.Utils.clean_scene()
            _purge_orphan_data()
            force_nvidia_gpu()

            # Dinamik JSON'ları tek isimle overwrite et
            scene_filename, geom_filename = write_dynamic_scene_files(
                self.temp_dir, spice_data, self.camera_config, base_energy, idx=0
            )

            # State ve sahne kurulumu
            self._State = corto.State(scene=scene_filename, geometry=geom_filename, body=self.body_files, scenario=self.scenario_name)
            add_asset_paths(self._State, dem_path=self.dem_path, body_names=self.body_files)
            self._ENV, self._cam, self._sun, _bodies, self._mat_nodes, _tree = build_scene_and_materials(
                self._State, self.body_files,
                use_displacement=self.use_displacement,
                use_atmosphere=self.use_atmosphere,
                atm_color_mode=self.atm_color_mode
            )

            self._prepared = True
            self._batch_count = 0
        else:
            # Sahneyi koru, sadece JSON'ları güncelle (transformlar PositionAll ile yenilenecek)
            write_dynamic_scene_files(self.temp_dir, spice_data, self.camera_config, base_energy, idx=0)

        # Transformları güncelle
        self._ENV.PositionAll(self._State, index=0)

    
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
            params: Photometric parameter array [10 values]
            output_path: Where to save rendered PNG
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get mission config from img_info (if available)
            mission_cfg = img_info.get('mission_config', None)
            
            if mission_cfg is not None and not mission_cfg.use_spice:
                # ── Label path (OSIRIS, generic) ──
                spice_data = self.spice_cache.get_label_data(
                    img_info['pds_path'], mission_cfg
                )
                # Update camera config from mission
                if mission_cfg.camera_config is not None:
                    self.camera_config = mission_cfg.camera_config
            else:
                # ── SPICE path (HRSC, original behavior) ──
                spice_data = self.spice_cache.get_spice_data(img_info['utc_time'])
            
            self._build_or_reuse_scene(spice_data)

            # Parametreleri uygula (yalnızca node değerleri güncellenir)
            params_dict = self._params_to_dict(params)
            set_phobos_params(self._mat_nodes, params_dict)
            # Apply Mars hybrid shader parameters
            set_mars_params(self._mat_nodes, params_dict)
            # Apply atmosphere parameters
            set_atmosphere_params(self._mat_nodes, params_dict)

            # q_eff'e göre güneş enerjisi
            # Use sun_to_target for label path, sun_to_phobos for SPICE path
            dist_key = getattr(mission_cfg, 'solar_distance_key', 'sun_to_phobos') if mission_cfg else 'sun_to_phobos'
            solar_distance_km = float(
                spice_data['distances'].get(dist_key,
                    spice_data['distances'].get('sun_to_phobos', 0))
            )
            q_eff = float(params_dict['q_eff'])
            sun_energy = calculate_sun_strength(solar_distance_km, q_eff=q_eff, sun_blender_scaler=self.sun_blender_scaler)
            self._sun.set_energy(sun_energy)

            # Kamera ve render ayarları
            corto.Camera.select_camera(self._cam.name)
            # Hedef dosyayı mutlaklaştır ve klasörü garanti et
            out_base = Path(output_path).resolve()
            out_base.parent.mkdir(parents=True, exist_ok=True)
            # Blender, yolun dizin + isim (uzantısız) şeklini ister
            bpy.context.scene.render.filepath = out_base.with_suffix('').as_posix()
            bpy.context.scene.render.use_file_extension = True
            bpy.context.scene.render.image_settings.file_format = 'PNG'
            bpy.context.scene.render.image_settings.color_depth = '16'
            bpy.context.scene.render.image_settings.compression = 0
            bpy.context.scene.view_settings.view_transform = 'Raw'

            bpy.ops.render.render(write_still=True)
            self._batch_count += 1
            # ============ BLENDER RAM CLEANUP  ============
            if self._batch_count % 100 == 0:  # Her 100 render'da bir
                try:
                    for img in list(bpy.data.images):  # list() ile copy al
                        if img.users == 0 and not img.is_dirty:
                            bpy.data.images.remove(img)
                    print(f"  🧹 Blender cache cleaned at render #{self._batch_count}")
                except Exception as e:
                    print(f"  ⚠️ Cleanup warning: {e}")
            # ===================================================
            # Kaydedilen gerçekteki PNG (mutlak)
            rendered_file = out_base.with_suffix('.png')
            if not rendered_file.exists():
                # Bazı durumlarda Blender frame numarası ekleyebilir (0001)
                fallback = rendered_file.with_name(rendered_file.stem + "0001.png")
                if fallback.exists():
                    rendered_file = fallback
                else:
                    print(f"Error: Rendered file not found: {rendered_file}")
                    return False

            print(f"✓ Rendered: {rendered_file}")
            return True

        except Exception as e:
            print(f"Error rendering for {img_info['filename']}: {e}")
            import traceback; traceback.print_exc()
            return False
    
    def _params_to_dict(self, params: np.ndarray) -> Dict[str, float]:
        """Convert parameter array to dictionary using canonical names."""
        from optimization_helper import PARAMETER_NAMES
        return {name: float(val) for name, val in zip(PARAMETER_NAMES, params)}
    
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
    particle_id: int,
    persistent: bool = True,
    batch_size: Optional[int] = None,
    use_displacement: bool = True,
    use_atmosphere: bool = True,
    atm_color_mode: str = 'single',
    dem_path: Optional[str] = None
) -> List[Path]:
    """
    Render synthetic images for multiple real IMG files with given parameters.
    
    Args:
        img_info_list: List of IMG info dicts (filename, utc_time, etc.)
        params: Photometric parameters
        output_dir: Directory for output files
        particle_id: Particle/individual ID for naming
        persistent: Keep Blender scene between renders
        batch_size: Rebuild scene every N renders
        use_displacement: True displacement vs bump mapping
        use_atmosphere: Create volumetric atmosphere sphere
        atm_color_mode: 'single' or 'rgb'
        dem_path: Optional Mars DEM TIFF path
        
    Returns:
        List of paths to rendered PNG files
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    renderer = get_renderer(
        persistent=persistent, batch_size=batch_size,
        use_displacement=use_displacement,
        use_atmosphere=use_atmosphere,
        atm_color_mode=atm_color_mode,
        dem_path=dem_path
    )
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

def save_debug_scene(output_dir: Path, particle_id: int) -> Optional[Path]:
    """Persist the current Blender scene for debugging best-particle states."""

    output_dir = Path(output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(f"Warning: Could not create debug directory {output_dir}: {exc}")
        return None

    debug_path = output_dir / f"debug_particle_{particle_id:03d}.blend"

    try:
        bpy.ops.wm.save_as_mainfile(filepath=str(debug_path), check_existing=False)
    except Exception as exc:
        print(f"Warning: Could not save debug blend {debug_path}: {exc}")
        return None

    alias_path = output_dir / "debug.blend"
    if alias_path != debug_path:
        try:
            shutil.copyfile(debug_path, alias_path)
        except Exception as exc:
            print(f"Warning: Could not update convenience debug file {alias_path}: {exc}")

    return debug_path


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
    params = np.array([0.15, 0.30, 0.45, 0.60, 0.25, 1.40, 0.85, 0.10])
    
    # Render
    renderer = CORTORenderer()
    output_path = Path("test_output/test_render.png")
    success = renderer.render_for_image(img_info, params, output_path)
    
    print(f"Render {'successful' if success else 'failed'}!")