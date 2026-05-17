"""Blender scene manager for photometric optimization.

Follows the exact cortopy API as used in S07_Mars_Phobos_Deimos_optimized.py
and corto_renderer.py.  Key patterns:

  corto.State(scene=..., geometry=..., body=..., scenario=...)
  corto.Camera('name', State.properties_cam)
  corto.Sun('name', State.properties_sun)
  corto.Body(name, State.properties_body_N)
  corto.Rendering(State.properties_rendering)
  corto.Environment(cam, [bodies], sun, rendering_engine)
  ENV.PositionAll(State, index=idx)
  ENV.RenderOne(cam, State, index=idx, depth_flag=True)

Handles:
  - Scene setup (color management, GPU, cortopy objects)
  - Shader node creation matching the optimizer's exact node tree
  - Per-frame rendering with camera-specific FOV and sun energy
  - Dark current post-processing
"""
from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path

import numpy as np

from config import (
    ALL_X0,
    BODY_NAMES,
    CAM_DC,
    CAM_SUN_SCALER,
    COEFF_PATH,
    DC_IN_BLENDER,
    EXPOSURE_REF_MS,
    INPUT_DIR,
    PDS_IMAGES,
    RENDER_SAMPLES,
    SCENARIO,
    USE_ATMOSPHERE,
)

logger = logging.getLogger(__name__)

# Solar irradiance at 1 AU (W/m^2, Blender units)
W_1AU = 427.815
# 1 AU in km
AU_KM = 149597870.7
# Blender Sun scaler (matches corto_renderer.py)
SUN_BLENDER_SCALER = 3.90232e-1


class SceneManager:
    """Manages the Blender scene using the cortopy API."""

    def __init__(self):
        # Cortopy objects
        self._state = None
        self._env = None
        self._cam = None
        self._sun = None
        self._bodies = []
        self._rendering = None

        # Shader node references for parameter updates
        self.nodes: dict = {}

        # Full parameter vector (populated by update_all)
        self._x_full: list[float] = []

        # Per-frame render params from last render_all() call
        self.last_render_info: list[dict] = []

        # Camera FOV cache
        self._hrsc_fov: float = 0.0
        self._osiris_fov: float = math.radians(
            2.0 * math.degrees(math.atan(27.648 / (2.0 * 712.4)))
        )

    def build(self) -> None:
        """Initialize the full Blender scene following cortopy conventions."""
        import bpy
        import cortopy as corto

        # -- Color Management: Raw (linear, no gamma) & 16-bit PNG --
        try:
            bpy.context.scene.display_settings.display_device = "NONE"
        except TypeError:
            pass  # OCIO config may not have NONE
        bpy.context.scene.view_settings.view_transform = "Raw"
        
        # Explicitly enforce 16-bit PNG to ensure high dynamic range photometric accuracy
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.image_settings.color_depth = '16'
        bpy.context.scene.render.image_settings.compression = 0

        # -- Clean scene --
        corto.Utils.clean_scene()

        # -- GPU setup (from tutorial) --
        self._setup_gpu()

        # -- State --
        self._state = corto.State(
            scene="scene_optimized.json",
            geometry="geometry_optimized.json",
            body=BODY_NAMES,
            scenario=SCENARIO,
        )

        # -- Asset paths (from tutorial) --
        inp = self._state.path["input_path"]
        self._state.add_path(
            "albedo_path_1",
            os.path.join(inp, "body", "albedo", "Phobos_Viking_dimmed_ior133.tif"),
        )
        self._state.add_path(
            "uv_data_path_1",
            os.path.join(inp, "body", "uv data", "g_phobos_036m_spc_0000n00000_v002.json"),
        )
        self._state.add_path(
            "albedo_path_2",
            os.path.join(inp, "body", "albedo", "Mars_MOC_f32.tif"),
        )
        self._state.add_path(
            "uv_data_path_2",
            os.path.join(inp, "body", "uv data", "Mars_65k.json"),
        )
        # Mars DEM displacement map
        dem_path = os.path.abspath(
            os.path.join(inp, "body", "displacement", "Mars_MOLA_DEM_f32_4x_4x.tif")
        )
        if os.path.exists(dem_path):
            self._state.add_path("displacement_path_2", dem_path)
            logger.info("Mars DEM registered: %s", os.path.basename(dem_path))
        self._state.add_path(
            "albedo_path_3",
            os.path.join(inp, "body", "albedo", "Deimos grayscale.jpg"),
        )
        self._state.add_path(
            "uv_data_path_3",
            os.path.join(inp, "body", "uv data", "g_deimos_162m_spc_0000n00000_v001.json"),
        )

        # -- Scene objects (cortopy native) --
        self._cam = corto.Camera("WFOV_Camera", self._state.properties_cam)
        self._sun = corto.Sun("Sun", self._state.properties_sun)

        name_1 = os.path.splitext(BODY_NAMES[0])[0]
        name_2 = os.path.splitext(BODY_NAMES[1])[0]
        name_3 = os.path.splitext(BODY_NAMES[2])[0]

        body_1 = corto.Body(name_1, self._state.properties_body_1)
        body_2 = corto.Body(name_2, self._state.properties_body_2)
        body_3 = corto.Body(name_3, self._state.properties_body_3)
        self._bodies = [body_1, body_2, body_3]

        self._rendering = corto.Rendering(self._state.properties_rendering)
        self._env = corto.Environment(
            self._cam, self._bodies, self._sun, self._rendering
        )

        # -- Shaders (matching tutorial's exact node tree) --
        self._build_phobos_shader(body_1)
        self._build_mars_shader(body_2)
        self._build_deimos_shader(body_3)

        # -- Mars atmosphere (helpers/atmosphere.py — same as tutorial) --
        if USE_ATMOSPHERE:
            import sys as _sys
            _helpers = str(Path(__file__).resolve().parent.parent / "helpers")
            if _helpers not in _sys.path:
                _sys.path.insert(0, _helpers)
            try:
                from atmosphere import create_atmosphere
                create_atmosphere(
                    name="Mars_Atmosphere",
                    center_object=name_2,
                    body_radius=3389.5,
                    atmosphere_ratio=1.0177,
                    beta0=ALL_X0[14],
                    scale_height=ALL_X0[15],
                    anisotropy=ALL_X0[16],
                    color=(ALL_X0[17], ALL_X0[18], ALL_X0[19]),
                )
                logger.info("Mars atmosphere created")
            except Exception as exc:
                logger.warning("Atmosphere skipped: %s", exc)
        else:
            logger.info("Atmosphere disabled (USE_ATMOSPHERE=False)")

        # -- Compositing (ID mask) --
        tree = corto.Compositing.create_compositing()
        render_node = corto.Compositing.rendering_node(tree, (0, 0))
        corto.Compositing.create_img_denoise_branch(tree, render_node)
        corto.Compositing.create_depth_branch(tree, render_node)
        corto.Compositing.create_slopes_branch(tree, render_node, self._state)
        # CORTO's create_maskID_branch hard-codes index=1 (Phobos only).
        # mask_ID_1 + mask_ID_shadow_1 produced here.
        corto.Compositing.create_maskID_branch(tree, render_node, self._state)
        # Mars (pass_index=2) ID mask — CORTO doesn't emit it, add manually so
        # the scorer can derive the Mars region from true geometry instead of
        # the synthetic render (which drops Mars's shadowed/dark pixels).
        out_path = self._state.path["output_path"]
        mid2 = corto.Compositing.maskID_node(tree, (200, -700))
        mid2.index = 2
        out2 = corto.Compositing.file_output_node(tree, (600, -700))
        out2.format.color_mode = "BW"
        out2.format.color_depth = "8"
        out2.base_path = out_path
        out2.file_slots[0].path = os.path.join("mask_ID_2", "######")
        corto.Compositing.link_nodes(
            tree, render_node.outputs["IndexOB"], mid2.inputs["ID value"]
        )
        corto.Compositing.link_nodes(
            tree, mid2.outputs["Alpha"], out2.inputs[0]
        )

        # -- Render quality --
        cycles = bpy.context.scene.cycles
        cycles.samples = RENDER_SAMPLES
        cycles.preview_samples = 4
        cycles.diffuse_bounces = 4
        cycles.volume_bounces = 2
        cycles.max_bounces = 8

        # -- FOV cache --
        self._hrsc_fov = self._cam.CAM_Blender.data.angle

        # -- Albedo colorspace fix (Non-Color for Mars) --
        import bpy
        for akey in ["albedo_path_1", "albedo_path_2", "albedo_path_3"]:
            img_name = os.path.basename(self._state.path.get(akey, ""))
            if img_name and img_name in bpy.data.images:
                bpy.data.images[img_name].colorspace_settings.name = "Non-Color"

        # -- Body scale --
        for b in self._bodies:
            b.set_scale(np.array([1, 1, 1]))

        logger.info("Scene built. %d shader node refs.", len(self.nodes))

    # -- GPU Setup (from tutorial) --------------------------------------------

    @staticmethod
    def _setup_gpu():
        """Force NVIDIA GPU for Cycles (OPTIX > CUDA fallback)."""
        import bpy
        try:
            cyc = bpy.context.preferences.addons["cycles"].preferences
            bpy.context.scene.render.engine = "CYCLES"
            bpy.context.scene.cycles.device = "GPU"
            for backend in ["OPTIX", "CUDA"]:
                cyc.compute_device_type = backend
                try:
                    cyc.get_devices()
                except Exception:
                    pass
                gpus = [d for d in cyc.devices if d.type == backend]
                if gpus:
                    for d in cyc.devices:
                        d.use = d.type == backend
                    logger.info("GPU: %s (%d devices)", backend, len(gpus))
                    return
        except Exception as e:
            logger.warning("GPU setup failed: %s", e)

    # -- Shader Builders (matching tutorial exactly) --------------------------

    def _build_phobos_shader(self, body):
        """Create Phobos shader nodes matching tutorial's exact node tree."""
        import cortopy as corto

        mat = corto.Shading.create_new_material("Phobos_Optimized")
        corto.Shading.create_branch_albedo_mix(mat, self._state, 1)

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        tex = next(n for n in nodes if n.bl_idname == "ShaderNodeTexImage")
        dif = next(n for n in nodes if n.bl_idname == "ShaderNodeBsdfDiffuse")
        pri = next(n for n in nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
        mix = next(n for n in nodes if n.bl_idname == "ShaderNodeMixShader")

        # Remove TexImage→Principled link
        for link in list(links):
            if link.from_node == tex and link.to_node == pri:
                links.remove(link)

        # Create hybrid nodes
        rgbbw = corto.Shading.create_node("ShaderNodeRGBToBW", mat, (-300, 100))
        val = corto.Shading.create_node("ShaderNodeValue", mat, (-300, -100))
        mixf = corto.Shading.create_node("ShaderNodeMix", mat, (0, 0))
        mixf.data_type = "FLOAT"

        # Link chain
        corto.Shading.link_nodes(mat, tex.outputs["Color"], rgbbw.inputs["Color"])
        corto.Shading.link_nodes(mat, rgbbw.outputs["Val"], mixf.inputs["A"])
        corto.Shading.link_nodes(mat, val.outputs["Value"], mixf.inputs["B"])
        corto.Shading.link_nodes(mat, mixf.outputs["Result"], dif.inputs["Color"])
        corto.Shading.link_nodes(mat, mixf.outputs["Result"], pri.inputs["Base Color"])

        # Store node refs
        self.nodes["p_val"] = val
        self.nodes["p_mixf"] = mixf
        self.nodes["p_dif"] = dif
        self.nodes["p_pri"] = pri
        self.nodes["p_mix"] = mix

        corto.Shading.load_uv_data(body, self._state, 1)
        corto.Shading.assign_material_to_object(mat, body)

    def _build_mars_shader(self, body):
        """Create Mars shader nodes matching tutorial's exact node tree.

        Uses DEM displacement when displacement_path_2 is available.
        """
        import bpy
        import cortopy as corto

        mat = corto.Shading.create_new_material("Mars_Standard")

        # Use DEM displacement branch if available (matches tutorial)
        if "displacement_path_2" in self._state.path:
            mars_settings = {
                "displacement": {
                    "scale": 0.001,
                    "mid_level": 0.0,
                    "colorspace_name": "Non-Color",
                },
                "albedo": {"weight_diffuse": 0.95},
            }
            corto.Shading.create_branch_albedo_and_displacement_mix(
                mat, self._state, settings=mars_settings, id_body=2
            )
            mat.cycles.displacement_method = "BOTH"
            bpy.context.scene.cycles.feature_set = "EXPERIMENTAL"
            bpy.context.scene.cycles.dicing_rate = 1.0
            mars_obj = bpy.data.objects.get(os.path.splitext(BODY_NAMES[1])[0])
            if mars_obj:
                sub = mars_obj.modifiers.new("Mars_Adaptive", "SUBSURF")
                sub.subdivision_type = "SIMPLE"
                sub.levels = 0
                sub.render_levels = 0
            logger.info("Mars displacement enabled (DEM)")
        else:
            corto.Shading.create_branch_albedo_mix(mat, self._state, 2)
            logger.info("Mars displacement skipped (no DEM)")

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Find texture node (not displacement)
        tex = next(
            n for n in nodes
            if n.bl_idname == "ShaderNodeTexImage"
            and "displacement" not in (n.image.name if n.image else "").lower()
        )

        # Remove old CORTO shader nodes
        for old in [
            next((n for n in nodes if n.bl_idname == "ShaderNodeBsdfDiffuse"), None),
            next((n for n in nodes if n.bl_idname == "ShaderNodeMixShader"), None),
            next((n for n in nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None),
        ]:
            if old:
                nodes.remove(old)

        # BrightContrast
        bc = nodes.new("ShaderNodeBrightContrast")
        bc.location = (-450, 100)

        # RGB→BW
        rgbbw = nodes.new("ShaderNodeRGBToBW")
        rgbbw.location = (-300, 100)

        # Value (mars_base_gray)
        val = nodes.new("ShaderNodeValue")
        val.location = (-300, -100)

        # MixFloat (mars_tex_mix)
        mixf = nodes.new("ShaderNodeMix")
        mixf.data_type = "FLOAT"
        mixf.location = (0, 0)

        # Albedo multiplier (VectorMath MULTIPLY)
        mul = nodes.new("ShaderNodeVectorMath")
        mul.operation = "MULTIPLY"
        mul.location = (200, 100)

        # OrenNayar BSDF
        dif = corto.Shading.diffuse_BSDF(mat, location=(450, 200))

        # Principled BSDF
        pri = corto.Shading.principled_BSDF(mat, location=(450, -200))
        for sp in ("Specular IOR Level", "Specular"):
            if sp in pri.inputs:
                pri.inputs[sp].default_value = 0.50
                break
        for sl in ("Metallic", "Clearcoat", "Sheen", "Coat Weight"):
            if sl in pri.inputs:
                pri.inputs[sl].default_value = 0.0

        # MixShader
        mix = corto.Shading.mix_node(mat, location=(700, 0))

        # Material output
        mat_out = next(n for n in nodes if n.bl_idname == "ShaderNodeOutputMaterial")

        # Link chain
        links.new(tex.outputs["Color"], bc.inputs["Color"])
        links.new(bc.outputs["Color"], rgbbw.inputs["Color"])
        links.new(rgbbw.outputs["Val"], mixf.inputs["A"])
        links.new(val.outputs["Value"], mixf.inputs["B"])
        links.new(mixf.outputs["Result"], mul.inputs[0])
        links.new(mul.outputs["Vector"], dif.inputs["Color"])
        links.new(mul.outputs["Vector"], pri.inputs["Base Color"])
        links.new(dif.outputs["BSDF"], mix.inputs[1])
        links.new(pri.outputs["BSDF"], mix.inputs[2])

        # Remove old surface link and connect MixShader
        for lnk in list(links):
            if lnk.to_node == mat_out and getattr(lnk.to_socket, "name", "") == "Surface":
                links.remove(lnk)
        links.new(mix.outputs["Shader"], mat_out.inputs["Surface"])

        # Store node refs
        self.nodes["m_val"] = val
        self.nodes["m_mixf"] = mixf
        self.nodes["m_bc"] = bc
        self.nodes["m_mul"] = mul
        self.nodes["m_dif"] = dif
        self.nodes["m_pri"] = pri
        self.nodes["m_mix"] = mix

        corto.Shading.load_uv_data(body, self._state, 2)
        corto.Shading.assign_material_to_object(mat, body)

    def _build_deimos_shader(self, body):
        """Deimos: standard CORTO material (not optimized)."""
        import cortopy as corto
        mat = corto.Shading.create_new_material("Deimos_Standard")
        corto.Shading.create_branch_albedo_mix(mat, self._state, 3)
        corto.Shading.load_uv_data(body, self._state, 3)
        corto.Shading.assign_material_to_object(mat, body)

    # -- Parameter Update -----------------------------------------------------

    def update_all(self, x_full: list[float]) -> None:
        """Apply all 24 parameters to shader nodes.

        Indices 0-5: Phobos, 6-13: Mars, 14-19: Atmosphere
        Indices 20-23: Sun scaler + DC (applied per-frame in render_all)
        """
        self._x_full = x_full

        # Phobos (0-5): base_gray, tex_mix, oren_rough, princ_rough, shader_mix, ior
        if "p_val" in self.nodes:
            self.nodes["p_val"].outputs[0].default_value = x_full[0]
        if "p_mixf" in self.nodes:
            self.nodes["p_mixf"].inputs["Factor"].default_value = x_full[1]
        if "p_dif" in self.nodes:
            self.nodes["p_dif"].inputs["Roughness"].default_value = x_full[2]
        if "p_pri" in self.nodes:
            self.nodes["p_pri"].inputs["Roughness"].default_value = x_full[3]
            self.nodes["p_pri"].inputs["IOR"].default_value = x_full[5]
        if "p_mix" in self.nodes:
            self.nodes["p_mix"].inputs[0].default_value = x_full[4]

        # Mars (6-13): base_gray, tex_mix, oren_rough, princ_rough, shader_mix, ior, albedo_mul, contrast
        if "m_val" in self.nodes:
            self.nodes["m_val"].outputs[0].default_value = x_full[6]
        if "m_mixf" in self.nodes:
            self.nodes["m_mixf"].inputs["Factor"].default_value = x_full[7]
        if "m_dif" in self.nodes:
            self.nodes["m_dif"].inputs["Roughness"].default_value = x_full[8]
        if "m_pri" in self.nodes:
            self.nodes["m_pri"].inputs["Roughness"].default_value = x_full[9]
            self.nodes["m_pri"].inputs["IOR"].default_value = x_full[11]
        if "m_mix" in self.nodes:
            self.nodes["m_mix"].inputs[0].default_value = x_full[10]
        if "m_mul" in self.nodes:
            m = x_full[12]
            self.nodes["m_mul"].inputs[1].default_value = (m, m, m)
        if "m_bc" in self.nodes:
            self.nodes["m_bc"].inputs["Contrast"].default_value = x_full[13]

        # Atmosphere (14-19) — applied via helpers/atmosphere.py if loaded
        self._update_atmosphere(x_full[14:20])

    def _update_atmosphere(self, p: list[float]) -> None:
        """Update atmosphere parameters if atmosphere nodes exist."""
        import bpy
        atm_obj = bpy.data.objects.get("Mars_Atmosphere")
        if atm_obj is None or not atm_obj.data.materials:
            return
        mat = atm_obj.data.materials[0]
        for node in mat.node_tree.nodes:
            if node.name == "beta0":
                node.outputs[0].default_value = p[0]
            elif node.name == "H":
                node.outputs[0].default_value = p[1]
            elif node.bl_idname == "ShaderNodeVolumePrincipled":
                node.inputs["Anisotropy"].default_value = p[2]
                node.inputs["Color"].default_value = (p[3], p[4], p[5], 1.0)

    # -- Render Loop ----------------------------------------------------------

    def render_all(self) -> dict[str, list[str]]:
        """Render all frames with per-PDS dynamic geometry.

        For each PDS image:
          1. Extract geometry (SPICE or PDS label) for that observation
          2. Write a dynamic geometry JSON with correct positions
          3. Re-import geometry into cortopy State
          4. Set camera FOV (HRSC or OSIRIS)
          5. Set sun energy (inverse-square law)
          6. PositionAll + RenderOne (always index=0 since each JSON has 1 frame)
        """
        import sys
        import bpy
        import cortopy as corto

        # Import SPICE/label helpers from old pipeline
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                               "post_project_org" / "post_processing" / "optimization"))
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                               "post_project_org"))

        from mission_config import detect_mission, extract_pose_from_label
        from phobos_data import (
            get_spice_data_for_time,
            write_dynamic_scene_files,
            get_camera_config,
            calculate_sun_strength,
            HAS_SPICE,
            SpiceDataProcessor,
        )

        # SPICE processor (reused across frames)
        sdp = SpiceDataProcessor() if HAS_SPICE else None

        paths: dict[str, list[str]] = {
            "img": [], "mask": [], "shadow": [], "mask_mars": []
        }
        self.last_render_info = []

        for idx, pds_meta in enumerate(PDS_IMAGES):
            pds_path = Path(pds_meta["file"])
            camera = pds_meta["camera"]
            logger.info("  Rendering frame %d: %s (%s)",
                        idx, pds_path.name, camera)

            # --- Get observation geometry from SPICE or PDS label ---
            mission_cfg = detect_mission(pds_path)
            camera_cfg = get_camera_config(mission_cfg)

            if camera == "hrsc" and HAS_SPICE:
                # HRSC: get UTC time from label header, then SPICE for geometry
                utc_time = self._get_utc_from_label(pds_path)
                spice_data = get_spice_data_for_time(sdp, utc_time)
                solar_dist_km = float(
                    spice_data["distances"]["sun_to_phobos"]
                )
            elif hasattr(mission_cfg, 'use_spice') and not mission_cfg.use_spice:
                # OSIRIS: use PDS label for geometry (has position vectors)
                label_data = extract_pose_from_label(pds_path, mission_cfg)
                spice_data = label_data
                solar_dist_km = float(
                    label_data.get("distances", {}).get(
                        "sun_to_phobos",
                        label_data.get("distances", {}).get(
                            "sun_to_target", 227943200
                        ),
                    )
                )
            else:
                # Fallback: try SPICE with UTC from label
                utc_time = self._get_utc_from_label(pds_path)
                if HAS_SPICE and utc_time:
                    spice_data = get_spice_data_for_time(sdp, utc_time)
                    solar_dist_km = float(
                        spice_data["distances"]["sun_to_phobos"]
                    )
                else:
                    label_data = extract_pose_from_label(pds_path, mission_cfg)
                    spice_data = label_data
                    solar_dist_km = 227943200.0

            # --- Sun energy ---
            cam_scaler = self._x_full[CAM_SUN_SCALER[camera]]
            base_energy = calculate_sun_strength(
                solar_dist_km, q_eff=1.0,
                sun_blender_scaler=SUN_BLENDER_SCALER,
            )
            energy = cam_scaler * base_energy
            self._sun.set_energy(energy)

            # --- Per-image exposure time → Blender film_exposure ---
            # Uncalibrated images (HRSC): raw DN ∝ exposure time, so scale
            # film_exposure relative to EXPOSURE_REF_MS.
            # Calibrated images (OSIRIS/Rosetta): radiance is already divided
            # by exposure during calibration → film_exposure stays 1.0,
            # and sun_scaler alone handles the brightness match.
            is_calibrated = pds_meta.get("calibrated", False)
            if is_calibrated:
                film_exp = 1.0
            else:
                exposure_ms = float(pds_meta.get("exposure_ms", EXPOSURE_REF_MS))
                film_exp = exposure_ms / EXPOSURE_REF_MS
            bpy.context.scene.cycles.film_exposure = film_exp
            self.last_render_info.append({
                "frame": pds_path.name,
                "sun_energy": energy,
                "film_exp": film_exp,
            })
            logger.info("    Film exposure: %.4f (calibrated=%s)",
                        film_exp, is_calibrated)

            # --- Write per-image dynamic geometry JSON ---
            scene_fn, geom_fn = write_dynamic_scene_files(
                base_output_dir=Path("input") / SCENARIO,
                spice_data=spice_data,
                camera_cfg=camera_cfg,
                sun_energy=energy,
                idx=idx,
            )

            # --- Re-import geometry into cortopy State ---
            geom_path = os.path.join(
                "input", SCENARIO, "geometry", geom_fn
            )
            self._state.import_geometry(geom_path)

            # --- Camera FOV switch ---
            if camera == "osiris":
                self._cam.CAM_Blender.data.angle = self._osiris_fov
            else:
                self._cam.CAM_Blender.data.angle = self._hrsc_fov

            # --- Position all bodies and render (index=0, single-frame JSON) ---
            self._env.PositionAll(self._state, index=0)
            self._env.RenderOne(self._cam, self._state, index=0, depth_flag=True)

            # RenderOne(index=0) always saves as 000000.png.
            # Rename to frame_{idx} to prevent next render from overwriting.
            out = self._state.path["output_path"]
            import shutil
            for subdir in ("img", "mask_ID_1", "mask_ID_2", "mask_ID_shadow_1",
                            "slopes", "depth"):
                src = os.path.join(out, subdir, "000000.png")
                dst = os.path.join(out, subdir, f"frame_{idx:06d}.png")
                if os.path.exists(src):
                    if os.path.exists(dst):
                        os.remove(dst)
                    shutil.move(src, dst)

            img_path = os.path.join(out, "img", f"frame_{idx:06d}.png")

            # DC post-processing (if not done in Blender compositing)
            dc_val = self._x_full[CAM_DC[camera]]
            if not DC_IN_BLENDER and dc_val > 0:
                self._apply_dc_postprocess(img_path, dc_val)

            paths["img"].append(img_path)
            paths["mask"].append(
                os.path.join(out, "mask_ID_1", f"frame_{idx:06d}.png")
            )
            paths["mask_mars"].append(
                os.path.join(out, "mask_ID_2", f"frame_{idx:06d}.png")
            )
            paths["shadow"].append(
                os.path.join(out, "mask_ID_shadow_1", f"frame_{idx:06d}.png")
            )
            logger.info("    Rendered: %s (solar=%.0f km, energy=%.4f)",
                        img_path, solar_dist_km, energy)

        return paths

    def save_debug_blend(self, output_dir: str | Path, label: str = "debug") -> Path | None:
        """Save current Blender scene as .blend for inspection.

        Args:
            output_dir: Directory to save into.
            label: Filename label (e.g. 'best', 'gen_005').

        Returns:
            Path to saved .blend file, or None on failure.
        """
        import bpy

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        blend_path = out / f"{label}.blend"
        try:
            # Use a unique temp name to avoid Blender's "file saved with @"
            # lock that prevents overwriting old-format .blend files.
            import tempfile, shutil
            with tempfile.NamedTemporaryFile(suffix=".blend", delete=False) as tmp:
                tmp_path = tmp.name
            bpy.ops.wm.save_as_mainfile(filepath=tmp_path, check_existing=False)
            shutil.move(tmp_path, str(blend_path))
            logger.info("Debug blend saved: %s", blend_path)
            return blend_path
        except Exception as exc:
            logger.warning("Debug blend save failed: %s", exc)
            return None

    @staticmethod
    def _apply_dc_postprocess(img_path: str, dc_val: float) -> None:
        """Apply dark current threshold to rendered image."""
        from PIL import Image

        img = np.array(Image.open(img_path))
        threshold = dc_val * 65535
        img[img < threshold] = 0
        Image.fromarray(img).save(img_path)

    @staticmethod
    def _get_utc_from_label(pds_path: Path) -> str:
        """Extract UTC mid-time from PDS label header.

        Parses START_TIME and STOP_TIME, returns the midpoint as UTC string.
        Falls back to START_TIME if STOP_TIME is missing.
        """
        import re
        from datetime import datetime, timedelta

        with open(pds_path, "rb") as f:
            header = f.read(32000).decode("latin-1")

        pattern = re.compile(r"^\s*(START_TIME|STOP_TIME)\s*=\s*(.+)$", re.MULTILINE)
        times = {}
        for m in pattern.finditer(header):
            key = m.group(1).strip()
            val = m.group(2).strip().strip('"').strip("'").rstrip("Z")
            try:
                # Handle various PDS time formats
                for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
                            "%Y-%jT%H:%M:%S.%f", "%Y-%jT%H:%M:%S"):
                    try:
                        times[key] = datetime.strptime(val, fmt)
                        break
                    except ValueError:
                        continue
            except Exception:
                pass

        if "START_TIME" in times and "STOP_TIME" in times:
            mid = times["START_TIME"] + (times["STOP_TIME"] - times["START_TIME"]) / 2
        elif "START_TIME" in times:
            mid = times["START_TIME"]
        else:
            raise ValueError(f"No START_TIME found in {pds_path.name}")

        return mid.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-2] + "Z"
