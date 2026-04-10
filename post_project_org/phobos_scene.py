# -*- coding: utf-8 -*-
"""
phobos_scene.py
Phobos Photometric Template Generator - Scene Management Module

Sorumluluklar:
- GPU konfigürasyonu
- Blender sahne oluşturma
- Malzeme ve shader kurulumu
- Compositing branch'leri
- 16-bit PNG kaydetme
"""

import os
import sys
import numpy as np
import bpy
from PIL import Image

sys.path.append(os.getcwd())
import cortopy as corto


# ============================================================================
# MARS BILIMSEL PARAMETRELER (NASA/IAU/MOLA)
# ============================================================================
MARS_MEAN_RADIUS_KM = 3389.5        # IAU 2015 ortalama yarıçap
MARS_TOTAL_RELIEF_KM = 28.381       # MOLA: Olympus Mons (+21.229) + Hellas (-7.152)
# Displacement scale: metre → km (Mars_65k_km.obj: 1 BU = 1 km)
MARS_DISP_SCALE = 0.001
# Displacement mid_level: OBJ radius (equatorial) vs areoid mean radius offset
# OBJ vertex radius = 3396.190 km, MOLA areoid mean = 3389.500 km
# Fark = 6.690 km = 6690 m → mid_level bunu telafi eder:
#   DEM=0 (areoid): offset = (0 - 6690) × 0.001 = -6.69 km → vertex: 3396.19 - 6.69 = 3389.5 km ✓
MARS_OBJ_RADIUS_KM = 3396.190       # OBJ dosyasından ölçülen yarıçap (km)
MARS_DISP_MID_LEVEL = (MARS_OBJ_RADIUS_KM - MARS_MEAN_RADIUS_KM) * 1000.0  # = 6690.0 metre
# Bump strength (legacy, used when displacement is disabled)
MARS_SCENE_SCALE = 1000.0           # Blender sahne scale faktörü
MARS_BUMP_STRENGTH = MARS_TOTAL_RELIEF_KM / (MARS_OBJ_RADIUS_KM * MARS_SCENE_SCALE / 1000.0)
# ≈ 0.0083568


# ============================================================================
# RENDER QUALITY DEFAULTS
# Buradaki değerleri doğrudan değiştirerek render kalitesini yönetin.
# CORTO scene_mmx.json üzerinden set ettiği değerleri burada override ediyoruz.
# ============================================================================

# Cycles render samples
# CORTO default: 4  |  Blender default: 128
# Önerilen: Optimizasyon → 4-16, Final render → 64-256
RENDER_SAMPLES = 64
RENDER_PREVIEW_SAMPLES = 4

# Light bounces
# CORTO default: 0  |  Blender default: 12
# Önerilen: 1-4 (Mars → Phobos yansıması için en az 1 gerekli)
DIFFUSE_BOUNCES = 4

# Volume bounces (atmosfer scattering için)
# CORTO default: YOK  |  Blender default: 0
# Önerilen: 0 = single-scatter, 1-2 = multi-scatter (daha gerçekçi)
VOLUME_BOUNCES = 2

# Maximum total bounces
# CORTO default: YOK  |  Blender default: 12
# Önerilen: 4-12 (uzay sahnesi için 4 yeterli)
MAX_BOUNCES = 4

# Glossy bounces (specular yansımalar)
# CORTO default: YOK  |  Blender default: 4
# Önerilen: 1-4
GLOSSY_BOUNCES = 2

# Transmission bounces (saydam malzeme geçişleri)
# CORTO default: YOK  |  Blender default: 12
# Önerilen: 0-2 (sahnemizde saydam malzeme yok)
TRANSMISSION_BOUNCES = 0

# Transparent bounces (alpha transparency)
# CORTO default: YOK  |  Blender default: 8
# Önerilen: 4-8
TRANSPARENT_MAX_BOUNCES = 4


# ============================================================================
# GPU FORCING
# ============================================================================
def force_nvidia_gpu():
    """NVIDIA GPU'yu Cycles için zorunlu kullan"""
    try:
        cyc = bpy.context.preferences.addons['cycles'].preferences
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.device = 'GPU'
        
        prefer_order = ['OPTIX', 'CUDA']
        nvidia_keywords = ['nvidia', 'geforce', 'rtx', 'quadro', 'tesla']
        enabled_any = False
        chosen_backend = None

        def _refresh_devices():
            try:
                cyc.get_devices()
            except Exception:
                try:
                    cyc.refresh_devices()
                except Exception:
                    pass

        for backend in prefer_order:
            cyc.compute_device_type = backend
            _refresh_devices()
            devices = list(cyc.devices)
            gpu_candidates = []
            
            for d in devices:
                if d.type == backend:
                    name_lc = d.name.lower()
                    is_nvidia = any(k in name_lc for k in nvidia_keywords)
                    d.use = bool(is_nvidia)
                    if d.use:
                        gpu_candidates.append(d)
                else:
                    d.use = False
            
            if gpu_candidates:
                enabled_any = True
                chosen_backend = backend
                break

        if not enabled_any:
            for backend in prefer_order:
                cyc.compute_device_type = backend
                _refresh_devices()
                devices = list(cyc.devices)
                first_gpu = next((d for d in devices if d.type == backend), None)
                if first_gpu:
                    for d in devices:
                        d.use = (d is first_gpu)
                    enabled_any = True
                    chosen_backend = backend
                    break

        print(f"GPU backend: {chosen_backend}" if enabled_any else "No GPU enabled")
    except Exception as e:
        print(f"GPU selection failed: {e}")


# ============================================================================
# 16-BIT PNG HANDLING
# ============================================================================
def save_raw_16bit_png(image_data, output_path):
    """16-bit PNG olarak yüksek hassasiyetle kaydet"""
    try:
        if image_data.dtype == np.uint16:
            raw_uint16 = image_data
        elif image_data.dtype == np.uint8:
            raw_uint16 = (image_data.astype(np.uint16) << 8)
        else:
            if image_data.max() <= 1.0:
                raw_uint16 = (image_data * 65535).astype(np.uint16)
            else:
                raw_uint16 = np.clip(image_data, 0, 65535).astype(np.uint16)
        
        Image.fromarray(raw_uint16, mode='I;16').save(str(output_path))
        return True
    except Exception as e:
        print(f" 16-bit PNG save failed: {e}")
        return False


# ============================================================================
# PHOBOS ADVANCED SHADING
# ============================================================================
def build_scene_and_materials(State, body_names,
                              use_displacement=True,
                              use_atmosphere=True,
                              atm_color_mode='single'):
    """
    Sahne ve gelişmiş Phobos malzemesini oluşturur
    
    Args:
        State: CORTO State object
        body_names: Body dosya adları listesi [phobos.obj, mars.obj, deimos.obj]
        use_displacement: True = MOLA DEM displacement, False = bump mapping
        use_atmosphere: True = create volumetric atmosphere sphere
        atm_color_mode: 'single' or 'rgb' (how color is parameterized)
    
    Returns:
        tuple: (ENV, cam, sun, bodies, mat_nodes, tree)
    """
    # ============ MATERIAL CLEANUP (YENİ) ============
    # Eski material varsa önce sil
    existing_materials = ['Phobos_Advanced_Hybrid', 'Mars_Standard', 'Deimos_Standard']
    for mat_name in existing_materials:
        if mat_name in bpy.data.materials:
            old_mat = bpy.data.materials[mat_name]
            bpy.data.materials.remove(old_mat)
    # =================================================
    # Setup bodies
    cam = corto.Camera('WFOV_Camera', State.properties_cam)
    sun = corto.Sun('Sun', State.properties_sun)
    
    name_1, _ = os.path.splitext(body_names[0])
    name_2, _ = os.path.splitext(body_names[1])
    name_3, _ = os.path.splitext(body_names[2])
    
    body_1 = corto.Body(name_1, State.properties_body_1)
    body_2 = corto.Body(name_2, State.properties_body_2)
    body_3 = corto.Body(name_3, State.properties_body_3)
    
    rendering_engine = corto.Rendering(State.properties_rendering)
    ENV = corto.Environment(cam, [body_1, body_2, body_3], sun, rendering_engine)

    try:
        bpy.context.scene.cycles.use_persistent_data = True
    except Exception:
        pass

    # ---- Cycles Experimental + Adaptive Subdivision (Mars true displacement) ----
    if use_displacement:
        try:
            bpy.context.scene.cycles.feature_set = 'EXPERIMENTAL'
            bpy.context.scene.cycles.dicing_rate = 1.0  # pixel/micropolygon
            print("  ✅ Cycles Experimental + adaptive subdivision enabled (dicing_rate=1.0)")
        except Exception as e:
            print(f"  ⚠️ Adaptive subdivision setup failed: {e}")

    # Advanced Phobos material
    phobos_material = corto.Shading.create_new_material('Phobos_Advanced_Hybrid')

    # Shader models
    oren_node = corto.Shading.diffuse_BSDF(phobos_material, location=(200, 200))
    principled_node = corto.Shading.principled_BSDF(phobos_material, location=(200, -200))

    # Grayscale sources
    albedo_texture_node = corto.Shading.texture_node(
        phobos_material, State.path["albedo_path_1"], location=(-600, 100)
    )
    try:
        bpy.data.images[os.path.basename(State.path["albedo_path_1"])].colorspace_settings.name = 'Non-Color'
    except Exception:
        pass

    uv_map_node = corto.Shading.uv_map(phobos_material, location=(-800, 100))
    rgb_to_bw_node = corto.Shading.create_node("ShaderNodeRGBToBW", phobos_material, location=(-300, 100))
    value_node = corto.Shading.create_node("ShaderNodeValue", phobos_material, location=(-300, -100))
    mix_value_node = corto.Shading.create_node("ShaderNodeMix", phobos_material, location=(0, 0))
    mix_value_node.data_type = 'FLOAT'

    # Final mix
    mix_shader_node = corto.Shading.mix_node(phobos_material, location=(500, 0))
    material_output_node = corto.Shading.material_output(phobos_material, location=(800, 0))

    # Default parameters
    value_node.outputs["Value"].default_value = 0.15
    mix_value_node.inputs["Factor"].default_value = 0.8
    oren_node.inputs["Roughness"].default_value = 0.9
    principled_node.inputs["Roughness"].default_value = 0.85

    # IOR default
    if 'IOR' in principled_node.inputs:
        principled_node.inputs['IOR'].default_value = 1.50

    # Specular
    for spec_name in ("Specular IOR Level", "Specular"):
        if spec_name in principled_node.inputs:
            principled_node.inputs[spec_name].default_value = 0.50
            break

    # Safety: dielectric settings
    for slot in ('Metallic', 'Clearcoat', 'Sheen', 'Coat Weight'):
        if slot in principled_node.inputs:
            principled_node.inputs[slot].default_value = 0.0

    mix_shader_node.inputs[0].default_value = 0.1

    # Link nodes
    corto.Shading.link_nodes(phobos_material, uv_map_node.outputs["UV"], albedo_texture_node.inputs["Vector"])
    corto.Shading.link_nodes(phobos_material, albedo_texture_node.outputs["Color"], rgb_to_bw_node.inputs["Color"])
    corto.Shading.link_nodes(phobos_material, rgb_to_bw_node.outputs["Val"], mix_value_node.inputs["A"])
    corto.Shading.link_nodes(phobos_material, value_node.outputs["Value"], mix_value_node.inputs["B"])
    
    final_gray = mix_value_node.outputs["Result"]
    corto.Shading.link_nodes(phobos_material, final_gray, oren_node.inputs["Color"])
    corto.Shading.link_nodes(phobos_material, final_gray, principled_node.inputs["Base Color"])
    corto.Shading.link_nodes(phobos_material, oren_node.outputs["BSDF"], mix_shader_node.inputs[1])
    corto.Shading.link_nodes(phobos_material, principled_node.outputs["BSDF"], mix_shader_node.inputs[2])
    corto.Shading.link_nodes(phobos_material, mix_shader_node.outputs["Shader"], material_output_node.inputs["Surface"])

    # Assign to Phobos
    corto.Shading.load_uv_data(body_1, State, 1)
    corto.Shading.assign_material_to_object(phobos_material, body_1)

    # ============ MARS HYBRID MATERIAL (Phobos tarzı) ============
    material_2 = corto.Shading.create_new_material('Mars_Standard')
    material_3 = corto.Shading.create_new_material('Deimos_Standard')

    # --- Mars displacement (CORTO branch — fixed, optimize edilmiyor) ---
    if use_displacement and 'mars_dem_path' in State.path:
        displacement_mars = {
            'scale': MARS_DISP_SCALE,
            'mid_level': MARS_DISP_MID_LEVEL,
            'colorspace_name': 'Non-Color'
        }
        albedo_mars = {'weight_diffuse': 0.95}
        settings_mars = {'displacement': displacement_mars, 'albedo': albedo_mars}
        corto.Shading.create_branch_albedo_and_displacement_mix(
            material_2, State, settings=settings_mars, id_body=2
        )

        # --- True displacement: BUMP → BOTH (gerçek vertex displacement) ---
        material_2.cycles.displacement_method = 'BOTH'

        # --- Subdivision Surface modifier (MONET pattern — adaptive tessellation) ---
        mars_obj = bpy.data.objects.get(name_2)
        if mars_obj:
            # Remove existing subsurf if re-running
            for mod in mars_obj.modifiers:
                if mod.type == 'SUBSURF' and mod.name == 'Mars_Adaptive':
                    mars_obj.modifiers.remove(mod)
                    break
            subsurf = mars_obj.modifiers.new('Mars_Adaptive', 'SUBSURF')
            subsurf.subdivision_type = 'SIMPLE'
            subsurf.levels = 0       # viewport
            subsurf.render_levels = 0  # adaptive subdivision controls this
            print(f"  ✅ Mars true displacement + adaptive subdivision enabled")
        else:
            print(f"  ⚠️ Mars object '{name_2}' not found for SUBSURF modifier")

        print(f"  ✅ Mars displacement mapping enabled (scale={MARS_DISP_SCALE})")
    else:
        corto.Shading.create_branch_albedo_mix(material_2, State, 2)
        print("  ℹ️ Mars using albedo + bump mapping (no displacement)")

    corto.Shading.create_branch_albedo_mix(material_3, State, 3)

    # --- Mars hybrid shader node tree ---
    mars_nodes = material_2.node_tree.nodes
    mars_links = material_2.node_tree.links

    # Find existing nodes from CORTO branch
    mars_texture = next((n for n in mars_nodes if getattr(n, "bl_idname", "") == 'ShaderNodeTexImage'), None)
    mars_uv = next((n for n in mars_nodes if getattr(n, "bl_idname", "") == 'ShaderNodeUVMap'), None)
    mars_output = next((n for n in mars_nodes if getattr(n, "bl_idname", "") == 'ShaderNodeOutputMaterial'), None)

    # Remove old CORTO-generated shader links (Diffuse, MixShader, Principled)
    old_diffuse = next((n for n in mars_nodes if getattr(n, "bl_idname", "") == 'ShaderNodeBsdfDiffuse'), None)
    old_mix_shader = next((n for n in mars_nodes if getattr(n, "bl_idname", "") == 'ShaderNodeMixShader'), None)
    old_principled = next((n for n in mars_nodes if getattr(n, "bl_idname", "") == 'ShaderNodeBsdfPrincipled'), None)
    for old_node in [old_diffuse, old_mix_shader, old_principled]:
        if old_node:
            mars_nodes.remove(old_node)

    # RGB → BW
    mars_rgb_to_bw = mars_nodes.new('ShaderNodeRGBToBW')
    mars_rgb_to_bw.location = (-300, 100)

    # Value node (mars_base_gray)
    mars_value_node = mars_nodes.new('ShaderNodeValue')
    mars_value_node.location = (-300, -100)
    mars_value_node.outputs['Value'].default_value = 0.15

    # MixFloat (mars_tex_mix): texture vs base_gray karışımı
    mars_mix_value_node = mars_nodes.new('ShaderNodeMix')
    mars_mix_value_node.data_type = 'FLOAT'
    mars_mix_value_node.location = (0, 0)
    mars_mix_value_node.inputs['Factor'].default_value = 0.8

    # Albedo multiplier (VectorMath MULTIPLY)
    mars_multiplier = mars_nodes.new('ShaderNodeVectorMath')
    mars_multiplier.operation = 'MULTIPLY'
    mars_multiplier.location = (200, 100)
    mars_multiplier.inputs[1].default_value = (1.0, 1.0, 1.0)

    # OrenNayar BSDF
    mars_oren_node = corto.Shading.diffuse_BSDF(material_2, location=(450, 200))
    mars_oren_node.inputs['Roughness'].default_value = 0.9

    # Principled BSDF
    mars_principled_node = corto.Shading.principled_BSDF(material_2, location=(450, -200))
    mars_principled_node.inputs['Roughness'].default_value = 0.85
    if 'IOR' in mars_principled_node.inputs:
        mars_principled_node.inputs['IOR'].default_value = 1.50
    for spec_name in ('Specular IOR Level', 'Specular'):
        if spec_name in mars_principled_node.inputs:
            mars_principled_node.inputs[spec_name].default_value = 0.50
            break
    # Dielectric settings
    for slot in ('Metallic', 'Clearcoat', 'Sheen', 'Coat Weight'):
        if slot in mars_principled_node.inputs:
            mars_principled_node.inputs[slot].default_value = 0.0

    # MixShader (mars_shader_mix)
    mars_mix_shader_node = corto.Shading.mix_node(material_2, location=(700, 0))
    mars_mix_shader_node.inputs[0].default_value = 0.1

    # --- Link nodes ---
    # Texture → RGB→BW
    if mars_texture:
        mars_links.new(mars_texture.outputs['Color'], mars_rgb_to_bw.inputs['Color'])
    # RGB→BW → MixFloat.A
    mars_links.new(mars_rgb_to_bw.outputs['Val'], mars_mix_value_node.inputs['A'])
    # Value → MixFloat.B
    mars_links.new(mars_value_node.outputs['Value'], mars_mix_value_node.inputs['B'])
    # MixFloat → Albedo multiplier
    mars_links.new(mars_mix_value_node.outputs['Result'], mars_multiplier.inputs[0])
    # Albedo multiplier → OrenNayar Color + Principled Base Color
    mars_links.new(mars_multiplier.outputs['Vector'], mars_oren_node.inputs['Color'])
    mars_links.new(mars_multiplier.outputs['Vector'], mars_principled_node.inputs['Base Color'])
    # OrenNayar → MixShader.1, Principled → MixShader.2
    mars_links.new(mars_oren_node.outputs['BSDF'], mars_mix_shader_node.inputs[1])
    mars_links.new(mars_principled_node.outputs['BSDF'], mars_mix_shader_node.inputs[2])
    # MixShader → Material Output
    if mars_output:
        # Remove old surface link
        for link in list(mars_links):
            if link.to_node == mars_output and getattr(link.to_socket, 'name', '') == 'Surface':
                mars_links.remove(link)
        mars_links.new(mars_mix_shader_node.outputs['Shader'], mars_output.inputs['Surface'])

    print("  ✅ Mars hybrid shader created (OrenNayar + Principled BSDF)")

    # --- Mars bump mapping (fallback when no displacement) ---
    if not (use_displacement and 'mars_dem_path' in State.path):
        if 'mars_bump_path' in State.path:
            mars_bump_tex = mars_nodes.new('ShaderNodeTexImage')
            mars_bump_tex.location = (-500, -300)
            try:
                mars_bump_tex.image = bpy.data.images.load(State.path['mars_bump_path'])
                mars_bump_tex.image.colorspace_settings.name = 'Non-Color'
            except Exception as e:
                print(f"  ⚠️ Mars bump texture yüklenemedi: {e}")

            mars_bump_node = mars_nodes.new('ShaderNodeBump')
            mars_bump_node.location = (200, -300)
            mars_bump_node.inputs['Strength'].default_value = MARS_BUMP_STRENGTH
            mars_bump_node.inputs['Distance'].default_value = 1.0

            if mars_uv:
                mars_links.new(mars_uv.outputs['UV'], mars_bump_tex.inputs['Vector'])

            mars_links.new(mars_bump_tex.outputs['Color'], mars_bump_node.inputs['Height'])
            mars_links.new(mars_bump_node.outputs['Normal'], mars_oren_node.inputs['Normal'])
            mars_links.new(mars_bump_node.outputs['Normal'], mars_principled_node.inputs['Normal'])
            print(f"  ✅ Mars 6K bump mapping etkinleştirildi (strength={MARS_BUMP_STRENGTH:.10f})")

    corto.Shading.load_uv_data(body_2, State, 2)
    corto.Shading.assign_material_to_object(material_2, body_2)
    corto.Shading.load_uv_data(body_3, State, 3)
    corto.Shading.assign_material_to_object(material_3, body_3)

    # COMPOSITING
    tree = corto.Compositing.create_compositing()
    render_node = corto.Compositing.rendering_node(tree, (0, 0))
    corto.Compositing.create_img_denoise_branch(tree, render_node)
    corto.Compositing.create_depth_branch(tree, render_node)
    corto.Compositing.create_slopes_branch(tree, render_node, State)
    corto.Compositing.create_maskID_branch(tree, render_node, State)

    # Fix compositor paths for cross-platform compatibility
    try:
        import platform
        is_windows = platform.system() == 'Windows'
        
        for node in tree.nodes:
            if getattr(node, "bl_idname", "") == "CompositorNodeOutputFile":
                for slot in node.file_slots:
                    current_path = slot.path
                    if '\\' in current_path:
                        current_path = current_path.replace('\\', '/')
                    if '#' not in current_path:
                        if '/' in current_path:
                            current_path = current_path.rstrip('/') + '/######'
                        else:
                            current_path = '######'
                    slot.path = current_path
                    
        print(f" ✅ Compositor paths configured for {platform.system()}")
                    
    except Exception as e:
        print(f" ⚠️ Warning: compositor path fix failed: {e}")

    # Scales — Mars_65k_km.obj already in km, no need for 1e3
    body_1.set_scale(np.array([1, 1, 1]))
    body_2.set_scale(np.array([1, 1, 1]))
    body_3.set_scale(np.array([1, 1, 1]))

    # Pack refs
    mat_nodes = {
        'value_node': value_node,
        'mix_value_node': mix_value_node,
        'oren_node': oren_node,
        'principled_node': principled_node,
        'mix_shader_node': mix_shader_node,
        # Mars hybrid shader nodes
        'mars_value_node': mars_value_node,
        'mars_mix_value_node': mars_mix_value_node,
        'mars_oren_node': mars_oren_node,
        'mars_principled_node': mars_principled_node,
        'mars_mix_shader_node': mars_mix_shader_node,
        'mars_albedo_multiplier': mars_multiplier,
    }

    # ============ ATMOSPHERE ============
    if use_atmosphere:
        try:
            # Import atmosphere helper — CWD is project root (corto_test/)
            helpers_path = os.path.join(os.getcwd(), 'helpers')
            if helpers_path not in sys.path:
                sys.path.insert(0, helpers_path)
            from atmosphere import create_atmosphere

            # Create atmosphere (default params — will be updated per iteration)
            atm_obj = create_atmosphere(
                name='Mars_Atmosphere',
                center_object=name_2,
                body_radius=MARS_MEAN_RADIUS_KM,
                atmosphere_ratio=1.0177,
                beta0=3e-4,
                scale_height=11.0,
                anisotropy=0.6,
                color=(0.8, 0.5, 0.3),
            )

            # Store atmosphere node references for optimizer
            atm_mat = atm_obj.data.materials[0]
            atm_nodes = atm_mat.node_tree.nodes
            mat_nodes['atm_volume_node'] = atm_nodes.get('Principled Volume')
            mat_nodes['atm_beta0_node'] = atm_nodes.get('beta0')
            mat_nodes['atm_H_node'] = atm_nodes.get('H')
            mat_nodes['atm_color_mode'] = atm_color_mode

            print("  ✅ Mars atmosphere created (beta0=3e-4, H=11km)")
        except Exception as e:
            print(f"  ⚠️ Atmosphere creation failed: {e}")
            import traceback; traceback.print_exc()

    # ============ RENDER QUALITY OVERRIDES ============
    # CORTO'nun JSON'dan set ettiği değerleri burada override ediyoruz.
    # Değerleri değiştirmek için dosya başındaki RENDER_QUALITY_DEFAULTS bölümüne bakın.
    cycles = bpy.context.scene.cycles
    cycles.samples = RENDER_SAMPLES
    cycles.preview_samples = RENDER_PREVIEW_SAMPLES
    cycles.diffuse_bounces = DIFFUSE_BOUNCES
    cycles.volume_bounces = VOLUME_BOUNCES
    cycles.max_bounces = MAX_BOUNCES
    cycles.glossy_bounces = GLOSSY_BOUNCES
    cycles.transmission_bounces = TRANSMISSION_BOUNCES
    cycles.transparent_max_bounces = TRANSPARENT_MAX_BOUNCES

    print(f"  ✅ Render quality: samples={RENDER_SAMPLES}, "
          f"diffuse={DIFFUSE_BOUNCES}, volume={VOLUME_BOUNCES}, "
          f"max={MAX_BOUNCES}, glossy={GLOSSY_BOUNCES}")

    return ENV, cam, sun, (body_1, body_2, body_3), mat_nodes, tree


# ============================================================================
# PARAMETER APPLICATION
# ============================================================================
def set_phobos_params(mat_nodes, params):
    """
    Phobos malzeme node'larına parametreleri uygular
    IOR taranır, Specular IOR Level 0.5'te sabit
    """
    if 'base_gray' in params:
        mat_nodes['value_node'].outputs['Value'].default_value = float(params['base_gray'])
    
    if 'tex_mix' in params:
        mat_nodes['mix_value_node'].inputs['Factor'].default_value = float(params['tex_mix'])
    
    if 'oren_rough' in params:
        mat_nodes['oren_node'].inputs['Roughness'].default_value = float(params['oren_rough'])
    
    if 'princ_rough' in params:
        mat_nodes['principled_node'].inputs['Roughness'].default_value = float(params['princ_rough'])
    
    # IOR taraması
    if 'ior' in params and 'IOR' in mat_nodes['principled_node'].inputs:
        mat_nodes['principled_node'].inputs['IOR'].default_value = float(params['ior'])
    
    # Specular IOR Level'i 0.5'te kilitle
    for spec_name in ('Specular IOR Level', 'Specular'):
        if spec_name in mat_nodes['principled_node'].inputs:
            mat_nodes['principled_node'].inputs[spec_name].default_value = 0.5
            break
    
    if 'shader_mix' in params:
        mat_nodes['mix_shader_node'].inputs[0].default_value = float(params['shader_mix'])
    
    # Dielectric uygunluk için sabitlemeler
    for slot in ('Metallic', 'Clearcoat', 'Sheen', 'Coat Weight'):
        if slot in mat_nodes['principled_node'].inputs:
            mat_nodes['principled_node'].inputs[slot].default_value = 0.0


# ============================================================================
# MARS PARAMETER APPLICATION
# ============================================================================
def set_mars_params(mat_nodes, params):
    """
    Mars hybrid shader parametrelerini uygular.
    OrenNayar + Principled BSDF mix with texture/gray blending.
    """
    if 'mars_base_gray' in params and 'mars_value_node' in mat_nodes:
        mat_nodes['mars_value_node'].outputs['Value'].default_value = float(params['mars_base_gray'])

    if 'mars_tex_mix' in params and 'mars_mix_value_node' in mat_nodes:
        mat_nodes['mars_mix_value_node'].inputs['Factor'].default_value = float(params['mars_tex_mix'])

    if 'mars_oren_rough' in params and 'mars_oren_node' in mat_nodes:
        mat_nodes['mars_oren_node'].inputs['Roughness'].default_value = float(params['mars_oren_rough'])

    if 'mars_princ_rough' in params and 'mars_principled_node' in mat_nodes:
        mat_nodes['mars_principled_node'].inputs['Roughness'].default_value = float(params['mars_princ_rough'])

    if 'mars_ior' in params and 'mars_principled_node' in mat_nodes:
        if 'IOR' in mat_nodes['mars_principled_node'].inputs:
            mat_nodes['mars_principled_node'].inputs['IOR'].default_value = float(params['mars_ior'])

    # Specular IOR Level sabit 0.5
    if 'mars_principled_node' in mat_nodes:
        for spec_name in ('Specular IOR Level', 'Specular'):
            if spec_name in mat_nodes['mars_principled_node'].inputs:
                mat_nodes['mars_principled_node'].inputs[spec_name].default_value = 0.5
                break

    if 'mars_shader_mix' in params and 'mars_mix_shader_node' in mat_nodes:
        mat_nodes['mars_mix_shader_node'].inputs[0].default_value = float(params['mars_shader_mix'])

    # Dielectric sabitler
    if 'mars_principled_node' in mat_nodes:
        for slot in ('Metallic', 'Clearcoat', 'Sheen', 'Coat Weight'):
            if slot in mat_nodes['mars_principled_node'].inputs:
                mat_nodes['mars_principled_node'].inputs[slot].default_value = 0.0

    if 'mars_albedo_mul' in params and 'mars_albedo_multiplier' in mat_nodes:
        mul = float(params['mars_albedo_mul'])
        try:
            mat_nodes['mars_albedo_multiplier'].inputs[1].default_value = (mul, mul, mul)
        except Exception:
            vec = getattr(mat_nodes['mars_albedo_multiplier'].inputs[1], 'default_value', None)
            if vec is not None:
                try:
                    vec[:] = (mul, mul, mul)
                except Exception:
                    mat_nodes['mars_albedo_multiplier'].inputs[1].default_value = (mul, mul, mul, 1.0)


# ============================================================================
# ATMOSPHERE PARAMETER APPLICATION
# ============================================================================
def set_atmosphere_params(mat_nodes, params):
    """
    Update atmosphere shader node values from optimizer params.
    
    Args:
        mat_nodes: Dictionary of material node references (from build_scene_and_materials)
        params: Dictionary of parameter values
    """
    if 'atm_volume_node' not in mat_nodes or mat_nodes['atm_volume_node'] is None:
        return
    
    vol = mat_nodes['atm_volume_node']  # Principled Volume
    color_mode = mat_nodes.get('atm_color_mode', 'single')
    
    if 'atm_beta0' in params and mat_nodes.get('atm_beta0_node'):
        mat_nodes['atm_beta0_node'].outputs[0].default_value = float(params['atm_beta0'])
    
    if 'atm_scale_height' in params and mat_nodes.get('atm_H_node'):
        mat_nodes['atm_H_node'].outputs[0].default_value = float(params['atm_scale_height'])
    
    if 'atm_anisotropy' in params:
        vol.inputs['Anisotropy'].default_value = float(params['atm_anisotropy'])
    
    # Color: RGB mode or single-channel mode
    if color_mode == 'rgb' and all(k in params for k in ['atm_color_r', 'atm_color_g', 'atm_color_b']):
        r = float(params['atm_color_r'])
        g = float(params['atm_color_g'])
        b = float(params['atm_color_b'])
        vol.inputs['Color'].default_value = (r, g, b, 1.0)
    elif 'atm_color_r' in params:
        r = float(params['atm_color_r'])
        # Proportional G/B from Mars dust ratio (0.8, 0.5, 0.3)
        vol.inputs['Color'].default_value = (r, r * 0.625, r * 0.375, 1.0)


# ============================================================================
# ASSET PATH HELPER
# ============================================================================
def add_asset_paths(state, dem_path=None, body_names=None):
    """Input/UV/albedo varlık yollarını State'e ekler
    
    Args:
        state: CORTO State object
        dem_path: Optional path to Mars DEM TIFF for displacement
        body_names: Optional list of body OBJ filenames [phobos, mars, deimos].
                    UV data dosyaları bu isimlerden türetilir (.obj → .json).
    """
    base = state.path["input_path"]
    
    # UV data dosya adlarını body OBJ isimlerinden türet
    if body_names:
        phobos_uv = body_names[0].replace('.obj', '.json')
        deimos_uv = body_names[2].replace('.obj', '.json') if len(body_names) > 2 else 'g_deimos_162m_spc_0000n00000_v001.json'
    else:
        # Fallback: phobos_main.py ile uyumlu varsayılan
        phobos_uv = 'g_phobos_018m_spc_0000n00000_v002.json'
        deimos_uv = 'g_deimos_162m_spc_0000n00000_v001.json'
    
    state.add_path('albedo_path_1', os.path.join(base, 'body', 'albedo', 'Phobos grayscale.jpg'))
    state.add_path('uv_data_path_1', os.path.join(base, 'body', 'uv data', phobos_uv))
    state.add_path('albedo_path_2', r'C:\Users\tolga\.gemini\antigravity\scratch\corto_test\input\S07_Mars_Phobos_Deimos\body\albedo\Mars_Viking_MDIM21_ClrMosaic_global_32k_shifted.tif')
    state.add_path('uv_data_path_2', os.path.join(base, 'body', 'uv data', 'Mars_65k.json'))
    state.add_path('mars_bump_path', os.path.join(base, 'body', 'displacement', 'Mars_MOLA_DEM_f32.tif'))
    state.add_path('albedo_path_3', os.path.join(base, 'body', 'albedo', 'Deimos grayscale.jpg'))
    state.add_path('uv_data_path_3', os.path.join(base, 'body', 'uv data', deimos_uv))
    
    # DEM displacement path (configurable)
    if dem_path:
        state.add_path('mars_dem_path', dem_path)
        state.add_path('displacement_path_2', dem_path)
        print(f"  ✅ Mars DEM path set: {dem_path}")
    else:
        # Default DEM location
        default_dem = os.path.join(base, 'body', 'displacement', 'Mars_MOLA_DEM_f32.tif')
        if os.path.exists(default_dem):
            state.add_path('mars_dem_path', default_dem)
            state.add_path('displacement_path_2', default_dem)
            print(f"  ✅ Mars DEM found at default path")
        else:
            print(f"  ℹ️ No Mars DEM found (checked: {default_dem})")