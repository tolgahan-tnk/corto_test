# -*- coding: utf-8 -*-
"""
phobos_scene.py
Phobos Photometric Template Generator - Scene Management Module

Tasks:
- GPU forcing
- Blender scene mang.
- Shader and material installation
- Compositing branchs
- 16-bit PNG 
"""

import os
import sys
import numpy as np
import bpy
from PIL import Image

sys.path.append(os.getcwd())
import cortopy as corto


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
def build_scene_and_materials(State, body_names):
    """
    Sahne ve gelişmiş Phobos malzemesini oluşturur
    
    Args:
        State: CORTO State object
        body_names: Body dosya adları listesi [phobos.obj, mars.obj, deimos.obj]
    
    Returns:
        tuple: (ENV, cam, sun, bodies, mat_nodes, tree)
    """
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

    # Standard for Mars/Deimos
    material_2 = corto.Shading.create_new_material('Mars_Standard')
    material_3 = corto.Shading.create_new_material('Deimos_Standard')
    corto.Shading.create_branch_albedo_mix(material_2, State, 2)
    corto.Shading.create_branch_albedo_mix(material_3, State, 3)
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
                    # Get current path
                    current_path = slot.path
                    
                    # Fix Windows backslashes to forward slashes
                    if '\\' in current_path:
                        current_path = current_path.replace('\\', '/')
                    
                    # Ensure frame number placeholder exists
                    if '#' not in current_path:
                        # Add frame numbering if missing
                        if '/' in current_path:
                            # Has directory, append to end
                            current_path = current_path.rstrip('/') + '/######'
                        else:
                            # No directory, just numbering
                            current_path = '######'
                    
                    # Update the path
                    slot.path = current_path
                    
        print(f" ✅ Compositor paths configured for {platform.system()}")
                    
    except Exception as e:
        print(f" ⚠️ Warning: compositor path fix failed: {e}")

    # Scales
    body_1.set_scale(np.array([1, 1, 1]))
    body_2.set_scale(np.array([1e3, 1e3, 1e3]))
    body_3.set_scale(np.array([1, 1, 1]))

    # Pack refs
    mat_nodes = {
        'value_node': value_node,
        'mix_value_node': mix_value_node,
        'oren_node': oren_node,
        'principled_node': principled_node,
        'mix_shader_node': mix_shader_node,
    }

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
# ASSET PATH HELPER
# ============================================================================
def add_asset_paths(state):
    """Input/UV/albedo varlık yollarını State'e ekler"""
    base = state.path["input_path"]
    state.add_path('albedo_path_1', os.path.join(base, 'body', 'albedo', 'Phobos grayscale.jpg'))
    state.add_path('uv_data_path_1', os.path.join(base, 'body', 'uv data', 'g_phobos_036m_spc_0000n00000_v002.json'))
    state.add_path('albedo_path_2', os.path.join(base, 'body', 'albedo', 'mars_1k_color.jpg'))
    state.add_path('uv_data_path_2', os.path.join(base, 'body', 'uv data', 'Mars_65k.json'))
    state.add_path('albedo_path_3', os.path.join(base, 'body', 'albedo', 'Deimos grayscale.jpg'))
    state.add_path('uv_data_path_3', os.path.join(base, 'body', 'uv data', 'g_deimos_162m_spc_0000n00000_v001.json'))
