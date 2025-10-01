# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 21:39:36 2025

@author: tolga
"""

"""
Tutorial script to render images of the S07_Mars_Phobos_Deimos scenario. 
This version has been revised to use an advanced, custom shading network for Phobos
based on the flowchart, while Mars and Deimos use the standard setup.

To run this tutorial, you first need to put data in the input folder. 
You can download the tutorial data from:

https://drive.google.com/drive/folders/1K3e5MyQin6T9d_EXLG_gFywJt3I18r6H?usp=sharing

"""
import sys
import os
import bpy # Imported to set colorspace
sys.path.append(os.getcwd())
import cortopy as corto
import numpy as np
from pathlib import Path


# Proje kökünü hesapla
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
print("Script is running in:", os.getcwd())


## Clean all existing/Default objects in the scene 
corto.Utils.clean_scene()

### (1) DEFINE INPUT ### 
scenario_name = "S07_Mars_Phobos_Deimos" # Name of the scenario folder
scene_name = "scene_mmx.json" # name of the scene input
geometry_name = "geometry_mmx.json" # name of the geometry input
body_name = ["g_phobos_018m_spc_0000n00000_v002.obj",
             "Mars_65k.obj",
             "g_deimos_162m_spc_0000n00000_v001.obj"] # name of the body input

# Load inputs and settings into the State object
State = corto.State(scene = scene_name, geometry = geometry_name, body = body_name, scenario = scenario_name)
# Add extra inputs 
State.add_path('albedo_path_1',os.path.join(State.path["input_path"],'body','albedo','Phobos grayscale.jpg'))
State.add_path('uv_data_path_1',os.path.join(State.path["input_path"],'body','uv data','g_phobos_018m_spc_0000n00000_v002.json'))
State.add_path('albedo_path_2',os.path.join(State.path["input_path"],'body','albedo','mars_1k_color.jpg'))
State.add_path('uv_data_path_2',os.path.join(State.path["input_path"],'body','uv data','Mars_65k.json'))
State.add_path('albedo_path_3',os.path.join(State.path["input_path"],'body','albedo','Deimos grayscale.jpg'))
State.add_path('uv_data_path_3',os.path.join(State.path["input_path"],'body','uv data','g_deimos_162m_spc_0000n00000_v001.json'))

### (2) SETUP THE SCENE ###
# Setup bodies
cam = corto.Camera('WFOV_Camera', State.properties_cam)
sun = corto.Sun('Sun',State.properties_sun)
name_1, _ = os.path.splitext(body_name[0])
body_1 = corto.Body(name_1,State.properties_body_1)
name_2, _ = os.path.splitext(body_name[1])
body_2 = corto.Body(name_2,State.properties_body_2)
name_3, _ = os.path.splitext(body_name[2])
body_3 = corto.Body(name_3,State.properties_body_3)

# Setup rendering engine
rendering_engine = corto.Rendering(State.properties_rendering)
# Setup environment
ENV = corto.Environment(cam, [body_1,body_2,body_3], sun, rendering_engine)

### (3) MATERIAL PROPERTIES ###

# --- Advanced Material for Phobos (body_1) based on the flowchart ---
phobos_material = corto.Shading.create_new_material('Phobos_Advanced_Hybrid_Grayscale')

# --- 1. Shader Models ---
oren_node = corto.Shading.diffuse_BSDF(phobos_material, location=(200, 200))
principled_node = corto.Shading.principled_BSDF(phobos_material, location=(200, -200))

# --- 2. Grayscale Sources & Mix ---
albedo_texture_node = corto.Shading.texture_node(phobos_material, State.path["albedo_path_1"], location=(-600, 100))
# Set colorspace to 'Non-Color' for accurate grayscale data
bpy.data.images[os.path.basename(State.path["albedo_path_1"])].colorspace_settings.name = 'Non-Color'

uv_map_node = corto.Shading.uv_map(phobos_material, location=(-800, 100)) # ADDED: UV Map node for texture coordinates

rgb_to_bw_node = corto.Shading.create_node("ShaderNodeRGBToBW", phobos_material, location=(-300, 100))
value_node = corto.Shading.create_node("ShaderNodeValue", phobos_material, location=(-300, -100))
mix_value_node = corto.Shading.create_node("ShaderNodeMix", phobos_material, location=(0, 0))
mix_value_node.data_type = 'FLOAT'

# --- 3. Final Shader Mix ---
mix_shader_node = corto.Shading.mix_node(phobos_material, location=(500, 0))
material_output_node = corto.Shading.material_output(phobos_material, location=(800, 0))

# --- 4. Set Tunable Parameters ---
# P1: Texture-Free Value (base gray color)
value_node.outputs["Value"].default_value = 0.15 
# P2: Texture Mix Factor (0.0 = 100% texture-free, 1.0 = 100% texture)
mix_value_node.inputs["Factor"].default_value = 0.8 
# P3: Oren-Nayar Roughness
oren_node.inputs["Roughness"].default_value = 0.9 
# P4: Principled BSDF Roughness
principled_node.inputs["Roughness"].default_value = 0.85 
# P5: Principled BSDF Specular
principled_node.inputs["Specular IOR Level"].default_value = 0.05 
# P6: Shader Mix Factor (0.0 = 100% Oren-Nayar, 1.0 = 100% Principled)
mix_shader_node.inputs[0].default_value = 0.1 

# --- 5. Link All Nodes for Phobos ---
# ADDED: Link the UV Map to the texture node to ensure correct projection
corto.Shading.link_nodes(phobos_material, uv_map_node.outputs["UV"], albedo_texture_node.inputs["Vector"])

# Link grayscale sources to the value mixer
corto.Shading.link_nodes(phobos_material, albedo_texture_node.outputs["Color"], rgb_to_bw_node.inputs["Color"])
corto.Shading.link_nodes(phobos_material, rgb_to_bw_node.outputs["Val"], mix_value_node.inputs["A"])
corto.Shading.link_nodes(phobos_material, value_node.outputs["Value"], mix_value_node.inputs["B"])
# Link mixed value to both shaders' color inputs
final_grayscale_output = mix_value_node.outputs["Result"]
corto.Shading.link_nodes(phobos_material, final_grayscale_output, oren_node.inputs["Color"])
corto.Shading.link_nodes(phobos_material, final_grayscale_output, principled_node.inputs["Base Color"])
# Link shaders to the shader mixer
corto.Shading.link_nodes(phobos_material, oren_node.outputs["BSDF"], mix_shader_node.inputs[1])
corto.Shading.link_nodes(phobos_material, principled_node.outputs["BSDF"], mix_shader_node.inputs[2])
# Link final shader to the output
corto.Shading.link_nodes(phobos_material, mix_shader_node.outputs["Shader"], material_output_node.inputs["Surface"])

# --- Assign the created material to Phobos ---
corto.Shading.load_uv_data(body_1,State,1)
corto.Shading.assign_material_to_object(phobos_material, body_1)


# --- Standard Materials for Mars (body_2) and Deimos (body_3) ---
# Create and assign materials using the standard library function
material_2 = corto.Shading.create_new_material('properties body 2')
material_3 = corto.Shading.create_new_material('properties body 3')

corto.Shading.create_branch_albedo_mix(material_2, State, 2)
corto.Shading.create_branch_albedo_mix(material_3, State, 3)

corto.Shading.load_uv_data(body_2,State,2)
corto.Shading.assign_material_to_object(material_2, body_2)
corto.Shading.load_uv_data(body_3,State,3)
corto.Shading.assign_material_to_object(material_3, body_3)


### (4) COMPOSITING PROPERTIES ###
# Build image-label pairs pipeline
tree = corto.Compositing.create_compositing()
render_node = corto.Compositing.rendering_node(tree, (0,0)) # Create Render node
corto.Compositing.create_img_denoise_branch(tree,render_node) # Create img_denoise branch
corto.Compositing.create_depth_branch(tree,render_node) # Create depth branch
corto.Compositing.create_slopes_branch(tree,render_node,State) # Create slopes branch
corto.Compositing.create_maskID_branch(tree,render_node,State) # Create ID mask branch

### (5) GENERATION OF IMG-LBL PAIRS ###
body_1.set_scale(np.array([1, 1, 1])) # adjust body scale for better test renderings
body_2.set_scale(np.array([1e3, 1e3, 1e3])) # adjust body scale for better test renderings
body_3.set_scale(np.array([1, 1, 1])) # adjust body scale for better test renderings

n_img = 1 # Render the first "n_img" images
for idx in range(0,n_img):
    ENV.PositionAll(State,index=idx)
    ENV.RenderOne(cam, State, index=idx, depth_flag = True)

# Save .blend as debug
corto.Utils.save_blend(State)

