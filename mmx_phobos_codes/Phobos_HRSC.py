"""
tt

"""
import sys
import os
sys.path.append(os.getcwd())
import cortopy as corto
import numpy as np
from mmx_phobos_codes import phobos_geometry

## Clean all existing/Default objects in the scene 
corto.Utils.clean_scene()

### (1) DEFINE INPUT ### 
scenario_name = "S07_Mars_Phobos_Deimos" # Name of the scenario folder
scene_name = "scene_mmx.json" # name of the scene input
geometry_name = "geometry_mmx.json" # name of the geometry input
body_name = ["g_phobos_287m_spc_0000n00000_v002.obj",
             "Mars_65k.obj",
             "g_deimos_162m_spc_0000n00000_v001.obj"] # name of the body input

# Load inputs and settings into the State object
State = corto.State(scene = scene_name, geometry = geometry_name, body = body_name, scenario = scenario_name)
# Add extra inputs 
State.add_path('albedo_path_1',os.path.join(State.path["input_path"],'body','albedo','Phobos grayscale.jpg'))
State.add_path('uv_data_path_1',os.path.join(State.path["input_path"],'body','uv data','g_phobos_287m_spc_0000n00000_v002.json'))
State.add_path('albedo_path_2',os.path.join(State.path["input_path"],'body','albedo','mars_1k_color.jpg'))
State.add_path('uv_data_path_2',os.path.join(State.path["input_path"],'body','uv data','Mars_65k.json'))
State.add_path('albedo_path_3',os.path.join(State.path["input_path"],'body','albedo','Deimos grayscale.jpg'))
State.add_path('uv_data_path_3',os.path.join(State.path["input_path"],'body','uv data','g_deimos_162m_spc_0000n00000_v001.json'))

### (1.1) LOAD SPICE GEOMETRY ###
geometry = phobos_geometry.compute_geometry()
positions = geometry["positions"]
orientations = geometry["orientations"]

### (2) SETUP THE SCENE ###
# Setup bodies
cam = corto.Camera('WFOV_Camera', State.properties_cam)

# ---------------------------------------------------------------------------
# Additional camera representing the HRSC SRC channel
# ---------------------------------------------------------------------------
hrsc_camera_properties = State.properties_cam.copy()

# HRSC SRC intrinsic parameters
focal_length_mm = 975.0
pixel_size_mm = 9.0 / 1000.0
active_pixels_x = 1024
active_pixels_y = 1032

# Compute horizontal field of view in degrees
fov_rad = 2 * np.arctan(((active_pixels_x * pixel_size_mm) / 2) / focal_length_mm)
hrsc_camera_properties["fov"] = np.degrees(fov_rad)
hrsc_camera_properties["res_x"] = active_pixels_x
hrsc_camera_properties["res_y"] = active_pixels_y

# Compute the camera intrinsic matrix K
f_px = active_pixels_x / (2 * np.tan(fov_rad / 2))
cx = active_pixels_x / 2
cy = active_pixels_y / 2
hrsc_camera_properties["K"] = np.array([[f_px, 0, cx], [0, f_px, cy], [0, 0, 1]])

cam_hrsc = corto.Camera('HRSC_SRC_Camera', hrsc_camera_properties)

##########Edited
sun = corto.Sun('Sun',State.properties_sun)
name_1, _ = os.path.splitext(body_name[0])
body_1 = corto.Body(name_1,State.properties_body_1)
name_2, _ = os.path.splitext(body_name[1])
body_2 = corto.Body(name_2,State.properties_body_2)
name_3, _ = os.path.splitext(body_name[2])
body_3 = corto.Body(name_3,State.properties_body_3)

# Setup rendering engine
rendering_engine = corto.Rendering(State.properties_rendering)

# Setup environment using the default camera
ENV = corto.Environment(cam, [body_1,body_2,body_3], sun, rendering_engine)

### (3) MATERIAL PROPERTIES ###
material_1 = corto.Shading.create_new_material('properties body 1')
material_2 = corto.Shading.create_new_material('properties body 2')
material_3 = corto.Shading.create_new_material('properties body 3')

corto.Shading.create_branch_albedo_mix(material_1, State,1)
corto.Shading.create_branch_albedo_mix(material_2, State,2)
corto.Shading.create_branch_albedo_mix(material_3, State,3)

corto.Shading.load_uv_data(body_1,State,1)
corto.Shading.assign_material_to_object(material_1, body_1)
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
body_1.set_scale(np.array([1, 1, 1]))
body_2.set_scale(np.array([1e3, 1e3, 1e3]))
body_3.set_scale(np.array([1, 1, 1]))

n_img = 1
for idx in range(n_img):
    body_1.set_position(np.array([0, 0, 0]))
    body_1.set_orientation(np.array([1, 0, 0, 0]))
    body_2.set_position(positions["mars"])
    body_2.set_orientation(orientations["mars"])
    body_3.set_position(positions["deimos"])
    body_3.set_orientation(orientations["deimos"])
    cam.set_position(positions["sc"])
    cam.set_orientation(orientations["sc"])
    cam_hrsc.set_position(positions["sc"])
    cam_hrsc.set_orientation(orientations["sc"])
    sun.set_position(positions["sun"])
    ENV.RenderOne(cam, State, index=idx, depth_flag=True)
    ENV.RenderOne(cam_hrsc, State, index=idx, depth_flag=True)

# Save .blend as debug
corto.Utils.save_blend(State)

# Unload kernels
phobos_geometry.unload_kernels()
