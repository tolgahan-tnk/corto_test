"""
tt

"""
import sys
import os
sys.path.append(os.getcwd())
import cortopy as corto
import numpy as np
import spiceypy as spice
from scipy.spatial.transform import Rotation as R
import time
import requests

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

### (1.1) LOAD SPICE KERNELS AND COMPUTE GEOMETRY ###
UTC_TIME = "2018-08-02T08:48:03.686"

KERNEL_LIST = {
    "generic_kernels/lsk/naif0012.tls",
    "generic_kernels/pck/pck00010.tpc",
    "MEX/kernels/sclk/former_versions/MEX_250417_STEP.TSC",
    "MEX/kernels/spk/MAR097_030101_300101_V0001.BSP",
    "generic_kernels/spk/planets/a_old_versions/de405.bsp",
    "MEX/kernels/spk/ORMM_T19_180801000000_01460.BSP",
    "MEX/kernels/ck/ATNM_MEASURED_180101_181231_V01.BC",
    "MEX/kernels/fk/MEX_V16.TF",
}

MIRRORS = [
    "https://naif.jpl.nasa.gov/pub/naif",
    "http://naif.jpl.nasa.gov/pub/naif",
    "https://spiftp.esac.esa.int/data/SPICE",
]

def dl(rel, mirrors, ret=3):
    fn = os.path.basename(rel)
    if os.path.isfile(fn) and os.path.getsize(fn):
        return
    for m in mirrors:
        url = f"{m}/{rel}"
        for n in range(1, ret + 1):
            try:
                with requests.get(url, timeout=(10, 300), stream=True) as r:
                    r.raise_for_status()
                    with open(fn, "wb") as f:
                        for c in r.iter_content(1 << 20):
                            f.write(c)
                return
            except (requests.Timeout, requests.ConnectionError):
                time.sleep(2)
    raise RuntimeError(f"{fn} could not be downloaded")

for k in KERNEL_LIST:
    dl(k, MIRRORS)
for fn in map(os.path.basename, KERNEL_LIST):
    spice.furnsh(fn)

et = spice.str2et(UTC_TIME)
v_pho_mars = spice.spkezr("PHOBOS", et, "J2000", "NONE", "MARS")[0][:3]
v_dei_mars = spice.spkezr("DEIMOS", et, "J2000", "NONE", "MARS")[0][:3]
v_mex_mars = spice.spkezr("-41", et, "J2000", "NONE", "MARS")[0][:3]
v_sun_mars = spice.spkezr("SUN", et, "J2000", "NONE", "MARS")[0][:3]

R_i_mars = spice.pxform("J2000", "IAU_MARS", et)
R_i_pho = spice.pxform("J2000", "IAU_PHOBOS", et)
R_i_dei = spice.pxform("J2000", "IAU_DEIMOS", et)
R_i_sc = spice.pxform("J2000", "MEX_SPACECRAFT", et)

v_sc_pho = R_i_pho @ (v_mex_mars - v_pho_mars)
v_mar_pho = R_i_pho @ (-v_pho_mars)
v_dei_pho = R_i_pho @ (v_dei_mars - v_pho_mars)
v_sun_pho = R_i_pho @ (v_sun_mars - v_pho_mars)

R_pho_sc = R_i_sc @ R_i_pho.T
R_pho_mars = R_i_mars @ R_i_pho.T
R_pho_dei = R_i_dei @ R_i_pho.T

def rot_to_quat(Rm):
    q = R.from_matrix(Rm).as_quat()  # returns [x,y,z,w]
    return np.array([q[3], q[0], q[1], q[2]])

quat_sc = rot_to_quat(R_pho_sc)
quat_mars = rot_to_quat(R_pho_mars)
quat_dei = rot_to_quat(R_pho_dei)

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
    body_2.set_position(v_mar_pho)
    body_2.set_orientation(quat_mars)
    body_3.set_position(v_dei_pho)
    body_3.set_orientation(quat_dei)
    cam.set_position(v_sc_pho)
    cam.set_orientation(quat_sc)
    cam_hrsc.set_position(v_sc_pho)
    cam_hrsc.set_orientation(quat_sc)
    sun.set_position(v_sun_pho)
    ENV.RenderOne(cam, State, index=idx, depth_flag=True)
    ENV.RenderOne(cam_hrsc, State, index=idx, depth_flag=True)

# Save .blend as debug
corto.Utils.save_blend(State)

# Unload kernels
for fn in map(os.path.basename, KERNEL_LIST):
    spice.unload(fn)
