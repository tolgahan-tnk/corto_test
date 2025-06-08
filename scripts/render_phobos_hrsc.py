#!/usr/bin/env python3
"""Render Phobos image using HRSC kernels for Mars Express.

This script downloads the required SPICE kernels, computes the position and
orientation of Phobos, Mars Express, and the Sun at
UTC ``2018-08-02T08:48:03.686`` and uses :mod:`cortopy` to generate a single
image with HRSC camera parameters.

It expects that the S07_Mars_Phobos_Deimos scenario from the ``input`` folder is
available. Only the Phobos model is used.
"""

from __future__ import annotations

import os
import time
import requests
import numpy as np
import spiceypy as spice
import cortopy as corto

UTC_TIME = "2018-08-02T08:48:03.686"

KERNEL_LIST = {
    # Generic kernels
    "generic_kernels/lsk/naif0012.tls",
    "generic_kernels/pck/pck00010.tpc",
    # Mars Express mission kernels
    "MEX/kernels/sclk/former_versions/MEX_250417_STEP.TSC",
    "MEX/kernels/spk/MAR097_030101_300101_V0001.BSP",
    "generic_kernels/spk/planets/a_old_versions/de405.bsp",
    "MEX/kernels/spk/ORMM_T19_180801000000_01460.BSP",
    # Spacecraft attitude
    "MEX/kernels/ck/ATNM_MEASURED_180101_181231_V01.BC",
    # Frame kernel
    "MEX/kernels/fk/MEX_V16.TF",
}

MIRRORS = [
    "https://naif.jpl.nasa.gov/pub/naif",
    "http://naif.jpl.nasa.gov/pub/naif",
    "https://spiftp.esac.esa.int/data/SPICE",
]


def dl(rel: str, mirrors: list[str], ret: int = 3) -> None:
    """Download ``rel`` from the first available mirror."""
    fn = os.path.basename(rel)
    if os.path.isfile(fn) and os.path.getsize(fn):
        print(f"✓ {fn} mevcut")
        return
    for m in mirrors:
        url = f"{m}/{rel}"
        for n in range(1, ret + 1):
            try:
                print(f"[{fn}] {n}/{ret} → {url}")
                with requests.get(url, timeout=(10, 300), stream=True) as r:
                    r.raise_for_status()
                    with open(fn, "wb") as f:
                        for c in r.iter_content(1 << 20):
                            f.write(c)
                print(f"✓ {fn} indirildi")
                return
            except (requests.Timeout, requests.ConnectionError) as e:
                print(f"⚠️  {e}")
                time.sleep(2)
    raise RuntimeError(f"{fn} indirilemedi")


for k in KERNEL_LIST:
    dl(k, MIRRORS)
for fn in map(os.path.basename, KERNEL_LIST):
    spice.furnsh(fn)

# ---------------------------------------------------------------------------
# Geometry computation
# ---------------------------------------------------------------------------
et = spice.str2et(UTC_TIME)

v_pho_mars = spice.spkezr("PHOBOS", et, "J2000", "NONE", "MARS")[0][:3]
v_mex_mars = spice.spkezr("-41", et, "J2000", "NONE", "MARS")[0][:3]
v_sun_mars = spice.spkezr("SUN", et, "J2000", "NONE", "MARS")[0][:3]

R_i_pho = spice.pxform("J2000", "IAU_PHOBOS", et)
R_i_sc = spice.pxform("J2000", "MEX_SPACECRAFT", et)

v_sc_pho = R_i_pho @ (v_mex_mars - v_pho_mars)
v_sun_pho = R_i_pho @ (v_sun_mars - v_pho_mars)
R_pho_sc = R_i_sc @ R_i_pho.T

q_sc = spice.m2q(R_pho_sc)
q_pho = spice.m2q(R_i_pho)

geometry = {
    "sun": {"position": np.array([v_sun_pho])},
    "camera": {
        "position": np.array([v_sc_pho]),
        "orientation": np.array([q_sc]),
    },
    "body": {
        "position": np.array([[0.0, 0.0, 0.0]]),
        "orientation": np.array([q_pho]),
    },
}

# ---------------------------------------------------------------------------
# CORTO scene setup
# ---------------------------------------------------------------------------

corto.Utils.clean_scene()

scenario_name = "S07_Mars_Phobos_Deimos"
# The default dataset ships with ``scene.json`` and ``geometry.json``
# files inside the scenario folder.  Adjust these names if your
# local copy differs.
scene_name = "scene.json"
geometry_name = "geometry.json"
body_name = "g_phobos_287m_spc_0000n00000_v002.obj"
State = corto.State(
    scene=scene_name,
    geometry=geometry_name,
    body=body_name,
    scenario=scenario_name,
)
State.geometry = geometry

cam = corto.Camera("WFOV_Camera", State.properties_cam)
sun = corto.Sun("Sun", State.properties_sun)
name, _ = os.path.splitext(body_name)
body = corto.Body(name, State.properties_body)
rendering_engine = corto.Rendering(State.properties_rendering)
ENV = corto.Environment(cam, body, sun, rendering_engine)

material = corto.Shading.create_new_material("phobos_material")
corto.Shading.create_branch_albedo_mix(material, State)
corto.Shading.load_uv_data(body, State)
corto.Shading.assign_material_to_object(material, body)

tree = corto.Compositing.create_compositing()
render_node = corto.Compositing.rendering_node(tree, (0, 0))
corto.Compositing.create_img_denoise_branch(tree, render_node)
corto.Compositing.create_depth_branch(tree, render_node)
corto.Compositing.create_slopes_branch(tree, render_node, State)
corto.Compositing.create_maskID_branch(tree, render_node, State)

body.set_scale(np.array([1, 1, 1]))
ENV.PositionAll(State, index=0)
ENV.RenderOne(cam, State, index=0, depth_flag=True)

corto.Utils.save_blend(State)

for fn in map(os.path.basename, KERNEL_LIST):
    spice.unload(fn)
