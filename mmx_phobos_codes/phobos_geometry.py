import os
import time
import requests
import numpy as np
import spiceypy as spice
from scipy.spatial.transform import Rotation as R

# Default UTC time used if none provided
DEFAULT_UTC_TIME = "2018-08-02T08:48:03.686"

# SPICE kernels required for geometry computation
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

# Mirrors used if a kernel is missing locally
MIRRORS = [
    "https://naif.jpl.nasa.gov/pub/naif",
    "http://naif.jpl.nasa.gov/pub/naif",
    "https://spiftp.esac.esa.int/data/SPICE",
]

def dl(rel, mirrors, ret=3):
    """Download `rel` from `mirrors` if it does not exist."""
    fn = os.path.basename(rel)
    if os.path.isfile(fn) and os.path.getsize(fn):
        return
    for m in mirrors:
        url = f"{m}/{rel}"
        for _ in range(1, ret + 1):
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


def load_kernels():
    for k in KERNEL_LIST:
        dl(k, MIRRORS)
    for fn in map(os.path.basename, KERNEL_LIST):
        spice.furnsh(fn)


def unload_kernels():
    for fn in map(os.path.basename, KERNEL_LIST):
        spice.unload(fn)


def _rot_to_quat(Rm):
    q = R.from_matrix(Rm).as_quat()  # [x, y, z, w]
    return np.array([q[3], q[0], q[1], q[2]])


def compute_geometry(utc_time=DEFAULT_UTC_TIME):
    """Return positions and orientations relative to Phobos."""
    load_kernels()
    et = spice.str2et(utc_time)

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

    geometry = {
        "et": et,
        "positions": {
            "sc": v_sc_pho,
            "mars": v_mar_pho,
            "deimos": v_dei_pho,
            "sun": v_sun_pho,
        },
        "orientations": {
            "sc": _rot_to_quat(R_pho_sc),
            "mars": _rot_to_quat(R_pho_mars),
            "deimos": _rot_to_quat(R_pho_dei),
        },
    }
    return geometry


if __name__ == "__main__":
    geo = compute_geometry()
    for k, v in geo.items():
        print(f"{k}: {v}")
    unload_kernels()
