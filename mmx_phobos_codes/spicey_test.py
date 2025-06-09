#!/usr/bin/env python3
# Mars Express / HRSC Phobos görüntüsü (HI454_0005_SR2.IMG)
# Konum-tutum hesaplaması – doğru CK (ATNM_MEASURED) ile

import os, time, requests, numpy as np, spiceypy as spice
from subprocess import run

UTC_TIME = "2018-08-02T08:48:03.686"          # IMAGE_TIME

KERNEL_LIST = {
    # genel çekirdekler
    "generic_kernels/lsk/naif0012.tls",
    "generic_kernels/pck/pck00010.tpc",

    # Mars-Express görev çekirdekleri
    "MEX/kernels/sclk/former_versions/MEX_250417_STEP.TSC",
    "MEX/kernels/spk/MAR097_030101_300101_V0001.BSP",
    "generic_kernels/spk/planets/a_old_versions/de405.bsp",
    "MEX/kernels/spk/ORMM_T19_180801000000_01460.BSP",

    # **** düzeltme: gerçek uzay-aracı tutumu ****
    "MEX/kernels/ck/ATNM_MEASURED_180101_181231_V01.BC",

    # SA CK gerekmiyor ama isterseniz ek bırakabilirsiniz
    # "MEX/kernels/ck/MEX_SA_2018_V0010.BC",

    "MEX/kernels/fk/MEX_V16.TF",
}

MIRRORS = [
    "https://naif.jpl.nasa.gov/pub/naif",
    "http://naif.jpl.nasa.gov/pub/naif",
    "https://spiftp.esac.esa.int/data/SPICE"
]

# ---------- indirme yardimcisi (değişmedi) ----------
def dl(rel, mirrors, ret=3):
    fn = os.path.basename(rel)
    if os.path.isfile(fn) and os.path.getsize(fn):
        print(f"✓ {fn} mevcut"); return
    for m in mirrors:
        url = f"{m}/{rel}"
        for n in range(1, ret+1):
            try:
                print(f"[{fn}] {n}/{ret} → {url}")
                with requests.get(url, timeout=(10,300), stream=True) as r:
                    r.raise_for_status()
                    with open(fn,"wb") as f:
                        for c in r.iter_content(1<<20): f.write(c)
                print(f"✓ {fn} indirildi"); return
            except (requests.Timeout,requests.ConnectionError) as e:
                print(f"⚠️  {e}"); time.sleep(2)
    raise RuntimeError(f"{fn} indirilemedi")

for k in KERNEL_LIST: dl(k, MIRRORS)
for fn in map(os.path.basename, KERNEL_LIST): spice.furnsh(fn)

# ---------- geometri ----------
et = spice.str2et(UTC_TIME)
v_pho_mars = spice.spkezr("PHOBOS", et, "J2000","NONE","MARS")[0][:3]
v_mex_mars = spice.spkezr("-41",    et, "J2000","NONE","MARS")[0][:3]
v_sun_mars = spice.spkezr("SUN",    et, "J2000","NONE","MARS")[0][:3]

R_i_mars = spice.pxform("J2000","IAU_MARS",et)
R_i_pho  = spice.pxform("J2000","IAU_PHOBOS",et)
R_i_sc   = spice.pxform("J2000","MEX_SPACECRAFT",et)   # artık hatasız ✔️

v_sc_pho = R_i_pho @ (v_mex_mars - v_pho_mars)
v_mar_pho= R_i_pho @ (-v_pho_mars)
v_sun_pho= R_i_pho @ (v_sun_mars - v_pho_mars)
R_pho_sc = R_i_sc @ R_i_pho.T

def m2s(M): return "\n".join("    "+"  ".join(f"{x: .6f}" for x in r) for r in M)
out = [
  f"UTC  : {UTC_TIME}",
  f"ET   : {et:.3f}",
  "",
  "Mars-merkez (km)",
  f"  Mars→Phobos  : {v_pho_mars}",
  f"  Mars→SC      : {v_mex_mars}",
  f"  Mars→Sun     : {v_sun_mars}",
  "",
  "Dönüş matrisleri",
  "  IAU_MARS:",      m2s(R_i_mars),
  "  IAU_PHOBOS:",    m2s(R_i_pho),
  "  MEX_SPACECRAFT:",m2s(R_i_sc),
  "",
  "Phobos-çerçeve vektörler (km)",
  f"  Pho→SC  : {v_sc_pho}",
  f"  Pho→Mars: {v_mar_pho}",
  f"  Pho→Sun : {v_sun_pho}",
  "",
  "Phobos→SC DCM:",
  m2s(R_pho_sc)
]
print("\n".join(out))
with open("output_phobos_geometry.txt","w") as f: f.write("\n".join(out))
print("\n✓ output_phobos_geometry.txt yazıldı")

# ---------- temizlik ----------
for fn in map(os.path.basename, KERNEL_LIST): spice.unload(fn) 