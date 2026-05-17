"""Visual comparison: Frame 01 alignment strategies.

Produces 3 overlay images:
  A) Current (clamped dx=-48.6)
  B) Unclamped (dx=-114)
  C) Largest connected component mask
"""
import numpy as np
from PIL import Image
from pathlib import Path
from skimage.registration import phase_cross_correlation
from skimage.transform import AffineTransform, warp
from scipy.ndimage import label, binary_dilation

eval_dir = Path("debug_opt_test/eval_0001")
FRAME = "frame_01_H7982_0006_SR2"
OUT = Path("debug_opt_test")

# Load
syn_aligned = np.array(Image.open(eval_dir / f"{FRAME}_mars_syn_aligned.tif")).astype(float)
pds_aligned = np.array(Image.open(eval_dir / f"{FRAME}_mars_real_aligned.tif")).astype(float)
syn_masked = np.array(Image.open(eval_dir / f"{FRAME}_mars_syn_masked.tif")).astype(float)
pds_masked = np.array(Image.open(eval_dir / f"{FRAME}_mars_real_masked.tif")).astype(float)

mars_mask = pds_masked > 0
pds_safe = np.nan_to_num(pds_aligned, nan=0.0)

# ── Strategy A: Current (clamped) ──
print("Strategy A: Current clamped shift...")
shift_full, _, _ = phase_cross_correlation(
    pds_safe, syn_aligned, upsample_factor=100,
    reference_mask=mars_mask, moving_mask=mars_mask
)
print(f"  Raw shift: dy={shift_full[0]:.2f}, dx={shift_full[1]:.2f}")

# Crop to mars bbox
ys, xs = np.where(mars_mask)
pad = 20
y0, y1 = max(0, ys.min()-pad), min(pds_aligned.shape[0], ys.max()+pad+1)
x0, x1 = max(0, xs.min()-pad), min(pds_aligned.shape[1], xs.max()+pad+1)

syn_crop = syn_aligned[y0:y1, x0:x1].copy()
pds_crop = pds_safe[y0:y1, x0:x1].copy()
reg_crop = mars_mask[y0:y1, x0:x1].copy()

shift_crop, _, _ = phase_cross_correlation(
    pds_crop, syn_crop, upsample_factor=100,
    reference_mask=reg_crop, moving_mask=reg_crop
)
print(f"  Cropped shift: dy={shift_crop[0]:.2f}, dx={shift_crop[1]:.2f}")

max_dy = syn_crop.shape[0] * 0.05
max_dx = syn_crop.shape[1] * 0.05
clamped_dy = float(np.clip(shift_crop[0], -max_dy, max_dy))
clamped_dx = float(np.clip(shift_crop[1], -max_dx, max_dx))
print(f"  Clamped: dy={clamped_dy:.2f}, dx={clamped_dx:.2f}")

tform_a = AffineTransform(translation=(clamped_dx, clamped_dy))
syn_a = warp(syn_crop, tform_a.inverse, output_shape=syn_crop.shape,
             preserve_range=True, mode="constant", cval=0, order=1)

# ── Strategy B: Unclamped ──
print("\nStrategy B: Unclamped shift...")
unclamped_dy = float(shift_crop[0])
unclamped_dx = float(shift_crop[1])
print(f"  Using: dy={unclamped_dy:.2f}, dx={unclamped_dx:.2f}")

tform_b = AffineTransform(translation=(unclamped_dx, unclamped_dy))
syn_b = warp(syn_crop, tform_b.inverse, output_shape=syn_crop.shape,
             preserve_range=True, mode="constant", cval=0, order=1)

# ── Strategy C: Largest Connected Component ──
print("\nStrategy C: Largest connected component...")
labeled, n_components = label(mars_mask)
print(f"  Total connected components: {n_components}")

# Find largest
sizes = [(labeled == i).sum() for i in range(1, n_components + 1)]
for i, s in enumerate(sizes):
    print(f"    Component {i+1}: {s} px ({100*s/mars_mask.sum():.1f}%)")

largest_id = np.argmax(sizes) + 1
lcc_mask = labeled == largest_id
print(f"  Using component {largest_id}: {lcc_mask.sum()} px")

# Crop to LCC bbox
ys_l, xs_l = np.where(lcc_mask)
y0l, y1l = max(0, ys_l.min()-pad), min(pds_aligned.shape[0], ys_l.max()+pad+1)
x0l, x1l = max(0, xs_l.min()-pad), min(pds_aligned.shape[1], xs_l.max()+pad+1)

syn_crop_c = syn_aligned[y0l:y1l, x0l:x1l].copy()
pds_crop_c = pds_safe[y0l:y1l, x0l:x1l].copy()
lcc_crop = lcc_mask[y0l:y1l, x0l:x1l].copy()

shift_lcc, _, _ = phase_cross_correlation(
    pds_crop_c, syn_crop_c, upsample_factor=100,
    reference_mask=lcc_crop, moving_mask=lcc_crop
)
print(f"  LCC shift: dy={shift_lcc[0]:.2f}, dx={shift_lcc[1]:.2f}")

tform_c = AffineTransform(translation=(float(shift_lcc[1]), float(shift_lcc[0])))
syn_c = warp(syn_crop_c, tform_c.inverse, output_shape=syn_crop_c.shape,
             preserve_range=True, mode="constant", cval=0, order=1)

# ── Generate overlay images ──
def make_overlay(syn, pds, mask, title):
    """Red=syn, Green=PDS, Yellow=overlap. Mask outline in white."""
    h, w = pds.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:,:,0] = np.clip(syn * 255, 0, 255).astype(np.uint8)
    img[:,:,1] = np.clip(pds * 255, 0, 255).astype(np.uint8)
    # Mask outline
    outline = binary_dilation(mask, iterations=1) & ~mask
    img[outline] = [255, 255, 255]
    return img

# A: Clamped
img_a = make_overlay(syn_a, pds_crop, reg_crop, "Clamped")
Image.fromarray(img_a).save(OUT / "compare_A_clamped.png")
print(f"\nSaved: compare_A_clamped.png (dy={clamped_dy:.1f}, dx={clamped_dx:.1f})")

# B: Unclamped
img_b = make_overlay(syn_b, pds_crop, reg_crop, "Unclamped")
Image.fromarray(img_b).save(OUT / "compare_B_unclamped.png")
print(f"Saved: compare_B_unclamped.png (dy={unclamped_dy:.1f}, dx={unclamped_dx:.1f})")

# C: LCC
img_c = make_overlay(syn_c, pds_crop_c, lcc_crop, "LCC")
Image.fromarray(img_c).save(OUT / "compare_C_lcc.png")
print(f"Saved: compare_C_lcc.png (dy={shift_lcc[0]:.1f}, dx={shift_lcc[1]:.1f})")

# Also save the LCC mask visualization
lcc_vis = np.zeros((*mars_mask.shape, 3), dtype=np.uint8)
lcc_vis[mars_mask, 1] = 100  # all mars in dim green
lcc_vis[lcc_mask, 1] = 255   # LCC in bright green
lcc_vis[~lcc_mask & mars_mask, 0] = 200  # non-LCC mars in red
ph_hole = (pds_aligned > 0) & ~mars_mask
lcc_vis[ph_hole, 2] = 200  # Phobos hole in blue
Image.fromarray(lcc_vis).save(OUT / "compare_lcc_mask.png")
print(f"Saved: compare_lcc_mask.png (Green=LCC, Red=excluded fragments, Blue=Phobos)")

print("\nDone! Check debug_opt_test/compare_*.png")
