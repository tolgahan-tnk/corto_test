"""
Post-processor for tutorial renders.

Replicates the optimizer's post-render pipeline:
  1. Load 16-bit PNG → normalize to [0,1]
  2. Apply DN threshold (zero out weak signal)
  3. Re-normalize positive pixels to [0,1]
  4. Save as 16-bit TIFF

Usage:
    from post_processor_for_tutorial import post_process_renders
    post_process_renders(output_dir, threshold_value=0.002)
"""

import os
import numpy as np
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'Pillow'])
    from PIL import Image


def post_process_single(
    input_path: str,
    output_path: str,
    threshold_value: float = 0.002,
):
    """Process a single rendered image: threshold + re-normalize.

    Args:
        input_path:      Path to 16-bit PNG from Blender
        output_path:     Path to save processed 16-bit TIFF
        threshold_value: DN threshold in [0,1] scale (from coefficients.json)
    """
    img = np.array(Image.open(input_path))

    # Normalize to [0, 1]
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)

    img = np.clip(img, 0.0, 1.0)

    # Threshold: zero out weak signal
    img = np.where(img < threshold_value, np.float32(0.0), img)

    # Re-normalize positive values to [0, 1]
    positive_mask = img > 0.0
    if np.any(positive_mask):
        pos_vals = img[positive_mask]
        min_val = float(pos_vals.min())
        max_val = float(pos_vals.max())
        if max_val > min_val:
            img[positive_mask] = (pos_vals - min_val) / (max_val - min_val)
        else:
            img[positive_mask] = 1.0
    else:
        img = np.zeros_like(img, dtype=np.float32)

    # Save as 16-bit TIFF
    out_uint16 = np.clip(img * 65535.0, 0, 65535).astype(np.uint16)
    Image.fromarray(out_uint16).save(output_path)

    return img


def post_process_renders(
    output_dir: str,
    threshold_value: float = 0.002,
    pattern: str = "*.png",
    suffix: str = "_processed",
):
    """Process all rendered PNGs in a directory.

    Args:
        output_dir:      Directory containing Blender render outputs
        threshold_value: DN threshold in [0,1] scale
        pattern:         Glob pattern for input files
        suffix:          Suffix added to processed filenames

    Returns:
        List of processed file paths
    """
    output_dir = Path(output_dir)
    png_files = sorted(output_dir.glob(pattern))

    if not png_files:
        print(f"  No files matching '{pattern}' in {output_dir}")
        return []

    processed = []
    for png in png_files:
        out_name = png.stem + suffix + ".tif"
        out_path = output_dir / out_name

        post_process_single(str(png), str(out_path), threshold_value)
        processed.append(str(out_path))
        print(f"  {png.name} -> {out_name} (threshold={threshold_value:.6f})")

    return processed
