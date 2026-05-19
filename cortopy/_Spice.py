import spiceypy as spice
import numpy as np
from typing import List, Tuple


def load_kernels(kernel_paths: List[str]) -> None:
    """Load SPICE kernels from a list of paths."""
    for path in kernel_paths:
        spice.furnsh(path)


def get_pose(target: str, observer: str, frame: str, et: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return position and orientation quaternion of target relative to observer."""
    state, _ = spice.spkezr(target, et, frame, 'NONE', observer)
    pos = np.array(state[:3])
    try:
        rot = spice.pxform(f'IAU_{target.upper()}', frame, et)
    except spice.stypes.SpiceyError:
        try:
            rot = spice.pxform(target, frame, et)
        except spice.stypes.SpiceyError:
            rot = np.eye(3)
    quat = spice.m2q(rot)
    return pos, np.array(quat)

