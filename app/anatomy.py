# anatomy.py
# Heuristic anatomical region labelling from GradCAM centre-of-mass position
# Based on approximate axial slice anatomy (no atlas registration)

import numpy as np


# Axial slice z-position → approximate anatomical level
# Assumes standard ABIDE-I NIfTI orientation (inferior → superior along z)
# These are heuristic ranges, not atlas-registered
_Z_REGIONS = [
    (0,   60,  "cerebellum / brainstem"),
    (60,  100, "inferior temporal / occipital"),
    (100, 140, "temporal lobe / basal ganglia"),
    (140, 180, "parietal / temporal"),
    (180, 220, "superior parietal / frontal"),
    (220, 256, "prefrontal / superior frontal"),
]

# GradCAM spatial CoM → approximate sub-region
# CoM is normalised (0=left/top, 1=right/bottom)
def _spatial_label(com_x: float, com_y: float) -> str:
    """
    Map normalised centre-of-mass (x, y) to a spatial descriptor.
    x: 0=left hemisphere, 1=right hemisphere
    y: 0=anterior, 1=posterior
    """
    side = "left" if com_x < 0.45 else ("right" if com_x > 0.55 else "bilateral/midline")
    pos  = "anterior" if com_y < 0.4 else ("posterior" if com_y > 0.6 else "central")
    return f"{pos} {side}"


def label_region(slice_idx: int, heatmap: np.ndarray) -> dict:
    """
    Given a slice z-index and GradCAM heatmap (224×224 float32),
    return anatomical region labels.

    Returns dict with:
        lobe       : approximate lobe/structure
        spatial    : spatial descriptor (anterior/posterior left/right)
        full_label : combined human-readable string
        com_x, com_y: normalised centre-of-mass
        peak_x, peak_y: normalised peak activation location
    """
    # Z-based lobe estimate
    lobe = "unknown region"
    for z_lo, z_hi, name in _Z_REGIONS:
        if z_lo <= slice_idx < z_hi:
            lobe = name
            break

    # Centre of mass of activation
    h, w = heatmap.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    total  = heatmap.sum() + 1e-9
    com_x  = float((xx * heatmap).sum() / total / w)   # normalised
    com_y  = float((yy * heatmap).sum() / total / h)

    # Peak location
    peak_flat  = np.argmax(heatmap)
    peak_y_px  = peak_flat // w
    peak_x_px  = peak_flat %  w
    peak_x     = float(peak_x_px / w)
    peak_y     = float(peak_y_px / h)

    spatial    = _spatial_label(com_x, com_y)
    energy     = float((heatmap > 0.5).mean())

    # If energy is very low, the heatmap is flat — don't make spatial claims
    if energy < 0.005:
        full_label = f"{lobe} (activation too diffuse for localisation)"
        spatial    = "diffuse / no focal activation"
    else:
        full_label = f"{lobe} — {spatial}"

    return {
        'lobe'      : lobe,
        'spatial'   : spatial,
        'full_label': full_label,
        'com_x'     : round(com_x, 3),
        'com_y'     : round(com_y, 3),
        'peak_x'    : round(peak_x, 3),
        'peak_y'    : round(peak_y, 3),
        'energy'    : round(energy, 4),
    }