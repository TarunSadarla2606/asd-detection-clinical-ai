# pipeline.py
# Full subject-level pipeline: NIfTI → slices → filter → predict → aggregate

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

# ── Normalisation stats from training ─────────────────────────────────────
# These were computed from the ABIDE-I training set
MEAN = [0.1290, 0.1290, 0.1290]
STD  = [0.1741, 0.1741, 0.1741]

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])


# ── Step 1: NIfTI → list of 2D axial slices as numpy arrays ───────────────

def load_nifti_slices(nifti_path: str) -> list[np.ndarray]:
    """
    Load a NIfTI file and return all axial slices as a list of
    uint8 grayscale numpy arrays (H, W).
    """
    import nibabel as nib

    nii  = nib.load(nifti_path)
    data = nii.get_fdata()

    # Ensure we have a 3D volume
    if data.ndim == 4:
        data = data[:, :, :, 0]   # take first volume if 4D
    if data.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {data.shape}")

    slices = []
    n_z = data.shape[2]

    for z in range(n_z):
        sl = data[:, :, z].astype(float)
        max_val = sl.max()
        if max_val <= 0:
            # Completely blank — keep as zeros, quality filter will drop it
            slices.append(np.zeros(sl.shape, dtype=np.uint8))
        else:
            rescaled = (np.maximum(sl, 0) / max_val) * 255
            slices.append(np.uint8(rescaled))

    return slices   # list of (H, W) uint8 arrays, length = n_z


# ── Step 2: Brain ROI crop ─────────────────────────────────────────────────

def crop_brain_roi(arr: np.ndarray, padding: int = 12) -> np.ndarray:
    """
    Crop the black background from a grayscale MRI slice.
    Finds the bounding box of the brain region and crops to it.
    Returns cropped array (may be non-square).
    Falls back to original if no brain region found.
    """
    _, binary = cv2.threshold(arr, int(0.05 * 255), 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return arr

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Add padding, clamp to image bounds
    H, W = arr.shape
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(W, x + w + padding)
    y2 = min(H, y + h + padding)

    cropped = arr[y1:y2, x1:x2]
    return cropped if cropped.size > 0 else arr


# ── Step 3: Slice quality filter ──────────────────────────────────────────

def score_slice_quality(arr: np.ndarray) -> dict:
    """
    Compute quality metrics for one MRI slice.
    Returns dict with is_valid (bool), quality_score (float), and metrics.
    Higher quality_score = better candidate for XAI visualisation.
    """
    arr_f = arr.astype(np.float32) / 255.0

    mean_i     = float(arr_f.mean())
    brain_frac = float((arr_f > 0.05).mean())
    std_i      = float(arr_f.std())
    lap_var    = float(cv2.Laplacian(arr, cv2.CV_64F).var())

    # Hard rejection rules
    if mean_i     < 0.04:                          return {'is_valid': False, 'quality_score': 0.0, 'reason': 'blank'}
    if brain_frac > 0.75 and lap_var > 500:        return {'is_valid': False, 'quality_score': 0.0, 'reason': 'noise'}
    if lap_var    > 8000:                          return {'is_valid': False, 'quality_score': 0.0, 'reason': 'high noise'}
    if brain_frac < 0.12:                          return {'is_valid': False, 'quality_score': 0.0, 'reason': 'low coverage'}
    if std_i      < 0.03:                          return {'is_valid': False, 'quality_score': 0.0, 'reason': 'uniform'}

    # Circularity check
    _, binary = cv2.threshold(arr, int(0.08 * 255), 255, cv2.THRESH_BINARY)
    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed    = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {'is_valid': False, 'quality_score': 0.0, 'reason': 'no contour'}

    largest = max(contours, key=cv2.contourArea)
    area    = cv2.contourArea(largest)
    perim   = cv2.arcLength(largest, True)
    if area < 500 or perim < 1:
        return {'is_valid': False, 'quality_score': 0.0, 'reason': 'tiny ROI'}

    circularity = (4 * np.pi * area) / (perim ** 2)
    roi_frac    = area / (arr.shape[0] * arr.shape[1])

    if circularity < 0.45:
        return {'is_valid': False, 'quality_score': 0.0, 'reason': f'non-circular ({circularity:.2f})'}
    if roi_frac < 0.10 or roi_frac > 0.80:
        return {'is_valid': False, 'quality_score': 0.0, 'reason': f'bad ROI size ({roi_frac:.2f})'}

    # Quality score: higher is better for XAI selection
    # Rewards good brain coverage, sharpness, and circularity
    quality_score = (
        circularity * 0.4 +
        min(brain_frac / 0.5, 1.0) * 0.3 +
        min(std_i / 0.2, 1.0) * 0.2 +
        min(lap_var / 1000, 1.0) * 0.1
    )

    return {
        'is_valid'      : True,
        'quality_score' : float(quality_score),
        'circularity'   : circularity,
        'brain_frac'    : brain_frac,
        'lap_var'       : lap_var,
        'reason'        : 'ok'
    }


# ── Step 4: Inference on a single slice ───────────────────────────────────

def predict_slice(arr: np.ndarray, model, device: str = 'cpu') -> dict:
    """
    Run the CNN on one grayscale slice array.
    Returns dict with prob_asd (float), prob_tc (float), pred_class (int).
    """
    pil = Image.fromarray(arr).convert('RGB')
    t   = EVAL_TRANSFORM(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(t)
        probs  = F.softmax(logits, dim=1)[0].cpu().numpy()

    return {
        'prob_tc'   : float(probs[0]),
        'prob_asd'  : float(probs[1]),
        'pred_class': int(np.argmax(probs))   # 0=TC, 1=ASD
    }


# ── Step 5: Full subject pipeline ─────────────────────────────────────────

def run_subject_pipeline(
    nifti_path : str,
    model,
    device      : str = 'cpu',
    top_k       : int = 5,
    progress_fn = None    # optional callback(message, fraction) for UI progress
) -> dict:
    """
    Full pipeline from NIfTI file to subject-level prediction.

    Returns:
        subject_pred   : 'ASD' or 'TC'
        subject_conf   : float — weighted confidence (0–1)
        pred_asd_votes : int — number of slices predicting ASD
        pred_tc_votes  : int — number of slices predicting TC
        total_valid    : int — total slices that passed quality filter
        top_slices     : list of dicts, each with:
                            'slice_idx', 'arr' (original uint8),
                            'arr_cropped' (brain-cropped uint8),
                            'prob_asd', 'quality_score', 'pred_class'
    """
    def _progress(msg, frac):
        if progress_fn:
            progress_fn(msg, frac)

    # 1. Load slices
    _progress("Loading NIfTI volume...", 0.05)
    slices = load_nifti_slices(nifti_path)
    n_total = len(slices)
    _progress(f"Loaded {n_total} axial slices", 0.15)

    # 2. Quality filter all slices
    _progress("Filtering slice quality...", 0.20)
    valid_slices = []
    for i, arr in enumerate(slices):
        q = score_slice_quality(arr)
        if q['is_valid']:
            valid_slices.append({
                'slice_idx'    : i,
                'arr'          : arr,
                'arr_cropped'  : crop_brain_roi(arr),
                'quality_score': q['quality_score'],
            })

    if not valid_slices:
        return {'error': 'No valid slices found in this NIfTI file.'}

    _progress(f"{len(valid_slices)}/{n_total} slices passed quality filter", 0.35)

    # 3. Run inference on all valid slices
    _progress("Running CNN inference on valid slices...", 0.40)
    for vs in valid_slices:
        pred = predict_slice(vs['arr'], model, device)
        vs.update(pred)

    _progress("Inference complete", 0.70)

    # 4. Subject-level aggregation
    # Weighted average: weight = quality_score × confidence
    prob_asd_values = [vs['prob_asd'] for vs in valid_slices]
    weights         = [vs['quality_score'] * max(vs['prob_asd'], vs['prob_tc'])
                       for vs in valid_slices]
    total_weight    = sum(weights)

    if total_weight > 0:
        weighted_prob_asd = sum(p * w for p, w in zip(prob_asd_values, weights)) / total_weight
    else:
        weighted_prob_asd = float(np.mean(prob_asd_values))

    pred_asd_votes = sum(1 for vs in valid_slices if vs['pred_class'] == 1)
    pred_tc_votes  = len(valid_slices) - pred_asd_votes

    subject_pred = 'ASD' if weighted_prob_asd >= 0.5 else 'TC'
    subject_conf = weighted_prob_asd if subject_pred == 'ASD' else (1 - weighted_prob_asd)

    # 5. Select top-K slices for XAI (highest quality_score among valid)
    top_slices = sorted(valid_slices, key=lambda x: x['quality_score'], reverse=True)[:top_k]

    _progress("Pipeline complete", 1.0)

    return {
        'subject_pred'  : subject_pred,
        'subject_conf'  : round(subject_conf, 4),
        'weighted_prob_asd': round(weighted_prob_asd, 4),
        'pred_asd_votes': pred_asd_votes,
        'pred_tc_votes' : pred_tc_votes,
        'total_valid'   : len(valid_slices),
        'n_total_slices': n_total,
        'top_slices'    : top_slices,
    }