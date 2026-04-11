"""
quality_filter.py
-----------------
4-metric slice quality filter for ABIDE-I sMRI axial slices.

This module implements the quality filtering pipeline developed for the
graduate clinical AI system (2026). It rejects low-quality slices
before CNN inference to improve model performance and XAI reliability.

The filter removes ~6.5% of slices consistently across train/val/test splits,
improving AUC from 0.97 (B.Tech baseline) to 0.994 (graduate system).

Filtering criteria:
    1. Mean intensity     > 0.04    — blank slice rejection
    2. Brain coverage     > 0.12    — low-coverage rejection
    3. Pixel std dev      > 0.03    — uniform/featureless rejection
    4. Laplacian variance > 20.0    — blur rejection
    5. Contour circularity >= 0.45  — non-brain shape rejection

Usage:
    from quality_filter import score_slice_quality, filter_csv

    # Single slice
    metrics = score_slice_quality(arr_uint8)
    if metrics['is_valid']:
        # use slice

    # Whole CSV
    clean_df = filter_csv('extracted_random_labels_train.csv',
                          path_prefix='/kaggle/input/autism/')
    clean_df.to_csv('clean_train.csv', index=False)
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


# ── Rejection thresholds ───────────────────────────────────────────────────
# Derived empirically from 500-slice quality distribution analysis.
# See: notebooks/phase2_graduate_2026/01_training_xai_gradcam_lime.ipynb Section 1B

THRESHOLD_MEAN_INTENSITY   = 0.04   # fraction of max; below = blank
THRESHOLD_BRAIN_COVERAGE   = 0.12   # fraction of pixels > 0.05; below = low coverage
THRESHOLD_STD_DEV          = 0.03   # pixel std (normalised); below = uniform
THRESHOLD_LAPLACIAN_VAR    = 20.0   # Laplacian variance; below = too blurry
THRESHOLD_CIRCULARITY      = 0.45   # 4πA/P²; below = non-circular (not brain)
THRESHOLD_ROI_MIN          = 0.10   # min fraction of image area; below = tiny ROI
THRESHOLD_ROI_MAX          = 0.80   # max fraction of image area; above = full coverage / noise


def score_slice_quality(arr: np.ndarray) -> dict:
    """
    Compute quality metrics for a single MRI slice and determine validity.

    Args:
        arr: 2D uint8 numpy array (grayscale MRI slice, values 0–255)

    Returns:
        dict with keys:
            is_valid      (bool)    — True if slice passes all filters
            quality_score (float)  — 0–1 composite score (higher = better)
            reason        (str)    — 'ok' or rejection reason
            mean_i        (float)  — mean normalised intensity
            brain_frac    (float)  — fraction of pixels above threshold
            std_i         (float)  — pixel std dev (normalised)
            lap_var       (float)  — Laplacian variance
            circularity   (float)  — contour circularity (0=line, 1=circle)
    """
    arr_f = arr.astype(np.float32) / 255.0

    mean_i     = float(arr_f.mean())
    brain_frac = float((arr_f > 0.05).mean())
    std_i      = float(arr_f.std())
    lap_var    = float(cv2.Laplacian(arr, cv2.CV_64F).var())

    # ── Hard rejection rules ───────────────────────────────────────────────
    if mean_i < THRESHOLD_MEAN_INTENSITY:
        return _reject('blank', mean_i, brain_frac, std_i, lap_var)

    if brain_frac < THRESHOLD_BRAIN_COVERAGE:
        return _reject('low coverage', mean_i, brain_frac, std_i, lap_var)

    if std_i < THRESHOLD_STD_DEV:
        return _reject('uniform', mean_i, brain_frac, std_i, lap_var)

    if lap_var > 8000:
        return _reject('high noise', mean_i, brain_frac, std_i, lap_var)

    if brain_frac > 0.75 and lap_var > 500:
        return _reject('noise/artefact', mean_i, brain_frac, std_i, lap_var)

    # ── Circularity / contour check ────────────────────────────────────────
    _, binary = cv2.threshold(arr, int(0.08 * 255), 255, cv2.THRESH_BINARY)
    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed    = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return _reject('no contour', mean_i, brain_frac, std_i, lap_var)

    largest = max(contours, key=cv2.contourArea)
    area    = cv2.contourArea(largest)
    perim   = cv2.arcLength(largest, True)

    if area < 500 or perim < 1:
        return _reject('tiny ROI', mean_i, brain_frac, std_i, lap_var)

    circularity = (4 * np.pi * area) / (perim ** 2)
    roi_frac    = area / (arr.shape[0] * arr.shape[1])

    if circularity < THRESHOLD_CIRCULARITY:
        return _reject(f'non-circular ({circularity:.2f})',
                       mean_i, brain_frac, std_i, lap_var, circularity)

    if roi_frac < THRESHOLD_ROI_MIN:
        return _reject(f'ROI too small ({roi_frac:.2f})',
                       mean_i, brain_frac, std_i, lap_var, circularity)

    if roi_frac > THRESHOLD_ROI_MAX:
        return _reject(f'ROI too large ({roi_frac:.2f})',
                       mean_i, brain_frac, std_i, lap_var, circularity)

    if lap_var < THRESHOLD_LAPLACIAN_VAR:
        return _reject('blurry', mean_i, brain_frac, std_i, lap_var, circularity)

    # ── Quality score (higher = better for XAI slice selection) ───────────
    # Weights: circularity (0.4) + brain coverage (0.3) + sharpness (0.2) + std (0.1)
    quality_score = (
        circularity               * 0.4 +
        min(brain_frac / 0.5, 1.) * 0.3 +
        min(lap_var / 1000., 1.)  * 0.2 +
        min(std_i / 0.2, 1.)      * 0.1
    )

    return {
        'is_valid'      : True,
        'quality_score' : float(quality_score),
        'reason'        : 'ok',
        'mean_i'        : round(mean_i, 4),
        'brain_frac'    : round(brain_frac, 4),
        'std_i'         : round(std_i, 4),
        'lap_var'       : round(lap_var, 2),
        'circularity'   : round(circularity, 4),
    }


def _reject(reason: str, mean_i: float, brain_frac: float,
            std_i: float, lap_var: float,
            circularity: float = 0.0) -> dict:
    return {
        'is_valid'      : False,
        'quality_score' : 0.0,
        'reason'        : reason,
        'mean_i'        : round(mean_i, 4),
        'brain_frac'    : round(brain_frac, 4),
        'std_i'         : round(std_i, 4),
        'lap_var'       : round(lap_var, 2),
        'circularity'   : round(circularity, 4),
    }


def filter_csv(
    csv_path: str,
    path_prefix: str = '/kaggle/input/autism/',
    local_prefix: str = 'E:\\TARUN\\Projects\\Autism Detection\\Data\\data_png',
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Apply quality filter to all slices in a label CSV file.

    Reads each PNG slice, computes quality metrics, and returns a DataFrame
    containing only valid slices.

    Args:
        csv_path:     Path to label CSV (columns: index, Image_path, Image_name, LABEL)
        path_prefix:  Runtime path prefix for PNG files
        local_prefix: Windows development path to replace
        verbose:      Print progress summary

    Returns:
        pd.DataFrame with only valid slices (same columns as input CSV)

    Example:
        clean_df = filter_csv('extracted_random_labels_train.csv')
        clean_df.to_csv('clean_train.csv', index=False)
    """
    df = pd.read_csv(csv_path)
    n_original = len(df)

    valid_indices = []

    for idx, row in df.iterrows():
        img_path = str(row.iloc[1])
        img_path = img_path.replace(local_prefix, path_prefix).replace('\\', '/')

        arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if arr is None:
            continue

        metrics = score_slice_quality(arr)
        if metrics['is_valid']:
            valid_indices.append(idx)

    clean_df = df.loc[valid_indices].reset_index(drop=True)

    if verbose:
        n_removed = n_original - len(clean_df)
        pct = n_removed / n_original * 100
        print(f"Quality filter: {n_original:,} → {len(clean_df):,} slices "
              f"({n_removed:,} removed, {pct:.1f}%)")

    return clean_df


def crop_brain_roi(arr: np.ndarray, padding: int = 12) -> np.ndarray:
    """
    Crop the black background from a grayscale MRI slice using contour detection.

    Used before XAI visualisation to eliminate background superpixels
    in LIME explanations.

    Args:
        arr:     2D uint8 numpy array
        padding: Pixels of padding to add around the detected ROI

    Returns:
        Cropped array. Falls back to original if no contour found.
    """
    _, binary = cv2.threshold(arr, int(0.05 * 255), 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return arr

    largest  = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    H, W     = arr.shape
    x1 = max(0, x - padding);       y1 = max(0, y - padding)
    x2 = min(W, x + w + padding);   y2 = min(H, y + h + padding)

    cropped = arr[y1:y2, x1:x2]
    return cropped if cropped.size > 0 else arr


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python quality_filter.py <csv_path> [path_prefix]")
        sys.exit(1)

    csv_path    = sys.argv[1]
    path_prefix = sys.argv[2] if len(sys.argv) > 2 else '/kaggle/input/autism/'

    clean = filter_csv(csv_path, path_prefix=path_prefix)
    out   = csv_path.replace('.csv', '_clean.csv')
    clean.to_csv(out, index=False)
    print(f"Saved clean CSV → {out}")
