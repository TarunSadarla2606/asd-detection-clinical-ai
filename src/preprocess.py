"""
preprocess.py
-------------
Data preprocessing pipeline for ASD detection from ABIDE-I sMRI data.

Pipeline:
    NIfTI (.nii.gz) → DICOM → PNG slices → cleaning → CSV label files

Starting dataset: 1,112 ABIDE-I subjects
After cleaning:   1,067 subjects, 100,510 usable PNG slices

Two preprocessing variants are implemented:
    - CNN:  resize → 3-channel → mean/std normalization
    - ViT:  Canny edge crop → resize → 3-channel → min-max norm → patchify

Usage:
    python preprocess.py --input_dir /path/to/png_slices \
                         --csv_dir   /path/to/csv_files \
                         --mode      cnn
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────
IMG_SIZE = (224, 224)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PATCH_SIZE    = 16  # ViT patch size → 196 patches per 224×224 image


# ── CNN Preprocessing ──────────────────────────────────────────────

def preprocess_cnn(image_path: str) -> np.ndarray:
    """
    CNN-specific preprocessing:
        1. Read DICOM/PNG slice (grayscale)
        2. Resize to 224×224
        3. Convert single-channel to 3-channel (RGB)
        4. Mean/std normalization (ImageNet statistics)

    Returns: np.ndarray of shape (224, 224, 3), float32, normalized
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img = cv2.resize(img, IMG_SIZE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    img_norm = (img_rgb - IMAGENET_MEAN) / (IMAGENET_STD + 1e-8)
    return img_norm


# ── ViT Preprocessing ───────────────────────────────────────────────

def edge_based_crop(image: np.ndarray) -> np.ndarray:
    """
    Apply Canny edge detection and crop to the bounding box of the
    largest contour (brain boundary). Removes empty background.
    """
    edges = cv2.Canny(image, threshold1=30, threshold2=100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return image  # fallback: no crop
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    # Add small padding (5px) on each side
    pad = 5
    x = max(0, x - pad);  y = max(0, y - pad)
    w = min(image.shape[1] - x, w + 2*pad)
    h = min(image.shape[0] - y, h + 2*pad)
    return image[y:y+h, x:x+w]


def preprocess_vit(image_path: str) -> np.ndarray:
    """
    ViT-specific preprocessing:
        1. Read slice (grayscale)
        2. Canny edge detection + bounding-box crop (remove background)
        3. Resize to 224×224
        4. Convert to 3-channel
        5. Min-max normalization to [0, 1]

    Returns: np.ndarray of shape (224, 224, 3), float32
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img = edge_based_crop(img)
    img = cv2.resize(img, IMG_SIZE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.float32)

    # Min-max normalization
    mn, mx = img_rgb.min(), img_rgb.max()
    if mx - mn > 1e-6:
        img_rgb = (img_rgb - mn) / (mx - mn)
    return img_rgb


def patchify(image_tensor, patch_size: int = PATCH_SIZE):
    """
    Divide a (C, H, W) image tensor into non-overlapping patches.

    Returns: tensor of shape (num_patches, C, patch_size, patch_size)
             For 224×224 with patch_size=16: (196, C, 16, 16)
    """
    import torch
    _, H, W = image_tensor.shape
    patches = []
    for i in range(0, H - patch_size + 1, patch_size):
        for j in range(0, W - patch_size + 1, patch_size):
            patches.append(image_tensor[:, i:i+patch_size, j:j+patch_size])
    return torch.stack(patches)  # (196, C, 16, 16)


# ── Dataset Statistics ──────────────────────────────────────────────

def dataset_info():
    """Print ABIDE-I dataset statistics after cleaning."""
    print("ABIDE-I Dataset Statistics")
    print("-" * 40)
    print(f"Original subjects:              1,112")
    print(f"Removed (missing/unclear sMRI):    19")
    print(f"Removed (Unknown label):           26")
    print(f"Final subjects after cleaning:  1,067")
    print(f"Total DICOM slices (raw):     274,871")
    print(f"Usable slices after filtering: 100,510")
    print()
    print("Train/Val/Test splits (slice-level):")
    print(f"  Full dataset:  Train 72,367 | Val 8,041 | Test 20,102")
    print(f"  Half dataset:  Train 40,431 | Val 8,041 | Test  5,054")
    print(f"  Small subset:  Train  2,400 | Val   300 | Test    300")


if __name__ == "__main__":
    dataset_info()
