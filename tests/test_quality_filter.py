"""Tests for src/quality_filter.py.

Does NOT call filter_csv — that requires a real CSV file.
All tests use synthetic uint8 numpy arrays (matching the function contract).
"""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from quality_filter import score_slice_quality, crop_brain_roi


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _brain_slice(size: int = 128) -> np.ndarray:
    """Synthetic brain-like slice: bright uint8 circle on dark background.

    score_slice_quality() divides by 255 internally, so input must be uint8.
    """
    arr = np.zeros((size, size), dtype=np.uint8)
    cx, cy, r = size // 2, size // 2, size // 3
    Y, X = np.ogrid[:size, :size]
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
    rng  = np.random.default_rng(42)
    # Values 128–153: well above all intensity thresholds once / 255
    arr[mask] = (128 + 25 * rng.random(mask.sum())).astype(np.uint8)
    return arr


# ---------------------------------------------------------------------------
# score_slice_quality
# ---------------------------------------------------------------------------

def test_valid_brain_slice_passes():
    result = score_slice_quality(_brain_slice(128))
    assert result["is_valid"] is True, \
        f"Expected valid, got rejected: {result['reason']}"


def test_blank_slice_rejected():
    result = score_slice_quality(np.zeros((64, 64), dtype=np.uint8))
    assert result["is_valid"] is False


def test_result_has_all_keys():
    result = score_slice_quality(_brain_slice())
    for key in ["is_valid", "quality_score", "reason",
                "mean_i", "brain_frac", "std_i", "lap_var", "circularity"]:
        assert key in result, f"Missing key in result: '{key}'"


def test_quality_score_in_unit_range():
    score = score_slice_quality(_brain_slice())["quality_score"]
    assert 0.0 <= score <= 1.0, f"quality_score out of [0,1]: {score}"


def test_mean_intensity_positive_for_brain():
    result = score_slice_quality(_brain_slice())
    assert result["mean_i"] > 0.0


def test_brain_frac_positive_for_brain():
    result = score_slice_quality(_brain_slice())
    assert result["brain_frac"] > 0.0


# ---------------------------------------------------------------------------
# crop_brain_roi
# ---------------------------------------------------------------------------

def test_crop_brain_roi_smaller_or_equal():
    arr     = _brain_slice(128)
    cropped = crop_brain_roi(arr)
    assert cropped.shape[0] <= 128
    assert cropped.shape[1] <= 128


def test_crop_brain_roi_nonempty():
    arr     = _brain_slice(128)
    cropped = crop_brain_roi(arr)
    assert cropped.size > 0


def test_crop_brain_roi_preserves_dtype():
    # crop_brain_roi returns a numpy slice of the input — same dtype
    arr     = _brain_slice(128)
    cropped = crop_brain_roi(arr)
    assert cropped.dtype == arr.dtype
