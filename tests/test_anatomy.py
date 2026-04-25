"""Tests for app/anatomy.py label_region.

Note: peak_x / peak_y are normalised [0,1], not pixel coordinates.
energy = fraction of heatmap pixels > 0.5.
"""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from anatomy import label_region


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat(val: float = 0.5, size: int = 224) -> np.ndarray:
    return np.full((size, size), val, dtype=np.float32)


def _hot(row: int, col: int, size: int = 224) -> np.ndarray:
    """Single bright pixel at (row, col) on black background."""
    arr = np.zeros((size, size), dtype=np.float32)
    arr[row, col] = 1.0
    return arr


# ---------------------------------------------------------------------------
# Return keys
# ---------------------------------------------------------------------------

def test_output_keys():
    r = label_region(100, _flat())
    for key in ["lobe", "spatial", "full_label",
                "com_x", "com_y", "peak_x", "peak_y", "energy"]:
        assert key in r, f"Missing key: '{key}'"


# ---------------------------------------------------------------------------
# Z-range lobe labels
# ---------------------------------------------------------------------------

def test_cerebellum_range():
    r = label_region(30, _flat())
    label = r["lobe"].lower()
    assert "cerebellum" in label or "brainstem" in label, \
        f"z=30 should map to cerebellum/brainstem, got: '{r['lobe']}'"


def test_prefrontal_range():
    r = label_region(240, _flat())
    label = r["lobe"].lower()
    assert "frontal" in label or "prefrontal" in label, \
        f"z=240 should map to prefrontal region, got: '{r['lobe']}'"


def test_mid_slice_lobe_is_string():
    r = label_region(120, _flat())
    assert isinstance(r["lobe"], str) and len(r["lobe"]) > 0


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------

def test_energy_zero_for_flat_05():
    # energy = fraction > 0.5; flat 0.5 is NOT > 0.5
    r = label_region(100, _flat(0.5))
    assert r["energy"] == 0.0


def test_energy_positive_for_hot_pixel():
    r = label_region(100, _hot(50, 75))
    assert r["energy"] > 0.0


# ---------------------------------------------------------------------------
# Peak coordinates are normalised [0, 1]
# ---------------------------------------------------------------------------

def test_peak_coords_normalised():
    r = label_region(100, _hot(50, 75))
    assert 0.0 <= r["peak_x"] <= 1.0, f"peak_x not in [0,1]: {r['peak_x']}"
    assert 0.0 <= r["peak_y"] <= 1.0, f"peak_y not in [0,1]: {r['peak_y']}"


def test_peak_x_matches_hot_column():
    size = 224
    col  = 75
    r = label_region(100, _hot(50, col, size))
    expected = col / size
    assert abs(r["peak_x"] - expected) < 0.01, \
        f"peak_x={r['peak_x']:.3f} expected ~{expected:.3f}"


def test_com_coords_normalised():
    r = label_region(100, _flat(0.8))
    assert 0.0 <= r["com_x"] <= 1.0
    assert 0.0 <= r["com_y"] <= 1.0
