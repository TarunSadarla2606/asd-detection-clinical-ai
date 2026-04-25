"""Tests for compute_xai_agreement in app/xai.py.

Only tests the pure-numpy agreement metric — no model weights, no inference.
Importing xai.py triggers 'from model import ...' so app/ must be on sys.path.
"""
import sys
import os

import numpy as np
import pytest

# app/ must be first so Python finds model.py when xai.py does 'from model import ...'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))


def _rand(seed=0):
    return np.random.default_rng(seed).random((14, 14)).astype(np.float32)


# ---------------------------------------------------------------------------
# compute_xai_agreement
# ---------------------------------------------------------------------------

def test_identical_maps_high_iou():
    """Same heatmap for both inputs → IoU = 1.0."""
    from xai import compute_xai_agreement
    m = _rand(0)
    r = compute_xai_agreement(m, m.copy())
    assert r["iou"] > 0.90, f"Expected IoU > 0.90 for identical maps, got {r['iou']:.3f}"


def test_orthogonal_maps_near_zero_iou():
    """Non-overlapping hot regions → IoU near 0."""
    from xai import compute_xai_agreement
    gc   = np.zeros((14, 14), np.float32); gc[:7, :]  = 1.0
    attn = np.zeros((14, 14), np.float32); attn[7:, :] = 1.0
    r = compute_xai_agreement(gc, attn)
    assert r["iou"] < 0.05, f"Expected IoU < 0.05 for orthogonal maps, got {r['iou']:.3f}"


def test_output_keys_present():
    from xai import compute_xai_agreement
    r = compute_xai_agreement(_rand(1), _rand(2))
    for key in ["iou", "pearson_r", "gc_energy", "attn_energy",
                "agreement", "interpretation"]:
        assert key in r, f"Missing key: '{key}'"


def test_agreement_values():
    """agreement field must be one of the three allowed strings."""
    from xai import compute_xai_agreement
    r = compute_xai_agreement(_rand(3), _rand(4))
    assert r["agreement"] in ("high", "moderate", "low"), \
        f"Unexpected agreement value: '{r['agreement']}'"


def test_iou_in_unit_range():
    from xai import compute_xai_agreement
    r = compute_xai_agreement(_rand(5), _rand(6))
    assert 0.0 <= r["iou"] <= 1.0, f"IoU out of [0,1]: {r['iou']}"


def test_pearson_r_in_range():
    from xai import compute_xai_agreement
    r = compute_xai_agreement(_rand(7), _rand(8))
    assert -1.0 <= r["pearson_r"] <= 1.0, \
        f"Pearson r out of [-1,1]: {r['pearson_r']}"


def test_interpretation_is_string():
    from xai import compute_xai_agreement
    r = compute_xai_agreement(_rand(9), _rand(10))
    assert isinstance(r["interpretation"], str) and len(r["interpretation"]) > 0
