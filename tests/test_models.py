"""Architecture smoke-tests — verifies output shapes and checkpoint-critical
attribute names without loading any weights.
"""
import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from models import (
    ASD_CNN, ASD_SkipCNN, ViT_PyTorch,
    ASDClassifierCNN, HybridCNNViT, build_model,
)

BATCH = 2


# ---------------------------------------------------------------------------
# Phase 1 models  (3-channel, 224×224)
# ---------------------------------------------------------------------------

def test_asd_cnn_output_shape():
    m = ASD_CNN(num_classes=2).eval()
    assert m(torch.randn(BATCH, 3, 224, 224)).shape == (BATCH, 2)


def test_asd_skipcnn_output_shape():
    m = ASD_SkipCNN(num_classes=2).eval()
    assert m(torch.randn(BATCH, 3, 224, 224)).shape == (BATCH, 2)


def test_vit_pytorch_output_shape():
    # Small config keeps CPU test fast
    m = ViT_PyTorch(
        img_size=64, patch_size=8, in_chans=3,
        embed_dim=64, depth=2, num_heads=2,
    ).eval()
    assert m(torch.randn(BATCH, 3, 64, 64)).shape == (BATCH, 2)


# ---------------------------------------------------------------------------
# Phase 2 models  (3-channel greyscale-to-RGB, 224×224)
# ---------------------------------------------------------------------------

def test_asd_classifier_cnn_output_shape():
    m = ASDClassifierCNN(num_classes=2).eval()
    assert m(torch.randn(BATCH, 3, 224, 224)).shape == (BATCH, 2)


def test_asd_classifier_cnn_attributes():
    """Checkpoint-critical attribute names must all be present."""
    m = ASDClassifierCNN()
    required = [
        "conv1", "conv2", "conv3", "conv4", "conv5",
        "lrelu1", "lrelu2", "lrelu3", "lrelu4", "lrelu5",
        "bn1", "bn2", "bn3", "bn4",
        "pool1", "pool2", "pool3", "pool4", "pool5",
        "drop1", "drop2", "drop3", "drop4", "drop5",
        "flatten", "fc1", "lrelu_fc", "fc2",
    ]
    for attr in required:
        assert hasattr(m, attr), f"ASDClassifierCNN missing attribute: {attr}"


def test_hybrid_cnn_vit_output_shape():
    m = HybridCNNViT(num_classes=2).eval()
    assert m(torch.randn(BATCH, 3, 224, 224)).shape == (BATCH, 2)


def test_hybrid_cnn_vit_attributes():
    """Checkpoint-critical attribute names must all be present."""
    m = HybridCNNViT()
    required = [
        "backbone", "token_proj", "pos_emb",
        "cls_tok", "cls_pe",
        "transformer", "norm", "head", "drop",
    ]
    for attr in required:
        assert hasattr(m, attr), f"HybridCNNViT missing attribute: {attr}"


def test_hybrid_get_cls_attn_shape():
    """get_cls_attn() returns head-averaged (B, 196) after a forward pass."""
    m = HybridCNNViT(num_classes=2).eval()
    _ = m(torch.randn(1, 3, 224, 224))
    attn = m.get_cls_attn()
    assert attn is not None, "get_cls_attn() returned None after forward pass"
    assert attn.shape == (1, 196), \
        f"Expected (1, 196), got {attn.shape}"


def test_hybrid_get_cls_attn_before_forward():
    """get_cls_attn() returns None before any forward pass."""
    m = HybridCNNViT(num_classes=2)
    assert m.get_cls_attn() is None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def test_build_model_all_names():
    configs = [
        ("asd_cnn",     {},                                            (3, 224, 224)),
        ("asd_skipcnn", {},                                            (3, 224, 224)),
        ("vit",         {"img_size": 64, "patch_size": 8,
                         "embed_dim": 64, "depth": 2, "num_heads": 2}, (3, 64,  64)),
        ("cnn",         {},                                            (3, 224, 224)),
        ("hybrid",      {},                                            (3, 224, 224)),
    ]
    for name, kw, shape in configs:
        m = build_model(name, **kw).eval()
        out = m(torch.randn(1, *shape))
        assert out.shape == (1, 2), f"{name}: expected (1,2), got {out.shape}"


def test_build_model_unknown_raises():
    with pytest.raises(ValueError):
        build_model("nonexistent_model")
