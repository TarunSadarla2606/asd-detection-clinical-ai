"""
src/
----
Reusable Python modules for ASD detection from sMRI.

Modules:
    models         — CNN architectures (5-layer plain + skip-connected)
    quality_filter — 4-metric slice quality filter (new in graduate system)
    dataset        — PyTorch Dataset class and DataLoader factory
    preprocess     — CNN / ViT preprocessing pipelines
    train          — Training loop with early stopping
    evaluate       — Metrics, ROC curve, confusion matrix, training curves
"""
