# Phase 1 — B.Tech Capstone Notebooks (2023)

The complete set of B.Tech training notebooks is preserved in the original repository:

**https://github.com/TarunSadarla2606/asd-detection-neuroimaging**

---

## Notebooks in original repo

| Notebook | Architecture | Optimizer | Test Acc | AUC |
|---|---|---|---|---|
| `cnn_adam.ipynb` | 5-layer CNN | Adam | 0.90 | 0.97 |
| `cnn_nadam.ipynb` | 5-layer CNN | NAdam | 0.91 | 0.97 |
| `cnn_nadam_and_rmsprop.ipynb` | 5-layer CNN | NAdam + RMSprop | 0.91 | 0.97 |
| `cnn_skip_adam.ipynb` | Skip-CNN | Adam | 0.90 | 0.97 |
| `cnn_skip_nadam.ipynb` | Skip-CNN | NAdam | 0.91 | **0.98** |
| `cnn_skip_rmsprop.ipynb` | Skip-CNN | RMSprop | 0.91 | 0.97 |
| `vit_keras_rmsprop.ipynb` | Custom ViT (Keras) | RMSprop | 0.62 | 0.62 |
| `vit_pytorch_full.ipynb` | Custom ViT (PyTorch) | — | — | — |
| `vit_pytorch_compact.ipynb` | Custom ViT (PyTorch) | — | — | — |
| `vit_pretrained_timm.ipynb` | ViT-B/16 (timm) | — | — | — |
| `vit_pretrained_huggingface.ipynb` | google/vit-base-patch16 | — | — | — |
| `lime_explainability.ipynb` | LIME on CNN | — | — | — |

## What the B.Tech work established

- CNNs consistently outperform from-scratch ViTs on ABIDE-I at this data scale (AUC 0.97–0.98 vs 0.62)
- Skip connections provide marginal but consistent AUC improvement (0.97 → 0.98 with NAdam)
- Optimizer choice (Adam/NAdam/RMSprop) has smaller impact than architecture selection
- LIME visualization pipeline partial — explainer invoked but final annotated output incomplete
- GradCAM explored in development but not retained in notebooks
- Doubling training subjects (533 → 1,067) produced consistent metric improvement

## Why these are not duplicated here

The B.Tech notebooks train on *unfiltered* raw slices using ImageNet normalisation statistics — different from the graduate system which uses quality-filtered data and ABIDE-I statistics. Mixing them in the same repository would create confusion. The original repo is preserved exactly as submitted.

This repository picks up where the B.Tech work ended — the open threads listed in the 2023 README (GradCAM, LIME completion, subgroup analysis, clinical deployment).
