# ASD Detection — Clinical AI System

> Upgrading a 2023 B.Tech capstone into a production-grade neuroimaging AI:  
> NIfTI-to-prediction pipeline · Hybrid CNN-ViT architecture · Grad-CAM + LIME explainability · MC-Dropout uncertainty · LLM clinical narrative · Deployed on Hugging Face Spaces.

[![CI](https://github.com/TarunSadarla2606/asd-detection-clinical-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/TarunSadarla2606/asd-detection-clinical-ai/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/Live_Demo-Hugging_Face_Spaces-FF9900?logo=huggingface&logoColor=white)](https://huggingface.co/spaces/TarunSadarla2606/asd-detection-demo)
[![Original B.Tech Repo](https://img.shields.io/badge/B.Tech_2023-Original_Capstone-9d7fe8?logo=github)](https://github.com/TarunSadarla2606/asd-detection-neuroimaging)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-latest-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [What's New vs B.Tech Baseline](#whats-new-vs-btech-baseline)
- [Live Demo](#live-demo)
- [Model Performance](#model-performance)
- [Repository Structure](#repository-structure)
- [Pipeline Architecture](#pipeline-architecture)
- [XAI Analysis](#xai-analysis)
- [Subgroup Analysis](#subgroup-analysis)
- [Deployment](#deployment)
- [Running Locally](#running-locally)
- [Clinical & Regulatory Context](#clinical--regulatory-context)
- [Relationship to B.Tech Work](#relationship-to-btech-work)
- [Author](#author)

---

## Overview

This repository is the **graduate-level continuation** of my [2023 B.Tech capstone](https://github.com/TarunSadarla2606/asd-detection-neuroimaging) — which compared CNNs and Vision Transformers for ASD detection from structural MRI and established that CNNs significantly outperform from-scratch ViTs on ABIDE-I at this data scale (AUC 0.97–0.98 vs 0.62).

That work answered: *"Can a CNN detect ASD from sMRI slices?"*

This repository answers: **"What does it take to go from a model that works in a notebook to a system a clinician could actually interact with?"**

The answer involves: a proper inference pipeline that accepts clinical-format inputs, a Hybrid CNN-ViT architecture that combines spatial feature extraction with global attention, explainability methods that show *why* the model predicts what it predicts, uncertainty quantification that flags when it's guessing, subgroup analysis that documents where it fails, and a governance framework that honestly states what would be required for clinical deployment.

---

## What's New vs B.Tech Baseline

| Capability | 2023 B.Tech Baseline | 2026 Clinical AI System |
|---|---|---|
| **Input format** | Pre-extracted PNG slices | Raw NIfTI volumes (.nii / .nii.gz) |
| **Data quality** | Raw slices including blanks and noise | 4-metric quality filter — ~6.5% removed |
| **Model architecture** | 5-layer CNN | CNN (AUC 0.994) + Hybrid CNN-ViT (AUC 0.997) |
| **Evaluation** | Accuracy + AUC | + AUPRC, Brier score, calibration, threshold analysis |
| **Explainability** | LIME attempt (visualization incomplete) | Grad-CAM + LIME + spatial agreement (Pearson r, IoU) |
| **Uncertainty** | None | MC-Dropout (30 stochastic passes) |
| **Subgroup analysis** | None | Sex-stratified + 7-site stratified performance |
| **Failure mode analysis** | None | GradCAM on FP/FN cases, failure log, causal analysis |
| **Clinical output** | None | LLM narrative (Claude Haiku), PDF report, anatomical region labelling |
| **Deployment** | Kaggle notebook | Streamlit app on Hugging Face Spaces (always-on) |
| **Governance** | None | Model Card, regulatory framing (FDA SaMD Class II), fairness documentation |
| **Automated testing** | None | 36 pytest unit tests, CI on every push (GitHub Actions) |

---

## Live Demo

**[https://huggingface.co/spaces/TarunSadarla2606/asd-detection-demo](https://huggingface.co/spaces/TarunSadarla2606/asd-detection-demo)**

Upload any NIfTI structural MRI file (.nii or .nii.gz). The system will:

1. Extract all axial slices from the 3D volume
2. Apply a 4-metric quality filter (blank/blurry/low-coverage/non-circular rejection)
3. Run CNN inference on all valid slices
4. Aggregate to a **subject-level prediction** via quality-weighted confidence voting
5. Generate Grad-CAM + LIME explanations on the top-K highest-quality slices
6. Estimate prediction uncertainty via MC-Dropout (30 passes)
7. Look up phenotypic metadata and site reliability from ABIDE-I
8. Generate an LLM-written clinical interpretation (Claude Haiku)
9. Produce a downloadable **PDF clinical report**

Four demo subjects from ABIDE-I are pre-loaded: `32016` (TC), `32067` (ASD), `32152` (ASD), `32164` (TC).

---

## Model Performance

All results on the held-out test set: **18,814 quality-filtered axial slices** from ABIDE-I.

> Note: These are slice-level metrics. The deployed system uses subject-level aggregation — see [Pipeline Architecture](#pipeline-architecture).

### CNN (Deployed Model)

| Metric | Value | Clinical Relevance |
|---|---|---|
| **AUC-ROC** | **0.994** | Near-perfect discrimination between ASD and TC at slice level |
| **AUPRC** | **0.994** | Robust performance under class imbalance |
| **Sensitivity** | **95.6%** | 95.6% of ASD slices correctly identified |
| **Specificity** | **97.2%** | 97.2% of TC slices correctly cleared |
| **Precision (PPV)** | **97.1%** | When model predicts ASD, 97.1% correct |
| **F1 Score** | **96.3%** | Harmonic mean of precision and recall |
| **Brier Score** | **0.027** | Well-calibrated probabilities |
| **False Negative Rate** | **4.4%** | Clinically most costly error: ASD missed |

### Model Comparison

| Model | AUC-ROC | Sensitivity | Specificity | Parameters | Notes |
|---|---|---|---|---|---|
| B.Tech 2023 CNN | 0.97 | ~90% | ~91% | 1.6M | Pre-filtered data, NAdam |
| **ASDClassifierCNN** (deployed) | **0.994** | **95.6%** | **97.2%** | 1.6M | Quality-filtered training |
| **HybridCNNViT** (research) | **0.997** | — | — | ~4.2M | CNN backbone + 4-block Transformer |

The +0.003 AUC gain from CNN → Hybrid CNN-ViT comes from the Transformer's ability to model long-range spatial dependencies across the full axial slice — complementing the CNN's local texture features.

---

## Repository Structure

```
asd-detection-clinical-ai/
│
├── README.md                        ← This file
├── CHANGELOG.md                     ← B.Tech → Graduate upgrade history
├── MODEL_CARD.md                    ← Full governance documentation
├── LICENSE
├── .gitignore
├── requirements.txt                 ← Full dependency list
├── packages.txt                     ← System deps for HF Spaces (libgl1)
│
├── .github/
│   ├── workflows/
│   │   └── ci.yml                   ← GitHub Actions CI (pytest, CPU PyTorch)
│   └── ISSUE_TEMPLATE/
│       └── bug_report.md
│
├── app/                             ← Streamlit application
│   ├── app.py                       ← Main entry point
│   ├── model.py                     ← CNN + Hybrid CNN-ViT + load_model()
│   ├── pipeline.py                  ← NIfTI → slices → filter → predict → aggregate
│   ├── xai.py                       ← Grad-CAM + LIME + MC-Dropout
│   ├── anatomy.py                   ← Heuristic anatomical region labelling
│   ├── narrator.py                  ← LLM narrative via Claude Haiku API
│   ├── phenotypic.py                ← ABIDE-I phenotypic CSV + site mapping
│   └── report.py                    ← PDF report generation (ReportLab)
│
├── src/                             ← Reusable Python modules (non-app)
│   ├── __init__.py
│   ├── models.py                    ← All architectures: CNN, SkipCNN, ViT, HybridCNNViT
│   ├── quality_filter.py            ← 4-metric slice quality filter
│   ├── dataset.py                   ← PyTorch Dataset + DataLoader factory
│   ├── preprocess.py                ← CNN / ViT preprocessing pipelines
│   ├── train.py                     ← Training loop with early stopping
│   └── evaluate.py                  ← Metrics, ROC curve, confusion matrix
│
├── tests/                           ← Automated test suite (36 tests, CI-verified)
│   ├── __init__.py
│   ├── test_models.py               ← Architecture smoke-tests (no weights needed)
│   ├── test_quality_filter.py       ← Quality filter unit tests
│   ├── test_xai_utils.py            ← XAI agreement metric tests
│   └── test_anatomy.py              ← Anatomical labelling tests
│
├── notebooks/                       ← Kaggle analysis notebooks
│   ├── README.md
│   ├── phase1_btech_2023/
│   └── phase2_graduate_2026/
│       ├── 01_training_xai_gradcam_lime.ipynb
│       └── 02_xai_analysis_full.ipynb
│
├── configs/                         ← Training configuration files
│   ├── cnn_nadam.yaml               ← Best-performing CNN config
│   ├── cnn_adam.yaml
│   ├── cnn_rmsprop.yaml
│   ├── skip_cnn_nadam.yaml
│   └── hybrid_cnn_vit.yaml          ← Hybrid CNN-ViT training config
│
├── results/                         ← All experimental outputs
│   ├── README.md
│   ├── experiment_results.csv
│   ├── phase2_metrics.csv
│   ├── hybrid_metrics.csv           ← CNN vs Hybrid CNN-ViT comparison
│   ├── subgroup_sex.csv
│   ├── subgroup_site.csv
│   └── figures/
│       ├── phase1/
│       ├── phase2/
│       └── xai/
│
├── data/
│   └── README.md                    ← ABIDE-I download instructions
│
└── docs/
    └── pipeline_diagram.md
```

**Model weights** hosted on Hugging Face (too large for GitHub):  
→ `https://huggingface.co/spaces/TarunSadarla2606/asd-detection-demo/tree/main/weights`

---

## Pipeline Architecture

### Subject-Level Inference

```
NIfTI volume (.nii / .nii.gz)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│ STEP 1: Axial Slice Extraction (nibabel)            │
└──────────────────────────┬──────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────┐
│ STEP 2: Slice Quality Filter (4 metrics)            │
│   Mean intensity > 0.04 · Brain coverage > 0.12    │
│   Pixel std > 0.03 · Laplacian var > 20            │
│   Contour circularity ≥ 0.45 → removes ~6.5%       │
└──────────────────────────┬──────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────┐
│ STEP 3: Brain ROI Crop (contour + 12px padding)    │
└──────────────────────────┬──────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────┐
│ STEP 4: CNN Inference (all valid slices)            │
│   Per-slice P(ASD), quality score                   │
└──────────────────────────┬──────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────┐
│ STEP 5: Subject-Level Aggregation                   │
│   weight = quality_score × max(P(ASD), P(TC))      │
│   P(ASD)_subject = Σ(P(ASD)_i × w_i) / Σw_i       │
└──────────────────────────┬──────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────┐
│ STEP 6: XAI on Top-K Slices                        │
│   Grad-CAM · LIME · MC-Dropout · Anatomy labels    │
└──────────────────────────┬──────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────┐
│ STEP 7: Clinical Output                             │
│   LLM narrative · PDF report · Phenotypic context  │
└─────────────────────────────────────────────────────┘
```

### CNN Architecture

```
Input: (B, 3, 224, 224)

Block 1:  Conv(3→16,  3×3) + LeakyReLU(0.1) + MaxPool(2×2) + Dropout2d(0.2)
Block 2:  Conv(16→32, 3×3) + BatchNorm + LeakyReLU + MaxPool + Dropout2d
Block 3:  Conv(32→64, 3×3) + BatchNorm + LeakyReLU + MaxPool + Dropout2d
Block 4:  Conv(64→128,3×3) + BatchNorm + LeakyReLU + MaxPool + Dropout2d
Block 5:  Conv(128→256,3×3)+ BatchNorm + LeakyReLU + MaxPool + Dropout2d
          ↑ GradCAM target layer
Flatten → FC(7×7×256 → 100) + LeakyReLU → FC(100 → 2)

Optimizer: NAdam  Loss: CrossEntropyLoss  Epochs: 50  Batch: 64  Params: 1,648,270
```

### Hybrid CNN-ViT Architecture

```
Input: (B, 3, 224, 224)

CNNBackbone (conv1–conv4, 4×MaxPool) → (B, 128, 14, 14)  [196 spatial tokens]
        │
        ├── Brain mask (Otsu threshold + morphological closing)
        │   → zero-out non-brain tokens
        │
        ▼
Linear(128→256) + LayerNorm + 2D sinusoidal pos embedding
        │
CLS token prepend → (B, 197, 256)
        │
4 × TransformerBlock(8 heads, FF=512, Pre-LN, Dropout=0.1)
        │
CLS token → LayerNorm → Linear(256→2) → (B, 2) logits

Optimizer: AdamW  Epochs: 60 (two-phase)  Batch: 32  Params: ~4.2M
AUC: 0.997  (vs CNN: 0.994)
```

---

## XAI Analysis

Full analysis in `notebooks/phase2_graduate_2026/02_xai_analysis_full.ipynb` — 9 sections.

### Grad-CAM vs LIME Agreement

| Case | Pearson r | IoU (top-25%) | GradCAM Energy |
|---|---|---|---|
| ASD — Correct (TP) | N/A (gradient saturation) | 0.288 | 0.000 |
| TC — Correct (TN) | 0.564 | 0.445 | 0.035 |
| ASD — Missed (FN) | −0.286 | 0.042 | 0.196 |
| TC — False Alarm (FP) | 0.035 | 0.185 | 0.184 |

### MC-Dropout Uncertainty

| Case | Point Pred | MC Mean | MC Std | Assessment |
|---|---|---|---|---|
| ASD — Correct (TP) | 1.000 | 1.000 | 0.000 | Low |
| TC — Correct (TN) | 0.000 | 0.000 | 0.000 | Low |
| ASD — Missed (FN) | 0.011 | 0.111 | **0.150** | **High** |
| TC — False Alarm (FP) | 0.999 | 0.997 | 0.005 | Low |

---

## Subgroup Analysis

### By Sex

| Group | N Subjects | Sensitivity | Specificity | AUC |
|---|---|---|---|---|
| Male | 948 | **95.8%** | 97.4% | 0.995 |
| Female | 164 | **93.4%** | 96.7% | 0.989 |

### By Acquisition Site

| Site | N Subjects | Sensitivity | Specificity | AUC |
|---|---|---|---|---|
| USM | 59 | 96.4% | 99.2% | 0.999 |
| UM_1 | 81 | **98.5%** | 97.9% | 0.997 |
| NYU | 73 | 95.2% | 96.9% | 0.991 |
| UM_2 | 61 | 90.2% | 98.5% | 0.991 |
| OLIN | 54 | 97.9% | 99.6% | 0.999 |
| OHSU | 55 | 100.0%* | 95.0% | 1.000* |
| PITT | 56 | **88.5%** | 95.6% | 0.976 |

*OHSU: only 18 ASD slices — statistically unreliable.

---

## Deployment

Deployed as a [Streamlit app on Hugging Face Spaces](https://huggingface.co/spaces/TarunSadarla2606/asd-detection-demo):
- **Free CPU tier** — always-on, no cold starts
- **Git LFS** — NIfTI demo files and model weights tracked via LFS
- **Secret management** — `ANTHROPIC_API_KEY` stored as HF Space secret

---

## Running Locally

```bash
git clone https://github.com/TarunSadarla2606/asd-detection-clinical-ai.git
cd asd-detection-clinical-ai
pip install -r requirements.txt
# Place xai_cnn_best_weights.pth in app/weights/
export ANTHROPIC_API_KEY=sk-ant-...
streamlit run app/app.py
```

**Run tests:**
```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
pip install pytest opencv-python-headless pandas matplotlib Pillow
pytest tests/ -v
```

---

## Clinical & Regulatory Context

This system would be classified as **SaMD Class II** under FDA 21st Century Cures Act guidance. **Current status: Research-grade only.**

Minimum requirements before any clinical deployment:
1. Prospective multi-site validation independent of ABIDE-I
2. Subject-level performance evaluation with confidence intervals
3. Neuroradiologist validation of XAI saliency maps
4. Site-specific calibration for scanner heterogeneity
5. De-biasing for sex performance disparity
6. FDA De Novo premarket submission

Full governance documentation: [`MODEL_CARD.md`](MODEL_CARD.md).

---

## Relationship to B.Tech Work

The original repository ([asd-detection-neuroimaging](https://github.com/TarunSadarla2606/asd-detection-neuroimaging)) contains the complete 2023 academic work — 11 training notebooks, architecture diagrams, and the full B.Tech report. That repository is preserved as-is.

---

## Author

**Tarun Sadarla**  
MS in Artificial Intelligence (Biomedical Concentration)  
University of North Texas — graduating May 2026

- Portfolio: [tarunsadarla2606.github.io](https://tarunsadarla2606.github.io)  
- LinkedIn: [linkedin.com/in/tarun-sadarla](https://www.linkedin.com/in/tarun-sadarla)
- GitHub: [@TarunSadarla2606](https://github.com/TarunSadarla2606)

---

*Research use only. Not validated for clinical deployment. Not a medical diagnosis.*  
*See `MODEL_CARD.md` for full governance documentation.*
