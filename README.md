# ASD Detection — Clinical AI System

> Upgrading a 2023 B.Tech capstone into a production-grade neuroimaging AI:  
> NIfTI-to-prediction pipeline · Grad-CAM + LIME explainability · MC-Dropout uncertainty · LLM clinical narrative · Deployed on Hugging Face Spaces.

[![Live Demo](https://img.shields.io/badge/Live_Demo-Hugging_Face_Spaces-FF9900?logo=huggingface&logoColor=white)](https://huggingface.co/spaces/TarunSadarla2606/asd-detection-demo)
[![Original B.Tech Repo](https://img.shields.io/badge/B.Tech_2023-Original_Capstone-9d7fe8?logo=github)](https://github.com/TarunSadarla2606/asd-detection-neuroimaging)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.56-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
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

The answer involves: a proper inference pipeline that accepts clinical-format inputs, explainability methods that show *why* the model predicts what it predicts, uncertainty quantification that flags when it's guessing, subgroup analysis that documents where it fails, and a governance framework that honestly states what would be required for clinical deployment.

---

## What's New vs B.Tech Baseline

| Capability | 2023 B.Tech Baseline | 2026 Clinical AI System |
|---|---|---|
| **Input format** | Pre-extracted PNG slices | Raw NIfTI volumes (.nii / .nii.gz) |
| **Data quality** | Raw slices including blanks and noise | 4-metric quality filter — ~6.5% removed |
| **Evaluation** | Accuracy + AUC | + AUPRC, Brier score, calibration, threshold analysis |
| **Explainability** | LIME attempt (visualization incomplete) | Grad-CAM + LIME + spatial agreement (Pearson r, IoU) |
| **Uncertainty** | None | MC-Dropout (30 stochastic passes) |
| **Subgroup analysis** | None | Sex-stratified + 7-site stratified performance |
| **Failure mode analysis** | None | GradCAM on FP/FN cases, failure log, causal analysis |
| **Clinical output** | None | LLM narrative (Claude Haiku), PDF report, anatomical region labelling |
| **Deployment** | Kaggle notebook | Streamlit app on Hugging Face Spaces (always-on) |
| **Governance** | None | Model Card, regulatory framing (FDA SaMD Class II), fairness documentation |

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

| Metric | Value | Clinical Relevance |
|---|---|---|
| **AUC-ROC** | **0.994** | Near-perfect discrimination between ASD and TC at slice level |
| **AUPRC** | **0.994** | Robust performance under class imbalance |
| **Sensitivity** | **95.6%** | 95.6% of ASD slices correctly identified (true positive rate) |
| **Specificity** | **97.2%** | 97.2% of TC slices correctly cleared (true negative rate) |
| **Precision (PPV)** | **97.1%** | When model predicts ASD, 97.1% correct |
| **F1 Score** | **96.3%** | Harmonic mean of precision and recall |
| **Brier Score** | **0.027** | Well-calibrated — probabilities match observed frequencies |
| **False Negative Rate** | **4.4%** | Clinically most costly error: ASD missed |
| **False Positive Rate** | **2.8%** | Unnecessary follow-up rate |

Comparison vs B.Tech baseline (same architecture, pre-filtered data, B.Tech NAdam run):

| Metric | B.Tech 2023 | Graduate 2026 | Δ |
|---|---|---|---|
| AUC | 0.97 | **0.994** | +0.024 |
| Sensitivity | ~0.90 | **0.956** | +0.056 |
| Specificity | ~0.91 | **0.972** | +0.062 |
| Brier Score | — | **0.027** | new metric |

The improvement is entirely attributable to the **4-metric quality filter** — removing ~6.5% of garbage slices (blanks, noise, partial scans) from training data produced a substantial performance gain.

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
├── app/                             ← Streamlit application
│   ├── app.py                       ← Main entry point
│   ├── model.py                     ← CNN architecture + load_model()
│   ├── pipeline.py                  ← NIfTI → slices → filter → predict → aggregate
│   ├── xai.py                       ← Grad-CAM + LIME + MC-Dropout
│   ├── anatomy.py                   ← Heuristic anatomical region labelling
│   ├── narrator.py                  ← LLM narrative via Claude Haiku API
│   ├── phenotypic.py                ← ABIDE-I phenotypic CSV + site mapping
│   └── report.py                    ← PDF report generation (ReportLab)
│
├── src/                             ← Reusable Python modules (non-app)
│   ├── __init__.py
│   ├── models.py                    ← CNN architectures (plain + skip-connected)
│   ├── quality_filter.py            ← 4-metric slice quality filter (NEW)
│   ├── dataset.py                   ← PyTorch Dataset + DataLoader factory
│   ├── preprocess.py                ← CNN / ViT preprocessing pipelines
│   ├── train.py                     ← Training loop with early stopping
│   └── evaluate.py                  ← Metrics, ROC curve, confusion matrix
│
├── notebooks/                       ← Kaggle analysis notebooks
│   ├── README.md                    ← Notebook index and run instructions
│   ├── phase1_btech_2023/           ← Original B.Tech training notebooks (reference)
│   │   └── README.md               ← Points to original repo
│   └── phase2_graduate_2026/
│       ├── 01_training_xai_gradcam_lime.ipynb    ← Training + cleaning + GradCAM + LIME
│       └── 02_xai_analysis_full.ipynb            ← Full 9-section XAI analysis
│
├── configs/                         ← Training configuration files
│   ├── cnn_nadam.yaml               ← Best-performing CNN config
│   ├── cnn_adam.yaml
│   ├── cnn_rmsprop.yaml
│   └── skip_cnn_nadam.yaml
│
├── results/                         ← All experimental outputs
│   ├── README.md                    ← Full metrics tables with sources
│   ├── experiment_results.csv       ← All 11 experimental runs
│   ├── phase2_metrics.csv           ← Graduate system full metrics
│   ├── subgroup_sex.csv             ← Sex-stratified performance
│   ├── subgroup_site.csv            ← Site-stratified performance
│   └── figures/
│       ├── phase1/                  ← B.Tech figures (from original repo)
│       │   ├── accuracy_comparison.png
│       │   ├── metrics_all_models.png
│       │   ├── cnn_vs_vit.png
│       │   ├── radar_cnn_optimizers.png
│       │   └── dataset_size_comparison.png
│       ├── phase2/                  ← Graduate system training figures
│       │   ├── training_curves.png
│       │   ├── evaluation_plots.png
│       │   ├── quality_distributions.png
│       │   ├── cleaning_summary.png
│       │   └── cleaning_visual_check.png
│       └── xai/                     ← XAI analysis figures
│           ├── gradcam_panel.png
│           ├── lime_panel.png
│           ├── xai_comparison_panel.png
│           ├── gradcam_energy_dist.png
│           ├── mc_dropout.png
│           ├── subgroup_performance.png
│           └── summary_dashboard.png
│
├── data/
│   └── README.md                    ← ABIDE-I download instructions + CSV format
│
├── docs/
│   └── pipeline_diagram.md          ← Text-based pipeline diagram
│
└── .github/
    └── ISSUE_TEMPLATE/
        └── bug_report.md
```

**Model weights** are hosted on Hugging Face (too large for GitHub):  
→ `https://huggingface.co/spaces/TarunSadarla2606/asd-detection-demo/tree/main/weights`  
Download `xai_cnn_best_weights.pth` and place in `app/weights/`.

---

## Pipeline Architecture

### Subject-Level Inference

```
NIfTI volume (.nii / .nii.gz)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│ STEP 1: Axial Slice Extraction (nibabel)            │
│   • Load 3D volume (256×182×256 typical)            │
│   • Extract all axial cross-sections                │
│   • Rescale intensity to uint8                      │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│ STEP 2: Slice Quality Filter (4 metrics)            │
│   • Mean intensity > 0.04         (blank rejection) │
│   • Brain coverage fraction > 0.12 (low coverage)  │
│   • Pixel std dev > 0.03          (uniform slices)  │
│   • Laplacian variance > 20       (blur rejection)  │
│   • Contour circularity ≥ 0.45    (non-brain shape) │
│   → Removes ~6.5% of slices                        │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│ STEP 3: Brain ROI Crop                              │
│   • Contour bounding box + 12px padding             │
│   • Removes black background before XAI            │
│   • Resize to 224×224                               │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│ STEP 4: CNN Inference (all valid slices)            │
│   • 5-layer CNN, NAdam, 1.6M params                 │
│   • Per-slice P(ASD), quality score                 │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│ STEP 5: Subject-Level Aggregation                   │
│   weight = quality_score × max(P(ASD), P(TC))      │
│   P(ASD)_subject = Σ(P(ASD)_i × w_i) / Σw_i       │
│   → Subject predicted as ASD if P(ASD) ≥ 0.5       │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│ STEP 6: XAI on Top-K Slices (ranked by quality)    │
│   • Grad-CAM → spatial heatmap on conv5             │
│   • LIME → 500-sample superpixel perturbation       │
│   • MC-Dropout → 30-pass uncertainty estimate       │
│   • Anatomical region → heuristic z-position map   │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│ STEP 7: Clinical Output                             │
│   • LLM narrative (Claude Haiku, ~$0.002/call)     │
│   • Phenotypic context + site reliability           │
│   • PDF report (ReportLab)                         │
└─────────────────────────────────────────────────────┘
```

### CNN Architecture

```
Input: (B, 3, 224, 224)

Block 1:  Conv(3→16,  3×3, p=1) + LeakyReLU(0.1) + MaxPool(2×2) + Dropout2d(0.2)
Block 2:  Conv(16→32, 3×3, p=1) + BatchNorm + LeakyReLU + MaxPool + Dropout2d
Block 3:  Conv(32→64, 3×3, p=1) + BatchNorm + LeakyReLU + MaxPool + Dropout2d
Block 4:  Conv(64→128,3×3, p=1) + BatchNorm + LeakyReLU + MaxPool + Dropout2d
Block 5:  Conv(128→256,3×3,p=1) + BatchNorm + LeakyReLU + MaxPool + Dropout2d
          ↑ GradCAM target layer
Flatten → FC(256×7×7 → 100) + LeakyReLU → FC(100 → 2)

Optimizer: NAdam (lr=1e-3, β₁=0.9, β₂=0.999, ε=1e-8)
Loss: CrossEntropyLoss
Epochs: 50 (early stopping, patience=5)
Batch size: 64
Parameters: 1,648,270
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

**Key finding:** Anti-correlation (r = −0.29) between Grad-CAM and LIME on the false negative case is a clinically meaningful signal — the two explanation methods are pointing at different regions, indicating unstable model attention on this prediction. The MC-Dropout result for this same case confirms: σ = 0.150 (high uncertainty).

### MC-Dropout Uncertainty

| Case | Point Pred | MC Mean | MC Std | Assessment |
|---|---|---|---|---|
| ASD — Correct (TP) | 1.000 | 1.000 | 0.000 | Low |
| TC — Correct (TN) | 0.000 | 0.000 | 0.000 | Low |
| ASD — Missed (FN) | 0.011 | 0.111 | **0.150** | **High** |
| TC — False Alarm (FP) | 0.999 | 0.997 | 0.005 | Low |

The dangerous failure mode: the false positive has σ = 0.005 — the model is confidently wrong with no uncertainty signal. This argues for always displaying uncertainty alongside prediction in any clinical interface.

### GradCAM Faithfulness (Occlusion Test)

| Case | Conf (original) | Conf (occluded) | Drop | Faithful? |
|---|---|---|---|---|
| ASD — Correct | 1.000 | 0.003 | 0.997 | ✓ Faithful |
| TC — Correct | 1.000 | 1.000 | 0.000 | ⚠ Saturation |

The ASD-correct case shows definitively faithful GradCAM — occluding the highlighted region destroys model confidence. The TC-correct case exhibits gradient saturation at high confidence (known GradCAM limitation; GradCAM++ would address this).

---

## Subgroup Analysis

### By Sex

| Group | N Subjects | Sensitivity | Specificity | AUC |
|---|---|---|---|---|
| Male | 948 | **95.8%** | 97.4% | 0.995 |
| Female | 164 | **93.4%** | 96.7% | 0.989 |

2.4 percentage point sensitivity gap. ABIDE-I is ~85% male — the female gap is expected and explicitly documented in `MODEL_CARD.md`.

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

*OHSU: only 18 ASD slices — statistically unreliable. PITT is the worst-performing site at 88.5% sensitivity — 9.4 percentage points below UM_1. Scanner heterogeneity is the primary driver.

---

## Deployment

The system is deployed as a [Streamlit app on Hugging Face Spaces](https://huggingface.co/spaces/TarunSadarla2606/asd-detection-demo):
- **Free CPU tier** — always-on, no cold starts
- **Git LFS** — NIfTI demo files and model weights tracked via LFS
- **Secret management** — `ANTHROPIC_API_KEY` stored as HF Space secret
- **Fallback narrative** — if API unavailable, template-based narrative generated

---

## Running Locally

```bash
# Clone
git clone https://github.com/TarunSadarla2606/asd-detection-clinical-ai.git
cd asd-detection-clinical-ai

# Install dependencies
pip install -r requirements.txt

# Download model weights from HF Space
# Place xai_cnn_best_weights.pth in app/weights/

# Set API key (optional — enables LLM narrative)
export ANTHROPIC_API_KEY=sk-ant-...

# Run
streamlit run app/app.py
```

**Data access:** The app accepts any NIfTI structural MRI file. For ABIDE-I data, register at [https://fcon_1000.projects.nitrc.org/indi/abide/](https://fcon_1000.projects.nitrc.org/indi/abide/). Name files with the anonymised subject ID (e.g. `32016.nii`) for automatic phenotypic metadata lookup. See `data/README.md` for full instructions.

---

## Clinical & Regulatory Context

This system would be classified as **Software as a Medical Device (SaMD) Class II** under FDA 21st Century Cures Act / Digital Health Center of Excellence guidance — software that analyses medical images to inform clinical decisions. The **De Novo pathway** would likely be required.

**Current status: Research-grade only.**

Minimum requirements before any clinical deployment:

1. Prospective multi-site validation independent of ABIDE-I
2. Subject-level (not slice-level) performance evaluation with CIs
3. Neuroradiologist validation of XAI saliency maps against anatomical ground truth
4. Site-specific calibration or harmonisation to address scanner heterogeneity
5. De-biasing for sex performance disparity (93.4% vs 95.8%)
6. FDA De Novo premarket submission with full clinical evidence package
7. IRB approval for any prospective data collection

Full governance documentation: see [`MODEL_CARD.md`](MODEL_CARD.md).

---

## Relationship to B.Tech Work

The original repository ([asd-detection-neuroimaging](https://github.com/TarunSadarla2606/asd-detection-neuroimaging)) contains the complete 2023 academic work — 11 training notebooks, architecture diagrams, and the full B.Tech report. That repository is preserved as-is.

This repository is what happened when those open threads were picked up:

> *"GradCAM explored but not retained. LIME setup ran but visualization was incomplete. Knowing a model is accurate is necessary but not sufficient — what it's actually looking at in the brain scan is the next question."*  
> — B.Tech project reflection, 2023 README

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
