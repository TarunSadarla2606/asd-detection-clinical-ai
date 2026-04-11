# Results

All experimental results for both Phase 1 (B.Tech 2023) and Phase 2 (Graduate 2026).

---

## Phase 2 — Graduate Clinical AI System (2026)

### Full Metrics (Test Set — 18,814 quality-filtered slices)

| Metric | Value | Source |
|---|---|---|
| AUC-ROC | **0.9944** | `notebooks/phase2_graduate_2026/02_xai_analysis_full.ipynb` Section A |
| AUPRC | **0.9943** | Same |
| Accuracy | 0.9644 | Same |
| Sensitivity | **0.9559** | Same |
| Specificity | **0.9725** | Same |
| Precision (PPV) | 0.9705 | Same |
| F1 Score | 0.9631 | Same |
| Brier Score | **0.0270** | Same |
| False Negative Rate | 0.0441 | Same |
| False Positive Rate | 0.0275 | Same |
| Confusion Matrix (TN/FP/FN/TP) | 9391/266/404/8753 | Same |

### Subgroup Analysis

**By Sex:**

| Group | N Subjects | N Slices | Sensitivity | Specificity | AUC |
|---|---|---|---|---|---|
| Male | 948 | 16,415 | 0.9583 | 0.9735 | 0.9950 |
| Female | 164 | 2,399 | 0.9345 | 0.9666 | 0.9891 |

**By Site:**

| Site | N Subjects | Sensitivity | Specificity | AUC |
|---|---|---|---|---|
| USM | 59 | 0.9636 | 0.9922 | 0.9985 |
| UM_1 | 81 | 0.9846 | 0.9796 | 0.9974 |
| NYU | 73 | 0.9517 | 0.9691 | 0.9910 |
| UM_2 | 61 | 0.9024 | 0.9852 | 0.9912 |
| OLIN | 54 | 0.9795 | 0.9961 | 0.9990 |
| OHSU | 55 | 1.0000† | 0.9504 | 1.0000† |
| PITT | 56 | 0.8854 | 0.9561 | 0.9761 |

†OHSU sensitivity unreliable — only 18 ASD slices.

### Quality Filter Results

| Split | Original | After Filter | Removed | % |
|---|---|---|---|---|
| Train | 72,367 | 67,625 | 4,742 | 6.6% |
| Val | 8,041 | 7,507 | 534 | 6.6% |
| Test | 20,102 | 18,814 | 1,288 | 6.4% |

Rejection reasons (from 500-slice sample analysis):
- Blank (mean intensity < 0.04): 6.2%
- Low brain coverage (< 0.12): 4.6%
- Blurry (Laplacian < 20): 1.0%
- Uniform (std dev < 0.03): 1.2%

---

## Phase 1 — B.Tech Baseline (2023)

Data from: `experiment_results.csv`

### CNN — Full Dataset (1,067 subjects)

| Optimizer | Train Acc | Test Acc | AUC | Precision | Recall | Specificity | F1 |
|---|---|---|---|---|---|---|---|
| Adam | 0.92 | 0.90 | 0.97 | 0.87 | 0.93 | 0.87 | 0.90 |
| NAdam | 0.93 | 0.91 | 0.97 | 0.90 | 0.90 | 0.91 | 0.90 |
| RMSprop | 0.94 | 0.91 | 0.97 | 0.90 | 0.90 | 0.91 | 0.90 |

### Skip-Connected CNN — Full Dataset (1,067 subjects)

| Optimizer | Train Acc | Test Acc | AUC | Precision | Recall | Specificity | F1 |
|---|---|---|---|---|---|---|---|
| Adam | 0.91 | 0.90 | 0.97 | 0.90 | 0.88 | 0.91 | 0.89 |
| NAdam | 0.93 | 0.91 | **0.98** | 0.93 | 0.88 | 0.94 | 0.90 |
| RMSprop | 0.93 | 0.91 | 0.97 | 0.93 | 0.87 | 0.94 | 0.90 |

### CNN — Half Dataset (533 subjects, early runs)

| Optimizer | Train Acc | Test Acc | AUC |
|---|---|---|---|
| Adam | 0.94 | 0.92 | 0.98 |
| NAdam | 0.94 | 0.92 | 0.98 |
| RMSprop | 0.93 | 0.91 | 0.98 |

### Vision Transformer (Keras, from scratch, 1,067 subjects)

| Metric | Value |
|---|---|
| Train Accuracy | 0.597 |
| Test Accuracy | 0.616 |
| AUC | 0.619 |
| F1 | 0.636 |

Near-random performance. Expected: from-scratch ViTs require far more data to be effective. See `MODEL_CARD.md` for detailed analysis.

---

## Figures

### Phase 1 (`figures/phase1/`)
- `accuracy_comparison.png` — Train vs test accuracy across all CNN variants
- `metrics_all_models.png` — AUC/Precision/Recall/F1 for all 6 CNN experiments
- `cnn_vs_vit.png` — CNN vs ViT performance comparison
- `radar_cnn_optimizers.png` — Radar chart of all metrics across optimizers
- `dataset_size_comparison.png` — 533 vs 1,067 subjects accuracy/AUC

### Phase 2 (`figures/phase2/`)
- `training_curves.png` — Training/validation accuracy and loss (50 epochs)
- `evaluation_plots.png` — ROC curve (AUC=0.994) + confusion matrix
- `quality_distributions.png` — 4-metric quality score distributions with thresholds
- `cleaning_summary.png` — Dataset size before/after filtering
- `cleaning_visual_check.png` — Visual: rejected vs kept slices

### XAI (`figures/xai/`)
- `gradcam_panel.png` — GradCAM heatmaps on 4 cases (TP, TN, FP, FN)
- `lime_panel.png` — LIME superpixel explanations on same 4 cases
- `xai_comparison_panel.png` — Side-by-side GradCAM vs LIME comparison
- `gradcam_energy_dist.png` — Population-level GradCAM energy by class
- `mc_dropout.png` — MC-Dropout uncertainty distributions (30 passes)
- `subgroup_performance.png` — Sex + site subgroup bar charts
- `summary_dashboard.png` — Full XAI analysis summary dashboard

---

*All Phase 2 metrics verified against notebook outputs in `02_xai_analysis_full.ipynb`.*
