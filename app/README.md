---
title: ASD Detection Clinical AI Demo
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# ASD Detection from Structural MRI — Clinical AI Research Demo

**Author:** Tarun Sadarla · MS Artificial Intelligence (Biomedical Concentration) · University of North Texas  
**Dataset:** ABIDE-I (Autism Brain Imaging Data Exchange) · 1,067 subjects · 17 acquisition sites  
**Status:** Research grade · Not validated for clinical deployment

---

## What This Demo Does

Upload a structural brain MRI (NIfTI format). The system extracts quality-filtered axial slices, runs two independently trained models, generates spatial explanations from both, fuses their predictions using uncertainty-aware weighting, and produces three downloadable clinical PDF reports — one per tab.

Everything visible in the app, including the clinical narratives, is generated from the actual model outputs. No hardcoded templates are used in the narrative path: all three reports are written by Claude Haiku from structured numerical data passed at inference time.

---

## Three Analysis Modes

### Tab 1 — Pure CNN

A 5-layer custom convolutional neural network trained end-to-end on ABIDE-I 2D axial slices. This is the primary model.

**Architecture:** Conv(3→16) → Conv(16→32) → Conv(32→64) → Conv(64→128) → Conv(128→256) → FC(100) → FC(2). Each block includes LeakyReLU, BatchNorm, MaxPool, and Dropout2d. Total: 1,648,270 parameters.

**Training:** NAdam optimiser, 50 epochs with early stopping, quality-filtered slices (4-metric pipeline: brightness, brain coverage, variance, Laplacian sharpness). 67,625 training slices after filtering.

**Validation performance (held-out test set):**

| AUC | Sensitivity | Specificity | AUPRC | Brier Score |
|-----|-------------|-------------|-------|-------------|
| 0.994 | 95.6% | 97.2% | 0.994 | 0.027 |

**Sex-stratified:** Male sensitivity 95.8% · Female sensitivity 93.4%  
**Site sensitivity range:** PITT 88.5% to UM_1 98.5% (9.4pp gap across 17 sites)

**Explainability:**
- GradCAM (Selvaraju et al., 2017) on conv5 — gradient-weighted class activation maps
- LIME (optional) — superpixel perturbation analysis
- MC-Dropout (30 passes, Dropout2d active) — stochastic uncertainty estimation

---

### Tab 2 — Hybrid CNN-ViT

A hybrid architecture combining the CNN backbone with a Transformer encoder. The first four convolutional blocks (conv1–conv4) act as a spatial feature extractor, producing a 14×14 grid of 128-dimensional tokens. These tokens pass through a 4-layer Transformer encoder with 8 attention heads that learns global self-attention relationships across the brain slice.

**Key design choices:**

*Brain masking:* sMRI axial slices contain 30–40% black skull background. Rather than feeding background tokens to the Transformer, a binary brain mask (Otsu threshold + morphological closing) is computed from each input image and downsampled to 14×14 token resolution. Background tokens are zeroed before the projection layer. This prevents the Transformer from wasting attention capacity on uninformative regions and eliminates spurious positional biases from skull/corner positions.

*2D sinusoidal positional embeddings:* Row and column positions are encoded independently using sinusoidal functions, making spatial structure explicit for the Transformer without learnable parameters.

*Two-phase training:* The CNN backbone is frozen for the first 5 epochs while the Transformer head initialises against stable features, then unfrozen with a 10× lower learning rate (2e-5 vs 2e-4) for joint fine-tuning. Gradient clipping (max norm 1.0) prevents early training instability.

**Architecture:** CNNBackbone (conv1–conv4) → 196 tokens → Linear(128→256) → 2D pos emb → CLS prepend → 4 × TransformerBlock(8 heads, FF=512) → CLS → LayerNorm → Linear(256,2). Total: ~1.7M parameters.

**Validation performance:**

| AUC | Sensitivity | Specificity | AUPRC | Brier Score |
|-----|-------------|-------------|-------|-------------|
| 0.943 | 84.5% | 86.8% | 0.944 | 0.097 |

The lower performance relative to the pure CNN is expected and architecturally informative: Transformer models lack the translation equivariance inductive bias that CNNs have built in, requiring more training data to learn it from scratch. With ~67K slices, the CNN's structural advantage holds. The architectural value of the Hybrid model lies in complementary spatial attention mechanisms, not raw classification performance.

**Explainability:**
- CLS→token attention rollout from the last Transformer layer, averaged across 8 heads — shows which spatial positions across the full slice the model weights globally, distinct from local gradient saliency
- GradCAM on backbone.conv4 — gradients flow through the Transformer back into the CNN backbone
- MC-Dropout (30 passes, both Dropout2d and Transformer Dropout active) — characteristically higher σ than pure CNN because both dropout pathways contribute stochastic variance simultaneously

---

### Tab 3 — Uncertainty-Gated Ensemble

Both models run independently on the same scan. Their predictions are fused using uncertainty-aware weighting:

**Fusion formula:**
```
w_CNN = (1 − σ_CNN) × AUC_CNN
w_ViT = (1 − σ_ViT) × AUC_ViT
P_ensemble = (w_CNN × P_CNN + w_ViT × P_ViT) / (w_CNN + w_ViT)
```

The weight each model receives is inversely proportional to its MC-Dropout standard deviation on the specific scan being evaluated. A model that is highly uncertain about a particular brain scan receives a lower weight regardless of its average validation performance. This adapts the ensemble to per-scan model confidence rather than using fixed weights derived from population-level metrics.

**Model disagreement:** When the CNN and Hybrid ViT independently predict different classes, the app surfaces a clinical disagreement flag. Disagreement between architecturally diverse models trained on identical data indicates the scan contains features that different representational strategies resolve differently — the ensemble cannot resolve this ambiguity algorithmically, and the result is explicitly flagged for expert review.

**XAI comparison:** GradCAM (local gradient saliency, CNN) and Attention Rollout (global self-attention, ViT) are computed on the same slices and compared using Jaccard IoU and Pearson correlation. High agreement (IoU > 0.30) means both methods identify overlapping spatial regions as discriminative. Low agreement (IoU < 0.15) indicates complementary explanations — the CNN focuses on local textural gradients while the Transformer attends to global spatial relationships. Low IoU is not an error condition.

**Three downloadable PDF reports:** One per tab, each containing the model's prediction, subject/site context, XAI figures, uncertainty estimates, and a Claude Haiku-generated clinical narrative written from the structured numerical outputs. The ensemble report additionally includes a side-by-side model comparison table, the explicit fusion weight calculation, and a mandatory expert review box when models disagree.

---

## Pipeline

```
NIfTI (.nii / .nii.gz)
  ↓
Axial slice extraction (nibabel)
  ↓
4-metric quality filter
  brightness · brain coverage fraction · intensity std · Laplacian variance
  ~6.5% rejection rate on ABIDE-I
  ↓
Brain ROI crop (largest contour bounding box + 12px padding)
  ↓
CNN inference on all valid slices
  Confidence-weighted subject-level aggregation
  weight = quality_score × max(P_ASD, P_TC)
  ↓
Hybrid CNN-ViT inference (same slices, separate normalisation stats)
  ↓
GradCAM (CNN) + Attention Rollout (ViT)
  ↓
MC-Dropout uncertainty (both models, 30 passes each)
  ↓
Uncertainty-gated ensemble fusion
  ↓
LLM narratives (Claude Haiku, 3 separate prompts)
  ↓
PDF report generation (ReportLab, 3 separate reports)
```

---

## Clinical Limitations

This system was developed on the ABIDE-I dataset, which is retrospective, multi-site, and collected under heterogeneous scanning protocols. Several limitations must be understood before interpreting any output:

**2D slice-level inference:** The models process individual 2D axial slices, not full 3D volumetric context. Slice-level predictions are aggregated by confidence-weighted voting, but spatial relationships across slices (e.g. volumetric asymmetry) are not captured.

**Domain shift:** Models trained on ABIDE-I (1.5T and 3T sMRI, multiple vendors) may not generalise to scans acquired under different protocols, field strengths, or preprocessing pipelines. Site-stratified sensitivity varies 9.4pp across the 17 ABIDE-I sites.

**Sex imbalance:** Female subjects represent 164/1,067 (15.4%) of the training cohort. Female sensitivity is 93.4% vs 95.8% for males. Results for female subjects should be interpreted with additional caution.

**Age range:** ABIDE-I subjects span 6–56 years. Model behaviour outside this range is unknown.

**No 3D volumetric features, no EHR integration, no longitudinal data:** Features known to be relevant for ASD diagnosis — cortical thickness, sulcal depth, white matter connectivity, developmental trajectory — are not captured in 2D axial slice classification.

**Regulatory status:** This system has not been submitted for FDA SaMD classification. Clinical deployment would require prospective multi-site validation, institutional review, and regulatory clearance.

---

## Dataset

ABIDE-I is a publicly available dataset aggregated by the Autism Brain Imaging Data Exchange initiative. It contains resting-state fMRI and structural MRI data from 17 international sites. The structural MRI (sMRI) component was used for this project. Subject identifiers have been anonymised.

Preprocessing pipeline applied: NIfTI loading → axial slice extraction → per-slice normalisation → 4-metric quality filtering → PNG export at native resolution. No skull stripping, no atlas registration, no spatial normalisation was applied — the quality filter and brain mask generation handle background suppression at inference time.

---

## Architecture Comparison

| | Pure CNN | Hybrid CNN-ViT |
|---|---|---|
| Layers | 5 conv + 2 FC | 4 conv + 4 transformer |
| Parameters | 1,648,270 | ~1,700,000 |
| AUC | 0.994 | 0.943 |
| Sensitivity | 95.6% | 84.5% |
| XAI method | GradCAM (gradients) | Attention Rollout (self-attention) |
| Uncertainty | MC-Dropout2d | MC-Dropout2d + Transformer Dropout |
| Inductive bias | Translation equivariance | Global self-attention |

---

## Future Work

- **Structural pruning (Phase 3A):** ILR-scored filter pruning targeting 40–50% parameter reduction with fairness-preserving guard rails (site sensitivity std constraint alongside AUC and sensitivity thresholds). Motivated by the 5,730 MC-Dropout forward passes required per subject in the current deployment.
- **Hybrid model pruning (Phase 3B-ii):** FFN hidden dimension pruning for the Transformer blocks, paired with the CNN→projection layer constraint (analogous to U-Net skip-connection pruning in the fetal head segmentation project).
- **3D volumetric modelling:** Replace 2D slice classification with 3D U-Net or 3D Swin Transformer operating on full brain volumes.
- **EHR integration:** Fuse imaging features with developmental history, ADOS/ADI-R scores, and comorbidity data via cross-modal attention.
- **Prospective validation:** Evaluate on ABIDE-II and site-held-out splits to measure true generalisation.

---

## References

- Di Martino et al. (2014). The autism brain imaging data exchange: towards a large-scale evaluation of the intrinsic brain architecture in autism. *Molecular Psychiatry.*
- Selvaraju et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV.*
- Dosovitskiy et al. (2021). An image is worth 16×16 words: Transformers for image recognition at scale. *ICLR.*
- Gal & Ghahramani (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. *ICML.*
- Liu et al. (2017). Learning efficient convolutional networks through network slimming. *ICCV.*

---

*Tarun Sadarla · MS Artificial Intelligence, Biomedical Concentration · University of North Texas · 2026*  
*This demo is part of a portfolio of clinical AI systems developed during graduate study.*
