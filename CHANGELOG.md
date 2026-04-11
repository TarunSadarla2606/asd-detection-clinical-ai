# Changelog

All notable changes from the B.Tech 2023 baseline to this graduate system are documented here.

---

## [2.0.0] — 2026-04 — Graduate Clinical AI System

### Added
- **NIfTI input pipeline** — system now accepts raw `.nii` / `.nii.gz` volumes instead of requiring pre-extracted PNG slices
- **4-metric slice quality filter** — mean intensity, brain coverage fraction, pixel std dev, Laplacian variance, circularity threshold (≥0.45); removes ~6.5% of slices per split
- **Brain ROI crop** — contour-based bounding box extraction before XAI visualisation, eliminates background superpixel artefact in LIME
- **Subject-level aggregation** — quality-weighted confidence voting across all valid slices → single subject-level P(ASD) and prediction
- **Grad-CAM implementation** — `pytorch-grad-cam` on conv5 (final convolutional layer); heatmap, energy, centre-of-mass computed per slice
- **LIME explainability** — 500-sample superpixel perturbation via `lime_image`; green/red visualisation of supporting and counter-evidence regions
- **Grad-CAM × LIME spatial agreement analysis** — Pearson correlation and IoU of top-25% activated regions across all 4 prediction outcomes (TP, TN, FP, FN)
- **MC-Dropout uncertainty quantification** — 30 stochastic forward passes with Dropout2d active at inference; mean P(ASD), σ, entropy
- **GradCAM faithfulness test** — occlusion-based validation of GradCAM saliency maps
- **Calibration analysis** — reliability diagram, Brier score (0.027), confidence histogram
- **Threshold analysis** — Youden (0.37) and cost-optimal (0.10, FN cost=10×) thresholds; DET curve
- **AUPRC metric** — 0.994; addresses class imbalance more robustly than AUC-ROC alone
- **Subgroup analysis** — sex-stratified (Male vs Female) and site-stratified (7 ABIDE-I sites) performance
- **Failure mode analysis** — GradCAM on FP/FN cases, circularity-based slice selection, likely cause attribution
- **Model Card** (Section I of analysis notebook, `MODEL_CARD.md`) — intended use, out-of-scope uses, known limitations, regulatory framing
- **Anatomical region labelling** — heuristic z-position lookup mapping GradCAM CoM to approximate brain region
- **LLM clinical narrative** — Claude Haiku (claude-haiku-4-5) generates adaptive 3-paragraph interpretation from structured model outputs
- **Streamlit deployment app** — NIfTI upload, pipeline progress, prediction card, XAI expanders, uncertainty, narrative, PDF download
- **PDF clinical report** — ReportLab-generated 2-page report with subject metadata, performance table, GradCAM figures, LLM narrative, disclaimer
- **Hugging Face Spaces deployment** — always-on CPU deployment, Git LFS for NIfTI and weights, secrets management
- **Phenotypic metadata integration** — ABIDE-I CSV lookup for subject age, sex, site; site-reliability indicator in UI

### Changed
- **Model retrained on quality-filtered data** — same 5-layer CNN architecture (NAdam, 50 epochs) but trained on `clean_train.csv` instead of raw CSV; AUC improved from 0.97 → 0.994
- **Output layer bug fixed** — B.Tech model was missing `fc2` output layer in forward pass; corrected
- **Normalisation constants** — updated from ImageNet statistics to ABIDE-I training set statistics (mean=0.129, std=0.174)
- **Data splits** — Train 72,367 → 67,625 | Val 8,041 → 7,507 | Test 20,102 → 18,814 (after quality filtering)

### Fixed
- LIME visualization was incomplete in B.Tech version (error before final annotated output) — fully implemented
- GradCAM explored but not retained in B.Tech — now fully implemented with population-level analysis

---

## [1.0.0] — 2023-11 — B.Tech Capstone (Baseline)

> Original work documented at: https://github.com/TarunSadarla2606/asd-detection-neuroimaging

### Established
- Custom 5-layer CNN for ASD/TC binary classification
- Skip-connected CNN variant (residual connection layers 1→3)
- Custom Vision Transformer (PyTorch from scratch, Keras from scratch)
- Pretrained ViT fine-tuning (ViT-B/16 via timm, google/vit-base via HuggingFace)
- ABIDE-I dataset pipeline: NIfTI → DICOM → PNG, 1,067 subjects, 100,510 slices
- Systematic optimizer comparison: Adam, NAdam, RMSprop across all architectures
- Key finding: CNNs (AUC 0.97–0.98) vastly outperform from-scratch ViTs (AUC 0.62) at this data scale
- LIME setup partially implemented (explainer invoked, visualization incomplete)
- Half-dataset vs full-dataset comparison (533 vs 1,067 subjects)

### Known Open Issues (resolved in v2.0.0)
- LIME visualization pipeline incomplete
- GradCAM implementation not retained
- No subject-level aggregation (slice-level classification only)
- No uncertainty quantification
- No subgroup or multi-site analysis
- No clinical output format
