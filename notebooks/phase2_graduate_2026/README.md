# Phase 2 Notebooks — Graduate System (2026)

Both notebooks were run on Kaggle (free GPU tier).  
Add the notebooks from your Kaggle account to this folder:

- `01_training_xai_gradcam_lime.ipynb`
- `02_xai_analysis_full.ipynb`

See the parent `notebooks/README.md` for full section descriptions and Kaggle dataset setup.

## Quick reference — key outputs

### Notebook 01
- Clean CSVs: `clean_train.csv` (67,625 slices), `clean_val.csv` (7,507), `clean_test.csv` (18,814)
- Weights: `xai_cnn_best_weights.pth` (1,648,270 parameters, 6.4MB)
- Rejection rate: ~6.5% consistent across all splits

### Notebook 02
- AUC: 0.9944 · AUPRC: 0.9943 · Brier: 0.0270
- GradCAM faithfulness: ASD-correct confidence drop 0.997 (faithful)
- FN case MC-Dropout σ: 0.150 (correctly flagged high uncertainty)
- FP case MC-Dropout σ: 0.005 (confidently wrong — dangerous failure mode documented)
- Site sensitivity range: PITT 88.5% to UM_1 98.5% (9.4pp gap)
