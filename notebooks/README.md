# Notebooks

All notebooks were developed and run on **Kaggle** (free GPU tier — Tesla P100 / T4).

---

## Phase 2 — Graduate System (2026)

Located in `phase2_graduate_2026/`. These are the two core notebooks for this repository.

### `01_training_xai_gradcam_lime.ipynb`

**Purpose:** Training pipeline with integrated data quality analysis and XAI generation.

**Sections:**
- **1A** — Data loading, split construction from ABIDE-I CSVs, class balance verification
- **1B** — Slice quality filtering: 4-metric pipeline implementation, distribution analysis, visual check of rejected vs kept slices
- **2** — Model training: 5-layer CNN, NAdam, 50 epochs, early stopping; training curves
- **3** — Grad-CAM generation on representative slices (TP, TN, FP, FN cases)
- **4** — LIME superpixel explanations on same 4 cases

**Key outputs:**
- `clean_train.csv`, `clean_val.csv`, `clean_test.csv` — quality-filtered label CSVs
- `xai_cnn_best_weights.pth` — trained model weights (6.4MB)
- Training curve plots, quality distribution figures, GradCAM/LIME panels

**Run on Kaggle:**
```
Input datasets:
  - autism/           (ABIDE-I PNG slices)
  - autism-csv/       (label CSV files)
  
Accelerator: GPU (T4 or P100)
Estimated runtime: ~2.5 hours
```

---

### `02_xai_analysis_full.ipynb`

**Purpose:** Comprehensive 9-section clinical AI analysis.

**Sections:**

| Section | Content |
|---|---|
| A | Performance: ROC (AUC=0.994), PR curve, calibration diagram, threshold analysis |
| B | Confidence distribution by true class (bimodal U-shape confirmed) |
| C | Grad-CAM energy distributions: ASD (mean 0.052) vs TC (mean 0.022) |
| D | LIME weight distributions: positive/negative segment counts and weights |
| E | GradCAM–LIME spatial agreement: Pearson r, IoU for all 4 prediction outcomes |
| F | MC-Dropout uncertainty: 30 stochastic passes, 4 cases (TP/TN/FP/FN) |
| G | Clinical deployment analysis: cost threshold (0.10), DET curve, faithfulness test |
| H | Summary dashboard: all metrics in one figure |
| I | Governance: sex + site subgroup analysis, failure mode analysis, Model Card generation |

**Key outputs:** 30+ CSV/PNG files in `/kaggle/working/xai_analysis/`, `MODEL_CARD.md`

**Run on Kaggle:**
```
Input datasets:
  - autism/           (ABIDE-I PNG slices)
  - autism-csv/       (original label CSVs)
  - clean-csvs/       (quality-filtered CSVs from notebook 01)
  - xai-weights/      (xai_cnn_best_weights.pth from notebook 01)
  - phenotypics-csv/  (5320_ABIDE_Phenotypics_20230908.csv)

Accelerator: CPU (GradCAM/LIME are CPU-friendly; GPU speeds up inference)
Estimated runtime: ~3 hours (varies with LIME sample count)
```

---

## Phase 1 — B.Tech Reference (2023)

> Located in `phase1_btech_2023/`. See the original repository for full notebooks:
> **https://github.com/TarunSadarla2606/asd-detection-neuroimaging**

The B.Tech notebooks are not duplicated here to avoid confusion between the two phases. The `phase1_btech_2023/README.md` provides a summary and links.

---

## Common Setup (Kaggle)

For both Phase 2 notebooks, the Kaggle dataset structure expected:

```
/kaggle/input/autism/
    32016/               ← subject folder name = anonymised ID
        32016_001.png
        32016_002.png
        ...
    32067/
        ...
    (1067 subject folders, 100,510+ PNG files total)

/kaggle/input/autism-csv/
    extracted_random_labels_train.csv
    extracted_random_labels_validation.csv
    extracted_random_labels_test.csv

/kaggle/input/phenotypics-csv/
    5320_ABIDE_Phenotypics_20230908.csv
```

See `data/README.md` for ABIDE-I download instructions.
