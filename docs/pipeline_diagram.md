# Pipeline Documentation

## End-to-End Inference Pipeline (Graduate System)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ASD DETECTION CLINICAL AI PIPELINE                       │
│                    NIfTI Volume → Subject-Level Prediction                  │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT: NIfTI volume (.nii or .nii.gz)
       Typical dimensions: 256×182×256 (varies by site/scanner)
       File size: 13–25MB
       ↓

┌──────────────────────────────────────────────────┐
│ STEP 1: AXIAL SLICE EXTRACTION                   │
│  Library: nibabel                                │
│  • nib.load(path).get_fdata()                   │
│  • Extract data[:,:,z] for all z                │
│  • Rescale to uint8: (slice/max)*255            │
│  Output: list of (H,W) uint8 arrays             │
│  Typical: ~256 slices per volume                │
└───────────────────────┬──────────────────────────┘
                        │
                        ↓
┌──────────────────────────────────────────────────┐
│ STEP 2: SLICE QUALITY FILTER (4 metrics)        │
│  See: src/quality_filter.py                     │
│                                                  │
│  Metric 1: mean_i > 0.04                        │
│    → Rejects completely blank slices            │
│    → Catches top/bottom of skull coverage       │
│                                                  │
│  Metric 2: brain_frac > 0.12                    │
│    → Fraction of pixels > 5% max intensity     │
│    → Rejects slices with minimal brain tissue   │
│                                                  │
│  Metric 3: std_i > 0.03                         │
│    → Pixel standard deviation                   │
│    → Rejects uniform/featureless slices         │
│                                                  │
│  Metric 4: circularity ≥ 0.45                   │
│    → 4πA/P² from largest contour               │
│    → Rejects slices of noise / partial scans   │
│    → Brain cross-sections score 0.7–0.95        │
│                                                  │
│  Also: Laplacian variance check (blur)          │
│  Result: ~6.5% of slices rejected              │
└───────────────────────┬──────────────────────────┘
                        │
                        ↓
┌──────────────────────────────────────────────────┐
│ STEP 3: BRAIN ROI CROP                           │
│  • Contour bounding box + 12px padding          │
│  • Removes black background                     │
│  • Critical for LIME quality (no bg superpixels)│
│  • Falls back to original if no contour found  │
└───────────────────────┬──────────────────────────┘
                        │
                        ↓
┌──────────────────────────────────────────────────┐
│ STEP 4: CNN INFERENCE                            │
│  Architecture: 5-layer CNN                      │
│  Input: (1, 3, 224, 224) per slice             │
│  Transform: Resize(224) → ToTensor → Normalize  │
│    mean=[0.129,0.129,0.129]                    │
│    std=[0.174,0.174,0.174]                     │
│  Output: softmax → [P(TC), P(ASD)]             │
│  Device: CPU (no CUDA required)                │
│  Speed: ~50 slices/second on modern CPU        │
└───────────────────────┬──────────────────────────┘
                        │
                        ↓
┌──────────────────────────────────────────────────┐
│ STEP 5: SUBJECT-LEVEL AGGREGATION               │
│                                                  │
│  quality_score = circularity×0.4               │
│               + brain_frac/0.5×0.3             │
│               + lap_var/1000×0.2               │
│               + std_i/0.2×0.1                  │
│                                                  │
│  weight_i = quality_score_i × max(P(ASD), P(TC))│
│                                                  │
│  P(ASD)_subject = Σ(P(ASD)_i × w_i) / Σw_i    │
│                                                  │
│  Prediction: ASD if P(ASD)_subject ≥ 0.5       │
│  Confidence: P(ASD) if ASD, else 1-P(ASD)     │
└───────────────────────┬──────────────────────────┘
                        │
                        ↓
┌──────────────────────────────────────────────────┐
│ STEP 6: TOP-K SLICE SELECTION FOR XAI           │
│  Rank all valid slices by quality_score         │
│  Select top-K (default K=3, configurable 1–5)  │
│  These are the slices shown in the UI           │
└───────────────────────┬──────────────────────────┘
                        │
                   ┌────┴────┐
                   │         │
                   ↓         ↓
┌─────────────────────┐  ┌───────────────────────┐
│ GRAD-CAM            │  │ LIME                  │
│                     │  │                       │
│ Target: conv5       │  │ 500 perturbed samples │
│ Backprop gradients  │  │ Local linear model    │
│ → class activation  │  │ → superpixel weights  │
│ heatmap (224×224)   │  │                       │
│                     │  │ ~20s/slice on CPU     │
│ Energy: fraction of │  │ Optional (toggle)     │
│ pixels > 0.5        │  │                       │
└──────────┬──────────┘  └────────────┬──────────┘
           │                          │
           └─────────┬────────────────┘
                     ↓
┌──────────────────────────────────────────────────┐
│ STEP 7: MC-DROPOUT UNCERTAINTY                  │
│  Dropout2d stays active at inference            │
│  30 stochastic forward passes                  │
│  Output: mean P(ASD), std σ, entropy           │
│  σ < 0.02 → Low uncertainty                   │
│  σ > 0.08 → High (flag for clinical review)   │
└───────────────────────┬──────────────────────────┘
                        │
                        ↓
┌──────────────────────────────────────────────────┐
│ STEP 8: PHENOTYPIC METADATA LOOKUP              │
│  Match anonymised ID from filename              │
│  → age, sex, site, true label from CSV         │
│  → site-specific validation performance         │
│  Site reliability: green if sens≥0.92 & spec≥0.95│
└───────────────────────┬──────────────────────────┘
                        │
                        ↓
┌──────────────────────────────────────────────────┐
│ STEP 9: LLM NARRATIVE (Claude Haiku)            │
│  Model: claude-haiku-4-5                        │
│  Cost: ~$0.002/call                             │
│  Inputs passed to LLM:                         │
│    • prediction + confidence + vote counts     │
│    • per-slice GradCAM energy + region + CoM  │
│    • LIME status (ran / not ran)               │
│    • MC-Dropout σ + assessment                │
│    • site performance + sex note              │
│    • ground truth match (ABIDE-I)             │
│  Output: 3-paragraph clinical interpretation  │
│  Adaptive: content changes with every input   │
└───────────────────────┬──────────────────────────┘
                        │
                        ↓
┌──────────────────────────────────────────────────┐
│ STEP 10: PDF REPORT GENERATION (ReportLab)      │
│  Page 1: prediction card, subject metadata,    │
│          model performance table               │
│  Page 2: GradCAM figures (all top-K slices)   │
│          LLM narrative (3 paragraphs)          │
│          Regulatory disclaimer                │
│  Pre-generated at pipeline completion         │
│  Instant download (no rerun required)         │
└─────────────────────────────────────────────────┘

OUTPUT: Subject-level prediction (ASD/TC) + confidence
        GradCAM/LIME figures × K slices
        MC-Dropout uncertainty estimate
        LLM clinical narrative
        Downloadable PDF report
```

---

## Training Pipeline

```
ABIDE-I PNGs + Label CSVs
        ↓
Apply quality filter (src/quality_filter.py)
        → clean_train.csv (67,625 slices)
        → clean_val.csv   (7,507 slices)
        → clean_test.csv  (18,814 slices)
        ↓
ASDDataset (src/dataset.py)
        → preprocess_cnn: resize(224) → 3ch → normalize
        → DataLoader(batch=64, shuffle=True)
        ↓
5-layer CNN (src/models.py: ASD_CNN)
NAdam (lr=1e-3, β₁=0.9, β₂=0.999)
CrossEntropyLoss
        ↓
Training loop (src/train.py)
        → 50 epochs max, early stopping (patience=5)
        → Best val accuracy checkpoint saved
        ↓
Evaluation (src/evaluate.py)
        → AUC, AUPRC, Brier, subgroup analysis
        ↓
xai_cnn_best_weights.pth
```

---

## Key Design Decisions

**Why 2D axial slices instead of 3D volumes?**  
ABIDE-I has 1,067 subjects — insufficient for 3D CNNs or volumetric transformers which require much larger datasets. 2D slice-level classification scales the effective dataset size from 1,067 to 100,510 samples. The trade-off is loss of inter-slice context, which is a documented limitation.

**Why weighted aggregation instead of majority vote?**  
Simple majority vote treats all slices equally. Quality-weighted voting down-weights uninformative slices (low coverage, blurry, small ROI) and up-weights high-quality central slices, producing more reliable subject-level predictions.

**Why Grad-CAM target layer conv5?**  
The final convolutional layer has the highest semantic abstraction and the coarsest spatial resolution (7×7), producing the most interpretable spatial attention maps. Earlier layers have finer resolution but lower semantic content.

**Why 500 LIME samples?**  
Empirically sufficient for stable superpixel attributions on 224×224 MRI slices while remaining tractable on CPU (~15–20 seconds). Increasing to 1000 produces marginally better results at 2× compute cost.

**Why NAdam over Adam/RMSprop?**  
All three optimizers produced comparable results in systematic comparison (AUC 0.97–0.98 with B.Tech baseline). NAdam was selected as the canonical optimizer for the graduate system due to slightly higher F1 consistency across runs. The choice has minor practical impact.
