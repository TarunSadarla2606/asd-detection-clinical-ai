# Model Card — ASD Detection from sMRI (Clinical AI System)

**Version:** 2.0 (Graduate System)  
**Date:** April 2026  
**Author:** Tarun Sadarla — University of North Texas (MS AI, Biomedical Concentration)  
**Contact:** GitHub — [@TarunSadarla2606](https://github.com/TarunSadarla2606)  
**Live Demo:** https://huggingface.co/spaces/TarunSadarla2606/asd-detection-demo

---

## Model Details

| Field | Value |
|---|---|
| Architecture | 5-layer CNN with LeakyReLU, BatchNorm, Dropout2d |
| Optimiser | NAdam (lr=1e-3, β₁=0.9, β₂=0.999, ε=1e-8) |
| Training epochs | 50 (early stopping, patience=5) |
| Input | 2D axial sMRI slices, 224×224, 3-channel (grayscale replicated) |
| Output | Binary: ASD (1) vs Typical Control (0) |
| Trainable parameters | 1,648,270 |
| Framework | PyTorch 2.6.0 |
| XAI methods | Grad-CAM (gradient-based) + LIME (perturbation-based) |
| Uncertainty | MC-Dropout (30 stochastic passes, Dropout2d) |
| Weights | Hosted on Hugging Face Spaces (Git LFS) |

### HybridCNNViT (Research Model)

| Field | Value |
|---|---|
| Architecture | CNNBackbone (conv1–conv4) + 4-block Transformer (8 heads, embed_dim=256) |
| Optimiser | AdamW, two-phase training |
| Training epochs | 60 |
| Input | 2D axial sMRI slices, 224×224, 3-channel |
| Output | Binary: ASD (1) vs Typical Control (0) |
| Trainable parameters | ~4,200,000 |
| Framework | PyTorch 2.6.0 |
| XAI methods | Attention Rollout (CLS→token, head-averaged) + GradCAM on backbone |
| AUC-ROC | 0.997 (vs CNN: 0.994) |
| Weights | Hosted on Hugging Face Spaces (Git LFS) |

---

## Intended Use

**Primary intended use:** Research tool to investigate the feasibility of deep learning-based ASD biomarker detection from structural MRI, and to demonstrate clinical AI best practices (explainability, uncertainty quantification, governance documentation, and regulatory framing) as a graduate capstone project.

**Intended users:** Academic researchers, AI/ML practitioners in neuroimaging, and hiring/clinical audiences evaluating graduate-level medical AI work.

**Out-of-scope uses (explicitly not for):**
- Clinical diagnosis or screening of ASD in any patient population
- Deployment in any medical device, EHR system, or clinical decision support tool
- Use on imaging data acquired at sites not represented in ABIDE-I
- Use in paediatric populations under age 7 or adults over age 64
- Replacement of or adjunct to behavioural and clinical ASD assessment (ADOS-2, ADI-R)

---

## Training Data

| Field | Value |
|---|---|
| Dataset | Autism Brain Imaging Data Exchange I (ABIDE-I) |
| Source | https://fcon_1000.projects.nitrc.org/indi/abide/ |
| Original size | 1,112 subjects from 17 international sites |
| After subject cleaning | 1,067 subjects (removed missing/unclear sMRI and unknown labels) |
| Input pipeline | NIfTI (.nii.gz) → PNG slices → 4-metric quality filter |
| Slices after quality filter | Train 67,625 · Val 7,507 · Test 18,814 |
| Label encoding | ASD=1, Typical Control (TC)=0 |
| Age range | 7–64 years (median 14.7) |
| Sex distribution | ~85% male (948 male / 164 female subjects) |

**Data limitations:**
- ABIDE-I is heavily male-skewed (~85% male) — limits generalisability to female subjects
- Multi-site acquisition (17 sites) introduces scanner heterogeneity as a potential confound
- All subjects retrospectively aggregated — no prospective data collection
- Slice-level classification: each 2D axial slice treated independently, losing inter-slice volumetric context

---

## Performance (Test Set — Slice Level)

| Metric | Value |
|---|---|
| AUC-ROC | 0.9944 |
| AUPRC | 0.9943 |
| Accuracy | 0.9644 |
| Sensitivity (Recall) | 0.9559 |
| Specificity | 0.9725 |
| Precision (PPV) | 0.9705 |
| F1 Score | 0.9631 |
| Brier Score | 0.0270 |
| False Negative Rate | 0.0441 |
| False Positive Rate | 0.0275 |
| Total test slices | 18,814 |

**Important:** Slice-level metrics ≠ subject-level clinical performance. The deployed system uses subject-level aggregation via quality-weighted confidence voting. Subject-level performance evaluation with confidence intervals on a held-out cohort would be required before any clinical consideration.

---

## Subgroup Performance

### Sex

| Group | N Subjects | N Slices | Sensitivity | Specificity | AUC |
|---|---|---|---|---|---|
| Male | 948 | 16,415 | 0.9583 | 0.9735 | 0.9950 |
| Female | 164 | 2,399 | **0.9345** | 0.9666 | 0.9891 |

The 2.4 percentage point sensitivity gap is directly attributable to the training data imbalance (948 male vs 164 female subjects). Performance on female subjects should be interpreted with additional caution.

### Acquisition Site (7 sites with sufficient representation)

| Site | N Subjects | Sensitivity | Specificity | AUC |
|---|---|---|---|---|
| USM | 59 | 0.9636 | 0.9922 | 0.9985 |
| UM_1 | 81 | **0.9846** | 0.9796 | 0.9974 |
| NYU | 73 | 0.9517 | 0.9691 | 0.9910 |
| UM_2 | 61 | 0.9024 | **0.9852** | 0.9912 |
| OLIN | 54 | 0.9795 | 0.9961 | 0.9990 |
| OHSU | 55 | 1.0000† | 0.9504 | 1.0000† |
| PITT | 56 | **0.8854** | 0.9561 | 0.9761 |

†OHSU: only 18 ASD slices — sensitivity estimate is statistically unreliable.

The 9.4 percentage point sensitivity gap between PITT (88.5%) and UM_1 (98.5%) is driven by scanner and acquisition protocol differences between sites. Site-specific calibration or harmonisation would be required before clinical deployment.

---

## Explainability

### Grad-CAM

Gradient-weighted Class Activation Mapping applied to conv5 (final convolutional layer). Produces spatial heatmaps showing which regions of the axial slice contributed most to the prediction.

**Limitation:** At very high confidence (softmax → 1.0), backpropagated gradients approach zero — GradCAM produces flat heatmaps (gradient saturation). This is expected mathematical behaviour, not a bug. GradCAM++ would address this for high-confidence cases.

### LIME

Local Interpretable Model-Agnostic Explanations — 500-sample superpixel perturbation. Identifies which image regions causally affect the prediction. Slower (~20s/slice on CPU) but provides complementary perturbation-based evidence.

### GradCAM × LIME Agreement

| Case | Pearson r | IoU (top-25%) | Interpretation |
|---|---|---|---|
| TC — Correct (TN) | 0.564 | 0.445 | Moderate agreement — supports explanation confidence |
| ASD — Missed (FN) | −0.286 | 0.042 | Anti-correlated — unstable model attention on this prediction |
| TC — False Alarm (FP) | 0.035 | 0.185 | Low agreement — explanation unreliable |

Disagreement between GradCAM and LIME on a prediction should be treated as a signal requiring additional scrutiny, not as a reason to trust either explanation independently.

---

## Known Limitations

1. **2D slice-level classification** — the model sees individual axial slices, not full 3D volumes. Inter-slice spatial relationships and full volumetric context are lost.

2. **Site confound** — with 17 sites and variable acquisition protocols, the model may learn site-specific imaging characteristics in addition to neurobiological ASD features. Scanner bias cannot be excluded without cross-site validation experiments.

3. **Sex imbalance** — ~85% male training data. Sensitivity for female subjects (93.4%) is 2.4 pp below male subjects (95.8%). Not representative of clinical ASD populations.

4. **No prospective validation** — all results from a retrospective held-out test set drawn from the same distribution as training data. Real-world performance on new sites, scanners, or protocols is unknown.

5. **XAI interpretability bounds** — GradCAM activations reflect what the model attends to, not necessarily what is clinically meaningful. Highlighted regions have not been validated against neuroradiologist annotations.

6. **Confident wrong predictions** — the false positive case in the failure mode analysis shows σ = 0.005 (low uncertainty) while being a misclassification. MC-Dropout uncertainty does not reliably flag all incorrect predictions.

7. **sMRI only** — ASD has multimodal neurobiological correlates (fMRI functional connectivity, white matter microstructure, cortical thickness). Structural MRI alone has limited sensitivity for high-functioning ASD presentations.

---

## Ethical Considerations

- ABIDE-I data is de-identified and shared under the 1000 Functional Connectomes Project data-sharing agreement. No additional patient data was collected for this project.
- ASD is a spectrum condition with significant heterogeneity. Binary classification necessarily oversimplifies the diagnostic landscape.
- Automated ASD screening tools carry risk of over-medicalisation or stigmatisation if deployed without appropriate clinical oversight.
- The system's PDF report and UI prominently display the research-grade disclaimer and out-of-scope uses on every prediction.

---

## Regulatory Framing

As a software system that analyses medical images to inform clinical decisions regarding ASD, this system would be classified as **Software as a Medical Device (SaMD) Class II** under FDA 21st Century Cures Act guidance and the FDA Digital Health Center of Excellence framework. The **De Novo pathway** would likely be required given the novel intended use.

**Current status: Research-grade only.** Before any clinical deployment:

1. Prospective multi-site validation on data independent of ABIDE-I
2. Subject-level (not slice-level) performance evaluation with confidence intervals
3. Comparative performance against existing clinical assessments (ADOS-2, ADI-R)
4. De-biasing or stratified training to address sex and site performance disparities
5. Neuroradiologist validation of XAI saliency maps against anatomical ground truth
6. IRB approval for any prospective data collection
7. A 510(k) or De Novo premarket submission with full clinical evidence package

This project demonstrates technical feasibility and responsible AI documentation practices in the context of a graduate research project. It is not and does not claim to be a validated clinical tool.
