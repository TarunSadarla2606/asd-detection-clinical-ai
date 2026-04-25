# narrator.py — v4
# LLM-generated clinical narratives for all three tabs
# Uses Claude Haiku. Every function sends rich structured data to the LLM —
# no hardcoded templates in the happy path.

import os
import anthropic

# ── Shared client helper ──────────────────────────────────────────────────────

def _get_client():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return None, "API key not set — set ANTHROPIC_API_KEY env variable."
    return anthropic.Anthropic(api_key=key), None


def _call_haiku(system_prompt: str, user_prompt: str,
                max_tokens: int = 900) -> str:
    client, err = _get_client()
    if err:
        return f"[Narrative unavailable: {err}]"
    try:
        msg = client.messages.create(
            model     = "claude-haiku-4-5",
            max_tokens= max_tokens,
            system    = system_prompt,
            messages  = [{"role": "user", "content": user_prompt}]
        )
        return msg.content[0].text.strip()
    except anthropic.AuthenticationError:
        return "[Narrative unavailable: API authentication failed — check ANTHROPIC_API_KEY.]"
    except anthropic.RateLimitError:
        return "[Narrative unavailable: API rate limit reached — try again shortly.]"
    except Exception as e:
        return f"[Narrative unavailable: {str(e)[:100]}]"


# ── Shared subject/site block builder ────────────────────────────────────────

def _build_subject_block(result: dict, subject_info: dict | None) -> str:
    if not subject_info:
        return ("Subject demographics unavailable — no phenotypic match found. "
                "Site-specific reliability and demographic considerations cannot be assessed.")
    si = subject_info
    pred = result['subject_pred']
    truth = si.get('true_label', 'Unknown')
    match_str = 'MATCHES' if pred == truth else 'DIFFERS FROM'
    s = (f"Age {si['age']} years, Sex: {si['sex']}, Site: {si['site']}. "
         f"ABIDE-I ground truth: {truth}. "
         f"Model prediction {match_str} ground truth. ")
    if si.get('site_perf'):
        sp = si['site_perf']
        low = sp['Sensitivity'] < 0.92 or sp['Specificity'] < 0.95
        s += (f"Site {si['site']} validation (n={sp['N_subjects']}): "
              f"Sensitivity {sp['Sensitivity']:.1%}, Specificity {sp['Specificity']:.1%}, "
              f"AUC {sp['AUC']:.3f}. "
              f"{'Site performance BELOW reliability threshold — extra caution warranted.' if low else 'Site within reliable range.'} ")
    else:
        s += f"Site {si['site']} not in validation cohort — site-specific reliability unknown. "
    if si.get('sex') == 'Female':
        s += ("Sex note: Female subjects are underrepresented in training data "
              "(n=164 vs 948 male). Model sensitivity for females: 93.4% vs 95.8% for males.")
    else:
        s += "Sex note: Male subject — primary demographic of training cohort."
    return s


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CNN narrative (existing function, signature unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def generate_clinical_narrative(
    result       : dict,
    subject_info : dict | None,
    uncertainty  : dict | None,
    gradcam_stats: list[dict],
    lime_stats   : list[dict] | None = None,
) -> str:
    """
    3-paragraph clinical narrative for the CNN model (Tab 1).
    Paragraph 1: Prediction & vote consensus
    Paragraph 2: GradCAM, LIME (if run), MC-Dropout
    Paragraph 3: Site, demographics, limitations, disclaimer
    """
    pred      = result['subject_pred']
    conf      = result['subject_conf']
    prob_asd  = result['weighted_prob_asd']
    asd_votes = result['pred_asd_votes']
    tc_votes  = result['pred_tc_votes']
    total     = result['total_valid']
    vote_pct  = asd_votes / total * 100 if total > 0 else 0

    demo_str = _build_subject_block(result, subject_info)

    gc_lines = []
    for gc in gradcam_stats[:5]:
        gc_lines.append(
            f"  Slice z={gc['slice_idx']}: P(ASD)={gc['prob_asd']:.4f}, "
            f"GradCAM_energy={gc['energy']:.4f}, region={gc.get('region','unknown')}, "
            f"centre_of_mass=({gc.get('com_x',0):.2f},{gc.get('com_y',0):.2f})"
        )
    gc_str = "\n".join(gc_lines) or "  No GradCAM data available."

    if lime_stats and lime_stats[0].get('lime_ran'):
        lime_str = (f"LIME WAS RUN on {len(lime_stats)} slices. Superpixel perturbation "
                    "analysis was performed alongside GradCAM. Discuss concordance or "
                    "divergence between methods and what it implies for explanation confidence.")
    else:
        lime_str = ("LIME WAS NOT RUN — only GradCAM is available. Note this as a limitation: "
                    "LIME provides complementary perturbation-based evidence that would either "
                    "corroborate or challenge GradCAM findings.")

    unc_str = (
        f"MC-Dropout ({uncertainty.get('n_passes',30)} passes): "
        f"mean P(ASD)={uncertainty['mean_prob_asd']:.4f}, std={uncertainty['std']:.4f}. "
        f"Assessment: {uncertainty['uncertainty']}."
        if uncertainty else "MC-Dropout uncertainty was not computed."
    )

    system_prompt = """You are a clinical AI research assistant. Translate structured brain MRI 
analysis outputs into clear, appropriately cautious clinical language for a neurologist reviewing 
AI-assisted findings. Rules: write in third person ("The model predicts..."), reference specific 
numbers, never diagnose — interpret model outputs only, write plain flowing paragraphs (no headers, 
no bold, no markdown), each paragraph 3-4 sentences. The third paragraph must end with exactly: 
"This report is research-grade only and must not inform clinical decision-making." """

    user_prompt = f"""Write a 3-paragraph clinical interpretation of these CNN-based ASD detection results.

PREDICTION
- Result: {pred} (ASD=Autism Spectrum Disorder, TC=Typical Control)
- Weighted confidence: {conf:.1%} | Weighted P(ASD): {prob_asd:.4f}
- Slice votes: {asd_votes} ASD / {tc_votes} TC from {total} valid slices ({vote_pct:.1f}% ASD)
- Total NIfTI slices: {result['n_total_slices']}

SUBJECT AND SITE
{demo_str}

GRAD-CAM (gradient-weighted class activation per slice)
{gc_str}

LIME (superpixel perturbation)
{lime_str}

UNCERTAINTY (MC-Dropout)
{unc_str}

MODEL CONTEXT
- Architecture: 5-layer CNN | Dataset: ABIDE-I (1,067 subjects, 17 sites, 2D axial sMRI)
- Validation: AUC 0.994, Sensitivity 95.6%, Specificity 97.2%, Brier 0.027

Paragraph 1 — PREDICTION AND CONSENSUS: prediction, confidence, exact vote counts, ground truth match, 
what the vote margin implies about consensus strength.
Paragraph 2 — EXPLAINABILITY AND UNCERTAINTY: GradCAM energy and anatomical regions per slice, 
LIME findings or absence, MC-Dropout uncertainty and clinical implications.
Paragraph 3 — CLINICAL CONTEXT: site reliability, demographic considerations, 2D slice-level limitation, 
steps required before clinical use. End exactly: 
"This report is research-grade only and must not inform clinical decision-making." """

    return _call_haiku(system_prompt, user_prompt, max_tokens=900)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Hybrid CNN-ViT narrative
# ══════════════════════════════════════════════════════════════════════════════

def generate_hybrid_narrative(
    result         : dict,
    subject_info   : dict | None,
    uncertainty    : dict | None,     # from mc_dropout_hybrid()
    attn_stats     : list[dict],      # list of get_attention_rollout() outputs per slice
) -> str:
    """
    3-paragraph clinical narrative for the Hybrid CNN-ViT model (Tab 2).

    Key difference from CNN narrative:
    - Explains self-attention mechanism and what it captures vs GradCAM
    - Contextualises the lower overall AUC (0.943 vs 0.994) honestly
    - Interprets high MC-Dropout sigma (typical σ≈0.24 on ASD positives)
      as architectural feature: both CNN and transformer dropouts contribute variance
    - Reports brain token coverage as data quality signal
    """
    pred     = result['subject_pred']
    conf     = result['subject_conf']
    prob_asd = result['weighted_prob_asd']
    total    = result['total_valid']
    asd_v    = result['pred_asd_votes']
    tc_v     = result['pred_tc_votes']
    vote_pct = asd_v / total * 100 if total > 0 else 0

    demo_str = _build_subject_block(result, subject_info)

    attn_lines = []
    for i, a in enumerate(attn_stats[:5]):
        attn_lines.append(
            f"  Slice {i+1}: brain_tokens={a.get('n_brain',0)}/196 "
            f"({a.get('n_brain',0)/196*100:.1f}% brain coverage at token resolution)"
        )
    attn_str = "\n".join(attn_lines) or "  No attention data available."

    unc_str = (
        f"MC-Dropout (CNN Dropout2d + Transformer Dropout, {uncertainty.get('n_passes',30)} passes): "
        f"mean P(ASD)={uncertainty['mean_prob_asd']:.4f}, std={uncertainty['std']:.4f}. "
        f"Assessment: {uncertainty['uncertainty']}. "
        f"Note: sigma values characteristically higher in the hybrid model than in the pure CNN "
        f"because both dropout pathways contribute stochastic variance."
        if uncertainty else "MC-Dropout uncertainty was not computed."
    )

    system_prompt = """You are a clinical AI research assistant explaining a Hybrid CNN-ViT 
model for brain MRI analysis. This architecture combines a CNN feature extractor with a transformer 
that uses self-attention over spatial brain regions. Write in third person, plain paragraphs, 
no headers, no bold, no markdown. Reference specific numbers. Never diagnose. End the third 
paragraph exactly with: "This report is research-grade only and must not inform clinical decision-making." """

    user_prompt = f"""Write a 3-paragraph clinical interpretation of these Hybrid CNN-ViT ASD detection results.

PREDICTION
- Result: {pred} | Confidence: {conf:.1%} | Weighted P(ASD): {prob_asd:.4f}
- Votes: {asd_v} ASD / {tc_v} TC from {total} valid slices ({vote_pct:.1f}% ASD)

ARCHITECTURE CONTEXT (explain this accurately to a neurologist)
- The Hybrid CNN-ViT uses a pretrained CNN (conv1-conv4) to extract spatial feature tokens from 
  the MRI slice, then a 4-layer Transformer encoder learns global self-attention relationships 
  across those tokens. Background/skull tokens are masked before the transformer sees them.
- Validation performance: AUC 0.943, Sensitivity 84.5%, Specificity 86.8% (vs CNN: AUC 0.994).
- The attention maps from the CLS classification token directly show which spatial locations 
  the model weights when making its decision — this is different from GradCAM gradients, 
  which show local intensity changes. Self-attention captures global spatial relationships 
  across the entire slice simultaneously.
- The model was trained for 30 epochs with differential learning rates (backbone frozen initially).

SUBJECT AND SITE
{demo_str}

BRAIN TOKEN COVERAGE (data quality signal)
{attn_str}

UNCERTAINTY (MC-Dropout — both CNN and transformer dropouts active)
{unc_str}

Paragraph 1 — PREDICTION AND ARCHITECTURE: What the hybrid model predicted, with what confidence, 
from how many slices. Briefly explain in plain clinical language what a CNN-Transformer hybrid does 
differently from a pure CNN. Note the lower AUC vs the CNN (0.943 vs 0.994) honestly and explain 
why this is expected for transformer models in limited-data regimes (~67K slices).

Paragraph 2 — SPATIAL ATTENTION AND UNCERTAINTY: Describe what the self-attention mechanism 
highlights in spatial terms. Explain what the brain token coverage numbers mean as a data quality 
signal. Interpret the MC-Dropout uncertainty — specifically explain why σ is characteristically 
higher in this hybrid model, and what that means clinically (the model is exploring multiple spatial 
hypotheses simultaneously via its attention mechanism).

Paragraph 3 — CLINICAL CONTEXT AND COMPARISON: Compare what attention maps provide vs GradCAM 
(global spatial relationships vs local gradient sensitivity). Note site and demographic considerations. 
State the 2D slice-level limitation and what additional validation would be required. End exactly: 
"This report is research-grade only and must not inform clinical decision-making." """

    return _call_haiku(system_prompt, user_prompt, max_tokens=950)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Ensemble narrative
# ══════════════════════════════════════════════════════════════════════════════

def generate_ensemble_narrative(
    cnn_result     : dict,
    hybrid_result  : dict,
    ensemble_result: dict,
    subject_info   : dict | None,
    cnn_uncertainty: dict | None,
    hybrid_uncertainty: dict | None,
    xai_agreement  : dict | None,     # from compute_xai_agreement()
) -> str:
    """
    3-paragraph clinical narrative for the Ensemble tab (Tab 3).

    This is the richest prompt: it reasons about WHY two architecturally
    different models agree or disagree, what that implies for clinical confidence,
    and how the uncertainty-gated weighting works.
    """
    cnn_pred   = cnn_result['subject_pred']
    cnn_conf   = cnn_result['subject_conf']
    cnn_prob   = cnn_result['weighted_prob_asd']
    hyb_pred   = hybrid_result['subject_pred']
    hyb_conf   = hybrid_result['subject_conf']
    hyb_prob   = hybrid_result['weighted_prob_asd']
    ens_pred   = ensemble_result['ensemble_pred']
    ens_prob   = ensemble_result['ensemble_prob_asd']
    ens_w_cnn  = ensemble_result['weight_cnn']
    ens_w_vit  = ensemble_result['weight_vit']
    disagree   = cnn_pred != hyb_pred
    prob_diff  = abs(cnn_prob - hyb_prob)
    cnn_sig    = cnn_uncertainty['std']   if cnn_uncertainty   else None
    hyb_sig    = hybrid_uncertainty['std'] if hybrid_uncertainty else None
    agree_text = xai_agreement['interpretation'] if xai_agreement else "XAI agreement not computed."
    iou        = xai_agreement['iou'] if xai_agreement else None

    demo_str = _build_subject_block(cnn_result, subject_info)

    system_prompt = """You are a clinical AI research scientist explaining an ensemble of two 
complementary neural networks for brain MRI analysis. Your audience is a neurologist or radiologist. 
Write in third person, plain paragraphs, no headers, no bold, no markdown. Reference specific 
numbers throughout. Never diagnose — interpret model outputs. Reason carefully about what model 
agreement and disagreement mean clinically. End the third paragraph exactly with: 
"This report is research-grade only and must not inform clinical decision-making." """

    disagreement_section = ""
    if disagree:
        disagreement_section = f"""
IMPORTANT — MODEL DISAGREEMENT DETECTED
The CNN predicts {cnn_pred} (P(ASD)={cnn_prob:.4f}) while the Hybrid CNN-ViT predicts 
{hyb_pred} (P(ASD)={hyb_prob:.4f}). Probability difference: {prob_diff:.4f}.
When architecturally diverse models trained on identical data disagree, it indicates that 
the scan contains ambiguous features that different representational strategies resolve 
differently. The CNN, using local gradient-based features, and the transformer, using 
global spatial attention, have reached different conclusions. This disagreement itself is 
clinically important signal — it does not resolve to a confident prediction and should 
trigger mandatory expert radiologist review before any use."""
    else:
        disagreement_section = f"""
MODEL AGREEMENT
Both models independently predict {cnn_pred}. CNN P(ASD)={cnn_prob:.4f}, 
Hybrid P(ASD)={hyb_prob:.4f}, difference={prob_diff:.4f}.
Agreement between a CNN (local gradient features) and a transformer (global spatial attention) 
trained on the same data using different inductive biases provides stronger evidence than 
either model alone. The ensemble confidence is correspondingly higher."""

    user_prompt = f"""Write a 3-paragraph clinical interpretation of this two-model ensemble analysis.

CNN MODEL (5-layer CNN, primary model)
- Prediction: {cnn_pred} | Confidence: {cnn_conf:.1%} | P(ASD)={cnn_prob:.4f}
- Votes: {cnn_result['pred_asd_votes']} ASD / {cnn_result['pred_tc_votes']} TC 
  from {cnn_result['total_valid']} slices
- Validation: AUC 0.994, Sensitivity 95.6%, Specificity 97.2%
- MC-Dropout σ: {f"{cnn_sig:.4f}" if cnn_sig is not None else "not computed"}

HYBRID CNN-ViT MODEL (CNN backbone + Transformer encoder)
- Prediction: {hyb_pred} | Confidence: {hyb_conf:.1%} | P(ASD)={hyb_prob:.4f}
- Votes: {hybrid_result['pred_asd_votes']} ASD / {hybrid_result['pred_tc_votes']} TC
  from {hybrid_result['total_valid']} slices
- Validation: AUC 0.943, Sensitivity 84.5%, Specificity 86.8%
- MC-Dropout σ: {f"{hyb_sig:.4f}" if hyb_sig is not None else "not computed"}

UNCERTAINTY-GATED ENSEMBLE FUSION
- Method: weights inversely proportional to MC-Dropout sigma, then scaled by baseline AUC
- CNN weight: {ens_w_cnn:.3f} | ViT weight: {ens_w_vit:.3f}
- Ensemble P(ASD): {ens_prob:.4f} | Ensemble prediction: {ens_pred}
- The ensemble dynamically upweights whichever model is more certain on this specific scan.
  If CNN sigma is low and ViT sigma is high, CNN dominates; the reverse when ViT is more certain.

{disagreement_section}

SUBJECT AND SITE
{demo_str}

XAI SPATIAL AGREEMENT (CNN GradCAM vs ViT Attention Rollout)
{agree_text}
{f"Jaccard IoU between top-25% activation regions: {iou:.3f}" if iou is not None else ""}
Interpretation: CNN uses gradient saliency (local, layer-specific), ViT uses global self-attention 
(simultaneous, full-slice). Different methods capturing different aspects is expected and valuable.

Paragraph 1 — INDEPENDENT PREDICTIONS AND ENSEMBLE FUSION: Describe what each model independently 
predicted, with exact probabilities and vote counts. Explain the uncertainty-gated ensemble fusion 
mechanism in plain terms — what the weights {ens_w_cnn:.3f}/{ens_w_vit:.3f} mean and why they 
differ. State the ensemble verdict clearly.

Paragraph 2 — MODEL AGREEMENT/DISAGREEMENT AND XAI: {"Explain the clinical significance of the disagreement between the two models and what it implies about the certainty of the prediction. Stress that this ambiguity warrants expert review." if disagree else "Explain why agreement between a CNN and a transformer trained on identical data is stronger evidence than either alone, and what complementary inductive biases each model brings."} 
Then describe the XAI spatial agreement — what the IoU value means, and whether the two 
explanation methods are showing the same or different brain regions as discriminative.

Paragraph 3 — CLINICAL SYNTHESIS: Synthesise the overall clinical picture across both models 
and the ensemble. Note site and demographic factors. State what a radiologist should do with this 
result. {"Specifically recommend mandatory expert review given the model disagreement." if disagree else "Note that model agreement increases but does not guarantee clinical reliability."}
End exactly: "This report is research-grade only and must not inform clinical decision-making." """

    return _call_haiku(system_prompt, user_prompt, max_tokens=1050)


# ── Fallback (used only on API failure) ──────────────────────────────────────

def _fallback(result: dict, subject_info: dict | None,
              tab: str = 'CNN', note: str = '') -> str:
    pred  = result['subject_pred']
    conf  = result['subject_conf']
    total = result['total_valid']
    votes = result['pred_asd_votes']
    demo  = ""
    if subject_info:
        si   = subject_info
        demo = f" Subject: {si.get('age')}-year-old {si.get('sex')}, site {si.get('site')}."
    text = (f"[{tab} model] The model predicts {pred} with {conf:.1%} confidence "
            f"from {total} valid slices ({votes} ASD votes).{demo} "
            f"This is a research-grade result and must not inform clinical decision-making.")
    if note:
        text += f"\n\n[System note: {note}]"
    return text
