# narrator.py — v3 fully adaptive
# LLM-generated clinical narrative using Claude Haiku
# Adapts to: prediction, confidence, votes, demographics, site,
#            GradCAM energy+region, LIME (if run), MC-Dropout uncertainty

import os
import anthropic


def generate_clinical_narrative(
    result        : dict,
    subject_info  : dict | None,
    uncertainty   : dict | None,
    gradcam_stats : list[dict],
    lime_stats    : list[dict] | None = None,
) -> str:
    """
    Calls Claude Haiku to generate a 3-paragraph clinical interpretation.
    Every parameter is passed to the LLM — no hardcoded templates in the happy path.
    Falls back to a brief template only on API failure.

    Cost: ~$0.002 per call (Haiku 4.5 pricing)
    Latency: ~2-4 seconds
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return _fallback(result, subject_info,
                         note="API key not set — set ANTHROPIC_API_KEY env variable.")

    # ── Build structured data payload ──────────────────────────────────────
    pred      = result['subject_pred']
    conf      = result['subject_conf']
    prob_asd  = result['weighted_prob_asd']
    asd_votes = result['pred_asd_votes']
    tc_votes  = result['pred_tc_votes']
    total     = result['total_valid']
    n_total   = result['n_total_slices']
    vote_pct  = asd_votes / total * 100 if total > 0 else 0
    vote_ratio = f"{tc_votes}:{asd_votes}" if pred == 'TC' else f"{asd_votes}:{tc_votes}"

    # ── Subject & site ──────────────────────────────────────────────────────
    if subject_info:
        si       = subject_info
        truth    = si.get('true_label', 'Unknown')
        match    = 'MATCHES' if pred == truth else 'DIFFERS FROM'
        demo_str = (
            f"Age {si['age']} years, Sex: {si['sex']}, Site: {si['site']}. "
            f"ABIDE-I ground truth: {truth}. "
            f"Model prediction {match} ground truth."
        )
        if si.get('site_perf'):
            sp       = si['site_perf']
            low_site = sp['Sensitivity'] < 0.92 or sp['Specificity'] < 0.95
            demo_str += (
                f" Site {si['site']} validation cohort (n={sp['N_subjects']}): "
                f"Sensitivity {sp['Sensitivity']:.1%}, "
                f"Specificity {sp['Specificity']:.1%}, "
                f"AUC {sp['AUC']:.3f}. "
                f"{'Site performance BELOW reliability threshold — extra caution warranted.' if low_site else 'Site performance within reliable range.'}"
            )
        else:
            demo_str += f" Site {si['site']} not in validation cohort — site-specific reliability unknown."

        if si.get('sex') == 'Female':
            demo_str += (" Sex note: Female subjects are underrepresented in training data "
                         "(n=164 vs 948 male). Model sensitivity for females is 93.4% vs 95.8% for males.")
        else:
            demo_str += " Sex note: Male subject — within the primary demographic of the training cohort."
    else:
        demo_str = ("Subject demographics unavailable — no phenotypic match found. "
                    "Site-specific reliability and demographic considerations cannot be assessed.")

    # ── GradCAM per-slice ───────────────────────────────────────────────────
    gc_lines = []
    for gc in gradcam_stats[:5]:
        gc_lines.append(
            f"  Slice z={gc['slice_idx']}: "
            f"P(ASD)={gc['prob_asd']:.4f}, "
            f"GradCAM_energy={gc['energy']:.4f}, "
            f"anatomical_region={gc.get('region', 'unknown')}, "
            f"centre_of_mass=({gc.get('com_x', 0):.2f}, {gc.get('com_y', 0):.2f})"
        )
    gc_str = "\n".join(gc_lines) if gc_lines else "  No GradCAM data available."

    # ── LIME ───────────────────────────────────────────────────────────────
    # lime_stats is None when LIME was not run
    # lime_stats is a list with lime_ran=True when LIME was run
    if lime_stats and lime_stats[0].get('lime_ran'):
        n_slices_lime = len(lime_stats)
        lime_str = (
            f"LIME WAS RUN on {n_slices_lime} slices. "
            "Superpixel perturbation analysis was performed alongside GradCAM. "
            "Interpret whether the two methods show agreement or divergence in the "
            "regions they highlight. LIME complements GradCAM by testing which image "
            "patches are causally important (not just gradient-sensitive). "
            "Discuss what agreement or disagreement between the methods implies for "
            "confidence in the explanation."
        )
    else:
        lime_str = (
            "LIME WAS NOT RUN for this analysis — only GradCAM gradient-based "
            "explanations are available. Note this as a limitation: LIME would provide "
            "complementary perturbation-based evidence to either corroborate or "
            "challenge the GradCAM findings, and its absence reduces explanation robustness."
        )

    # ── Uncertainty ─────────────────────────────────────────────────────────
    if uncertainty:
        unc_str = (
            f"MC-Dropout ({uncertainty.get('n_passes', 30)} stochastic passes): "
            f"mean P(ASD)={uncertainty['mean_prob_asd']:.4f}, "
            f"std={uncertainty['std']:.4f}. "
            f"Assessment: {uncertainty['uncertainty']}. "
            f"{'High std (>0.05) indicates genuine model uncertainty — flag for clinical review.' if uncertainty['std'] > 0.05 else 'Low std confirms the prediction is stable across model samples.'}"
        )
    else:
        unc_str = "MC-Dropout uncertainty was not computed for this run."

    # ── System prompt ───────────────────────────────────────────────────────
    system_prompt = """You are a clinical AI research assistant. Your role is to translate 
structured automated brain MRI analysis outputs into clear, factual, appropriately cautious 
clinical language suitable for a neurologist or radiologist reviewing AI-assisted findings.

Strict rules:
- Write entirely in third person: "The model predicts...", "Grad-CAM reveals..."
- Reference specific numbers from the data — never make up values
- Adapt your interpretation to what the data actually shows
  (e.g. if energy is very low say so; if LIME ran, discuss it; if site is unknown say so)
- Never diagnose — interpret model outputs only
- Do not use markdown headers (##) or bold (**) in your output
- Write in plain flowing paragraphs
- Each paragraph: 3-4 sentences
- The third paragraph must end with exactly:
  "This report is research-grade only and must not inform clinical decision-making."
"""

    # ── User prompt ─────────────────────────────────────────────────────────
    user_prompt = f"""Write a 3-paragraph clinical interpretation of these automated 
ASD detection results. Use specific numbers. Write plain paragraphs — no headers, no bold.

PREDICTION
- Result: {pred} (ASD = Autism Spectrum Disorder, TC = Typical Control)
- Weighted confidence: {conf:.1%}
- Weighted P(ASD): {prob_asd:.4f}
- Slice voting: {asd_votes} ASD / {tc_votes} TC out of {total} valid slices
  ({vote_pct:.1f}% voted ASD; ratio {vote_ratio} in favour of prediction)
- Total slices in NIfTI volume: {n_total}

SUBJECT AND SITE
{demo_str}

GRAD-CAM (gradient-weighted spatial attention per slice)
{gc_str}

LIME (superpixel perturbation analysis)
{lime_str}

UNCERTAINTY
{unc_str}

MODEL CONTEXT
- Architecture: 5-layer CNN trained on ABIDE-I (1,067 subjects, 17 sites, 2D axial sMRI)
- Overall validation: AUC 0.994, Sensitivity 95.6%, Specificity 97.2%, Brier 0.027
- Sex-stratified: Male sensitivity 95.8%, Female sensitivity 93.4%
- Evaluation is slice-level, aggregated by weighted confidence voting

Paragraph 1 — PREDICTION AND CONSENSUS:
Describe what the model predicted, the confidence level, the vote distribution 
(exact numbers), whether it matches ground truth, and what the vote margin implies 
about consensus strength.

Paragraph 2 — EXPLAINABILITY AND UNCERTAINTY:
Describe the GradCAM activation energy values and anatomical regions for each slice.
If LIME ran: describe what it adds and whether GradCAM and LIME are concordant.
If LIME did not run: state it clearly as a limitation and explain what it would have added.
Then describe what the MC-Dropout uncertainty result means for this prediction.

Paragraph 3 — CLINICAL CONTEXT AND LIMITATIONS:
Describe site-specific reliability, demographic considerations for this subject,
the 2D slice-level limitation, and concrete steps required before any clinical use.
End this paragraph with exactly:
"This report is research-grade only and must not inform clinical decision-making."
"""

    # ── API call ─────────────────────────────────────────────────────────────
    try:
        client  = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model      = "claude-haiku-4-5",
            max_tokens = 750,
            system     = system_prompt,
            messages   = [{"role": "user", "content": user_prompt}]
        )
        return message.content[0].text.strip()

    except anthropic.AuthenticationError:
        return _fallback(result, subject_info,
                         note="API authentication failed — check ANTHROPIC_API_KEY.")
    except anthropic.RateLimitError:
        return _fallback(result, subject_info,
                         note="API rate limit reached — try again in a moment.")
    except Exception as e:
        return _fallback(result, subject_info,
                         note=f"API error: {str(e)[:80]}")


def _fallback(result: dict, subject_info: dict | None, note: str = "") -> str:
    """Minimal template fallback — used only when API is unreachable."""
    pred  = result['subject_pred']
    conf  = result['subject_conf']
    votes = result['pred_asd_votes']
    total = result['total_valid']

    demo = ""
    if subject_info:
        si   = subject_info
        demo = f" Subject: {si.get('age')}-year-old {si.get('sex')}, site {si.get('site')}."

    text = (
        f"The model predicts {pred} with {conf:.1%} confidence based on consensus "
        f"across {total} quality-filtered axial slices ({votes} ASD votes).{demo} "
        f"This is a research-grade result only and must not inform clinical decision-making."
    )
    if note:
        text += f"\n\n[System note: {note}]"
    return text
