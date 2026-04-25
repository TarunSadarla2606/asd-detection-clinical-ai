# app.py — v8.1 bulletproof three-tab ensemble
# Rewritten from scratch. Compatible with Streamlit 1.32+ on Python 3.13.
#
# Defensive patterns applied throughout:
#   1. Figures rendered to numpy RGB arrays (bypass PIL/BytesIO/bytes type edge cases
#      in st.image on Python 3.13 / newer Streamlit)
#   2. use_container_width=True (NOT width='stretch') — the former works on every
#      Streamlit from 1.32 to current; the latter only on 1.51+
#   3. Every figure render wrapped in try/except; partial failures don't kill the page
#   4. Length-guarded indexing; never iterate past the shorter list
#   5. Format-spec ternaries removed from f-strings (broken syntax in v7 caused
#      misleading TypeErrors attributed to unrelated lines)
#   6. Defensive ensemble math (zero-division guards, missing uncertainty handling)

import streamlit as st
import tempfile, os, io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image as PILImage

from model import load_model, load_hybrid_model, HYBRID_TRANSFORM
from pipeline import run_subject_pipeline, predict_slice, EVAL_TRANSFORM
from xai import (
    run_gradcam, mc_dropout_uncertainty,
    make_explanation_figure,
    get_attention_rollout, mc_dropout_hybrid,
    make_hybrid_xai_figure, make_dual_xai_figure,
)
from phenotypic import load_phenotypic, get_subject_info, extract_anon_num_from_filename
from anatomy import label_region
from narrator import (
    generate_clinical_narrative,
    generate_hybrid_narrative,
    generate_ensemble_narrative,
)
from report import generate_report


# ════════════════════════════════════════════════════════════════════════════
# UTILITIES — defensive figure rendering
# ════════════════════════════════════════════════════════════════════════════

def _fig_to_array(fig, dpi=100):
    """
    Render a matplotlib figure to an RGB numpy array.
    Returns None if rendering fails so callers can skip gracefully.
    """
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        img = PILImage.open(buf).convert('RGB')   # force RGB, drop alpha
        return np.asarray(img)
    except Exception as e:
        st.warning(f"Figure render failed: {type(e).__name__}: {e}")
        return None
    finally:
        plt.close(fig)


def _show_image(arr_or_none, caption=None):
    """
    Display an image array via st.image. Uses use_container_width=True which is
    supported across Streamlit 1.32 through current. In newer versions it emits
    a deprecation warning but does NOT raise — width='stretch' would raise TypeError
    on 1.32, so we deliberately avoid it.
    Silently skips if arr is None (e.g. earlier render failed).
    """
    if arr_or_none is None:
        st.info("Image unavailable for this slice.")
        return
    try:
        st.image(arr_or_none, caption=caption, use_container_width=True)
    except TypeError:
        # Extreme fallback: drop the kwarg entirely
        try:
            st.image(arr_or_none, caption=caption)
        except Exception as e:
            st.warning(f"Image display failed: {type(e).__name__}: {e}")


def _array_to_png_bytes(arr):
    """Convert numpy RGB array back to PNG bytes for PDF embedding."""
    if arr is None:
        return None
    try:
        buf = io.BytesIO()
        PILImage.fromarray(arr).save(buf, format='PNG')
        return buf.getvalue()
    except Exception:
        return None


def _fmt_sigma(unc_dict):
    """Safely format sigma from an uncertainty dict; returns 'N/A' if missing."""
    if unc_dict and 'std' in unc_dict:
        return f"{unc_dict['std']:.4f}"
    return "N/A"


# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG + STYLES
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ASD Detection — Clinical AI Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background:#0e1117; }
[data-testid="stSidebar"] { background:#161b27; border-right:1px solid #2d3748; }
h1,h2,h3,h4 { color:#e2e8f0 !important; }
p, li, label { color:#cbd5e0; }
.pred-card { padding:28px; border-radius:14px; text-align:center; margin:14px 0; }
.pred-asd  { background:linear-gradient(135deg,#c0392b18,#c0392b35); border:2px solid #c0392b; }
.pred-tc   { background:linear-gradient(135deg,#27ae6018,#27ae6035); border:2px solid #27ae60; }
.model-card{ background:#1a202c; border:1px solid #2d3748; border-radius:10px;
             padding:14px 16px; margin:6px 0; }
.narrative-box { background:#1a202c; border-left:4px solid #4a90d9;
                 padding:16px 20px; border-radius:6px; margin:12px 0;
                 font-size:0.93em; line-height:1.7; color:#cbd5e0; }
.agree-high     { border-left:4px solid #27ae60; background:#00200a;
                  padding:10px 14px; border-radius:6px; margin:8px 0; }
.agree-moderate { border-left:4px solid #f39c12; background:#2a1f00;
                  padding:10px 14px; border-radius:6px; margin:8px 0; }
.agree-low      { border-left:4px solid #e74c3c; background:#200010;
                  padding:10px 14px; border-radius:6px; margin:8px 0; }
.disagree-banner{ border:2px solid #e74c3c; background:#200010;
                  padding:14px 18px; border-radius:8px; margin:12px 0; }
.ensemble-weights{ background:#1a202c; border:1px solid #4a90d9;
                   padding:12px 16px; border-radius:8px; margin:8px 0; }
.region-tag { display:inline-block; background:#2d3748; color:#90cdf4;
              padding:3px 10px; border-radius:12px; font-size:0.8em; margin:2px; }
.site-warn  { border-left:4px solid #f39c12; padding:10px 14px;
              background:#2a1f00; border-radius:6px; font-size:0.88em; margin:8px 0; }
.site-ok    { border-left:4px solid #27ae60; padding:10px 14px;
              background:#00200a; border-radius:6px; font-size:0.88em; margin:8px 0; }
.disclaimer { background:#1a1a2e; border-left:4px solid #f39c12;
              padding:14px 18px; border-radius:6px; font-size:0.80em;
              color:#a0aec0; margin-top:20px; line-height:1.6; }
[data-testid="metric-container"] {
    background:#1a202c; border:1px solid #2d3748; border-radius:8px; padding:10px; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════════════════

_defaults = {
    'cnn_result': None, 'hybrid_result': None, 'ensemble_result': None,
    'subject_info': None, 'source_name': None,
    'cnn_gradcam_stats': [], 'cnn_uncertainty': None, 'cnn_narrative': None,
    'cnn_fig_arrays': [], 'cnn_pdf': None,
    'hybrid_attn_stats': [], 'hybrid_uncertainty': None, 'hybrid_narrative': None,
    'hybrid_fig_arrays': [], 'hybrid_pdf': None,
    'dual_fig_arrays': [], 'xai_agreement': None,
    'ensemble_narrative': None, 'ensemble_pdf': None,
    'compute_key': None, 'saved_nifti': None, 'saved_name': None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ════════════════════════════════════════════════════════════════════════════
# CACHED LOADERS
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_cnn():
    return load_model("weights/xai_cnn_best_weights.pth", device="cpu")

@st.cache_resource
def get_hybrid():
    return load_hybrid_model("weights/hybrid_best.pth", device="cpu")

@st.cache_resource
def get_pheno():
    return load_phenotypic()


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🧠 ASD Detection")
    st.markdown("**Research Demo** · University of North Texas")
    st.markdown("---")
    st.markdown("### Settings")
    top_k           = st.slider("XAI slices per model", 1, 5, 3)
    run_lime_flag   = st.checkbox("LIME explanations (~20s/slice)", value=False)
    run_uncertainty = st.checkbox("MC-Dropout uncertainty", value=True)
    run_narrative   = st.checkbox("LLM clinical narratives", value=True)
    st.markdown("---")
    st.markdown("### Models")

    st.markdown("""<div class='model-card'>
    <b style='color:#4a90d9'>Tab 1 — Pure CNN</b><br>
    <small style='color:#718096'>5-layer CNN · Phase 2</small><br><br>
    <table style='font-size:0.82em;color:#cbd5e0'>
    <tr><td>AUC</td><td><b>0.994</b></td></tr>
    <tr><td>Sensitivity</td><td><b>95.6%</b></td></tr>
    <tr><td>Specificity</td><td><b>97.2%</b></td></tr>
    <tr><td>XAI</td><td>GradCAM</td></tr></table></div>""", unsafe_allow_html=True)

    st.markdown("""<div class='model-card'>
    <b style='color:#8e44ad'>Tab 2 — Hybrid CNN-ViT</b><br>
    <small style='color:#718096'>CNN backbone + 4-layer Transformer</small><br><br>
    <table style='font-size:0.82em;color:#cbd5e0'>
    <tr><td>AUC</td><td><b>0.943</b></td></tr>
    <tr><td>Sensitivity</td><td><b>84.5%</b></td></tr>
    <tr><td>Specificity</td><td><b>86.8%</b></td></tr>
    <tr><td>XAI</td><td>Attention Rollout</td></tr></table></div>""", unsafe_allow_html=True)

    st.markdown("""<div class='model-card'>
    <b style='color:#e67e22'>Tab 3 — Ensemble</b><br>
    <small style='color:#718096'>Uncertainty-gated fusion</small><br><br>
    <small style='color:#a0aec0'>Dynamic weights from MC-Dropout sigma.<br>
    Disagreement triggers expert review flag.<br>
    Side-by-side XAI comparison.</small></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-size:0.78em;color:#718096'>Research use only. "
                "Not for clinical deployment.</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════════

c1, c2 = st.columns([1, 8])
with c1:
    st.markdown("<div style='font-size:3.5em;margin-top:8px'>🧠</div>", unsafe_allow_html=True)
with c2:
    st.title("ASD Detection from Structural MRI")
    st.markdown(
        "Upload a **NIfTI (.nii / .nii.gz)** brain scan. Three analysis modes run in parallel: "
        "**Tab 1** — Pure CNN (AUC 0.994); "
        "**Tab 2** — Hybrid CNN-ViT with self-attention maps; "
        "**Tab 3** — Uncertainty-gated ensemble with XAI comparison."
    )
st.markdown("---")


# ════════════════════════════════════════════════════════════════════════════
# FILE INPUT
# ════════════════════════════════════════════════════════════════════════════

uploaded = st.file_uploader("Upload NIfTI file", type=["nii", "gz"])

st.markdown("#### Or try a demo subject:")
demo_cols = st.columns(4)
demo_file = None
demo_dir  = "demo_subjects"
if os.path.exists(demo_dir):
    demo_files = sorted([f for f in os.listdir(demo_dir) if f.endswith(('.nii', '.nii.gz'))])
    for i, fname in enumerate(demo_files[:4]):
        label = fname.replace('.nii.gz', '').replace('.nii', '')
        if demo_cols[i % 4].button(f"📂 {label}", use_container_width=True):
            demo_file = os.path.join(demo_dir, fname)
            st.session_state.saved_nifti = None
            st.session_state.saved_name  = None

if uploaded is not None:
    file_id = f"{uploaded.name}_{uploaded.size}"
    if st.session_state.saved_name != file_id:
        suffix = '.nii.gz' if uploaded.name.endswith('.gz') else '.nii'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            st.session_state.saved_nifti = tmp.name
            st.session_state.saved_name  = file_id

nifti_path  = st.session_state.saved_nifti if uploaded else (demo_file if demo_file else None)
source_name = (uploaded.name if uploaded else
               (os.path.basename(demo_file) if demo_file else None))


# ════════════════════════════════════════════════════════════════════════════
# COMPUTATION STAGE — runs only when input changes
# ════════════════════════════════════════════════════════════════════════════

compute_key = (f"{nifti_path}_{top_k}_{run_lime_flag}_{run_uncertainty}_{run_narrative}"
               if nifti_path else None)

if compute_key and compute_key != st.session_state.compute_key:
    cnn_model    = get_cnn()
    hybrid_model = get_hybrid()
    pheno_df     = get_pheno()

    anon_num     = extract_anon_num_from_filename(source_name) if source_name else None
    subject_info = (get_subject_info(anon_num, pheno_df)
                    if anon_num and pheno_df is not None else None)

    pbar   = st.progress(0)
    status = st.empty()

    # ── Step 1/9: CNN pipeline ──────────────────────────────────────────────
    status.text("Step 1/9 — CNN inference pipeline...")
    cnn_result = run_subject_pipeline(nifti_path, cnn_model, device='cpu',
                                      top_k=top_k, progress_fn=None)
    pbar.progress(10)
    if 'error' in cnn_result:
        st.error(cnn_result['error'])
        st.stop()

    # ── Step 2/9: Hybrid inference on the same slices ──────────────────────
    status.text("Step 2/9 — Hybrid CNN-ViT inference...")
    import torch
    import torch.nn.functional as F_nn
    from torchvision import transforms as T_tf

    _hyb_tf = T_tf.Compose([
        T_tf.Resize((224, 224)),
        T_tf.ToTensor(),
        T_tf.Normalize([0.1425056904554367] * 3, [0.19151894748210907] * 3),
    ])

    def _predict_hybrid(arr, model):
        pil = PILImage.fromarray(arr).convert('RGB')
        t = _hyb_tf(pil).unsqueeze(0)
        with torch.no_grad():
            p = F_nn.softmax(model(t), dim=1)[0].cpu().numpy()
        return {
            'prob_tc': float(p[0]),
            'prob_asd': float(p[1]),
            'pred_class': int(p.argmax()),
        }

    hybrid_valid = []
    for vs in cnn_result.get('top_slices', []):
        try:
            ph = _predict_hybrid(vs['arr'], hybrid_model)
        except Exception as e:
            st.warning(f"Hybrid inference failed on slice {vs.get('slice_idx','?')}: {e}")
            ph = {'prob_tc': 0.5, 'prob_asd': 0.5, 'pred_class': 0}
        hybrid_valid.append({**vs, **ph})

    asd_v = sum(1 for v in hybrid_valid if v['pred_class'] == 1)
    tc_v  = len(hybrid_valid) - asd_v
    probs = [v['prob_asd'] for v in hybrid_valid]
    wts   = [v['quality_score'] * max(v['prob_asd'], v['prob_tc']) for v in hybrid_valid]
    wtot  = sum(wts)
    wp    = sum(p * w for p, w in zip(probs, wts)) / wtot if wtot > 0 else float(np.mean(probs)) if probs else 0.5
    hyb_pred = 'ASD' if wp >= 0.5 else 'TC'
    hyb_conf = wp if hyb_pred == 'ASD' else 1 - wp
    hybrid_result = {
        'subject_pred': hyb_pred, 'subject_conf': round(hyb_conf, 4),
        'weighted_prob_asd': round(wp, 4),
        'pred_asd_votes': asd_v, 'pred_tc_votes': tc_v,
        'total_valid': len(hybrid_valid),
        'n_total_slices': cnn_result['n_total_slices'],
        'top_slices': hybrid_valid,
    }
    pbar.progress(20)

    # ── Step 3/9: CNN GradCAM + anatomy ────────────────────────────────────
    status.text("Step 3/9 — CNN GradCAM explanations...")
    cnn_gradcam_stats = []
    cnn_fig_arrays    = []
    for sl in cnn_result['top_slices']:
        try:
            gc = run_gradcam(sl['arr_cropped'], cnn_model, sl['pred_class'], device='cpu')
            ar = label_region(sl['slice_idx'], gc['heatmap'])
            cnn_gradcam_stats.append({
                'slice_idx': sl['slice_idx'], 'prob_asd': sl['prob_asd'],
                'energy': gc['energy'], 'region': ar['full_label'],
                'lobe': ar['lobe'], 'spatial': ar['spatial'],
                'com_x': ar['com_x'], 'com_y': ar['com_y'],
            })
        except Exception as e:
            st.warning(f"GradCAM stats failed slice {sl.get('slice_idx','?')}: {e}")
            cnn_gradcam_stats.append({
                'slice_idx': sl.get('slice_idx', 0), 'prob_asd': sl.get('prob_asd', 0),
                'energy': 0.0, 'region': 'unknown', 'lobe': '', 'spatial': '',
                'com_x': 0, 'com_y': 0,
            })
        try:
            fig = make_explanation_figure(
                sl['arr_cropped'], cnn_model,
                sl['pred_class'], sl['slice_idx'],
                device='cpu', run_lime_flag=run_lime_flag,
            )
            cnn_fig_arrays.append(_fig_to_array(fig))
        except Exception as e:
            st.warning(f"CNN figure render failed slice {sl.get('slice_idx','?')}: {e}")
            cnn_fig_arrays.append(None)
    pbar.progress(35)

    # ── Step 4/9: Hybrid attention rollout ─────────────────────────────────
    status.text("Step 4/9 — Hybrid attention maps...")
    hybrid_attn_stats  = []
    hybrid_fig_arrays  = []
    for sl in cnn_result['top_slices']:
        try:
            attn = get_attention_rollout(sl['arr_cropped'], hybrid_model, device='cpu')
            hybrid_attn_stats.append(attn)
        except Exception as e:
            st.warning(f"Attention rollout failed slice {sl.get('slice_idx','?')}: {e}")
            hybrid_attn_stats.append({'n_brain': 0})
        try:
            hyb_sl = next((v for v in hybrid_valid if v['slice_idx'] == sl['slice_idx']), sl)
            fig = make_hybrid_xai_figure(
                sl['arr_cropped'], hybrid_model,
                hyb_sl.get('pred_class', sl['pred_class']),
                sl['slice_idx'], device='cpu',
            )
            hybrid_fig_arrays.append(_fig_to_array(fig))
        except Exception as e:
            st.warning(f"Hybrid figure render failed slice {sl.get('slice_idx','?')}: {e}")
            hybrid_fig_arrays.append(None)
    pbar.progress(50)

    # ── Step 5/9: Dual XAI comparison ──────────────────────────────────────
    status.text("Step 5/9 — Dual XAI comparison figures...")
    dual_fig_arrays = []
    xai_agreement   = None
    for sl in cnn_result['top_slices'][:top_k]:
        try:
            hyb_sl = next((v for v in hybrid_valid if v['slice_idx'] == sl['slice_idx']), sl)
            fig, agree = make_dual_xai_figure(
                sl['arr_cropped'], cnn_model, hybrid_model,
                sl['pred_class'], hyb_sl.get('pred_class', sl['pred_class']),
                sl['slice_idx'], device='cpu',
            )
            if xai_agreement is None:
                xai_agreement = agree
            dual_fig_arrays.append(_fig_to_array(fig))
        except Exception as e:
            st.warning(f"Dual XAI failed slice {sl.get('slice_idx','?')}: {e}")
            dual_fig_arrays.append(None)
    pbar.progress(62)

    # ── Step 6/9: MC-Dropout uncertainty ───────────────────────────────────
    status.text("Step 6/9 — Uncertainty estimation (both models)...")
    cnn_uncertainty    = None
    hybrid_uncertainty = None
    if run_uncertainty and cnn_result['top_slices']:
        arr0 = cnn_result['top_slices'][0]['arr_cropped']
        try:
            cnn_uncertainty = mc_dropout_uncertainty(arr0, cnn_model, n_passes=30)
        except Exception as e:
            st.warning(f"CNN uncertainty failed: {e}")
        try:
            hybrid_uncertainty = mc_dropout_hybrid(arr0, hybrid_model, n_passes=30)
        except Exception as e:
            st.warning(f"Hybrid uncertainty failed: {e}")
    pbar.progress(72)

    # ── Step 7/9: Ensemble fusion ──────────────────────────────────────────
    status.text("Step 7/9 — Uncertainty-gated ensemble fusion...")
    cnn_prob   = cnn_result['weighted_prob_asd']
    hyb_prob   = hybrid_result['weighted_prob_asd']
    sig_cnn    = cnn_uncertainty['std']    if cnn_uncertainty    else 0.05
    sig_hyb    = hybrid_uncertainty['std'] if hybrid_uncertainty else 0.10
    w_cnn_raw  = (1 - sig_cnn) * 0.994
    w_vit_raw  = (1 - sig_hyb) * 0.943
    w_total    = w_cnn_raw + w_vit_raw
    w_cnn      = w_cnn_raw / w_total if w_total > 0 else 0.5
    w_vit      = w_vit_raw / w_total if w_total > 0 else 0.5
    ens_prob   = w_cnn * cnn_prob + w_vit * hyb_prob
    ens_pred   = 'ASD' if ens_prob >= 0.5 else 'TC'
    ens_conf   = ens_prob if ens_pred == 'ASD' else 1 - ens_prob
    ensemble_result = {
        'ensemble_pred': ens_pred,
        'ensemble_prob_asd': round(ens_prob, 4),
        'ensemble_conf': round(ens_conf, 4),
        'weight_cnn': round(w_cnn, 3),
        'weight_vit': round(w_vit, 3),
        'models_agree': cnn_result['subject_pred'] == hybrid_result['subject_pred'],
        'prob_diff': round(abs(cnn_prob - hyb_prob), 4),
    }
    pbar.progress(80)

    # ── Step 8/9: LLM narratives ───────────────────────────────────────────
    cnn_narrative = hybrid_narrative = ensemble_narrative = None
    if run_narrative:
        status.text("Step 8/9 — Generating clinical narratives (3 tabs)...")
        try:
            cnn_narrative = generate_clinical_narrative(
                cnn_result, subject_info, cnn_uncertainty, cnn_gradcam_stats)
        except Exception as e:
            st.warning(f"CNN narrative failed: {e}")
        try:
            hybrid_narrative = generate_hybrid_narrative(
                hybrid_result, subject_info, hybrid_uncertainty, hybrid_attn_stats)
        except Exception as e:
            st.warning(f"Hybrid narrative failed: {e}")
        try:
            ensemble_narrative = generate_ensemble_narrative(
                cnn_result, hybrid_result, ensemble_result, subject_info,
                cnn_uncertainty, hybrid_uncertainty, xai_agreement)
        except Exception as e:
            st.warning(f"Ensemble narrative failed: {e}")
    pbar.progress(90)

    # ── Step 9/9: PDF reports ──────────────────────────────────────────────
    status.text("Step 9/9 — Building PDF reports...")

    def _arrays_to_report_figs(arr_list):
        figs = []
        for a in arr_list:
            if a is None:
                continue
            try:
                fig, ax = plt.subplots(figsize=(16, 4))
                ax.imshow(a)
                ax.axis('off')
                fig.patch.set_facecolor('#0e1117')
                figs.append(fig)
            except Exception:
                pass
        return figs

    cnn_pdf = hybrid_pdf = ensemble_pdf = None
    try:
        cnn_figs_r    = _arrays_to_report_figs(cnn_fig_arrays)
        hybrid_figs_r = _arrays_to_report_figs(hybrid_fig_arrays)
        dual_figs_r   = _arrays_to_report_figs(dual_fig_arrays)

        cnn_pdf = generate_report(
            result=cnn_result, subject_info=subject_info,
            xai_figures=cnn_figs_r, narrative=cnn_narrative, mode='cnn')

        hybrid_pdf = generate_report(
            result=hybrid_result, subject_info=subject_info,
            xai_figures=hybrid_figs_r, narrative=hybrid_narrative,
            mode='hybrid', hybrid_uncertainty=hybrid_uncertainty)

        ensemble_pdf = generate_report(
            result=cnn_result, subject_info=subject_info,
            xai_figures=cnn_figs_r, narrative=ensemble_narrative, mode='ensemble',
            hybrid_result=hybrid_result, ensemble_result=ensemble_result,
            cnn_uncertainty=cnn_uncertainty, hybrid_uncertainty=hybrid_uncertainty,
            xai_agreement=xai_agreement,
            hybrid_xai_figures=hybrid_figs_r, dual_xai_figures=dual_figs_r)

        for f in cnn_figs_r + hybrid_figs_r + dual_figs_r:
            plt.close(f)
    except Exception as e:
        st.warning(f"PDF generation failed: {e}")

    pbar.progress(100)
    status.text("Complete.")

    # Single batch write to session state
    st.session_state.update({
        'cnn_result': cnn_result, 'hybrid_result': hybrid_result,
        'ensemble_result': ensemble_result,
        'subject_info': subject_info, 'source_name': source_name,
        'cnn_gradcam_stats': cnn_gradcam_stats,
        'cnn_uncertainty': cnn_uncertainty,
        'cnn_narrative': cnn_narrative,
        'cnn_fig_arrays': cnn_fig_arrays,
        'cnn_pdf': cnn_pdf,
        'hybrid_attn_stats': hybrid_attn_stats,
        'hybrid_uncertainty': hybrid_uncertainty,
        'hybrid_narrative': hybrid_narrative,
        'hybrid_fig_arrays': hybrid_fig_arrays,
        'hybrid_pdf': hybrid_pdf,
        'dual_fig_arrays': dual_fig_arrays,
        'xai_agreement': xai_agreement,
        'ensemble_narrative': ensemble_narrative,
        'ensemble_pdf': ensemble_pdf,
        'compute_key': compute_key,
    })
    pbar.empty()
    status.empty()


# ════════════════════════════════════════════════════════════════════════════
# DISPLAY STAGE — pure rendering, no computation
# ════════════════════════════════════════════════════════════════════════════

if st.session_state.cnn_result is not None:
    cnn_result    = st.session_state.cnn_result
    hybrid_result = st.session_state.hybrid_result
    ens_result    = st.session_state.ensemble_result
    subject_info  = st.session_state.subject_info
    cnn_gc        = st.session_state.cnn_gradcam_stats
    cnn_unc       = st.session_state.cnn_uncertainty
    hyb_unc       = st.session_state.hybrid_uncertainty
    xai_agree     = st.session_state.xai_agreement

    cnn_fig_arrays    = st.session_state.cnn_fig_arrays
    hybrid_fig_arrays = st.session_state.hybrid_fig_arrays
    dual_fig_arrays   = st.session_state.dual_fig_arrays

    tab1, tab2, tab3 = st.tabs([
        "🔵 Tab 1 — Pure CNN (AUC 0.994)",
        "🟣 Tab 2 — Hybrid CNN-ViT (AUC 0.943)",
        "🟠 Tab 3 — Ensemble & XAI Comparison",
    ])

    # ════════════════════════════════════════════════════════════════════
    # TAB 1 — CNN
    # ════════════════════════════════════════════════════════════════════
    with tab1:
        pred = cnn_result['subject_pred']
        conf = cnn_result['subject_conf']
        prob = cnn_result['weighted_prob_asd']
        color = '#e74c3c' if pred == 'ASD' else '#2ecc71'
        icon  = '🔴' if pred == 'ASD' else '🟢'

        st.markdown("## Subject-Level Prediction — CNN")
        pred_class = "asd" if pred == "ASD" else "tc"
        st.markdown(f"""<div class='pred-card pred-{pred_class}'>
        <div style='font-size:3.2em;font-weight:900;color:{color};font-family:monospace'>{icon} {pred}</div>
        <div style='font-size:1.6em;color:#e2e8f0;margin:6px 0'>Confidence: {conf:.1%}</div>
        <div style='color:#a0aec0;font-size:0.9em'>Weighted consensus · {cnn_result["total_valid"]} valid slices</div>
        </div>""", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ASD votes",    cnn_result['pred_asd_votes'])
        m2.metric("TC votes",     cnn_result['pred_tc_votes'])
        m3.metric("Valid slices", cnn_result['total_valid'])
        m4.metric("Total slices", cnn_result['n_total_slices'])

        bar_col = '#e74c3c' if prob >= 0.5 else '#2ecc71'
        st.markdown("**Weighted P(ASD):**")
        st.markdown(f"""<div style='background:#2d3748;border-radius:8px;padding:4px;margin:4px 0'>
        <div style='background:{bar_col};width:{prob*100:.1f}%;min-width:44px;height:26px;
        border-radius:6px;text-align:center;line-height:26px;color:white;font-weight:700;font-size:0.9em'>
        {prob:.1%} P(ASD)</div></div>""", unsafe_allow_html=True)

        # Subject context
        if subject_info:
            st.markdown("---")
            st.markdown("## Subject & Site Context")
            si = subject_info
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sex", si['sex'])
            c2.metric("Age", f"{si['age']} yrs" if si['age'] else "N/A")
            c3.metric("Site", si['site'])
            c4.metric("ABIDE Diagnosis", si['true_label'])
            if si['true_label'] != 'Unknown':
                if pred == si['true_label']:
                    st.success(f"Prediction **{pred}** matches ABIDE-I ground truth **{si['true_label']}**")
                else:
                    st.error(f"Prediction **{pred}** differs from ABIDE-I ground truth **{si['true_label']}**")
            if si.get('site_perf'):
                sp = si['site_perf']
                low = sp['Sensitivity'] < 0.92 or sp['Specificity'] < 0.95
                cls = 'site-warn' if low else 'site-ok'
                ico = '⚠️' if low else '✅'
                tail = ' — <b>Below threshold</b>' if low else ''
                st.markdown(
                    f"<div class='{cls}'>{ico} <b>Site {si['site']}:</b> "
                    f"Sens {sp['Sensitivity']:.1%} · Spec {sp['Specificity']:.1%} · "
                    f"AUC {sp['AUC']:.3f}{tail}</div>",
                    unsafe_allow_html=True
                )

        # Uncertainty
        if cnn_unc:
            st.markdown("---")
            st.markdown("## Prediction Uncertainty (MC-Dropout, 30 passes)")
            u1, u2, u3 = st.columns(3)
            u1.metric("Mean P(ASD)", f"{cnn_unc['mean_prob_asd']:.4f}")
            u2.metric("Std Dev (σ)", f"{cnn_unc['std']:.4f}")
            u3.metric("Assessment",  cnn_unc['uncertainty'].split('—')[0].strip())
            st.info(f"🔍 {cnn_unc['uncertainty']}")

        # XAI
        st.markdown("---")
        st.markdown("## Explainability — GradCAM")
        n = min(len(cnn_result['top_slices']), len(cnn_fig_arrays))
        for i in range(n):
            sl  = cnn_result['top_slices'][i]
            arr = cnn_fig_arrays[i]
            gc_info = cnn_gc[i] if i < len(cnn_gc) else {}
            slice_pred = 'ASD' if sl['pred_class'] == 1 else 'TC'
            with st.expander(
                f"Slice z={sl['slice_idx']} · P(ASD)={sl['prob_asd']:.3f} · "
                f"Quality={sl['quality_score']:.3f} · Pred={slice_pred}",
                expanded=(i == 0),
            ):
                _show_image(arr)
                r1, r2, r3 = st.columns(3)
                r1.metric("P(ASD)",  f"{sl['prob_asd']:.4f}")
                r2.metric("P(TC)",   f"{sl['prob_tc']:.4f}")
                r3.metric("Quality", f"{sl['quality_score']:.4f}")
                if gc_info.get('region'):
                    st.markdown(
                        f"📍 **Region:** <span class='region-tag'>{gc_info['region']}</span>",
                        unsafe_allow_html=True
                    )

        # Narrative
        if st.session_state.cnn_narrative:
            st.markdown("---")
            st.markdown("## Clinical Interpretation (LLM)")
            st.markdown(
                f"<div class='narrative-box'>{st.session_state.cnn_narrative}</div>",
                unsafe_allow_html=True
            )

        # Download
        if st.session_state.cnn_pdf:
            st.markdown("---")
            st.download_button(
                "📄 Download CNN Report (PDF)",
                data=st.session_state.cnn_pdf,
                file_name="asd_cnn_report.pdf",
                mime="application/pdf",
            )

        st.markdown(
            "<div class='disclaimer'>⚠️ Research use only. Not validated for clinical deployment. "
            "5-layer CNN · ABIDE-I · University of North Texas.</div>",
            unsafe_allow_html=True
        )


    # ════════════════════════════════════════════════════════════════════
    # TAB 2 — HYBRID CNN-ViT
    # ════════════════════════════════════════════════════════════════════
    with tab2:
        pred = hybrid_result['subject_pred']
        conf = hybrid_result['subject_conf']
        prob = hybrid_result['weighted_prob_asd']
        color = '#e74c3c' if pred == 'ASD' else '#2ecc71'
        icon  = '🔴' if pred == 'ASD' else '🟢'

        st.markdown("## Subject-Level Prediction — Hybrid CNN-ViT")
        st.info("ℹ️ This model uses self-attention over spatial brain-region tokens. "
                "AUC 0.943 vs CNN 0.994 — expected in limited-data regime. "
                "Attention maps provide complementary spatial interpretation.")
        pred_class2 = "asd" if pred == "ASD" else "tc"
        st.markdown(f"""<div class='pred-card pred-{pred_class2}'>
        <div style='font-size:3.2em;font-weight:900;color:{color};font-family:monospace'>{icon} {pred}</div>
        <div style='font-size:1.6em;color:#e2e8f0;margin:6px 0'>Confidence: {conf:.1%}</div>
        <div style='color:#a0aec0;font-size:0.9em'>Transformer attention consensus · {hybrid_result["total_valid"]} slices</div>
        </div>""", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ASD votes",    hybrid_result['pred_asd_votes'])
        m2.metric("TC votes",     hybrid_result['pred_tc_votes'])
        m3.metric("Valid slices", hybrid_result['total_valid'])
        m4.metric("Hybrid AUC",   "0.943")

        bar_col = '#e74c3c' if prob >= 0.5 else '#2ecc71'
        st.markdown("**Weighted P(ASD) (Hybrid):**")
        st.markdown(f"""<div style='background:#2d3748;border-radius:8px;padding:4px;margin:4px 0'>
        <div style='background:{bar_col};width:{prob*100:.1f}%;min-width:44px;height:26px;
        border-radius:6px;text-align:center;line-height:26px;color:white;font-weight:700;font-size:0.9em'>
        {prob:.1%} P(ASD)</div></div>""", unsafe_allow_html=True)

        # Uncertainty
        if hyb_unc:
            st.markdown("---")
            st.markdown("## Uncertainty (CNN + Transformer Dropout, 30 passes)")
            st.markdown("Both CNN Dropout2d and Transformer Dropout layers are active — "
                        "σ reflects uncertainty from both pathways simultaneously.")
            u1, u2, u3 = st.columns(3)
            u1.metric("Mean P(ASD)",   f"{hyb_unc['mean_prob_asd']:.4f}")
            u2.metric("Std Dev (σ)",   f"{hyb_unc['std']:.4f}")
            u3.metric("Interpretation", hyb_unc['uncertainty'].split('—')[0].strip())
            st.info(f"🔍 {hyb_unc['uncertainty']}")

        # Attention maps
        st.markdown("---")
        st.markdown("## Explainability — Transformer Attention Maps")
        st.markdown("CLS→token attention from the last transformer layer, averaged across 8 heads. "
                    "Shows which spatial locations across the full slice the model weights when deciding ASD vs TC.")
        n = min(len(cnn_result['top_slices']), len(hybrid_fig_arrays))
        for i in range(n):
            sl  = cnn_result['top_slices'][i]
            arr = hybrid_fig_arrays[i]
            attn_info = (st.session_state.hybrid_attn_stats[i]
                         if i < len(st.session_state.hybrid_attn_stats) else {})
            n_brain = attn_info.get('n_brain', 0)
            with st.expander(
                f"Slice z={sl['slice_idx']} · {n_brain}/196 brain tokens",
                expanded=(i == 0),
            ):
                _show_image(arr)
                c1, c2 = st.columns(2)
                c1.metric("Brain tokens",   f"{n_brain}/196")
                c2.metric("Brain coverage", f"{n_brain/196*100:.1f}%")

        # Narrative
        if st.session_state.hybrid_narrative:
            st.markdown("---")
            st.markdown("## Clinical Interpretation (LLM)")
            st.markdown(
                f"<div class='narrative-box'>{st.session_state.hybrid_narrative}</div>",
                unsafe_allow_html=True
            )

        # Download
        if st.session_state.hybrid_pdf:
            st.markdown("---")
            st.download_button(
                "📄 Download Hybrid CNN-ViT Report (PDF)",
                data=st.session_state.hybrid_pdf,
                file_name="asd_hybrid_vit_report.pdf",
                mime="application/pdf",
            )

        st.markdown(
            "<div class='disclaimer'>⚠️ Research use only. Hybrid CNN-ViT · ABIDE-I · UNT.</div>",
            unsafe_allow_html=True
        )


    # ════════════════════════════════════════════════════════════════════
    # TAB 3 — ENSEMBLE
    # ════════════════════════════════════════════════════════════════════
    with tab3:
        ens_pred     = ens_result['ensemble_pred']
        ens_prob     = ens_result['ensemble_prob_asd']
        ens_conf     = ens_result['ensemble_conf']
        w_cnn        = ens_result['weight_cnn']
        w_vit        = ens_result['weight_vit']
        cnn_prob     = cnn_result['weighted_prob_asd']
        hyb_prob     = hybrid_result['weighted_prob_asd']
        models_agree = ens_result['models_agree']
        prob_diff    = ens_result['prob_diff']
        color = '#e74c3c' if ens_pred == 'ASD' else '#2ecc71'
        icon  = '🔴' if ens_pred == 'ASD' else '🟢'

        # Pre-compute sigma strings (avoid invalid f-string ternary syntax)
        sig_cnn_str = _fmt_sigma(cnn_unc)
        sig_hyb_str = _fmt_sigma(hyb_unc)

        st.markdown("## Ensemble Analysis")

        # Disagreement banner
        if not models_agree:
            st.markdown(
                f"<div class='disagree-banner'>"
                f"<b style='color:#e74c3c;font-size:1.1em'>⚠️ Model Disagreement Detected</b><br>"
                f"CNN predicts <b>{cnn_result['subject_pred']}</b> (P(ASD)={cnn_prob:.4f}) · "
                f"Hybrid ViT predicts <b>{hybrid_result['subject_pred']}</b> (P(ASD)={hyb_prob:.4f}) · "
                f"Difference: {prob_diff:.4f}<br>"
                f"<small>When architecturally diverse models disagree, the ambiguity cannot be resolved "
                f"algorithmically. Expert radiologist review is recommended before any clinical use.</small>"
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            st.success(f"✅ Both models independently predict **{ens_pred}** — "
                       f"cross-architecture agreement (P diff={prob_diff:.4f})")

        # Ensemble verdict
        ens_class = "asd" if ens_pred == "ASD" else "tc"
        st.markdown(f"""<div class='pred-card pred-{ens_class}'>
        <div style='font-size:3.2em;font-weight:900;color:{color};font-family:monospace'>{icon} {ens_pred}</div>
        <div style='font-size:1.6em;color:#e2e8f0;margin:6px 0'>Ensemble Confidence: {ens_conf:.1%}</div>
        <div style='color:#a0aec0;font-size:0.9em'>Uncertainty-gated fusion ·
        CNN weight {w_cnn:.3f} · ViT weight {w_vit:.3f}</div>
        </div>""", unsafe_allow_html=True)

        # Side-by-side model comparison
        st.markdown("---")
        st.markdown("## Individual Model Predictions")
        c1, c2 = st.columns(2)
        with c1:
            cnn_c = '#e74c3c' if cnn_result['subject_pred'] == 'ASD' else '#2ecc71'
            st.markdown(
                f"<div class='model-card'>"
                f"<b style='color:#4a90d9'>CNN (AUC 0.994)</b><br>"
                f"<div style='font-size:2em;color:{cnn_c};font-weight:900'>{cnn_result['subject_pred']}</div>"
                f"P(ASD)={cnn_prob:.4f} · Conf={cnn_result['subject_conf']:.1%}<br>"
                f"Votes: {cnn_result['pred_asd_votes']} ASD / {cnn_result['pred_tc_votes']} TC<br>"
                f"σ={sig_cnn_str} · Weight: {w_cnn:.3f}"
                f"</div>",
                unsafe_allow_html=True
            )
        with c2:
            hyb_c = '#e74c3c' if hybrid_result['subject_pred'] == 'ASD' else '#2ecc71'
            st.markdown(
                f"<div class='model-card'>"
                f"<b style='color:#8e44ad'>Hybrid CNN-ViT (AUC 0.943)</b><br>"
                f"<div style='font-size:2em;color:{hyb_c};font-weight:900'>{hybrid_result['subject_pred']}</div>"
                f"P(ASD)={hyb_prob:.4f} · Conf={hybrid_result['subject_conf']:.1%}<br>"
                f"Votes: {hybrid_result['pred_asd_votes']} ASD / {hybrid_result['pred_tc_votes']} TC<br>"
                f"σ={sig_hyb_str} · Weight: {w_vit:.3f}"
                f"</div>",
                unsafe_allow_html=True
            )

        # Ensemble weights explanation
        st.markdown("---")
        st.markdown("## Uncertainty-Gated Ensemble Fusion")
        st.markdown(
            f"<div class='ensemble-weights'>"
            f"<b>How the ensemble weight is computed:</b><br>"
            f"w_CNN = (1 − σ_CNN) × AUC_CNN = (1 − {sig_cnn_str}) × 0.994 = {w_cnn:.4f}<br>"
            f"w_ViT = (1 − σ_ViT) × AUC_ViT = (1 − {sig_hyb_str}) × 0.943 = {w_vit:.4f}<br>"
            f"P_ensemble = ({w_cnn:.4f} × {cnn_prob:.4f} + {w_vit:.4f} × {hyb_prob:.4f}) "
            f"/ ({w_cnn:.4f}+{w_vit:.4f}) = <b>{ens_prob:.4f}</b>"
            f"</div>",
            unsafe_allow_html=True
        )
        st.caption("The model with lower MC-Dropout σ on this specific scan receives higher weight — "
                   "regardless of its average performance. This adapts to per-scan confidence.")

        # XAI agreement
        if xai_agree:
            st.markdown("---")
            st.markdown("## XAI Agreement — CNN GradCAM vs ViT Attention")
            agree_lvl  = xai_agree.get('agreement', 'low')
            agree_cls  = f"agree-{agree_lvl}"
            agree_icon = {'high': '✅', 'moderate': '⚡', 'low': '⚠️'}.get(agree_lvl, '⚠️')
            st.markdown(
                f"<div class='{agree_cls}'>"
                f"{agree_icon} <b>Spatial Agreement: {agree_lvl.upper()}</b> · "
                f"Jaccard IoU={xai_agree['iou']:.3f} · Pearson r={xai_agree['pearson_r']:.3f}<br>"
                f"<small>{xai_agree['interpretation']}</small>"
                f"</div>",
                unsafe_allow_html=True
            )

            for i, arr in enumerate(dual_fig_arrays):
                with st.expander(f"XAI Comparison — Slice {i+1}", expanded=(i == 0)):
                    _show_image(arr)
                    st.caption("Left→Right: Original MRI | CNN GradCAM | CNN overlay | "
                               "ViT Attention | ViT overlay. IoU and Pearson r in title bar.")

        # Narrative
        if st.session_state.ensemble_narrative:
            st.markdown("---")
            st.markdown("## Clinical Interpretation (LLM)")
            st.markdown(
                f"<div class='narrative-box'>{st.session_state.ensemble_narrative}</div>",
                unsafe_allow_html=True
            )

        # Download
        if st.session_state.ensemble_pdf:
            st.markdown("---")
            st.download_button(
                "📄 Download Ensemble Report (PDF)",
                data=st.session_state.ensemble_pdf,
                file_name="asd_ensemble_report.pdf",
                mime="application/pdf",
            )

        st.markdown(
            "<div class='disclaimer'>⚠️ Research use only. Ensemble: CNN + Hybrid CNN-ViT · "
            "Uncertainty-gated fusion · ABIDE-I · University of North Texas.</div>",
            unsafe_allow_html=True
        )
