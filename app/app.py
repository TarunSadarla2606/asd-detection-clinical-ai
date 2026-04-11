# app.py — v6 final stable
# Fixes: no mid-loop status updates, MC-Dropout checkbox restored,
# uploaded file handled once via session state to prevent rerun on download

import streamlit as st
import tempfile, os, io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model      import load_model
from pipeline   import run_subject_pipeline
from xai        import run_gradcam, mc_dropout_uncertainty, make_explanation_figure
from phenotypic import load_phenotypic, get_subject_info, extract_anon_num_from_filename
from anatomy    import label_region
from narrator   import generate_clinical_narrative
from report     import generate_report

st.set_page_config(
    page_title = "ASD Detection — Clinical AI Demo",
    page_icon  = "🧠",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background:#0e1117; }
    [data-testid="stSidebar"]          { background:#161b27; border-right:1px solid #2d3748; }
    h1,h2,h3,h4                        { color:#e2e8f0 !important; }
    p, li, label                       { color:#cbd5e0; }
    .pred-card { padding:28px; border-radius:14px; text-align:center; margin:14px 0; }
    .pred-asd  { background:linear-gradient(135deg,#c0392b18,#c0392b35); border:2px solid #c0392b; }
    .pred-tc   { background:linear-gradient(135deg,#27ae6018,#27ae6035); border:2px solid #27ae60; }
    .narrative-box { background:#1a202c; border-left:4px solid #4a90d9;
                     padding:16px 20px; border-radius:6px; margin:12px 0;
                     font-size:0.93em; line-height:1.7; color:#cbd5e0; }
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
        background:#1a202c; border:1px solid #2d3748; border-radius:8px; padding:10px;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────
for key, default in [
    ('result',        None),
    ('subject_info',  None),
    ('source_name',   None),
    ('gradcam_stats', []),
    ('uncertainty',   None),
    ('narrative',     None),
    ('fig_pngs',      []),
    ('pdf_bytes',     None),
    ('compute_key',   None),
    ('saved_nifti',   None),   # persists uploaded file path across reruns
    ('saved_name',    None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


@st.cache_resource
def get_model():
    return load_model("weights/xai_cnn_best_weights.pth", device="cpu")

@st.cache_resource
def get_pheno():
    return load_phenotypic()


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 ASD Detection")
    st.markdown("**Research Demo** · University of North Texas")
    st.markdown("---")
    st.markdown("### Settings")
    top_k           = st.slider("XAI slices", 1, 5, 3)
    run_lime_flag   = st.checkbox("LIME explanations (~20s/slice)", value=False)
    run_uncertainty = st.checkbox("MC-Dropout uncertainty", value=True)
    run_narrative   = st.checkbox("LLM clinical narrative", value=True)
    st.markdown("---")
    st.markdown("### Model")
    st.markdown("""
| Metric | Value |
|--------|-------|
| AUC | **0.994** |
| Sensitivity | **95.6%** |
| Specificity | **97.2%** |
| Brier Score | **0.027** |
| Architecture | 5-layer CNN |
| Dataset | ABIDE-I |
""")
    st.markdown("---")
    st.markdown("<div style='font-size:0.78em;color:#718096'>Research use only. Not for clinical deployment.</div>",
                unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("<div style='font-size:3.5em;margin-top:8px'>🧠</div>", unsafe_allow_html=True)
with col_title:
    st.title("ASD Detection from Structural MRI")
    st.markdown(
        "Upload a **NIfTI (.nii / .nii.gz)** brain scan. The pipeline extracts "
        "quality-filtered axial slices, classifies each one with a trained CNN, "
        "aggregates to a **subject-level prediction**, and explains the decision "
        "using **Grad-CAM**, **LIME**, and an **LLM-generated clinical narrative**."
    )

st.markdown("---")

# ── File input ─────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload NIfTI file", type=["nii", "gz"])

st.markdown("#### Or try a demo subject:")
demo_cols = st.columns(4)
demo_file = None
demo_dir  = "demo_subjects"
if os.path.exists(demo_dir):
    demo_files = sorted([f for f in os.listdir(demo_dir)
                         if f.endswith(('.nii', '.nii.gz'))])
    for i, fname in enumerate(demo_files[:4]):
        label = fname.replace('.nii.gz', '').replace('.nii', '')
        if demo_cols[i % 4].button(f"📂 {label}", use_container_width=True):
            demo_file = os.path.join(demo_dir, fname)
            # Clear saved upload when demo button clicked
            st.session_state.saved_nifti = None
            st.session_state.saved_name  = None

# ── Resolve input — save uploaded file ONCE to session state ──────────────
# This prevents re-saving on every rerun (which changes compute_key)
if uploaded is not None:
    file_id = f"{uploaded.name}_{uploaded.size}"
    if st.session_state.saved_name != file_id:
        suffix = '.nii.gz' if uploaded.name.endswith('.gz') else '.nii'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            st.session_state.saved_nifti = tmp.name
            st.session_state.saved_name  = file_id

nifti_path  = st.session_state.saved_nifti if uploaded is not None else (
              demo_file if demo_file else None)
source_name = (uploaded.name if uploaded is not None else
               (os.path.basename(demo_file) if demo_file else None))


# ══════════════════════════════════════════════════════════════════════════
# STAGE 1 — ALL COMPUTATION
# Runs only when compute_key changes. No display code here.
# Status updates use a placeholder updated ONCE per step (not in loops).
# ══════════════════════════════════════════════════════════════════════════
compute_key = (f"{nifti_path}_{top_k}_{run_lime_flag}"
               f"_{run_uncertainty}_{run_narrative}") if nifti_path else None

if compute_key and compute_key != st.session_state.compute_key:
    model    = get_model()
    pheno_df = get_pheno()

    anon_num     = extract_anon_num_from_filename(source_name) if source_name else None
    subject_info = (get_subject_info(anon_num, pheno_df)
                    if anon_num and pheno_df is not None else None)

    # Single progress bar + single status text — updated only between steps
    pbar   = st.progress(0)
    status = st.empty()

    # Step 1: Pipeline
    status.text("Step 1/6 — Running inference pipeline...")
    result = run_subject_pipeline(
        nifti_path, model, device='cpu', top_k=top_k,
        progress_fn=None   # no per-slice callbacks — avoids mid-loop reruns
    )
    pbar.progress(20)
    if 'error' in result:
        st.error(result['error'])
        st.stop()

    # Step 2: GradCAM + anatomy (no per-slice UI updates)
    status.text("Step 2/6 — Computing GradCAM explanations...")
    gradcam_stats = []
    for sl in result['top_slices']:
        gc = run_gradcam(sl['arr_cropped'], model, sl['pred_class'], device='cpu')
        ar = label_region(sl['slice_idx'], gc['heatmap'])
        gradcam_stats.append({
            'slice_idx': sl['slice_idx'],
            'prob_asd' : sl['prob_asd'],
            'energy'   : gc['energy'],
            'region'   : ar['full_label'],
            'lobe'     : ar['lobe'],
            'spatial'  : ar['spatial'],
            'com_x'    : ar['com_x'],
            'com_y'    : ar['com_y'],
        })
    pbar.progress(40)

    # Step 3: MC-Dropout
    uncertainty = None
    if run_uncertainty and result['top_slices']:
        status.text("Step 3/6 — Estimating prediction uncertainty...")
        uncertainty = mc_dropout_uncertainty(
            result['top_slices'][0]['arr_cropped'], model, n_passes=30
        )
    pbar.progress(55)

    # Step 4: XAI figures → PNG bytes
    status.text("Step 4/6 — Rendering XAI figures...")
    fig_pngs = []
    for sl in result['top_slices']:
        fig = make_explanation_figure(
            sl['arr_cropped'], model,
            pred_class    = sl['pred_class'],
            slice_idx     = sl['slice_idx'],
            device        = 'cpu',
            run_lime_flag = run_lime_flag
        )
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        fig_pngs.append(buf.getvalue())
        plt.close(fig)
    pbar.progress(75)

    # Step 5: LLM narrative
    narrative = None
    if run_narrative:
        status.text("Step 5/6 — Generating clinical narrative...")
        lime_stats_for_narrator = None
        if run_lime_flag and fig_pngs:
            lime_stats_for_narrator = [
                {'slice_idx': sl['slice_idx'], 'lime_ran': True,
                 'n_pos': 10, 'n_neg': 10, 'max_weight': 0.3,
                 'sum_pos': 1.0, 'sum_neg': -0.5}
                for sl in result['top_slices']
            ]
        narrative = generate_clinical_narrative(
            result, subject_info, uncertainty, gradcam_stats,
            lime_stats=lime_stats_for_narrator
        )
    pbar.progress(88)

    # Step 6: PDF
    status.text("Step 6/6 — Building PDF report...")
    report_figs = []
    for png_bytes in fig_pngs:
        fig, ax = plt.subplots(figsize=(16, 4))
        img = plt.imread(io.BytesIO(png_bytes))
        ax.imshow(img)
        ax.axis('off')
        fig.patch.set_facecolor('#0e1117')
        report_figs.append(fig)
    pdf_bytes = generate_report(
        result=result, subject_info=subject_info,
        xai_figures=report_figs, narrative=narrative,
    )
    for fig in report_figs:
        plt.close(fig)
    pbar.progress(100)
    status.text("Complete.")

    # Single batch write to session state
    st.session_state.update({
        'result':        result,
        'subject_info':  subject_info,
        'source_name':   source_name,
        'gradcam_stats': gradcam_stats,
        'uncertainty':   uncertainty,
        'fig_pngs':      fig_pngs,
        'narrative':     narrative,
        'pdf_bytes':     pdf_bytes,
        'compute_key':   compute_key,
    })

    pbar.empty()
    status.empty()


# ══════════════════════════════════════════════════════════════════════════
# STAGE 2 — PURE DISPLAY (zero computation, zero session state writes)
# ══════════════════════════════════════════════════════════════════════════
if st.session_state.result is not None:
    result        = st.session_state.result
    subject_info  = st.session_state.subject_info
    source_name   = st.session_state.source_name
    gradcam_stats = st.session_state.gradcam_stats
    uncertainty   = st.session_state.uncertainty
    fig_pngs      = st.session_state.fig_pngs
    narrative     = st.session_state.narrative
    pdf_bytes     = st.session_state.pdf_bytes

    pred     = result['subject_pred']
    conf     = result['subject_conf']
    prob_asd = result['weighted_prob_asd']
    color    = '#e74c3c' if pred == 'ASD' else '#2ecc71'
    icon     = '🔴' if pred == 'ASD' else '🟢'

    # 1. Prediction
    st.markdown("## Subject-Level Prediction")
    st.markdown(f"""
    <div class='pred-card pred-{"asd" if pred=="ASD" else "tc"}'>
        <div style='font-size:3.2em;font-weight:900;color:{color};
                    font-family:monospace;letter-spacing:2px'>{icon} {pred}</div>
        <div style='font-size:1.6em;color:#e2e8f0;margin:6px 0'>Confidence: {conf:.1%}</div>
        <div style='color:#a0aec0;font-size:0.9em'>
            Weighted consensus across {result["total_valid"]} quality-filtered slices
        </div>
    </div>""", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ASD votes",    result['pred_asd_votes'])
    m2.metric("TC votes",     result['pred_tc_votes'])
    m3.metric("Valid slices", result['total_valid'])
    m4.metric("Total slices", result['n_total_slices'])

    bar_color = '#e74c3c' if prob_asd >= 0.5 else '#2ecc71'
    st.markdown("**Weighted P(ASD):**")
    st.markdown(f"""
    <div style='background:#2d3748;border-radius:8px;padding:4px;margin:4px 0'>
        <div style='background:{bar_color};width:{prob_asd*100:.1f}%;min-width:44px;
                    height:26px;border-radius:6px;text-align:center;
                    line-height:26px;color:white;font-weight:700;font-size:0.9em'>
            {prob_asd:.1%} P(ASD)
        </div>
    </div>""", unsafe_allow_html=True)

    # 2. Subject context
    st.markdown("---")
    st.markdown("## Subject & Site Context")
    if subject_info:
        si = subject_info
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sex",             si['sex'])
        c2.metric("Age",             f"{si['age']} yrs" if si['age'] else "N/A")
        c3.metric("Site",            si['site'])
        c4.metric("ABIDE Diagnosis", si['true_label'])
        if si['true_label'] != 'Unknown':
            if pred == si['true_label']:
                st.success(f"Prediction **{pred}** matches ABIDE-I ground truth **{si['true_label']}**")
            else:
                st.error(f"Prediction **{pred}** differs from ABIDE-I ground truth **{si['true_label']}**")
        if si['site_perf']:
            sp  = si['site_perf']
            low = sp['Sensitivity'] < 0.92 or sp['Specificity'] < 0.95
            cls = 'site-warn' if low else 'site-ok'
            ico = '⚠️' if low else '✅'
            st.markdown(f"""<div class='{cls}'>
            {ico} <b>Site {si['site']}:</b>
            Sens {sp['Sensitivity']:.1%} · Spec {sp['Specificity']:.1%} ·
            AUC {sp['AUC']:.3f} · n={sp['N_subjects']}
            {"<br><b>Below reliability threshold.</b>" if low else ""}
            </div>""", unsafe_allow_html=True)
        with st.expander("Full phenotypic record"):
            st.json({k: v for k, v in si.items() if k != 'site_perf'})
    else:
        st.info("Subject not found in phenotypic database.")

    # 3. Uncertainty
    if uncertainty:
        st.markdown("---")
        st.markdown("## Prediction Uncertainty (MC-Dropout)")
        st.markdown("30 stochastic forward passes. Low σ = confident. High σ = treat with caution.")
        u1, u2, u3 = st.columns(3)
        u1.metric("Mean P(ASD)", f"{uncertainty['mean_prob_asd']:.4f}")
        u2.metric("Std Dev (σ)", f"{uncertainty['std']:.4f}")
        u3.metric("Assessment",  uncertainty['uncertainty'].split('—')[0].strip())
        st.info(f"🔍 {uncertainty['uncertainty']}")

    # 4. XAI (st.image from bytes — never triggers rerun)
    st.markdown("---")
    st.markdown("## Explainability — Top Slices")
    st.markdown("Grad-CAM: red/yellow = high activation. LIME: green = supports, red = contradicts.")
    for i, (sl, png_bytes) in enumerate(zip(result['top_slices'], fig_pngs)):
        gc_info = gradcam_stats[i] if i < len(gradcam_stats) else {}
        region  = gc_info.get('region', '')
        with st.expander(
            f"Slice z={sl['slice_idx']}  ·  P(ASD)={sl['prob_asd']:.3f}  ·  "
            f"Quality={sl['quality_score']:.3f}  ·  "
            f"Pred={'ASD' if sl['pred_class']==1 else 'TC'}",
            expanded=(i == 0)
        ):
            st.image(png_bytes, use_container_width=True)
            r1, r2, r3 = st.columns(3)
            r1.metric("P(ASD)",        f"{sl['prob_asd']:.4f}")
            r2.metric("P(TC)",         f"{sl['prob_tc']:.4f}")
            r3.metric("Quality score", f"{sl['quality_score']:.4f}")
            if region:
                st.markdown(f"📍 **Region:** <span class='region-tag'>{region}</span>",
                            unsafe_allow_html=True)

    # 5. Anatomical regions
    if gradcam_stats:
        st.markdown("---")
        st.markdown("## Activation Region Summary")
        st.markdown("Approximate anatomical region of peak GradCAM activation "
                    "(heuristic z-position, not atlas-registered).")
        cols = st.columns(len(gradcam_stats))
        for i, (gc, col) in enumerate(zip(gradcam_stats, cols)):
            with col:
                st.markdown(f"**z={gc['slice_idx']}**")
                col.metric("P(ASD)", f"{gc['prob_asd']:.3f}")
                col.metric("Energy", f"{gc['energy']:.3f}")
                st.markdown(
                    f"<span class='region-tag'>{gc['lobe']}</span>"
                    f"<span class='region-tag'>{gc['spatial']}</span>",
                    unsafe_allow_html=True
                )

    # 6. Narrative
    if narrative:
        st.markdown("---")
        st.markdown("## Clinical Interpretation")
        st.markdown("*Generated by Claude Haiku (claude-haiku-4-5). Research use only — not a clinical opinion.*")
        clean = narrative.replace('## ', '').replace('# ', '').replace('**', '')
        st.markdown(
            f"<div class='narrative-box'>{clean.replace(chr(10), '<br><br>')}</div>",
            unsafe_allow_html=True
        )

    # 7. PDF — pre-generated, instant download, no rerun
    st.markdown("---")
    st.markdown("## Clinical Report (PDF)")
    st.markdown("Pre-generated report including prediction, subject metadata, "
                "GradCAM figures, and clinical narrative.")
    if pdf_bytes:
        fname = f"ASD_Report_{(source_name or 'subject').replace('.nii.gz','').replace('.nii','')}.pdf"
        st.download_button(
            label     = "Download Clinical Report (PDF)",
            data      = pdf_bytes,
            file_name = fname,
            mime      = "application/pdf",
            type      = "primary"
        )

    # 8. Disclaimer
    st.markdown("---")
    st.markdown("""
    <div class='disclaimer'>
    <b>Research Use Only — Not a Medical Device</b><br>
    Trained on ABIDE-I (retrospective, multi-site, 2D slice-level sMRI).
    Not prospectively validated. Not a medical diagnosis.
    Clinical deployment requires prospective validation, FDA SaMD Class II clearance,
    neuroradiologist review, and IRB approval.<br><br>
    <b>Author:</b> Tarun Sadarla, University of North Texas (MS AI, Biomedical) ·
    <b>AUC:</b> 0.994 · <b>Sensitivity:</b> 95.6% · <b>Specificity:</b> 97.2%
    </div>""", unsafe_allow_html=True)
