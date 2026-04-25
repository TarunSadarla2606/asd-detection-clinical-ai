# report.py — v2
# Generate PDF clinical summary reports for all three tabs
# Tab 1: CNN report  |  Tab 2: Hybrid CNN-ViT report  |  Tab 3: Ensemble report
# All reports use LLM-generated narratives — no template text in the happy path.

import io
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage, KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


# ── Style constants ───────────────────────────────────────────────────────────
DARK      = colors.HexColor('#1a1a2e')
MID       = colors.HexColor('#2c3e50')
LIGHT_BG  = colors.HexColor('#f8f9fa')
AMBER     = colors.HexColor('#f39c12')
RED_CLR   = colors.HexColor('#c0392b')
GREEN_CLR = colors.HexColor('#27ae60')
BLUE_CLR  = colors.HexColor('#2980b9')
PURPLE    = colors.HexColor('#8e44ad')
GRAY_TEXT = colors.HexColor('#7f8c8d')


def _fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()


def _make_doc(buf) -> SimpleDocTemplate:
    return SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2.5*cm
    )


def _styles():
    s  = getSampleStyleSheet()
    h1 = ParagraphStyle('H1', parent=s['Title'],
                        fontSize=18, textColor=DARK, spaceAfter=4,
                        fontName='Helvetica-Bold')
    h2 = ParagraphStyle('H2', parent=s['Heading2'],
                        fontSize=11, textColor=DARK, fontName='Helvetica-Bold',
                        spaceBefore=10, spaceAfter=4)
    sub = ParagraphStyle('Sub', parent=s['Normal'],
                         fontSize=8.5, textColor=GRAY_TEXT, spaceAfter=2)
    body = ParagraphStyle('Body', parent=s['Normal'],
                          fontSize=8.5, textColor=MID, leading=13, spaceAfter=5)
    narr = ParagraphStyle('Narr', parent=s['Normal'],
                          fontSize=9, textColor=colors.HexColor('#2c3e50'),
                          leading=14, spaceAfter=7)
    disc = ParagraphStyle('Disc', parent=s['Normal'],
                          fontSize=7, textColor=GRAY_TEXT, leading=10)
    warn = ParagraphStyle('Warn', parent=s['Normal'],
                          fontSize=8.5, textColor=RED_CLR,
                          fontName='Helvetica-Bold', leading=13)
    return dict(h1=h1, h2=h2, sub=sub, body=body, narr=narr, disc=disc, warn=warn, base=s)


def _header_block(story, styles, title_line: str, mode_label: str,
                  pred: str, conf: float,
                  pred_col, result: dict):
    """Shared header: title + prediction box + vote metrics."""
    S = styles
    story.append(Paragraph(title_line, S['h1']))
    story.append(Paragraph(
        f"Research Grade · University of North Texas · ABIDE-I Dataset · {mode_label}",
        S['sub']))
    story.append(Paragraph(
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        S['sub']))
    story.append(HRFlowable(width='100%', thickness=2, color=DARK, spaceAfter=10))

    # Prediction box
    pt = Table(
        [[Paragraph(pred, ParagraphStyle('PL', fontSize=28,
                                         textColor=pred_col, fontName='Helvetica-Bold',
                                         alignment=TA_CENTER))],
         [Paragraph(f"Confidence: {conf:.1%}",
                    ParagraphStyle('CF', fontSize=14, textColor=MID,
                                   alignment=TA_CENTER))],
         [Paragraph(
             f"Based on {result['total_valid']} valid slices — "
             f"ASD votes: {result['pred_asd_votes']} | TC votes: {result['pred_tc_votes']}",
             ParagraphStyle('VT', fontSize=8, textColor=MID, alignment=TA_CENTER)
         )]],
        colWidths=['100%']
    )
    pt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), LIGHT_BG),
        ('BOX',        (0,0), (-1,-1), 2, pred_col),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(pt)
    story.append(Spacer(1, 10))


def _subject_block(story, styles, subject_info):
    S = styles
    story.append(Paragraph("Subject Information", S['h2']))
    if subject_info:
        si = subject_info
        site_note = ''
        if si.get('site_perf'):
            sp = si['site_perf']
            site_note = (f" | Sens={sp['Sensitivity']:.3f} Spec={sp['Specificity']:.3f} "
                         f"AUC={sp['AUC']:.3f} (n={sp['N_subjects']})")
        rows = [
            ['Field', 'Value'],
            ['Subject ID (anonymised)', str(si.get('anon_id','N/A'))],
            ['ABIDE Diagnosis (ground truth)', si.get('true_label','Unknown')],
            ['Sex', si.get('sex','Unknown')],
            ['Age at Scan', f"{si.get('age','N/A')} years" if si.get('age') else 'N/A'],
            ['Acquisition Site', f"{si.get('site','N/A')}{site_note}"],
        ]
    else:
        rows = [['Field','Value'],['Subject ID','Unknown — no phenotypic match found']]

    t = Table(rows, colWidths=[5*cm, 12*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), MID),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 8),
        ('BACKGROUND', (0,1), (-1,-1), LIGHT_BG),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white, LIGHT_BG]),
        ('GRID',       (0,0), (-1,-1), 0.5, colors.HexColor('#dee2e6')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING',(0,0),(-1,-1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))


def _narrative_block(story, styles, narrative: str, source_label: str):
    S = styles
    story.append(Paragraph(f"Clinical Interpretation — {source_label}", S['h2']))
    story.append(Paragraph(
        "Generated by Claude Haiku from structured model outputs. Research use only.",
        ParagraphStyle('it', fontSize=7, textColor=GRAY_TEXT, fontName='Helvetica-Oblique')
    ))
    story.append(Spacer(1, 4))
    clean = (narrative or '').replace('**','').replace('*','')
    for para in clean.split('\n\n'):
        para = para.strip()
        if para:
            story.append(Paragraph(para, S['narr']))


def _disclaimer(story, styles, model_note: str = '5-layer CNN'):
    S = styles
    story.append(HRFlowable(width='100%', thickness=1, color=AMBER, spaceAfter=6))
    story.append(Paragraph(
        f"⚠️ RESEARCH USE ONLY — This report was generated by a research-grade AI system "
        f"trained on the ABIDE-I dataset (retrospective, multi-site, 2D axial sMRI). "
        f"It has NOT been validated for clinical deployment, does not constitute a medical "
        f"diagnosis, and must not guide clinical decisions. Clinical deployment would require "
        f"prospective validation and FDA SaMD Class II regulatory clearance. "
        f"Model: {model_note} | Dataset: ABIDE-I | Author: Tarun Sadarla, UNT.",
        S['disc']
    ))


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(
    result       : dict,
    subject_info : dict | None,
    xai_figures  : list,          # matplotlib figures
    narrative    : str | None = None,
    mode         : str = 'cnn',   # 'cnn' | 'hybrid' | 'ensemble'
    # Ensemble-only extras
    hybrid_result      : dict | None = None,
    ensemble_result    : dict | None = None,
    cnn_uncertainty    : dict | None = None,
    hybrid_uncertainty : dict | None = None,
    xai_agreement      : dict | None = None,
    hybrid_xai_figures : list | None = None,
    dual_xai_figures   : list | None = None,
) -> bytes:
    """
    Generate a PDF report. mode controls structure and content.

    Tab 1 → mode='cnn'
    Tab 2 → mode='hybrid'
    Tab 3 → mode='ensemble'
    """
    if mode == 'cnn':
        return _generate_cnn_report(result, subject_info, xai_figures, narrative)
    elif mode == 'hybrid':
        return _generate_hybrid_report(result, subject_info, xai_figures, narrative,
                                       hybrid_uncertainty)
    else:
        return _generate_ensemble_report(
            cnn_result=result,
            hybrid_result=hybrid_result or {},
            ensemble_result=ensemble_result or {},
            subject_info=subject_info,
            narrative=narrative,
            cnn_uncertainty=cnn_uncertainty,
            hybrid_uncertainty=hybrid_uncertainty,
            xai_agreement=xai_agreement,
            cnn_xai_figures=xai_figures,
            hybrid_xai_figures=hybrid_xai_figures or [],
            dual_xai_figures=dual_xai_figures or [],
        )


# ── Tab 1: CNN report ─────────────────────────────────────────────────────────

def _generate_cnn_report(result, subject_info, xai_figures, narrative) -> bytes:
    buf   = io.BytesIO()
    doc   = _make_doc(buf)
    S     = _styles()
    story = []

    pred     = result['subject_pred']
    conf     = result['subject_conf']
    pred_col = RED_CLR if pred == 'ASD' else GREEN_CLR

    _header_block(story, S, "ASD Detection — Clinical AI Report",
                  "Pure CNN · Phase 2", pred, conf, pred_col, result)

    # Model performance table
    story.append(Paragraph("Model Performance (Validation Cohort)", S['h2']))
    story.append(Paragraph(
        "5-layer custom CNN trained on ABIDE-I. Slice-level inference aggregated by "
        "confidence-weighted voting. Performance evaluated on held-out test set.",
        S['body']))
    perf = Table(
        [['AUC','Sensitivity','Specificity','F1','Brier Score','Architecture'],
         ['0.994','95.6%','97.2%','95.1%','0.027','5-layer CNN']],
        colWidths=[2.8*cm]*6
    )
    perf.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0), MID),('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),9),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),('BACKGROUND',(0,1),(-1,-1),LIGHT_BG),
        ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#dee2e6')),
        ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
    ]))
    story.append(perf)
    story.append(Spacer(1, 10))

    _subject_block(story, S, subject_info)

    # XAI figures
    if xai_figures:
        story.append(Paragraph("Explainability — Grad-CAM Analysis", S['h2']))
        story.append(Paragraph(
            "GradCAM (Gradient-weighted Class Activation Mapping) on conv5 layer. "
            "Red/yellow = high model activation. Slices selected by quality score "
            "(circularity · brain coverage · sharpness).", S['body']))
        story.append(Spacer(1, 4))
        for fig in xai_figures:
            story.append(RLImage(io.BytesIO(_fig_to_bytes(fig)), width=16*cm, height=4*cm))
            story.append(Spacer(1, 4))

    _narrative_block(story, S, narrative, "CNN Model (Claude Haiku)")
    _disclaimer(story, S, model_note='5-layer CNN | AUC 0.994')
    doc.build(story)
    return buf.getvalue()


# ── Tab 2: Hybrid CNN-ViT report ──────────────────────────────────────────────

def _generate_hybrid_report(result, subject_info, xai_figures, narrative,
                             hybrid_uncertainty) -> bytes:
    buf   = io.BytesIO()
    doc   = _make_doc(buf)
    S     = _styles()
    story = []

    pred     = result['subject_pred']
    conf     = result['subject_conf']
    pred_col = RED_CLR if pred == 'ASD' else GREEN_CLR

    _header_block(story, S, "ASD Detection — Hybrid CNN-ViT Report",
                  "CNN + Transformer Encoder", pred, conf, pred_col, result)

    # Architecture explanation (clinical language)
    story.append(Paragraph("Architecture Overview", S['h2']))
    story.append(Paragraph(
        "The Hybrid CNN-ViT processes each MRI slice in two stages. First, a pretrained "
        "4-block convolutional neural network (conv1–conv4, initialised from Phase 2 weights) "
        "extracts a 14×14 grid of spatial feature tokens — 196 positions representing distinct "
        "regions of the axial brain slice. Background and skull regions are automatically "
        "masked out at token resolution so the transformer never processes uninformative "
        "black-region tokens.",
        S['body']))
    story.append(Paragraph(
        "Second, a 4-layer Transformer encoder (8 attention heads, 256-dimensional embedding) "
        "learns global self-attention relationships across all surviving brain-region tokens "
        "simultaneously. Unlike GradCAM, which highlights local gradient sensitivity, the "
        "transformer's CLS classification token aggregates information from the entire slice "
        "through self-attention — capturing long-range spatial dependencies between brain regions.",
        S['body']))
    story.append(Spacer(1, 6))

    # Model comparison table
    story.append(Paragraph("Performance Comparison", S['h2']))
    comp = Table(
        [['Metric','CNN (Phase 2)','Hybrid CNN-ViT'],
         ['AUC','0.994','0.943'],
         ['Sensitivity','95.6%','84.5%'],
         ['Specificity','97.2%','86.8%'],
         ['AUPRC','0.994','0.944'],
         ['Brier Score','0.027','0.097'],
         ['Architecture','5-layer CNN','CNN backbone + 4-layer Transformer']],
        colWidths=[5*cm, 4*cm, 8*cm]
    )
    comp.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),MID),('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8.5),
        ('BACKGROUND',(0,1),(-1,-1),LIGHT_BG),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,LIGHT_BG]),
        ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#dee2e6')),
        ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),
        ('ALIGN',(1,0),(-1,-1),'CENTER'),
    ]))
    story.append(comp)
    story.append(Paragraph(
        "Note: Lower Hybrid CNN-ViT performance is expected in this limited-data regime "
        "(~67K slices). Transformer architectures typically require substantially more training "
        "data than CNNs to match or exceed CNN performance. The architectural value lies in "
        "complementary spatial attention mechanisms, not raw performance.",
        ParagraphStyle('note', fontSize=7.5, textColor=GRAY_TEXT, leading=11, spaceAfter=6)
    ))
    story.append(Spacer(1, 6))

    _subject_block(story, S, subject_info)

    # Uncertainty
    if hybrid_uncertainty:
        story.append(Paragraph("Uncertainty Estimation (MC-Dropout)", S['h2']))
        story.append(Paragraph(
            f"Both CNN Dropout2d and Transformer Dropout layers were activated for "
            f"{hybrid_uncertainty['n_passes']} stochastic forward passes. "
            f"Mean P(ASD): {hybrid_uncertainty['mean_prob_asd']:.4f} | "
            f"Std Dev (σ): {hybrid_uncertainty['std']:.4f}. "
            f"Assessment: {hybrid_uncertainty['uncertainty']}. "
            f"The hybrid model characteristically produces higher σ than the pure CNN "
            f"because both dropout pathways contribute stochastic variance — the model "
            f"is simultaneously uncertain about local texture features (CNN pathway) and "
            f"global spatial relationships (transformer pathway).",
            S['body']))
        story.append(Spacer(1, 6))

    # Attention figures
    if xai_figures:
        story.append(Paragraph("Explainability — Attention Maps", S['h2']))
        story.append(Paragraph(
            "Each panel shows: original MRI | brain token mask (14×14, green=brain, red=masked) | "
            "CLS→token self-attention (head-averaged, last transformer layer) | attention overlay. "
            "High attention weight = region the transformer classified as globally important "
            "for the ASD/TC decision. Unlike GradCAM, attention captures cross-region "
            "relationships rather than local gradient sensitivity.",
            S['body']))
        story.append(Spacer(1, 4))
        for fig in xai_figures:
            story.append(RLImage(io.BytesIO(_fig_to_bytes(fig)), width=16*cm, height=4*cm))
            story.append(Spacer(1, 4))

    _narrative_block(story, S, narrative, "Hybrid CNN-ViT Model (Claude Haiku)")
    _disclaimer(story, S, model_note='Hybrid CNN-ViT | AUC 0.943')
    doc.build(story)
    return buf.getvalue()


# ── Tab 3: Ensemble report ────────────────────────────────────────────────────

def _generate_ensemble_report(
    cnn_result, hybrid_result, ensemble_result, subject_info,
    narrative, cnn_uncertainty, hybrid_uncertainty, xai_agreement,
    cnn_xai_figures, hybrid_xai_figures, dual_xai_figures
) -> bytes:
    buf   = io.BytesIO()
    doc   = _make_doc(buf)
    S     = _styles()
    story = []

    ens_pred = ensemble_result.get('ensemble_pred', '?')
    ens_prob = ensemble_result.get('ensemble_prob_asd', 0.5)
    ens_conf = ensemble_result.get('ensemble_conf', 0.5)
    disagree = cnn_result.get('subject_pred') != hybrid_result.get('subject_pred')
    pred_col = RED_CLR if ens_pred == 'ASD' else GREEN_CLR

    story.append(Paragraph("ASD Detection — Ensemble Analysis Report", S['h1']))
    story.append(Paragraph(
        "Research Grade · University of North Texas · ABIDE-I Dataset · "
        "Uncertainty-Gated CNN + Transformer Ensemble",
        S['sub']))
    story.append(Paragraph(
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        S['sub']))
    story.append(HRFlowable(width='100%', thickness=2, color=DARK, spaceAfter=10))

    # Model disagreement banner (critical clinical alert)
    if disagree:
        warn_table = Table(
            [[Paragraph(
                "⚠️ MODEL DISAGREEMENT — Expert Radiologist Review Required\n"
                f"CNN predicts {cnn_result.get('subject_pred')} · "
                f"Hybrid ViT predicts {hybrid_result.get('subject_pred')}. "
                "When architecturally diverse models trained on identical data disagree, "
                "the ambiguity cannot be resolved algorithmically. "
                "This result must not be used without expert review.",
                S['warn']
            )]],
            colWidths=['100%']
        )
        warn_table.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#fff5f5')),
            ('BOX',(0,0),(-1,-1),2,RED_CLR),
            ('TOPPADDING',(0,0),(-1,-1),10),
            ('BOTTOMPADDING',(0,0),(-1,-1),10),
        ]))
        story.append(warn_table)
        story.append(Spacer(1, 8))

    # Ensemble verdict
    ens_table = Table(
        [[Paragraph(ens_pred,
                    ParagraphStyle('EP', fontSize=26, textColor=pred_col,
                                   fontName='Helvetica-Bold', alignment=TA_CENTER))],
         [Paragraph(f"Ensemble Confidence: {ens_conf:.1%}  |  Ensemble P(ASD): {ens_prob:.4f}",
                    ParagraphStyle('EC', fontSize=12, textColor=MID, alignment=TA_CENTER))],
         [Paragraph(
             f"CNN weight: {ensemble_result.get('weight_cnn',0.5):.3f}  |  "
             f"ViT weight: {ensemble_result.get('weight_vit',0.5):.3f}  "
             f"(uncertainty-gated)",
             ParagraphStyle('EW', fontSize=8, textColor=MID, alignment=TA_CENTER)
         )]],
        colWidths=['100%']
    )
    ens_table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,-1),LIGHT_BG),
        ('BOX',(0,0),(-1,-1),2,pred_col),
        ('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),
    ]))
    story.append(ens_table)
    story.append(Spacer(1, 10))

    # Side-by-side model comparison table
    story.append(Paragraph("Individual Model Results", S['h2']))

    cnn_sig = f"{cnn_uncertainty['std']:.4f}" if cnn_uncertainty else "N/A"
    hyb_sig = f"{hybrid_uncertainty['std']:.4f}" if hybrid_uncertainty else "N/A"

    comp = Table(
        [['','CNN (Primary)','Hybrid CNN-ViT'],
         ['Prediction', cnn_result.get('subject_pred','?'), hybrid_result.get('subject_pred','?')],
         ['Confidence', f"{cnn_result.get('subject_conf',0):.1%}", f"{hybrid_result.get('subject_conf',0):.1%}"],
         ['P(ASD)', f"{cnn_result.get('weighted_prob_asd',0):.4f}", f"{hybrid_result.get('weighted_prob_asd',0):.4f}"],
         ['ASD votes', str(cnn_result.get('pred_asd_votes',0)), str(hybrid_result.get('pred_asd_votes',0))],
         ['Valid slices', str(cnn_result.get('total_valid',0)), str(hybrid_result.get('total_valid',0))],
         ['MC-Dropout σ', cnn_sig, hyb_sig],
         ['Ensemble weight', f"{ensemble_result.get('weight_cnn',0.5):.3f}", f"{ensemble_result.get('weight_vit',0.5):.3f}"],
         ['Validation AUC','0.994','0.943'],
         ['Architecture','5-layer CNN','CNN backbone + 4-layer Transformer'],
        ],
        colWidths=[4*cm, 6.5*cm, 6.5*cm]
    )
    comp.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),MID),('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('BACKGROUND',(0,1),(0,-1),LIGHT_BG),('FONTNAME',(0,1),(0,-1),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,-1),8.5),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,LIGHT_BG]),
        ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#dee2e6')),
        ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),
        ('ALIGN',(1,0),(-1,-1),'CENTER'),
    ]))
    story.append(comp)
    story.append(Spacer(1, 10))

    # Ensemble fusion methodology
    story.append(Paragraph("Ensemble Fusion Methodology", S['h2']))
    story.append(Paragraph(
        "Uncertainty-gated weighting: each model's contribution is inversely proportional "
        "to its MC-Dropout standard deviation (σ), then scaled by its baseline AUC. "
        "A model that is highly uncertain on this specific scan (high σ) receives a lower "
        "weight regardless of its average performance. This produces a dynamic ensemble that "
        "adapts to per-scan model confidence rather than using fixed weights.",
        S['body']))
    story.append(Paragraph(
        f"Formula: w_cnn = (1 − σ_cnn) × AUC_cnn;  w_vit = (1 − σ_vit) × AUC_vit; "
        f"P_ensemble = (w_cnn × P_cnn + w_vit × P_vit) / (w_cnn + w_vit). "
        f"Applied values: σ_cnn={cnn_sig}, σ_vit={hyb_sig}, "
        f"w_cnn={ensemble_result.get('weight_cnn',0.5):.3f}, "
        f"w_vit={ensemble_result.get('weight_vit',0.5):.3f}.",
        S['body']))
    story.append(Spacer(1, 6))

    _subject_block(story, S, subject_info)

    # XAI agreement section
    if xai_agreement:
        story.append(Paragraph("XAI Spatial Agreement — CNN GradCAM vs ViT Attention Rollout", S['h2']))
        agr_color = {'high': GREEN_CLR, 'moderate': AMBER, 'low': RED_CLR}.get(
            xai_agreement.get('agreement','low'), RED_CLR)
        story.append(Paragraph(
            f"Spatial agreement: {xai_agreement.get('agreement','N/A').upper()}  |  "
            f"Jaccard IoU (top-25% regions): {xai_agreement.get('iou',0):.3f}  |  "
            f"Pearson r: {xai_agreement.get('pearson_r',0):.3f}",
            ParagraphStyle('agr', fontSize=9, textColor=agr_color,
                           fontName='Helvetica-Bold', leading=13)
        ))
        story.append(Paragraph(xai_agreement.get('interpretation',''), S['body']))
        story.append(Paragraph(
            "Method note: CNN GradCAM (Selvaraju et al., 2017) computes gradient-weighted "
            "class activation maps on conv5, highlighting local features. "
            "ViT Attention Rollout extracts CLS→token self-attention from the last "
            "transformer encoder layer, averaged across 8 heads — highlighting global "
            "spatial relationships across brain-region tokens.",
            S['body']))
        story.append(Spacer(1, 6))

    # Dual XAI figures
    if dual_xai_figures:
        story.append(Paragraph("XAI Comparison Figures", S['h2']))
        for fig in dual_xai_figures:
            story.append(RLImage(io.BytesIO(_fig_to_bytes(fig)), width=16*cm, height=3.8*cm))
            story.append(Spacer(1, 4))

    _narrative_block(story, S, narrative, "Ensemble Analysis (Claude Haiku)")
    _disclaimer(story, S, model_note='CNN + Hybrid CNN-ViT Ensemble | Uncertainty-gated fusion')
    doc.build(story)
    return buf.getvalue()
