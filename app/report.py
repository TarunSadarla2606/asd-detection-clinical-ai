# report.py
# Generate a one-page PDF clinical summary report

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
    HRFlowable, Image as RLImage
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT


def _fig_to_bytes(fig) -> bytes:
    """Convert a matplotlib figure to PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()

def generate_report(
    result: dict,
    subject_info: dict | None,
    xai_figures: list,
    narrative: str | None = None,   # ← add this
    output_path: str = None,
) -> bytes:   
    """
    Generate a PDF clinical summary report.

    Args:
        result       : output dict from run_subject_pipeline()
        subject_info : output dict from get_subject_info() or None
        xai_figures  : list of matplotlib figures (one per top slice)
        output_path  : save to file if provided, else return bytes

    Returns bytes of the PDF.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )

    styles = getSampleStyleSheet()
    story  = []

    # ── Colour palette ─────────────────────────────────────────────────────
    pred      = result['subject_pred']
    conf      = result['subject_conf']
    pred_col  = colors.HexColor('#c0392b') if pred == 'ASD' else colors.HexColor('#27ae60')
    dark      = colors.HexColor('#1a1a2e')
    mid       = colors.HexColor('#2c3e50')
    light_bg  = colors.HexColor('#f8f9fa')
    amber     = colors.HexColor('#f39c12')

    # ── Header ─────────────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        'Title', parent=styles['Title'],
        fontSize=18, textColor=dark, spaceAfter=4,
        fontName='Helvetica-Bold'
    )
    sub_style = ParagraphStyle(
        'Sub', parent=styles['Normal'],
        fontSize=9, textColor=colors.HexColor('#7f8c8d'),
        spaceAfter=2
    )
    story.append(Paragraph("ASD Detection — Clinical AI Report", title_style))
    story.append(Paragraph("Research Grade · University of North Texas · ABIDE-I Dataset", sub_style))
    story.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", sub_style))
    story.append(HRFlowable(width='100%', thickness=2, color=dark, spaceAfter=12))

    # ── Prediction result box ──────────────────────────────────────────────
    pred_label_style = ParagraphStyle(
        'PredLabel', parent=styles['Normal'],
        fontSize=28, textColor=pred_col,
        fontName='Helvetica-Bold', alignment=TA_CENTER
    )
    conf_style = ParagraphStyle(
        'Conf', parent=styles['Normal'],
        fontSize=14, textColor=mid,
        alignment=TA_CENTER
    )

    pred_table = Table(
        [[Paragraph(f"{pred}", pred_label_style)],
         [Paragraph(f"Confidence: {conf:.1%}", conf_style)],
         [Paragraph(
             f"Consensus across {result['total_valid']} valid slices "
             f"(ASD votes: {result['pred_asd_votes']} | TC votes: {result['pred_tc_votes']})",
             ParagraphStyle('small', parent=styles['Normal'],
                            fontSize=8, textColor=mid, alignment=TA_CENTER)
         )]],
        colWidths=['100%']
    )
    pred_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), light_bg),
        ('BOX',        (0,0), (-1,-1), 2, pred_col),
        ('ROUNDEDCORNERS', [8]),
        ('TOPPADDING',    (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(pred_table)
    story.append(Spacer(1, 12))

    # ── Subject information ────────────────────────────────────────────────
    story.append(Paragraph("Subject Information", ParagraphStyle(
        'SH', parent=styles['Heading2'], fontSize=12,
        textColor=dark, fontName='Helvetica-Bold'
    )))

    if subject_info:
        si = subject_info
        site_note = ''
        if si['site_perf']:
            sp = si['site_perf']
            site_note = (f"  ·  Site validation: Sens={sp['Sensitivity']:.3f} "
                         f"Spec={sp['Specificity']:.3f} AUC={sp['AUC']:.3f} "
                         f"(n={sp['N_subjects']})")

        info_data = [
            ['Field', 'Value'],
            ['Subject ID (anonymised)', str(si['anon_id'])],
            ['ABIDE SubID',             str(si['sub_id'] or 'N/A')],
            ['True Diagnosis (ABIDE)',  si['true_label']],
            ['Sex',                     si['sex']],
            ['Age at Scan',             f"{si['age']} years" if si['age'] else 'N/A'],
            ['Acquisition Site',        f"{si['site']}{site_note}"],
        ]
    else:
        info_data = [
            ['Field', 'Value'],
            ['Subject ID', 'Unknown (phenotypic data not matched)'],
        ]

    info_table = Table(info_data, colWidths=[5*cm, 12*cm])
    info_table.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0), mid),
        ('TEXTCOLOR',     (0,0), (-1,0), colors.white),
        ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0,0), (-1,-1), 8),
        ('BACKGROUND',    (0,1), (-1,-1), light_bg),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [colors.white, light_bg]),
        ('GRID',          (0,0), (-1,-1), 0.5, colors.HexColor('#dee2e6')),
        ('TOPPADDING',    (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 12))

    # ── Model performance ──────────────────────────────────────────────────
    story.append(Paragraph("Model Performance (Validation Cohort)", ParagraphStyle(
        'SH', parent=styles['Heading2'], fontSize=12,
        textColor=dark, fontName='Helvetica-Bold'
    )))
    perf_data = [
        ['AUC', 'Sensitivity', 'Specificity', 'Precision', 'F1', 'Brier'],
        ['0.994', '95.6%', '97.2%', '97.1%', '96.3%', '0.027'],
    ]
    perf_table = Table(perf_data, colWidths=[2.8*cm]*6)
    perf_table.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0), mid),
        ('TEXTCOLOR',     (0,0), (-1,0), colors.white),
        ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0,0), (-1,-1), 9),
        ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
        ('BACKGROUND',    (0,1), (-1,-1), light_bg),
        ('GRID',          (0,0), (-1,-1), 0.5, colors.HexColor('#dee2e6')),
        ('TOPPADDING',    (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(perf_table)
    story.append(Spacer(1, 12))

    # ── XAI figures ────────────────────────────────────────────────────────
    if xai_figures:
        story.append(Paragraph("Explainability — Grad-CAM Analysis", ParagraphStyle(
            'SH', parent=styles['Heading2'], fontSize=12,
            textColor=dark, fontName='Helvetica-Bold'
        )))
        story.append(Paragraph(
            "Each row shows: original MRI slice · GradCAM heatmap · GradCAM overlay · LIME explanation. "
            "Red regions = high model activation. Slices selected by quality score "
            "(circularity + brain coverage + sharpness).",
            ParagraphStyle('body', parent=styles['Normal'], fontSize=8, textColor=mid)
        ))
        story.append(Spacer(1, 6))

        for i, fig in enumerate(xai_figures):   # all figures passed in
            fig_bytes = _fig_to_bytes(fig)
            img       = RLImage(io.BytesIO(fig_bytes), width=16*cm, height=4*cm)
            story.append(img)
            story.append(Spacer(1, 6))
    
    # ── LLM Narrative ──────────────────────────────────────────────────────
    if narrative:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Clinical Interpretation (AI-Generated)", ParagraphStyle(
            'SH', parent=styles['Heading2'], fontSize=12,
            textColor=dark, fontName='Helvetica-Bold'
        )))
        story.append(Paragraph(
            "Generated by Claude Haiku from structured model outputs. "
            "Research use only — not a clinical opinion.",
            ParagraphStyle('italic', parent=styles['Normal'],
                           fontSize=7, textColor=mid, fontName='Helvetica-Oblique')
        ))
        story.append(Spacer(1, 4))

        # Split into paragraphs and render each
        clean = narrative.replace('**', '').replace('*', '')   # strip markdown bold
        for para in clean.split('\n\n'):
            para = para.strip()
            if para:
                story.append(Paragraph(para, ParagraphStyle(
                    'narr', parent=styles['Normal'],
                    fontSize=8.5, textColor=colors.HexColor('#2c3e50'),
                    leading=13, spaceAfter=6
                )))

    # ── Disclaimer ─────────────────────────────────────────────────────────
    story.append(HRFlowable(width='100%', thickness=1, color=amber, spaceAfter=6))
    disclaimer_style = ParagraphStyle(
        'Disc', parent=styles['Normal'],
        fontSize=7, textColor=colors.HexColor('#7f8c8d'),
        leading=10
    )
    story.append(Paragraph(
        "⚠️  RESEARCH USE ONLY — This report was generated by a research-grade AI system "
        "trained on the ABIDE-I dataset (retrospective, multi-site, slice-level). "
        "It has NOT been validated for clinical deployment and does not constitute a medical "
        "diagnosis. It should not be used to guide clinical decisions. Any clinical deployment "
        "would require prospective validation, FDA SaMD Class II regulatory clearance, and "
        "neuroradiologist review. Model: 5-layer CNN | Data: ABIDE-I | "
        "Author: Tarun Sadarla, University of North Texas.",
        disclaimer_style
    ))

    doc.build(story)
    pdf_bytes = buf.getvalue()

    if output_path:
        with open(output_path, 'wb') as f:
            f.write(pdf_bytes)

    return pdf_bytes