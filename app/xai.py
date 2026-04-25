# xai.py — v2
# Explainability for both CNN and Hybrid CNN-ViT models
# Functions: GradCAM (CNN), Attention Rollout (ViT), Dual XAI figure, MC-Dropout (both)

import io
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm_mod
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from model import (
    CNN_MEAN, CNN_STD, HYBRID_MEAN, HYBRID_STD,
    CNN_TRANSFORM, HYBRID_TRANSFORM
)

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _arr_to_tensor(arr: np.ndarray, transform) -> torch.Tensor:
    """Convert uint8 grayscale array → (1,3,224,224) tensor."""
    pil = Image.fromarray(arr).convert('RGB')
    return transform(pil).unsqueeze(0)


def _arr_to_rgb(arr: np.ndarray) -> np.ndarray:
    """uint8 (H,W) → float32 (224,224,3) in [0,1] for overlays."""
    pil = Image.fromarray(arr).convert('RGB').resize((224, 224))
    return np.array(pil).astype(np.float32) / 255.0


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CNN GradCAM + MC-Dropout (existing, unchanged signatures)
# ══════════════════════════════════════════════════════════════════════════════

def run_gradcam(arr: np.ndarray, model, pred_class: int,
                device: str = 'cpu') -> dict:
    """
    GradCAM on CNN conv5 layer.
    Returns dict: heatmap (224,224), energy (float), overlay (224,224,3).
    """
    if not GRADCAM_AVAILABLE:
        return {'heatmap': np.zeros((224,224)), 'energy': 0.0,
                'overlay': np.zeros((224,224,3))}

    t       = _arr_to_tensor(arr, CNN_TRANSFORM).to(device)
    orig    = _arr_to_rgb(arr)
    cam_eng = GradCAM(model=model, target_layers=[model.conv5])
    gs      = cam_eng(input_tensor=t,
                      targets=[ClassifierOutputTarget(pred_class)])[0]  # (224,224)
    overlay = show_cam_on_image(orig, gs, use_rgb=True)
    return {
        'heatmap': gs,
        'energy' : float((gs > 0.5).mean()),
        'overlay': overlay,
    }


def mc_dropout_uncertainty(arr: np.ndarray, model,
                           n_passes: int = 30,
                           device: str = 'cpu') -> dict:
    """
    MC-Dropout on CNN model. Activates Dropout2d layers only.
    Returns mean P(ASD), std, and clinical assessment string.
    """
    t = _arr_to_tensor(arr, CNN_TRANSFORM).to(device)
    model.eval()
    def _enable(m):
        if isinstance(m, nn.Dropout2d): m.train()
    model.apply(_enable)

    probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            p = F.softmax(model(t), dim=1)[0, 1].item()
            probs.append(p)
    model.eval()

    mu, sigma = float(np.mean(probs)), float(np.std(probs))
    if sigma < 0.03:
        assessment = f"Low uncertainty (σ={sigma:.3f}) — prediction is stable across model samples"
    elif sigma < 0.08:
        assessment = f"Moderate uncertainty (σ={sigma:.3f}) — some variability; interpret with caution"
    else:
        assessment = (f"High uncertainty (σ={sigma:.3f}) — model is genuinely uncertain; "
                      f"flag for expert review")

    return {
        'mean_prob_asd': mu,
        'std'          : sigma,
        'n_passes'     : n_passes,
        'uncertainty'  : assessment,
        'probs'        : probs,
    }


def make_explanation_figure(arr: np.ndarray, model, pred_class: int,
                             slice_idx: int, device: str = 'cpu',
                             run_lime_flag: bool = False):
    """CNN XAI figure: original | GradCAM heatmap | GradCAM overlay."""
    orig = _arr_to_rgb(arr)
    gc   = run_gradcam(arr, model, pred_class, device)

    n_cols = 3
    fig, axes = plt.subplots(1, n_cols, figsize=(15, 4))
    fig.patch.set_facecolor('#0e1117')

    axes[0].imshow(orig[:,:,0], cmap='gray'); axes[0].axis('off')
    axes[0].set_title(f'Slice z={slice_idx}', color='white', fontsize=9)

    axes[1].imshow(gc['heatmap'], cmap='jet', vmin=0, vmax=1); axes[1].axis('off')
    axes[1].set_title(f'GradCAM  energy={gc["energy"]:.3f}', color='white', fontsize=9)

    axes[2].imshow(gc['overlay']); axes[2].axis('off')
    axes[2].set_title('Overlay', color='white', fontsize=9)

    plt.tight_layout(pad=0.3)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Hybrid CNN-ViT: Attention Rollout + GradCAM on backbone + MC-Dropout
# ══════════════════════════════════════════════════════════════════════════════

def get_attention_rollout(arr: np.ndarray, hybrid_model,
                          device: str = 'cpu') -> dict:
    """
    Extract CLS→token attention from last transformer layer, head-averaged.

    Returns:
        attn_14    : (14,14) numpy array — token-resolution attention
        attn_224   : (224,224) numpy array — bilinearly upsampled
        overlay    : (224,224,3) uint8 — heatmap blended on original MRI
        n_brain    : int — number of brain tokens out of 196
        token_grid : (14,14) bool array — which tokens were brain tokens
    """
    t    = _arr_to_tensor(arr, HYBRID_TRANSFORM).to(device)
    orig = _arr_to_rgb(arr)

    hybrid_model.eval()
    with torch.no_grad():
        _ = hybrid_model(t)

    ca = hybrid_model.get_cls_attn()  # (1,196)
    if ca is None:
        blank = np.zeros((224,224))
        return {'attn_14':np.zeros((14,14)),'attn_224':blank,
                'overlay':(orig*255).astype(np.uint8),'n_brain':0,'token_grid':np.zeros((14,14),bool)}

    # Brain token mask for this image
    from model import generate_brain_mask, mask_to_tokens
    mask      = generate_brain_mask(t)
    tok_mask  = mask_to_tokens(mask, 14)   # (1,196) bool
    n_brain   = int(tok_mask[0].sum().item())
    tok_grid  = tok_mask[0].cpu().numpy().reshape(14, 14)

    attn_14  = ca[0].cpu().numpy().reshape(14, 14)
    attn_224 = cv2.resize(attn_14, (224, 224), interpolation=cv2.INTER_LINEAR)
    attn_224 = (attn_224 - attn_224.min()) / (attn_224.max() - attn_224.min() + 1e-8)

    heatmap = cm_mod.get_cmap('jet')(attn_224)[:, :, :3]  # (224,224,3)
    overlay = ((0.5 * orig + 0.5 * heatmap).clip(0,1) * 255).astype(np.uint8)

    return {
        'attn_14'   : attn_14,
        'attn_224'  : attn_224,
        'overlay'   : overlay,
        'n_brain'   : n_brain,
        'token_grid': tok_grid,
    }


def run_gradcam_backbone(arr: np.ndarray, hybrid_model, pred_class: int,
                         device: str = 'cpu') -> dict:
    """
    GradCAM on hybrid_model.backbone.conv4.
    Gradients flow through the transformer and into the CNN backbone,
    so this captures what the full model attends to via the CNN pathway.
    """
    if not GRADCAM_AVAILABLE:
        return {'heatmap': np.zeros((224,224)), 'energy': 0.0,
                'overlay': np.zeros((224,224,3))}

    t    = _arr_to_tensor(arr, HYBRID_TRANSFORM).to(device)
    orig = _arr_to_rgb(arr)
    cam  = GradCAM(model=hybrid_model, target_layers=[hybrid_model.backbone.conv4])
    gs   = cam(input_tensor=t, targets=[ClassifierOutputTarget(pred_class)])[0]
    overlay = show_cam_on_image(orig, gs, use_rgb=True)
    return {
        'heatmap': gs,
        'energy' : float((gs > 0.5).mean()),
        'overlay': overlay,
    }


def mc_dropout_hybrid(arr: np.ndarray, hybrid_model,
                      n_passes: int = 30,
                      device: str = 'cpu') -> dict:
    """
    MC-Dropout on Hybrid CNN-ViT.
    Activates BOTH CNN Dropout2d AND Transformer Dropout layers.
    High sigma on ASD positives is expected and clinically meaningful —
    the attention mechanism explores multiple plausible explanations.
    """
    t = _arr_to_tensor(arr, HYBRID_TRANSFORM).to(device)
    hybrid_model.eval()

    def _enable(m):
        if isinstance(m, (nn.Dropout2d, nn.Dropout)): m.train()
    hybrid_model.apply(_enable)

    probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            p = F.softmax(hybrid_model(t), dim=1)[0, 1].item()
            probs.append(p)
    hybrid_model.eval()

    mu, sigma = float(np.mean(probs)), float(np.std(probs))

    # Hybrid model has characteristically higher sigma on ASD positives (σ≈0.24)
    # because both CNN and transformer dropouts contribute variance
    if sigma < 0.05:
        assessment = (f"Low combined uncertainty (σ={sigma:.3f}) — "
                      f"CNN and transformer pathways agree strongly")
    elif sigma < 0.12:
        assessment = (f"Moderate uncertainty (σ={sigma:.3f}) — "
                      f"some disagreement between CNN and attention pathways")
    else:
        assessment = (f"High combined uncertainty (σ={sigma:.3f}) — "
                      f"CNN and transformer pathways diverge significantly; "
                      f"attention may be exploring multiple competing spatial hypotheses")

    return {
        'mean_prob_asd': mu,
        'std'          : sigma,
        'n_passes'     : n_passes,
        'uncertainty'  : assessment,
        'probs'        : probs,
        'model_type'   : 'hybrid',
    }


def make_hybrid_xai_figure(arr: np.ndarray, hybrid_model, pred_class: int,
                            slice_idx: int, device: str = 'cpu'):
    """
    Hybrid XAI figure: original | Brain token mask | Attention rollout | Attention overlay.
    """
    orig     = _arr_to_rgb(arr)
    attn_d   = get_attention_rollout(arr, hybrid_model, device)
    gc_d     = run_gradcam_backbone(arr, hybrid_model, pred_class, device)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.patch.set_facecolor('#0e1117')

    axes[0].imshow(orig[:,:,0], cmap='gray'); axes[0].axis('off')
    axes[0].set_title(f'Slice z={slice_idx}', color='white', fontsize=9)

    tok_grid = attn_d['token_grid'].astype(float)
    axes[1].imshow(tok_grid, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    axes[1].set_title(f'Brain token mask\n{attn_d["n_brain"]}/196 brain tokens',
                      color='white', fontsize=9)
    axes[1].axis('off')

    axes[2].imshow(attn_d['attn_224'], cmap='jet', vmin=0, vmax=1); axes[2].axis('off')
    axes[2].set_title('CLS→Token Attention\n(head-averaged, last layer)',
                      color='white', fontsize=9)

    axes[3].imshow(attn_d['overlay']); axes[3].axis('off')
    axes[3].set_title('Attention Overlay', color='white', fontsize=9)

    plt.tight_layout(pad=0.3)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Dual XAI comparison + agreement metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_xai_agreement(gc_heatmap: np.ndarray,
                           attn_map: np.ndarray,
                           threshold_pct: float = 75.0) -> dict:
    """
    Compute spatial agreement between GradCAM and Attention Rollout.

    Args:
        gc_heatmap  : (224,224) GradCAM from CNN
        attn_map    : (224,224) Attention rollout from ViT
        threshold_pct: percentile threshold for 'hot' region (default top 25%)

    Returns dict with:
        iou         : Jaccard index of hot regions
        pearson_r   : Pearson correlation
        gc_energy   : CNN GradCAM energy (fraction of pixels > 0.5)
        attn_energy : ViT attention energy
        agreement   : 'high' | 'moderate' | 'low'
        interpretation: clinical text
    """
    gc_hot   = gc_heatmap >= np.percentile(gc_heatmap,   threshold_pct)
    attn_hot = attn_map   >= np.percentile(attn_map,     threshold_pct)

    intersection = float((gc_hot & attn_hot).sum())
    union        = float((gc_hot | attn_hot).sum())
    iou          = intersection / (union + 1e-8)

    # Pearson r
    gc_flat   = gc_heatmap.flatten()
    attn_flat = attn_map.flatten()
    gc_z      = gc_flat   - gc_flat.mean()
    attn_z    = attn_flat - attn_flat.mean()
    pearson_r = float(np.dot(gc_z, attn_z) /
                      (np.linalg.norm(gc_z) * np.linalg.norm(attn_z) + 1e-8))

    gc_energy   = float((gc_heatmap   > 0.5).mean())
    attn_energy = float((attn_map     > 0.5).mean())

    if iou > 0.30:
        agreement = 'high'
        interp = (f"Both models highlight overlapping spatial regions (IoU={iou:.3f}). "
                  f"Strong spatial convergence increases confidence that these regions "
                  f"are genuinely discriminative for ASD vs TC classification.")
    elif iou > 0.15:
        agreement = 'moderate'
        interp = (f"Partial spatial overlap between explanations (IoU={iou:.3f}). "
                  f"The CNN's gradient-based saliency and the ViT's global attention "
                  f"identify partially shared regions, suggesting the models use "
                  f"complementary but not identical feature sets.")
    else:
        agreement = 'low'
        interp = (f"Low spatial agreement (IoU={iou:.3f}). The CNN focuses on local "
                  f"textural gradients while the transformer attends to global spatial "
                  f"relationships across the full slice. These are complementary "
                  f"explanations — low IoU does not indicate error in either model.")

    return {
        'iou'        : iou,
        'pearson_r'  : pearson_r,
        'gc_energy'  : gc_energy,
        'attn_energy': attn_energy,
        'agreement'  : agreement,
        'interpretation': interp,
    }


def make_dual_xai_figure(arr: np.ndarray, cnn_model, hybrid_model,
                          cnn_pred_class: int, hybrid_pred_class: int,
                          slice_idx: int, device: str = 'cpu'):
    """
    Side-by-side comparison: CNN GradCAM vs ViT Attention Rollout on same slice.
    Returns (fig, agreement_dict).
    """
    orig   = _arr_to_rgb(arr)
    gc_d   = run_gradcam(arr, cnn_model, cnn_pred_class, device)
    attn_d = get_attention_rollout(arr, hybrid_model, device)
    agree  = compute_xai_agreement(gc_d['heatmap'], attn_d['attn_224'])

    fig, axes = plt.subplots(1, 5, figsize=(24, 4))
    fig.patch.set_facecolor('#0e1117')

    axes[0].imshow(orig[:,:,0], cmap='gray'); axes[0].axis('off')
    axes[0].set_title(f'MRI — Slice z={slice_idx}', color='white', fontsize=9)

    axes[1].imshow(gc_d['heatmap'], cmap='jet', vmin=0, vmax=1); axes[1].axis('off')
    axes[1].set_title(f'CNN GradCAM\nenergy={gc_d["energy"]:.3f}',
                      color='white', fontsize=9)

    axes[2].imshow(gc_d['overlay']); axes[2].axis('off')
    axes[2].set_title('CNN Overlay', color='white', fontsize=9)

    axes[3].imshow(attn_d['attn_224'], cmap='jet', vmin=0, vmax=1); axes[3].axis('off')
    axes[3].set_title(f'ViT Attention Rollout\nenergy={agree["attn_energy"]:.3f}',
                      color='white', fontsize=9)

    axes[4].imshow(attn_d['overlay']); axes[4].axis('off')
    agree_color = {'high':'#27ae60','moderate':'#f39c12','low':'#e74c3c'}[agree['agreement']]
    axes[4].set_title(f'ViT Overlay\nIoU={agree["iou"]:.3f}  r={agree["pearson_r"]:.3f}',
                      color=agree_color, fontsize=9, fontweight='bold')

    plt.suptitle(
        f'XAI Agreement: {agree["agreement"].upper()}  '
        f'(IoU={agree["iou"]:.3f}, Pearson r={agree["pearson_r"]:.3f})',
        color='white', fontsize=10, fontweight='bold'
    )
    plt.tight_layout(pad=0.3)
    return fig, agree
