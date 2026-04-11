# xai.py
# GradCAM + LIME explanations for the top-K slices

import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from lime import lime_image
from skimage.segmentation import mark_boundaries

MEAN = [0.1290, 0.1290, 0.1290]
STD  = [0.1741, 0.1741, 0.1741]

_LIME_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

_EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])


# ── GradCAM ───────────────────────────────────────────────────────────────

def run_gradcam(arr_cropped: np.ndarray, model, pred_class: int,
                device: str = 'cpu') -> dict:
    """
    Run GradCAM on a brain-cropped uint8 grayscale array.
    Returns dict with:
        original_rgb : (224,224,3) float32 [0,1]  — for overlay
        cam_overlay  : (224,224,3) uint8           — GradCAM on image
        heatmap      : (224,224)   float32          — raw activation map
        energy       : float                        — fraction of pixels > 0.5
    """
    pil          = Image.fromarray(arr_cropped).convert('RGB').resize((224, 224))
    original_rgb = np.array(pil).astype(np.float32) / 255.0
    t            = _EVAL_TF(pil).unsqueeze(0).to(device)

    cam_engine = GradCAM(model=model, target_layers=[model.conv5])

    with torch.no_grad():
        probs = F.softmax(model(t), dim=1)[0].cpu().numpy()

    targets  = [ClassifierOutputTarget(pred_class)]
    heatmap  = cam_engine(input_tensor=t, targets=targets)[0]
    overlay  = show_cam_on_image(original_rgb, heatmap, use_rgb=True)
    energy   = float((heatmap > 0.5).mean())

    return {
        'original_rgb': original_rgb,
        'cam_overlay' : overlay,
        'heatmap'     : heatmap,
        'energy'      : energy,
        'prob_asd'    : float(probs[1]),
        'prob_tc'     : float(probs[0]),
    }


# ── LIME ──────────────────────────────────────────────────────────────────

def run_lime(arr_cropped: np.ndarray, model, pred_class: int,
             device: str = 'cpu', num_samples: int = 500) -> dict:
    """
    Run LIME on a brain-cropped uint8 grayscale array.
    num_samples=500 is fast enough for a live demo on CPU (~15s).
    Increase to 1000 for better quality if time allows.
    Returns dict with:
        img_np       : (224,224,3) uint8  — resized input
        positive_vis : (224,224,3) uint8  — top supporting regions only
        full_vis     : (224,224,3) uint8  — green=pro, red=counter
    """
    pil    = Image.fromarray(arr_cropped).convert('RGB').resize((224, 224))
    img_np = np.array(pil)

    def _predict(images_np):
        model.eval()
        batch = torch.stack([
            _LIME_TF(Image.fromarray(img.astype(np.uint8)))
            for img in images_np
        ]).to(device)
        with torch.no_grad():
            return F.softmax(model(batch), dim=1).cpu().numpy()

    explainer   = lime_image.LimeImageExplainer(random_state=42)
    explanation = explainer.explain_instance(
        img_np,
        classifier_fn=_predict,
        top_labels=2,
        hide_color=0,
        num_samples=num_samples,
        random_seed=42
    )

    top_label = pred_class
    temp_pos, mask_pos = explanation.get_image_and_mask(
        top_label, positive_only=True, num_features=5, hide_rest=True
    )
    temp_full, mask_full = explanation.get_image_and_mask(
        top_label, positive_only=False, num_features=10, hide_rest=False
    )

    positive_vis = mark_boundaries(temp_pos.astype(np.uint8), mask_pos)
    full_vis     = mark_boundaries(temp_full.astype(np.uint8), mask_full)

    return {
        'img_np'      : img_np,
        'positive_vis': (positive_vis * 255).astype(np.uint8),
        'full_vis'    : (full_vis * 255).astype(np.uint8),
    }


# ── MC-Dropout uncertainty ─────────────────────────────────────────────────

def mc_dropout_uncertainty(arr_cropped: np.ndarray, model,
                            n_passes: int = 30,
                            device: str = 'cpu') -> dict:
    """
    Estimate prediction uncertainty via MC-Dropout.
    Returns mean P(ASD), std, and uncertainty label.
    """
    pil = Image.fromarray(arr_cropped).convert('RGB')
    t   = _EVAL_TF(pil).unsqueeze(0).to(device)

    # Enable dropout at inference time
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout2d):
            m.train()

    probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            p = F.softmax(model(t), dim=1)[0, 1].item()
            probs.append(p)

    model.eval()   # reset

    mean_p = float(np.mean(probs))
    std_p  = float(np.std(probs))

    if std_p < 0.02:
        uncertainty = 'Low — high confidence prediction'
    elif std_p < 0.08:
        uncertainty = 'Moderate — review recommended'
    else:
        uncertainty = 'High — clinical review strongly recommended'

    return {
        'mean_prob_asd': round(mean_p, 4),
        'std'          : round(std_p, 4),
        'uncertainty'  : uncertainty,
        'n_passes'     : n_passes,
    }


# ── Combined figure for one slice ─────────────────────────────────────────

def make_explanation_figure(arr_cropped: np.ndarray, model,
                             pred_class: int, slice_idx: int,
                             device: str = 'cpu',
                             run_lime_flag: bool = True) -> plt.Figure:
    """
    Produces a 1×4 figure:
      Col 1: Original MRI (cropped)
      Col 2: GradCAM heatmap
      Col 3: GradCAM overlay
      Col 4: LIME full explanation  (skipped if run_lime_flag=False)
    Returns a matplotlib Figure (caller must close it).
    """
    gc   = run_gradcam(arr_cropped, model, pred_class, device)
    cols = 4 if run_lime_flag else 3

    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5))
    fig.patch.set_facecolor('#0e1117')

    label = 'ASD' if pred_class == 1 else 'TC'
    conf  = gc['prob_asd'] if pred_class == 1 else gc['prob_tc']

    axes[0].imshow(gc['original_rgb'], cmap='gray')
    axes[0].set_title(f'MRI Slice z={slice_idx}', color='white', fontsize=11)
    axes[0].axis('off')

    im = axes[1].imshow(gc['heatmap'], cmap='jet', vmin=0, vmax=1)
    axes[1].set_title(f'GradCAM Heatmap\n(energy={gc["energy"]:.3f})', color='white', fontsize=11)
    axes[1].axis('off')
    plt.colorbar(mpl_cm.ScalarMappable(cmap='jet'), ax=axes[1], fraction=0.046)

    axes[2].imshow(gc['cam_overlay'])
    axes[2].set_title(f'GradCAM Overlay\nPred: {label} ({conf:.1%})', color='white', fontsize=11)
    axes[2].axis('off')

    if run_lime_flag:
        lm = run_lime(arr_cropped, model, pred_class, device)
        axes[3].imshow(lm['full_vis'])
        axes[3].set_title('LIME\n(green=pro, red=counter)', color='white', fontsize=11)
        axes[3].axis('off')

    for ax in axes:
        ax.set_facecolor('#0e1117')

    plt.tight_layout()
    return fig