"""
lime_explain.py
---------------
LIME (Local Interpretable Model-Agnostic Explanations) for ASD CNN models.

LIME identifies which regions (superpixels) of an MRI slice most influence
the model's ASD/TC classification decision, providing a pixel-level
interpretability map without modifying the model architecture.

Background:
    LIME works by generating perturbed versions of an input image (randomly
    masking superpixels), running each through the model, and fitting a
    simpler linear model to learn which regions drive the prediction.
    The output is a binary mask highlighting positively/negatively
    contributing superpixels.

    Clinical relevance: identifies whether the model attends to anatomically
    meaningful brain regions (e.g., prefrontal cortex, amygdala, cerebellum)
    that are known to be structurally different in ASD.

Usage:
    python lime_explain.py --weights SXAI_weights.pth \
                           --image   data/png/32018.nii/32018_148.png \
                           --output  results/lime_explanation.png

Note:
    This notebook was developed post-submission. The LIME explainer ran
    successfully and generated an explanation object, but the final
    visualization encountered an error before producing the annotated image.
    The core LIME–PyTorch integration below is complete and functional.

Requirements:
    pip install lime scikit-image
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from skimage.segmentation import mark_boundaries

from models import ASD_CNN


# ── Image preprocessing ────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def preprocess_for_model(image: np.ndarray) -> torch.Tensor:
    """
    Preprocess a NumPy RGB image (H, W, 3) for model input.
    Returns a (1, 3, 224, 224) float tensor.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transform(image).unsqueeze(0)


def predict_fn(images: np.ndarray, model: torch.nn.Module,
               device: torch.device) -> np.ndarray:
    """
    Prediction function for LIME. Takes a batch of NumPy images (N, H, W, 3)
    and returns probability arrays (N, 2) for [TC, ASD].
    """
    model.eval()
    probs_all = []
    with torch.no_grad():
        for img in images:
            inp = preprocess_for_model(img).to(device)
            logits = model(inp)
            probs  = F.softmax(logits, dim=1).cpu().numpy()[0]
            probs_all.append(probs)
    return np.array(probs_all)


# ── LIME explanation ───────────────────────────────────────────────

def explain_with_lime(image_path: str, model: torch.nn.Module,
                      device: torch.device,
                      num_samples: int = 1000,
                      top_labels: int = 1,
                      save_path: str = None):
    """
    Generate a LIME explanation for a single MRI slice.

    Args:
        image_path:  Path to a PNG MRI slice.
        model:       Trained ASD_CNN or ASD_SkipCNN model.
        device:      torch.device.
        num_samples: Number of perturbed images for LIME (higher = more accurate).
        top_labels:  Number of top labels to explain.
        save_path:   If set, save the explanation figure to this path.

    Returns:
        explanation: LIME ImageExplanation object.
    """
    try:
        from lime import lime_image
    except ImportError:
        raise ImportError("Install LIME: pip install lime")

    # Load and convert image to RGB NumPy array
    img_pil = Image.open(image_path).convert('RGB')
    img_np  = np.array(img_pil)
    print(f"Image: {image_path} | Shape: {img_np.shape} | dtype: {img_np.dtype}")

    # Create LIME explainer
    explainer = lime_image.LimeImageExplainer()

    # Wrap predict_fn to bind model and device
    def _predict(images):
        return predict_fn(images, model, device)

    print(f"Running LIME with {num_samples} samples…")
    explanation = explainer.explain_instance(
        img_np,
        classifier_fn=_predict,
        top_labels=top_labels,
        hide_color=0,
        num_samples=num_samples,
    )

    # Visualize
    top_label = explanation.top_labels[0]
    label_name = 'ASD' if top_label == 1 else 'TC'
    print(f"Top predicted label: {top_label} ({label_name})")

    # Positive superpixels (contribute toward ASD prediction)
    temp_pos, mask_pos = explanation.get_image_and_mask(
        top_label, positive_only=True, hide_rest=False, num_features=5
    )
    # All superpixels (positive and negative)
    temp_all, mask_all = explanation.get_image_and_mask(
        top_label, positive_only=False, hide_rest=False, num_features=10
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(img_np)
    axes[0].set_title('Original MRI Slice')
    axes[0].axis('off')

    axes[1].imshow(mark_boundaries(temp_pos / 2 + 0.5, mask_pos))
    axes[1].set_title(f'LIME — Positive Regions\n(supporting {label_name})')
    axes[1].axis('off')

    axes[2].imshow(mark_boundaries(temp_all / 2 + 0.5, mask_all))
    axes[2].set_title('LIME — All Contributing Regions\n(green=+, red=−)')
    axes[2].axis('off')

    plt.suptitle(f'LIME Explanation | Predicted: {label_name}', fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved explanation → {save_path}")
    plt.show()
    plt.close()

    return explanation


# ── Main ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LIME explainability for ASD model")
    parser.add_argument('--weights',   required=True, help="Path to trained model .pth weights")
    parser.add_argument('--image',     required=True, help="Path to PNG MRI slice")
    parser.add_argument('--output',    default='results/lime_explanation.png')
    parser.add_argument('--samples',   type=int, default=1000)
    parser.add_argument('--model',     default='cnn', choices=['cnn', 'skip_cnn'])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from models import get_model
    model = get_model(args.model).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    print(f"Model loaded from: {args.weights}")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    explain_with_lime(args.image, model, device,
                      num_samples=args.samples,
                      save_path=args.output)
