"""
evaluate.py
-----------
Model evaluation: metrics, ROC curve, confusion matrix, and training curves.

Usage:
    python evaluate.py --model skip_cnn \
                       --weights best_model.pth \
                       --test_csv /kaggle/input/autism-csv/extracted_random_labels_test.csv \
                       --output_dir results/

All metrics are computed from real experimental results documented in
the project notebooks. Reference numbers from final report:

    Skip-CNN (NAdam, 1,067 subjects):
        Test Acc=0.91, AUC=0.98, Precision=0.93, Recall=0.88, F1=0.90

    CNN (RMSprop, 1,067 subjects):
        Test Acc=0.91, AUC=0.97, Precision=0.90, Recall=0.90, F1=0.90
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, ConfusionMatrixDisplay)
from tqdm import tqdm

from models import get_model
from dataset import ASDDataset
from torch.utils.data import DataLoader


def evaluate(model, loader, device, threshold: float = 0.5):
    """
    Run inference on a DataLoader and return predictions and probabilities.

    Returns:
        preds:       np.ndarray of predicted labels (0 or 1)
        targets:     np.ndarray of true labels
        probs:       np.ndarray of ASD (class 1) probabilities
    """
    model.eval()
    preds_all, targets_all, probs_all = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Evaluating', leave=False):
            images = images.to(device)
            logits = model(images)
            probs  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds  = (probs >= threshold).astype(int)
            preds_all.extend(preds)
            targets_all.extend(labels.numpy())
            probs_all.extend(probs)
    return np.array(preds_all), np.array(targets_all), np.array(probs_all)


def print_metrics(preds, targets, probs):
    """Print the full classification report."""
    cm = confusion_matrix(targets, preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print("\n" + "="*45)
    print("  CLASSIFICATION RESULTS")
    print("="*45)
    print(f"  Accuracy    : {accuracy_score(targets, preds):.4f}")
    print(f"  AUC         : {roc_auc_score(targets, probs):.4f}")
    print(f"  Precision   : {precision_score(targets, preds, zero_division=0):.4f}")
    print(f"  Recall      : {recall_score(targets, preds, zero_division=0):.4f}")
    print(f"  Specificity : {specificity:.4f}")
    print(f"  F1-Score    : {f1_score(targets, preds, zero_division=0):.4f}")
    print(f"\n  Confusion Matrix:  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print("="*45 + "\n")


def plot_roc_curve(targets, probs, save_path: str = None):
    fpr, tpr, _ = roc_curve(targets, probs)
    auc = roc_auc_score(targets, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    ax.plot([0,1], [0,1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate (Recall)')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved ROC curve → {save_path}")
    plt.show()
    plt.close()


def plot_confusion_matrix(preds, targets, save_path: str = None):
    cm = confusion_matrix(targets, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=['TC (0)', 'ASD (1)'])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap='Blues', colorbar=True)
    ax.set_title('Confusion Matrix — Test Set')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved confusion matrix → {save_path}")
    plt.show()
    plt.close()


def plot_training_curves(history: dict, save_path: str = None):
    """Plot training vs validation accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history['train_accs']) + 1)
    axes[0].plot(epochs, history['train_accs'], 'o-', label='Train', color='steelblue')
    axes[0].plot(epochs, history['val_accs'],   'o-', label='Val',   color='darkorange')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Training vs Validation Accuracy')
    axes[0].legend(); axes[0].grid(alpha=0.4)

    axes[1].plot(epochs, history['train_losses'], label='Train', color='steelblue')
    axes[1].plot(epochs, history['val_losses'],   label='Val',   color='darkorange')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].set_title('Training vs Validation Loss')
    axes[1].legend(); axes[1].grid(alpha=0.4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved training curves → {save_path}")
    plt.show()
    plt.close()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model = get_model(args.model).to(device)
    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"Loaded weights from: {args.weights}")

    # Test data
    test_ds     = ASDDataset(args.test_csv, mode='cnn',
                             path_prefix=args.path_prefix)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                             num_workers=2, pin_memory=True)
    print(f"Test set: {len(test_ds):,} slices")

    # Evaluate
    preds, targets, probs = evaluate(model, test_loader, device)
    print_metrics(preds, targets, probs)

    # Plots
    plot_roc_curve(targets, probs,
                   save_path=os.path.join(args.output_dir, 'roc_curve.png'))
    plot_confusion_matrix(preds, targets,
                   save_path=os.path.join(args.output_dir, 'confusion_matrix.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ASD detection model")
    parser.add_argument('--model',       default='skip_cnn', choices=['cnn','skip_cnn','vit'])
    parser.add_argument('--weights',     default=None)
    parser.add_argument('--test_csv',    required=True)
    parser.add_argument('--output_dir',  default='results/')
    parser.add_argument('--path_prefix', default='/kaggle/input/autism/')
    main(parser.parse_args())
