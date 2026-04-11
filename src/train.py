"""
train.py
--------
Training loop for ASD detection models.

Supports all three optimizers (Adam, NAdam, RMSprop) and both
CNN architectures. Includes early stopping, training curve logging,
and model weight saving.

Usage:
    python train.py --model skip_cnn --optimizer nadam \
                    --train_csv /kaggle/input/autism-csv/extracted_random_labels_train.csv \
                    --val_csv   /kaggle/input/autism-csv/extracted_random_labels_validation.csv \
                    --epochs 50 --batch_size 64 --save_path model_weights.pth

Canonical hyperparameters (from report):
    Learning rate:  0.001
    Adam/NAdam β₁:  0.9,   β₂: 0.999,  ε: 1e-7 (Adam) / 1e-8 (NAdam)
    RMSprop:        lr=0.001, ε=1e-7
    Epochs:         50 (CNN), 100 (ViT Keras)
    Batch size:     64
    Loss:           CrossEntropyLoss
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from models import get_model
from dataset import build_loaders


def build_optimizer(model: nn.Module, name: str,
                    lr: float = 0.001) -> optim.Optimizer:
    """Return the optimizer matching the canonical hyperparameters."""
    name = name.lower()
    if name == 'adam':
        return optim.Adam(model.parameters(), lr=lr,
                          betas=(0.9, 0.999), eps=1e-7)
    elif name == 'nadam':
        return optim.NAdam(model.parameters(), lr=lr,
                           betas=(0.9, 0.999), eps=1e-8)
    elif name == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=lr, eps=1e-7)
    else:
        raise ValueError(f"Unknown optimizer '{name}'. Use: adam | nadam | rmsprop")


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, preds_all, targets_all = 0.0, [], []
    for images, labels in tqdm(loader, desc='Training', leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds_all.extend(outputs.argmax(dim=1).cpu().tolist())
        targets_all.extend(labels.cpu().tolist())
    acc = accuracy_score(targets_all, preds_all)
    return total_loss / len(loader), acc


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, preds_all, targets_all = 0.0, [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validation', leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds_all.extend(outputs.argmax(dim=1).cpu().tolist())
            targets_all.extend(labels.cpu().tolist())
    acc = accuracy_score(targets_all, preds_all)
    return total_loss / len(loader), acc


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_loader, val_loader, _ = build_loaders(
        args.train_csv, args.val_csv, args.val_csv,   # test reuse val for simplicity
        mode='cnn', batch_size=args.batch_size,
        path_prefix=args.path_prefix,
    )
    print(f"Train: {len(train_loader.dataset):,} | Val: {len(val_loader.dataset):,}")

    # Model + optimizer + loss
    model     = get_model(args.model).to(device)
    optimizer = build_optimizer(model, args.optimizer, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # History
    train_losses, val_losses       = [], []
    train_accs,   val_accs         = [], []
    best_val_acc, patience_counter = 0.0, 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = eval_epoch(model, val_loader, criterion, device)

        train_losses.append(tr_loss); train_accs.append(tr_acc)
        val_losses.append(va_loss);   val_accs.append(va_acc)

        print(f"Epoch [{epoch:02d}/{args.epochs}] "
              f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f} | "
              f"Val Loss: {va_loss:.4f}  Acc: {va_acc:.4f}")

        # Early stopping
        if va_acc > best_val_acc + args.min_delta:
            best_val_acc = va_acc
            patience_counter = 0
            if args.save_path:
                torch.save(model.state_dict(), args.save_path)
                print(f"  ✓ Saved best model → {args.save_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    return model, {
        'train_losses': train_losses, 'val_losses': val_losses,
        'train_accs': train_accs, 'val_accs': val_accs,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASD detection model")
    parser.add_argument('--model',      default='skip_cnn', choices=['cnn','skip_cnn','vit'])
    parser.add_argument('--optimizer',  default='nadam',    choices=['adam','nadam','rmsprop'])
    parser.add_argument('--train_csv',  required=True)
    parser.add_argument('--val_csv',    required=True)
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch_size', type=int,   default=64)
    parser.add_argument('--lr',         type=float, default=0.001)
    parser.add_argument('--patience',   type=int,   default=5)
    parser.add_argument('--min_delta',  type=float, default=0.001)
    parser.add_argument('--save_path',  default='best_model.pth')
    parser.add_argument('--path_prefix', default='/kaggle/input/autism/')
    args = parser.parse_args()
    train(args)
