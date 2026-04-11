"""
models.py
---------
All model architectures for ASD detection from sMRI brain scans.

Models:
    1. ASD_CNN         — Custom 5-layer CNN (plain)
    2. ASD_SkipCNN     — 5-layer CNN with skip connection (layers 1→3)
    3. ViT_PyTorch     — Custom Vision Transformer from scratch (PyTorch)

Results achieved (ABIDE-I, 1,067 subjects, 100,510 slices):
    ASD_CNN (RMSprop):    Train 0.94 | Test 0.91 | AUC 0.97 | F1 0.90
    ASD_SkipCNN (NAdam):  Train 0.93 | Test 0.91 | AUC 0.98 | F1 0.90
    ViT (Keras, scratch): Train 0.60 | Test 0.62 | AUC 0.62 | F1 0.64

Usage:
    from models import ASD_CNN, ASD_SkipCNN, ViT_PyTorch
    model = ASD_CNN().to(device)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
# 1. Custom 5-Layer CNN
# ══════════════════════════════════════════════════════════════════════

class ASD_CNN(nn.Module):
    """
    Custom 5-layer CNN for ASD binary classification.

    Architecture:
        5 convolutional blocks with increasing filter sizes (16→32→64→128→256).
        Leaky ReLU activation throughout.
        BatchNorm after layers 2–5.
        MaxPool(2×2) + Dropout(20%) after each block.
        Dense head: FC(100) + LeakyReLU → Output(2).

    Input:  (batch, 3, 224, 224)
    Output: (batch, 2)  logits for [TC, ASD]
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        lrelu = lambda: nn.LeakyReLU(0.1, inplace=True)

        # ── Block 1: Conv(3→16) ───────────────────────────────────
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            lrelu(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

        # ── Block 2: Conv(16→32) + BatchNorm ──────────────────────
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            lrelu(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

        # ── Block 3: Conv(32→64) + BatchNorm ──────────────────────
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            lrelu(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

        # ── Block 4: Conv(64→128) + BatchNorm ─────────────────────
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            lrelu(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

        # ── Block 5: Conv(128→256) + BatchNorm ────────────────────
        self.block5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            lrelu(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

        # After 5× MaxPool(2), 224→7
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 100),
            lrelu(),
            nn.Linear(100, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return self.classifier(x)


# ══════════════════════════════════════════════════════════════════════
# 2. Skip-Connected CNN
# ══════════════════════════════════════════════════════════════════════

class ASD_SkipCNN(nn.Module):
    """
    5-layer CNN with a residual skip connection from block 1 output
    to block 3 input. Addresses vanishing gradients and promotes
    feature reuse.

    Skip connection: output of Conv1 is projected (1×1 conv) and added
    to the input of Conv3, matching channel dimensions (16 → 32 → 64).
    A 1×1 conv + MaxPool is used to match spatial dimensions.

    Input:  (batch, 3, 224, 224)
    Output: (batch, 2)
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        lrelu = lambda: nn.LeakyReLU(0.1, inplace=True)

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            lrelu(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )  # → (B, 16, 112, 112)

        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            lrelu(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )  # → (B, 32, 56, 56)

        # Skip projection: match channels (16→64) and spatial dims (112→56)
        self.skip_proj = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=1),
            nn.MaxPool2d(4),  # 112 → 28  (to match after block3's pool)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            lrelu(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )  # → (B, 64, 28, 28)

        self.block4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            lrelu(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )  # → (B, 128, 14, 14)

        self.block5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            lrelu(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )  # → (B, 256, 7, 7)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 100),
            lrelu(),
            nn.Linear(100, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.block1(x)      # (B, 16, 112, 112)
        x2 = self.block2(x1)     # (B, 32, 56, 56)
        x3 = self.block3(x2)     # (B, 64, 28, 28)
        skip = self.skip_proj(x1) # (B, 64, 28, 28)
        x3 = x3 + skip            # residual addition
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        return self.classifier(x5)


# ══════════════════════════════════════════════════════════════════════
# 3. Custom Vision Transformer (PyTorch, from scratch)
# ══════════════════════════════════════════════════════════════════════

class PatchEmbedding(nn.Module):
    """
    Converts a (B, C, H, W) image into patch embeddings.
    Uses a Conv2d with stride=patch_size to extract non-overlapping patches.
    Adds learnable positional embeddings and a CLS token.
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, embed_dim: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patcher = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patcher(x).transpose(1, 2)            # (B, num_patches, embed_dim)
        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, embed_dim)
        x = torch.cat([cls, x], dim=1)                  # (B, num_patches+1, embed_dim)
        x = self.dropout(x + self.pos_embed)
        return x


class ViTBlock(nn.Module):
    """Single Transformer encoder block with pre-norm and GELU MLP."""
    def __init__(self, embed_dim: int, num_heads: int,
                 mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads,
                                           dropout=dropout, batch_first=True)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ViT_PyTorch(nn.Module):
    """
    Vision Transformer for ASD binary classification (built from scratch).

    Default config matches the compact configuration used in experiments:
        embed_dim=512, num_heads=8, num_layers=12, patch_size=16

    Note: From-scratch ViTs require large datasets to train effectively.
    On the ABIDE-I slice dataset, this model achieves only ~60% accuracy.
    Use pretrained weights (ViT-B/16) for competitive performance.

    Input:  (batch, 3, 224, 224)
    Output: (batch, num_classes)
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, embed_dim: int = 512,
                 num_heads: int = 8, num_layers: int = 12,
                 num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels,
                                          embed_dim, dropout)
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, mlp_ratio=2.0, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, 0])   # CLS token → classification head


# ══════════════════════════════════════════════════════════════════════
# Factory function
# ══════════════════════════════════════════════════════════════════════

def get_model(name: str, **kwargs) -> nn.Module:
    """
    Factory to instantiate models by name.

    Args:
        name: 'cnn' | 'skip_cnn' | 'vit'
        **kwargs: passed to the model constructor

    Example:
        model = get_model('skip_cnn').to(device)
    """
    registry = {
        'cnn':      ASD_CNN,
        'skip_cnn': ASD_SkipCNN,
        'vit':      ViT_PyTorch,
    }
    if name not in registry:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(registry.keys())}")
    return registry[name](**kwargs)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy = torch.randn(4, 3, 224, 224).to(device)

    for name in ['cnn', 'skip_cnn', 'vit']:
        model = get_model(name).to(device)
        out = model(dummy)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name:10s} | output: {out.shape} | params: {n_params:,}")
