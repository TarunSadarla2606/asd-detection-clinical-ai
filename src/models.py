"""ASD detection model architectures — Phase 1 (B.Tech) and Phase 2 (Graduate).

Phase 1  (B.Tech 2023, asd-detection-neuroimaging):
    ASD_CNN      — 5-layer CNN, plain
    ASD_SkipCNN  — 5-layer CNN with skip connection (block1→block3)
    ViT_PyTorch  — from-scratch ViT (failed to converge at ABIDE-I scale)

Phase 2  (Graduate 2026, this repo):
    ASDClassifierCNN  — 67K quality-filtered slices, AUC 0.994
    HybridCNNViT      — CNN backbone + 4-block Transformer, AUC 0.997

Phase 2 attribute names match app/model.py checkpoints — do NOT rename.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Phase 1 — B.Tech baselines  (3-channel input, 224×224)
# ---------------------------------------------------------------------------

class ASD_CNN(nn.Module):
    """5-layer CNN, Phase 1 B.Tech baseline.  Input: (B, 3, 224, 224)."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()

        def _block(in_ch, out_ch, bn=True):
            layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1)]
            if bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers += [nn.LeakyReLU(0.1, inplace=True),
                       nn.MaxPool2d(2), nn.Dropout2d(dropout)]
            return nn.Sequential(*layers)

        self.block1 = _block(3,   16,  bn=False)
        self.block2 = _block(16,  32)
        self.block3 = _block(32,  64)
        self.block4 = _block(64,  128)
        self.block5 = _block(128, 256)
        # 5× MaxPool(2) on 224 → 7
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 100),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(100, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in [self.block1, self.block2,
                    self.block3, self.block4, self.block5]:
            x = blk(x)
        return self.classifier(x)


class ASD_SkipCNN(nn.Module):
    """5-layer CNN with residual skip from block1 to block3.  Input: (B, 3, 224, 224)."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        lrelu = lambda: nn.LeakyReLU(0.1, inplace=True)

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), lrelu(),
            nn.MaxPool2d(2), nn.Dropout2d(dropout))          # (B,16,112,112)

        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), lrelu(),
            nn.MaxPool2d(2), nn.Dropout2d(dropout))          # (B,32,56,56)

        # project block1 output to match block3 output: channels 16→64, spatial 112→28
        self.skip_proj = nn.Sequential(
            nn.Conv2d(16, 64, 1),
            nn.MaxPool2d(4),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), lrelu(),
            nn.MaxPool2d(2), nn.Dropout2d(dropout))          # (B,64,28,28)

        self.block4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), lrelu(),
            nn.MaxPool2d(2), nn.Dropout2d(dropout))          # (B,128,14,14)

        self.block5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), lrelu(),
            nn.MaxPool2d(2), nn.Dropout2d(dropout))          # (B,256,7,7)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 100), lrelu(),
            nn.Linear(100, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2) + self.skip_proj(x1)   # residual addition
        return self.classifier(self.block5(self.block4(x3)))


class ViT_PyTorch(nn.Module):
    """Minimal ViT (Phase 1 — failed to converge from scratch; included for comparison).

    Input: (B, in_chans, img_size, img_size).
    Default img_size=64 / patch_size=8 for fast testing; use 224/16 to match B.Tech.
    """

    def __init__(self, img_size: int = 64, patch_size: int = 8,
                 in_chans: int = 3, embed_dim: int = 128,
                 depth: int = 4, num_heads: int = 4,
                 num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.patch_size  = patch_size
        num_patches      = (img_size // patch_size) ** 2
        patch_dim        = in_chans * patch_size * patch_size

        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed   = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout     = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            batch_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)                # (B,C,nH,nW,p,p)
        x = x.contiguous().view(B, C, -1, p * p).permute(0, 2, 1, 3)
        x = x.reshape(B, x.shape[1], -1)                      # (B,N,C*p*p)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1) + self.pos_embed
        x   = self.dropout(x)
        x   = self.norm(self.transformer(x)[:, 0])            # CLS token
        return self.head(x)


# ---------------------------------------------------------------------------
# Phase 2 — Graduate system  (1-channel input, 224×224)
# Attribute names match app/model.py checkpoints — do NOT rename.
# ---------------------------------------------------------------------------

class ASDClassifierCNN(nn.Module):
    """5-block CNN trained on 67K quality-filtered ABIDE-I axial slices.

    Input: (B, 1, 224, 224) greyscale, normalised with CNN_MEAN / CNN_STD.
    AUC 0.994 on held-out test set (18,814 slices).
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        # Block 1 — 32 filters, no BN
        self.conv1    = nn.Conv2d(1, 32, 3, padding=1)
        self.lrelu1   = nn.LeakyReLU(0.1)
        self.pool1    = nn.MaxPool2d(2)
        self.drop1    = nn.Dropout2d(dropout)
        # Block 2 — 64 filters + BN
        self.conv2    = nn.Conv2d(32, 64, 3, padding=1)
        self.lrelu2   = nn.LeakyReLU(0.1)
        self.bn1      = nn.BatchNorm2d(64)
        self.pool2    = nn.MaxPool2d(2)
        self.drop2    = nn.Dropout2d(dropout)
        # Block 3 — 128 filters + BN
        self.conv3    = nn.Conv2d(64, 128, 3, padding=1)
        self.lrelu3   = nn.LeakyReLU(0.1)
        self.bn2      = nn.BatchNorm2d(128)
        self.pool3    = nn.MaxPool2d(2)
        self.drop3    = nn.Dropout2d(dropout)
        # Block 4 — 256 filters + BN
        self.conv4    = nn.Conv2d(128, 256, 3, padding=1)
        self.lrelu4   = nn.LeakyReLU(0.1)
        self.bn3      = nn.BatchNorm2d(256)
        self.pool4    = nn.MaxPool2d(2)
        self.drop4    = nn.Dropout2d(dropout)
        # Block 5 — 512 filters + BN + AdaptivePool
        self.conv5    = nn.Conv2d(256, 512, 3, padding=1)
        self.lrelu5   = nn.LeakyReLU(0.1)
        self.bn4      = nn.BatchNorm2d(512)
        self.pool5    = nn.AdaptiveAvgPool2d((1, 1))
        self.drop5    = nn.Dropout2d(dropout)
        # Head
        self.flatten  = nn.Flatten()
        self.fc1      = nn.Linear(512, 256)
        self.lrelu_fc = nn.LeakyReLU(0.1)
        self.fc2      = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.pool1(self.lrelu1(self.conv1(x))))
        x = self.drop2(self.pool2(self.bn1(self.lrelu2(self.conv2(x)))))
        x = self.drop3(self.pool3(self.bn2(self.lrelu3(self.conv3(x)))))
        x = self.drop4(self.pool4(self.bn3(self.lrelu4(self.conv4(x)))))
        x = self.drop5(self.pool5(self.bn4(self.lrelu5(self.conv5(x)))))
        return self.fc2(self.lrelu_fc(self.fc1(self.flatten(x))))


class TransformerBlock(nn.Module):
    """Single Transformer encoder block.  Attribute names match checkpoints."""

    def __init__(self, dim: int, heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.n1   = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True)
        self.n2   = nn.LayerNorm(dim)
        mlp_dim   = int(dim * mlp_ratio)
        self.ffn  = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout),
        )
        self.attn_w = None   # populated during forward; used by get_cls_attn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.n1(x)
        attn_out, self.attn_w = self.attn(
            normed, normed, normed,
            need_weights=True, average_attn_weights=False)
        x = x + attn_out
        return x + self.ffn(self.n2(x))


class CNNBackbone(nn.Module):
    """4-block CNN backbone.  Output: (B, 128, 14, 14) for 224×224 input."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.LeakyReLU(0.1), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.LeakyReLU(0.1), nn.BatchNorm2d(64), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.1), nn.BatchNorm2d(128), nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.LeakyReLU(0.1), nn.BatchNorm2d(128), nn.MaxPool2d(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv4(self.conv3(self.conv2(self.conv1(x))))


class HybridCNNViT(nn.Module):
    """Hybrid CNN + Vision Transformer for ASD classification.

    CNNBackbone → 196 spatial tokens → 4-block Transformer → CLS-token head.
    Input: (B, 1, 224, 224), normalised with HYBRID_MEAN / HYBRID_STD.
    AUC 0.997 on held-out test set.
    Attribute names match app/model.py checkpoints — do NOT rename.
    """

    def __init__(self, num_classes: int = 2, embed_dim: int = 256,
                 depth: int = 4, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.backbone    = CNNBackbone()                   # (B,128,14,14)
        self.token_proj  = nn.Linear(128, embed_dim)

        num_patches = 14 * 14  # 196
        self.register_buffer("pos_emb",
                             torch.zeros(1, num_patches, embed_dim))
        self.cls_tok = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_pe  = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.transformer = nn.ModuleList([
            TransformerBlock(embed_dim, heads, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.drop = nn.Dropout(dropout)

        nn.init.trunc_normal_(self.cls_tok, std=0.02)
        nn.init.trunc_normal_(self.cls_pe,  std=0.02)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        feat   = self.backbone(x)                               # (B,128,14,14)
        tokens = feat.flatten(2).transpose(1, 2)               # (B,196,128)
        tokens = self.token_proj(tokens) + self.pos_emb        # (B,196,256)
        cls    = self.cls_tok.expand(B, -1, -1) + self.cls_pe  # (B,1,256)
        tokens = self.drop(torch.cat([cls, tokens], dim=1))    # (B,197,256)
        for blk in self.transformer:
            tokens = blk(tokens)
        return self.head(self.norm(tokens[:, 0]))

    def get_cls_attn(self):
        """Last-layer CLS→spatial attention weights: (B, heads, 196)."""
        w = self.transformer[-1].attn_w
        return None if w is None else w[:, :, 0, 1:]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(name: str, **kwargs) -> nn.Module:
    """Instantiate a model by name.

    Names: ``asd_cnn``, ``asd_skipcnn``, ``vit``, ``cnn``, ``hybrid``.
    """
    registry = {
        "asd_cnn":    ASD_CNN,
        "asd_skipcnn": ASD_SkipCNN,
        "vit":        ViT_PyTorch,
        "cnn":        ASDClassifierCNN,
        "hybrid":     HybridCNNViT,
    }
    if name not in registry:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(registry)}")
    return registry[name](**kwargs)
