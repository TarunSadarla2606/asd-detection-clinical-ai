"""ASD detection model architectures — Phase 1 (B.Tech) and Phase 2 (Graduate).

Phase 1  (B.Tech 2023, asd-detection-neuroimaging):
    ASD_CNN      — 5-layer CNN, plain
    ASD_SkipCNN  — 5-layer CNN with skip connection (block1→block3)
    ViT_PyTorch  — from-scratch ViT (failed to converge at ABIDE-I scale)

Phase 2  (Graduate 2026, this repo):
    ASDClassifierCNN  — 67K quality-filtered slices, AUC 0.994
    HybridCNNViT      — CNN backbone + 4-block Transformer, AUC 0.997

Phase 2 architectures and attribute names mirror app/model.py exactly so
that src/models.py can be used to verify state_dict compatibility.
"""

import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


# ── Normalisation constants (match app/model.py) ─────────────────────────────
CNN_MEAN    = [0.1290, 0.1290, 0.1290]
CNN_STD     = [0.1741, 0.1741, 0.1741]
HYBRID_MEAN = [0.1425056904554367,  0.1425056904554367,  0.1425056904554367]
HYBRID_STD  = [0.19151894748210907, 0.19151894748210907, 0.19151894748210907]


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
            nn.MaxPool2d(2), nn.Dropout2d(dropout))
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), lrelu(),
            nn.MaxPool2d(2), nn.Dropout2d(dropout))
        self.skip_proj = nn.Sequential(
            nn.Conv2d(16, 64, 1), nn.MaxPool2d(4))
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), lrelu(),
            nn.MaxPool2d(2), nn.Dropout2d(dropout))
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), lrelu(),
            nn.MaxPool2d(2), nn.Dropout2d(dropout))
        self.block5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), lrelu(),
            nn.MaxPool2d(2), nn.Dropout2d(dropout))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 100), lrelu(),
            nn.Linear(100, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2) + self.skip_proj(x1)
        return self.classifier(self.block5(self.block4(x3)))


class ViT_PyTorch(nn.Module):
    """Minimal ViT (Phase 1 — failed to converge from scratch; included for comparison).

    Input: (B, in_chans, img_size, img_size).
    Default img_size=64 / patch_size=8 for fast testing.
    """

    def __init__(self, img_size: int = 64, patch_size: int = 8,
                 in_chans: int = 3, embed_dim: int = 128,
                 depth: int = 4, num_heads: int = 4,
                 num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.patch_size = patch_size
        num_patches     = (img_size // patch_size) ** 2
        patch_dim       = in_chans * patch_size * patch_size

        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed   = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout     = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(B, C, -1, p * p).permute(0, 2, 1, 3)
        x = x.reshape(B, x.shape[1], -1)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1) + self.pos_embed
        x   = self.dropout(x)
        x   = self.norm(self.transformer(x)[:, 0])
        return self.head(x)


# ---------------------------------------------------------------------------
# Phase 2 helpers  (match app/model.py exactly)
# ---------------------------------------------------------------------------

def generate_brain_mask(img_tensor: torch.Tensor,
                        threshold: float = 0.05) -> torch.Tensor:
    """Binary brain mask from HYBRID_MEAN/STD normalised input.
    Returns (B, 1, H, W) float32 {0, 1}.
    """
    gray      = img_tensor.mean(dim=1, keepdim=True)
    gray_orig = (gray * HYBRID_STD[0] + HYBRID_MEAN[0]).clamp(0, 1)
    mask      = (gray_orig > threshold).float()
    kernel    = np.ones((15, 15), np.uint8)
    closed    = []
    for b in range(mask.shape[0]):
        m = mask[b, 0].cpu().numpy().astype(np.uint8)
        closed.append(torch.from_numpy(
            cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)).float())
    return torch.stack(closed).unsqueeze(1).to(img_tensor.device)


def mask_to_tokens(mask: torch.Tensor, feat_size: int = 14) -> torch.Tensor:
    """Downsample brain mask (B,1,H,W) → token mask (B, feat_size^2) bool."""
    down     = F.adaptive_avg_pool2d(mask, (feat_size, feat_size))
    tok_mask = (down > 0.3).squeeze(1)
    return tok_mask.reshape(tok_mask.shape[0], -1)


def sinusoidal_2d_pos_emb(fs: int, ed: int) -> torch.Tensor:
    """(1, fs*fs, ed) 2D sinusoidal positional embedding."""
    hd    = ed // 2
    scale = math.log(10000) / (hd // 2 - 1)
    emb   = torch.exp(torch.arange(hd // 2) * -scale)

    def enc(pos):
        return torch.cat([torch.sin(pos * emb), torch.cos(pos * emb)], dim=1)

    rows = torch.arange(fs).float().unsqueeze(1)
    cols = torch.arange(fs).float().unsqueeze(1)
    re, ce = enc(rows), enc(cols)
    pe = [torch.cat([re[r], ce[c]]) for r in range(fs) for c in range(fs)]
    return torch.stack(pe).unsqueeze(0)  # (1, N, ed)


# ---------------------------------------------------------------------------
# Phase 2 — Graduate system  (3-channel input, 224×224)
# Attribute names mirror app/model.py exactly — do NOT rename.
# ---------------------------------------------------------------------------

class ASDClassifierCNN(nn.Module):
    """5-block CNN trained on 67K quality-filtered ABIDE-I axial slices.

    Input:  (B, 3, 224, 224) greyscale replicated to 3 channels.
    Output: (B, 2) logits.
    AUC 0.994 on held-out test set (18,814 slices).
    Attribute names match app/model.py checkpoints — do NOT rename.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.conv1  = nn.Conv2d(3,   16,  3, padding=1)
        self.lrelu1 = nn.LeakyReLU(0.1)
        self.pool1  = nn.MaxPool2d(2, 2)
        self.drop1  = nn.Dropout2d(dropout)

        self.conv2  = nn.Conv2d(16,  32,  3, padding=1)
        self.lrelu2 = nn.LeakyReLU(0.1)
        self.bn1    = nn.BatchNorm2d(32)
        self.pool2  = nn.MaxPool2d(2, 2)
        self.drop2  = nn.Dropout2d(dropout)

        self.conv3  = nn.Conv2d(32,  64,  3, padding=1)
        self.lrelu3 = nn.LeakyReLU(0.1)
        self.bn2    = nn.BatchNorm2d(64)
        self.pool3  = nn.MaxPool2d(2, 2)
        self.drop3  = nn.Dropout2d(dropout)

        self.conv4  = nn.Conv2d(64,  128, 3, padding=1)
        self.lrelu4 = nn.LeakyReLU(0.1)
        self.bn3    = nn.BatchNorm2d(128)
        self.pool4  = nn.MaxPool2d(2, 2)
        self.drop4  = nn.Dropout2d(dropout)

        self.conv5  = nn.Conv2d(128, 256, 3, padding=1)
        self.lrelu5 = nn.LeakyReLU(0.1)
        self.bn4    = nn.BatchNorm2d(256)
        self.pool5  = nn.MaxPool2d(2, 2)
        self.drop5  = nn.Dropout2d(dropout)

        self.flatten  = nn.Flatten()
        self.fc1      = nn.Linear(7 * 7 * 256, 100)
        self.lrelu_fc = nn.LeakyReLU(0.1)
        self.fc2      = nn.Linear(100, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.pool1(self.lrelu1(self.conv1(x))))
        x = self.drop2(self.pool2(self.bn1(self.lrelu2(self.conv2(x)))))
        x = self.drop3(self.pool3(self.bn2(self.lrelu3(self.conv3(x)))))
        x = self.drop4(self.pool4(self.bn3(self.lrelu4(self.conv4(x)))))
        x = self.drop5(self.pool5(self.bn4(self.lrelu5(self.conv5(x)))))
        return self.fc2(self.lrelu_fc(self.fc1(self.flatten(x))))


class TransformerBlock(nn.Module):
    """Pre-LN Transformer encoder block.
    Attribute names: n1, n2, attn, ffn, attn_w — match app/model.py checkpoints.
    """

    def __init__(self, dim: int, heads: int,
                 ff_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.n1   = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True)
        self.n2   = nn.LayerNorm(dim)
        self.ffn  = nn.Sequential(
            nn.Linear(dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, dim), nn.Dropout(dropout),
        )
        self.attn_w = None   # stored after forward; shape (B, heads, seq, seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.n1(x)
        ao, aw = self.attn(xn, xn, xn,
                           need_weights=True, average_attn_weights=False)
        self.attn_w = aw.detach()  # detach to avoid holding the computation graph
        x = x + ao
        return x + self.ffn(self.n2(x))


class CNNBackbone(nn.Module):
    """First 4 blocks of ASDClassifierCNN.  Output: (B, 128, 14, 14) for 224×224.
    Flat attribute names (conv1–conv4, lrelu1–lrelu4, bn1–bn3, pool1–pool4,
    drop1–drop4) match app/model.py checkpoints — do NOT rename.
    """

    def __init__(self):
        super().__init__()
        self.conv1  = nn.Conv2d(3,  16,  3, padding=1)
        self.lrelu1 = nn.LeakyReLU(0.1)
        self.pool1  = nn.MaxPool2d(2, 2)
        self.drop1  = nn.Dropout2d(0.2)

        self.conv2  = nn.Conv2d(16, 32,  3, padding=1)
        self.lrelu2 = nn.LeakyReLU(0.1)
        self.bn1    = nn.BatchNorm2d(32)
        self.pool2  = nn.MaxPool2d(2, 2)
        self.drop2  = nn.Dropout2d(0.2)

        self.conv3  = nn.Conv2d(32, 64,  3, padding=1)
        self.lrelu3 = nn.LeakyReLU(0.1)
        self.bn2    = nn.BatchNorm2d(64)
        self.pool3  = nn.MaxPool2d(2, 2)
        self.drop3  = nn.Dropout2d(0.2)

        self.conv4  = nn.Conv2d(64, 128, 3, padding=1)
        self.lrelu4 = nn.LeakyReLU(0.1)
        self.bn3    = nn.BatchNorm2d(128)
        self.pool4  = nn.MaxPool2d(2, 2)
        self.drop4  = nn.Dropout2d(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.pool1(self.lrelu1(self.conv1(x))))
        x = self.drop2(self.pool2(self.bn1(self.lrelu2(self.conv2(x)))))
        x = self.drop3(self.pool3(self.bn2(self.lrelu3(self.conv3(x)))))
        return self.drop4(self.pool4(self.bn3(self.lrelu4(self.conv4(x)))))


class HybridCNNViT(nn.Module):
    """Hybrid CNN + Vision Transformer for ASD classification.

    CNNBackbone → 196 spatial tokens (brain-masked) → 4-block Transformer → CLS head.
    Input:  (B, 3, 224, 224) greyscale replicated to 3 channels.
    Output: (B, 2) logits.
    AUC 0.997 on held-out test set.
    Attribute names mirror app/model.py checkpoints — do NOT rename.
    """

    def __init__(self, num_classes: int = 2,
                 cnn_out: int = 128, fs: int = 14,
                 ed: int = 256, nh: int = 8, nl: int = 4,
                 ff: int = 512, dr: float = 0.1):
        super().__init__()
        self.fs = fs
        self.nt = fs * fs   # 196

        self.backbone   = CNNBackbone()
        self.token_proj = nn.Sequential(
            nn.Linear(cnn_out, ed),
            nn.LayerNorm(ed),
        )
        pe = sinusoidal_2d_pos_emb(fs, ed)
        self.register_buffer('pos_emb', pe)          # (1, 196, ed)

        self.cls_tok = nn.Parameter(torch.zeros(1, 1, ed))
        self.cls_pe  = nn.Parameter(torch.zeros(1, 1, ed))
        nn.init.trunc_normal_(self.cls_tok, std=0.02)
        nn.init.trunc_normal_(self.cls_pe,  std=0.02)

        self.transformer = nn.ModuleList([
            TransformerBlock(ed, nh, ff_dim=ff, dropout=dr)
            for _ in range(nl)
        ])
        self.norm = nn.LayerNorm(ed)
        self.head = nn.Linear(ed, num_classes)
        self.drop = nn.Dropout(dr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        mask     = generate_brain_mask(x)
        tok_mask = mask_to_tokens(mask, self.fs).float().unsqueeze(-1)   # (B,196,1)
        feat     = self.backbone(x)                                       # (B,128,14,14)
        tokens   = feat.permute(0, 2, 3, 1).reshape(B, self.nt, -1)     # (B,196,128)
        tokens   = tokens * tok_mask
        tokens   = self.drop(self.token_proj(tokens)) + self.pos_emb    # (B,196,ed)
        cls      = (self.cls_tok + self.cls_pe).expand(B, -1, -1)       # (B,1,ed)
        tokens   = torch.cat([cls, tokens], dim=1)                       # (B,197,ed)
        for blk in self.transformer:
            tokens = blk(tokens)
        return self.head(self.norm(tokens[:, 0]))

    def get_cls_attn(self):
        """Head-averaged CLS→token attention from last transformer layer.
        Returns (B, 196) normalised to [0,1], or None before first forward pass.
        """
        aw = self.transformer[-1].attn_w          # (B, heads, 197, 197) or None
        if aw is None:
            return None
        ca = aw[:, :, 0, 1:].mean(dim=1)          # (B, 196)
        mn = ca.min(1, keepdim=True)[0]
        mx = ca.max(1, keepdim=True)[0]
        return (ca - mn) / (mx - mn + 1e-8)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(name: str, **kwargs) -> nn.Module:
    """Instantiate a model by name.

    Names: ``asd_cnn``, ``asd_skipcnn``, ``vit``, ``cnn``, ``hybrid``.
    """
    registry = {
        "asd_cnn":     ASD_CNN,
        "asd_skipcnn": ASD_SkipCNN,
        "vit":         ViT_PyTorch,
        "cnn":         ASDClassifierCNN,
        "hybrid":      HybridCNNViT,
    }
    if name not in registry:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(registry)}")
    return registry[name](**kwargs)
