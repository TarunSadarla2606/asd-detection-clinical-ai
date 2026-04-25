# model.py — v3
# CNN + Hybrid CNN-ViT for ASD detection from sMRI.
# IMPORTANT: TransformerBlock and HybridCNNViT attribute names match the
# Colab training notebook EXACTLY so that load_state_dict works without
# any key remapping. Do not rename any nn.Module attributes.

import math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


# ── Normalisation stats ───────────────────────────────────────────────────────
# CNN Phase 2 — computed on ABIDE-I training set in original Kaggle notebook
CNN_MEAN = [0.1290, 0.1290, 0.1290]
CNN_STD  = [0.1741, 0.1741, 0.1741]

# Hybrid CNN-ViT — stats used during Colab training
HYBRID_MEAN = [0.1425056904554367,  0.1425056904554367,  0.1425056904554367]
HYBRID_STD  = [0.19151894748210907, 0.19151894748210907, 0.19151894748210907]

CNN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(CNN_MEAN, CNN_STD),
])

HYBRID_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(HYBRID_MEAN, HYBRID_STD),
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ASDClassifierCNN  (Phase 2, 5-layer CNN)
# ══════════════════════════════════════════════════════════════════════════════

class ASDClassifierCNN(nn.Module):
    """
    5-layer CNN for ASD vs TC classification from 2D axial sMRI slices.
    Input : (B, 3, 224, 224)
    Output: (B, 2) raw logits
    Trained on ABIDE-I (1,067 subjects, 17 sites).
    Test performance: AUC 0.994, Sensitivity 95.6%, Specificity 97.2%
    """
    def __init__(self):
        super().__init__()
        self.conv1  = nn.Conv2d(3,   16,  3, padding=1)
        self.lrelu1 = nn.LeakyReLU(0.1)
        self.pool1  = nn.MaxPool2d(2, 2)
        self.drop1  = nn.Dropout2d(0.2)

        self.conv2  = nn.Conv2d(16,  32,  3, padding=1)
        self.lrelu2 = nn.LeakyReLU(0.1)
        self.bn1    = nn.BatchNorm2d(32)
        self.pool2  = nn.MaxPool2d(2, 2)
        self.drop2  = nn.Dropout2d(0.2)

        self.conv3  = nn.Conv2d(32,  64,  3, padding=1)
        self.lrelu3 = nn.LeakyReLU(0.1)
        self.bn2    = nn.BatchNorm2d(64)
        self.pool3  = nn.MaxPool2d(2, 2)
        self.drop3  = nn.Dropout2d(0.2)

        self.conv4  = nn.Conv2d(64,  128, 3, padding=1)
        self.lrelu4 = nn.LeakyReLU(0.1)
        self.bn3    = nn.BatchNorm2d(128)
        self.pool4  = nn.MaxPool2d(2, 2)
        self.drop4  = nn.Dropout2d(0.2)

        self.conv5  = nn.Conv2d(128, 256, 3, padding=1)   # ← GradCAM target
        self.lrelu5 = nn.LeakyReLU(0.1)
        self.bn4    = nn.BatchNorm2d(256)
        self.pool5  = nn.MaxPool2d(2, 2)
        self.drop5  = nn.Dropout2d(0.2)

        self.flatten  = nn.Flatten()
        self.fc1      = nn.Linear(7 * 7 * 256, 100)
        self.lrelu_fc = nn.LeakyReLU(0.1)
        self.fc2      = nn.Linear(100, 2)

    def forward(self, x):
        x = self.drop1(self.pool1(self.lrelu1(self.conv1(x))))
        x = self.drop2(self.pool2(self.bn1(self.lrelu2(self.conv2(x)))))
        x = self.drop3(self.pool3(self.bn2(self.lrelu3(self.conv3(x)))))
        x = self.drop4(self.pool4(self.bn3(self.lrelu4(self.conv4(x)))))
        x = self.drop5(self.pool5(self.bn4(self.lrelu5(self.conv5(x)))))
        x = self.flatten(x)
        x = self.lrelu_fc(self.fc1(x))
        return self.fc2(x)


def load_model(weights_path: str, device: str = 'cpu') -> ASDClassifierCNN:
    """Load Phase 2 CNN weights. Returns model in eval mode."""
    model = ASDClassifierCNN()
    model.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Hybrid CNN-ViT
# Attribute names below are intentionally kept identical to the Colab training
# notebook so that torch.load_state_dict() works without key remapping.
# ══════════════════════════════════════════════════════════════════════════════

# ── Hybrid model constants (match training notebook config cell) ──────────────
_EMBED_DIM = 256
_N_HEADS   = 8
_N_LAYERS  = 4
_FF_DIM    = 512
_DROPOUT   = 0.1
_CNN_OUT   = 128   # conv4 output channels
_FEAT_SIZE = 14    # 224 / 2^4 = 14
_N_TOKENS  = _FEAT_SIZE ** 2  # 196


# ── Brain masking utilities (used inside HybridCNNViT.forward) ────────────────

def generate_brain_mask(img_tensor: torch.Tensor,
                         threshold: float = 0.05) -> torch.Tensor:
    """
    Binary brain mask from normalised input (HYBRID_MEAN/STD space).
    Otsu-style threshold + morphological closing.
    Returns (B, 1, H, W) float32 {0, 1}.
    """
    gray     = img_tensor.mean(dim=1, keepdim=True)
    gray_orig = (gray * HYBRID_STD[0] + HYBRID_MEAN[0]).clamp(0, 1)
    mask     = (gray_orig > threshold).float()

    kernel = np.ones((15, 15), np.uint8)
    closed = []
    for b in range(mask.shape[0]):
        m = mask[b, 0].cpu().numpy().astype(np.uint8)
        closed.append(torch.from_numpy(
            cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)).float())
    return torch.stack(closed).unsqueeze(1).to(img_tensor.device)


def mask_to_tokens(mask: torch.Tensor,
                   feat_size: int = _FEAT_SIZE) -> torch.Tensor:
    """
    Downsample brain mask (B,1,H,W) → token mask (B, N_tokens) bool.
    Tokens with >30% brain coverage are treated as brain tokens.
    """
    down     = F.adaptive_avg_pool2d(mask, (feat_size, feat_size))
    tok_mask = (down > 0.3).squeeze(1)
    return tok_mask.reshape(tok_mask.shape[0], -1)


# ── Sinusoidal 2D positional embeddings ──────────────────────────────────────

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


# ── TransformerBlock — attribute names MUST match training notebook ───────────

class TransformerBlock(nn.Module):
    """
    Pre-LN transformer encoder block.
    Attribute names: n1, n2, attn, ffn, attn_w — match training notebook exactly.
    """
    def __init__(self, ed: int, nh: int, ff: int, dr: float = 0.1):
        super().__init__()
        self.n1   = nn.LayerNorm(ed)
        self.attn = nn.MultiheadAttention(ed, nh, dropout=dr, batch_first=True)
        self.n2   = nn.LayerNorm(ed)
        self.ffn  = nn.Sequential(
            nn.Linear(ed, ff),
            nn.GELU(),
            nn.Dropout(dr),
            nn.Linear(ff, ed),
            nn.Dropout(dr),
        )
        self.attn_w = None  # stored after each forward for attention rollout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.n1(x)
        ao, aw = self.attn(xn, xn, xn,
                           need_weights=True, average_attn_weights=False)
        self.attn_w = aw.detach()  # (B, n_heads, seq, seq)
        x = x + ao
        x = x + self.ffn(self.n2(x))
        return x


# ── CNNBackbone — attribute names MUST match training notebook ────────────────

class CNNBackbone(nn.Module):
    """
    conv1–conv4 from ASDClassifierCNN. Output: (B, 128, 14, 14).
    Attribute names match the Colab training notebook exactly.
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


# ── HybridCNNViT — attribute names MUST match training notebook ───────────────

class HybridCNNViT(nn.Module):
    """
    Hybrid CNN-ViT for ASD classification from 2D axial sMRI slices.

    Architecture:
        Input (B,3,224,224)
        → Brain mask (Otsu threshold + morphological closing)
        → CNNBackbone (conv1–conv4) → (B,128,14,14) [196 spatial tokens]
        → Background zeroing at token resolution
        → Linear projection 128→256 + 2D sinusoidal pos embeddings
        → CLS token prepend → (B,197,256)
        → 4 × TransformerBlock(8 heads, FF=512, Pre-LN)
        → CLS → LayerNorm → Linear(256,2) → (B,2) logits

    IMPORTANT — attribute names below match the Colab training notebook:
        backbone, token_proj, pos_emb (buffer), cls_tok, cls_pe,
        transformer, norm, head, drop
        fs, nt, ed (instance attributes)
    Do NOT rename these — state_dict keys depend on them.
    """

    def __init__(self,
                 cnn_out: int  = _CNN_OUT,
                 fs: int       = _FEAT_SIZE,
                 ed: int       = _EMBED_DIM,
                 nh: int       = _N_HEADS,
                 nl: int       = _N_LAYERS,
                 ff: int       = _FF_DIM,
                 dr: float     = _DROPOUT):
        super().__init__()
        self.fs = fs
        self.nt = fs * fs   # 196
        self.ed = ed

        self.backbone   = CNNBackbone()
        self.token_proj = nn.Sequential(
            nn.Linear(cnn_out, ed),
            nn.LayerNorm(ed),
        )
        pe = sinusoidal_2d_pos_emb(fs, ed)
        self.register_buffer('pos_emb', pe)              # (1, 196, ed)

        self.cls_tok = nn.Parameter(torch.zeros(1, 1, ed))
        self.cls_pe  = nn.Parameter(torch.zeros(1, 1, ed))
        nn.init.trunc_normal_(self.cls_tok, std=0.02)
        nn.init.trunc_normal_(self.cls_pe,  std=0.02)

        self.transformer = nn.ModuleList([
            TransformerBlock(ed, nh, ff, dr) for _ in range(nl)
        ])
        self.norm = nn.LayerNorm(ed)
        self.head = nn.Linear(ed, 2)
        self.drop = nn.Dropout(dr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # Brain mask → token mask
        mask     = generate_brain_mask(x)                          # (B,1,224,224)
        tok_mask = mask_to_tokens(mask, self.fs).float().unsqueeze(-1)  # (B,196,1)
        # CNN features
        feat   = self.backbone(x)                                  # (B,128,14,14)
        tokens = feat.permute(0,2,3,1).reshape(B, self.nt, -1)    # (B,196,128)
        tokens = tokens * tok_mask                                  # zero background
        # Project + positional encoding
        tokens = self.drop(self.token_proj(tokens)) + self.pos_emb # (B,196,ed)
        # CLS token
        cls    = (self.cls_tok + self.cls_pe).expand(B, -1, -1)   # (B,1,ed)
        tokens = torch.cat([cls, tokens], dim=1)                   # (B,197,ed)
        # Transformer
        for blk in self.transformer:
            tokens = blk(tokens)
        return self.head(self.norm(tokens[:, 0]))                  # (B,2)

    def get_cls_attn(self) -> torch.Tensor | None:
        """
        CLS→token attention from the last transformer layer, head-averaged.
        Returns (B, 196) normalised to [0,1], or None if not yet run.
        Reshape to (B,14,14) for spatial visualisation.
        NOTE: method name matches training notebook (get_cls_attn, not get_cls_attention).
        """
        aw = self.transformer[-1].attn_w   # (B, n_heads, 197, 197)
        if aw is None:
            return None
        ca = aw[:, :, 0, 1:].mean(dim=1)  # (B, 196) — CLS row, all patch tokens
        mn = ca.min(1, keepdim=True)[0]
        mx = ca.max(1, keepdim=True)[0]
        return (ca - mn) / (mx - mn + 1e-8)

    def get_token_brain_coverage(self, img_tensor: torch.Tensor) -> dict:
        """Returns brain token count and fraction for a single input tensor."""
        mask     = generate_brain_mask(img_tensor)
        tok_mask = mask_to_tokens(mask, self.fs)
        n_brain  = int(tok_mask[0].sum().item())
        return {
            'n_brain'  : n_brain,
            'n_total'  : self.nt,
            'fraction' : n_brain / self.nt,
        }


def load_hybrid_model(weights_path: str, device: str = 'cpu') -> HybridCNNViT:
    """
    Load Hybrid CNN-ViT weights saved by the Colab training notebook.
    Returns model in eval mode.
    """
    model = HybridCNNViT()
    state = torch.load(weights_path, map_location=device, weights_only=True)

    # Diagnostic: verify key alignment before loading
    model_keys   = set(model.state_dict().keys())
    ckpt_keys    = set(state.keys())
    missing      = model_keys - ckpt_keys
    unexpected   = ckpt_keys - model_keys

    if missing:
        print(f"[load_hybrid_model] WARNING: {len(missing)} keys in model not in checkpoint:")
        for k in sorted(missing)[:10]:
            print(f"  missing: {k}")
    if unexpected:
        print(f"[load_hybrid_model] WARNING: {len(unexpected)} keys in checkpoint not in model:")
        for k in sorted(unexpected)[:10]:
            print(f"  unexpected: {k}")

    if missing or unexpected:
        # Attempt partial load — load what matches, skip the rest
        model.load_state_dict(state, strict=False)
        print(f"[load_hybrid_model] Loaded with strict=False. "
              f"Matched {len(model_keys & ckpt_keys)}/{len(model_keys)} keys.")
    else:
        model.load_state_dict(state)
        print(f"[load_hybrid_model] All {len(model_keys)} keys matched perfectly.")

    model.eval()
    return model
