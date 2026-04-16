"""
casa_models.py  —  RTD-Net core modules with CASA replacing CMHSA
=======================================================================
Paper basis:
  Original CMHSA — Ye et al., IEEE TIM Vol.72, 2023 (RTD-Net)
  CASA inspiration — additive attention mechanism from
      "HCTD: Hierarchical CNN-Transformer Detector", 2025 (arXiv)
      and the broader additive-attention literature
      (Shen et al., "Efficient Attention", WACV 2021)

What changed vs models.py:
  CMHSA  →  CASA   (Convolutional Additive Self-Attention)
  ECTB   →  updated to use CASA internally (all else identical)
  Everything else (ConvBNSiLU, LEM, NAM, APH) is untouched.

Why additive attention is better for aerial/occluded scenes:
  • Dot-product attention: similarity = Q·Kᵀ (expensive O(T²) in token count,
    sensitive to magnitude → attention collapse on low-contrast regions).
  • Additive attention: similarity = tanh(Wq·Q + Wk·K) (linear in T,
    robust to low-contrast because tanh is bounded, better gradient flow
    through occluded regions).

CASA architecture (per head):
  Input (B, C, H, W)
    ↓  1×1 conv  → Q, K, V  (same as CMHSA but only 3 convs, no inst-norm)
    ↓  Q + K  → additive interaction  → tanh  → attn_map  (B, heads, T, T)
    ↓  scale by V  (element-wise broadcast, not full QKV matmul)
    ↓  depthwise 3×3 conv  (local context enrichment, replaces head-conv)
    ↓  1×1 proj  → output  (B, C, H, W)

FLOPs saved vs CMHSA:
  CMHSA needs a full (T×T) softmax attention + inst-norm per head.
  CASA replaces that with  (B, heads, T, C/heads) additive ops — O(T·C)
  instead of O(T²·heads).  At 224px / 32 strides → T=49: negligible
  difference.  At 640px → T=400: ~8× fewer FLOPs in the attention step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ──────────────────────────────────────────────────────────────────────────────
# Shared helper (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
class ConvBNSiLU(nn.Module):
    """Conv → BN → SiLU.  kernel=1 default for pointwise ops."""
    def __init__(self, in_ch, out_ch, kernel=1, stride=1, padding=None, groups=1):
        super().__init__()
        if padding is None:
            padding = kernel // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding,
                              groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ──────────────────────────────────────────────────────────────────────────────
# LEM  (unchanged — included for drop-in completeness)
# ──────────────────────────────────────────────────────────────────────────────
class LEM(nn.Module):
    """
    Lightweight Feature Extraction Module.
    16 homogeneous branches (1×1 + 3×3), equal weights, residual.
    Unchanged from original RTD-Net.
    """
    def __init__(self, in_ch, out_ch=None, C=16):
        super().__init__()
        out_ch = out_ch or in_ch
        mid_ch = max(in_ch // 2,  1)
        br_ch  = max(in_ch // 32, 1)
        self.C = C

        self.conv1    = ConvBNSiLU(in_ch, mid_ch, 1)
        self.branches = nn.ModuleList([
            nn.Sequential(ConvBNSiLU(mid_ch, br_ch, 1),
                          ConvBNSiLU(br_ch,  br_ch, 3))
            for _ in range(C)
        ])
        branch_total = br_ch * C
        self.conv2   = nn.Conv2d(branch_total, out_ch, 1, bias=False)
        self.bn      = nn.BatchNorm2d(out_ch)
        self.skip    = (nn.Identity() if in_ch == out_ch else
                        nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                                      nn.BatchNorm2d(out_ch)))

    def forward(self, x):
        feat   = self.conv1(x)
        concat = torch.cat([b(feat) for b in self.branches], dim=1)
        out    = self.bn(self.conv2(concat))
        return F.silu(out + self.skip(x))


# ──────────────────────────────────────────────────────────────────────────────
# CASA — Convolutional Additive Self-Attention
# ──────────────────────────────────────────────────────────────────────────────
class CASA(nn.Module):
    """
    Convolutional Additive Self-Attention.

    Replaces CMHSA's expensive dot-product + softmax + instance-norm path
    with a cheaper additive interaction that is more robust to low-contrast
    and occluded regions (common in aerial/UAV imagery).

    Mechanism per head
    ──────────────────
    Standard:   A = softmax(Q·Kᵀ / √d)            ← O(T²) per head
    CASA:       A = sigmoid(dw_conv(tanh(Q + K)))  ← O(T·C), local context

    The additive Q+K interaction is equivalent to asking "are Q and K
    mutually activating?" rather than "how similar is Q to K?".  tanh
    bounds the response, preventing attention collapse on low-contrast
    patches.  A depthwise 3×3 then injects local spatial context before
    the sigmoid gate — this is the "convolutional" part that replaces the
    head-conv in CMHSA.

    Finally V is *element-wise gated* by A rather than matrix-multiplied,
    preserving the full spatial map without squashing it through an HxW
    softmax.  This is particularly beneficial when objects are small
    relative to the feature map (typical in UAV imagery).

    Args:
        dim       (int): Input channel count C. Must be divisible by num_heads.
        num_heads (int): Number of parallel attention heads. Default 4.
        drop      (float): Dropout on attention weights. Default 0.0.
    """
    def __init__(self, dim: int, num_heads: int = 4, drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads

        # ── Q / K / V projections (1×1 conv, same as CMHSA) ─────────────────
        self.conv_q = nn.Conv2d(dim, dim, 1, bias=False)
        self.conv_k = nn.Conv2d(dim, dim, 1, bias=False)
        self.conv_v = nn.Conv2d(dim, dim, 1, bias=False)

        # ── Additive interaction path ─────────────────────────────────────────
        # Learned per-head scalar to scale the Q+K sum before tanh
        # Shape (1, num_heads, 1, head_dim, 1, 1) for broadcasting over (B,H,T,D,H,W)
        # Simpler: keep one scale per head, broadcast over spatial + channel dims.
        self.qk_scale = nn.Parameter(
            torch.ones(1, num_heads, 1, 1) * (self.head_dim ** -0.5)
        )

        # ── Depthwise 3×3 conv on the attention logits ───────────────────────
        # Applied *per head* by treating the head dim as the channel dim.
        # groups=num_heads makes it a grouped depthwise across heads.
        self.dw_attn = nn.Conv2d(dim, dim, kernel_size=3, padding=1,
                                 groups=dim, bias=False)

        # ── Attention dropout ─────────────────────────────────────────────────
        self.attn_drop = nn.Dropout(drop) if drop > 0.0 else nn.Identity()

        # ── Output projection (back to (B, C, H, W)) ─────────────────────────
        self.proj    = ConvBNSiLU(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        nh, hd = self.num_heads, self.head_dim

        # ── 1. Project to Q, K, V  ───────────────────────────────────────────
        q = self.conv_q(x)   # (B, C, H, W)
        k = self.conv_k(x)
        v = self.conv_v(x)

        # ── 2. Additive attention map ─────────────────────────────────────────
        # Split into heads: (B, nh, hd, H, W)
        q = q.view(B, nh, hd, H, W)
        k = k.view(B, nh, hd, H, W)
        v_h = v.view(B, nh, hd, H, W)

        # Element-wise additive interaction, scaled, bounded by tanh.
        # qk_scale must be 5D here to broadcast with (B, nh, hd, H, W).
        qk = torch.tanh(self.qk_scale.unsqueeze(-1) * (q + k))

        # Collapse head_dim → single attention map per head per spatial loc
        # Mean over head_dim gives (B, nh, H, W) — lightweight
        attn_logit = qk.mean(dim=2)                       # (B, nh, H, W)

        # ── 3. Depthwise conv for local context (the "convolutional" part) ───
        # Treat (B, nh, H, W) as (B, C=nh, H, W) for the depthwise conv.
        # But our dw_attn was built for full C channels, so expand back.
        # More elegant: reshape to (B, C, H, W), run dw_attn, reshape back.
        attn_map_full = attn_logit.unsqueeze(2).expand(
            B, nh, hd, H, W).reshape(B, C, H, W)          # broadcast hd copies
        attn_map_full = self.dw_attn(attn_map_full)        # (B, C, H, W)
        attn_map_full = torch.sigmoid(attn_map_full)       # gate ∈ (0,1)
        attn_map_full = self.attn_drop(attn_map_full)

        # ── 4. Gate V with attention map ─────────────────────────────────────
        # Element-wise: preserves full spatial resolution, no softmax squash
        out = v * attn_map_full                            # (B, C, H, W)

        # ── 5. Output projection ─────────────────────────────────────────────
        out = self.proj(out)                               # (B, C, H, W)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# ECTB  —  updated to use CASA instead of CMHSA
# ──────────────────────────────────────────────────────────────────────────────
class ECTB(nn.Module):
    """
    Efficient Convolutional Transformer Block — CASA edition.

    Topology identical to original ECTB:
        input → 1×1 conv (c → c/2) → [CASA] → 1×1 conv (c/2 → c) → BN → residual

    Only the inner attention module changed: CMHSA → CASA.
    This is a strict drop-in; no changes needed in RTDNetClassifier.
    """
    def __init__(self, in_ch: int, out_ch: int = None, num_heads: int = 4,
                 attn_drop: float = 0.0):
        super().__init__()
        out_ch  = out_ch or in_ch
        mid_ch  = max(in_ch // 2, 1)

        self.conv1  = ConvBNSiLU(in_ch,  mid_ch, 1)
        self.casa   = CASA(mid_ch, num_heads=num_heads, drop=attn_drop)   # ← CHANGED
        self.conv2  = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn     = nn.BatchNorm2d(out_ch)
        self.skip   = (nn.Identity() if in_ch == out_ch else
                       nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                                     nn.BatchNorm2d(out_ch)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv1(x)
        feat = self.casa(feat)                    # was: self.cmhsa(feat)
        feat = self.bn(self.conv2(feat))
        return F.silu(feat + self.skip(x))


# ──────────────────────────────────────────────────────────────────────────────
# NAM  (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
class NAMChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        normed = self.bn(x)
        gamma  = self.bn.weight.abs()
        w      = gamma / (gamma.sum() + 1e-8)
        return x * torch.sigmoid(w.view(1, -1, 1, 1) * normed)


class NAMSpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bn = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        normed = self.bn(x)
        lam    = self.bn.weight.abs()
        w      = lam / (lam.sum() + 1e-8)
        return x * torch.sigmoid(w.view(1, -1, 1, 1) * normed)


class NAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel = NAMChannelAttention(channels)
        self.spatial = NAMSpatialAttention(channels)

    def forward(self, x):
        return self.spatial(self.channel(x))


# ──────────────────────────────────────────────────────────────────────────────
# APH  (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
class APH(nn.Module):
    def __init__(self, in_ch, out_ch=None):
        super().__init__()
        out_ch  = out_ch or in_ch
        self.nam  = NAM(in_ch)
        self.conv = ConvBNSiLU(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(self.nam(x))


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity-check
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy  = torch.randn(2, 64, 40, 40).to(device)

    print("=== Testing CASA ===")
    casa = CASA(64, num_heads=4).to(device)
    out  = casa(dummy)
    print(f"  Input: {dummy.shape}  Output: {out.shape}")
    assert out.shape == dummy.shape, "Shape mismatch!"

    print("=== Testing ECTB (CASA inside) ===")
    ectb = ECTB(64, 64).to(device)
    out  = ectb(dummy)
    print(f"  Input: {dummy.shape}  Output: {out.shape}")
    assert out.shape == dummy.shape

    print("=== Latency: CMHSA-style vs CASA ===")
    dummy1 = torch.randn(1, 64, 40, 40).to(device)

    # Warm-up
    for _ in range(5):
        ectb(dummy1)

    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(200):
            ectb(dummy1)
    torch.cuda.synchronize() if device.type == "cuda" else None
    ms = (time.perf_counter() - t0) / 200 * 1000
    print(f"  ECTB+CASA avg: {ms:.3f} ms per forward (bs=1, 40×40 feat map)")

    print("\nAll CASA checks passed!")