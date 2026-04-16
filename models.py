"""
models.py — Core building blocks of RTD-Net
Paper: "Real-Time Object Detection Network in UAV-Vision Based on CNN and Transformer"
Ye et al., IEEE Transactions on Instrumentation and Measurement, Vol. 72, 2023

Modules implemented:
  - LEM   : Lightweight Feature Extraction Module (homogeneous multibranch)
  - CMHSA : Convolutional Multi-Head Self-Attention
  - ECTB  : Efficient Convolutional Transformer Block
  - NAM   : Normalization-based Attention Module (channel + spatial)
  - APH   : Attention Prediction Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# Helper: standard Conv-BN-SiLU block
# ---------------------------------------------------------------------------
class ConvBNSiLU(nn.Module):
    """Standard Conv → BN → SiLU block used throughout the network."""
    def __init__(self, in_ch, out_ch, kernel=1, stride=1, padding=None, groups=1):
        super().__init__()
        if padding is None:
            padding = kernel // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding, groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ---------------------------------------------------------------------------
# LEM — Lightweight Feature Extraction Module  (Section III-B, Eq. 1)
# ---------------------------------------------------------------------------
class LEM(nn.Module):
    """
    Homogeneous multi-branch architecture.
    Architecture:
        input → conv1 (1×1, c→c/2)
              → [C branches, each: 1×1 (c/2 → c/32) + 3×3 (c/32 → c/32)]
              → concatenate  → conv2 (1×1, c/32*C → c) → BN
    C = 16 as stated in the paper (total branch output = c/2 when C=16, c/32*16=c/2).
    We keep a residual skip from input.
    """
    def __init__(self, in_ch, out_ch=None, C=16):
        super().__init__()
        out_ch  = out_ch or in_ch
        mid_ch  = max(in_ch // 2, 1)           # c/2
        br_ch   = max(in_ch // 32, 1)           # c/32 per branch
        self.C  = C

        # First 1×1 to project to mid_ch
        self.conv1 = ConvBNSiLU(in_ch, mid_ch, 1)

        # C homogeneous branches (1×1 + 3×3)
        self.branches = nn.ModuleList([
            nn.Sequential(
                ConvBNSiLU(mid_ch, br_ch, 1),
                ConvBNSiLU(br_ch,  br_ch, 3),
            )
            for _ in range(C)
        ])

        branch_total = br_ch * C   # should equal mid_ch when br_ch=in_ch/32, C=16
        self.conv2 = nn.Conv2d(branch_total, out_ch, 1, bias=False)
        self.bn    = nn.BatchNorm2d(out_ch)

        # Residual: project input if channels differ
        self.skip = nn.Identity() if in_ch == out_ch else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        feat   = self.conv1(x)
        branch_outs = [b(feat) for b in self.branches]
        concat = torch.cat(branch_outs, dim=1)
        out    = self.bn(self.conv2(concat))
        return F.silu(out + self.skip(x))


# ---------------------------------------------------------------------------
# CMHSA — Convolutional Multi-Head Self-Attention  (Section III-C, Eq. 2-3)
# ---------------------------------------------------------------------------
class CMHSA(nn.Module):
    """
    Replaces the linear Q/K/V projections in standard MHSA with 1×1 convolutions,
    then flattens tokens for attention.  Adds instance normalisation after Softmax
    (Eq. 3) and reprojects the result with a linear layer back to 2-D.
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads  = num_heads
        self.head_dim   = dim // num_heads
        self.scale      = self.head_dim ** -0.5

        # Convolutional projections for Q, K, V  (kernel s=1 as stated in paper)
        self.conv_q = nn.Conv2d(dim, dim, 1, bias=False)
        self.conv_k = nn.Conv2d(dim, dim, 1, bias=False)
        self.conv_v = nn.Conv2d(dim, dim, 1, bias=False)

        # Instance normalisation applied after softmax (Eq. 3)
        self.inst_norm = nn.InstanceNorm2d(num_heads, affine=True)

        # Conv between heads to improve inter-head interaction (paper Eq. 3)
        self.head_conv = nn.Conv2d(num_heads, num_heads, 1, bias=False)

        # Output projection
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        T = H * W   # number of tokens

        # Convolutional projection + flatten  (Eq. 2)
        q = self.conv_q(x).flatten(2)  # (B, C, T)
        k = self.conv_k(x).flatten(2)
        v = self.conv_v(x).flatten(2)

        # Split into heads: (B, heads, head_dim, T)
        q = q.view(B, self.num_heads, self.head_dim, T)
        k = k.view(B, self.num_heads, self.head_dim, T)
        v = v.view(B, self.num_heads, self.head_dim, T)

        # Scaled dot-product:  attn shape (B, heads, T, T)
        attn = torch.einsum('bhdT,bhdt->bhdT', q, k) * self.scale
        # Note: we need (B, heads, T, T)
        attn = torch.einsum('bhdq,bhdT->bhqT', q, k) * self.scale

        # Conv between heads (Eq. 3: Conv(QK^T / sqrt(c/k)))
        # We treat the "heads" dim as channels for the 1×1 conv
        # attn: (B, heads, T, T) — apply conv across heads per (q, T) pair
        # For tractability: apply head_conv on attention logits per query position
        attn_conv = self.head_conv(attn.view(B, self.num_heads, T, T))  # (B, heads, T, T)

        # Softmax + instance normalisation (Eq. 3)
        attn_soft = F.softmax(attn_conv, dim=-1)               # (B, heads, T, T)
        # IN expects (B, C, H, W); treat T as H, T as W
        attn_norm = self.inst_norm(attn_soft)                   # (B, heads, T, T)

        # Weighted sum of V
        # v: (B, heads, head_dim, T) → need (B, heads, T, head_dim)
        v_t  = v.permute(0, 1, 3, 2)                           # (B, heads, T, head_dim)
        out  = torch.einsum('bhqT,bhTd->bhqd', attn_norm, v_t) # (B, heads, T, head_dim)

        # Concatenate heads → (B, T, C)
        out  = out.contiguous().view(B, T, C)

        # Linear projection back to C
        out  = self.proj(out)   # (B, T, C)

        # Reshape to 2-D feature map
        out  = out.permute(0, 2, 1).view(B, C, H, W)
        return out


# ---------------------------------------------------------------------------
# ECTB — Efficient Convolutional Transformer Block  (Section III-C, Fig. 3c)
# ---------------------------------------------------------------------------
class ECTB(nn.Module):
    """
    Same topology as LEM but replaces the homogeneous multibranch architecture
    with a single CMHSA self-attention branch.
    Structure:
        input → 1×1Conv (c → c/2) → CMHSA → 1×1Conv (c/2 → c) → BN → residual
    """
    def __init__(self, in_ch, out_ch=None, num_heads=4):
        super().__init__()
        out_ch  = out_ch or in_ch
        mid_ch  = max(in_ch // 2, 1)

        self.conv1  = ConvBNSiLU(in_ch,  mid_ch, 1)
        self.cmhsa  = CMHSA(mid_ch, num_heads=num_heads)
        self.conv2  = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn     = nn.BatchNorm2d(out_ch)

        self.skip = nn.Identity() if in_ch == out_ch else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.cmhsa(feat)
        feat = self.bn(self.conv2(feat))
        return F.silu(feat + self.skip(x))


# ---------------------------------------------------------------------------
# NAM — Normalization-based Attention Module  (Section III-E, Eq. 7-9)
# ---------------------------------------------------------------------------
class NAMChannelAttention(nn.Module):
    """
    Channel attention using BN scale factor γ to reflect channel importance.
    Mc = sigmoid(W_γ · BN(F))     (Eq. 8)
    """
    def __init__(self, channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        # BN computes γ (weight) and β (bias) for each channel
        normed = self.bn(x)
        # Importance weights W_γ = γ_i / Σ γ_j  (normalise across channels)
        gamma  = self.bn.weight.abs()                    # (C,)
        w      = gamma / (gamma.sum() + 1e-8)            # (C,)
        w      = w.view(1, -1, 1, 1)
        attn   = torch.sigmoid(w * normed)
        return x * attn


class NAMSpatialAttention(nn.Module):
    """
    Spatial attention using pixel-level normalisation.
    Ms = sigmoid(W_λ · BN_s(F))   (Eq. 9)
    """
    def __init__(self, channels):
        super().__init__()
        # InstanceNorm acts as pixel-level normalisation
        self.bn = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        normed = self.bn(x)
        # Importance weights over spatial channels
        lam    = self.bn.weight.abs()                    # (C,)
        w      = lam / (lam.sum() + 1e-8)
        w      = w.view(1, -1, 1, 1)
        attn   = torch.sigmoid(w * normed)
        return x * attn


class NAM(nn.Module):
    """
    Full NAM: channel attention followed by spatial attention (CBAM-style integration).
    """
    def __init__(self, channels):
        super().__init__()
        self.channel = NAMChannelAttention(channels)
        self.spatial = NAMSpatialAttention(channels)

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x


# ---------------------------------------------------------------------------
# APH — Attention Prediction Head  (Section III-E)
# ---------------------------------------------------------------------------
class APH(nn.Module):
    """
    Attention Prediction Head: NAM attention applied before the final prediction.
    Infers channel + spatial attention maps, multiplies them with the feature map,
    then applies a small conv to produce the output feature.
    """
    def __init__(self, in_ch, out_ch=None):
        super().__init__()
        out_ch = out_ch or in_ch
        self.nam  = NAM(in_ch)
        self.conv = ConvBNSiLU(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.nam(x)
        x = self.conv(x)
        return x


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy  = torch.randn(2, 64, 40, 40).to(device)

    print("=== Testing LEM ===")
    lem = LEM(64, 64).to(device)
    print(f"  Input: {dummy.shape}  Output: {lem(dummy).shape}")

    print("=== Testing CMHSA ===")
    attn = CMHSA(64, num_heads=4).to(device)
    print(f"  Input: {dummy.shape}  Output: {attn(dummy).shape}")

    print("=== Testing ECTB ===")
    ectb = ECTB(64, 64).to(device)
    print(f"  Input: {dummy.shape}  Output: {ectb(dummy).shape}")

    print("=== Testing NAM ===")
    nam  = NAM(64).to(device)
    print(f"  Input: {dummy.shape}  Output: {nam(dummy).shape}")

    print("=== Testing APH ===")
    aph  = APH(64, 128).to(device)
    print(f"  Input: {dummy.shape}  Output: {aph(dummy).shape}")

    print("\nAll modules passed!")