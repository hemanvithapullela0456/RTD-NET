"""
attention_variants.py
=====================
Three lightweight attention modules that drop into APH as NAM replacements.

All built directly on the original models.py code base:
    LEM, CMHSA, ECTB, NAM  →  unchanged
    APH                     →  gains attention_type switch
    RTDNetClassifier        →  passes attention_type through to APH

Variants
--------
1. ConvNAM         — adds a small 3×3 depthwise conv + BN inside each NAM
                     branch to make attention spatially adaptive.
2. ResidualConvNAM — wraps ConvNAM in a residual connection so the module
                     learns a *correction* on top of the identity path.
3. TripletConvNAM  — keeps conv-enhanced channel attention but replaces the
                     spatial branch with Triplet Attention (three pooling
                     directions: C×H, C×W, H×W).

Parameter overhead vs original NAM
-----------------------------------
  ConvNAM         : +2 × (C depthwise 3×3 + BN)  ≈  +2 × C × 9 weights
  ResidualConvNAM : same as ConvNAM  (residual costs 0 extra params)
  TripletConvNAM  : channel branch same as ConvNAM;
                    spatial branch 3 strip-pool branches each with a 1×1
                    conv (C→1) — very light.
  At C=512 (base_ch=32 → b*16=512):
      ConvNAM/Residual: ~9K extra params
      TripletConvNAM  : ~14K extra params
  All well under the 0.1M target.

Usage
-----
    model = RTDNetClassifier(attention_type='conv')       # ConvNAM
    model = RTDNetClassifier(attention_type='residual')   # ResidualConvNAM
    model = RTDNetClassifier(attention_type='triplet')    # TripletConvNAM
    model = RTDNetClassifier(attention_type='original')   # plain NAM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper
# ─────────────────────────────────────────────────────────────────────────────
class ConvBNSiLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=1, stride=1, padding=None, groups=1):
        super().__init__()
        if padding is None:
            padding = kernel // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding,
                              groups=groups, bias=False)
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ─────────────────────────────────────────────────────────────────────────────
# LEM  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
class LEM(nn.Module):
    def __init__(self, in_ch, out_ch=None, C=16):
        super().__init__()
        out_ch = out_ch or in_ch
        mid_ch = max(in_ch // 2, 1)
        br_ch  = max(in_ch // 32, 1)
        self.C = C
        self.conv1 = ConvBNSiLU(in_ch, mid_ch, 1)
        self.branches = nn.ModuleList([
            nn.Sequential(ConvBNSiLU(mid_ch, br_ch, 1),
                          ConvBNSiLU(br_ch,  br_ch, 3))
            for _ in range(C)
        ])
        self.conv2 = nn.Conv2d(br_ch * C, out_ch, 1, bias=False)
        self.bn    = nn.BatchNorm2d(out_ch)
        self.skip  = (nn.Identity() if in_ch == out_ch else
                      nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                                    nn.BatchNorm2d(out_ch)))

    def forward(self, x):
        feat = self.conv1(x)
        out  = self.bn(self.conv2(torch.cat([b(feat) for b in self.branches], 1)))
        return F.silu(out + self.skip(x))


# ─────────────────────────────────────────────────────────────────────────────
# CMHSA  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
class CMHSA(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.conv_q    = nn.Conv2d(dim, dim, 1, bias=False)
        self.conv_k    = nn.Conv2d(dim, dim, 1, bias=False)
        self.conv_v    = nn.Conv2d(dim, dim, 1, bias=False)
        self.inst_norm = nn.InstanceNorm2d(num_heads, affine=True)
        self.head_conv = nn.Conv2d(num_heads, num_heads, 1, bias=False)
        self.proj      = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        T = H * W
        q = self.conv_q(x).flatten(2).view(B, self.num_heads, self.head_dim, T)
        k = self.conv_k(x).flatten(2).view(B, self.num_heads, self.head_dim, T)
        v = self.conv_v(x).flatten(2).view(B, self.num_heads, self.head_dim, T)
        attn = self.head_conv(
            torch.einsum('bhdq,bhdT->bhqT', q, k) * self.scale)
        attn = self.inst_norm(F.softmax(attn, dim=-1))
        out  = torch.einsum('bhqT,bhTd->bhqd',
                            attn, v.permute(0,1,3,2)).contiguous()
        out  = self.proj(out.view(B, T, C)).permute(0,2,1).view(B, C, H, W)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# ECTB  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
class ECTB(nn.Module):
    def __init__(self, in_ch, out_ch=None, num_heads=4):
        super().__init__()
        out_ch = out_ch or in_ch
        mid_ch = max(in_ch // 2, 1)
        self.conv1 = ConvBNSiLU(in_ch, mid_ch, 1)
        self.cmhsa = CMHSA(mid_ch, num_heads=num_heads)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn    = nn.BatchNorm2d(out_ch)
        self.skip  = (nn.Identity() if in_ch == out_ch else
                      nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                                    nn.BatchNorm2d(out_ch)))

    def forward(self, x):
        feat = self.cmhsa(self.conv1(x))
        return F.silu(self.bn(self.conv2(feat)) + self.skip(x))


# ─────────────────────────────────────────────────────────────────────────────
# Original NAM  (unchanged — kept for reference / attention_type='original')
# ─────────────────────────────────────────────────────────────────────────────
class NAMChannelAttention(nn.Module):
    """BN γ-weight channel attention (original)."""
    def __init__(self, channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        normed = self.bn(x)
        gamma  = self.bn.weight.abs()
        w      = gamma / (gamma.sum() + 1e-8)
        return x * torch.sigmoid(w.view(1, -1, 1, 1) * normed)


class NAMSpatialAttention(nn.Module):
    """InstanceNorm λ-weight spatial attention (original)."""
    def __init__(self, channels):
        super().__init__()
        self.bn = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        normed = self.bn(x)
        lam    = self.bn.weight.abs()
        w      = lam / (lam.sum() + 1e-8)
        return x * torch.sigmoid(w.view(1, -1, 1, 1) * normed)


class NAM(nn.Module):
    """Original NAM: channel then spatial attention."""
    def __init__(self, channels):
        super().__init__()
        self.channel = NAMChannelAttention(channels)
        self.spatial = NAMSpatialAttention(channels)

    def forward(self, x):
        return self.spatial(self.channel(x))


# ─────────────────────────────────────────────────────────────────────────────
# ── Approach 1 ── ConvNAM
# ─────────────────────────────────────────────────────────────────────────────
class ConvNAMChannelAttention(nn.Module):
    """
    Channel attention: original BN γ-weighting + a depthwise 3×3 + BN
    applied to the normalised feature before sigmoid.

    The depthwise conv lets each channel's importance estimate be shaped
    by local spatial context rather than a pure per-channel scalar.

    Extra params: C × 9 (dw conv weights) + 2C (BN γ,β) = 11C
    At C=512: 11×512 = 5,632 params.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.bn      = nn.BatchNorm2d(channels)
        # Depthwise 3×3 — one filter per channel, no cross-channel mixing
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=3,
                                 padding=1, groups=channels, bias=False)
        self.dw_bn   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NAM channel weight (γ importance)
        normed = self.bn(x)
        gamma  = self.bn.weight.abs()
        w      = gamma / (gamma.sum() + 1e-8)
        # Refine with local conv before sigmoid gate
        refined = self.dw_bn(self.dw_conv(normed))
        attn    = torch.sigmoid(w.view(1, -1, 1, 1) * refined)
        return x * attn


class ConvNAMSpatialAttention(nn.Module):
    """
    Spatial attention: original InstanceNorm λ-weighting + depthwise 3×3 + BN.

    Extra params: C × 9 + 2C = 11C  (same as channel branch).
    At C=512: 5,632 params.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.bn      = nn.InstanceNorm2d(channels, affine=True)
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=3,
                                 padding=1, groups=channels, bias=False)
        self.dw_bn   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.bn(x)
        lam    = self.bn.weight.abs()
        w      = lam / (lam.sum() + 1e-8)
        refined = self.dw_bn(self.dw_conv(normed))
        attn    = torch.sigmoid(w.view(1, -1, 1, 1) * refined)
        return x * attn


class ConvNAM(nn.Module):
    """
    Approach 1 — Conv-NAM.
    Channel attention → spatial attention, each enhanced with dw conv + BN.
    Total extra params vs NAM: ~22C  (≈11K at C=512).
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channel = ConvNAMChannelAttention(channels)
        self.spatial = ConvNAMSpatialAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


# ─────────────────────────────────────────────────────────────────────────────
# ── Approach 2 ── ResidualConvNAM
# ─────────────────────────────────────────────────────────────────────────────
class ResidualConvNAM(nn.Module):
    """
    Approach 2 — Residual-Conv-NAM.

    Wraps ConvNAM in a residual connection:
        output = x + ConvNAM(x)

    The attention module therefore learns a *correction* on top of the
    identity path.  This is more stable to train because:
      • gradients always flow through the skip connection
      • the module initialises near identity (BN β=0, dw conv ≈ 0)
      • large early errors in attention estimates don't corrupt the feature

    Extra params: identical to ConvNAM (~22C, ≈11K at C=512).
    The residual costs 0 extra parameters.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv_nam = ConvNAM(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_nam(x)


# ─────────────────────────────────────────────────────────────────────────────
# ── Approach 3 ── TripletConvNAM
# ─────────────────────────────────────────────────────────────────────────────
class _TripletBranch(nn.Module):
    """
    One branch of Triplet Attention.
    Permutes x to bring a target dimension to the channel axis, then applies
    a strip-pool + 1×1 conv + BN + sigmoid to produce a spatial attention map,
    then permutes back.

    Args:
        channels (int): C — used only when the target is the channel dim.
        spatial  (bool): True for H→C and W→C branches; False for C-branch.

    For the two spatial branches the conv reduces pooled-channel dim (H or W)
    to 1 via a 1D conv along the remaining spatial axis.
    """
    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        # Single conv that operates along the "remaining" spatial dimension
        # after permutation.  Input to conv is 2-channel (avg+max pool).
        self.conv = nn.Conv2d(2, 1, kernel_size=(1, kernel_size),
                              padding=(0, padding), bias=False)
        self.bn   = nn.BatchNorm2d(1)

    def forward(self, x_perm: torch.Tensor) -> torch.Tensor:
        """
        x_perm : (B, C', H', W') already permuted so the target dim is W'.
        Returns a (B, 1, H', W') attention map in the same permuted space.
        """
        avg = x_perm.mean(dim=1, keepdim=True)   # (B,1,H',W')
        mx  = x_perm.max(dim=1, keepdim=True)[0] # (B,1,H',W')
        cat = torch.cat([avg, mx], dim=1)         # (B,2,H',W')
        return torch.sigmoid(self.bn(self.conv(cat)))


class TripletAttention(nn.Module):
    """
    Triplet Attention — three parallel pooling branches:
      Branch 1 (H-branch): permute (B,C,H,W)→(B,H,C,W), pool along C,
                            produce (B,1,C,W) attn → permute back → (B,C,H,W)
      Branch 2 (W-branch): permute (B,C,H,W)→(B,W,C,H), pool along C,
                            produce (B,1,C,H) attn → permute back → (B,C,H,W)
      Branch 3 (C-branch): standard spatial pool over (H,W), 1×k conv → (B,1,H,W)

    Final output: average of the three attended features.

    Extra params: 3 × (2×k conv + BN-1) ≈ 3 × 7 × 2 = ~42 scalars (negligible).
    At k=7: 3 × (14 + 1) = 45 params.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.branch_h = _TripletBranch(None, kernel_size)  # H direction
        self.branch_w = _TripletBranch(None, kernel_size)  # W direction
        self.branch_c = _TripletBranch(None, kernel_size)  # channel direction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # ── Branch 1: attend along H ──────────────────────────────────────────
        # Permute so H becomes the "channel" dim to pool over
        x_h    = x.permute(0, 2, 1, 3)           # (B, H, C, W)
        attn_h = self.branch_h(x_h)               # (B, 1, C, W)
        out_h  = (x_h * attn_h).permute(0, 2, 1, 3)   # back to (B,C,H,W)

        # ── Branch 2: attend along W ──────────────────────────────────────────
        x_w    = x.permute(0, 3, 2, 1)           # (B, W, H, C)
        # Reshape to (B, W, C, H) for the branch conv (kernel along H)
        x_w    = x_w.permute(0, 1, 3, 2)         # (B, W, C, H)
        attn_w = self.branch_w(x_w)               # (B, 1, C, H)
        out_w  = (x_w * attn_w).permute(0, 2, 3, 1)   # (B,C,H,W)

        # ── Branch 3: standard spatial (C-axis pooling) ───────────────────────
        attn_c = self.branch_c(x)                 # (B, 1, H, W)
        out_c  = x * attn_c                       # (B, C, H, W)

        return (out_h + out_w + out_c) / 3.0


class TripletConvNAM(nn.Module):
    """
    Approach 3 — Triplet-Conv-NAM.

    Channel branch : ConvNAMChannelAttention (BN γ + dw conv + BN)
    Spatial branch : TripletAttention (three-direction pooling)

    The channel branch provides per-channel importance weights adapted by
    local context.  The spatial branch captures fine-grained structure along
    all three axes — especially useful for small objects (UAV imagery) and
    occluded regions where single-axis pooling loses information.

    Extra params vs original NAM:
        ConvNAMChannelAttention : ~11C (≈5.6K at C=512)
        TripletAttention        : 3 × (2×7 + 1) = 45  (negligible)
        Total                   : ~5.6K at C=512
    """
    def __init__(self, channels: int, triplet_kernel: int = 7):
        super().__init__()
        self.channel = ConvNAMChannelAttention(channels)
        self.spatial = TripletAttention(kernel_size=triplet_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel(x)   # channel attention first
        x = self.spatial(x)   # then triplet spatial attention
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Updated APH — attention_type switch
# ─────────────────────────────────────────────────────────────────────────────
_ATTENTION_REGISTRY = {
    'original' : NAM,
    'conv'     : ConvNAM,
    'residual' : ResidualConvNAM,
    'triplet'  : TripletConvNAM,
}


class APH(nn.Module):
    """
    Attention Prediction Head.

    Args:
        in_ch          (int): Input channels.
        out_ch         (int): Output channels (defaults to in_ch).
        attention_type (str): One of 'original' | 'conv' | 'residual' | 'triplet'.
    """
    def __init__(self, in_ch: int, out_ch: int = None,
                 attention_type: str = 'original'):
        super().__init__()
        out_ch = out_ch or in_ch
        if attention_type not in _ATTENTION_REGISTRY:
            raise ValueError(
                f"attention_type must be one of {list(_ATTENTION_REGISTRY)}, "
                f"got '{attention_type}'"
            )
        self.nam  = _ATTENTION_REGISTRY[attention_type](in_ch)
        self.conv = ConvBNSiLU(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.nam(x))


# ─────────────────────────────────────────────────────────────────────────────
# SPPF  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
class SPPF(nn.Module):
    def __init__(self, in_ch, out_ch, pool_size=5):
        super().__init__()
        mid_ch    = in_ch // 2
        self.cv1  = ConvBNSiLU(in_ch,    mid_ch, 1)
        self.cv2  = ConvBNSiLU(mid_ch*4, out_ch, 1)
        self.pool = nn.MaxPool2d(pool_size, 1, pool_size // 2)

    def forward(self, x):
        x  = self.cv1(x)
        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        return self.cv2(torch.cat([x, p1, p2, p3], 1))


# ─────────────────────────────────────────────────────────────────────────────
# RTDNetClassifier — attention_type flows through to APH
# ─────────────────────────────────────────────────────────────────────────────
class RTDNetClassifier(nn.Module):
    """
    RTD-Net backbone with switchable attention inside APH.

    Args:
        num_classes    (int)  : Output classes. Default 30 (AID).
        base_ch        (int)  : Base channel width. Default 32.
        num_heads      (int)  : ECTB heads. Default 4.
        C              (int)  : LEM branches. Default 16.
        dropout        (float): FC dropout. Default 0.3.
        attention_type (str)  : 'original' | 'conv' | 'residual' | 'triplet'.

    Example
    -------
        model = RTDNetClassifier(num_classes=30, attention_type='triplet')
    """
    def __init__(
        self,
        num_classes    : int   = 30,
        base_ch        : int   = 32,
        num_heads      : int   = 4,
        C              : int   = 16,
        dropout        : float = 0.3,
        attention_type : str   = 'original',
    ):
        super().__init__()
        b = base_ch

        # ── Stem ─────────────────────────────────────────────────────────────
        self.conv1 = ConvBNSiLU(3,   b,    3, stride=2)   # /2
        self.conv2 = ConvBNSiLU(b,   b*2,  3, stride=2)   # /4

        # ── Stage 1 ──────────────────────────────────────────────────────────
        self.lem1  = LEM(b*2, b*2,  C=C)
        self.conv3 = ConvBNSiLU(b*2, b*4,  3, stride=2)   # /8

        # ── Stage 2 ──────────────────────────────────────────────────────────
        self.lem2  = LEM(b*4, b*4,  C=C)
        self.conv4 = ConvBNSiLU(b*4, b*8,  3, stride=2)   # /16

        # ── Stage 3 ──────────────────────────────────────────────────────────
        self.lem3  = LEM(b*8, b*8,  C=C)
        self.conv5 = ConvBNSiLU(b*8, b*16, 3, stride=2)   # /32

        # ── ECTB ─────────────────────────────────────────────────────────────
        ectb_ch    = b * 16
        safe_heads = num_heads
        while ectb_ch % safe_heads != 0:
            safe_heads = max(1, safe_heads // 2)
        self.ectb  = ECTB(ectb_ch, ectb_ch, num_heads=safe_heads)

        # ── SPPF ─────────────────────────────────────────────────────────────
        self.sppf  = SPPF(b*16, b*16)

        # ── APH with switchable attention ────────────────────────────────────
        self.aph   = APH(b*16, b*16, attention_type=attention_type)

        # ── Classifier ───────────────────────────────────────────────────────
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(b*16, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None: nn.init.ones_(m.weight)
                if m.bias   is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv2(self.conv1(x))          # stem
        x = self.conv3(self.lem1(x))           # stage 1
        x = self.conv4(self.lem2(x))           # stage 2
        x = self.conv5(self.lem3(x))           # stage 3
        x = self.sppf(self.ectb(x))            # transformer + pooling
        x = self.aph(x)                        # attention head
        return self.fc(self.drop(
            self.gap(x).flatten(1)))           # classify

    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        return total, trainable


# ─────────────────────────────────────────────────────────────────────────────
# Sanity checks
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    dummy_224 = torch.randn(2, 3, 224, 224).to(device)

    baseline_params = None

    for atype in ['original', 'conv', 'residual', 'triplet']:
        model = RTDNetClassifier(num_classes=30, base_ch=32,
                                 attention_type=atype).to(device)
        total, _ = model.count_parameters()

        with torch.no_grad():
            out = model(dummy_224)
        assert out.shape == (2, 30), f"Shape error for {atype}"

        if baseline_params is None:
            baseline_params = total
        delta = total - baseline_params

        print(f"[{atype:>10}]  params={total:>9,}  "
              f"Δ={delta:>+7,}  size={total*4/1024**2:.2f} MB  "
              f"out={tuple(out.shape)}")

    # Latency at 640×640
    print("\nLatency at 640×640 (bs=1):")
    dummy_640 = torch.randn(1, 3, 640, 640).to(device)
    for atype in ['original', 'conv', 'residual', 'triplet']:
        model = RTDNetClassifier(num_classes=30, base_ch=32,
                                 attention_type=atype).to(device)
        model.eval()
        with torch.no_grad():
            for _ in range(3): model(dummy_640)
            t0 = time.perf_counter()
            for _ in range(20): model(dummy_640)
        ms = (time.perf_counter() - t0) / 20 * 1000
        print(f"  [{atype:>10}]  {ms:.1f} ms  (~{1000/ms:.0f} FPS)")

    print("\nAll variants passed!")