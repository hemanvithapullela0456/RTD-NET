"""
rtdnet_v2.py  —  RTD-Net V2
============================
Builds on exp2 (Slim + LiteASPP + RepLEM) with three targeted architectural
changes designed to push AID 50/50 accuracy from ~94.2% → 96–97%.

CHANGES OVER EXP2 (rtdnet_replem.py)
--------------------------------------

CHANGE A — ECTBPlus  (restored mid_ch + depthwise positional bias)
  exp2 ECTB used mid_ch = C//4 = 128  (head_dim=32, severely bottlenecked)
  V2   ECTB uses mid_ch = C//2 = 256  (head_dim=64, full attention capacity)
  + parallel depthwise 3×3 conv on mid features, mixed with attention via
    learnable scalar alpha (init 0.5). Adds local inductive bias that pure
    self-attention lacks (attention is permutation-equivariant; DW is not).
  Params: ~+328K (ECTB restoration) + negligible (DW, same mid_ch channels)
  Expected gain: +0.8–1.2%

CHANGE B — LateralFusion  (skip from lem3 /16 → post-SPPF /32)
  The stride-2 DSConv5 discards fine-grained object-level detail.
  lem3's output (/16, b*8 channels) is saved, projected to b*16, spatially
  halved with MaxPool2d(2), and added residually to the post-SPPF feature.
  Cost: b*8 × b*16 = 131K params for base_ch=32.
  Expected gain: +0.5–0.8%

CHANGE C — TwoScalePool  (replaces single AdaptiveAvgPool2d(1))
  Pools at 1×1 AND 2×2, concatenates (giving 5×C features), then projects
  back to C with a Linear layer. The 2×2 quadrant descriptors preserve
  spatial layout cues (elongation, asymmetry) that single GAP destroys —
  critical for aerial scenes like 'viaduct', 'beach', 'airport'.
  Cost: C×5 → C linear = 131K params for base_ch=32.
  Expected gain: +0.3–0.5%

CUMULATIVE EXPECTED GAIN: +1.6–2.5%  →  AID 50/50 ≈ 95.8–96.7%
PARAM BUDGET: ~2.701M (exp2) + ~590K (changes) ≈ 3.29M

REPARAMETERIZATION
------------------
Rep3x3 branches inside RepLEM are fused at inference by calling:
    model.reparameterize()
Do this AFTER training and BEFORE saving the final inference checkpoint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

class ConvBNSiLU(nn.Module):
    """Standard Conv → BN → SiLU."""
    def __init__(self, in_ch, out_ch, kernel=1, stride=1,
                 padding=None, groups=1):
        super().__init__()
        if padding is None:
            padding = kernel // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding,
                              groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DSConvBNSiLU(nn.Module):
    """Depthwise-Separable Conv → BN → SiLU  (replaces conv5)."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.dw  = nn.Conv2d(in_ch, in_ch, 3, stride, 1,
                             groups=in_ch, bias=False)
        self.pw  = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


# ─────────────────────────────────────────────────────────────────────────────
# Rep3x3 — structural reparameterization block
# ─────────────────────────────────────────────────────────────────────────────

class Rep3x3(nn.Module):
    """
    Three parallel paths during training:
        path A: 3×3 conv + BN
        path B: 1×1 conv + BN  (zero-padded to 3×3 at merge)
        path C: identity  + BN
    All outputs are summed then SiLU-activated.
    After training, call .reparameterize() to fuse into one 3×3 conv
    with zero inference overhead.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.conv3    = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn3      = nn.BatchNorm2d(channels)
        self.conv1    = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn1      = nn.BatchNorm2d(channels)
        self.bn_id    = nn.BatchNorm2d(channels)
        self._fused   = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._fused:
            return F.silu(self._fused_conv(x))
        return F.silu(self.bn3(self.conv3(x)) +
                      self.bn1(self.conv1(x)) +
                      self.bn_id(x))

    def reparameterize(self):
        if self._fused:
            return
        k3, b3 = self._fuse_bn(self.conv3, self.bn3)
        k1, b1 = self._fuse_bn_pad(self.conv1, self.bn1)
        ki, bi = self._identity_branch()
        W, B   = k3 + k1 + ki, b3 + b1 + bi
        fused  = nn.Conv2d(self.channels, self.channels, 3, 1, 1, bias=True)
        fused.weight.data = W
        fused.bias.data   = B
        self._fused_conv  = fused.to(k3.device)
        del self.conv3, self.bn3, self.conv1, self.bn1, self.bn_id
        self._fused = True

    @staticmethod
    def _fuse_bn(conv, bn):
        s = bn.weight / (bn.running_var + bn.eps).sqrt()
        return (conv.weight * s.view(-1, 1, 1, 1),
                bn.bias - bn.running_mean * s)

    @staticmethod
    def _fuse_bn_pad(conv, bn):
        w = F.pad(conv.weight, [1, 1, 1, 1])
        s = bn.weight / (bn.running_var + bn.eps).sqrt()
        return (w * s.view(-1, 1, 1, 1),
                bn.bias - bn.running_mean * s)

    def _identity_branch(self):
        C   = self.channels
        dev = self.bn_id.weight.device
        w   = torch.zeros(C, C, 3, 3, device=dev)
        for i in range(C):
            w[i, i, 1, 1] = 1.0
        s = self.bn_id.weight / (self.bn_id.running_var + self.bn_id.eps).sqrt()
        return (w * s.view(-1, 1, 1, 1),
                self.bn_id.bias - self.bn_id.running_mean * s)


# ─────────────────────────────────────────────────────────────────────────────
# RepLEM — LEM with Rep3x3 inside each branch (from exp2)
# ─────────────────────────────────────────────────────────────────────────────

class RepLEM(nn.Module):
    """LEM with Rep3x3 replacing the 3×3 ConvBNSiLU in each branch."""
    def __init__(self, in_ch: int, out_ch: int = None, C: int = 16):
        super().__init__()
        out_ch = out_ch or in_ch
        mid_ch = max(in_ch // 2,  1)
        br_ch  = max(in_ch // 32, 1)
        self.C = C
        self.conv1    = ConvBNSiLU(in_ch, mid_ch, 1)
        self.branches = nn.ModuleList([
            nn.Sequential(ConvBNSiLU(mid_ch, br_ch, 1), Rep3x3(br_ch))
            for _ in range(C)
        ])
        self.conv2 = nn.Conv2d(br_ch * C, out_ch, 1, bias=False)
        self.bn    = nn.BatchNorm2d(out_ch)
        self.skip  = (nn.Identity() if in_ch == out_ch else
                      nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                                    nn.BatchNorm2d(out_ch)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv1(x)
        out  = self.bn(self.conv2(
            torch.cat([b(feat) for b in self.branches], dim=1)))
        return F.silu(out + self.skip(x))


# ─────────────────────────────────────────────────────────────────────────────
# CMHSA — unchanged from original
# ─────────────────────────────────────────────────────────────────────────────

class CMHSA(nn.Module):
    """Convolutional Multi-Head Self-Attention."""
    def __init__(self, dim: int, num_heads: int = 4):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        T = H * W
        q = self.conv_q(x).flatten(2).view(B, self.num_heads, self.head_dim, T)
        k = self.conv_k(x).flatten(2).view(B, self.num_heads, self.head_dim, T)
        v = self.conv_v(x).flatten(2).view(B, self.num_heads, self.head_dim, T)
        attn = self.head_conv(
            torch.einsum('bhdq,bhdT->bhqT', q, k) * self.scale)
        attn = self.inst_norm(F.softmax(attn, dim=-1))
        out  = torch.einsum('bhqT,bhTd->bhqd',
                            attn, v.permute(0, 1, 3, 2)).contiguous()
        return self.proj(out.view(B, T, C)).permute(0, 2, 1).view(B, C, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# CHANGE A — ECTBPlus
# ─────────────────────────────────────────────────────────────────────────────

class ECTBPlus(nn.Module):
    """
    Efficient Convolutional Transformer Block — restored capacity + DW bias.

    vs exp2 ECTB:
        mid_ch  : C//4 (128) → C//2 (256)   restores head_dim 32 → 64
        + parallel depthwise 3×3 conv fused with attention via
          learnable scalar alpha (clamped to [0,1], init 0.5)

    The depthwise path injects local spatial structure that CMHSA lacks
    (self-attention is permutation-equivariant; DW conv is not).
    alpha is learned per-training-run, letting the model decide how much
    to rely on local vs global context.
    """
    def __init__(self, in_ch: int, out_ch: int = None, num_heads: int = 4):
        super().__init__()
        out_ch = out_ch or in_ch
        mid_ch = max(in_ch // 2, 1)          # restored from C//4 → C//2
        while mid_ch % num_heads != 0:
            num_heads = max(1, num_heads // 2)

        self.conv1  = ConvBNSiLU(in_ch, mid_ch, 1)
        self.cmhsa  = CMHSA(mid_ch, num_heads=num_heads)
        self.dw_pos = nn.Conv2d(mid_ch, mid_ch, 3, 1, 1,
                                groups=mid_ch, bias=False)
        self.bn_pos = nn.BatchNorm2d(mid_ch)
        self.alpha  = nn.Parameter(torch.tensor(0.5))   # learnable mix scalar

        self.conv2  = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn     = nn.BatchNorm2d(out_ch)
        self.skip   = (nn.Identity() if in_ch == out_ch else
                       nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                                     nn.BatchNorm2d(out_ch)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat  = self.conv1(x)
        attn  = self.cmhsa(feat)
        local = self.bn_pos(self.dw_pos(feat))
        a     = self.alpha.clamp(0., 1.)
        mixed = a * attn + (1. - a) * local
        return F.silu(self.bn(self.conv2(mixed)) + self.skip(x))


# ─────────────────────────────────────────────────────────────────────────────
# LiteASPP — from exp1, unchanged
# ─────────────────────────────────────────────────────────────────────────────

class LiteASPP(nn.Module):
    """
    Lightweight Atrous Spatial Pyramid Pooling.
    Parallel dilated 3×3 convs at rates {1, 2, 4} + global avg pool branch.
    Effective receptive fields {3, 7, 15, global} — orthogonal scales.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid = max(in_ch // 4, 1)
        self.b0  = ConvBNSiLU(in_ch, mid, 1)
        self.b1  = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(mid), nn.SiLU(inplace=True))
        self.b2  = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(mid), nn.SiLU(inplace=True))
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid), nn.SiLU(inplace=True))
        self.proj = ConvBNSiLU(mid * 4, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        g    = F.interpolate(self.gap(x), size=(h, w),
                             mode='bilinear', align_corners=False)
        return self.proj(torch.cat([self.b0(x), self.b1(x),
                                    self.b2(x), g], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# CHANGE B — LateralFusion
# ─────────────────────────────────────────────────────────────────────────────

class LateralFusion(nn.Module):
    """
    Fuses lem3 output (/16, b*8 ch) into post-SPPF feature (/32, b*16 ch).

    The /16 features still carry fine-grained object-level detail lost
    through the stride-2 DSConv. A 1×1 projection + MaxPool2d(2) aligns
    spatial size, then a residual add merges into the main stream.

    Handles odd spatial sizes robustly via adaptive_avg_pool2d fallback.
    """
    def __init__(self, skip_ch: int, main_ch: int):
        super().__init__()
        self.proj = ConvBNSiLU(skip_ch, main_ch, 1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, skip: torch.Tensor,
                main: torch.Tensor) -> torch.Tensor:
        s = self.pool(self.proj(skip))
        if s.shape[-2:] != main.shape[-2:]:
            s = F.adaptive_avg_pool2d(s, main.shape[-2:])
        return main + s


# ─────────────────────────────────────────────────────────────────────────────
# CHANGE C — TwoScalePool
# ─────────────────────────────────────────────────────────────────────────────

class TwoScalePool(nn.Module):
    """
    Replaces AdaptiveAvgPool2d(1) with dual-scale pooling.

    1×1 pool  → global descriptor           (B, C)
    2×2 pool  → 4 quadrant descriptors       (B, 4C)
    concat    → (B, 5C)
    project   → (B, C)   via Linear

    The 2×2 quadrants capture spatial asymmetries important for aerial
    scenes: beach (sea on one side), viaduct (elongated), airport (runways
    at edges), etc.  The projection keeps the FC layer unchanged.
    """
    def __init__(self, in_ch: int):
        super().__init__()
        self.proj = nn.Linear(in_ch * 5, in_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g1 = F.adaptive_avg_pool2d(x, 1).flatten(1)   # (B, C)
        g2 = F.adaptive_avg_pool2d(x, 2).flatten(1)   # (B, 4C)
        return self.proj(torch.cat([g1, g2], dim=1))   # (B, C)


# ─────────────────────────────────────────────────────────────────────────────
# NAM + APH — unchanged from original
# ─────────────────────────────────────────────────────────────────────────────

class NAMChannelAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.bn(x)
        gamma  = self.bn.weight.abs()
        w      = gamma / (gamma.sum() + 1e-8)
        return x * torch.sigmoid(w.view(1, -1, 1, 1) * normed)


class NAMSpatialAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.bn = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.bn(x)
        lam    = self.bn.weight.abs()
        w      = lam / (lam.sum() + 1e-8)
        return x * torch.sigmoid(w.view(1, -1, 1, 1) * normed)


class NAM(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channel = NAMChannelAttention(channels)
        self.spatial = NAMSpatialAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


class APH(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = None):
        super().__init__()
        out_ch   = out_ch or in_ch
        self.nam  = NAM(in_ch)
        self.conv = ConvBNSiLU(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.nam(x))


# ─────────────────────────────────────────────────────────────────────────────
# RTDNetV2 — full model
# ─────────────────────────────────────────────────────────────────────────────

class RTDNetV2(nn.Module):
    """
    RTD-Net V2  —  ~3.29 M parameters.

    Architecture stack (cumulative over baseline):
        Slim      : DSConvBNSiLU conv5, SPPF mid//4, ECTB mid//4
        exp1      : LiteASPP replaces SPPFSlim
        exp2      : RepLEM replaces LEM
        V2 (this) : ECTBPlus (mid//2 + DW bias)
                    LateralFusion (lem3 /16 skip → /32 main)
                    TwoScalePool  (1×1 + 2×2 GAP → projected)

    Args:
        num_classes (int)  : Output classes. Default 30 (AID).
        base_ch     (int)  : Base channel width. Default 32.
        num_heads   (int)  : CMHSA heads. Default 4.
        C           (int)  : LEM/RepLEM branches. Default 16.
        dropout     (float): Dropout before FC. Default 0.3.
    """
    def __init__(
        self,
        num_classes : int   = 30,
        base_ch     : int   = 32,
        num_heads   : int   = 4,
        C           : int   = 16,
        dropout     : float = 0.3,
    ):
        super().__init__()
        b = base_ch

        # ── Stem ─────────────────────────────────────────────────────────────
        self.conv1 = ConvBNSiLU(3,   b,    3, stride=2)   # /2
        self.conv2 = ConvBNSiLU(b,   b*2,  3, stride=2)   # /4

        # ── Stage 1 ──────────────────────────────────────────────────────────
        self.lem1  = RepLEM(b*2, b*2, C=C)
        self.conv3 = ConvBNSiLU(b*2, b*4, 3, stride=2)    # /8

        # ── Stage 2 ──────────────────────────────────────────────────────────
        self.lem2  = RepLEM(b*4, b*4, C=C)
        self.conv4 = ConvBNSiLU(b*4, b*8, 3, stride=2)    # /16

        # ── Stage 3 — output saved as lateral skip ────────────────────────────
        self.lem3  = RepLEM(b*8, b*8, C=C)                # /16  ← skip saved here

        # ── Downsample /16 → /32 (DSConv, slim change 1) ─────────────────────
        self.conv5 = DSConvBNSiLU(b*8, b*16, stride=2)    # /32

        # ── CHANGE A: ECTBPlus (restored mid_ch + DW positional bias) ────────
        ectb_ch    = b * 16
        safe_heads = num_heads
        while ectb_ch // 2 % safe_heads != 0:             # mid = C//2
            safe_heads = max(1, safe_heads // 2)
        self.ectb  = ECTBPlus(ectb_ch, ectb_ch, num_heads=safe_heads)

        # ── LiteASPP (exp1) ──────────────────────────────────────────────────
        self.sppf  = LiteASPP(b*16, b*16)

        # ── CHANGE B: Lateral fusion (/16 skip → /32 main) ───────────────────
        self.lateral = LateralFusion(b*8, b*16)

        # ── APH (unchanged) ──────────────────────────────────────────────────
        self.aph   = APH(b*16, b*16)

        # ── CHANGE C: Two-scale pooling ───────────────────────────────────────
        self.pool  = TwoScalePool(b*16)

        self.drop  = nn.Dropout(dropout)
        self.fc    = nn.Linear(b*16, num_classes)

        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────────
    def _init_weights(self) -> None:
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

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x    = self.conv2(self.conv1(x))
        # Stages 1–2
        x    = self.conv3(self.lem1(x))
        x    = self.conv4(self.lem2(x))
        # Stage 3 — save /16 features for lateral
        skip = self.lem3(x)               # /16, b*8 channels
        # /32 stream
        x    = self.conv5(skip)
        x    = self.sppf(self.ectb(x))
        # Fuse /16 skip into /32 main
        x    = self.lateral(skip, x)
        # Attention head
        x    = self.aph(x)
        # Two-scale classify
        return self.fc(self.drop(self.pool(x)))

    # ── Reparameterize ────────────────────────────────────────────────────────
    def reparameterize(self) -> None:
        """
        Fuse all Rep3x3 training branches into single 3×3 convs.
        Call ONCE after training, before saving the inference checkpoint.
        After this call the model is inference-equivalent but smaller/faster.
        """
        fused_count = 0
        for m in self.modules():
            if isinstance(m, Rep3x3):
                m.reparameterize()
                fused_count += 1
        print(f'Rep3x3 branches fused ({fused_count} blocks). '
              'Model is now inference-optimised.')

    # ── Utilities ─────────────────────────────────────────────────────────────
    def count_parameters(self) -> tuple[int, int]:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        return total, trainable

    def per_module_params(self) -> dict:
        return {
            name: sum(p.numel() for p in mod.parameters())
            for name, mod in self.named_children()
        }


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import time

    EXP2_PARAMS = 2_701_000   # approximate exp2 baseline

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    model = RTDNetV2(num_classes=30, base_ch=32).to(device)
    total, trainable = model.count_parameters()

    print('── Per-module breakdown ──────────────────────────────────')
    for name, p in model.per_module_params().items():
        pct = p / total * 100
        print(f'  {name:<12}  {p:>9,}  ({pct:5.1f}%)')

    print(f'\n── Summary ───────────────────────────────────────────────')
    print(f'  exp2 (RepLEM+LiteASPP)  : ~{EXP2_PARAMS:>9,}  (2.701 M)')
    print(f'  RTDNet-V2               :  {total:>9,}  ({total/1e6:.3f} M)')
    print(f'  Delta                   :  {total-EXP2_PARAMS:>+9,}')
    print(f'  Model size MB           :  {total*4/1024**2:.2f}')

    # Forward pass sanity
    for res in [224, 640]:
        dummy = torch.randn(2, 3, res, res).to(device)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (2, 30), f'Shape mismatch at {res}px'
        print(f'\n  {res}×{res}  →  {out.shape}  ✓')

    # Throughput
    model.eval()
    dummy1 = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(5):
            model(dummy1)
        t0 = time.perf_counter()
        for _ in range(100):
            model(dummy1)
    ms = (time.perf_counter() - t0) / 100 * 1000
    print(f'\n  Avg latency (224px, bs=1): {ms:.1f} ms  (~{1000/ms:.0f} FPS)')

    # Reparameterize test
    model.reparameterize()
    total_fused, _ = model.count_parameters()
    print(f'\n  Params after reparameterize: {total_fused:,}  ({total_fused/1e6:.3f} M)')
    with torch.no_grad():
        out2 = model(dummy1)
    print(f'  Post-fuse forward OK: {out2.shape}  ✓')
    print('\nAll checks passed!')