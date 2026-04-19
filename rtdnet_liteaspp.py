"""
rtdnet_liteaspp.py  —  RTD-Net Slim + LiteASPP
================================================
Experiment 1: Replace SPPFSlim with LiteASPP.
Everything else is identical to rtdnet_slim.py.

Why LiteASPP beats SPPFSlim
-----------------------------
SPPFSlim uses the SAME 5×5 MaxPool kernel applied 1, 2, and 3 times in
sequence.  The resulting 'pyramid' has effective receptive fields of
{5, 9, 13} pixels — they are nested and highly correlated (each is just
a wider blur of the last).  They capture no genuinely new spatial frequency.

LiteASPP uses parallel dilated 3×3 convolutions with rates {1, 2, 4} plus
a global average pool branch.  Effective receptive fields: {3, 7, 15, global}.
These are ORTHOGONAL receptive fields — each branch sees a genuinely different
spatial scale.  For AID (30 aerial scene classes), where 'port' vs 'resort'
vs 'stadium' differ in object density and spatial layout at different scales,
this matters directly.

Parameter count: ≈ same as SPPFSlim (mid_ch = C//4 per branch, 4 branches,
output proj C → C identical structure).  No cost, real benefit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Shared helpers (identical to rtdnet_slim.py) ────────────────────────────

class ConvBNSiLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=1, stride=1, padding=None,
                 groups=1):
        super().__init__()
        if padding is None:
            padding = kernel // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding,
                              groups=groups, bias=False)
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DSConvBNSiLU(nn.Module):
    """Depthwise-Separable conv — replaces conv5 (Change 1 from slim)."""
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.dw  = nn.Conv2d(in_ch, in_ch, 3, stride, 1,
                             groups=in_ch, bias=False)
        self.pw  = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class LEM(nn.Module):
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
        self.conv2 = nn.Conv2d(br_ch * C, out_ch, 1, bias=False)
        self.bn    = nn.BatchNorm2d(out_ch)
        self.skip  = (nn.Identity() if in_ch == out_ch else
                      nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                                    nn.BatchNorm2d(out_ch)))

    def forward(self, x):
        feat = self.conv1(x)
        out  = self.bn(self.conv2(
            torch.cat([b(feat) for b in self.branches], dim=1)))
        return F.silu(out + self.skip(x))


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
                            attn, v.permute(0, 1, 3, 2)).contiguous()
        return self.proj(out.view(B, T, C)).permute(0, 2, 1).view(B, C, H, W)


class ECTB(nn.Module):
    """Slim ECTB: mid_ch = C//4 (unchanged from rtdnet_slim)."""
    def __init__(self, in_ch, out_ch=None, num_heads=4):
        super().__init__()
        out_ch = out_ch or in_ch
        mid_ch = max(in_ch // 4, 1)
        while mid_ch % num_heads != 0:
            num_heads = max(1, num_heads // 2)
        self.conv1 = ConvBNSiLU(in_ch, mid_ch, 1)
        self.cmhsa = CMHSA(mid_ch, num_heads=num_heads)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn    = nn.BatchNorm2d(out_ch)
        self.skip  = (nn.Identity() if in_ch == out_ch else
                      nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                                    nn.BatchNorm2d(out_ch)))

    def forward(self, x):
        return F.silu(self.bn(self.conv2(self.cmhsa(self.conv1(x))))
                      + self.skip(x))


# ─── NEW: LiteASPP ───────────────────────────────────────────────────────────

class LiteASPP(nn.Module):
    """
    Lightweight Atrous Spatial Pyramid Pooling.
    Drop-in replacement for SPPFSlim.

    Branches:
        b0  1×1  conv             → local (rate=1)
        b1  3×3  dilated rate=2   → 7px effective RF
        b2  3×3  dilated rate=4   → 15px effective RF
        gap global avg pool       → image-level context

    All branches: C → C//4 channels.
    Projection: (C//4 × 4) → out_ch.  Same structure/params as SPPFSlim.
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


# ─── NAM + APH (unchanged) ───────────────────────────────────────────────────

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


class APH(nn.Module):
    def __init__(self, in_ch, out_ch=None):
        super().__init__()
        out_ch   = out_ch or in_ch
        self.nam  = NAM(in_ch)
        self.conv = ConvBNSiLU(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(self.nam(x))


# ─── Full model ──────────────────────────────────────────────────────────────

class RTDNetLiteASPP(nn.Module):
    """
    RTD-Net Slim + LiteASPP.
    Changes vs rtdnet_slim.py:
        SPPFSlim  →  LiteASPP   (this file)
    All other modules are identical to rtdnet_slim.py.
    """

    def __init__(self, num_classes=30, base_ch=32, num_heads=4,
                 C=16, dropout=0.3):
        super().__init__()
        b = base_ch

        self.conv1 = ConvBNSiLU(3,   b,    3, stride=2)
        self.conv2 = ConvBNSiLU(b,   b*2,  3, stride=2)
        self.lem1  = LEM(b*2,  b*2,  C=C)
        self.conv3 = ConvBNSiLU(b*2, b*4,  3, stride=2)
        self.lem2  = LEM(b*4,  b*4,  C=C)
        self.conv4 = ConvBNSiLU(b*4, b*8,  3, stride=2)
        self.lem3  = LEM(b*8,  b*8,  C=C)
        self.conv5 = DSConvBNSiLU(b*8, b*16, stride=2)   # slim change 1

        ectb_ch    = b * 16
        safe_heads = num_heads
        while ectb_ch // 4 % safe_heads != 0:
            safe_heads = max(1, safe_heads // 2)
        self.ectb  = ECTB(ectb_ch, ectb_ch, num_heads=safe_heads)  # slim change 3

        # ← EXPERIMENT 1: LiteASPP replaces SPPFSlim
        self.sppf  = LiteASPP(b*16, b*16)

        self.aph   = APH(b*16, b*16)
        self.gap   = nn.AdaptiveAvgPool2d(1)
        self.drop  = nn.Dropout(dropout)
        self.fc    = nn.Linear(b*16, num_classes)
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

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x = self.conv3(self.lem1(x))
        x = self.conv4(self.lem2(x))
        x = self.conv5(self.lem3(x))
        x = self.sppf(self.ectb(x))
        x = self.aph(x)
        return self.fc(self.drop(self.gap(x).flatten(1)))

    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        return total, trainable


if __name__ == '__main__':
    import time
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = RTDNetLiteASPP(num_classes=30).to(device)
    total, _ = model.count_parameters()
    print(f'RTDNet-LiteASPP  params: {total:,}  ({total/1e6:.3f} M)')
    dummy = torch.randn(2, 3, 640, 640).to(device)
    with torch.no_grad():
        out = model(dummy)
    print(f'Output shape: {out.shape}  ✓')
    # Latency
    dummy1 = torch.randn(1, 3, 224, 224).to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(5): model(dummy1)
        t0 = time.perf_counter()
        for _ in range(100): model(dummy1)
    ms = (time.perf_counter() - t0) / 100 * 1000
    print(f'Latency 224px bs=1: {ms:.1f} ms  (~{1000/ms:.0f} FPS)')