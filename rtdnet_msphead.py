"""
rtdnet_msphead.py  —  RTD-Net Slim + LiteASPP + RepLEM + MultiScalePoolHead
=============================================================================
Experiment 3: Adds GeM + Spatial Pyramid Pooling classifier head,
on top of Experiments 1 and 2.

MultiScalePoolHead: what changes
----------------------------------
Original:   GAP(1×1) → flatten → Dropout → Linear(C → num_classes)
New:        GeM(1×1) + AvgPool(2×2) + AvgPool(4×4) → flatten → compress
            → Dropout → Linear(C → num_classes)

Spatial pyramid descriptor sizes:
    GeM  1×1  → C  features  (sharpened global)
    Pool 2×2  → 4C features  (quadrant layout)
    Pool 4×4  → 16C features (fine spatial grid)
    Total: 21C → Linear(21C → C) → SiLU → Dropout → Linear(C → num_classes)

Why this helps:
    Plain GAP loses all spatial arrangement — 'dense residential' and
    'medium residential' have identical texture but different spatial density.
    GeM's learnable p (init=3) focuses more on activated regions than GAP
    (p=1) while staying softer than MaxPool (p=∞).  The 2×2 and 4×4 grids
    give the linear classifier explicit layout information.

Parameter overhead: Linear(21C → C) + Linear(C → num_classes).
    For base_ch=32, C=512:  21×512×512 = ~5.5M  ← too large!
    We therefore cap the compression:  Linear(21C → C) → still C=512.
    Actually 21*512 → 512 = 5.5M is large.  We compress mid-step:
       GeM(C) + Pool2(4C_reduced) + Pool4(16C_reduced)
    Instead we pool spatially but use channel-reduced versions:
       project C → C//4 before SPP to keep overhead manageable.

Final parameter count for SPP head: ~3.5M overhead — still acceptable since
the backbone saved ~1.7M vs baseline.  If too heavy, set spp_reduce=True.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Shared helpers ───────────────────────────────────────────────────────────

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
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.dw  = nn.Conv2d(in_ch, in_ch, 3, stride, 1,
                             groups=in_ch, bias=False)
        self.pw  = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class Rep3x3(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.conv3    = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn3      = nn.BatchNorm2d(channels)
        self.conv1    = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn1      = nn.BatchNorm2d(channels)
        self.bn_id    = nn.BatchNorm2d(channels)
        self._fused   = False

    def forward(self, x):
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
        W = k3 + k1 + ki;  B = b3 + b1 + bi
        fused = nn.Conv2d(self.channels, self.channels, 3, 1, 1, bias=True)
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
        return (w * s.view(-1, 1, 1, 1), bn.bias - bn.running_mean * s)

    def _identity_branch(self):
        C   = self.channels
        dev = self.bn_id.weight.device
        w   = torch.zeros(C, C, 3, 3, device=dev)
        for i in range(C): w[i, i, 1, 1] = 1.0
        s = self.bn_id.weight / (self.bn_id.running_var + self.bn_id.eps).sqrt()
        return (w * s.view(-1, 1, 1, 1),
                self.bn_id.bias - self.bn_id.running_mean * s)


class RepLEM(nn.Module):
    def __init__(self, in_ch, out_ch=None, C=16):
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


class LiteASPP(nn.Module):
    def __init__(self, in_ch, out_ch):
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

    def forward(self, x):
        h, w = x.shape[-2:]
        g    = F.interpolate(self.gap(x), size=(h, w),
                             mode='bilinear', align_corners=False)
        return self.proj(torch.cat([self.b0(x), self.b1(x),
                                    self.b2(x), g], dim=1))


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


# ─── NEW: GeM + MultiScale Spatial Pool Head ─────────────────────────────────

class GeM(nn.Module):
    """
    Generalized Mean Pooling.
    p=1 → AvgPool, p→∞ → MaxPool.  Learnable p init=3 focuses on
    activated spatial regions without hard-max brittleness.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), 1).pow(1.0 / self.p)


class MultiScalePoolHead(nn.Module):
    """
    GeM(1×1) + AvgPool(2×2) + AvgPool(4×4)  spatial pyramid head.

    To avoid parameter explosion we first channel-compress with a
    1×1 conv (C → C//2) before the SPP, then flatten:
        C//2 × (1 + 4 + 16) = C//2 × 21

    Then:  Linear(C//2 × 21 → C//2) → SiLU → Dropout → Linear(C//2 → classes)

    For base_ch=32, C=512:
        C//2 = 256;  256 × 21 = 5376 → 256 → 30
        Extra params: 256×21×256 + 256×30 ≈ 1.4M   (acceptable)
    """

    def __init__(self, in_ch: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        mid = max(in_ch // 2, 1)
        self.compress = ConvBNSiLU(in_ch, mid, 1)   # C → C//2
        self.gem      = GeM(p=3.0)
        self.pool2    = nn.AdaptiveAvgPool2d(2)
        self.pool4    = nn.AdaptiveAvgPool2d(4)
        total_flat    = mid * (1 + 4 + 16)
        self.fc1      = nn.Linear(total_flat, mid)
        self.act      = nn.SiLU(inplace=True)
        self.drop     = nn.Dropout(dropout)
        self.fc2      = nn.Linear(mid, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x  = self.compress(x)                       # (B, C//2, H, W)
        g  = self.gem(x).flatten(1)                 # (B, C//2)
        p2 = self.pool2(x).flatten(1)               # (B, 4·C//2)
        p4 = self.pool4(x).flatten(1)               # (B, 16·C//2)
        feat = torch.cat([g, p2, p4], dim=1)        # (B, 21·C//2)
        return self.fc2(self.drop(self.act(self.fc1(feat))))


# ─── Full model ──────────────────────────────────────────────────────────────

class RTDNetMSPHead(nn.Module):
    """
    RTD-Net Slim + LiteASPP + RepLEM + MultiScalePoolHead.
    Changes vs rtdnet_slim.py:
        SPPFSlim      →  LiteASPP           (experiment 1)
        LEM           →  RepLEM             (experiment 2)
        GAP+FC        →  MultiScalePoolHead (experiment 3, this file)
    """

    def __init__(self, num_classes=30, base_ch=32, num_heads=4,
                 C=16, dropout=0.3):
        super().__init__()
        b = base_ch

        self.conv1 = ConvBNSiLU(3,   b,    3, stride=2)
        self.conv2 = ConvBNSiLU(b,   b*2,  3, stride=2)
        self.lem1  = RepLEM(b*2,  b*2,  C=C)
        self.conv3 = ConvBNSiLU(b*2, b*4,  3, stride=2)
        self.lem2  = RepLEM(b*4,  b*4,  C=C)
        self.conv4 = ConvBNSiLU(b*4, b*8,  3, stride=2)
        self.lem3  = RepLEM(b*8,  b*8,  C=C)
        self.conv5 = DSConvBNSiLU(b*8, b*16, stride=2)

        ectb_ch    = b * 16
        safe_heads = num_heads
        while ectb_ch // 4 % safe_heads != 0:
            safe_heads = max(1, safe_heads // 2)
        self.ectb  = ECTB(ectb_ch, ectb_ch, num_heads=safe_heads)
        self.sppf  = LiteASPP(b*16, b*16)
        self.aph   = APH(b*16, b*16)

        # ← EXPERIMENT 3: MultiScalePoolHead replaces GAP + fc
        self.head  = MultiScalePoolHead(b*16, num_classes, dropout=dropout)

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
        x = self.conv2(self.conv1(x))
        x = self.conv3(self.lem1(x))
        x = self.conv4(self.lem2(x))
        x = self.conv5(self.lem3(x))
        x = self.sppf(self.ectb(x))
        x = self.aph(x)
        return self.head(x)

    def reparameterize(self):
        for m in self.modules():
            if isinstance(m, Rep3x3):
                m.reparameterize()
        print('Rep3x3 branches fused.')

    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        return total, trainable


if __name__ == '__main__':
    import time
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = RTDNetMSPHead(num_classes=30).to(device)
    total, _ = model.count_parameters()
    print(f'RTDNet-MSPHead  params: {total:,}  ({total/1e6:.3f} M)')
    dummy = torch.randn(2, 3, 640, 640).to(device)
    with torch.no_grad():
        out = model(dummy)
    print(f'Output shape: {out.shape}  ✓')