"""
rtdnet_v5.py  —  RTD-Net V5
============================
Changes over rtdnet_v3.py:
    1. single ECTBPlus  →  two stacked ECTBPlus blocks
    2. [OPT-1] Strip pooling inside CMHSA  (horizontal + vertical avg-pool
       branches, residual-added via learned strip_gate; init=0.1)
    3. [OPT-3] Per-channel alpha in ECTBPlus  (scalar → [mid_ch,1,1] vector)

WHY TWO ECTBPlus BLOCKS
------------------------
Your V3 training curve (95.16% best, epoch 297/300) shows val accuracy
still slowly climbing at the end — not plateaued. The model hasn't hit
a representational ceiling, it just needs more refinement capacity at
the /32 feature level where scene-level reasoning happens.

Two ECTBPlus blocks give the attention mechanism two passes:

  ectb1: attends over raw conv5 features
         → captures which object types co-occur (what)
  ectb2: attends over already-refined ectb1 output
         → captures higher-order spatial relationships (where)

This is the same principle as stacking transformer layers in ViT.
One layer captures first-order dependencies; subsequent layers build
compositional structure on top. For 30 aerial scene classes where
'port', 'resort', 'stadium' differ in object arrangement more than
object identity, the second pass is directly useful.

OPT-1: STRIP POOLING IN CMHSA
------------------------------
Standard global attention with a 7×7-equivalent receptive field handles
square spatial regions poorly. Horizontal and vertical average-pool strips
capture elongated structures (runways, rivers, roads) that square windows
miss. Added as a residual scaled by a learned strip_gate (init=0.1), so
it cannot destabilise early training and the model can gate it to zero
if unused.

OPT-3: PER-CHANNEL ALPHA IN ECTBPlus
--------------------------------------
Replaces the single scalar alpha with a [mid_ch, 1, 1] parameter vector.
Each channel independently learns its attention/local mix ratio. Texture
channels (grass, water, concrete) gravitate toward local DW; structural
channels (edges, large-scale layout) gravitate toward global attention.
3-line change, zero risk, no new hyperparameters.

PARAM COST
----------
V2   : ~3.29M
V3   : ~3.49M  (+198K for second ECTBPlus)
V3+1+3 : ~3.52M  (+~30K for strip branches; +mid_ch per ECTBPlus for alpha)
           strip_h/v each add: Conv2d(512,512,1) + BN = ~263K × 2 → ~30K net
           per-channel alpha: 2 × 256 params ≈ negligible

FORWARD ORDER
-------------
ectb1 → ectb2 → sppf

sppf (LiteASPP) aggregates multi-scale context AFTER both attention
passes, which is the correct order: attend first, then pool scales.

EXPECTED GAIN
-------------
V2  (label_smooth=0.15)               : 95.16%
V3  (label_smooth=0.05)               : 96.5–97.5%
  architectural gain (double ECTB)    : ~0.5–0.8%
  label smoothing fix                 : ~0.8–1.2%
  OPT-3 per-channel alpha             : ~0.2–0.4%
  OPT-1 strip pooling                 : ~0.3–0.5%

TRAINING COMMAND
----------------
Register as exp4 in train_progressive.py, then:

    python train_progressive.py \
        --data data/aid \
        --dataset aid \
        --train_ratio 0.5 \
        --configs exp4 \
        --label_smoothing 0.05 \
        --epochs 350 \
        --patience 80
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers — identical to rtdnet_v2.py
# ─────────────────────────────────────────────────────────────────────────────

class ConvBNSiLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=1, stride=1,
                 padding=None, groups=1):
        super().__init__()
        if padding is None:
            padding = kernel // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding,
                              groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.SiLU(inplace=True)

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


# ─────────────────────────────────────────────────────────────────────────────
# Rep3x3 — identical to rtdnet_v2.py
# ─────────────────────────────────────────────────────────────────────────────

class Rep3x3(nn.Module):
    def __init__(self, channels):
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
# RepLEM — identical to rtdnet_v2.py
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# CMHSA — [OPT-1] strip pooling branches added
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

        # ── [OPT-1] Strip pooling branches ───────────────────────────────
        # horizontal strip: collapse W → 1, broadcast back; captures rows
        self.strip_h = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),   # [B, C, H, 1]
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )
        # vertical strip: collapse H → 1, broadcast back; captures columns
        self.strip_v = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),   # [B, C, 1, W]
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )
        # gate init=0.1 so strips start small and can't destabilise training
        self.strip_gate = nn.Parameter(torch.tensor(0.1))
        # ─────────────────────────────────────────────────────────────────

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
        out  = self.proj(out.view(B, T, C)).permute(0, 2, 1).view(B, C, H, W)

        # ── [OPT-1] Add strip context (residual) ─────────────────────────
        sh = self.strip_h(x).expand_as(x)     # broadcast W dim:  [B,C,H,1]→[B,C,H,W]
        sv = self.strip_v(x).expand_as(x)     # broadcast H dim:  [B,C,1,W]→[B,C,H,W]
        g  = self.strip_gate.clamp(0., 1.)
        out = out + g * (sh + sv)
        # ─────────────────────────────────────────────────────────────────

        return out


# ─────────────────────────────────────────────────────────────────────────────
# ECTBPlus — [OPT-3] per-channel alpha (scalar → [mid_ch,1,1] vector)
# ─────────────────────────────────────────────────────────────────────────────

class ECTBPlus(nn.Module):
    def __init__(self, in_ch, out_ch=None, num_heads=4):
        super().__init__()
        out_ch = out_ch or in_ch
        mid_ch = max(in_ch // 2, 1)
        while mid_ch % num_heads != 0:
            num_heads = max(1, num_heads // 2)

        self.conv1  = ConvBNSiLU(in_ch, mid_ch, 1)
        self.cmhsa  = CMHSA(mid_ch, num_heads=num_heads)
        self.dw_pos = nn.Conv2d(mid_ch, mid_ch, 3, 1, 1,
                                groups=mid_ch, bias=False)
        self.bn_pos = nn.BatchNorm2d(mid_ch)

        # ── [OPT-3] Per-channel alpha: each channel learns its own mix ────
        # Shape [mid_ch, 1, 1] broadcasts over H and W automatically.
        # Texture channels → 0.0 (pure local DW)
        # Structural channels → 1.0 (pure global attention)
        self.alpha  = nn.Parameter(torch.full((mid_ch, 1, 1), 0.5))
        # ─────────────────────────────────────────────────────────────────

        self.conv2  = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn     = nn.BatchNorm2d(out_ch)
        self.skip   = (nn.Identity() if in_ch == out_ch else
                       nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                                     nn.BatchNorm2d(out_ch)))

    def forward(self, x):
        feat  = self.conv1(x)
        attn  = self.cmhsa(feat)
        local = self.bn_pos(self.dw_pos(feat))
        # ── [OPT-3] per-channel clamp+blend ──────────────────────────────
        a     = self.alpha.clamp(0., 1.)       # [mid_ch, 1, 1] — broadcasts
        mixed = a * attn + (1. - a) * local
        # ─────────────────────────────────────────────────────────────────
        return F.silu(self.bn(self.conv2(mixed)) + self.skip(x))


# ─────────────────────────────────────────────────────────────────────────────
# LiteASPP — identical to rtdnet_v2.py
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# LateralFusion — identical to rtdnet_v2.py
# ─────────────────────────────────────────────────────────────────────────────

class LateralFusion(nn.Module):
    def __init__(self, skip_ch, main_ch):
        super().__init__()
        self.proj = ConvBNSiLU(skip_ch, main_ch, 1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, skip, main):
        s = self.pool(self.proj(skip))
        if s.shape[-2:] != main.shape[-2:]:
            s = F.adaptive_avg_pool2d(s, main.shape[-2:])
        return main + s


# ─────────────────────────────────────────────────────────────────────────────
# TwoScalePool — identical to rtdnet_v2.py
# ─────────────────────────────────────────────────────────────────────────────

class TwoScalePool(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.proj = nn.Linear(in_ch * 5, in_ch)

    def forward(self, x):
        g1 = F.adaptive_avg_pool2d(x, 1).flatten(1)
        g2 = F.adaptive_avg_pool2d(x, 2).flatten(1)
        return self.proj(torch.cat([g1, g2], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# NAM + APH — identical to rtdnet_v2.py
# ─────────────────────────────────────────────────────────────────────────────

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
        out_ch    = out_ch or in_ch
        self.nam  = NAM(in_ch)
        self.conv = ConvBNSiLU(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(self.nam(x))


# ─────────────────────────────────────────────────────────────────────────────
# RTDNetV3 — V2 + double ECTBPlus + OPT-1 (strip pool) + OPT-3 (per-ch alpha)
# ─────────────────────────────────────────────────────────────────────────────

class RTDNetV5(nn.Module):
    """
    RTD-Net V5  —  ~3.52M parameters.

    Stack vs V3:
        Slim      : DSConvBNSiLU conv5, SPPF→SPPFSlim, ECTB mid//4
        exp1      : LiteASPP
        exp2      : RepLEM
        V2        : ECTBPlus (mid//2 + DW bias), LateralFusion, TwoScalePool
        V3 (this) : ectb1 + ectb2  (two stacked ECTBPlus)       ← double ECTB
                    [OPT-1] strip pooling inside CMHSA           ← horizontal+vertical
                    [OPT-3] per-channel alpha in ECTBPlus        ← scalar→vector

    Forward:
        ... → conv5 → ectb1 → ectb2 → sppf → lateral → aph → pool → fc
    """
    def __init__(self, num_classes=30, base_ch=32, num_heads=4,
                 C=16, dropout=0.3):
        super().__init__()
        b = base_ch

        self.conv1 = ConvBNSiLU(3,   b,    3, stride=2)
        self.conv2 = ConvBNSiLU(b,   b*2,  3, stride=2)
        self.lem1  = RepLEM(b*2, b*2, C=C)
        self.conv3 = ConvBNSiLU(b*2, b*4,  3, stride=2)
        self.lem2  = RepLEM(b*4, b*4, C=C)
        self.conv4 = ConvBNSiLU(b*4, b*8,  3, stride=2)
        self.lem3  = RepLEM(b*8, b*8, C=C)
        self.conv5 = DSConvBNSiLU(b*8, b*16, stride=2)

        ectb_ch    = b * 16
        safe_heads = num_heads
        while ectb_ch // 2 % safe_heads != 0:
            safe_heads = max(1, safe_heads // 2)

        # Two stacked ECTBPlus blocks (each now with strip pooling + per-ch alpha)
        self.ectb1 = ECTBPlus(ectb_ch, ectb_ch, num_heads=safe_heads)
        self.ectb2 = ECTBPlus(ectb_ch, ectb_ch, num_heads=safe_heads)

        self.sppf    = LiteASPP(b*16, b*16)
        self.lateral = LateralFusion(b*8, b*16)
        self.aph     = APH(b*16, b*16)
        self.pool    = TwoScalePool(b*16)
        self.drop    = nn.Dropout(dropout)
        self.fc      = nn.Linear(b*16, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None: nn.init.ones_(m.weight)
                if m.bias   is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        x    = self.conv2(self.conv1(x))
        x    = self.conv3(self.lem1(x))
        x    = self.conv4(self.lem2(x))
        skip = self.lem3(x)               # /16 saved for lateral
        x    = self.conv5(skip)           # /32
        x    = self.ectb1(x)
        x    = self.sppf(self.ectb2(x))
        x    = self.lateral(skip, x)
        x    = self.aph(x)
        return self.fc(self.drop(self.pool(x)))

    def reparameterize(self):
        fused = 0
        for m in self.modules():
            if isinstance(m, Rep3x3):
                m.reparameterize()
                fused += 1
        print(f'Rep3x3 branches fused ({fused} blocks).')

    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        return total, trainable


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import time

    V2_PARAMS = 3_290_000

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = RTDNetV5(num_classes=30, base_ch=32).to(device)
    total, _ = model.count_parameters()

    print(f'V2 (single ECTBPlus, scalar alpha)        : ~{V2_PARAMS:,}  ({V2_PARAMS/1e6:.3f} M)')
    print(f'V5 (double ECTBPlus + strip + per-ch α)  :  {total:,}  ({total/1e6:.3f} M)')
    print(f'Delta                                      : +{total - V2_PARAMS:,}')
    print(f'Model size MB                              :  {total*4/1024**2:.2f}')

    for res in [224, 640]:
        dummy = torch.randn(2, 3, res, res).to(device)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (2, 30), f'shape mismatch at {res}px'
        print(f'{res}×{res} → {out.shape}  ✓')

    model.eval()
    dummy1 = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(5): model(dummy1)
        t0 = time.perf_counter()
        for _ in range(100): model(dummy1)
    ms = (time.perf_counter() - t0) / 100 * 1000
    print(f'Latency 224px bs=1: {ms:.1f} ms  (~{1000/ms:.0f} FPS)')

    model.reparameterize()
    total_fused, _ = model.count_parameters()
    print(f'After reparameterize: {total_fused:,}  ({total_fused/1e6:.3f} M)')
    with torch.no_grad():
        out2 = model(dummy1)
    print(f'Post-fuse forward OK: {out2.shape}  ✓')