"""
rtdnet_nam_coordinate.py  —  RTD-Net with Hybrid NAM + Coordinate Attention
=============================================================================
Drop-in replacement for rtdnet_slim.py.

What changed vs rtdnet_slim.py:
  • NAMSpatialAttention (InstanceNorm-based) is replaced with
    CoordinateSpatialAttention (Fig. 5, Wang et al. 2024).
  • APH now wraps NAMWithCoordinateAttention instead of the old NAM.
  • Everything else (DSConvBNSiLU, LEM, CMHSA, ECTB, SPPFSlim) is unchanged.

Where the new module sits in the forward pass:
  conv1 → conv2 → lem1 → conv3 → lem2 → conv4 → lem3
    → conv5 (DSConv) → ectb → sppf → [APH w/ new NAM+CA] → GAP → fc

Module map
----------
CoordinateSpatialAttention   replaces NAMSpatialAttention
NAMWithCoordinateAttention   replaces NAM  (channel branch unchanged)
APH                          wraps NAMWithCoordinateAttention (unchanged API)
RTDNetNAMCoordinate          the full classifier (= entry point)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Shared primitives  (identical to rtdnet_slim.py — do not edit)
# ─────────────────────────────────────────────────────────────────────────────

class ConvBNSiLU(nn.Module):
    """Standard Conv → BN → SiLU."""
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


class DSConvBNSiLU(nn.Module):
    """
    Depthwise-Separable Conv → BN → SiLU  (CHANGE 1 from rtdnet_slim).
    Replaces the expensive conv5 (256→512, stride=2).
    Params: ~134K vs ~1.18M for standard conv.
    """
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.dw  = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.pw  = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class LEM(nn.Module):
    """Lightweight Extraction Module — unchanged from baseline."""
    def __init__(self, in_ch, out_ch=None, C=16):
        super().__init__()
        out_ch = out_ch or in_ch
        mid_ch = max(in_ch // 2, 1)
        br_ch  = max(in_ch // 32, 1)
        self.C = C
        self.conv1    = ConvBNSiLU(in_ch, mid_ch, 1)
        self.branches = nn.ModuleList([
            nn.Sequential(ConvBNSiLU(mid_ch, br_ch, 1),
                          ConvBNSiLU(br_ch, br_ch, 3))
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
    """Convolutional Multi-Head Self-Attention — unchanged."""
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
        B, C, H, W = x.shape; T = H * W
        q = self.conv_q(x).flatten(2).view(B, self.num_heads, self.head_dim, T)
        k = self.conv_k(x).flatten(2).view(B, self.num_heads, self.head_dim, T)
        v = self.conv_v(x).flatten(2).view(B, self.num_heads, self.head_dim, T)
        attn = self.inst_norm(F.softmax(
            self.head_conv(torch.einsum('bhdq,bhdT->bhqT', q, k) * self.scale),
            dim=-1))
        out = torch.einsum('bhqT,bhTd->bhqd',
                           attn, v.permute(0, 1, 3, 2)).contiguous()
        return self.proj(out.view(B, T, C)).permute(0, 2, 1).view(B, C, H, W)


class ECTB(nn.Module):
    """Efficient Convolutional Transformer Block — slim bottleneck (mid=C//4)."""
    def __init__(self, in_ch, out_ch=None, num_heads=4):
        super().__init__()
        out_ch = out_ch or in_ch
        mid_ch = max(in_ch // 4, 1)
        self.conv1 = ConvBNSiLU(in_ch, mid_ch, 1)
        self.cmhsa = CMHSA(mid_ch, num_heads=num_heads)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn    = nn.BatchNorm2d(out_ch)
        self.skip  = (nn.Identity() if in_ch == out_ch else
                      nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                                    nn.BatchNorm2d(out_ch)))

    def forward(self, x):
        return F.silu(self.bn(self.conv2(self.cmhsa(self.conv1(x)))) + self.skip(x))


class SPPFSlim(nn.Module):
    """Spatial Pyramid Pooling Fast — slim mid channels (mid=C//4)."""
    def __init__(self, in_ch, out_ch, pool_size=5):
        super().__init__()
        mid_ch    = in_ch // 4
        self.cv1  = ConvBNSiLU(in_ch,    mid_ch, 1)
        self.cv2  = ConvBNSiLU(mid_ch*4, out_ch, 1)
        self.pool = nn.MaxPool2d(pool_size, 1, pool_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        p1, p2, p3 = self.pool(x), self.pool(self.pool(x)), self.pool(self.pool(self.pool(x)))
        return self.cv2(torch.cat([x, p1, p2, p3], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Hybrid NAM + Coordinate Attention
# ─────────────────────────────────────────────────────────────────────────────

class NAMChannelAttention(nn.Module):
    """
    Original NAM Channel Attention — UNCHANGED.

    Uses BN scale factors (gamma) normalised to [0,1] as channel importance
    weights, then gates with sigmoid.  Liu et al., "NAM: Normalization-based
    Attention Module".
    """
    def __init__(self, channels: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.bn(x)
        gamma  = self.bn.weight.abs()                    # (C,)
        w      = gamma / (gamma.sum() + 1e-8)            # normalise to sum=1
        return x * torch.sigmoid(w.view(1, -1, 1, 1) * normed)


class CoordinateSpatialAttention(nn.Module):
    """
    Coordinate Attention — REPLACES NAMSpatialAttention.

    Implements the exact Fig. 5 pipeline from:
      Wang et al., "Urban traffic tiny object detection via attention and
      multi-scale feature driven in UAV-vision", Scientific Reports 2024.

    Equations referenced below:
      Eq.6  z^h_c(h) = (1/W) Σ x_c(h,i)          — pool along width
      Eq.7  z^w_c(w) = (1/H) Σ x_c(j,w)          — pool along height
      Eq.8  f = δ(F1([z^h, z^w]))                 — shared 1×1 conv+BN+SiLU
      Eq.9  g^h = σ(F_h(f^h))                     — height attention map
      Eq.10 g^w = σ(F_w(f^w))                     — width  attention map
      Eq.11 y_c(i,j) = x_c(i,j) × g^h_c(i) × g^w_c(j)  — re-weight
            + residual skip (Fig.5 explicit residual path)

    Args:
        channels : number of input channels C
        r        : reduction ratio for the shared bottleneck (default 32,
                   matches paper; min channel=1 to handle small C safely)
    """
    def __init__(self, channels: int, r: int = 32):
        super().__init__()
        mid = max(channels // r, 1)          # C/r

        # Eq.8  — shared 1×1 Conv operating on the concatenated strip
        self.shared_conv = nn.Conv2d(channels, mid, kernel_size=1, bias=False)
        self.shared_bn   = nn.BatchNorm2d(mid)
        # activation is SiLU (paper uses "Non-linear", SiLU matches RTD-Net style)

        # Eq.9,10 — per-direction projection back to C channels
        self.conv_h = nn.Conv2d(mid, channels, kernel_size=1, bias=False)  # g^h
        self.conv_w = nn.Conv2d(mid, channels, kernel_size=1, bias=False)  # g^w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # ── Eq.6: pool along width → (B, C, H, 1) ────────────────────────────
        z_h = x.mean(dim=3, keepdim=True)           # (B, C, H, 1)

        # ── Eq.7: pool along height → (B, C, 1, W) ───────────────────────────
        z_w = x.mean(dim=2, keepdim=True)           # (B, C, 1, W)

        # ── Eq.8: concat → shared conv → BN → SiLU ───────────────────────────
        # Transpose z_w to (B,C,W,1) so both strips share the last dim = 1
        z_w_t  = z_w.permute(0, 1, 3, 2)            # (B, C, W, 1)
        z_cat  = torch.cat([z_h, z_w_t], dim=2)     # (B, C, H+W, 1)
        f      = F.silu(self.shared_bn(
                     self.shared_conv(z_cat)))       # (B, C/r, H+W, 1)

        # ── Split back into per-direction strips ──────────────────────────────
        f_h = f[:, :, :H, :]                         # (B, C/r, H,   1)
        f_w = f[:, :, H:, :]                         # (B, C/r, W,   1)

        # ── Eq.9: g^h  ───────────────────────────────────────────────────────
        g_h = torch.sigmoid(self.conv_h(f_h))        # (B, C, H, 1)

        # ── Eq.10: g^w  ──────────────────────────────────────────────────────
        # restore to (B, C, 1, W) for correct broadcast with X
        g_w = torch.sigmoid(self.conv_w(
                  f_w.permute(0, 1, 3, 2)))          # (B, C, 1, W)

        # ── Eq.11: re-weight  (broadcast: H×1 and 1×W → H×W) ────────────────
        y = x * g_h * g_w

        # ── Fig.5 explicit residual ───────────────────────────────────────────
        return x + y


class NAMWithCoordinateAttention(nn.Module):
    """
    Hybrid NAM Channel + Coordinate Spatial Attention.

    Channel branch : original NAM (BN gamma weights + sigmoid) — UNCHANGED.
    Spatial branch : Coordinate Attention from Fig. 5 — NEW.

    Applied sequentially:  x → channel_attn → spatial_attn → output

    Args:
        channels : feature channels
        r        : CA reduction ratio (default 32)
    """
    def __init__(self, channels: int, r: int = 32):
        super().__init__()
        self.channel_attn = NAMChannelAttention(channels)
        self.spatial_attn  = CoordinateSpatialAttention(channels, r=r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


# Convenience alias  (matches existing import in APH)
ImprovedNAM = NAMWithCoordinateAttention


class APH(nn.Module):
    """
    Attention Prediction Head — same API as before, now uses
    NAMWithCoordinateAttention internally.
    """
    def __init__(self, in_ch: int, out_ch: int = None, r: int = 32):
        super().__init__()
        out_ch    = out_ch or in_ch
        self.nam  = NAMWithCoordinateAttention(in_ch, r=r)
        self.conv = ConvBNSiLU(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.nam(x))


# ─────────────────────────────────────────────────────────────────────────────
# Full classifier
# ─────────────────────────────────────────────────────────────────────────────

class RTDNetNAMCoordinate(nn.Module):
    """
    RTD-Net + Hybrid NAM-Coordinate Attention Classifier.

    Architecture (identical to RTDNetClassifier / rtdnet_slim.py except APH):
      conv1  ConvBNSiLU(3,    b,   3, /2)
      conv2  ConvBNSiLU(b,    b*2, 3, /2)
      lem1   LEM(b*2,  b*2)
      conv3  ConvBNSiLU(b*2,  b*4, 3, /2)
      lem2   LEM(b*4,  b*4)
      conv4  ConvBNSiLU(b*4,  b*8, 3, /2)
      lem3   LEM(b*8,  b*8)
      conv5  DSConvBNSiLU(b*8, b*16, /2)      ← slim CHANGE 1
      ectb   ECTB(b*16, mid=C//4)             ← slim CHANGE 3
      sppf   SPPFSlim(b*16, mid=C//4)         ← slim CHANGE 2
      aph    APH(b*16) with NAMWithCoordinateAttention  ← NEW
      gap    AdaptiveAvgPool2d(1)
      drop   Dropout
      fc     Linear(b*16, num_classes)

    Args:
        num_classes : output classes (default 30 for AID)
        base_ch     : base channel width (default 32)
        num_heads   : ECTB/CMHSA heads (default 4)
        C           : LEM branches (default 16)
        dropout     : dropout probability before FC (default 0.3)
        ca_r        : Coordinate Attention reduction ratio (default 32)
    """
    def __init__(
        self,
        num_classes : int   = 30,
        base_ch     : int   = 32,
        num_heads   : int   = 4,
        C           : int   = 16,
        dropout     : float = 0.3,
        ca_r        : int   = 32,
    ):
        super().__init__()
        b = base_ch

        # Stem
        self.conv1 = ConvBNSiLU(3,   b,    3, stride=2)
        self.conv2 = ConvBNSiLU(b,   b*2,  3, stride=2)

        # Stage 1
        self.lem1  = LEM(b*2,  b*2,  C=C)
        self.conv3 = ConvBNSiLU(b*2, b*4,  3, stride=2)

        # Stage 2
        self.lem2  = LEM(b*4,  b*4,  C=C)
        self.conv4 = ConvBNSiLU(b*4, b*8,  3, stride=2)

        # Stage 3
        self.lem3  = LEM(b*8,  b*8,  C=C)

        # DS conv  (slim change 1)
        self.conv5 = DSConvBNSiLU(b*8, b*16, stride=2)

        # ECTB — slim bottleneck (slim change 3)
        ectb_ch    = b * 16
        safe_heads = num_heads
        while ectb_ch // 4 % safe_heads != 0:
            safe_heads = max(1, safe_heads // 2)
        self.ectb  = ECTB(ectb_ch, ectb_ch, num_heads=safe_heads)

        # SPPF — slim mid (slim change 2)
        self.sppf  = SPPFSlim(b*16, b*16)

        # APH with new hybrid attention  ← KEY CHANGE
        self.aph   = APH(b*16, b*16, r=ca_r)

        # Classifier head
        self.gap   = nn.AdaptiveAvgPool2d(1)
        self.drop  = nn.Dropout(dropout)
        self.fc    = nn.Linear(b*16, num_classes)

        self._init_weights()

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv2(self.conv1(x))       # stem
        x = self.conv3(self.lem1(x))        # stage 1
        x = self.conv4(self.lem2(x))        # stage 2
        x = self.conv5(self.lem3(x))        # stage 3 + DS conv
        x = self.sppf(self.ectb(x))         # transformer + multi-scale pool
        x = self.aph(x)                     # hybrid NAM+CA attention head  ← NEW
        return self.fc(self.drop(self.gap(x).flatten(1)))

    # ------------------------------------------------------------------
    def count_parameters(self) -> tuple:
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
if __name__ == "__main__":
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = RTDNetNAMCoordinate(num_classes=30, base_ch=32, ca_r=32).to(device)
    total, trainable = model.count_parameters()

    print("── Per-module parameter breakdown ───────────────────")
    for name, p in model.per_module_params().items():
        print(f"  {name:<10}  {p:>9,}  ({p/total*100:5.1f}%)")

    print(f"\n── Summary ──────────────────────────────────────────")
    SLIM_PARAMS = 1_453_000   # approximate rtdnet_slim baseline
    print(f"  RTDNet-Slim baseline :  ~{SLIM_PARAMS/1e6:.3f} M")
    print(f"  RTDNetNAMCoordinate  :   {total:>9,}  ({total/1e6:.3f} M)")
    print(f"  Model size MB        :   {total*4/1024**2:.2f}")

    for res in [224, 640]:
        dummy = torch.randn(2, 3, res, res).to(device)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (2, 30), f"Shape error at {res}px: {out.shape}"
        print(f"\n  {res}×{res}  →  {out.shape}  ✓")

    dummy1 = torch.randn(1, 3, 224, 224).to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(5): model(dummy1)
        t0 = time.perf_counter()
        for _ in range(100): model(dummy1)
    ms = (time.perf_counter() - t0) / 100 * 1000
    print(f"\n  Avg latency (224px, bs=1): {ms:.1f} ms (~{1000/ms:.0f} FPS)")
    print("\nAll checks passed!")