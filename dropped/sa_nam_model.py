"""
sa_nam_model.py  —  RTD-Net with SA-NAM (Scale-Aware NAM) inside APH
=====================================================================

What changed vs model.py:
    NAM  →  SA_NAM  (Scale-Aware NAM)
    APH  →  updated to instantiate SA_NAM instead of NAM
    RTDNetClassifier → identical forward pass; only APH internals change

SA_NAM design:
    1. Apply original NAM at full resolution         → out1  (B,C,H,W)
    2. AvgPool2d(2) → NAM at half resolution → bilinear upsample → out2  (B,C,H,W)
    3. Fuse: final = sigmoid(w) * out1 + (1 − sigmoid(w)) * out2
       where w is a single learnable scalar (nn.Parameter), sigmoid-bounded
       so the blend weight stays in (0,1) throughout training.

Why this helps:
    Full-res NAM captures fine-grained channel/spatial importance (small
    objects, edges).  Half-res NAM captures coarse-grained context
    (scene-level texture, large structures).  The learnable blend lets the
    model weight both scales per-dataset — no manual tuning needed.

Parameter overhead:
    Exactly 1 extra scalar parameter (the blend weight w).
    No extra convolutions.  Memory overhead ≈ one extra activation map
    at half resolution during the forward pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper (unchanged)
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
        self.conv1    = ConvBNSiLU(in_ch, mid_ch, 1)
        self.branches = nn.ModuleList([
            nn.Sequential(ConvBNSiLU(mid_ch, br_ch, 1),
                          ConvBNSiLU(br_ch,  br_ch, 3))
            for _ in range(C)
        ])
        branch_total = br_ch * C
        self.conv2 = nn.Conv2d(branch_total, out_ch, 1, bias=False)
        self.bn    = nn.BatchNorm2d(out_ch)
        self.skip  = (nn.Identity() if in_ch == out_ch else
                      nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                                    nn.BatchNorm2d(out_ch)))

    def forward(self, x):
        feat   = self.conv1(x)
        concat = torch.cat([b(feat) for b in self.branches], dim=1)
        return F.silu(self.bn(self.conv2(concat)) + self.skip(x))


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
        q = self.conv_q(x).flatten(2)
        k = self.conv_k(x).flatten(2)
        v = self.conv_v(x).flatten(2)
        q = q.view(B, self.num_heads, self.head_dim, T)
        k = k.view(B, self.num_heads, self.head_dim, T)
        v = v.view(B, self.num_heads, self.head_dim, T)
        attn = torch.einsum('bhdq,bhdT->bhqT', q, k) * self.scale
        attn = self.head_conv(attn.view(B, self.num_heads, T, T))
        attn = F.softmax(attn, dim=-1)
        attn = self.inst_norm(attn)
        v_t  = v.permute(0, 1, 3, 2)
        out  = torch.einsum('bhqT,bhTd->bhqd', attn, v_t).contiguous().view(B, T, C)
        out  = self.proj(out).permute(0, 2, 1).view(B, C, H, W)
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
        feat = self.conv1(x)
        feat = self.cmhsa(feat)
        return F.silu(self.bn(self.conv2(feat)) + self.skip(x))


# ─────────────────────────────────────────────────────────────────────────────
# Original NAM sub-modules  (kept — SA_NAM reuses them internally)
# ─────────────────────────────────────────────────────────────────────────────
class NAMChannelAttention(nn.Module):
    """BN γ-weight channel attention (Eq. 8 in original paper)."""
    def __init__(self, channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        normed = self.bn(x)
        gamma  = self.bn.weight.abs()
        w      = gamma / (gamma.sum() + 1e-8)
        return x * torch.sigmoid(w.view(1, -1, 1, 1) * normed)


class NAMSpatialAttention(nn.Module):
    """InstanceNorm λ-weight spatial attention (Eq. 9 in original paper)."""
    def __init__(self, channels):
        super().__init__()
        self.bn = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        normed = self.bn(x)
        lam    = self.bn.weight.abs()
        w      = lam / (lam.sum() + 1e-8)
        return x * torch.sigmoid(w.view(1, -1, 1, 1) * normed)


class NAM(nn.Module):
    """Full NAM: channel → spatial attention."""
    def __init__(self, channels):
        super().__init__()
        self.channel = NAMChannelAttention(channels)
        self.spatial = NAMSpatialAttention(channels)

    def forward(self, x):
        return self.spatial(self.channel(x))


# ─────────────────────────────────────────────────────────────────────────────
# SA_NAM — Scale-Aware NAM                                        ← NEW
# ─────────────────────────────────────────────────────────────────────────────
class SA_NAM(nn.Module):
    """
    Scale-Aware Normalization-based Attention Module.

    Two NAM instances (shared weights via a single NAM object) applied at
    two spatial scales, then blended with a single learnable scalar w.

        full-res path:  x            → NAM → out1
        half-res path:  AvgPool(x,2) → NAM → bilinear upsample → out2
        output:         sigmoid(w) * out1 + (1 − sigmoid(w)) * out2

    Parameter overhead vs original NAM:  exactly +1 scalar (the blend w).

    Args:
        channels (int): Feature channel count C.
    """
    def __init__(self, channels: int):
        super().__init__()
        # A single NAM is applied at both scales to stay lightweight.
        # Sharing weights forces the same normalisation logic to work at
        # both resolutions — the blend weight decides which scale wins.
        self.nam = NAM(channels)

        # Learnable blend weight — raw (unbounded); pass through sigmoid
        # in forward so the effective weight stays strictly in (0, 1).
        # Initialised at 0.0 → sigmoid(0) = 0.5 (equal mix to start).
        self.w = nn.Parameter(torch.zeros(1))

        # Downsampler for the half-res path
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Full-resolution NAM ───────────────────────────────────────────────
        out1 = self.nam(x)                                    # (B, C, H, W)

        # ── Half-resolution NAM ───────────────────────────────────────────────
        x_down  = self.pool(x)                                # (B, C, H/2, W/2)
        out2_ds = self.nam(x_down)                            # (B, C, H/2, W/2)
        # Upsample back — bilinear with align_corners=False matches torchvision
        # convention and avoids checkerboard artefacts on odd spatial sizes.
        out2 = F.interpolate(out2_ds,
                             size=(x.shape[2], x.shape[3]),
                             mode='bilinear',
                             align_corners=False)             # (B, C, H, W)

        # ── Learnable scale-aware blend ───────────────────────────────────────
        alpha = torch.sigmoid(self.w)                         # scalar ∈ (0, 1)
        return alpha * out1 + (1.0 - alpha) * out2


# ─────────────────────────────────────────────────────────────────────────────
# APH — updated to use SA_NAM                                     ← CHANGED
# ─────────────────────────────────────────────────────────────────────────────
class APH(nn.Module):
    """
    Attention Prediction Head — SA_NAM edition.

    Identical interface to original APH; only the internal attention
    module is swapped from NAM to SA_NAM.

    Args:
        in_ch  (int): Input channels.
        out_ch (int): Output channels (defaults to in_ch).
    """
    def __init__(self, in_ch: int, out_ch: int = None):
        super().__init__()
        out_ch    = out_ch or in_ch
        self.nam  = SA_NAM(in_ch)                  # ← was NAM(in_ch)
        self.conv = ConvBNSiLU(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.nam(x)
        x = self.conv(x)
        return x


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
        return self.cv2(torch.cat([x, p1, p2, p3], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# RTDNetClassifier — forward pass identical; APH now wraps SA_NAM  ← note
# ─────────────────────────────────────────────────────────────────────────────
class RTDNetClassifier(nn.Module):
    """
    RTD-Net backbone with SA-NAM inside APH.

    The only difference from the original RTDNetClassifier is that APH
    now instantiates SA_NAM instead of NAM.  The forward pass is byte-for-
    byte identical — no structural change needed anywhere else.

    Args:
        num_classes (int)  : Output classes. Default 30 (AID).
        base_ch     (int)  : Base channel width. Default 32.
        num_heads   (int)  : ECTB heads. Default 4.
        C           (int)  : LEM branches. Default 16.
        dropout     (float): FC dropout. Default 0.3.
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

        # Stem
        self.conv1 = ConvBNSiLU(3,   b,    3, stride=2)
        self.conv2 = ConvBNSiLU(b,   b*2,  3, stride=2)

        # Stages
        self.lem1  = LEM(b*2, b*2,  C=C);  self.conv3 = ConvBNSiLU(b*2, b*4,  3, stride=2)
        self.lem2  = LEM(b*4, b*4,  C=C);  self.conv4 = ConvBNSiLU(b*4, b*8,  3, stride=2)
        self.lem3  = LEM(b*8, b*8,  C=C);  self.conv5 = ConvBNSiLU(b*8, b*16, 3, stride=2)

        # Transformer stage
        ectb_ch    = b * 16
        safe_heads = num_heads
        while ectb_ch % safe_heads != 0:
            safe_heads = max(1, safe_heads // 2)
        self.ectb = ECTB(ectb_ch, ectb_ch, num_heads=safe_heads)

        # Multi-scale pooling
        self.sppf = SPPF(b*16, b*16)

        # Attention head — APH now uses SA_NAM internally
        self.aph  = APH(b*16, b*16)

        # Classifier
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(b*16, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None: nn.init.ones_(m.weight)
                if m.bias  is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.conv2(self.conv1(x))
        # Stages
        x = self.conv3(self.lem1(x))
        x = self.conv4(self.lem2(x))
        x = self.conv5(self.lem3(x))
        # Transformer + pooling
        x = self.sppf(self.ectb(x))
        # Attention head (SA_NAM inside)
        x = self.aph(x)
        # Classify
        return self.fc(self.drop(self.gap(x).flatten(1)))

    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def blend_weight(self) -> float:
        """Returns the current effective scale-blend alpha = sigmoid(w)."""
        return torch.sigmoid(self.aph.nam.w).item()


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = RTDNetClassifier(num_classes=30, base_ch=32).to(device)
    total, trainable = model.count_parameters()
    print(f"Total params     : {total:,}")
    print(f"Trainable params : {trainable:,}")
    print(f"Model size (MB)  : {total * 4 / 1024**2:.2f}")
    print(f"SA_NAM overhead  : +1 parameter (blend scalar w)\n")

    dummy = torch.randn(4, 3, 224, 224).to(device)
    with torch.no_grad():
        out = model(dummy)
    print(f"Output shape     : {out.shape}")
    print(f"Initial alpha (blend weight) : {model.blend_weight():.4f}  (0.5 = equal mix)")

    # Latency
    dummy1 = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(5): model(dummy1)
        t0 = time.perf_counter()
        for _ in range(100): model(dummy1)
    ms = (time.perf_counter() - t0) / 100 * 1000
    print(f"Avg latency      : {ms:.1f} ms  (~{1000/ms:.0f} FPS @ bs=1, 224px)")

    print("\nSA_NAM model passed all checks!")