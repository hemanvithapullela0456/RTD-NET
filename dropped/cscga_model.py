"""
cscga_model.py — RTDNet + CSCGA (Collaborative Spatial-Channel Group Attention)

New module: CSCGA
  Inserted AFTER the last LEM stage (stage 3 / conv5 output) and BEFORE APH.

Architecture of CSCGA:
  1. Split channels into G=4 groups
  2. Depthwise 3×3 conv per group (grouped conv, groups=C)
  3. Coordinate attention:
       - Pool along H → (B, C, 1, W), along W → (B, C, H, 1)
       - Concat → (B, C, 1, H+W) → 1×1 conv (C → C//r) → SiLU
       - Split back → two 1×1 convs → sigmoid → channel-wise scale
  4. NAM-style channel attention (BN γ weights) fused via addition before sigmoid
  5. All ops are lightweight (GroupConv, no FC expansion)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import ConvBNSiLU, LEM, ECTB, APH, NAM


# ─────────────────────────────────────────────────────────────────────────────
# CSCGA — Collaborative Spatial-Channel Group Attention
# ─────────────────────────────────────────────────────────────────────────────
class CSCGA(nn.Module):
    """
    Collaborative Spatial-Channel Group Attention.

    Args:
        channels (int): Number of input (and output) channels C.
        num_groups (int): Number of channel groups G. Default 4.
        reduction (int): Reduction ratio r for coordinate-attention mid-channels. Default 8.
    """
    def __init__(self, channels: int, num_groups: int = 4, reduction: int = 8):
        super().__init__()
        assert channels % num_groups == 0, "channels must be divisible by num_groups"
        self.C = channels
        self.G = num_groups
        mid = max(channels // reduction, 8)   # coordinate-attention bottleneck

        # ── 1. Group depthwise conv (3×3, groups=C for full depthwise) ──────
        #    Applied per-group by using groups=C (one filter per channel).
        self.dw_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                      groups=channels, bias=False),           # depthwise
            nn.Conv2d(channels, channels, kernel_size=1,
                      groups=num_groups, bias=False),         # group pointwise
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

        # ── 2. Coordinate Attention ──────────────────────────────────────────
        # Encode H and W directions separately then fuse
        self.ca_pool_h  = nn.AdaptiveAvgPool2d((None, 1))  # (B,C,H,1)
        self.ca_pool_w  = nn.AdaptiveAvgPool2d((1, None))  # (B,C,1,W)

        # Shared encoding: concat along spatial → reduce channels
        self.ca_encode  = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
        )
        # Decode back to C for each direction
        self.ca_decode_h = nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        self.ca_decode_w = nn.Conv2d(mid, channels, kernel_size=1, bias=False)

        # ── 3. NAM-style channel attention (BN γ) ───────────────────────────
        self.nam_bn = nn.BatchNorm2d(channels)

        # ── 4. Fusion: learned scalar blend between coord-attn & NAM-attn ───
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

        # ── 5. Lightweight output conv (channel mixing after attention) ──────
        self.out_conv = ConvBNSiLU(channels, channels, 1)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _nam_attention(self, x: torch.Tensor) -> torch.Tensor:
        """NAM channel attention using BN γ as importance weights."""
        normed = self.nam_bn(x)
        gamma  = self.nam_bn.weight.abs()          # (C,)
        w      = gamma / (gamma.sum() + 1e-8)      # normalise
        w      = w.view(1, -1, 1, 1)
        return torch.sigmoid(w * normed)           # (B,C,H,W)

    def _coord_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Coordinate attention producing a (B,C,H,W) spatial-channel map."""
        B, C, H, W = x.shape

        # Pool along each axis
        x_h = self.ca_pool_h(x)                   # (B,C,H,1)
        x_w = self.ca_pool_w(x).permute(0,1,3,2)  # (B,C,W,1) → transpose to (B,C,W,1)

        # Concat along H dimension: (B, C, H+W, 1)
        y   = torch.cat([x_h, x_w], dim=2)        # (B,C,H+W,1)
        y   = self.ca_encode(y)                    # (B,mid,H+W,1)

        # Split back
        y_h, y_w = torch.split(y, [H, W], dim=2)  # (B,mid,H,1) and (B,mid,W,1)

        # Decode + sigmoid
        a_h = torch.sigmoid(self.ca_decode_h(y_h))            # (B,C,H,1)
        a_w = torch.sigmoid(self.ca_decode_w(y_w.permute(0,1,3,2)))  # (B,C,1,W)

        # Broadcast multiply: produces (B,C,H,W)
        return a_h * a_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Group depthwise conv (local spatial feature enrichment)
        feat = self.dw_conv(x)                     # (B,C,H,W)

        # 2. Coordinate attention map
        ca_map  = self._coord_attention(feat)      # (B,C,H,W)

        # 3. NAM channel attention map
        nam_map = self._nam_attention(feat)        # (B,C,H,W)

        # 4. Fuse coord + NAM attention (learned convex combination)
        w = torch.sigmoid(self.fusion_weight)
        attn = w * ca_map + (1.0 - w) * nam_map   # (B,C,H,W)

        # 5. Apply attention + residual + output conv
        out = self.out_conv(feat * attn + x)       # residual from original x
        return out


# ─────────────────────────────────────────────────────────────────────────────
# SPPF (unchanged from model.py)
# ─────────────────────────────────────────────────────────────────────────────
class SPPF(nn.Module):
    def __init__(self, in_ch, out_ch, pool_size=5):
        super().__init__()
        mid_ch = in_ch // 2
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
# RTDNetClassifier — updated with CSCGA
# ─────────────────────────────────────────────────────────────────────────────
class RTDNetClassifier(nn.Module):
    """
    RTD-Net backbone + CSCGA.

    Insertion point:
        … → LEM3 → Conv5 → [CSCGA] → ECTB → SPPF → APH → GAP → FC

    Args:
        num_classes  (int)  : Output classes. Default 30 (AID).
        base_ch      (int)  : Base channel width. Default 32.
        num_heads    (int)  : ECTB/CMHSA heads. Default 4.
        C            (int)  : LEM branches. Default 16.
        dropout      (float): Dropout before FC. Default 0.3.
        cscga_groups (int)  : CSCGA channel groups G. Default 4.
        cscga_reduction(int): CSCGA coordinate-attn reduction ratio. Default 8.
    """
    def __init__(
        self,
        num_classes: int   = 30,
        base_ch: int       = 32,
        num_heads: int     = 4,
        C: int             = 16,
        dropout: float     = 0.3,
        cscga_groups: int  = 4,
        cscga_reduction: int = 8,
    ):
        super().__init__()
        b = base_ch

        # ── Stem ────────────────────────────────────────────────────────────
        self.conv1 = ConvBNSiLU(3,   b,    3, stride=2)   # /2
        self.conv2 = ConvBNSiLU(b,   b*2,  3, stride=2)   # /4

        # ── Stage 1 ─────────────────────────────────────────────────────────
        self.lem1  = LEM(b*2,  b*2,  C=C)
        self.conv3 = ConvBNSiLU(b*2, b*4,  3, stride=2)   # /8

        # ── Stage 2 ─────────────────────────────────────────────────────────
        self.lem2  = LEM(b*4,  b*4,  C=C)
        self.conv4 = ConvBNSiLU(b*4, b*8,  3, stride=2)   # /16

        # ── Stage 3 ─────────────────────────────────────────────────────────
        self.lem3  = LEM(b*8,  b*8,  C=C)
        self.conv5 = ConvBNSiLU(b*8, b*16, 3, stride=2)   # /32

        # ── CSCGA (NEW) — after last LEM/conv5, before ECTB ─────────────────
        cscga_ch = b * 16
        # Ensure num_groups divides channels
        g = cscga_groups
        while cscga_ch % g != 0:
            g = max(1, g // 2)
        self.cscga = CSCGA(cscga_ch, num_groups=g, reduction=cscga_reduction)

        # ── ECTB (transformer stage) ─────────────────────────────────────────
        ectb_ch    = b * 16
        safe_heads = num_heads
        while ectb_ch % safe_heads != 0:
            safe_heads = max(1, safe_heads // 2)
        self.ectb  = ECTB(ectb_ch, ectb_ch, num_heads=safe_heads)

        # ── SPPF ─────────────────────────────────────────────────────────────
        self.sppf  = SPPF(b*16, b*16)

        # ── APH ──────────────────────────────────────────────────────────────
        self.aph   = APH(b*16, b*16)

        # ── Classification head ───────────────────────────────────────────────
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(b*16, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.conv1(x)
        x = self.conv2(x)

        # Stage 1
        x = self.lem1(x)
        x = self.conv3(x)

        # Stage 2
        x = self.lem2(x)
        x = self.conv4(x)

        # Stage 3
        x = self.lem3(x)
        x = self.conv5(x)

        # ── CSCGA (new) ──────────────────────────────────────────────────────
        x = self.cscga(x)

        # Transformer + multi-scale pooling
        x = self.ectb(x)
        x = self.sppf(x)

        # Attention prediction head
        x = self.aph(x)

        # Global pool + classify
        x = self.gap(x).flatten(1)
        x = self.drop(x)
        x = self.fc(x)
        return x

    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


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
    print(f"Model size (MB)  : {total * 4 / 1024**2:.2f}\n")

    dummy = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        out = model(dummy)
    print(f"Output shape     : {out.shape}")

    # Throughput
    dummy1 = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(5): model(dummy1)           # warm-up
        t0 = time.time()
        for _ in range(50): model(dummy1)
    ms = (time.time() - t0) / 50 * 1000
    print(f"Avg latency      : {ms:.1f} ms  (~{1000/ms:.0f} FPS @ bs=1)")

    print("\nCSCGA module test passed!")