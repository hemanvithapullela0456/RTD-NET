"""
rtdnet/adm/models.py — Building blocks for the RTS-Net variant (Paper 2)

Paper: "Urban Traffic Tiny Object Detection via Attention and Multi-Scale
        Feature Driven in UAV-Vision"
        Wang et al., Scientific Reports, 2024

Modules:
  - ConvBNSiLU : shared conv helper
  - RFEM       : Real-time Feature Extraction Module  (replaces LEM from Paper 1)
  - CMHSA      : Convolutional Multi-Head Self-Attention  (unchanged from Paper 1)
  - ECTB       : Efficient Convolutional Transformer Block (unchanged from Paper 1)
  - CADM       : Coordinated Attention Detection Module   (replaces APH from Paper 1)

Key architectural differences vs Paper 1 modules:
──────────────────────────────────────────────────
RFEM vs LEM
  LEM  : conv1(c→c/2) → every branch independently receives full c/2 map
         and projects it down c/2→c/32 via 1×1 before 3×3. C=16 branches.
  RFEM : conv1(c→c) → split into y1(c/2) + y2(c/2)
         → y1 sliced into C equal chunks (c/32 each, no per-branch 1×1)
         → y2 bypasses all branches (direct shortcut)
         → concat [y1 | branch_outs | y2] → 1×1 fusion
         Result: 26.5% less compute, 30.9% faster inference (Paper 2 ablation).

CADM vs APH/NAM
  APH/NAM : attention from BN γ (channel) + InstanceNorm λ (spatial).
            No positional encoding — global statistics only.
  CADM    : Coordinate Attention — 1D avg-pool along H-axis and W-axis
            separately, producing direction-aware gates g_h and g_w.
            Output = x * g_h * g_w  (position-preserving spatial gating).
            Better tiny-object localisation in wide UAV field-of-view.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper: Conv → BN → SiLU
# ---------------------------------------------------------------------------
class ConvBNSiLU(nn.Module):
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


# ---------------------------------------------------------------------------
# RFEM — Real-time Feature Extraction Module  (Paper 2, Eq. 1-2, Fig. 2)
# ---------------------------------------------------------------------------
class RFEM(nn.Module):
    """
    Flow:
        x ──► conv1 (1×1, c→c)
              │
              ▼
           split along channel dim
           ┌────────┬────────┐
           y1(c/2)          y2(c/2)  ──────────────────────────┐
           │                                                     │
           split into C=16 slices (each c/32)                   │
           │                                                     │
           C parallel 3×3 convs (one per slice)                 │
           │                                                     │
           concat branch outputs  (c/2)                         │
           │                                                     │
           concat [y1 | branch_outs | y2]  (3c/2 total) ◄───────┘
           │
           conv2 (1×1, 3c/2 → c) → BN
           │
           + residual(x) → SiLU → output
    """
    def __init__(self, in_ch, out_ch=None, C=16):
        super().__init__()
        out_ch  = out_ch or in_ch
        half_ch = max(in_ch // 2, 1)
        br_ch   = max(in_ch // 32, 1)   # size of each y1 slice fed to one branch
        self.C  = C

        # Initial channel mixing before split
        self.conv1 = ConvBNSiLU(in_ch, in_ch, 1)

        # C parallel 3×3 convs — no 1×1 projection per branch (unlike LEM)
        self.branches = nn.ModuleList([
            ConvBNSiLU(br_ch, br_ch, 3)
            for _ in range(C)
        ])

        # Fusion of [y1 | branch_outs | y2] = c/2 + c/2 + c/2 = 3c/2
        self.conv2 = nn.Conv2d(half_ch + br_ch * C + half_ch, out_ch, 1, bias=False)
        self.bn    = nn.BatchNorm2d(out_ch)

        self.skip = nn.Identity() if in_ch == out_ch else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        # Eq. 1: split after initial conv
        feat    = self.conv1(x)
        y1, y2  = feat.chunk(2, dim=1)              # (B, c/2, H, W) each

        # Eq. 2: slice y1 into C equal parts, one per branch
        slices      = y1.chunk(self.C, dim=1)        # C × (B, c/32, H, W)
        branch_outs = [b(s) for b, s in zip(self.branches, slices)]

        # Concat [y1 | branch_outputs | y2]
        out = torch.cat([y1, *branch_outs, y2], dim=1)   # (B, 3c/2, H, W)
        out = self.bn(self.conv2(out))
        return F.silu(out + self.skip(x))


# ---------------------------------------------------------------------------
# CMHSA — Convolutional Multi-Head Self-Attention (Paper 1, unchanged)
# Kept here so adm/ is self-contained.
# ---------------------------------------------------------------------------
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

        attn = torch.einsum('bhdq,bhdT->bhqT', q, k) * self.scale
        attn = self.head_conv(attn)
        attn = self.inst_norm(F.softmax(attn, dim=-1))

        out = torch.einsum('bhqT,bhTd->bhqd',
                           attn, v.permute(0, 1, 3, 2))
        out = out.contiguous().view(B, T, C)
        out = self.proj(out).permute(0, 2, 1).view(B, C, H, W)
        return out


# ---------------------------------------------------------------------------
# ECTB — Efficient Convolutional Transformer Block (Paper 1, unchanged)
# ---------------------------------------------------------------------------
class ECTB(nn.Module):
    def __init__(self, in_ch, out_ch=None, num_heads=4):
        super().__init__()
        out_ch = out_ch or in_ch
        mid_ch = max(in_ch // 2, 1)

        self.conv1 = ConvBNSiLU(in_ch, mid_ch, 1)
        self.cmhsa = CMHSA(mid_ch, num_heads=num_heads)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn    = nn.BatchNorm2d(out_ch)

        self.skip = nn.Identity() if in_ch == out_ch else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.cmhsa(feat)
        feat = self.bn(self.conv2(feat))
        return F.silu(feat + self.skip(x))


# ---------------------------------------------------------------------------
# CoordinateAttention — core of CADM  (Paper 2, Eq. 6-11, Fig. 5)
# ---------------------------------------------------------------------------
class CoordinateAttention(nn.Module):
    """
    Encodes spatial context along H and W directions independently.

    Steps:
      1. Pool along W → z_h  (B, C, H, 1)   — captures vertical position
      2. Pool along H → z_w  (B, C, 1, W)   — captures horizontal position
      3. Concatenate along spatial dim → (B, C, H+W, 1)
      4. Shared 1×1 conv + BN + activation  → (B, C/r, H+W, 1)
      5. Split back into f_h (B, C/r, H, 1) and f_w (B, C/r, W, 1)
      6. Separate 1×1 convs → sigmoid gates g_h, g_w
      7. Output = x * g_h * g_w
    """
    def __init__(self, in_ch, r=32):
        super().__init__()
        mid_ch = max(in_ch // r, 8)

        # Shared transform for concatenated H+W features (Eq. 8)
        self.conv_shared = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.Hardswish(inplace=True),
        )

        # Direction-specific 1×1 convs (Eq. 9, 10)
        self.conv_h = nn.Conv2d(mid_ch, in_ch, 1, bias=False)   # Fh
        self.conv_w = nn.Conv2d(mid_ch, in_ch, 1, bias=False)   # Fw

    def forward(self, x):
        B, C, H, W = x.shape

        # Eq. 6: pool along W-axis → vertical encoding
        z_h = x.mean(dim=3, keepdim=True)              # (B, C, H, 1)

        # Eq. 7: pool along H-axis → horizontal encoding
        # Transpose to (B, C, W, 1) for uniform spatial processing
        z_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)  # (B, C, W, 1)

        # Eq. 8: concat + shared conv
        f   = self.conv_shared(torch.cat([z_h, z_w], dim=2))   # (B, mid, H+W, 1)

        # Split back
        f_h, f_w = f.split([H, W], dim=2)              # (B, mid, H, 1), (B, mid, W, 1)

        # Eq. 9-10: direction-specific gates
        g_h = torch.sigmoid(self.conv_h(f_h))          # (B, C, H, 1)
        g_w = torch.sigmoid(self.conv_w(f_w))          # (B, C, W, 1)
        g_w = g_w.permute(0, 1, 3, 2)                  # (B, C, 1, W)

        # Eq. 11: re-weight — broadcasts g_h over W and g_w over H
        return x * g_h * g_w


# ---------------------------------------------------------------------------
# CADM — Coordinated Attention Detection Module  (Paper 2, Section "CADM")
# ---------------------------------------------------------------------------
class CADM(nn.Module):
    """
    Drop-in replacement for APH.
    Wraps CoordinateAttention with a final ConvBNSiLU projection.
    """
    def __init__(self, in_ch, out_ch=None, r=32):
        super().__init__()
        out_ch          = out_ch or in_ch
        self.coord_attn = CoordinateAttention(in_ch, r=r)
        self.conv       = ConvBNSiLU(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(self.coord_attn(x))


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 64, 40, 40).to(device)

    print("--- RFEM ---")
    m = RFEM(64, 64).to(device)
    print(f"  {x.shape} → {m(x).shape}  | params: {sum(p.numel() for p in m.parameters()):,}")

    print("--- ECTB ---")
    m = ECTB(64, 64).to(device)
    print(f"  {x.shape} → {m(x).shape}  | params: {sum(p.numel() for p in m.parameters()):,}")

    print("--- CoordinateAttention ---")
    m = CoordinateAttention(64).to(device)
    print(f"  {x.shape} → {m(x).shape}  | params: {sum(p.numel() for p in m.parameters()):,}")

    print("--- CADM ---")
    m = CADM(64, 128).to(device)
    print(f"  {x.shape} → {m(x).shape}  | params: {sum(p.numel() for p in m.parameters()):,}")

    print("\nAll passed!")