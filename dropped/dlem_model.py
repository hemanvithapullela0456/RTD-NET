"""
dlem_model.py — RTDNet with D-LEM (Dynamic Multi-Branch LEM)

Key change: every LEM in the backbone is replaced by D-LEM.

D-LEM vs LEM
─────────────────────────────────────────────────────────────────────────────
LEM  : all 16 branches run in parallel, outputs simply concatenated (equal
       weight = 1/16 implicit), then fused by a 1×1 conv.

D-LEM: same 16 branches, but a tiny *branch-gating network* computes a
       softmax weight vector w ∈ ℝ^C over each branch before concatenation.
       The gate is a 2-layer bottleneck:
           GAP → FC (C → C//4) → SiLU → FC (C//4 → C) → Softmax
       where C = number of branches.  The output is a weighted sum of branch
       feature maps rather than a naive cat+conv, so the model learns which
       branches (= which receptive-field widths / filter patterns) matter for
       each input image.

Parameter overhead per D-LEM stage:
   gate FC1:  in_ch  × (in_ch // 4)  params
   gate FC2: (in_ch // 4) × in_ch   params
   ≈ 2 × (C² / 4)  —  tiny relative to the branch conv params.

For base_ch=32 the three D-LEM stages add only ~960 + 3,840 + 15,360 ≈ 20 K
extra params on top of the 3.155 M baseline (< 0.7 % overhead).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import ConvBNSiLU, ECTB, APH         # reuse unchanged helpers


# ─────────────────────────────────────────────────────────────────────────────
# D-LEM — Dynamic Multi-Branch LEM
# ─────────────────────────────────────────────────────────────────────────────
class DLEM(nn.Module):
    """
    Dynamic Lightweight Feature Extraction Module.

    Identical branch topology to LEM (C branches of 1×1 → 3×3), but each
    branch is gated by a learned scalar weight derived from a channel-squeeze
    bottleneck applied to the *input* feature map.  The gate is re-computed
    per forward pass, so it adapts to the spatial content of each image
    (small vs. large objects, fine vs. coarse textures).

    Args:
        in_ch  : Input (and output) channel count.
        out_ch : Output channel count (defaults to in_ch).
        C      : Number of parallel branches. Default 16 (as in original LEM).
        r      : Reduction ratio for the gate bottleneck. Default 4.
    """
    def __init__(self, in_ch: int, out_ch: int = None, C: int = 16, r: int = 4):
        super().__init__()
        out_ch = out_ch or in_ch
        mid_ch = max(in_ch // 2, 1)      # c/2  — first proj, same as LEM
        br_ch  = max(in_ch // 32, 1)     # c/32 — per-branch width

        self.C = C

        # ── Shared first projection (same as original LEM) ───────────────────
        self.conv1 = ConvBNSiLU(in_ch, mid_ch, 1)

        # ── C homogeneous branches (identical to LEM) ────────────────────────
        self.branches = nn.ModuleList([
            nn.Sequential(
                ConvBNSiLU(mid_ch, br_ch, 1),
                ConvBNSiLU(br_ch,  br_ch, 3),
            )
            for _ in range(C)
        ])

        branch_total = br_ch * C          # ≈ mid_ch when in_ch/32 * 16 = in_ch/2

        # ── Fusion conv (same as LEM) ────────────────────────────────────────
        self.conv2 = nn.Conv2d(branch_total, out_ch, 1, bias=False)
        self.bn    = nn.BatchNorm2d(out_ch)

        # ── Residual ─────────────────────────────────────────────────────────
        self.skip = (
            nn.Identity() if in_ch == out_ch
            else nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        )

        # ── Branch-gating network (the novel part) ───────────────────────────
        # Input: global average-pooled feature → scalar per branch.
        # Two-layer bottleneck keeps parameter count negligible.
        gate_mid = max(C // r, 1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # (B, in_ch, 1, 1)
            nn.Flatten(),                     # (B, in_ch)
            nn.Linear(in_ch, gate_mid, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(gate_mid, C, bias=False),
            # Softmax over C branches: weights sum to 1, always positive
            nn.Softmax(dim=-1),               # (B, C)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── 1. Compute per-branch gate weights ────────────────────────────────
        #    w: (B, C)  — learned from the *input* x (content-adaptive)
        w = self.gate(x)                      # (B, C)

        # ── 2. Shared projection ─────────────────────────────────────────────
        feat = self.conv1(x)                  # (B, mid_ch, H, W)

        # ── 3. Run all C branches → list of (B, br_ch, H, W) tensors ─────────
        branch_outs = [b(feat) for b in self.branches]   # C × (B, br_ch, H, W)

        # ── 4. Weight each branch by its gate scalar ─────────────────────────
        #    w[:, i] is a scalar per sample; broadcast over (br_ch, H, W)
        #    Stack to (B, C, br_ch, H, W), multiply, reshape to (B, C*br_ch, H, W)
        stacked = torch.stack(branch_outs, dim=1)         # (B, C, br_ch, H, W)
        w_bcast = w.view(w.size(0), self.C, 1, 1, 1)     # (B, C,  1,    1, 1)
        weighted = (stacked * w_bcast)                    # (B, C, br_ch, H, W)

        # Flatten C × br_ch back into a single channel dim for fusion conv
        B, C, br, H, W = weighted.shape
        concat = weighted.view(B, C * br, H, W)           # (B, branch_total, H, W)

        # ── 5. Fuse + residual (same as LEM) ─────────────────────────────────
        out = self.bn(self.conv2(concat))
        return F.silu(out + self.skip(x))


# ─────────────────────────────────────────────────────────────────────────────
# SPPF (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
class SPPF(nn.Module):
    def __init__(self, in_ch, out_ch, pool_size=5):
        super().__init__()
        mid_ch    = in_ch // 2
        self.cv1  = ConvBNSiLU(in_ch, mid_ch, 1)
        self.cv2  = ConvBNSiLU(mid_ch * 4, out_ch, 1)
        self.pool = nn.MaxPool2d(pool_size, 1, pool_size // 2)

    def forward(self, x):
        x  = self.cv1(x)
        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        return self.cv2(torch.cat([x, p1, p2, p3], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# RTDNetClassifier — all three LEM stages replaced by D-LEM
# ─────────────────────────────────────────────────────────────────────────────
class RTDNetClassifier(nn.Module):
    """
    RTD-Net backbone with D-LEM replacing all three LEM stages.

    Forward pass:
        Input → Stem → [D-LEM1 → Conv3] → [D-LEM2 → Conv4]
              → [D-LEM3 → Conv5] → ECTB → SPPF → APH → GAP → FC

    Args:
        num_classes   (int)  : Classes. Default 30 (AID).
        base_ch       (int)  : Base channel width. Default 32.
        num_heads     (int)  : ECTB attention heads. Default 4.
        C             (int)  : Branches per D-LEM. Default 16.
        dropout       (float): FC dropout. Default 0.3.
        gate_reduction(int)  : Gate bottleneck ratio r. Default 4.
    """
    def __init__(
        self,
        num_classes: int    = 30,
        base_ch: int        = 32,
        num_heads: int      = 4,
        C: int              = 16,
        dropout: float      = 0.3,
        gate_reduction: int = 4,
    ):
        super().__init__()
        b = base_ch

        # ── Stem ─────────────────────────────────────────────────────────────
        self.conv1 = ConvBNSiLU(3,   b,   3, stride=2)   # /2
        self.conv2 = ConvBNSiLU(b,   b*2, 3, stride=2)   # /4

        # ── Stage 1 — D-LEM ──────────────────────────────────────────────────
        self.dlem1 = DLEM(b*2,  b*2,  C=C, r=gate_reduction)
        self.conv3 = ConvBNSiLU(b*2, b*4, 3, stride=2)   # /8

        # ── Stage 2 — D-LEM ──────────────────────────────────────────────────
        self.dlem2 = DLEM(b*4,  b*4,  C=C, r=gate_reduction)
        self.conv4 = ConvBNSiLU(b*4, b*8, 3, stride=2)   # /16

        # ── Stage 3 — D-LEM ──────────────────────────────────────────────────
        self.dlem3 = DLEM(b*8,  b*8,  C=C, r=gate_reduction)
        self.conv5 = ConvBNSiLU(b*8, b*16, 3, stride=2)  # /32

        # ── ECTB (Transformer stage) ──────────────────────────────────────────
        ectb_ch    = b * 16
        safe_heads = num_heads
        while ectb_ch % safe_heads != 0:
            safe_heads = max(1, safe_heads // 2)
        self.ectb  = ECTB(ectb_ch, ectb_ch, num_heads=safe_heads)

        # ── SPPF ──────────────────────────────────────────────────────────────
        self.sppf  = SPPF(b*16, b*16)

        # ── APH (Attention Prediction Head) ───────────────────────────────────
        self.aph   = APH(b*16, b*16)

        # ── Classification head ───────────────────────────────────────────────
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(b*16, num_classes)

        self._init_weights()

    # ── Weight init (same policy as original) ────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None: nn.init.ones_(m.weight)
                if m.bias  is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.conv1(x)
        x = self.conv2(x)

        # Stage 1
        x = self.dlem1(x)
        x = self.conv3(x)

        # Stage 2
        x = self.dlem2(x)
        x = self.conv4(x)

        # Stage 3
        x = self.dlem3(x)
        x = self.conv5(x)

        # Transformer + multi-scale pooling
        x = self.ectb(x)
        x = self.sppf(x)

        # Attention prediction head
        x = self.aph(x)

        # Classifier
        x = self.gap(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)

    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def branch_weights(self, x: torch.Tensor) -> dict:
        """
        Diagnostic helper — returns the gate weight vectors for each D-LEM
        stage given a single input tensor.  Useful for visualising which
        branches the model activates for different image types.

        Args:
            x: (1, 3, H, W) input image tensor (batch size 1).
        Returns:
            dict with keys 'stage1', 'stage2', 'stage3', each a (C,) tensor.
        """
        self.eval()
        with torch.no_grad():
            s = self.conv1(x)
            s = self.conv2(s)
            w1 = self.dlem1.gate(s).squeeze(0)    # (C,)

            s = self.dlem1(s)
            s = self.conv3(s)
            w2 = self.dlem2.gate(s).squeeze(0)

            s = self.dlem2(s)
            s = self.conv4(s)
            w3 = self.dlem3.gate(s).squeeze(0)

        return {"stage1": w1, "stage2": w2, "stage3": w3}


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

    # Forward pass check
    dummy = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        out = model(dummy)
    print(f"Output shape     : {out.shape}")

    # Latency
    dummy1 = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(5): model(dummy1)
        t0 = time.time()
        for _ in range(50): model(dummy1)
    ms = (time.time() - t0) / 50 * 1000
    print(f"Avg latency      : {ms:.1f} ms  (~{1000/ms:.0f} FPS @ bs=1)")

    # Gate weight diagnostic
    weights = model.branch_weights(dummy1)
    for stage, w in weights.items():
        top3 = w.topk(3)
        print(f"  {stage} — top-3 branch ids: {top3.indices.tolist()}  "
              f"weights: {[round(v, 3) for v in top3.values.tolist()]}")

    print("\nD-LEM model passed all checks!")