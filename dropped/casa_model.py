"""
casa_model.py — RTDNetClassifier using CASA attention (CMHSA replacement)

Drop-in replacement for model.py.  Only one import line changes:
    was:  from models import ConvBNSiLU, LEM, ECTB, APH
    now:  from casa_models import ConvBNSiLU, LEM, ECTB, APH

Everything else — SPPF, RTDNetClassifier, train.py, dataset.py — untouched.

To train:
    python train.py --data data/aid --dataset aid
    (just ensure train.py imports from casa_model instead of model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ← Only change vs model.py: source module
from dropped.casa_models import ConvBNSiLU, LEM, ECTB, APH


# ──────────────────────────────────────────────────────────────────────────────
# SPPF  (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────────
# RTDNetClassifier  (identical to model.py — ECTB now uses CASA internally)
# ──────────────────────────────────────────────────────────────────────────────
class RTDNetClassifier(nn.Module):
    """
    RTD-Net classification backbone with CASA attention in ECTB.

    Forward pass (unchanged from original):
        Input → Stem → Stage1(LEM+Conv) → Stage2(LEM+Conv) → Stage3(LEM+Conv)
              → ECTB[CASA] → SPPF → APH → GAP → FC

    Args:
        num_classes (int)  : Output classes. Default 30 (AID).
        base_ch     (int)  : Base channel width. Default 32.
        num_heads   (int)  : Attention heads inside CASA. Default 4.
        C           (int)  : LEM branches. Default 16.
        dropout     (float): FC dropout. Default 0.3.
        attn_drop   (float): Attention weight dropout inside CASA. Default 0.0.
    """
    def __init__(
        self,
        num_classes : int   = 30,
        base_ch     : int   = 32,
        num_heads   : int   = 4,
        C           : int   = 16,
        dropout     : float = 0.3,
        attn_drop   : float = 0.0,
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

        # ECTB — internally now uses CASA
        ectb_ch    = b * 16
        safe_heads = num_heads
        while ectb_ch % safe_heads != 0:
            safe_heads = max(1, safe_heads // 2)
        self.ectb  = ECTB(ectb_ch, ectb_ch, num_heads=safe_heads, attn_drop=attn_drop)

        # Pooling + head
        self.sppf  = SPPF(b*16, b*16)
        self.aph   = APH(b*16, b*16)
        self.gap   = nn.AdaptiveAvgPool2d(1)
        self.drop  = nn.Dropout(dropout)
        self.fc    = nn.Linear(b*16, num_classes)

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
        x = self.conv2(self.conv1(x))
        x = self.conv3(self.lem1(x))
        x = self.conv4(self.lem2(x))
        x = self.conv5(self.lem3(x))
        x = self.ectb(x)
        x = self.sppf(x)
        x = self.aph(x)
        return self.fc(self.drop(self.gap(x).flatten(1)))

    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ──────────────────────────────────────────────────────────────────────────────
# Sanity check
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = RTDNetClassifier(num_classes=30, base_ch=32).to(device)
    total, trainable = model.count_parameters()
    print(f"Total params     : {total:,}")
    print(f"Trainable params : {trainable:,}")
    print(f"Model size (MB)  : {total * 4 / 1024**2:.2f}\n")

    for res in [224, 640]:
        dummy = torch.randn(1, 3, res, res).to(device)
        with torch.no_grad():
            for _ in range(3): model(dummy)
            t0 = time.perf_counter()
            for _ in range(50): model(dummy)
        ms = (time.perf_counter() - t0) / 50 * 1000
        print(f"  {res}×{res} → {ms:.1f} ms/img  (~{1000/ms:.0f} FPS)")

    print("\ncasa_model.py passed all checks!")