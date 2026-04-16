"""
rtdnet/adm/model.py — RTS-Net classification model (Paper 2)

Paper: "Urban Traffic Tiny Object Detection via Attention and Multi-Scale
        Feature Driven in UAV-Vision"
        Wang et al., Scientific Reports, 2024

Architecture:
    Input (3, H, W)
    ├── Stem   : Conv_1 (stride 2) → Conv_2 (stride 2)         /4
    ├── Stage 1: RFEM → Conv_3 (stride 2)                       /8
    ├── Stage 2: RFEM → Conv_4 (stride 2)                       /16
    ├── Stage 3: RFEM → Conv_5 (stride 2)                       /32
    ├── ECTB   : transformer stage
    ├── SPPF   : Spatial Pyramid Pooling Fast
    ├── CADM   : Coordinated Attention Detection Module
    ├── GAP    : Global Average Pooling
    └── FC     → num_classes

Differences from the original RTD-Net (rtdnet/model.py):
  • RFEM replaces LEM at every feature extraction stage
  • CADM replaces APH as the attention head before classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import ConvBNSiLU, RFEM, ECTB, CADM


# ---------------------------------------------------------------------------
# SPPF — Spatial Pyramid Pooling Fast (unchanged from Paper 1)
# ---------------------------------------------------------------------------
class SPPF(nn.Module):
    def __init__(self, in_ch, out_ch, pool_size=5):
        super().__init__()
        mid_ch    = in_ch // 2
        self.cv1  = ConvBNSiLU(in_ch,     mid_ch,   1)
        self.cv2  = ConvBNSiLU(mid_ch * 4, out_ch,  1)
        self.pool = nn.MaxPool2d(pool_size, 1, pool_size // 2)

    def forward(self, x):
        x  = self.cv1(x)
        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        return self.cv2(torch.cat([x, p1, p2, p3], dim=1))


# ---------------------------------------------------------------------------
# RTS-Net Classifier
# ---------------------------------------------------------------------------
class RTSNetClassifier(nn.Module):
    """
    RTS-Net backbone repurposed for image classification.

    Args:
        num_classes : output classes (default 30 for AID).
        base_ch     : base channel width — scales all layer widths.
        num_heads   : attention heads for ECTB/CMHSA.
        C           : parallel branches in RFEM (paper uses 16).
        dropout     : dropout probability before FC head.
    """
    def __init__(self, num_classes=30, base_ch=32, num_heads=4, C=16, dropout=0.3):
        super().__init__()
        b = base_ch

        # ---- Stem: 4× downsampling ----
        self.conv1 = ConvBNSiLU(3,    b,    3, stride=2)   # /2
        self.conv2 = ConvBNSiLU(b,    b*2,  3, stride=2)   # /4

        # ---- Stage 1: RFEM + strided conv ----
        self.rfem1 = RFEM(b*2,  b*2,  C=C)
        self.conv3 = ConvBNSiLU(b*2,  b*4,  3, stride=2)   # /8

        # ---- Stage 2: RFEM + strided conv ----
        self.rfem2 = RFEM(b*4,  b*4,  C=C)
        self.conv4 = ConvBNSiLU(b*4,  b*8,  3, stride=2)   # /16

        # ---- Stage 3: RFEM + strided conv ----
        self.rfem3 = RFEM(b*8,  b*8,  C=C)
        self.conv5 = ConvBNSiLU(b*8,  b*16, 3, stride=2)   # /32

        # ---- ECTB: global context via CMHSA (unchanged from Paper 1) ----
        ectb_ch    = b * 16
        safe_heads = num_heads
        while ectb_ch % safe_heads != 0:
            safe_heads = max(1, safe_heads // 2)
        self.ectb  = ECTB(ectb_ch, ectb_ch, num_heads=safe_heads)

        # ---- SPPF: multi-scale pooling (unchanged from Paper 1) ----
        self.sppf  = SPPF(b*16, b*16)

        # ---- CADM: coordinated attention head (replaces APH/NAM) ----
        self.cadm  = CADM(b*16, b*16)

        # ---- Classification head ----
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(b*16, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
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

    def forward(self, x):
        # Stem
        x = self.conv2(self.conv1(x))

        # Stage 1
        x = self.conv3(self.rfem1(x))

        # Stage 2
        x = self.conv4(self.rfem2(x))

        # Stage 3
        x = self.conv5(self.rfem3(x))

        # Transformer + multi-scale pooling
        x = self.sppf(self.ectb(x))

        # Coordinated attention head
        x = self.cadm(x)

        # Classify
        x = self.fc(self.drop(self.gap(x).flatten(1)))
        return x

    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = RTSNetClassifier(num_classes=30, base_ch=32).to(device)
    total, trainable = model.count_parameters()
    print(f"Total params     : {total:,}")
    print(f"Trainable params : {trainable:,}")
    print(f"Model size (MB)  : {total * 4 / 1024**2:.2f}")

    dummy = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(3):          # warm-up
            _ = model(dummy)
        t0 = time.time()
        for _ in range(20):
            out = model(dummy)
        elapsed = (time.time() - t0) / 20

    print(f"Output shape     : {out.shape}")
    print(f"Avg latency      : {elapsed*1000:.1f} ms  (~{1/elapsed:.0f} FPS)")