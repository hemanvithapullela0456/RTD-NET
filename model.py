"""
model.py — RTDNet adapted for image classification
Paper: "Real-Time Object Detection Network in UAV-Vision Based on CNN and Transformer"
Ye et al., IEEE TIM Vol. 72, 2023

Architecture (classification variant):
    Input (3, 640, 640)
    ├── Stem:    Conv_1 (stride 2) → Conv_2 (stride 2)         → /4
    ├── Stage 1: LEM_1 → Conv_3 (stride 2)                     → /8
    ├── Stage 2: LEM_2 → Conv_4 (stride 2)                     → /16
    ├── Stage 3: LEM_3 → Conv_5 (stride 2)                     → /32
    ├── Stage 4: ECTB (transformer stage)
    ├── SPPF    (Spatial Pyramid Pooling - Fast)
    ├── APH     (Attention Prediction Head)
    ├── GAP     (Global Average Pooling)
    └── FC      → num_classes

For non-640 inputs (e.g. 224 or 256), the spatial dims scale accordingly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import ConvBNSiLU, LEM, ECTB, APH


# ---------------------------------------------------------------------------
# SPPF — Spatial Pyramid Pooling Fast  (same as YOLOv5's SPPF)
# ---------------------------------------------------------------------------
class SPPF(nn.Module):
    """
    Concatenates the output of three consecutive max-pools (each 5×5) with the
    input to build multi-scale features efficiently.
    """
    def __init__(self, in_ch, out_ch, pool_size=5):
        super().__init__()
        mid_ch = in_ch // 2
        self.cv1 = ConvBNSiLU(in_ch,  mid_ch, 1)
        self.cv2 = ConvBNSiLU(mid_ch * 4, out_ch, 1)
        self.pool = nn.MaxPool2d(pool_size, 1, pool_size // 2)

    def forward(self, x):
        x  = self.cv1(x)
        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        return self.cv2(torch.cat([x, p1, p2, p3], dim=1))


# ---------------------------------------------------------------------------
# RTDNet Classifier
# ---------------------------------------------------------------------------
class RTDNetClassifier(nn.Module):
    """
    RTD-Net backbone repurposed for scene / remote-sensing image classification.

    Args:
        num_classes (int): Number of output classes. Default 30 (AID dataset).
        base_ch     (int): Base channel width. Default 32. Scale up for more capacity.
        num_heads   (int): Attention heads in ECTB / CMHSA. Default 4.
        C           (int): Number of branches in LEM. Default 16.
        dropout     (float): Dropout before the FC head. Default 0.3.
    """
    def __init__(self, num_classes=30, base_ch=32, num_heads=4, C=16, dropout=0.3):
        super().__init__()
        b = base_ch   # channel shorthand

        # ---- Stem: two strided convolutions (×4 downsampling) ----
        self.conv1 = ConvBNSiLU(3,    b,    3, stride=2)   # /2
        self.conv2 = ConvBNSiLU(b,    b*2,  3, stride=2)   # /4

        # ---- Stage 1 ----
        self.lem1  = LEM(b*2,  b*2,  C=C)
        self.conv3 = ConvBNSiLU(b*2, b*4,  3, stride=2)   # /8

        # ---- Stage 2 ----
        self.lem2  = LEM(b*4,  b*4,  C=C)
        self.conv4 = ConvBNSiLU(b*4, b*8,  3, stride=2)   # /16

        # ---- Stage 3 ----
        self.lem3  = LEM(b*8,  b*8,  C=C)
        self.conv5 = ConvBNSiLU(b*8, b*16, 3, stride=2)   # /32

        # ---- ECTB (transformer stage) ----
        # Safe num_heads: b*16 must be divisible by num_heads
        ectb_ch    = b * 16
        safe_heads = num_heads
        while ectb_ch % safe_heads != 0:
            safe_heads = max(1, safe_heads // 2)
        self.ectb  = ECTB(ectb_ch, ectb_ch, num_heads=safe_heads)

        # ---- SPPF ----
        self.sppf  = SPPF(b*16, b*16)

        # ---- APH (Attention Prediction Head) ----
        self.aph   = APH(b*16, b*16)

        # ---- Classification head ----
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.drop   = nn.Dropout(dropout)
        self.fc     = nn.Linear(b*16, num_classes)

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

    def forward(self, x):
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

        # Transformer + pooling
        x = self.ectb(x)
        x = self.sppf(x)

        # Attention head
        x = self.aph(x)

        # Global pooling + FC
        x = self.gap(x).flatten(1)
        x = self.drop(x)
        x = self.fc(x)
        return x

    def count_parameters(self):
        total  = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = RTDNetClassifier(num_classes=30, base_ch=32).to(device)
    total, trainable = model.count_parameters()
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Model size (MB):  {total * 4 / 1024**2:.2f}")

    # Throughput test
    dummy = torch.randn(1, 3, 640, 640).to(device)
    with torch.no_grad():
        # Warm-up
        for _ in range(3):
            _ = model(dummy)
        # Timing
        t0 = time.time()
        for _ in range(20):
            out = model(dummy)
        elapsed = (time.time() - t0) / 20
    print(f"Output shape:     {out.shape}")
    print(f"Avg inference:    {elapsed*1000:.1f} ms  |  ~{1/elapsed:.0f} FPS")

    # Also test with 224×224 (faster training)
    dummy224 = torch.randn(4, 3, 224, 224).to(device)
    with torch.no_grad():
        out224 = model(dummy224)
    print(f"224×224 output:   {out224.shape}")