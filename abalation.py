"""
ablation.py — Ablation study runner for RTD-Net components
Trains 5 model variants (matching Table IV in the paper) and prints a comparison table.

Variants:
  1. Baseline     : Yolov5s-style backbone (LEM only, no ECTB / APH)
  2. + ECTB       : Add ECTB transformer block
  3. + LEM + ECTB : Combined backbone
  4. + FFM (skip) : Feature fusion (detection only; here we log the combined result)
  5. Full RTD-Net : All modules (LEM + ECTB + APH)

Usage:
    python ablation.py --data data/aid --epochs 50 --batch_size 32
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

from model import RTDNetClassifier, SPPF
from models import ConvBNSiLU, LEM, ECTB, APH, NAM
from dataset import get_dataloaders


# ---------------------------------------------------------------------------
# Minimal baseline variants (to isolate each contribution)
# ---------------------------------------------------------------------------
class BaselineClassifier(nn.Module):
    """LEM-only backbone, no ECTB, no APH."""
    def __init__(self, num_classes=30, base_ch=32):
        super().__init__()
        b = base_ch
        self.conv1 = ConvBNSiLU(3,   b,   3, stride=2)
        self.conv2 = ConvBNSiLU(b,   b*2, 3, stride=2)
        self.lem1  = LEM(b*2,  b*2)
        self.conv3 = ConvBNSiLU(b*2, b*4, 3, stride=2)
        self.lem2  = LEM(b*4,  b*4)
        self.conv4 = ConvBNSiLU(b*4, b*8, 3, stride=2)
        self.lem3  = LEM(b*8,  b*8)
        self.conv5 = ConvBNSiLU(b*8, b*16, 3, stride=2)
        self.sppf  = SPPF(b*16, b*16)
        self.gap   = nn.AdaptiveAvgPool2d(1)
        self.fc    = nn.Linear(b*16, num_classes)

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x = self.conv3(self.lem1(x))
        x = self.conv4(self.lem2(x))
        x = self.conv5(self.lem3(x))
        x = self.sppf(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)


class ECTBOnlyClassifier(nn.Module):
    """Standard Conv backbone + ECTB (no LEM, no APH) — analogous to Yolov5s+ECTB."""
    def __init__(self, num_classes=30, base_ch=32):
        super().__init__()
        b = base_ch
        self.stem  = nn.Sequential(
            ConvBNSiLU(3,    b,    3, stride=2),
            ConvBNSiLU(b,    b*2,  3, stride=2),
            ConvBNSiLU(b*2,  b*4,  3, stride=2),
            ConvBNSiLU(b*4,  b*8,  3, stride=2),
            ConvBNSiLU(b*8,  b*16, 3, stride=2),
        )
        ch = b * 16
        safe_h = 4
        while ch % safe_h != 0:
            safe_h //= 2
        self.ectb = ECTB(ch, ch, num_heads=safe_h)
        self.sppf = SPPF(ch, ch)
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(ch, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.ectb(x)
        x = self.sppf(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)


class LEMECTBClassifier(nn.Module):
    """LEM backbone + ECTB, no APH."""
    def __init__(self, num_classes=30, base_ch=32):
        super().__init__()
        b = base_ch
        self.conv1 = ConvBNSiLU(3,    b,    3, stride=2)
        self.conv2 = ConvBNSiLU(b,    b*2,  3, stride=2)
        self.lem1  = LEM(b*2,  b*2)
        self.conv3 = ConvBNSiLU(b*2,  b*4,  3, stride=2)
        self.lem2  = LEM(b*4,  b*4)
        self.conv4 = ConvBNSiLU(b*4,  b*8,  3, stride=2)
        self.lem3  = LEM(b*8,  b*8)
        self.conv5 = ConvBNSiLU(b*8,  b*16, 3, stride=2)
        ch = b * 16
        safe_h = 4
        while ch % safe_h != 0:
            safe_h //= 2
        self.ectb  = ECTB(ch, ch, num_heads=safe_h)
        self.sppf  = SPPF(ch, ch)
        self.gap   = nn.AdaptiveAvgPool2d(1)
        self.fc    = nn.Linear(ch, num_classes)

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x = self.conv3(self.lem1(x))
        x = self.conv4(self.lem2(x))
        x = self.conv5(self.lem3(x))
        x = self.ectb(x)
        x = self.sppf(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_eval(model, train_loader, val_loader, device, epochs=50, lr=0.01):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.937,
                          weight_decay=0.0005, nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler    = GradScaler() if device.type == "cuda" else None

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast(enabled=(scaler is not None)):
                loss = criterion(model(imgs), labels)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                pred = model(imgs).argmax(1)
                correct += (pred == labels).sum().item()
                total   += labels.size(0)
        acc = 100.0 * correct / total
        if acc > best_acc:
            best_acc = acc
        if epoch % 10 == 0:
            print(f"    Epoch {epoch}/{epochs}  val_acc={acc:.2f}%  best={best_acc:.2f}%")

    return best_acc, count_params(model)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default="data/aid")
    parser.add_argument("--num_classes",type=int, default=None)
    parser.add_argument("--train_ratio",type=float, default=0.8)
    parser.add_argument("--img_size",   type=int, default=224)
    parser.add_argument("--epochs",     type=int, default=50,
                        help="Epochs per variant (use 50 for quick ablation, 300 for full)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--base_ch",    type=int, default=32)
    parser.add_argument("--num_workers",type=int, default=4)
    args = parser.parse_args()

    if not os.path.isdir(args.data):
        print(f"[ERROR] Dataset not found: {args.data}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, class_names = get_dataloaders(
        root=args.data,
        train_ratio=args.train_ratio,
        image_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    nc = args.num_classes or len(class_names)

    variants = [
        ("Baseline (LEM only)",      BaselineClassifier(nc, args.base_ch)),
        ("+ ECTB (no LEM)",          ECTBOnlyClassifier(nc, args.base_ch)),
        ("LEM + ECTB",               LEMECTBClassifier(nc,  args.base_ch)),
        ("Full RTD-Net (LEM+ECTB+APH)", RTDNetClassifier(nc, args.base_ch)),
    ]

    results = []
    for name, model in variants:
        print(f"\n{'='*50}")
        print(f"Variant: {name}")
        t0 = time.time()
        best_acc, params = train_eval(
            model, train_loader, val_loader, device,
            epochs=args.epochs, lr=0.01)
        elapsed = time.time() - t0
        results.append({
            "variant":    name,
            "best_val_acc": round(best_acc, 2),
            "params_M":   round(params / 1e6, 2),
            "time_min":   round(elapsed / 60, 1),
        })
        print(f"  → Best Val Acc: {best_acc:.2f}%  |  Params: {params/1e6:.2f}M  |  Time: {elapsed/60:.1f}min")

    # Print summary table
    print("\n" + "=" * 65)
    print(f"{'Variant':<35} {'Val Acc':>8} {'Params(M)':>10} {'Time(min)':>10}")
    print("-" * 65)
    for r in results:
        print(f"{r['variant']:<35} {r['best_val_acc']:>7.2f}% {r['params_M']:>10.2f} {r['time_min']:>10.1f}")
    print("=" * 65)

    # Save results
    Path("runs/ablation").mkdir(parents=True, exist_ok=True)
    with open("runs/ablation/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to runs/ablation/results.json")


if __name__ == "__main__":
    main()