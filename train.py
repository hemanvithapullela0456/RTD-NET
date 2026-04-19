"""
train.py — Training script for RTD-Net Image Classifier
Usage:
    python train.py --data data/aid --dataset aid [options]
    python train.py --data data/nwpu --dataset nwpu --num_classes 45

Key defaults match the paper:
    - SGD, momentum=0.937, weight_decay=0.0005
    - Initial LR=0.01, reduce ×0.1 every 100 epochs
    - 300 max epochs
    - Batch size 64
    - Input 224×224 (full 640×640 is available via --img_size 640)
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.cuda.amp import GradScaler, autocast

from dropped.cscga_model import RTDNetClassifier as CSCGA_Model
from dropped.dlem_model import RTDNetClassifier as DLEM_Model
from dropped.casa_model import RTDNetClassifier as CASA_Model
from dataset import get_dataloaders


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logger(save_dir, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # clear old handlers
    logger.handlers = []

    fh = logging.FileHandler(Path(save_dir) / "train.log")
    sh = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter("%(asctime)s  %(message)s")
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Return top-k accuracy values."""
    maxk = max(topk)
    bsz  = target.size(0)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum()
        res.append(correct_k.mul_(100.0 / bsz).item())
    return res


# ---------------------------------------------------------------------------
# Train / Val one epoch
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, logger, epoch):
    model.train()
    loss_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    for step, (imgs, labels) in enumerate(loader):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=(device.type == "cuda")):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

        acc1, acc5 = accuracy(logits.detach(), labels, topk=(1, min(5, logits.size(1))))
        n = imgs.size(0)
        loss_m.update(loss.item(), n)
        top1_m.update(acc1, n)
        top5_m.update(acc5, n)

        if (step + 1) % 20 == 0:
            logger.info(f"  Epoch {epoch:3d} | Step {step+1:4d}/{len(loader)} "
                        f"| Loss {loss_m.avg:.4f} | Top1 {top1_m.avg:.2f}%")

    return loss_m.avg, top1_m.avg


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        loss   = criterion(logits, labels)

        acc1, acc5 = accuracy(logits, labels, topk=(1, min(5, logits.size(1))))
        n = imgs.size(0)
        loss_m.update(loss.item(), n)
        top1_m.update(acc1, n)
        top5_m.update(acc5, n)

    return loss_m.avg, top1_m.avg, top5_m.avg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="RTD-Net Classifier Training")

    # Data
    parser.add_argument("--data",        type=str, default="data/aid",
                        help="Path to dataset root (ImageFolder format)")
    parser.add_argument("--dataset",     type=str, default="aid",
                        choices=["aid", "nwpu"],
                        help="Dataset name (used for defaults)")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Number of classes (auto-detected if None)")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Train split ratio (0.8 → 80/20, 0.5 → 50/50)")
    parser.add_argument("--img_size",    type=int, default=224,
                        help="Input image resolution")

    # Model
    parser.add_argument("--base_ch",  type=int, default=32,
                        help="Base channel width (32=~15M params, 48=~33M)")
    parser.add_argument("--num_heads",type=int, default=4)
    parser.add_argument("--C",        type=int, default=16,
                        help="Number of LEM branches")
    parser.add_argument("--dropout",  type=float, default=0.3)

    # Training (paper defaults)
    parser.add_argument("--epochs",     type=int,   default=300)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=0.01)
    parser.add_argument("--momentum",   type=float, default=0.937)
    parser.add_argument("--weight_decay",type=float,default=0.0005)
    parser.add_argument("--lr_steps",   nargs="+",  type=int,
                        default=[100, 200],
                        help="Epochs at which LR is reduced by ×0.1")
    parser.add_argument("--patience",   type=int, default=50,
                        help="Early stopping patience (epochs)")

    # System
    parser.add_argument("--num_workers",type=int, default=4)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--model_save_dir",   type=str, default="runs/train")
    parser.add_argument("--no_amp",     action="store_true",
                        help="Disable automatic mixed precision")

    args = parser.parse_args()
    model_list = [
        ("cscga_model", CSCGA_Model),
        ("dlem_model", DLEM_Model),
        ("casa_model", CASA_Model),
    ]

    # ---- Save directory ----
    run_name = f"{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    base_save_dir = Path(args.model_save_dir) / run_name
    base_save_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(base_save_dir, "main")
    logger.info(f"Args: {vars(args)}")

    # ---- Reproducibility ----
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ---- Dataloaders ----
    if not os.path.isdir(args.data):
        logger.error(f"Dataset not found: {args.data}")
        logger.error("Please download AID from: https://captain-whu.github.io/AID/")
        logger.error("Extract so that: data/aid/<class_name>/*.jpg")
        sys.exit(1)

    train_loader, val_loader, class_names = get_dataloaders(
        root=args.data,
        train_ratio=args.train_ratio,
        image_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    num_classes = args.num_classes or len(class_names)
    logger.info(f"Classes: {num_classes}  |  "
                f"Train batches: {len(train_loader)}  |  "
                f"Val batches: {len(val_loader)}")

    # Save class mapping
    with open(base_save_dir / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)

    # ---- Model ----
    for model_name, ModelClass in model_list:

        logger.info(f"\n\n===== 🚀 Training {model_name} =====\n")

        # 👉 create separate folder for each model
        model_save_dir = base_save_dir / model_name
        model_save_dir.mkdir(parents=True, exist_ok=True)

        # 👉 NEW LOGGER for each model (IMPORTANT)
        logger = setup_logger(model_save_dir, model_name)
        model = ModelClass(
            num_classes=num_classes,
            base_ch=args.base_ch,
            num_heads=args.num_heads,
            C=args.C,
            dropout=args.dropout,
        ).to(device)

        total_params, trainable_params = model.count_parameters()
        logger.info(f"Total params:     {total_params:,}")
        logger.info(f"Trainable params: {trainable_params:,}")
        logger.info(f"Model size (MB):  {total_params * 4 / 1024**2:.2f}")

        # ---- Loss, Optimizer, Scheduler ----
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )

        scheduler = MultiStepLR(optimizer, milestones=args.lr_steps, gamma=0.1)

        # AMP scaler
        use_amp = (device.type == "cuda") and (not args.no_amp)
        scaler  = GradScaler() if use_amp else None
        logger.info(f"Mixed precision: {use_amp}")

        # ---- Training loop ----
        best_acc    = 0.0
        best_epoch  = 0
        no_improve  = 0
        history     = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, args.epochs + 1):
            t_start = time.time()

            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device, logger, epoch)

            val_loss, val_acc, val_acc5 = validate(
                model, val_loader, criterion, device)

            scheduler.step()
            elapsed = time.time() - t_start

            logger.info(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"LR {scheduler.get_last_lr()[0]:.6f} | "
                f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.2f}% | "
                f"Val Loss {val_loss:.4f} | Val Acc {val_acc:.2f}% | "
                f"Val Top5 {val_acc5:.2f}% | {elapsed:.0f}s"
            )

            # Save history
            history["train_loss"].append(round(train_loss, 4))
            history["train_acc"].append(round(train_acc, 2))
            history["val_loss"].append(round(val_loss, 4))
            history["val_acc"].append(round(val_acc, 2))

            # Checkpoint
            is_best = val_acc > best_acc
            if is_best:
                best_acc   = val_acc
                best_epoch = epoch
                no_improve = 0
                torch.save({
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "val_acc":     val_acc,
                    "class_names": class_names,
                    "args":        vars(args),
                }, model_save_dir / "best_model.pt")
                logger.info(f"  ★ New best: {best_acc:.2f}% (epoch {best_epoch})")
            else:
                no_improve += 1

            # Save latest
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_acc":     val_acc,
            }, model_save_dir / "last_model.pt")

            # Save history JSON every epoch
            with open(model_save_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

            # Early stopping
            if no_improve >= args.patience:
                logger.info(f"Early stopping triggered after {args.patience} epochs without improvement.")
                break

    # ---- Final summary ----
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best Val Accuracy : {best_acc:.2f}%  (epoch {best_epoch})")
    logger.info(f"Total params      : {total_params:,}")
    logger.info(f"Model size (MB)   : {total_params * 4 / 1024**2:.2f}")
    logger.info(f"Results saved to  : {model_save_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()