"""
train_nam_coordinate.py  —  Training script for RTDNetNAMCoordinate
====================================================================
Trains the RTD-Net + Hybrid NAM-Coordinate Attention classifier on
AID or NWPU-RESISC45 using the same hyperparameters as the original
paper so results are directly comparable.

Quick start (AID, 80/20 split):
    python train_nam_coordinate.py --data data/aid --dataset aid

Quick start (NWPU, 80/20 split):
    python train_nam_coordinate.py --data data/nwpu --dataset nwpu --num_classes 45

Compare against rtdnet_slim baseline:
    python train_nam_coordinate.py --data data/aid --tag slim_vs_ca --epochs 300

Resume from checkpoint:
    python train_nam_coordinate.py --data data/aid --resume runs/nam_ca/aid_.../best.pt

Ablate only the CA module:
    python train_nam_coordinate.py --data data/aid --ca_r 32   # default
    python train_nam_coordinate.py --data data/aid --ca_r 16   # stronger CA
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.cuda.amp import GradScaler, autocast

# ── local imports ──────────────────────────────────────────────────────────────
# dataset.py must be in the same directory (unchanged from original project)
from dataset import get_dataloaders
from dropped.rtdnet_nam_coordinate import RTDNetNAMCoordinate


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(save_dir: Path) -> logging.Logger:
    logger = logging.getLogger("train_nam_ca")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    for h in [
        logging.FileHandler(save_dir / "train.log"),
        logging.StreamHandler(sys.stdout),
    ]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.propagate = False
    return logger


class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0.0
    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


def topk_accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """Returns top-k accuracy list for each k in topk."""
    maxk = max(topk); bsz = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    correct  = pred.t().eq(target.view(1, -1).expand_as(pred.t()))
    return [correct[:k].reshape(-1).float().sum().mul_(100.0 / bsz).item()
            for k in topk]


# ─────────────────────────────────────────────────────────────────────────────
# Train / validate one epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                    logger, epoch, log_every=20):
    model.train()
    loss_m, acc_m = AverageMeter(), AverageMeter()

    for step, (imgs, labels) in enumerate(loader):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=(scaler is not None)):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

        with torch.no_grad():
            acc1, = topk_accuracy(logits, labels, topk=(1,))

        loss_m.update(loss.item(), imgs.size(0))
        acc_m .update(acc1,        imgs.size(0))

        if (step + 1) % log_every == 0:
            logger.info(
                f"  Ep{epoch:3d}  [{step+1:4d}/{len(loader)}]"
                f"  loss={loss_m.avg:.4f}  acc={acc_m.avg:.2f}%"
            )

    return loss_m.avg, acc_m.avg


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_m, top1_m, top5_m = AverageMeter(), AverageMeter(), AverageMeter()

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        out    = model(imgs)
        loss   = criterion(out, labels)

        k = min(5, out.size(1))
        a1, a5 = topk_accuracy(out, labels, topk=(1, k))
        n = imgs.size(0)
        loss_m.update(loss.item(), n)
        top1_m.update(a1, n)
        top5_m.update(a5, n)

    return loss_m.avg, top1_m.avg, top5_m.avg


# ─────────────────────────────────────────────────────────────────────────────
# Inference latency
# ─────────────────────────────────────────────────────────────────────────────

def measure_latency(model, device, img_size, warmup=10, reps=200) -> float:
    """Returns average inference time in milliseconds (batch=1)."""
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train RTDNetNAMCoordinate on AID / NWPU-RESISC45"
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    parser.add_argument("--data",        default="data/aid",
                        help="Root directory of the dataset")
    parser.add_argument("--dataset",     default="aid",
                        choices=["aid", "nwpu"],
                        help="Dataset name (used for save-dir label only)")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Override number of classes (auto-detected if None)")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Fraction of data for training (0.8 = 80/20 split)")
    parser.add_argument("--img_size",    type=int, default=640,
                        help="Input resolution (paper uses 640 for detection)")

    # ── Model ─────────────────────────────────────────────────────────────────
    parser.add_argument("--base_ch",    type=int,   default=32)
    parser.add_argument("--num_heads",  type=int,   default=4)
    parser.add_argument("--C",          type=int,   default=16,
                        help="Number of LEM branches")
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--ca_r",       type=int,   default=32,
                        help="Coordinate Attention channel reduction ratio")

    # ── Training ──────────────────────────────────────────────────────────────
    parser.add_argument("--epochs",       type=int,   default=300)
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=0.01)
    parser.add_argument("--momentum",     type=float, default=0.937)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--lr_steps",     nargs="+",  type=int,
                        default=[100, 200],
                        help="Epochs at which to decay LR by 0.1")
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--patience",     type=int,   default=50,
                        help="Early-stop if val acc does not improve for N epochs")
    parser.add_argument("--no_amp",       action="store_true",
                        help="Disable automatic mixed precision")

    # ── Misc ──────────────────────────────────────────────────────────────────
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--save_dir",    default="runs/nam_ca")
    parser.add_argument("--tag",         default="",
                        help="Optional extra tag appended to save directory")
    parser.add_argument("--resume",      default="",
                        help="Path to checkpoint .pt file to resume from")

    args = parser.parse_args()

    # ── Save directory ────────────────────────────────────────────────────────
    stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_str  = f"_{args.tag}" if args.tag else ""
    save_dir = Path(args.save_dir) / f"{args.dataset}{tag_str}_{stamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(save_dir)
    logger.info("=" * 70)
    logger.info("  RTDNetNAMCoordinate — Training")
    logger.info("=" * 70)

    # ── Reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Device      : {device}")
    logger.info(f"  Save dir    : {save_dir}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    if not os.path.isdir(args.data):
        logger.error(f"Dataset not found: {args.data}")
        sys.exit(1)

    train_loader, val_loader, class_names = get_dataloaders(
        root        = args.data,
        train_ratio = args.train_ratio,
        image_size  = args.img_size,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        seed        = args.seed,
    )
    num_classes = args.num_classes or len(class_names)
    logger.info(f"  Classes     : {num_classes}")
    logger.info(f"  Train / Val : {len(train_loader.dataset)} / "
                f"{len(val_loader.dataset)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = RTDNetNAMCoordinate(
        num_classes = num_classes,
        base_ch     = args.base_ch,
        num_heads   = args.num_heads,
        C           = args.C,
        dropout     = args.dropout,
        ca_r        = args.ca_r,
    ).to(device)

    total, trainable = model.count_parameters()
    logger.info(f"  Params      : {total:,}  ({total/1e6:.4f} M)")
    logger.info(f"  Model size  : {total*4/1024**2:.2f} MB")
    logger.info(f"  CA ratio r  : {args.ca_r}")

    # ── Optionally resume ─────────────────────────────────────────────────────
    start_epoch = 1
    best_acc    = 0.0
    history     = dict(train_loss=[], train_acc=[], val_loss=[], val_acc=[])

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc    = ckpt.get("val_acc", 0.0)
        logger.info(f"  Resumed from {args.resume}  (epoch {start_epoch-1}, "
                    f"best {best_acc:.2f}%)")

    # ── Optimiser / scheduler / loss ──────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.SGD(
        model.parameters(),
        lr           = args.lr,
        momentum     = args.momentum,
        weight_decay = args.weight_decay,
        nesterov     = True,
    )
    scheduler = MultiStepLR(optimizer, milestones=args.lr_steps, gamma=0.1)

    # Fast-forward scheduler to the resumed epoch
    for _ in range(start_epoch - 1):
        scheduler.step()

    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler  = GradScaler() if use_amp else None
    logger.info(f"  AMP         : {use_amp}")
    logger.info("=" * 70)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_epoch = start_epoch
    no_improve = 0

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer,
            scaler, device, logger, epoch,
        )
        va_loss, va_acc, va5 = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            f"Ep{epoch:3d}/{args.epochs}"
            f"  lr={scheduler.get_last_lr()[0]:.5f}"
            f"  | tr  {tr_loss:.4f} / {tr_acc:.2f}%"
            f"  | val {va_loss:.4f} / {va_acc:.2f}%"
            f"  top5={va5:.2f}%"
            f"  | {elapsed:.0f}s"
        )

        history["train_loss"].append(round(tr_loss, 4))
        history["train_acc"] .append(round(tr_acc,  4))
        history["val_loss"]  .append(round(va_loss, 4))
        history["val_acc"]   .append(round(va_acc,  4))

        if va_acc > best_acc:
            best_acc, best_epoch, no_improve = va_acc, epoch, 0
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "val_acc"    : va_acc,
                "args"       : vars(args),
            }, save_dir / "best.pt")
            logger.info(f"  ★ new best {best_acc:.2f}%  (epoch {best_epoch})")
        else:
            no_improve += 1

        # Save latest checkpoint (for safe resume)
        torch.save({
            "epoch"      : epoch,
            "model_state": model.state_dict(),
            "val_acc"    : va_acc,
            "args"       : vars(args),
        }, save_dir / "last.pt")

        # History JSON
        with open(save_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        if no_improve >= args.patience:
            logger.info(
                f"  Early stopping at epoch {epoch}."
                f"  Best: {best_acc:.2f}% @ epoch {best_epoch}"
            )
            break

    # ── Final report ──────────────────────────────────────────────────────────
    inf_ms = measure_latency(model, device, args.img_size)
    fps    = 1000.0 / inf_ms

    logger.info("=" * 70)
    logger.info("  TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Best val accuracy : {best_acc:.2f}%  (epoch {best_epoch})")
    logger.info(f"  Params            : {total/1e6:.4f} M")
    logger.info(f"  Inference latency : {inf_ms:.2f} ms  (~{fps:.0f} FPS)")
    logger.info(f"  Checkpoints saved : {save_dir}")

    summary = {
        "best_val_acc"  : round(best_acc, 2),
        "best_epoch"    : best_epoch,
        "params_M"      : round(total / 1e6, 4),
        "size_MB"       : round(total * 4 / 1024**2, 2),
        "inf_ms"        : round(inf_ms, 2),
        "fps"           : round(fps, 1),
        "settings"      : vars(args),
    }
    with open(save_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"  Summary JSON      : {save_dir / 'summary.json'}")


if __name__ == "__main__":
    main()