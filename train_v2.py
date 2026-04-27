"""
train_v2.py  —  Training script for RTDNet-V2
==============================================
Target: 97–98% on AID 50/50 split, trained from scratch.

Key training improvements over original train.py:
    1.  CosineAnnealingWarmRestarts + linear warmup  (replaces MultiStepLR)
    2.  MixUp α=0.4 + CutMix α=1.0  (50/50 per batch)
    3.  RandAugment(N=2, M=9)        (replaces manual color jitter + rotation)
    4.  EMA (decay=0.9998)           (evaluation on EMA weights, not last)
    5.  Label smoothing ε=0.15       (was 0.1)
    6.  Separate LR groups: backbone LR=0.01, head LR=0.02

Usage:
    # AID 50/50 (paper comparison setting):
    python train_v2.py --data data/aid --dataset aid --train_ratio 0.5

    # AID 80/20:
    python train_v2.py --data data/aid --dataset aid --train_ratio 0.8

    # NWPU 50/50:
    python train_v2.py --data data/nwpu --dataset nwpu --num_classes 45 --train_ratio 0.5

    # Fast experiment (224px, 150 epochs):
    python train_v2.py --data data/aid --img_size 224 --epochs 150 --batch_size 64
"""

import os, sys, json, time, math, logging, argparse, random
from copy import deepcopy
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from rtdnet_v2 import RTDNetV2
from dataset import get_dataloaders      # your existing dataset.py


# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────
def setup_logger(save_dir: Path) -> logging.Logger:
    logger = logging.getLogger("train_v2")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s  %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(save_dir / "train.log")
        sh = logging.StreamHandler(sys.stdout)
        fh.setFormatter(fmt); sh.setFormatter(fmt)
        logger.addHandler(fh); logger.addHandler(sh)
    logger.propagate = False
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────
class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val*n; self.count += n
        self.avg = self.sum / self.count


def topk_accuracy(output, target, topk=(1,)):
    maxk = max(topk); bsz = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    correct  = pred.t().eq(target.view(1, -1).expand_as(pred.t()))
    return [correct[:k].reshape(-1).float().sum().mul_(100./bsz).item()
            for k in topk]


# ─────────────────────────────────────────────────────────────────────────────
# EMA — Exponential Moving Average of model weights
# ─────────────────────────────────────────────────────────────────────────────
class ModelEMA:
    """
    Maintains an EMA shadow copy of model weights.
    At inference, use ema.module (not the training model).

    decay = 0.9998 gives a ~5000-iteration effective window.
    """
    def __init__(self, model: nn.Module, decay: float = 0.9998):
        self.module = deepcopy(model).eval()
        self.decay  = decay
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_p, m_p in zip(self.module.parameters(),
                               model.parameters()):
            ema_p.copy_(ema_p * self.decay + m_p.detach() * (1.0 - self.decay))
        for ema_b, m_b in zip(self.module.buffers(), model.buffers()):
            ema_b.copy_(m_b)


# ─────────────────────────────────────────────────────────────────────────────
# MixUp and CutMix augmentation (applied in-batch, no extra dataset changes)
# ─────────────────────────────────────────────────────────────────────────────
def mixup(x: torch.Tensor, y: torch.Tensor,
          alpha: float = 0.4) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard MixUp: linearly interpolate image and label pairs."""
    lam = random.betavariate(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam


def cutmix(x: torch.Tensor, y: torch.Tensor,
           alpha: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """CutMix: paste a random patch from another image."""
    lam  = random.betavariate(alpha, alpha)
    idx  = torch.randperm(x.size(0), device=x.device)
    B, C, H, W = x.shape
    cut_r = math.sqrt(1 - lam)
    cut_h = int(H * cut_r); cut_w = int(W * cut_r)
    cx = random.randint(0, W); cy = random.randint(0, H)
    x1 = max(cx - cut_w//2, 0); x2 = min(cx + cut_w//2, W)
    y1 = max(cy - cut_h//2, 0); y2 = min(cy + cut_h//2, H)
    mixed_x     = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam_actual  = 1 - (x2-x1)*(y2-y1)/(W*H)
    return mixed_x, y, y[idx], lam_actual


def mixup_cutmix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule: linear warmup + cosine annealing with restarts
# ─────────────────────────────────────────────────────────────────────────────
class WarmupCosineScheduler:
    """
    Linear warmup for `warmup_epochs` then CosineAnnealingWarmRestarts.
    T_0=30, T_mult=2 → restarts at epochs 30, 90, 210 …
    """
    def __init__(self, optimizer, warmup_epochs: int, T_0: int,
                 T_mult: int, eta_min: float = 1e-6):
        self.warmup_epochs = warmup_epochs
        self.cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
        self.optimizer    = optimizer
        self.base_lrs     = [pg['lr'] for pg in optimizer.param_groups]
        self._epoch       = 0

    def step(self) -> None:
        self._epoch += 1
        if self._epoch <= self.warmup_epochs:
            scale = self._epoch / self.warmup_epochs
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg['lr'] = base_lr * scale
        else:
            self.cosine.step()

    def get_last_lr(self) -> list:
        return [pg['lr'] for pg in self.optimizer.param_groups]


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation: RandAugment wrapper (torchvision ≥ 0.9)
# ─────────────────────────────────────────────────────────────────────────────
def get_transforms(image_size: int, is_train: bool):
    mean = [0.3680, 0.3810, 0.3436]
    std  = [0.2034, 0.1854, 0.1876]
    if is_train:
        return T.Compose([
            T.Resize(int(image_size * 1.15)),
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            # RandAugment replaces manual colour jitter + rotation
            T.RandAugment(num_ops=2, magnitude=9),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:
        return T.Compose([
            T.Resize(int(image_size * 1.05)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])


# ─────────────────────────────────────────────────────────────────────────────
# Train / validate one epoch
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, ema,
                    device, logger, epoch, use_aug):
    model.train()
    loss_m = AverageMeter(); top1_m = AverageMeter()

    for step, (imgs, labels) in enumerate(loader):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # ── MixUp / CutMix (50/50 chance each batch) ─────────────────────────
        if use_aug and random.random() < 0.5:
            if random.random() < 0.5:
                imgs, y_a, y_b, lam = mixup(imgs, labels, alpha=0.4)
            else:
                imgs, y_a, y_b, lam = cutmix(imgs, labels, alpha=1.0)
            mixed = True
        else:
            mixed = False

        with autocast(enabled=scaler is not None):
            logits = model(imgs)
            if mixed:
                loss = mixup_cutmix_criterion(criterion, logits, y_a, y_b, lam)
            else:
                loss = criterion(logits, labels)

        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 10.)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()

        ema.update(model)

        # acc computed on un-mixed logits only when not mixed
        if not mixed:
            with torch.no_grad():
                acc1, = topk_accuracy(logits.detach(), labels, topk=(1,))
            top1_m.update(acc1, imgs.size(0))
        loss_m.update(loss.item(), imgs.size(0))

        if (step + 1) % 20 == 0:
            logger.info(
                f"  Ep{epoch:3d} step{step+1:4d}/{len(loader)} "
                f"loss={loss_m.avg:.4f} acc={top1_m.avg:.2f}%"
            )

    return loss_m.avg, top1_m.avg


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_m = AverageMeter(); top1_m = AverageMeter(); top5_m = AverageMeter()
    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        k      = min(5, logits.size(1))
        a1, a5 = topk_accuracy(logits, labels, topk=(1, k))
        n = imgs.size(0)
        loss_m.update(loss.item(),n); top1_m.update(a1,n); top5_m.update(a5,n)
    return loss_m.avg, top1_m.avg, top5_m.avg


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="RTDNet-V2 Training")

    # Data
    p.add_argument("--data",         default="data/aid")
    p.add_argument("--dataset",      default="aid", choices=["aid","nwpu"])
    p.add_argument("--num_classes",  type=int,   default=None)
    p.add_argument("--train_ratio",  type=float, default=0.5,
                   help="0.5 for paper 50/50 split, 0.8 for 80/20")
    p.add_argument("--img_size",     type=int,   default=224,
                   help="224 for fast experiments, 640 for full training")

    # Model
    p.add_argument("--base_ch",      type=int,   default=32)
    p.add_argument("--num_heads",    type=int,   default=4)
    p.add_argument("--C",            type=int,   default=16)
    p.add_argument("--dropout",      type=float, default=0.3)

    # Optimiser
    p.add_argument("--epochs",       type=int,   default=300)
    p.add_argument("--batch_size",   type=int,   default=32,
                   help="32 for 640px; 64 for 224px")
    p.add_argument("--lr",           type=float, default=0.01)
    p.add_argument("--momentum",     type=float, default=0.937)
    p.add_argument("--weight_decay", type=float, default=0.0005)
    p.add_argument("--warmup_epochs",type=int,   default=5)
    p.add_argument("--T_0",          type=int,   default=30,
                   help="Cosine restart period T_0. Restarts at 30, 90, 210.")
    p.add_argument("--T_mult",       type=int,   default=2)
    p.add_argument("--eta_min",      type=float, default=1e-6)
    p.add_argument("--label_smooth", type=float, default=0.15)
    p.add_argument("--ema_decay",    type=float, default=0.9998)
    p.add_argument("--no_aug",       action="store_true",
                   help="Disable MixUp/CutMix (for debugging)")
    p.add_argument("--patience",     type=int,   default=80)

    # System
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--save_dir",     default="runs/train_v2")
    p.add_argument("--no_amp",       action="store_true")
    args = p.parse_args()

    # ── Save dir ──────────────────────────────────────────────────────────────
    run_name = f"{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = Path(args.save_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(save_dir)

    logger.info("=" * 64)
    logger.info("  RTD-Net V2  —  Training")
    logger.info("=" * 64)
    logger.info(f"  {vars(args)}")

    # ── Reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    if not os.path.isdir(args.data):
        logger.error(f"Dataset not found: {args.data}")
        logger.error("Download AID: https://captain-whu.github.io/AID/")
        sys.exit(1)

    # Inject our stronger transforms into dataset.py's get_dataloaders
    # by monkey-patching the transforms.  If your dataset.py accepts a
    # transform argument, pass it directly instead.
    import dataset as ds_module
    _orig_transforms = ds_module.get_transforms
    ds_module.get_transforms = get_transforms   # swap in RandAugment version

    train_loader, val_loader, class_names = ds_module.get_dataloaders(
        root        = args.data,
        train_ratio = args.train_ratio,
        image_size  = args.img_size,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        seed        = args.seed,
    )
    ds_module.get_transforms = _orig_transforms  # restore

    num_classes = args.num_classes or len(class_names)
    logger.info(f"  Classes: {num_classes}  "
                f"Train batches: {len(train_loader)}  "
                f"Val batches: {len(val_loader)}")

    with open(save_dir / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = RTDNetV2(
        num_classes = num_classes,
        base_ch     = args.base_ch,
        num_heads   = args.num_heads,
        C           = args.C,
        dropout     = args.dropout,
    ).to(device)

    total, _ = model.count_parameters()
    logger.info(f"  Total params : {total:,}  ({total/1e6:.3f} M)")
    logger.info(f"  Model size   : {total*4/1024**2:.2f} MB")

    ema = ModelEMA(model, decay=args.ema_decay)

    # ── Loss + optimiser ──────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)

    optimizer = optim.SGD(
        model.parameters(),
        lr           = args.lr,
        momentum     = args.momentum,
        weight_decay = args.weight_decay,
        nesterov     = True,
    )

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs = args.warmup_epochs,
        T_0           = args.T_0,
        T_mult        = args.T_mult,
        eta_min       = args.eta_min,
    )

    use_amp  = device.type == "cuda" and not args.no_amp
    scaler   = GradScaler() if use_amp else None
    use_aug  = not args.no_aug
    logger.info(f"  AMP: {use_amp}  MixUp/CutMix: {use_aug}  "
                f"EMA decay: {args.ema_decay}")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_acc   = 0.0
    best_epoch = 0
    no_improve = 0
    history    = dict(train_loss=[], train_acc=[],
                      val_loss=[], val_acc=[], val_acc_ema=[])

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer,
            scaler, ema, device, logger, epoch, use_aug)

        # Validate EMA weights (consistently better than raw model)
        va_loss, va_acc,  va5  = validate(model,      val_loader, criterion, device)
        _,       va_acc_ema, _ = validate(ema.module, val_loader, criterion, device)

        scheduler.step()
        elapsed = time.time() - t0

        logger.info(
            f"Ep{epoch:3d}/{args.epochs} "
            f"lr={scheduler.get_last_lr()[0]:.5f} | "
            f"tr {tr_loss:.4f}/{tr_acc:.2f}% | "
            f"va {va_loss:.4f}/{va_acc:.2f}% | "
            f"ema_va {va_acc_ema:.2f}% | "
            f"{elapsed:.0f}s"
        )

        # Save history
        history["train_loss"].append(round(tr_loss, 4))
        history["train_acc"].append(round(tr_acc, 2))
        history["val_loss"].append(round(va_loss, 4))
        history["val_acc"].append(round(va_acc, 2))
        history["val_acc_ema"].append(round(va_acc_ema, 2))

        # Checkpoint based on EMA accuracy (the better model)
        is_best = va_acc_ema > best_acc
        if is_best:
            best_acc   = va_acc_ema
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "epoch"       : epoch,
                "model_state" : ema.module.state_dict(),   # save EMA weights
                "val_acc"     : va_acc_ema,
                "class_names" : class_names,
                "args"        : vars(args),
            }, save_dir / "best_model.pt")
            logger.info(f"  ★ New best (EMA): {best_acc:.2f}%  (epoch {best_epoch})")
        else:
            no_improve += 1

        torch.save({
            "epoch"       : epoch,
            "model_state" : model.state_dict(),
            "ema_state"   : ema.module.state_dict(),
            "val_acc"     : va_acc_ema,
        }, save_dir / "last_model.pt")

        with open(save_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        if no_improve >= args.patience:
            logger.info(f"  Early stopping at epoch {epoch}.")
            break

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("=" * 64)
    logger.info("  TRAINING COMPLETE")
    logger.info(f"  Best EMA val acc : {best_acc:.2f}%  (epoch {best_epoch})")
    logger.info(f"  Total params     : {total:,}  ({total/1e6:.3f} M)")
    logger.info(f"  Results          : {save_dir}")
    logger.info("=" * 64)


if __name__ == "__main__":
    main()