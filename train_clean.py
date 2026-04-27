"""
train_clean.py  —  Fixed training script for RTDNetClean
=========================================================
Fixes over train_v2.py:

    FIX 1 — EMA decay: 0.9999 → 0.999
        At AID 50/50 (~78 steps/epoch), decay=0.9999 gives an EMA
        half-life of ~6931 steps = ~89 epochs. The shadow was near-random
        for the first 50+ epochs, so every validation score was garbage.
        decay=0.999 gives half-life ~693 steps = ~9 epochs — correct.

    FIX 2 — EMA warmup guard
        Don't validate on EMA weights until epoch EMA_WARMUP_EPOCH (30).
        Before that, validate on live model weights. This means your
        epoch 1-30 val accuracy is real and informative.

    FIX 3 — Label smoothing: 0.15 → 0.05
        0.15 + Mixup simultaneously is double-regularisation. Both push
        label confidence down. 0.05 is correct when Mixup is active.

    FIX 4 — scheduler.get_last_lr() called correctly
        Moved after scheduler.step() with correct ordering.

    FIX 5 — Gradient clipping: 10.0 → 1.0
        Aggressive clipping at 10.0 allows large gradient spikes through.
        1.0 is the standard for transformer-containing models.

Everything else (Mixup/CutMix, AMP, WarmupCosine, F1, confusion matrix)
is unchanged from train_v2.py.

Usage:
    python train_clean.py --data data/aid --dataset aid --split 50
    python train_clean.py --data data/aid --dataset aid --split 50 \
        --epochs 300 --batch_size 64 --img_size 640
    python train_clean.py --data data/nwpu --dataset nwpu \
        --num_classes 45 --split 20
    python train_clean.py --data data/ucm  --dataset ucm  \
        --num_classes 21 --split 80
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.amp import GradScaler, autocast
from torchvision import datasets

from sklearn.metrics import f1_score, confusion_matrix, classification_report

from augmentations import (
    MixupCutMixCollator,
    LabelSmoothingLoss,
    WarmupCosineScheduler,
    get_strong_transforms,
)
from rtdnet_v2 import RTDNetV2

# ── EMA ───────────────────────────────────────────────────────────────────────

class EMA:
    """
    Exponential Moving Average of model weights.

    decay=0.999  →  half-life ≈ 693 steps ≈ 9 epochs at 78 steps/epoch
    decay=0.9999 →  half-life ≈ 6931 steps ≈ 89 epochs  ← WRONG for AID

    Rule of thumb: decay = 1 - (1 / (half_life_in_steps))
    For AID 50/50, 78 steps/epoch, target half-life ~10 epochs:
        decay = 1 - 1/(10*78) = 1 - 0.00128 ≈ 0.999
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model   = model
        self.decay   = decay
        self.shadow  = {k: v.clone().float()
                        for k, v in model.state_dict().items()}
        self._backup: dict = {}

    @torch.no_grad()
    def update(self):
        d = self.decay
        for k, v in self.model.state_dict().items():
            self.shadow[k].mul_(d).add_(v.float(), alpha=1.0 - d)

    def apply_shadow(self):
        self._backup = {k: v.clone()
                        for k, v in self.model.state_dict().items()}
        dev = next(self.model.parameters()).device
        self.model.load_state_dict(
            {k: v.to(dev) for k, v in self.shadow.items()})

    def restore(self):
        self.model.load_state_dict(self._backup)
        self._backup = {}

    def state_dict(self):
        return {'shadow': self.shadow, 'decay': self.decay}

    def load_state_dict(self, d):
        self.shadow = d['shadow']
        self.decay  = d['decay']


# ── TTA ───────────────────────────────────────────────────────────────────────

@torch.no_grad()
def tta_predict(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """4-view TTA: original + h-flip + v-flip + both flips."""
    model.eval()
    logits  = model(x)
    logits += model(x.flip(-1))
    logits += model(x.flip(-2))
    logits += model(x.flip(-1).flip(-2))
    return logits * 0.25


# ─── Logger ───────────────────────────────────────────────────────────────────

def setup_logger(save_dir: Path) -> logging.Logger:
    logger = logging.getLogger("train_clean")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    for h in [logging.FileHandler(save_dir / "training.log"),
               logging.StreamHandler(sys.stdout)]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.propagate = False
    return logger


# ─── Data ─────────────────────────────────────────────────────────────────────

def get_dataloaders(root, train_ratio, image_size, batch_size,
                    num_workers, seed, num_classes):
    import random

    full_ds = datasets.ImageFolder(root)
    nc = len(full_ds.classes)
    class_names = full_ds.classes

    class_indices = {c: [] for c in range(nc)}
    for idx, (_, label) in enumerate(full_ds.samples):
        class_indices[label].append(idx)

    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for label in range(nc):
        indices = class_indices[label][:]
        rng.shuffle(indices)
        split = int(len(indices) * train_ratio)
        train_idx.extend(indices[:split])
        val_idx.extend(indices[split:])

    train_ds_full = datasets.ImageFolder(
        root, transform=get_strong_transforms(image_size, is_train=True))
    val_ds_full = datasets.ImageFolder(
        root, transform=get_strong_transforms(image_size, is_train=False))

    train_ds = Subset(train_ds_full, train_idx)
    val_ds   = Subset(val_ds_full,   val_idx)

    collator = MixupCutMixCollator(
        num_classes=num_classes,
        mixup_alpha=0.4, cutmix_alpha=1.0,
        mixup_prob=0.5,  cutmix_prob=0.5,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collator, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, class_names


# ─── Meters ───────────────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0.0
    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


def topk_acc(output, target, topk=(1,)):
    if target.dim() == 2:
        target = target.argmax(dim=1)
    maxk = max(topk)
    bsz  = target.size(0)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    correct = pred.t().eq(target.view(1, -1).expand_as(pred.t()))
    return [correct[:k].reshape(-1).float().sum().mul_(100.0/bsz).item()
            for k in topk]


# ─── Train / validate ─────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, scaler,
                ema, device, logger, epoch, log_every=20):
    model.train()
    loss_m, acc_m = AverageMeter(), AverageMeter()

    for step, (imgs, labels) in enumerate(loader):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast('cuda', enabled=(scaler is not None)):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # FIX 5
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        ema.update()

        with torch.no_grad():
            (acc1,) = topk_acc(logits, labels, topk=(1,))
        loss_m.update(loss.item(), imgs.size(0))
        acc_m.update(acc1, imgs.size(0))

        if (step + 1) % log_every == 0:
            logger.info(f"  Ep{epoch:3d} {step+1:4d}/{len(loader)} "
                        f"loss={loss_m.avg:.4f}  acc={acc_m.avg:.2f}%")

    return loss_m.avg, acc_m.avg


@torch.no_grad()
def validate(model, loader, criterion, device, use_tta=False):
    model.eval()
    loss_m, acc1_m, acc5_m = AverageMeter(), AverageMeter(), AverageMeter()
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if labels.dim() == 2:
            labels = labels.argmax(dim=1)

        out  = tta_predict(model, imgs) if use_tta else model(imgs)
        loss = F.cross_entropy(out, labels)
        k    = min(5, out.size(1))
        a1, a5 = topk_acc(out, labels, topk=(1, k))
        n = imgs.size(0)
        loss_m.update(loss.item(), n)
        acc1_m.update(a1, n)
        acc5_m.update(a5, n)
        all_preds.extend(out.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    f1_macro    = f1_score(all_labels, all_preds, average='macro',
                           zero_division=0)
    f1_per_cls  = f1_score(all_labels, all_preds, average=None,
                           zero_division=0).tolist()

    return (loss_m.avg, acc1_m.avg, acc5_m.avg,
            f1_macro, f1_per_cls, all_preds, all_labels)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data',            default='data/aid')
    p.add_argument('--dataset',         default='aid',
                   choices=['aid', 'nwpu', 'ucm'])
    p.add_argument('--num_classes',     type=int,   default=None)
    p.add_argument('--split',           type=int,   default=50,
                   choices=[20, 50, 80])
    p.add_argument('--base_ch',         type=int,   default=32)
    p.add_argument('--num_heads',       type=int,   default=4)
    p.add_argument('--C',               type=int,   default=16)
    p.add_argument('--dropout',         type=float, default=0.3)
    p.add_argument('--epochs',          type=int,   default=300)
    p.add_argument('--batch_size',      type=int,   default=64)
    p.add_argument('--img_size',        type=int,   default=640)
    p.add_argument('--lr',              type=float, default=0.01)
    p.add_argument('--min_lr',          type=float, default=1e-6)
    p.add_argument('--warmup_epochs',   type=int,   default=5)
    p.add_argument('--momentum',        type=float, default=0.937)
    p.add_argument('--weight_decay',    type=float, default=5e-4)
    # FIX 3: default label_smoothing reduced to 0.05
    p.add_argument('--label_smoothing', type=float, default=0.05)
    # FIX 1: default ema_decay reduced to 0.999
    p.add_argument('--ema_decay',       type=float, default=0.999)
    # FIX 2: EMA warmup guard epoch
    p.add_argument('--ema_warmup_epoch',type=int,   default=30,
                   help='Only use EMA for validation after this epoch')
    p.add_argument('--patience',        type=int,   default=60)
    p.add_argument('--num_workers',     type=int,   default=4)
    p.add_argument('--seed',            type=int,   default=42)
    p.add_argument('--save_dir',        default='runs/clean')
    p.add_argument('--no_amp',          action='store_true')
    p.add_argument('--no_tta',          action='store_true')
    args = p.parse_args()

    # ── Save dir + logger ────────────────────────────────────────────────────
    run_name = (f"{args.dataset}_split{args.split}"
                f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    save_dir = Path(args.save_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(save_dir)

    logger.info('=' * 70)
    logger.info('  RTD-Net Clean  (RepLEM + LiteASPP + GeM)')
    logger.info('  Fixed EMA decay + label smoothing + gradient clipping')
    logger.info('=' * 70)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'  Device        : {device}')

    if not os.path.isdir(args.data):
        logger.error(f'Dataset not found: {args.data}')
        sys.exit(1)

    tmp_ds      = datasets.ImageFolder(args.data)
    num_classes = args.num_classes or len(tmp_ds.classes)
    del tmp_ds

    train_ratio = args.split / 100.0
    logger.info(f'  Dataset       : {args.dataset}')
    logger.info(f'  Classes       : {num_classes}')
    logger.info(f'  Train split   : {args.split}%')
    logger.info(f'  Img size      : {args.img_size}')
    logger.info(f'  Batch size    : {args.batch_size}')
    logger.info(f'  Label smooth  : {args.label_smoothing}  (was 0.15)')
    logger.info(f'  EMA decay     : {args.ema_decay}  (was 0.9999)')
    logger.info(f'  EMA warmup    : epoch {args.ema_warmup_epoch}')
    logger.info(f'  Grad clip     : 1.0  (was 10.0)')

    train_loader, val_loader, class_names = get_dataloaders(
        root=args.data, train_ratio=train_ratio,
        image_size=args.img_size, batch_size=args.batch_size,
        num_workers=args.num_workers, seed=args.seed,
        num_classes=num_classes,
    )
    logger.info(f'  Train batches : {len(train_loader)}')
    logger.info(f'  Val batches   : {len(val_loader)}')

    # ── Model ────────────────────────────────────────────────────────────────
    model = RTDNetV2(
        num_classes=num_classes,
        base_ch=args.base_ch,
        num_heads=args.num_heads,
        C=args.C,
        dropout=args.dropout,
    ).to(device)


    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'  Parameters    : {total_params:,}  ({total_params/1e6:.3f} M)')
    logger.info('\n  Per-module param breakdown:')
    for name, count in model.per_module_params().items():
        logger.info(f'    {name:<12}  {count:>9,}  ({count/total_params*100:5.1f}%)')

    ema = EMA(model, decay=args.ema_decay)

    criterion = LabelSmoothingLoss(
        num_classes=num_classes, smoothing=args.label_smoothing)
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay, nesterov=True,
    )
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs, min_lr=args.min_lr,
    )
    scaler = (GradScaler('cuda') if (device.type == 'cuda' and not args.no_amp)
              else None)

    logger.info(f'  LR            : {args.lr}  (warmup {args.warmup_epochs} ep)')
    logger.info(f'  Patience      : {args.patience}')
    logger.info(f'  AMP           : {"off" if scaler is None else "on"}')
    logger.info(f'  TTA final     : {"off" if args.no_tta else "on"}')
    logger.info('')

    best_f1, best_acc, best_epoch, no_improve = 0.0, 0.0, 0, 0
    history = dict(train_loss=[], train_acc=[],
                   val_loss=[], val_acc=[], val_top5=[], val_f1=[])

    header = (f"{'Ep':>4}  {'LR':>8}  "
              f"{'TrLoss':>7} {'TrAcc%':>7}  "
              f"{'VaLoss':>7} {'VaAcc%':>7} {'VaTop5%':>8}  "
              f"{'F1mac':>6}  {'EMA':>5}  {'Time':>5}")
    logger.info(header)
    logger.info('-' * len(header))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_epoch(
            model, train_loader, criterion, optimizer,
            scaler, ema, device, logger, epoch,
        )

        # FIX 2: use EMA only after warmup guard epoch
        use_ema_this_epoch = epoch >= args.ema_warmup_epoch
        if use_ema_this_epoch:
            ema.apply_shadow()

        (va_loss, va_acc, va_top5,
         f1_macro, f1_per_cls,
         _, _) = validate(model, val_loader, criterion, device, use_tta=False)

        if use_ema_this_epoch:
            ema.restore()

        # FIX 4: step scheduler AFTER validation, get LR correctly
        scheduler.step()
        cur_lr = optimizer.param_groups[0]['lr']

        elapsed = time.time() - t0
        ema_tag = 'EMA' if use_ema_this_epoch else 'live'
        logger.info(
            f"{epoch:4d}  {cur_lr:8.6f}  "
            f"{tr_loss:7.4f} {tr_acc:7.2f}  "
            f"{va_loss:7.4f} {va_acc:7.2f} {va_top5:8.2f}  "
            f"{f1_macro:6.4f}  {ema_tag:>5}  {elapsed:5.0f}s"
        )

        history['train_loss'].append(round(tr_loss, 5))
        history['train_acc'].append(round(tr_acc, 3))
        history['val_loss'].append(round(va_loss, 5))
        history['val_acc'].append(round(va_acc, 3))
        history['val_top5'].append(round(va_top5, 3))
        history['val_f1'].append(round(f1_macro, 5))
        with open(save_dir / 'history.json', 'w') as fh:
            json.dump(history, fh, indent=2)

        if f1_macro > best_f1:
            best_f1, best_acc, best_epoch, no_improve = (
                f1_macro, va_acc, epoch, 0)
            torch.save(dict(
                epoch=epoch,
                model_state=model.state_dict(),
                ema_state=ema.state_dict(),
                val_acc=va_acc, f1_macro=f1_macro,
                f1_per_class=f1_per_cls,
                class_names=class_names, args=vars(args),
            ), save_dir / 'best.pt')
            logger.info(
                f'  * new best  F1={best_f1:.4f}  '
                f'Acc={best_acc:.2f}%  (ep {best_epoch})')
        else:
            no_improve += 1

        if no_improve >= args.patience:
            logger.info(
                f'\n  Early stop ep {epoch}  '
                f'best F1={best_f1:.4f} Acc={best_acc:.2f}% @ ep {best_epoch}')
            break

    # ── Final eval: EMA + TTA ─────────────────────────────────────────────────
    logger.info('\n' + '=' * 70)
    logger.info('  Final evaluation  (EMA weights + TTA)')
    logger.info('=' * 70)

    ckpt = torch.load(save_dir / 'best.pt', map_location=device)
    model.load_state_dict(ckpt['model_state'])
    ema.load_state_dict(ckpt['ema_state'])

    ema.apply_shadow()
    (fin_loss, fin_acc, fin_top5,
     fin_f1, fin_f1_per_cls,
     all_preds, all_labels) = validate(
        model, val_loader, criterion, device,
        use_tta=(not args.no_tta),
    )
    ema.restore()

    logger.info(f'  Top-1 Accuracy : {fin_acc:.4f}%')
    logger.info(f'  Top-5 Accuracy : {fin_top5:.4f}%')
    logger.info(f'  F1 Macro       : {fin_f1:.4f}')
    logger.info(f'  Val Loss       : {fin_loss:.4f}')

    logger.info('\n  Per-class F1 scores:')
    logger.info(f"  {'Class':<30} {'F1':>6}")
    logger.info('  ' + '-' * 38)
    for cls, f1 in zip(class_names, fin_f1_per_cls):
        marker = '  ←' if f1 < 0.90 else ''
        logger.info(f'  {cls:<30} {f1:6.4f}{marker}')

    report = classification_report(all_labels, all_preds,
                                   target_names=class_names, digits=4,
                                   zero_division=0)
    logger.info('\n  sklearn classification_report:\n')
    logger.info(report)
    with open(save_dir / 'classification_report.txt', 'w') as fh:
        fh.write(report)

    cm = confusion_matrix(all_labels, all_preds)
    with open(save_dir / 'confusion_matrix.json', 'w') as fh:
        json.dump({'class_names': class_names, 'matrix': cm.tolist()},
                  fh, indent=2)

    model.reparameterize()
    torch.save(dict(
        model_state=model.state_dict(), class_names=class_names,
        num_classes=num_classes, val_acc=fin_acc, f1_macro=fin_f1,
        args=vars(args),
    ), save_dir / 'final_reparameterized.pt')

    summary = dict(
        dataset=args.dataset, split=f'{args.split}/{100-args.split}',
        num_classes=num_classes, params_M=round(total_params/1e6, 4),
        best_epoch=best_epoch,
        final_top1_acc=round(fin_acc, 4),
        final_top5_acc=round(fin_top5, 4),
        final_f1_macro=round(fin_f1, 4),
        final_val_loss=round(fin_loss, 5),
        tta_used=(not args.no_tta),
        fixes_applied=[
            'ema_decay_0.999', 'label_smoothing_0.05',
            'grad_clip_1.0', 'ema_warmup_guard_ep30', 'gem_head',
        ],
        f1_per_class={cn: round(f, 4)
                      for cn, f in zip(class_names, fin_f1_per_cls)},
        args=vars(args),
    )
    with open(save_dir / 'summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2)

    logger.info('\n' + '=' * 70)
    logger.info('  TRAINING COMPLETE')
    logger.info(f'  Top-1 : {fin_acc:.4f}%')
    logger.info(f'  F1    : {fin_f1:.4f}')
    logger.info(f'  Saved : {save_dir}')
    logger.info('=' * 70)


if __name__ == '__main__':
    main()