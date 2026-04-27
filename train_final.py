"""
train_final.py  —  Clean training script for RTDNetFinal
=========================================================
Philosophy: simple, correct, no EMA complexity during training.

What this script does:
    - Trains RTDNetFinal with SGD + cosine LR + warmup
    - Validates on live model weights every epoch  (no EMA switching)
    - Saves best checkpoint by val accuracy
    - At the very end: applies 4-view TTA to the best checkpoint
    - That's it. No EMA during training. No cold-switch surprises.

Why no EMA during per-epoch validation:
    EMA is only useful when the model is near convergence (epoch 200+).
    Before that, the shadow lags the live model and gives WORSE val accuracy,
    which corrupts your best-checkpoint selection. The correct use of EMA
    is: train normally, save best live checkpoint, then at final test
    load that checkpoint and average it with a second run's checkpoint
    (proper model soup / checkpoint averaging). TTA gives the same benefit
    with zero complexity.

Augmentation recipe:
    Train: RandomResizedCrop + HFlip + VFlip + Rotation90 +
           ColorJitter + RandAugment(2,9) + RandomErasing
    Val  : Resize + CenterCrop (clean)

Loss: LabelSmoothing(0.05) + Mixup(0.4) + CutMix(1.0)
    Note: 0.05 not 0.15. At 0.15 + Mixup the model is double-regularised
    and confidence is suppressed — this directly hurts early learning.

Optimizer: SGD(lr=0.01, momentum=0.937, nesterov=True, wd=5e-4)
Scheduler: Linear warmup 5ep → cosine decay to 1e-6
Grad clip : 1.0 (not 10.0 — transformer blocks need tighter clipping)

Usage:
    # AID 50/50
    python train_final.py --data data/aid --dataset aid --split 50

    # NWPU 20%
    python train_final.py --data data/nwpu --dataset nwpu \
        --num_classes 45 --split 20

    # UCM 80%
    python train_final.py --data data/ucm --dataset ucm \
        --num_classes 21 --split 80
"""

import os, sys, time, json, math, random, logging, argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
from torchvision import datasets, transforms
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from rtdnet_final import RTDNetFinal


# ─── Augmentations ────────────────────────────────────────────────────────────

def get_transforms(image_size, is_train):
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                image_size, scale=(0.5, 1.0), ratio=(0.75, 1.333),
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=90),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.05),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3680, 0.3810, 0.3436],
                                  std=[0.2034, 0.1854, 0.1844]),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ])
    else:
        resize = int(image_size * 1.143)
        return transforms.Compose([
            transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3680, 0.3810, 0.3436],
                                  std=[0.2034, 0.1854, 0.1844]),
        ])


# ─── Mixup / CutMix collator ──────────────────────────────────────────────────

class MixupCutMixCollator:
    def __init__(self, num_classes, mixup_alpha=0.4, cutmix_alpha=1.0,
                 mixup_prob=0.5, cutmix_prob=0.5):
        self.num_classes  = num_classes
        self.mixup_alpha  = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_prob   = mixup_prob
        self.cutmix_prob  = cutmix_prob

    def __call__(self, batch):
        imgs, labels = zip(*batch)
        imgs   = torch.stack(imgs)
        labels = torch.tensor(labels, dtype=torch.long)
        r = random.random()
        if r < self.mixup_prob:
            return self._mixup(imgs, labels)
        elif r < self.mixup_prob + self.cutmix_prob:
            return self._cutmix(imgs, labels)
        else:
            return imgs, F.one_hot(labels, self.num_classes).float()

    def _mixup(self, imgs, labels):
        lam   = max(np.random.beta(self.mixup_alpha, self.mixup_alpha),
                    1 - np.random.beta(self.mixup_alpha, self.mixup_alpha))
        idx   = torch.randperm(imgs.size(0))
        mixed = lam * imgs + (1 - lam) * imgs[idx]
        y     = F.one_hot(labels, self.num_classes).float()
        return mixed, lam * y + (1 - lam) * y[idx]

    def _cutmix(self, imgs, labels):
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        idx = torch.randperm(imgs.size(0))
        B, C, H, W = imgs.shape
        cut_h = int(H * math.sqrt(1 - lam))
        cut_w = int(W * math.sqrt(1 - lam))
        cx, cy = random.randint(0, W), random.randint(0, H)
        x1 = max(cx - cut_w // 2, 0);  x2 = min(cx + cut_w // 2, W)
        y1 = max(cy - cut_h // 2, 0);  y2 = min(cy + cut_h // 2, H)
        mixed = imgs.clone()
        mixed[:, :, y1:y2, x1:x2] = imgs[idx, :, y1:y2, x1:x2]
        lam_real = 1 - (y2 - y1) * (x2 - x1) / (H * W)
        y = F.one_hot(labels, self.num_classes).float()
        return mixed, lam_real * y + (1 - lam_real) * y[idx]


# ─── Loss ─────────────────────────────────────────────────────────────────────

class LabelSmoothingLoss(nn.Module):
    """smoothing=0.05. NOT 0.15. See train_final.py docstring for why."""
    def __init__(self, num_classes, smoothing=0.05):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing   = smoothing
        self.confidence  = 1.0 - smoothing

    def forward(self, pred, target):
        log_p = F.log_softmax(pred, dim=-1)
        if target.dim() == 1:
            smooth = self.smoothing / (self.num_classes - 1)
            soft   = torch.full_like(log_p, smooth)
            soft.scatter_(1, target.unsqueeze(1), self.confidence)
        else:
            soft = (1 - self.smoothing) * target + self.smoothing / self.num_classes
        return -(soft * log_p).sum(dim=-1).mean()


# ─── Scheduler ────────────────────────────────────────────────────────────────

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs=5, total_epochs=300,
                 min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.min_lr        = min_lr
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        ep = self.last_epoch
        if ep < self.warmup_epochs:
            scale = (ep + 1) / max(self.warmup_epochs, 1)
        else:
            progress = (ep - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1)
            scale = 0.5 * (1 + math.cos(math.pi * progress))
        return [self.min_lr + (base - self.min_lr) * scale
                for base in self.base_lrs]


# ─── Data ─────────────────────────────────────────────────────────────────────

def get_dataloaders(root, train_ratio, image_size, batch_size,
                    num_workers, seed, num_classes):
    full_ds = datasets.ImageFolder(root)
    nc = len(full_ds.classes)
    class_names = full_ds.classes

    class_indices = {c: [] for c in range(nc)}
    for idx, (_, label) in enumerate(full_ds.samples):
        class_indices[label].append(idx)

    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for label in range(nc):
        idxs = class_indices[label][:]
        rng.shuffle(idxs)
        split = int(len(idxs) * train_ratio)
        train_idx.extend(idxs[:split])
        val_idx.extend(idxs[split:])

    train_full = datasets.ImageFolder(root, transform=get_transforms(image_size, True))
    val_full   = datasets.ImageFolder(root, transform=get_transforms(image_size, False))

    collator = MixupCutMixCollator(num_classes=num_classes)

    train_loader = DataLoader(
        Subset(train_full, train_idx), batch_size=batch_size, shuffle=True,
        collate_fn=collator, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        Subset(val_full, val_idx), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, class_names


# ─── Meters ───────────────────────────────────────────────────────────────────

class Meter:
    def __init__(self): self.reset()
    def reset(self): self.sum = self.count = 0.0
    def update(self, val, n=1): self.sum += val * n; self.count += n
    @property
    def avg(self): return self.sum / max(self.count, 1)


def topk_acc(output, target, k=1):
    if target.dim() == 2:
        target = target.argmax(1)
    _, pred = output.topk(k, dim=1)
    return pred.t().eq(target.view(1,-1).expand_as(pred.t()))\
               [:k].reshape(-1).float().mean().item() * 100.0


# ─── Train / Validate ─────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, scaler, device, logger, epoch):
    model.train()
    loss_m, acc_m = Meter(), Meter()
    log_every = max(1, len(loader) // 3)

    for step, (imgs, labels) in enumerate(loader):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=(scaler is not None)):
            out  = model(imgs)
            loss = criterion(out, labels)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        with torch.no_grad():
            a1 = topk_acc(out, labels, k=1)
        loss_m.update(loss.item(), imgs.size(0))
        acc_m.update(a1, imgs.size(0))

        if (step + 1) % log_every == 0:
            logger.info(f'  Ep{epoch:3d} {step+1:4d}/{len(loader)} '
                        f'loss={loss_m.avg:.4f}  acc={acc_m.avg:.2f}%')

    return loss_m.avg, acc_m.avg


@torch.no_grad()
def validate(model, loader, criterion, device, use_tta=False):
    model.eval()
    loss_m, acc1_m, acc5_m = Meter(), Meter(), Meter()
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if labels.dim() == 2:
            labels = labels.argmax(1)

        if use_tta:
            out = (model(imgs) + model(imgs.flip(-1)) +
                   model(imgs.flip(-2)) + model(imgs.flip(-1).flip(-2))) * 0.25
        else:
            out = model(imgs)

        loss = F.cross_entropy(out, labels)
        k    = min(5, out.size(1))
        loss_m.update(loss.item(), imgs.size(0))
        acc1_m.update(topk_acc(out, labels, 1), imgs.size(0))
        acc5_m.update(topk_acc(out, labels, k), imgs.size(0))
        all_preds.extend(out.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1c = f1_score(all_labels, all_preds, average=None,    zero_division=0).tolist()
    return loss_m.avg, acc1_m.avg, acc5_m.avg, f1, f1c, all_preds, all_labels


# ─── Logger ───────────────────────────────────────────────────────────────────

def setup_logger(save_dir):
    logger = logging.getLogger('train_final')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter('%(asctime)s  %(message)s', datefmt='%H:%M:%S')
    for h in [logging.FileHandler(save_dir / 'training.log'),
               logging.StreamHandler(sys.stdout)]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.propagate = False
    return logger


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data',            default='data/aid')
    p.add_argument('--dataset',         default='aid', choices=['aid','nwpu','ucm'])
    p.add_argument('--num_classes',     type=int,   default=None)
    p.add_argument('--split',           type=int,   default=50, choices=[20,50,80])
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
    p.add_argument('--label_smoothing', type=float, default=0.05)
    p.add_argument('--patience',        type=int,   default=60)
    p.add_argument('--num_workers',     type=int,   default=4)
    p.add_argument('--seed',            type=int,   default=42)
    p.add_argument('--save_dir',        default='runs/final')
    p.add_argument('--no_amp',          action='store_true')
    p.add_argument('--no_tta',          action='store_true')
    args = p.parse_args()

    run_name = f"{args.dataset}_split{args.split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = Path(args.save_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(save_dir)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info('=' * 68)
    logger.info('  RTD-Net Final  (RepLEM×3 + ECTB×2 + LiteASPP + GeMHead)')
    logger.info('=' * 68)
    logger.info(f'  Device        : {device}')
    logger.info(f'  Dataset       : {args.dataset}  split={args.split}%')
    logger.info(f'  Img size      : {args.img_size}  batch={args.batch_size}')
    logger.info(f'  LR            : {args.lr}  warmup={args.warmup_epochs}ep')
    logger.info(f'  Label smooth  : {args.label_smoothing}')
    logger.info(f'  Grad clip     : 1.0')
    logger.info(f'  No EMA during training — live weights validated every epoch')

    if not os.path.isdir(args.data):
        logger.error(f'Dataset not found: {args.data}')
        sys.exit(1)

    tmp_ds      = datasets.ImageFolder(args.data)
    num_classes = args.num_classes or len(tmp_ds.classes)
    del tmp_ds

    train_loader, val_loader, class_names = get_dataloaders(
        root=args.data, train_ratio=args.split/100,
        image_size=args.img_size, batch_size=args.batch_size,
        num_workers=args.num_workers, seed=args.seed,
        num_classes=num_classes,
    )
    logger.info(f'  Classes       : {num_classes}')
    logger.info(f'  Train batches : {len(train_loader)}')
    logger.info(f'  Val batches   : {len(val_loader)}')

    model = RTDNetFinal(
        num_classes=num_classes, base_ch=args.base_ch,
        num_heads=args.num_heads, C=args.C, dropout=args.dropout,
    ).to(device)

    total, _ = model.count_parameters()
    logger.info(f'  Parameters    : {total:,}  ({total/1e6:.3f} M)\n')
    logger.info('  Per-module breakdown:')
    for name, cnt in model.per_module_params().items():
        logger.info(f'    {name:<8}  {cnt:>9,}  ({cnt/total*100:5.1f}%)')
    logger.info('')

    criterion = LabelSmoothingLoss(num_classes, smoothing=args.label_smoothing)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay,
                          nesterov=True)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=args.warmup_epochs,
                                      total_epochs=args.epochs, min_lr=args.min_lr)
    scaler    = GradScaler() if (device.type == 'cuda' and not args.no_amp) else None

    logger.info(f'  AMP           : {"on" if scaler else "off"}')
    logger.info(f'  TTA at final  : {"on" if not args.no_tta else "off"}')
    logger.info(f'  Patience      : {args.patience} epochs\n')

    header = (f"{'Ep':>4}  {'LR':>9}  "
              f"{'TrLoss':>7} {'TrAcc%':>7}  "
              f"{'VaLoss':>7} {'VaAcc%':>7} {'VaTop5':>7}  "
              f"{'F1':>6}  {'Time':>5}")
    logger.info(header)
    logger.info('-' * len(header))

    best_acc   = 0.0
    best_epoch = 0
    no_improve = 0
    history    = dict(tr_loss=[], tr_acc=[], va_loss=[], va_acc=[], va_f1=[])

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, logger, epoch)

        va_loss, va_acc, va_top5, f1, f1_per_cls, _, _ = validate(
            model, val_loader, criterion, device, use_tta=False)

        scheduler.step()
        cur_lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        logger.info(f'{epoch:4d}  {cur_lr:9.6f}  '
                    f'{tr_loss:7.4f} {tr_acc:7.2f}  '
                    f'{va_loss:7.4f} {va_acc:7.2f} {va_top5:7.2f}  '
                    f'{f1:6.4f}  {elapsed:5.0f}s')

        history['tr_loss'].append(round(tr_loss, 5))
        history['tr_acc'].append(round(tr_acc, 3))
        history['va_loss'].append(round(va_loss, 5))
        history['va_acc'].append(round(va_acc, 3))
        history['va_f1'].append(round(f1, 5))
        with open(save_dir / 'history.json', 'w') as fh:
            json.dump(history, fh, indent=2)

        if va_acc > best_acc:
            best_acc, best_epoch, no_improve = va_acc, epoch, 0
            torch.save(dict(
                epoch=epoch, model_state=model.state_dict(),
                val_acc=va_acc, f1=f1, f1_per_class=f1_per_cls,
                class_names=class_names, args=vars(args),
            ), save_dir / 'best.pt')
            logger.info(f'  ★ new best  Acc={best_acc:.2f}%  (ep {best_epoch})')
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info(f'\n  Early stop ep {epoch}  '
                            f'best={best_acc:.2f}% @ ep {best_epoch}')
                break

    # ── Final eval on best checkpoint ─────────────────────────────────────────
    logger.info('\n' + '=' * 68)
    logger.info(f'  Final evaluation  (best checkpoint ep {best_epoch}'
                f'{"  + TTA" if not args.no_tta else ""})')
    logger.info('=' * 68)

    ckpt = torch.load(save_dir / 'best.pt', map_location=device)
    model.load_state_dict(ckpt['model_state'])

    fin_loss, fin_acc, fin_top5, fin_f1, fin_f1c, preds, labels = validate(
        model, val_loader, criterion, device, use_tta=(not args.no_tta))

    logger.info(f'  Top-1 : {fin_acc:.4f}%')
    logger.info(f'  Top-5 : {fin_top5:.4f}%')
    logger.info(f'  F1    : {fin_f1:.4f}')

    logger.info('\n  Per-class F1:')
    for cls, f in zip(class_names, fin_f1c):
        flag = '  ← low' if f < 0.90 else ''
        logger.info(f'    {cls:<30} {f:.4f}{flag}')

    report = classification_report(labels, preds, target_names=class_names,
                                   digits=4, zero_division=0)
    logger.info(f'\n{report}')
    with open(save_dir / 'report.txt', 'w') as fh:
        fh.write(report)

    cm = confusion_matrix(labels, preds)
    with open(save_dir / 'confusion_matrix.json', 'w') as fh:
        json.dump({'class_names': class_names, 'matrix': cm.tolist()}, fh, indent=2)

    model.reparameterize()
    torch.save(dict(model_state=model.state_dict(), class_names=class_names,
                    num_classes=num_classes, val_acc=fin_acc, f1=fin_f1,
                    args=vars(args)),
               save_dir / 'final_reparameterized.pt')

    summary = dict(
        dataset=args.dataset, split=f'{args.split}/{100-args.split}',
        num_classes=num_classes, params_M=round(total/1e6, 4),
        best_epoch=best_epoch, final_top1=round(fin_acc, 4),
        final_top5=round(fin_top5, 4), final_f1=round(fin_f1, 4),
        tta=not args.no_tta,
        f1_per_class={c: round(f, 4) for c, f in zip(class_names, fin_f1c)},
    )
    with open(save_dir / 'summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2)

    logger.info('\n' + '=' * 68)
    logger.info('  DONE')
    logger.info(f'  Top-1 : {fin_acc:.4f}%')
    logger.info(f'  F1    : {fin_f1:.4f}')
    logger.info(f'  Saved : {save_dir}')
    logger.info('=' * 68)


if __name__ == '__main__':
    main()