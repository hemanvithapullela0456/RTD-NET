"""
train_progressive.py  —  Progressive ablation: augmentation + 4 architecture runs
===================================================================================
Runs five configurations in order:

    baseline    — rtdnet_slim.py          + NEW training recipe (Mixup/CutMix +
                                            WarmupCosine + label smoothing 0.15)
    exp1        — rtdnet_liteaspp.py      + same recipe
    exp2        — rtdnet_replem.py        + same recipe
    exp3        — rtdnet_msphead.py       + same recipe
    all         — rtdnet_msphead.py       (all 3 arch changes, already included)
                  Note: exp3 == all since msphead already contains liteaspp+replem

Usage:
    python train_progressive.py --data data/aid --dataset aid
    python train_progressive.py --data data/aid --configs baseline exp1 exp2 exp3
"""

import os, sys, time, json, logging, argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
from torchvision import datasets

# Local imports
from augmentations import (MixupCutMixCollator, LabelSmoothingLoss,
                            WarmupCosineScheduler, get_strong_transforms)

# ─── Model registry ───────────────────────────────────────────────────────────

def build_model(config: str, num_classes: int, base_ch: int,
                num_heads: int, C: int, dropout: float):
    if config == 'baseline':
        from rtdnet_slim import RTDNetClassifier
        return RTDNetClassifier(num_classes=num_classes, base_ch=base_ch,
                                num_heads=num_heads, C=C, dropout=dropout)
    elif config == 'exp1':
        from rtdnet_liteaspp import RTDNetLiteASPP
        return RTDNetLiteASPP(num_classes=num_classes, base_ch=base_ch,
                              num_heads=num_heads, C=C, dropout=dropout)
    elif config == 'exp2':
        from rtdnet_replem import RTDNetRepLEM
        return RTDNetRepLEM(num_classes=num_classes, base_ch=base_ch,
                            num_heads=num_heads, C=C, dropout=dropout)
    elif config in ('exp3', 'all'):
        from rtdnet_msphead import RTDNetMSPHead
        return RTDNetMSPHead(num_classes=num_classes, base_ch=base_ch,
                             num_heads=num_heads, C=C, dropout=dropout)
    else:
        raise ValueError(f'Unknown config: {config}')


CONFIG_DESC = {
    'baseline': 'Slim baseline  + new training recipe only',
    'exp1':     'Slim + LiteASPP            (arch change 1)',
    'exp2':     'Slim + LiteASPP + RepLEM   (arch change 1+2)',
    'exp3':     'Slim + LiteASPP + RepLEM + MSPHead (all 3)',
    'all':      'Slim + LiteASPP + RepLEM + MSPHead (all 3)',
}

ALL_CONFIGS = ['baseline', 'exp1', 'exp2', 'exp3']


# ─── Data loading (uses strong transforms) ────────────────────────────────────

def get_dataloaders_strong(root, train_ratio, image_size, batch_size,
                           num_workers, seed, num_classes):
    """
    Like dataset.get_dataloaders but uses strong augmentation transforms
    and the MixupCutMix collator on the train loader.
    """
    import random
    full_ds = datasets.ImageFolder(root)
    class_names = full_ds.classes
    nc = len(class_names)

    # Stratified split
    class_indices = {c: [] for c in range(nc)}
    for idx, (_, label) in enumerate(full_ds.samples):
        class_indices[label].append(idx)

    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for label, indices in class_indices.items():
        sh = indices[:]
        rng.shuffle(sh)
        sp = int(len(sh) * train_ratio)
        train_idx.extend(sh[:sp])
        val_idx.extend(sh[sp:])

    train_ds_full = datasets.ImageFolder(
        root, transform=get_strong_transforms(image_size, True))
    val_ds_full   = datasets.ImageFolder(
        root, transform=get_strong_transforms(image_size, False))

    train_ds = Subset(train_ds_full, train_idx)
    val_ds   = Subset(val_ds_full,   val_idx)

    collator = MixupCutMixCollator(
        mixup_alpha=0.4, cutmix_alpha=1.0,
        cutmix_prob=0.5, mixup_prob=0.5,
        num_classes=num_classes)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collator, num_workers=num_workers,
        pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, class_names


# ─── Training helpers ─────────────────────────────────────────────────────────

def setup_logger(save_dir: Path) -> logging.Logger:
    logger = logging.getLogger('train_progressive')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter('%(asctime)s  %(message)s')
    for h in [logging.FileHandler(save_dir / 'training.log'),
              logging.StreamHandler(sys.stdout)]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.propagate = False
    return logger


class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val*n; self.count += n
        self.avg  = self.sum / self.count


def topk_acc(output, target, topk=(1,)):
    """Works for both hard (1-D) and soft (2-D) targets."""
    if target.dim() == 2:
        target = target.argmax(dim=1)
    maxk = max(topk); bsz = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    correct = pred.t().eq(target.view(1, -1).expand_as(pred.t()))
    return [correct[:k].reshape(-1).float().sum().mul_(100. / bsz).item()
            for k in topk]


def train_epoch(model, loader, criterion, optimizer, scaler, device,
                logger, ep, log_every=20):
    model.train()
    lm, am = AverageMeter(), AverageMeter()
    for step, (imgs, labels) in enumerate(loader):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)   # may be soft (2-D)

        with autocast(enabled=scaler is not None):
            logits = model(imgs)
            loss   = criterion(logits, labels)

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

        with torch.no_grad():
            acc1, = topk_acc(logits, labels, topk=(1,))
        lm.update(loss.item(), imgs.size(0))
        am.update(acc1,        imgs.size(0))

        if (step + 1) % log_every == 0:
            logger.info(f'  Ep{ep:3d} {step+1:4d}/{len(loader)} '
                        f'loss={lm.avg:.4f} acc={am.avg:.2f}%')
    return lm.avg, am.avg


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    lm, am1, am5 = AverageMeter(), AverageMeter(), AverageMeter()
    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        out    = model(imgs)
        # validation labels are always hard (not from collator)
        if labels.dim() == 2:
            labels = labels.argmax(dim=1)
        # use standard CE for val loss reporting
        loss   = F.cross_entropy(out, labels)
        k      = min(5, out.size(1))
        a1, a5 = topk_acc(out, labels, topk=(1, k))
        n = imgs.size(0)
        lm.update(loss.item(), n)
        am1.update(a1, n)
        am5.update(a5, n)
    return lm.avg, am1.avg, am5.avg


# ─── Single experiment run ────────────────────────────────────────────────────

def run_experiment(config, args, train_loader, val_loader,
                   num_classes, save_dir, logger, device):
    logger.info(f'\n{"="*70}')
    logger.info(f'  Config : {config}  —  {CONFIG_DESC[config]}')
    logger.info(f'{"="*70}')

    model = build_model(config, num_classes, args.base_ch,
                        args.num_heads, args.C, args.dropout).to(device)

    total = sum(p.numel() for p in model.parameters())
    logger.info(f'  Params : {total:,}  ({total/1e6:.3f} M)')

    criterion = LabelSmoothingLoss(num_classes=num_classes,
                                   smoothing=args.label_smoothing)

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay,
                          nesterov=True)

    scheduler = WarmupCosineScheduler(optimizer,
                                      warmup_epochs=args.warmup_epochs,
                                      total_epochs=args.epochs,
                                      min_lr=args.min_lr)

    scaler = (GradScaler()
              if (device.type == 'cuda' and not args.no_amp) else None)

    best_acc, best_ep, no_imp = 0., 0, 0
    history = dict(train_loss=[], train_acc=[], val_loss=[], val_acc=[])

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion,
                                      optimizer, scaler, device, logger, ep)
        va_loss, va_acc, va5 = validate(model, val_loader, criterion, device)
        scheduler.step()
        cur_lr = scheduler.get_last_lr()[0]

        logger.info(
            f'Ep{ep:3d}/{args.epochs}  lr={cur_lr:.6f} | '
            f'tr {tr_loss:.4f}/{tr_acc:.2f}% | '
            f'va {va_loss:.4f}/{va_acc:.2f}% top5={va5:.2f}%  '
            f'[{time.time()-t0:.0f}s]')

        for k, v in zip(['train_loss', 'train_acc', 'val_loss', 'val_acc'],
                        [tr_loss, tr_acc, va_loss, va_acc]):
            history[k].append(round(v, 4))

        if va_acc > best_acc:
            best_acc, best_ep, no_imp = va_acc, ep, 0
            torch.save(dict(epoch=ep, model_state=model.state_dict(),
                            val_acc=va_acc, config=config),
                       save_dir / f'{config}_best.pt')
            logger.info(f'  ★ new best {best_acc:.2f}%  (epoch {best_ep})')
        else:
            no_imp += 1

        with open(save_dir / f'{config}_history.json', 'w') as f:
            json.dump(history, f, indent=2)

        if no_imp >= args.patience:
            logger.info(f'  Early stop at epoch {ep}  '
                        f'(best {best_acc:.2f}% @ ep {best_ep})')
            break

    # Reparameterize RepLEM if present
    if hasattr(model, 'reparameterize'):
        model.reparameterize()

    # Inference latency
    model.eval()
    dummy = torch.randn(1, 3, args.img_size, args.img_size).to(device)
    with torch.no_grad():
        for _ in range(5): model(dummy)
        t0 = time.perf_counter()
        for _ in range(100): model(dummy)
    inf_ms = (time.perf_counter() - t0) / 100 * 1000

    return dict(
        config       = config,
        description  = CONFIG_DESC[config],
        best_val_acc = round(best_acc, 2),
        best_epoch   = best_ep,
        params_M     = round(total / 1e6, 4),
        size_MB      = round(total * 4 / 1024**2, 2),
        inf_ms       = round(inf_ms, 2),
        fps          = round(1000 / inf_ms, 1),
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data',            default='data/aid')
    p.add_argument('--dataset',         default='aid',
                   choices=['aid', 'nwpu'])
    p.add_argument('--num_classes',     type=int,   default=None)
    p.add_argument('--train_ratio',     type=float, default=0.8)
    p.add_argument('--img_size',        type=int,   default=640)
    p.add_argument('--base_ch',         type=int,   default=32)
    p.add_argument('--num_heads',       type=int,   default=4)
    p.add_argument('--C',               type=int,   default=16)
    p.add_argument('--dropout',         type=float, default=0.3)
    p.add_argument('--epochs',          type=int,   default=300)
    p.add_argument('--batch_size',      type=int,   default=64)
    p.add_argument('--lr',              type=float, default=0.01)
    p.add_argument('--min_lr',          type=float, default=1e-6)
    p.add_argument('--warmup_epochs',   type=int,   default=5)
    p.add_argument('--momentum',        type=float, default=0.937)
    p.add_argument('--weight_decay',    type=float, default=0.0005)
    p.add_argument('--label_smoothing', type=float, default=0.15)
    p.add_argument('--patience',        type=int,   default=60)
    p.add_argument('--configs',         nargs='+',  default=ALL_CONFIGS,
                   choices=ALL_CONFIGS + ['all'],
                   help='Which experiments to run (default: all 4)')
    p.add_argument('--num_workers',     type=int,   default=4)
    p.add_argument('--seed',            type=int,   default=42)
    p.add_argument('--save_dir',        default='runs/progressive')
    p.add_argument('--no_amp',          action='store_true')
    args = p.parse_args()

    save_dir = (Path(args.save_dir) /
                f"{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(save_dir)

    logger.info('=' * 70)
    logger.info('  Progressive Experiment: augmentation → arch changes')
    logger.info('  Training recipe: Mixup+CutMix, LabelSmooth=0.15, '
                'WarmupCosine, RandomErasing')
    logger.info('=' * 70)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'  Device : {device}')

    if not os.path.isdir(args.data):
        logger.error(f'Dataset not found: {args.data}')
        sys.exit(1)

    # Quick class count
    tmp_ds      = datasets.ImageFolder(args.data)
    num_classes = args.num_classes or len(tmp_ds.classes)
    del tmp_ds

    logger.info(f'  Classes     : {num_classes}')
    logger.info(f'  Label smooth: {args.label_smoothing}')
    logger.info(f'  Warmup eps  : {args.warmup_epochs}')
    logger.info(f'  Max epochs  : {args.epochs}')
    logger.info(f'  Patience    : {args.patience}')

    # Build dataloaders once — reused across all configs
    train_loader, val_loader, class_names = get_dataloaders_strong(
        root=args.data, train_ratio=args.train_ratio,
        image_size=args.img_size, batch_size=args.batch_size,
        num_workers=args.num_workers, seed=args.seed,
        num_classes=num_classes)

    logger.info(f'  Train batches : {len(train_loader)}')
    logger.info(f'  Val   batches : {len(val_loader)}')

    results = []
    for config in args.configs:
        r = run_experiment(config, args, train_loader, val_loader,
                           num_classes, save_dir, logger, device)
        results.append(r)

    # ── Final comparison table ─────────────────────────────────────────────────
    logger.info('\n' + '=' * 70)
    logger.info('  PROGRESSIVE EXPERIMENT RESULTS')
    logger.info('=' * 70)
    logger.info(f"  {'Config':<12} {'Acc%':>7} {'ΔAcc':>7} "
                f"{'Params(M)':>10} {'MB':>6} {'InfMS':>7} {'FPS':>6}")
    logger.info('-' * 70)

    baseline_acc = next(
        (r['best_val_acc'] for r in results if r['config'] == 'baseline'),
        None)

    for r in results:
        delta  = (r['best_val_acc'] - baseline_acc
                  if baseline_acc is not None else 0.0)
        marker = ('  ★' if r['best_val_acc'] ==
                  max(x['best_val_acc'] for x in results) else '')
        logger.info(
            f"  {r['config']:<12} {r['best_val_acc']:>7.2f} "
            f"{delta:>+7.2f} {r['params_M']:>10.4f} "
            f"{r['size_MB']:>6.2f} {r['inf_ms']:>7.2f} "
            f"{r['fps']:>6.1f}{marker}")

    logger.info('=' * 70)

    with open(save_dir / 'progressive_results.json', 'w') as f:
        json.dump(dict(results=results, settings=vars(args)), f, indent=2)
    logger.info(f'  Saved to: {save_dir}')


if __name__ == '__main__':
    main()