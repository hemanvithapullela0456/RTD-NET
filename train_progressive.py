"""
train_progressive.py  —  Progressive ablation: augmentation + 5 architecture runs
===================================================================================
Runs configurations in order:

    baseline  — rtdnet_slim.py
    exp1      — rtdnet_liteaspp.py       (LiteASPP)
    exp2      — rtdnet_replem.py         (LiteASPP + RepLEM)
    exp3      — rtdnet_v2.py             (LiteASPP + RepLEM + ECTBPlus
                                          + LateralFusion + TwoScalePool)

Usage — run only V2:
    python train_progressive.py --data data/aid --dataset aid --configs exp3

Usage — run full progressive sweep:
    python train_progressive.py --data data/aid --dataset aid

Usage — 50/50 split (matches your reported result):
    python train_progressive.py --data data/aid --dataset aid \
        --train_ratio 0.5 --configs exp3

Resume / compare against existing exp2:
    python train_progressive.py --data data/aid --configs exp2 exp3
"""

import os, sys, time, json, logging, argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.amp import GradScaler, autocast
from torchvision import datasets

# Local imports
from augmentations import (MixupCutMixCollator, LabelSmoothingLoss,
                            WarmupCosineScheduler, get_strong_transforms)
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


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
    elif config == 'exp3':
        from rtdnet_v2 import RTDNetV2
        return RTDNetV2(num_classes=num_classes, base_ch=base_ch,
                        num_heads=num_heads, C=C, dropout=dropout)
    elif config == 'exp4':
        from rtdnet_v3 import RTDNetV3
        return RTDNetV3(num_classes=num_classes, base_ch=base_ch,
                        num_heads=num_heads, C=C, dropout=dropout)
    elif config == 'exp5':
        from rtdnet_v4 import RTDNetV4
        return RTDNetV4(num_classes=num_classes, base_ch=base_ch,
                        num_heads=num_heads, C=C, dropout=dropout)
    elif config == 'exp6':
        from rtdnet_v5 import RTDNetV5
        return RTDNetV5(num_classes=num_classes, base_ch=base_ch,
                        num_heads=num_heads, C=C, dropout=dropout)
    else:
        raise ValueError(f'Unknown config: {config}')


CONFIG_DESC = {
    'baseline': 'Slim baseline + new training recipe only',
    'exp1':     'Slim + LiteASPP                        (arch +1)',
    'exp2':     'Slim + LiteASPP + RepLEM               (arch +2)',
    'exp3':     'Slim + LiteASPP + RepLEM + ECTBPlus + LateralFusion + TwoScalePool  (arch +3)',
    'exp4':     'Slim + LiteASPP + RepLEM + ECTBPlus + LateralFusion + TwoScalePool + V3  (arch +4)',
    'exp5': 'Slim + LiteASPP + RepLEM + ECTBPlus + MultiFPN + APH + TwoScalePool (arch +5)',
    'exp6': 'Slim + LiteASPP + RepLEM + ECTBPlus + MultiFPN + APH + TwoScalePool (arch +6)',
}

ALL_CONFIGS = ['baseline', 'exp1', 'exp2', 'exp3', 'exp4', 'exp5', 'exp6']


# ─── Data loading ─────────────────────────────────────────────────────────────

def get_dataloaders_strong(root, train_ratio, image_size, batch_size,
                           num_workers, seed, num_classes):
    import random
    full_ds = datasets.ImageFolder(root)
    nc      = len(full_ds.classes)

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

    return train_loader, val_loader, full_ds.classes


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
        self.val = val; self.sum += val * n; self.count += n
        self.avg  = self.sum / self.count


def topk_acc(output, target, topk=(1,)):
    if target.dim() == 2:
        target = target.argmax(dim=1)
    maxk = max(topk); bsz = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    correct = pred.t().eq(target.view(1, -1).expand_as(pred.t()))
    return [correct[:k].reshape(-1).float().sum().mul_(100. / bsz).item()
            for k in topk]


def train_epoch(model, loader, criterion, optimizer, scaler,
                device, logger, ep, log_every=20):
    model.train()
    lm, am = AverageMeter(), AverageMeter()
    for step, (imgs, labels) in enumerate(loader):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=scaler is not None):
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
        if labels.dim() == 2:
            labels = labels.argmax(dim=1)
        loss   = F.cross_entropy(out, labels)
        k      = min(5, out.size(1))
        a1, a5 = topk_acc(out, labels, topk=(1, k))
        n = imgs.size(0)
        lm.update(loss.item(), n)
        am1.update(a1, n)
        am5.update(a5, n)
    return lm.avg, am1.avg, am5.avg


# ─── Single experiment ────────────────────────────────────────────────────────

def run_experiment(config, args, train_loader, val_loader,
                   num_classes, save_dir, logger, device):
    logger.info(f'\n{"="*72}')
    logger.info(f'  Config : {config}')
    logger.info(f'  Desc   : {CONFIG_DESC[config]}')
    logger.info(f'{"="*72}')

    model = build_model(config, num_classes, args.base_ch,
                        args.num_heads, args.C, args.dropout).to(device)

    total = sum(p.numel() for p in model.parameters())
    logger.info(f'  Params : {total:,}  ({total/1e6:.3f} M)  '
                f'| {total*4/1024**2:.2f} MB')

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

    scaler = (
        GradScaler(device=device.type)
        if (device.type == 'cuda' and not args.no_amp)
        else None
    )

    best_acc, best_ep, no_imp = 0., 0, 0
    start_epoch = 1

    history = dict(train_loss=[], train_acc=[], val_loss=[], val_acc=[])

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        best_acc = ckpt.get('val_acc', 0.0)
        best_ep = ckpt.get('epoch', 0)
        start_epoch = best_ep + 1
        logger.info(f"Resumed from {args.resume} at epoch {best_ep}")

    for ep in range(start_epoch, args.epochs + 1):
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
            logger.info(f'   [BEST] new best {best_acc:.2f}%  (epoch {best_ep})')
        else:
            no_imp += 1

        with open(save_dir / f'{config}_history.json', 'w') as f:
            json.dump(history, f, indent=2)

        if no_imp >= args.patience:
            logger.info(f'  Early stop at epoch {ep}  '
                        f'(best {best_acc:.2f}% @ ep {best_ep})')
            break

    # ── Reparameterize if applicable ──────────────────────────────────────────
    if hasattr(model, 'reparameterize'):
        model.reparameterize()
        # Save inference-ready checkpoint
        torch.save(dict(epoch=best_ep,
                        model_state=model.state_dict(),
                        val_acc=best_acc,
                        config=config,
                        reparameterized=True),
                   save_dir / f'{config}_best_fused.pt')
        logger.info(f'  Saved fused checkpoint → {config}_best_fused.pt')

    # ── Latency benchmark ─────────────────────────────────────────────────────
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
    p = argparse.ArgumentParser(
        description='RTD-Net progressive architecture ablation trainer')
    p.add_argument('--data',            default='data/aid',
                   help='Path to dataset root (ImageFolder structure)')
    p.add_argument('--dataset',         default='aid',
                   choices=['aid', 'nwpu'])
    p.add_argument('--num_classes',     type=int,   default=None,
                   help='Override class count (auto-detected if omitted)')
    p.add_argument('--train_ratio',     type=float, default=0.5,
                   help='Train split fraction. 0.5 = 50/50, 0.8 = 80/20')
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
    p.add_argument('--configs',         nargs='+',  default=['exp3'],
                   choices=ALL_CONFIGS,
                   help='Which experiments to run. Default: exp3 only')
    p.add_argument('--num_workers',     type=int,   default=4)
    p.add_argument('--seed',            type=int,   default=42)
    p.add_argument('--save_dir',        default='runs/progressive')
    p.add_argument('--no_amp',          action='store_true',
                   help='Disable automatic mixed precision')
    p.add_argument('--resume', type=str, default='',
               help='Path to checkpoint to resume from')
    args = p.parse_args()

    save_dir = (Path(args.save_dir) /
                f"{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(save_dir)

    logger.info('=' * 72)
    logger.info('  RTD-Net Progressive Ablation Trainer')
    logger.info(f'  Configs      : {args.configs}')
    logger.info(f'  Train ratio  : {args.train_ratio}')
    logger.info(f'  Label smooth : {args.label_smoothing}')
    logger.info(f'  Epochs       : {args.epochs}  patience={args.patience}')
    logger.info(f'  Img size     : {args.img_size}')
    logger.info('=' * 72)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'  Device : {device}')

    if not os.path.isdir(args.data):
        logger.error(f'Dataset directory not found: {args.data}')
        sys.exit(1)

    tmp_ds      = datasets.ImageFolder(args.data)
    num_classes = args.num_classes or len(tmp_ds.classes)
    del tmp_ds
    logger.info(f'  Classes : {num_classes}')

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

    # ── Summary table ─────────────────────────────────────────────────────────
    logger.info('\n' + '=' * 72)
    logger.info('  RESULTS')
    logger.info('=' * 72)
    logger.info(f"  {'Config':<12} {'Acc%':>7} {'ΔAcc':>7} "
                f"{'Params(M)':>10} {'MB':>6} {'InfMS':>7} {'FPS':>6}")
    logger.info('-' * 72)

    baseline_acc = next(
        (r['best_val_acc'] for r in results if r['config'] == 'baseline'),
        None)

    for r in results:
        delta  = ((r['best_val_acc'] - baseline_acc)
                  if baseline_acc is not None else 0.0)
        marker = ('   [TOP]' if r['best_val_acc'] ==
                  max(x['best_val_acc'] for x in results) else '')
        logger.info(
            f"  {r['config']:<12} {r['best_val_acc']:>7.2f} "
            f"{delta:>+7.2f} {r['params_M']:>10.4f} "
            f"{r['size_MB']:>6.2f} {r['inf_ms']:>7.2f} "
            f"{r['fps']:>6.1f}{marker}")

    logger.info('=' * 72)

    summary = dict(results=results, settings=vars(args))
    with open(save_dir / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f'  Saved → {save_dir / "results.json"}')


if __name__ == '__main__':
    main()