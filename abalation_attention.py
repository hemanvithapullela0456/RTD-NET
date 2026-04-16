"""
ablation_attention.py
=====================
Trains all four RTDNetClassifier attention variants sequentially
with identical hyperparameters and reports a clean comparison table.

Usage
-----
    python ablation_attention.py --data data/aid --dataset aid
    python ablation_attention.py --data data/nwpu --dataset nwpu --num_classes 45

Outputs  (saved to runs/ablation/<timestamp>/)
-------
    ablation_results.json   — final comparison dict
    <variant>_history.json  — per-epoch train/val loss+acc
    <variant>_best.pt       — best checkpoint for each variant
    ablation.log            — full console output

Training settings match the paper exactly:
    SGD lr=0.01, momentum=0.937, weight_decay=0.0005, nesterov=True
    MultiStepLR milestones=[100, 200], gamma=0.1
    CrossEntropyLoss label_smoothing=0.1
    Grad clip max_norm=10.0
    AMP (if CUDA available)
    Input 640×640 (set --img_size 224 for faster experimentation)
"""

import os, sys, time, json, logging, argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.cuda.amp import GradScaler, autocast

# All four variants are in one file — just change attention_type
from attention_variants import RTDNetClassifier
from dataset import get_dataloaders   # your existing dataset.py


VARIANTS = ['original', 'conv', 'residual', 'triplet']


# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────
def setup_logger(save_dir: Path) -> logging.Logger:
    logger = logging.getLogger("ablation")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.FileHandler(save_dir / "ablation.log"))
        logger.addHandler(logging.StreamHandler(sys.stdout))
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
        self.avg  = self.sum / self.count


def topk_accuracy(output, target, topk=(1,)):
    maxk = max(topk); bsz = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    correct  = pred.t().eq(target.view(1,-1).expand_as(pred.t()))
    return [correct[:k].reshape(-1).float().sum().mul_(100./bsz).item()
            for k in topk]


def reset_peak_mem():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def peak_mem_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Train / validate
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, scaler, device, logger, ep):
    model.train()
    lm, am = AverageMeter(), AverageMeter()
    for step, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device, non_blocking=True), \
                       labels.to(device, non_blocking=True)
        with autocast(enabled=scaler is not None):
            loss = criterion(model(imgs), labels)
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
            acc1, = topk_accuracy(model(imgs) if not scaler else
                                  model(imgs), labels, topk=(1,))
        lm.update(loss.item(), imgs.size(0))
        am.update(acc1,        imgs.size(0))
        if (step+1) % 20 == 0:
            logger.info(f"  Ep{ep:3d} {step+1:4d}/{len(loader)} "
                        f"loss={lm.avg:.4f} acc={am.avg:.2f}%")
    return lm.avg, am.avg


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    lm, am5 = AverageMeter(), AverageMeter()
    am1 = AverageMeter()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device,non_blocking=True), \
                       labels.to(device,non_blocking=True)
        out  = model(imgs)
        loss = criterion(out, labels)
        k    = min(5, out.size(1))
        a1, a5 = topk_accuracy(out, labels, topk=(1, k))
        n = imgs.size(0)
        lm.update(loss.item(),n); am1.update(a1,n); am5.update(a5,n)
    return lm.avg, am1.avg, am5.avg


# ─────────────────────────────────────────────────────────────────────────────
# Single run
# ─────────────────────────────────────────────────────────────────────────────
def run(variant: str, args, train_loader, val_loader,
        class_names, save_dir: Path, logger, device) -> dict:

    num_classes = args.num_classes or len(class_names)

    model = RTDNetClassifier(
        num_classes    = num_classes,
        base_ch        = args.base_ch,
        num_heads      = args.num_heads,
        C              = args.C,
        dropout        = args.dropout,
        attention_type = variant,
    ).to(device)

    total, _ = model.count_parameters()
    logger.info(f"\n{'='*60}")
    logger.info(f"  Variant       : {variant}")
    logger.info(f"  Total params  : {total:,}  ({total/1e6:.4f} M)")
    logger.info(f"  Size (MB)     : {total*4/1024**2:.2f}")
    logger.info(f"{'='*60}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay, nesterov=True)
    scheduler = MultiStepLR(optimizer, milestones=args.lr_steps, gamma=0.1)
    use_amp   = device.type == "cuda" and not args.no_amp
    scaler    = GradScaler() if use_amp else None

    best_acc, best_ep, no_imp = 0., 0, 0
    history = dict(train_loss=[], train_acc=[], val_loss=[], val_acc=[])
    peak_train_mem = 0.

    for ep in range(1, args.epochs+1):
        reset_peak_mem()
        t0 = time.time()

        tr_loss, tr_acc = train_epoch(model, train_loader, criterion,
                                      optimizer, scaler, device, logger, ep)
        peak_train_mem  = max(peak_train_mem, peak_mem_mb())

        va_loss, va_acc, va5 = validate(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(
            f"Ep{ep:3d}/{args.epochs} lr={scheduler.get_last_lr()[0]:.5f} | "
            f"tr {tr_loss:.4f}/{tr_acc:.2f}% | "
            f"va {va_loss:.4f}/{va_acc:.2f}% top5={va5:.2f}% | "
            f"{time.time()-t0:.0f}s"
        )

        for k,v in zip(['train_loss','train_acc','val_loss','val_acc'],
                       [tr_loss, tr_acc, va_loss, va_acc]):
            history[k].append(round(v, 4))

        if va_acc > best_acc:
            best_acc, best_ep, no_imp = va_acc, ep, 0
            torch.save(dict(epoch=ep, model_state=model.state_dict(),
                            val_acc=va_acc, variant=variant),
                       save_dir / f"{variant}_best.pt")
            logger.info(f"  ★ new best {best_acc:.2f}%  (epoch {best_ep})")
        else:
            no_imp += 1

        with open(save_dir / f"{variant}_history.json", "w") as f:
            json.dump(history, f, indent=2)

        if no_imp >= args.patience:
            logger.info(f"  Early stopping at epoch {ep}.")
            break

    # Inference latency
    model.eval()
    dummy = torch.randn(1, 3, args.img_size, args.img_size).to(device)
    with torch.no_grad():
        for _ in range(5): model(dummy)
        reset_peak_mem()
        t0 = time.perf_counter()
        for _ in range(100): model(dummy)
    inf_ms       = (time.perf_counter()-t0)/100*1000
    peak_inf_mem = peak_mem_mb()

    return dict(
        variant           = variant,
        best_val_acc      = round(best_acc, 2),
        best_epoch        = best_ep,
        total_params      = total,
        total_params_M    = round(total/1e6, 4),
        model_size_MB     = round(total*4/1024**2, 2),
        peak_train_mem_MB = round(peak_train_mem, 1),
        peak_inf_mem_MB   = round(peak_inf_mem, 1),
        inf_ms            = round(inf_ms, 2),
        fps               = round(1000/inf_ms, 1),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",         default="data/aid")
    p.add_argument("--dataset",      default="aid", choices=["aid","nwpu"])
    p.add_argument("--num_classes",  type=int,   default=None)
    p.add_argument("--train_ratio",  type=float, default=0.8)
    p.add_argument("--img_size",     type=int,   default=640)

    p.add_argument("--base_ch",      type=int,   default=32)
    p.add_argument("--num_heads",    type=int,   default=4)
    p.add_argument("--C",            type=int,   default=16)
    p.add_argument("--dropout",      type=float, default=0.3)

    p.add_argument("--epochs",       type=int,   default=300)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=0.01)
    p.add_argument("--momentum",     type=float, default=0.937)
    p.add_argument("--weight_decay", type=float, default=0.0005)
    p.add_argument("--lr_steps",     nargs="+",  type=int, default=[100,200])
    p.add_argument("--patience",     type=int,   default=50)
    p.add_argument("--variants",     nargs="+",
                default=[v for v in VARIANTS if v != 'original'],
                choices=VARIANTS,
                help="Which variants to train (default: conv, residual, triplet)")
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--save_dir",     default="runs/ablation")
    p.add_argument("--no_amp",       action="store_true")
    args = p.parse_args()

    save_dir = (Path(args.save_dir) /
                f"{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(save_dir)

    logger.info("="*60)
    logger.info("  Ablation: original | conv | residual | triplet NAM")
    logger.info("="*60)
    logger.info(f"  Args: {vars(args)}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Device: {device}")

    if not os.path.isdir(args.data):
        logger.error(f"Dataset not found: {args.data}")
        sys.exit(1)

    train_loader, val_loader, class_names = get_dataloaders(
        root=args.data, train_ratio=args.train_ratio,
        image_size=args.img_size, batch_size=args.batch_size,
        num_workers=args.num_workers, seed=args.seed,
    )
    args.num_classes = args.num_classes or len(class_names)

    results = []
    for variant in args.variants:
        r = run(variant, args, train_loader, val_loader,
                class_names, save_dir, logger, device)
        results.append(r)

    # ── Comparison table ──────────────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("  FINAL RESULTS")
    logger.info("="*60)
    hdr = (f"{'Variant':<12} {'Acc%':>7} {'Params(M)':>10} "
           f"{'MB':>6} {'TrainMem':>9} {'InfMS':>7} {'FPS':>6}")
    logger.info(hdr)
    logger.info("-"*60)

    best_variant = max(results, key=lambda r: r['best_val_acc'])
    baseline_p = next((r['total_params'] for r in results if r['variant'] == 'original'), None)


    ORIGINAL_ACC    = 94.55
    ORIGINAL_PARAMS = baseline_p  # None if not run

    logger.info(f"{'original':<12} {ORIGINAL_ACC:>7.2f} {'(reference — not rerun)':>45}")

    for r in results:
        marker      = " ★" if r['variant'] == best_variant['variant'] else ""
        delta_p     = f"{r['total_params'] - ORIGINAL_PARAMS:+,}" if ORIGINAL_PARAMS else "N/A"
        delta_acc   = r['best_val_acc'] - ORIGINAL_ACC
        logger.info(
            f"{r['variant']:<12} {r['best_val_acc']:>7.2f} (Δacc={delta_acc:+.2f}%) "
            f"{r['total_params_M']:>10.4f} {r['model_size_MB']:>6.2f} "
            f"{r['peak_train_mem_MB']:>9.1f} {r['inf_ms']:>7.2f} "
            f"{r['fps']:>6.1f}  Δparams={delta_p}{marker}"
        )

    logger.info("="*60)
    logger.info(f"  Best variant: {best_variant['variant']}  "
                f"({best_variant['best_val_acc']:.2f}%)")

    comparison = dict(
        results    = results,
        best       = best_variant['variant'],
        settings   = vars(args),
    )
    with open(save_dir / "ablation_results.json", "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"\n  Saved to: {save_dir}")


if __name__ == "__main__":
    main()