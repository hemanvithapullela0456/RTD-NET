"""
ablation_nam_coordinate.py  —  Ablation: NAM vs NAM+CoordinateAttention
========================================================================
Runs four configs in order and produces a comparison table:

  slim_baseline   RTDNetNAMCoordinate with CA disabled (vanilla NAM only)
  ca_r64          NAM + Coordinate Attention, r=64  (light CA)
  ca_r32          NAM + Coordinate Attention, r=32  (default, paper setting)
  ca_r16          NAM + Coordinate Attention, r=16  (strong CA)

Usage:
    python ablation_nam_coordinate.py --data data/aid --dataset aid
    python ablation_nam_coordinate.py --data data/nwpu --dataset nwpu \
        --num_classes 45 --epochs 150
"""

import os, sys, time, json, logging, argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.cuda.amp import GradScaler, autocast

from dataset import get_dataloaders
from dropped.rtdnet_nam_coordinate import (
    RTDNetNAMCoordinate,
    NAMChannelAttention,
    CoordinateSpatialAttention,
)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline model: replace CoordinateSpatialAttention with original NAM spatial
# ─────────────────────────────────────────────────────────────────────────────
class NAMSpatialAttention(nn.Module):
    """Original NAM spatial attention (InstanceNorm-based) — for baseline only."""
    def __init__(self, channels: int):
        super().__init__()
        self.bn = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.bn(x)
        lam    = self.bn.weight.abs()
        w      = lam / (lam.sum() + 1e-8)
        return x * torch.sigmoid(w.view(1, -1, 1, 1) * normed)


class OriginalNAM(nn.Module):
    """Vanilla NAM (channel + original spatial) — baseline."""
    def __init__(self, channels: int):
        super().__init__()
        self.channel = NAMChannelAttention(channels)
        self.spatial = NAMSpatialAttention(channels)

    def forward(self, x):
        return self.spatial(self.channel(x))


class RTDNetNAMCoordinateBaseline(RTDNetNAMCoordinate):
    """
    Same as RTDNetNAMCoordinate but APH uses original NAM (no Coordinate Attention).
    Used as the slim_baseline config in the ablation.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Re-wire APH.nam to vanilla NAM
        import torch.nn as nn
        from dropped.rtdnet_nam_coordinate import ConvBNSiLU
        in_ch = kwargs.get("base_ch", 32) * 16

        class _APHBaseline(nn.Module):
            def __init__(self, in_ch):
                super().__init__()
                self.nam  = OriginalNAM(in_ch)
                self.conv = ConvBNSiLU(in_ch, in_ch, 1)
            def forward(self, x):
                return self.conv(self.nam(x))

        self.aph = _APHBaseline(in_ch).to(next(self.parameters()).device)


# ─────────────────────────────────────────────────────────────────────────────
# Config registry
# ─────────────────────────────────────────────────────────────────────────────
CONFIGS = ["slim_baseline", "ca_r64", "ca_r32", "ca_r16"]

CONFIG_DESC = {
    "slim_baseline": "RTDNet-Slim + original NAM    (no Coordinate Attention)",
    "ca_r64"       : "RTDNet-Slim + NAM + CA r=64  (light Coordinate Attention)",
    "ca_r32"       : "RTDNet-Slim + NAM + CA r=32  (default, paper setting)",
    "ca_r16"       : "RTDNet-Slim + NAM + CA r=16  (strong Coordinate Attention)",
}

CONFIG_CA_R = {
    "slim_baseline": None,   # no CA
    "ca_r64"       : 64,
    "ca_r32"       : 32,
    "ca_r16"       : 16,
}


def build_model(config: str, num_classes: int, base_ch: int, num_heads: int,
                C: int, dropout: float) -> nn.Module:
    ca_r = CONFIG_CA_R[config]
    if ca_r is None:
        return RTDNetNAMCoordinateBaseline(
            num_classes=num_classes, base_ch=base_ch,
            num_heads=num_heads, C=C, dropout=dropout,
        )
    return RTDNetNAMCoordinate(
        num_classes=num_classes, base_ch=base_ch,
        num_heads=num_heads, C=C, dropout=dropout, ca_r=ca_r,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (same as train_nam_coordinate.py)
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(save_dir):
    logger = logging.getLogger("ablation_nam_ca")
    logger.setLevel(logging.INFO); logger.handlers = []
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    for h in [logging.FileHandler(save_dir / "ablation.log"),
              logging.StreamHandler(sys.stdout)]:
        h.setFormatter(fmt); logger.addHandler(h)
    logger.propagate = False
    return logger


class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0.0
    def update(self, val, n=1):
        self.sum += val*n; self.count += n; self.avg = self.sum/self.count


def topk_acc(output, target, topk=(1,)):
    maxk = max(topk); bsz = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    correct = pred.t().eq(target.view(1,-1).expand_as(pred.t()))
    return [correct[:k].reshape(-1).float().sum().mul_(100./bsz).item() for k in topk]


def train_epoch(model, loader, criterion, optimizer, scaler, device, logger, ep):
    model.train()
    lm, am = AverageMeter(), AverageMeter()
    for step, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with autocast(enabled=scaler is not None):
            loss = criterion(model(imgs), labels)
        optimizer.zero_grad(set_to_none=True)
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
            acc1, = topk_acc(model(imgs), labels)
        lm.update(loss.item(), imgs.size(0))
        am.update(acc1, imgs.size(0))
        if (step+1) % 20 == 0:
            logger.info(f"  Ep{ep:3d} {step+1:4d}/{len(loader)}"
                        f"  loss={lm.avg:.4f}  acc={am.avg:.2f}%")
    return lm.avg, am.avg


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    lm, am1, am5 = AverageMeter(), AverageMeter(), AverageMeter()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        out  = model(imgs); loss = criterion(out, labels)
        k    = min(5, out.size(1))
        a1, a5 = topk_acc(out, labels, topk=(1, k))
        n = imgs.size(0)
        lm.update(loss.item(),n); am1.update(a1,n); am5.update(a5,n)
    return lm.avg, am1.avg, am5.avg


# ─────────────────────────────────────────────────────────────────────────────
# Single config run
# ─────────────────────────────────────────────────────────────────────────────

def run(config, args, train_loader, val_loader, save_dir, logger, device):
    num_classes = args.num_classes
    model = build_model(config, num_classes, args.base_ch,
                        args.num_heads, args.C, args.dropout).to(device)
    total = sum(p.numel() for p in model.parameters())

    logger.info(f"\n{'='*70}")
    logger.info(f"  Config  : {config}")
    logger.info(f"  Desc    : {CONFIG_DESC[config]}")
    logger.info(f"  CA r    : {CONFIG_CA_R[config]}")
    logger.info(f"  Params  : {total:,}  ({total/1e6:.4f} M)")
    logger.info(f"{'='*70}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay,
                          nesterov=True)
    scheduler = MultiStepLR(optimizer, milestones=args.lr_steps, gamma=0.1)
    scaler    = GradScaler() if (device.type == "cuda" and not args.no_amp) else None

    best_acc, best_ep, no_imp = 0., 0, 0
    history = dict(train_loss=[], train_acc=[], val_loss=[], val_acc=[])

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion,
                                      optimizer, scaler, device, logger, ep)
        va_loss, va_acc, va5 = validate(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(
            f"Ep{ep:3d}/{args.epochs}"
            f"  lr={scheduler.get_last_lr()[0]:.5f}"
            f"  | tr {tr_loss:.4f}/{tr_acc:.2f}%"
            f"  | val {va_loss:.4f}/{va_acc:.2f}%"
            f"  top5={va5:.2f}%  | {time.time()-t0:.0f}s"
        )
        for k, v in zip(["train_loss","train_acc","val_loss","val_acc"],
                         [tr_loss, tr_acc, va_loss, va_acc]):
            history[k].append(round(v, 4))

        if va_acc > best_acc:
            best_acc, best_ep, no_imp = va_acc, ep, 0
            torch.save({"epoch": ep, "model_state": model.state_dict(),
                        "val_acc": va_acc, "config": config},
                       save_dir / f"{config}_best.pt")
            logger.info(f"  ★ new best {best_acc:.2f}%  (epoch {best_ep})")
        else:
            no_imp += 1

        with open(save_dir / f"{config}_history.json", "w") as f:
            json.dump(history, f, indent=2)

        if no_imp >= args.patience:
            logger.info(f"  Early stop at epoch {ep}.")
            break

    # latency
    model.eval()
    dummy = torch.randn(1, 3, args.img_size, args.img_size).to(device)
    with torch.no_grad():
        for _ in range(5): model(dummy)
        t0 = time.perf_counter()
        for _ in range(100): model(dummy)
    inf_ms = (time.perf_counter() - t0) / 100 * 1000

    return dict(
        config      = config,
        description = CONFIG_DESC[config],
        ca_r        = CONFIG_CA_R[config],
        best_val_acc= round(best_acc, 2),
        best_epoch  = best_ep,
        params_M    = round(total/1e6, 4),
        size_MB     = round(total*4/1024**2, 2),
        inf_ms      = round(inf_ms, 2),
        fps         = round(1000/inf_ms, 1),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",         default="data/aid")
    p.add_argument("--dataset",      default="aid", choices=["aid","nwpu"])
    p.add_argument("--num_classes",  type=int, default=None)
    p.add_argument("--train_ratio",  type=float, default=0.8)
    p.add_argument("--img_size",     type=int, default=640)
    p.add_argument("--base_ch",      type=int, default=32)
    p.add_argument("--num_heads",    type=int, default=4)
    p.add_argument("--C",            type=int, default=16)
    p.add_argument("--dropout",      type=float, default=0.3)
    p.add_argument("--epochs",       type=int, default=300)
    p.add_argument("--batch_size",   type=int, default=64)
    p.add_argument("--lr",           type=float, default=0.01)
    p.add_argument("--momentum",     type=float, default=0.937)
    p.add_argument("--weight_decay", type=float, default=0.0005)
    p.add_argument("--lr_steps",     nargs="+", type=int, default=[100,200])
    p.add_argument("--patience",     type=int, default=50)
    p.add_argument("--configs",      nargs="+", default=CONFIGS, choices=CONFIGS)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--save_dir",     default="runs/ablation_nam_ca")
    p.add_argument("--no_amp",       action="store_true")
    args = p.parse_args()

    save_dir = (Path(args.save_dir) /
                f"{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(save_dir)

    logger.info("=" * 70)
    logger.info("  Ablation: slim_baseline → ca_r64 → ca_r32 → ca_r16")
    logger.info("=" * 70)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Device: {device}")

    if not os.path.isdir(args.data):
        logger.error(f"Dataset not found: {args.data}"); sys.exit(1)

    train_loader, val_loader, class_names = get_dataloaders(
        root=args.data, train_ratio=args.train_ratio, image_size=args.img_size,
        batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed,
    )
    args.num_classes = args.num_classes or len(class_names)

    results = []
    for config in args.configs:
        r = run(config, args, train_loader, val_loader, save_dir, logger, device)
        results.append(r)

    # ── Final table ────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("  ABLATION RESULTS — NAM vs NAM + Coordinate Attention")
    logger.info("=" * 70)
    logger.info(f"  {'Config':<16} {'CA_r':>5} {'Acc%':>7} {'ΔAcc':>7} "
                f"{'Params(M)':>10} {'MB':>6} {'ms':>7} {'FPS':>6}")
    logger.info("-" * 70)

    baseline_acc = next((r["best_val_acc"] for r in results
                         if r["config"] == "slim_baseline"), None)
    for r in results:
        delta  = (r["best_val_acc"] - baseline_acc) if baseline_acc else 0.0
        marker = "  ★" if r["best_val_acc"] == max(x["best_val_acc"] for x in results) else ""
        ca_r_s = str(r["ca_r"]) if r["ca_r"] else "none"
        logger.info(
            f"  {r['config']:<16} {ca_r_s:>5} {r['best_val_acc']:>7.2f} "
            f"{delta:>+7.2f} {r['params_M']:>10.4f} {r['size_MB']:>6.2f} "
            f"{r['inf_ms']:>7.2f} {r['fps']:>6.1f}{marker}"
        )
    logger.info("=" * 70)

    with open(save_dir / "ablation_results.json", "w") as f:
        json.dump({"results": results, "settings": vars(args)}, f, indent=2)
    logger.info(f"  Saved → {save_dir}")


if __name__ == "__main__":
    main()