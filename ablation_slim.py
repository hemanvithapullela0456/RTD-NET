"""
ablation_slim.py  —  Stepwise ablation for RTD-Net Slim changes
===============================================================
Trains 4 configurations in order:
    change1   — DSConvBNSiLU only  (conv5 replaced)
    change2   — SPPFSlim only      (SPPF mid_ch = C//4)
    change3   — ECTB slim only     (ECTB mid_ch = C//4)
    all3      — all three changes combined (= rtdnet_slim.py)

Usage:
    python ablation_slim.py --data data/aid --dataset aid
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

CONFIGS = ["change1", "change2", "change3", "all3"]


# ─────────────────────────────────────────────────────────────────────────────
# All building blocks (copy-pasted so this file is self-contained)
# ─────────────────────────────────────────────────────────────────────────────
class ConvBNSiLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=1, stride=1, padding=None, groups=1):
        super().__init__()
        if padding is None:
            padding = kernel // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding, groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.SiLU(inplace=True)
    def forward(self, x): return self.act(self.bn(self.conv(x)))


class DSConvBNSiLU(nn.Module):                          # CHANGE 1
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.dw  = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.pw  = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x): return self.act(self.bn(self.pw(self.dw(x))))


class LEM(nn.Module):
    def __init__(self, in_ch, out_ch=None, C=16):
        super().__init__()
        out_ch = out_ch or in_ch
        mid_ch = max(in_ch // 2, 1)
        br_ch  = max(in_ch // 32, 1)
        self.C = C
        self.conv1    = ConvBNSiLU(in_ch, mid_ch, 1)
        self.branches = nn.ModuleList([
            nn.Sequential(ConvBNSiLU(mid_ch, br_ch, 1), ConvBNSiLU(br_ch, br_ch, 3))
            for _ in range(C)
        ])
        self.conv2 = nn.Conv2d(br_ch * C, out_ch, 1, bias=False)
        self.bn    = nn.BatchNorm2d(out_ch)
        self.skip  = (nn.Identity() if in_ch == out_ch else
                      nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)))
    def forward(self, x):
        feat = self.conv1(x)
        out  = self.bn(self.conv2(torch.cat([b(feat) for b in self.branches], dim=1)))
        return F.silu(out + self.skip(x))


class CMHSA(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.conv_q    = nn.Conv2d(dim, dim, 1, bias=False)
        self.conv_k    = nn.Conv2d(dim, dim, 1, bias=False)
        self.conv_v    = nn.Conv2d(dim, dim, 1, bias=False)
        self.inst_norm = nn.InstanceNorm2d(num_heads, affine=True)
        self.head_conv = nn.Conv2d(num_heads, num_heads, 1, bias=False)
        self.proj      = nn.Linear(dim, dim)
    def forward(self, x):
        B, C, H, W = x.shape; T = H * W
        q = self.conv_q(x).flatten(2).view(B, self.num_heads, self.head_dim, T)
        k = self.conv_k(x).flatten(2).view(B, self.num_heads, self.head_dim, T)
        v = self.conv_v(x).flatten(2).view(B, self.num_heads, self.head_dim, T)
        attn = self.inst_norm(F.softmax(
            self.head_conv(torch.einsum('bhdq,bhdT->bhqT', q, k) * self.scale), dim=-1))
        out = torch.einsum('bhqT,bhTd->bhqd', attn, v.permute(0,1,3,2)).contiguous()
        return self.proj(out.view(B, T, C)).permute(0,2,1).view(B, C, H, W)


class ECTB_Original(nn.Module):                         # mid = C//2  (baseline)
    def __init__(self, in_ch, out_ch=None, num_heads=4):
        super().__init__()
        out_ch = out_ch or in_ch
        mid_ch = max(in_ch // 2, 1)
        self.conv1 = ConvBNSiLU(in_ch, mid_ch, 1)
        self.cmhsa = CMHSA(mid_ch, num_heads=num_heads)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn    = nn.BatchNorm2d(out_ch)
        self.skip  = (nn.Identity() if in_ch == out_ch else
                      nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)))
    def forward(self, x):
        return F.silu(self.bn(self.conv2(self.cmhsa(self.conv1(x)))) + self.skip(x))


class ECTB_Slim(nn.Module):                             # CHANGE 3: mid = C//4
    def __init__(self, in_ch, out_ch=None, num_heads=4):
        super().__init__()
        out_ch = out_ch or in_ch
        mid_ch = max(in_ch // 4, 1)
        # ensure mid_ch divisible by num_heads
        while mid_ch % num_heads != 0:
            num_heads = max(1, num_heads // 2)
        self.conv1 = ConvBNSiLU(in_ch, mid_ch, 1)
        self.cmhsa = CMHSA(mid_ch, num_heads=num_heads)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn    = nn.BatchNorm2d(out_ch)
        self.skip  = (nn.Identity() if in_ch == out_ch else
                      nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)))
    def forward(self, x):
        return F.silu(self.bn(self.conv2(self.cmhsa(self.conv1(x)))) + self.skip(x))


class SPPF_Original(nn.Module):                         # mid = C//2  (baseline)
    def __init__(self, in_ch, out_ch, pool_size=5):
        super().__init__()
        mid_ch   = in_ch // 2
        self.cv1  = ConvBNSiLU(in_ch,    mid_ch, 1)
        self.cv2  = ConvBNSiLU(mid_ch*4, out_ch, 1)
        self.pool = nn.MaxPool2d(pool_size, 1, pool_size // 2)
    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x, self.pool(x), self.pool(self.pool(x)),
                                   self.pool(self.pool(self.pool(x)))], dim=1))


class SPPF_Slim(nn.Module):                             # CHANGE 2: mid = C//4
    def __init__(self, in_ch, out_ch, pool_size=5):
        super().__init__()
        mid_ch    = in_ch // 4
        self.cv1  = ConvBNSiLU(in_ch,    mid_ch, 1)
        self.cv2  = ConvBNSiLU(mid_ch*4, out_ch, 1)
        self.pool = nn.MaxPool2d(pool_size, 1, pool_size // 2)
    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x, self.pool(x), self.pool(self.pool(x)),
                                   self.pool(self.pool(self.pool(x)))], dim=1))


class NAMChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)
    def forward(self, x):
        normed = self.bn(x)
        w = self.bn.weight.abs()
        w = w / (w.sum() + 1e-8)
        return x * torch.sigmoid(w.view(1,-1,1,1) * normed)

class NAMSpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bn = nn.InstanceNorm2d(channels, affine=True)
    def forward(self, x):
        normed = self.bn(x)
        w = self.bn.weight.abs()
        w = w / (w.sum() + 1e-8)
        return x * torch.sigmoid(w.view(1,-1,1,1) * normed)

class NAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel = NAMChannelAttention(channels)
        self.spatial = NAMSpatialAttention(channels)
    def forward(self, x): return self.spatial(self.channel(x))

class APH(nn.Module):
    def __init__(self, in_ch, out_ch=None):
        super().__init__()
        out_ch = out_ch or in_ch
        self.nam  = NAM(in_ch)
        self.conv = ConvBNSiLU(in_ch, out_ch, 1)
    def forward(self, x): return self.conv(self.nam(x))


# ─────────────────────────────────────────────────────────────────────────────
# Model factory — assembles any of the 5 configs from the blocks above
# ─────────────────────────────────────────────────────────────────────────────
class RTDNetAblation(nn.Module):
    """
    Builds one of 5 configs by toggling each change independently:
        use_ds_conv  — CHANGE 1: DSConv for conv5
        use_sppf_slim — CHANGE 2: SPPFSlim
        use_ectb_slim — CHANGE 3: ECTB slim bottleneck
    """
    def __init__(self, num_classes=30, base_ch=32, num_heads=4, C=16,
                 dropout=0.3, use_ds_conv=False, use_sppf_slim=False,
                 use_ectb_slim=False):
        super().__init__()
        b = base_ch

        self.conv1 = ConvBNSiLU(3,   b,   3, stride=2)
        self.conv2 = ConvBNSiLU(b,   b*2, 3, stride=2)
        self.lem1  = LEM(b*2, b*2, C=C)
        self.conv3 = ConvBNSiLU(b*2, b*4, 3, stride=2)
        self.lem2  = LEM(b*4, b*4, C=C)
        self.conv4 = ConvBNSiLU(b*4, b*8, 3, stride=2)
        self.lem3  = LEM(b*8, b*8, C=C)

        # CHANGE 1 toggle
        self.conv5 = (DSConvBNSiLU(b*8, b*16, stride=2) if use_ds_conv else
                      ConvBNSiLU(b*8, b*16, 3, stride=2))

        # CHANGE 3 toggle
        self.ectb  = (ECTB_Slim(b*16, b*16, num_heads=num_heads) if use_ectb_slim else
                      ECTB_Original(b*16, b*16, num_heads=num_heads))

        # CHANGE 2 toggle
        self.sppf  = (SPPF_Slim(b*16, b*16) if use_sppf_slim else
                      SPPF_Original(b*16, b*16))

        self.aph  = APH(b*16, b*16)
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(b*16, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None: nn.init.ones_(m.weight)
                if m.bias   is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x = self.conv3(self.lem1(x))
        x = self.conv4(self.lem2(x))
        x = self.conv5(self.lem3(x))
        x = self.sppf(self.ectb(x))
        x = self.aph(x)
        return self.fc(self.drop(self.gap(x).flatten(1)))

    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# Maps config name → which flags to enable
CONFIG_FLAGS = {
    "baseline": dict(use_ds_conv=False, use_sppf_slim=False, use_ectb_slim=False),
    "change1":  dict(use_ds_conv=True,  use_sppf_slim=False, use_ectb_slim=False),
    "change2":  dict(use_ds_conv=False, use_sppf_slim=True,  use_ectb_slim=False),
    "change3":  dict(use_ds_conv=False, use_sppf_slim=False, use_ectb_slim=True),
    "all3":     dict(use_ds_conv=True,  use_sppf_slim=True,  use_ectb_slim=True),
}

CONFIG_DESC = {
    "baseline": "Original model  (no changes)",
    "change1":  "Change 1 only   (DSConv replaces conv5)",
    "change2":  "Change 2 only   (SPPFSlim mid=C//4)",
    "change3":  "Change 3 only   (ECTB slim mid=C//4)",
    "all3":     "All 3 changes   (= rtdnet_slim.py)",
}


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers  (identical to your existing scripts)
# ─────────────────────────────────────────────────────────────────────────────
def setup_logger(save_dir):
    logger = logging.getLogger("ablation_slim")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter("%(asctime)s  %(message)s")
    for h in [logging.FileHandler(save_dir / "ablation.log"),
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
        self.avg = self.sum / self.count

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
            acc1, = topk_acc(model(imgs), labels, topk=(1,))
        lm.update(loss.item(), imgs.size(0))
        am.update(acc1,        imgs.size(0))
        if (step+1) % 20 == 0:
            logger.info(f"  Ep{ep:3d} {step+1:4d}/{len(loader)} loss={lm.avg:.4f} acc={am.avg:.2f}%")
    return lm.avg, am.avg

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    lm, am1, am5 = AverageMeter(), AverageMeter(), AverageMeter()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        out  = model(imgs)
        loss = criterion(out, labels)
        k = min(5, out.size(1))
        a1, a5 = topk_acc(out, labels, topk=(1, k))
        n = imgs.size(0)
        lm.update(loss.item(),n); am1.update(a1,n); am5.update(a5,n)
    return lm.avg, am1.avg, am5.avg


# ─────────────────────────────────────────────────────────────────────────────
# Single config run
# ─────────────────────────────────────────────────────────────────────────────
def run(config, args, train_loader, val_loader, class_names, save_dir, logger, device):
    num_classes = args.num_classes or len(class_names)
    flags = CONFIG_FLAGS[config]

    model = RTDNetAblation(
        num_classes=num_classes, base_ch=args.base_ch,
        num_heads=args.num_heads, C=args.C, dropout=args.dropout,
        **flags,
    ).to(device)

    total, _ = model.count_parameters()
    logger.info(f"\n{'='*65}")
    logger.info(f"  Config  : {config}  —  {CONFIG_DESC[config]}")
    logger.info(f"  Flags   : ds_conv={flags['use_ds_conv']}  "
                f"sppf_slim={flags['use_sppf_slim']}  "
                f"ectb_slim={flags['use_ectb_slim']}")
    logger.info(f"  Params  : {total:,}  ({total/1e6:.4f} M)")
    logger.info(f"{'='*65}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay,
                          nesterov=True)
    scheduler = MultiStepLR(optimizer, milestones=args.lr_steps, gamma=0.1)
    scaler    = GradScaler() if (device.type=="cuda" and not args.no_amp) else None

    best_acc, best_ep, no_imp = 0., 0, 0
    history = dict(train_loss=[], train_acc=[], val_loss=[], val_acc=[])

    for ep in range(1, args.epochs+1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion,
                                      optimizer, scaler, device, logger, ep)
        va_loss, va_acc, va5 = validate(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(
            f"Ep{ep:3d}/{args.epochs} lr={scheduler.get_last_lr()[0]:.5f} | "
            f"tr {tr_loss:.4f}/{tr_acc:.2f}% | "
            f"va {va_loss:.4f}/{va_acc:.2f}% top5={va5:.2f}% | {time.time()-t0:.0f}s"
        )
        for k, v in zip(['train_loss','train_acc','val_loss','val_acc'],
                        [tr_loss, tr_acc, va_loss, va_acc]):
            history[k].append(round(v, 4))

        if va_acc > best_acc:
            best_acc, best_ep, no_imp = va_acc, ep, 0
            torch.save(dict(epoch=ep, model_state=model.state_dict(),
                            val_acc=va_acc, config=config),
                       save_dir / f"{config}_best.pt")
            logger.info(f"  ★ new best {best_acc:.2f}%  (epoch {best_ep})")
        else:
            no_imp += 1

        with open(save_dir / f"{config}_history.json", "w") as f:
            json.dump(history, f, indent=2)

        if no_imp >= args.patience:
            logger.info(f"  Early stopping at epoch {ep}.")
            logger.info(f"  Best Val Accuracy : {best_acc:.2f}%  (epoch {best_ep})")
            break

    # Inference latency
    model.eval()
    dummy = torch.randn(1, 3, args.img_size, args.img_size).to(device)
    with torch.no_grad():
        for _ in range(5): model(dummy)
        t0 = time.perf_counter()
        for _ in range(100): model(dummy)
    inf_ms = (time.perf_counter()-t0)/100*1000

    return dict(
        config       = config,
        description  = CONFIG_DESC[config],
        best_val_acc = round(best_acc, 2),
        best_epoch   = best_ep,
        params_M     = round(total/1e6, 4),
        size_MB      = round(total*4/1024**2, 2),
        inf_ms       = round(inf_ms, 2),
        fps          = round(1000/inf_ms, 1),
        **{f"flag_{k}": v for k, v in flags.items()},
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
    p.add_argument("--configs",      nargs="+",  default=CONFIGS, choices=CONFIGS,
                   help="Which configs to run (default: all 5)")
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--save_dir",     default="runs/ablation_slim")
    p.add_argument("--no_amp",       action="store_true")
    args = p.parse_args()

    save_dir = (Path(args.save_dir) /
                f"{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(save_dir)

    logger.info("="*65)
    logger.info("  Stepwise Slim Ablation: baseline → C1 → C2 → C3 → all3")
    logger.info("="*65)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Device: {device}")

    if not os.path.isdir(args.data):
        logger.error(f"Dataset not found: {args.data}"); sys.exit(1)

    train_loader, val_loader, class_names = get_dataloaders(
        root=args.data, train_ratio=args.train_ratio,
        image_size=args.img_size, batch_size=args.batch_size,
        num_workers=args.num_workers, seed=args.seed,
    )
    args.num_classes = args.num_classes or len(class_names)

    results = []
    for config in args.configs:
        r = run(config, args, train_loader, val_loader,
                class_names, save_dir, logger, device)
        results.append(r)

    # ── Final comparison table ────────────────────────────────────────────────
    logger.info("\n" + "="*65)
    logger.info("  STEPWISE ABLATION RESULTS")
    logger.info("="*65)
    logger.info(f"  {'Config':<10} {'Acc%':>7} {'ΔAcc':>7} {'Params(M)':>10} "
                f"{'MB':>6} {'InfMS':>7} {'FPS':>6}")
    logger.info("-"*65)

    baseline_acc = next((r['best_val_acc'] for r in results if r['config']=='baseline'), 94.55)
    for r in results:
        delta = r['best_val_acc'] - baseline_acc
        marker = "  ★" if r['best_val_acc'] == max(x['best_val_acc'] for x in results) else ""
        logger.info(
            f"  {r['config']:<10} {r['best_val_acc']:>7.2f} {delta:>+7.2f} "
            f"{r['params_M']:>10.4f} {r['size_MB']:>6.2f} "
            f"{r['inf_ms']:>7.2f} {r['fps']:>6.1f}{marker}"
        )

    logger.info("="*65)

    with open(save_dir / "ablation_slim_results.json", "w") as f:
        json.dump(dict(results=results, settings=vars(args)), f, indent=2)
    logger.info(f"  Saved to: {save_dir}")


if __name__ == "__main__":
    main()