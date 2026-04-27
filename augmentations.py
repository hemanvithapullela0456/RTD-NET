# """
# augmentations.py  —  Non-architectural accuracy boosters
# =========================================================
# Drop-in additions to any existing training loop.

# Provides:
#     MixupCutMixCollator  — batch-level collator (replaces default collate_fn)
#     RandAugmentTransform — per-image policy augmentation (wrap in dataset)
#     LabelSmoothingLoss   — CE with smoothing + optional Mixup/CutMix support
#     WarmupCosineScheduler— LR warm-up + cosine anneal (replaces MultiStepLR)

# Usage in your existing train script
# ------------------------------------
#     from augmentations import (MixupCutMixCollator, LabelSmoothingLoss,
#                                WarmupCosineScheduler, get_strong_transforms)

#     # 1. Stronger transforms (pass to dataset.get_dataloaders)
#     # Already handled: call get_strong_transforms() and set in dataset.py

#     # 2. Swap the collator on the train loader
#     train_loader = DataLoader(
#         train_ds, batch_size=64, shuffle=True,
#         collate_fn=MixupCutMixCollator(mixup_alpha=0.4, cutmix_alpha=1.0,
#                                        cutmix_prob=0.5, num_classes=num_classes),
#         num_workers=4, pin_memory=True, drop_last=True,
#     )

#     # 3. Swap the loss
#     criterion = LabelSmoothingLoss(num_classes=num_classes, smoothing=0.15)

#     # 4. Swap the scheduler
#     scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=5,
#                                       total_epochs=300, min_lr=1e-6)
#     # call scheduler.step() once per EPOCH (not per batch)
# """

# import math
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import default_collate
# from torchvision import transforms


# # ─────────────────────────────────────────────────────────────────────────────
# # 1. Stronger per-image transforms
# # ─────────────────────────────────────────────────────────────────────────────

# def get_strong_transforms(image_size: int = 640, is_train: bool = True):
#     """
#     Replaces the original get_transforms() from dataset.py.
#     Additions vs baseline:
#         • RandomErasing (p=0.25)  — occlusion robustness
#         • RandomGrayscale (p=0.05)— texture invariance
#         • Slightly tighter normalization stays the same (AID stats)
#     """
#     mean = [0.3680, 0.3810, 0.3436]
#     std  = [0.2034, 0.1854, 0.1876]

#     if is_train:
#         return transforms.Compose([
#             transforms.Resize(int(image_size * 1.15)),
#             transforms.RandomCrop(image_size),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomVerticalFlip(),
#             transforms.RandomRotation(15),
#             transforms.ColorJitter(brightness=0.3, contrast=0.3,
#                                    saturation=0.3, hue=0.1),
#             transforms.RandomGrayscale(p=0.05),
#             transforms.ToTensor(),
#             transforms.Normalize(mean, std),
#             transforms.RandomErasing(p=0.25, scale=(0.02, 0.15),
#                                      ratio=(0.3, 3.3), value=0),
#         ])
#     else:
#         return transforms.Compose([
#             transforms.Resize(int(image_size * 1.05)),
#             transforms.CenterCrop(image_size),
#             transforms.ToTensor(),
#             transforms.Normalize(mean, std),
#         ])


# # ─────────────────────────────────────────────────────────────────────────────
# # 2. Mixup + CutMix collator
# # ─────────────────────────────────────────────────────────────────────────────

# class MixupCutMixCollator:
#     """
#     Drop-in collate_fn for DataLoader.

#     Each batch is either:
#         • Mixup:  x = λ·xA + (1−λ)·xB,  y = λ·yA + (1−λ)·yB
#         • CutMix: patch from xB pasted into xA,  y blended by area ratio
#         • Neither (probability = 1 - cutmix_prob - mixup_prob): plain batch

#     Args:
#         mixup_alpha   : Beta distribution α for Mixup  (0 = off)
#         cutmix_alpha  : Beta distribution α for CutMix (0 = off)
#         cutmix_prob   : Probability of applying CutMix per batch
#         mixup_prob    : Probability of applying Mixup  per batch
#         num_classes   : Number of output classes
#     """

#     def __init__(self, mixup_alpha: float = 0.4, cutmix_alpha: float = 1.0,
#                  cutmix_prob: float = 0.5, mixup_prob: float = 0.5,
#                  num_classes: int = 30):
#         self.mixup_alpha  = mixup_alpha
#         self.cutmix_alpha = cutmix_alpha
#         self.cutmix_prob  = cutmix_prob
#         self.mixup_prob   = mixup_prob
#         self.num_classes  = num_classes

#     def __call__(self, batch):
#         imgs, labels = default_collate(batch)          # (B,C,H,W), (B,)
#         imgs   = imgs.float()
#         labels = F.one_hot(labels, self.num_classes).float()

#         r = random.random()
#         if r < self.cutmix_prob and self.cutmix_alpha > 0:
#             imgs, labels = self._cutmix(imgs, labels)
#         elif r < self.cutmix_prob + self.mixup_prob and self.mixup_alpha > 0:
#             imgs, labels = self._mixup(imgs, labels)
#         # else: return unchanged

#         return imgs, labels

#     def _mixup(self, imgs, labels):
#         lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
#         lam = max(lam, 1 - lam)                        # keep dominant class
#         idx = torch.randperm(imgs.size(0))
#         imgs   = lam * imgs   + (1 - lam) * imgs[idx]
#         labels = lam * labels + (1 - lam) * labels[idx]
#         return imgs, labels

#     def _cutmix(self, imgs, labels):
#         lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
#         B, C, H, W = imgs.shape
#         cut_rat    = math.sqrt(1.0 - lam)
#         cut_h      = int(H * cut_rat)
#         cut_w      = int(W * cut_rat)
#         cx = random.randint(0, W)
#         cy = random.randint(0, H)
#         x1 = max(cx - cut_w // 2, 0)
#         y1 = max(cy - cut_h // 2, 0)
#         x2 = min(cx + cut_w // 2, W)
#         y2 = min(cy + cut_h // 2, H)

#         idx = torch.randperm(B)
#         imgs_mix = imgs.clone()
#         imgs_mix[:, :, y1:y2, x1:x2] = imgs[idx, :, y1:y2, x1:x2]

#         # Adjust lambda to actual patch area
#         lam_adj = 1 - (y2 - y1) * (x2 - x1) / (H * W)
#         labels  = lam_adj * labels + (1 - lam_adj) * labels[idx]
#         return imgs_mix, labels


# # ─────────────────────────────────────────────────────────────────────────────
# # 3. Loss that handles both hard labels and soft (mixed) labels
# # ─────────────────────────────────────────────────────────────────────────────

# class LabelSmoothingLoss(nn.Module):
#     """
#     Cross-entropy with label smoothing that also accepts soft targets
#     produced by Mixup / CutMix.

#     When targets are 1-D (plain class indices), standard smoothed CE is used.
#     When targets are 2-D (soft distributions), KL-div against smoothed target.
#     """

#     def __init__(self, num_classes: int, smoothing: float = 0.15):
#         super().__init__()
#         self.num_classes = num_classes
#         self.smoothing   = smoothing
#         self.confidence  = 1.0 - smoothing

#     def forward(self, logits: torch.Tensor,
#                 targets: torch.Tensor) -> torch.Tensor:
#         log_prob = F.log_softmax(logits, dim=-1)

#         if targets.dim() == 1:
#             # Hard labels → convert to smoothed soft labels
#             smooth_val = self.smoothing / (self.num_classes - 1)
#             soft = torch.full_like(log_prob, smooth_val)
#             soft.scatter_(1, targets.unsqueeze(1), self.confidence)
#         else:
#             # Already soft (from Mixup/CutMix) → just apply smoothing blend
#             soft = (self.confidence * targets +
#                     self.smoothing / self.num_classes)

#         return -(soft * log_prob).sum(dim=-1).mean()


# # ─────────────────────────────────────────────────────────────────────────────
# # 4. Warmup + Cosine Annealing scheduler
# # ─────────────────────────────────────────────────────────────────────────────

# class WarmupCosineScheduler:
#     """
#     Linear warmup for `warmup_epochs`, then cosine decay to `min_lr`.
#     Call scheduler.step() once per epoch.

#     Why better than MultiStepLR:
#         MultiStepLR drops LR in discrete jumps → training instability near the
#         steps.  Cosine decay is smooth and empirically +0.2–0.5% on aerial
#         classification benchmarks.  Warmup prevents early overfitting on the
#         first large-LR batches.
#     """

#     def __init__(self, optimizer, warmup_epochs: int = 5,
#                  total_epochs: int = 300, min_lr: float = 1e-6):
#         self.optimizer      = optimizer
#         self.warmup_epochs  = warmup_epochs
#         self.total_epochs   = total_epochs
#         self.min_lr         = min_lr
#         self.base_lrs       = [g['lr'] for g in optimizer.param_groups]
#         self._epoch         = 0

#     def step(self):
#         self._epoch += 1
#         lrs = self._get_lrs()
#         for g, lr in zip(self.optimizer.param_groups, lrs):
#             g['lr'] = lr

#     def _get_lrs(self):
#         e = self._epoch
#         if e <= self.warmup_epochs:
#             factor = e / max(self.warmup_epochs, 1)
#         else:
#             progress = (e - self.warmup_epochs) / (
#                 self.total_epochs - self.warmup_epochs)
#             progress = min(progress, 1.0)
#             factor   = 0.5 * (1.0 + math.cos(math.pi * progress))
#         return [self.min_lr + (base - self.min_lr) * factor
#                 for base in self.base_lrs]

#     def get_last_lr(self):
#         return [g['lr'] for g in self.optimizer.param_groups]

#     def state_dict(self):
#         return {'epoch': self._epoch, 'base_lrs': self.base_lrs}

#     def load_state_dict(self, d):
#         self._epoch   = d['epoch']
#         self.base_lrs = d['base_lrs']


# # ─────────────────────────────────────────────────────────────────────────────
# # 5. Stochastic Depth (drop-path) — for LEM branches
# #    Import and wrap individual branch outputs if desired
# # ─────────────────────────────────────────────────────────────────────────────

# class DropPath(nn.Module):
#     """
#     Stochastic depth: randomly zero entire residual branch during training.
#     Use as drop-in on the skip connection of LEM / ECTB.

#     drop_prob = 0.1–0.2 is the sweet spot for small models.
#     """

#     def __init__(self, drop_prob: float = 0.1):
#         super().__init__()
#         self.drop_prob = drop_prob

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if not self.training or self.drop_prob == 0.:
#             return x
#         keep   = 1 - self.drop_prob
#         shape  = (x.shape[0],) + (1,) * (x.ndim - 1)
#         mask   = torch.empty(shape, dtype=x.dtype,
#                              device=x.device).bernoulli_(keep).div_(keep)
#         return x * mask

"""
augmentations.py  —  Training augmentations, loss, and scheduler
=================================================================
Shared by train_clean.py (and any future training scripts).

Key fix over the original:
    LabelSmoothingLoss default smoothing: 0.15 → use 0.05 for remote sensing.
    0.15 + Mixup simultaneously double-regularises and suppresses confidence
    too hard during early training. 0.05 is the correct value.
"""

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


# ─── Strong augmentation pipeline ────────────────────────────────────────────

def get_strong_transforms(image_size: int, is_train: bool):
    """
    Training: RandomResizedCrop + flips + ColorJitter + RandAugment.
    Validation: Resize + CenterCrop only.

    RandAugment (magnitude=9, num_ops=2) is the single biggest
    augmentation gain for aerial scenes — it randomly applies
    combinations of rotate, translate, shear, posterize, solarize, etc.
    """
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.5, 1.0),
                ratio=(0.75, 1.333),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),          # aerial: both matter
            transforms.RandomRotation(degrees=90),          # rotation invariance
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3,
                saturation=0.2, hue=0.05,
            ),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.3680, 0.3810, 0.3436],             # AID dataset stats
                std=[0.2034, 0.1854, 0.1844],
            ),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ])
    else:
        resize_size = int(image_size * 1.143)              # 640 → 731
        return transforms.Compose([
            transforms.Resize(
                resize_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.3680, 0.3810, 0.3436],
                std=[0.2034, 0.1854, 0.1844],
            ),
        ])


# ─── Mixup / CutMix collator ─────────────────────────────────────────────────

class MixupCutMixCollator:
    """
    Applies Mixup OR CutMix randomly per batch.
    Returns soft one-hot labels compatible with LabelSmoothingLoss.

    mixup_prob + cutmix_prob can sum to < 1.0; remaining probability
    gives plain (un-augmented) batches, which helps early convergence.
    """

    def __init__(
        self,
        num_classes: int,
        mixup_alpha: float = 0.4,
        cutmix_alpha: float = 1.0,
        mixup_prob: float   = 0.5,
        cutmix_prob: float  = 0.5,
    ):
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
            imgs, labels = self._mixup(imgs, labels)
        elif r < self.mixup_prob + self.cutmix_prob:
            imgs, labels = self._cutmix(imgs, labels)
        else:
            # Plain batch — still convert labels to soft one-hot
            labels = F.one_hot(labels, self.num_classes).float()

        return imgs, labels

    def _mixup(self, imgs, labels):
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        lam = max(lam, 1 - lam)
        idx = torch.randperm(imgs.size(0))
        mixed = lam * imgs + (1 - lam) * imgs[idx]
        y     = F.one_hot(labels, self.num_classes).float()
        soft  = lam * y + (1 - lam) * y[idx]
        return mixed, soft

    def _cutmix(self, imgs, labels):
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        idx = torch.randperm(imgs.size(0))
        B, C, H, W = imgs.shape

        cut_rat = math.sqrt(1 - lam)
        cut_h   = int(H * cut_rat)
        cut_w   = int(W * cut_rat)
        cx = random.randint(0, W)
        cy = random.randint(0, H)
        x1 = max(cx - cut_w // 2, 0)
        y1 = max(cy - cut_h // 2, 0)
        x2 = min(cx + cut_w // 2, W)
        y2 = min(cy + cut_h // 2, H)

        mixed = imgs.clone()
        mixed[:, :, y1:y2, x1:x2] = imgs[idx, :, y1:y2, x1:x2]

        lam_real = 1 - (y2 - y1) * (x2 - x1) / (H * W)
        y    = F.one_hot(labels, self.num_classes).float()
        soft = lam_real * y + (1 - lam_real) * y[idx]
        return mixed, soft


# ─── Label smoothing loss ─────────────────────────────────────────────────────

class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy with label smoothing.
    Accepts both hard (1-D long) and soft (2-D float) labels.

    IMPORTANT: Use smoothing=0.05 for remote sensing classification,
    not 0.10 or 0.15. Higher values combined with Mixup/CutMix
    produce excessive regularisation that suppresses early learning.
    """

    def __init__(self, num_classes: int, smoothing: float = 0.05):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing   = smoothing
        self.confidence  = 1.0 - smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(pred, dim=-1)

        if target.dim() == 1:
            # Hard labels
            smooth_val = self.smoothing / (self.num_classes - 1)
            soft = torch.full_like(log_probs, smooth_val)
            soft.scatter_(1, target.unsqueeze(1), self.confidence)
        else:
            # Already soft (from Mixup/CutMix) — apply mild smoothing on top
            soft = (1 - self.smoothing) * target + \
                   self.smoothing / self.num_classes

        return -(soft * log_probs).sum(dim=-1).mean()


# ─── Warmup + Cosine scheduler ────────────────────────────────────────────────

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear warmup for warmup_epochs, then cosine decay to min_lr.

    Usage:
        scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=5,
                                          total_epochs=300, min_lr=1e-6)
        # After each epoch:
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs: int   = 5,
        total_epochs:  int   = 300,
        min_lr:        float = 1e-6,
        last_epoch:    int   = -1,
    ):
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

        return [
            self.min_lr + (base_lr - self.min_lr) * scale
            for base_lr in self.base_lrs
        ]