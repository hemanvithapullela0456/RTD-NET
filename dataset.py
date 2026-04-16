"""
dataset.py — Data loading utilities for AID and NWPU-RESISC45
Supports both 80/20 and 50/50 train/val splits.
"""

import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def get_transforms(image_size=224, is_train=True):
    """
    Returns augmentation pipeline.
    Training: RandomCrop, HorizontalFlip, VerticalFlip, Rotation, ColorJitter, Normalize
    Validation: CenterCrop, Normalize
    """
    mean = [0.3680, 0.3810, 0.3436]   # AID approximate stats
    std  = [0.2034, 0.1854, 0.1876]

    if is_train:
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.15)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.05)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ---------------------------------------------------------------------------
# Split helper: reproducible 80/20 or 50/50 per-class split
# ---------------------------------------------------------------------------
def split_dataset(root: str, train_ratio: float = 0.8,
                  image_size: int = 224, seed: int = 42):
    """
    Loads an ImageFolder dataset and returns (train_loader, val_loader, class_names).
    Performs a stratified per-class split so every class is represented in both sets.

    Args:
        root       : Path to the dataset folder (contains one sub-folder per class).
        train_ratio: Fraction of data for training (0.8 → 80/20, 0.5 → 50/50).
        image_size : Spatial size for resizing.
        seed       : Random seed for reproducibility.
    """
    # Load with dummy transform to get indices
    full_ds = datasets.ImageFolder(root)
    class_names = full_ds.classes
    num_classes = len(class_names)
    print(f"Dataset root  : {root}")
    print(f"Classes found : {num_classes}  →  {class_names[:5]} ...")
    print(f"Total images  : {len(full_ds)}")

    # Build per-class index lists
    class_indices = {c: [] for c in range(num_classes)}
    for idx, (_, label) in enumerate(full_ds.samples):
        class_indices[label].append(idx)

    # Stratified split
    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for label, indices in class_indices.items():
        indices_shuffled = indices[:]
        rng.shuffle(indices_shuffled)
        split = int(len(indices_shuffled) * train_ratio)
        train_idx.extend(indices_shuffled[:split])
        val_idx.extend(indices_shuffled[split:])

    print(f"Train samples : {len(train_idx)}")
    print(f"Val samples   : {len(val_idx)}")

    # Re-load with proper transforms
    train_ds_full = datasets.ImageFolder(root, transform=get_transforms(image_size, True))
    val_ds_full   = datasets.ImageFolder(root, transform=get_transforms(image_size, False))

    train_ds = Subset(train_ds_full, train_idx)
    val_ds   = Subset(val_ds_full,   val_idx)

    return train_ds, val_ds, class_names


def get_dataloaders(root: str, train_ratio: float = 0.8,
                    image_size: int = 224, batch_size: int = 32,
                    num_workers: int = 4, seed: int = 42):
    """
    Returns (train_loader, val_loader, class_names).
    """
    train_ds, val_ds, class_names = split_dataset(
        root, train_ratio, image_size, seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, class_names


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/aid"
    if not os.path.isdir(root):
        print(f"[WARN] {root} not found — skipping dataloader test.")
    else:
        tl, vl, names = get_dataloaders(root, batch_size=4, num_workers=0)
        imgs, labels = next(iter(tl))
        print(f"Batch images: {imgs.shape}, labels: {labels.shape}")
        print(f"Class names:  {names}")