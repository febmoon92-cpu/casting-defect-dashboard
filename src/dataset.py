"""Dataset / Transforms / DataLoaders for casting binary classification.

We treat 'def_front' as positive class (label = 1, "defective").

The Kaggle archive provides:
  data/raw/casting_data/casting_data/train/{def_front, ok_front}
  data/raw/casting_data/casting_data/test/{def_front, ok_front}

We stratified-split the train folder into train/val and keep test as-is.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from . import config
from .utils import find_data_root


CLASS_TO_IDX = {"ok_front": 0, "def_front": 1}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


@dataclass
class Sample:
    path: Path
    label: int


def _list_samples(split_dir: Path) -> list[Sample]:
    out: list[Sample] = []
    for cls_name, idx in CLASS_TO_IDX.items():
        cls_dir = split_dir / cls_name
        if not cls_dir.exists():
            continue
        for img_path in sorted(cls_dir.glob("*")):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                out.append(Sample(path=img_path, label=idx))
    return out


def gather_splits(raw_dir: Path = config.RAW_DIR, val_split: float = config.VAL_SPLIT, seed: int = config.SEED):
    """Return (train_samples, val_samples, test_samples) via stratified split of train/."""
    data_root = find_data_root(raw_dir)
    train_all = _list_samples(data_root / "train")
    test_samples = _list_samples(data_root / "test")

    rng = np.random.default_rng(seed)
    by_class: dict[int, list[Sample]] = {0: [], 1: []}
    for s in train_all:
        by_class[s.label].append(s)

    train_samples: list[Sample] = []
    val_samples: list[Sample] = []
    for label, samples in by_class.items():
        idx = np.arange(len(samples))
        rng.shuffle(idx)
        n_val = int(round(len(samples) * val_split))
        val_idx = set(idx[:n_val].tolist())
        for i, s in enumerate(samples):
            (val_samples if i in val_idx else train_samples).append(s)

    return train_samples, val_samples, test_samples


class CastingDataset(Dataset):
    def __init__(self, samples: list[Sample], transform: Callable):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        with Image.open(s.path) as img:
            img = img.convert("RGB")
        x = self.transform(img)
        return x, s.label, str(s.path)


def build_transforms(img_size: int = config.IMG_SIZE):
    """Conservative augmentation that does NOT distort defect signatures."""
    train_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


def make_weighted_sampler(samples: list[Sample]) -> WeightedRandomSampler:
    labels = np.array([s.label for s in samples])
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    class_weight = 1.0 / np.clip(counts, 1, None)
    weights = class_weight[labels]
    return WeightedRandomSampler(weights=torch.from_numpy(weights).double(), num_samples=len(samples), replacement=True)


def build_loaders(batch_size: int = config.RESNET_BATCH, num_workers: int = config.NUM_WORKERS):
    train_s, val_s, test_s = gather_splits()
    train_tf, eval_tf = build_transforms()
    train_ds = CastingDataset(train_s, train_tf)
    val_ds = CastingDataset(val_s, eval_tf)
    test_ds = CastingDataset(test_s, eval_tf)

    sampler = make_weighted_sampler(train_s)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "train_samples": train_s,
        "val_samples": val_s,
        "test_samples": test_s,
    }
