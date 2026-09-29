# ============================================================
# ODD²F — Dataset & Data Loading
# ============================================================
# Steps 1–3: Loads images, generates ELA, applies augmentation,
# returns paired (RGB, ELA) tensors for the dual-stream model.
# ============================================================

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split

from ..preprocessing.ela import compute_ela


# ─── Standard transforms (Step 1) ────────────────────────────

def get_transform(train: bool = True, image_size: int = 224) -> transforms.Compose:
    """Get torchvision transform pipeline.

    Training includes augmentation (Step 3); validation/test does not.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2,
                saturation=0.2, hue=0.1,
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))],
                p=0.1,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


# ─── CutMix utility (Step 3) ─────────────────────────────────

def cutmix_data(
    rgb: torch.Tensor,
    ela: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply CutMix augmentation to a batch.

    Args:
        rgb: RGB batch (B, 3, H, W).
        ela: ELA batch (B, 3, H, W).
        targets: Labels (B,).
        alpha: Beta distribution parameter.

    Returns:
        (mixed_rgb, mixed_ela, targets_a, targets_b, lam)
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = rgb.size(0)
    index = torch.randperm(batch_size, device=rgb.device)

    # Random bounding box
    _, _, h, w = rgb.shape
    cut_ratio = np.sqrt(1 - lam)
    cut_w = int(w * cut_ratio)
    cut_h = int(h * cut_ratio)
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    x1 = max(0, cx - cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    x2 = min(w, cx + cut_w // 2)
    y2 = min(h, cy + cut_h // 2)

    # Apply cutmix to both streams identically
    mixed_rgb = rgb.clone()
    mixed_ela = ela.clone()
    mixed_rgb[:, :, y1:y2, x1:x2] = rgb[index, :, y1:y2, x1:x2]
    mixed_ela[:, :, y1:y2, x1:x2] = ela[index, :, y1:y2, x1:x2]

    # Adjust lambda for the actual area ratio
    lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))

    return mixed_rgb, mixed_ela, targets, targets[index], lam


# ─── Dual-Stream Dataset ─────────────────────────────────────

class DeepfakeDataset(Dataset):
    """Dataset that returns paired (RGB, ELA) images for ODD²F.

    Expects a root directory with two subfolders:
        root/
        ├── real/    (label 0)
        └── fake/    (label 1)

    Args:
        root: Path to dataset root.
        transform: Torchvision transform for both RGB and ELA.
        ela_quality: JPEG quality for ELA computation.
        ela_scale: ELA amplification factor.
        image_size: Target image size.
    """

    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        ela_quality: int = 95,
        ela_scale: int = 20,
        image_size: int = 224,
    ):
        super().__init__()
        self.root = Path(root)
        self.transform = transform
        self.ela_quality = ela_quality
        self.ela_scale = ela_scale
        self.image_size = image_size

        # Collect file paths and labels
        self.samples: list[tuple[str, int]] = []
        self._load_samples()

    def _load_samples(self):
        """Scan real/ and fake/ subdirectories for images."""
        for label_name, label_idx in [("real", 0), ("fake", 1)]:
            folder = self.root / label_name
            if not folder.exists():
                # Try uppercase or other variants
                for variant in [label_name, label_name.upper(),
                                label_name.capitalize()]:
                    alt = self.root / variant
                    if alt.exists():
                        folder = alt
                        break
                else:
                    print(f"Warning: '{folder}' not found, skipping.")
                    continue

            for file in sorted(folder.iterdir()):
                if file.suffix.lower() in self.VALID_EXTENSIONS:
                    self.samples.append((str(file), label_idx))

        if len(self.samples) == 0:
            raise ValueError(
                f"No images found in {self.root}. "
                f"Expected subdirectories: real/ and fake/"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with keys:
                'rgb': Transformed RGB tensor (3, 224, 224).
                'ela': Transformed ELA tensor (3, 224, 224).
                'label': Integer label (0=Real, 1=Fake).
                'path': Original file path string.
        """
        path, label = self.samples[idx]

        # Load and resize image
        image = Image.open(path).convert("RGB")
        image_np = np.array(image)

        # Compute ELA (Step 2)
        ela_np = compute_ela(
            image_np,
            quality=self.ela_quality,
            scale=self.ela_scale,
        )
        ela_pil = Image.fromarray(ela_np)

        # Apply transforms (Step 3 augmentation included for training)
        if self.transform:
            rgb_tensor = self.transform(image)
            ela_tensor = self.transform(ela_pil)
        else:
            # Fallback: simple resize + normalize
            fallback = get_transform(train=False, image_size=self.image_size)
            rgb_tensor = fallback(image)
            ela_tensor = fallback(ela_pil)

        return {
            "rgb": rgb_tensor,
            "ela": ela_tensor,
            "label": label,
            "path": path,
        }


# ─── Multi-Dataset Loader ────────────────────────────────────

class MultiDatasetLoader:
    """Manages multiple datasets and creates train/val/test splits.

    Args:
        dataset_configs: List of dicts with 'name' and 'root' keys.
        val_ratio: Fraction reserved for validation.
        test_ratio: Fraction reserved for testing.
        ela_quality: JPEG quality for ELA.
        ela_scale: ELA amplification.
        image_size: Target resolution.
        batch_size: Dataloader batch size.
        num_workers: Dataloader workers.
        seed: Random seed for reproducible splits.
    """

    def __init__(
        self,
        dataset_configs: list[dict],
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        ela_quality: int = 95,
        ela_scale: int = 20,
        image_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 42,
    ):
        self.configs = dataset_configs
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.ela_quality = ela_quality
        self.ela_scale = ela_scale
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def prepare_splits(
        self,
        dataset_name: Optional[str] = None,
    ) -> dict[str, DataLoader]:
        """Create train/val/test DataLoaders.

        Args:
            dataset_name: If given, use only this dataset.
                          If None, merge all datasets.

        Returns:
            dict with 'train', 'val', 'test' DataLoaders.
        """
        # Collect all samples
        all_samples = []
        configs_to_use = self.configs
        if dataset_name:
            configs_to_use = [c for c in self.configs if c["name"] == dataset_name]

        for cfg in configs_to_use:
            root = Path(cfg["root"])
            if not root.exists():
                print(f"Warning: Dataset root '{root}' not found, skipping.")
                continue

            ds = DeepfakeDataset(
                root=str(root),
                ela_quality=self.ela_quality,
                ela_scale=self.ela_scale,
                image_size=self.image_size,
            )
            all_samples.extend(ds.samples)

        if len(all_samples) == 0:
            raise ValueError("No samples found across all datasets.")

        paths, labels = zip(*all_samples)
        paths, labels = list(paths), list(labels)

        # Stratified train/val/test split
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            paths, labels,
            test_size=self.test_ratio,
            stratify=labels,
            random_state=self.seed,
        )
        val_fraction = self.val_ratio / (1 - self.test_ratio)
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_paths, train_labels,
            test_size=val_fraction,
            stratify=train_labels,
            random_state=self.seed,
        )

        # Build datasets
        train_ds = _SplitDataset(
            train_paths, train_labels,
            transform=get_transform(train=True, image_size=self.image_size),
            ela_quality=self.ela_quality,
            ela_scale=self.ela_scale,
        )
        val_ds = _SplitDataset(
            val_paths, val_labels,
            transform=get_transform(train=False, image_size=self.image_size),
            ela_quality=self.ela_quality,
            ela_scale=self.ela_scale,
        )
        test_ds = _SplitDataset(
            test_paths, test_labels,
            transform=get_transform(train=False, image_size=self.image_size),
            ela_quality=self.ela_quality,
            ela_scale=self.ela_scale,
        )

        # Weighted sampler for class imbalance
        train_class_counts = np.bincount(train_labels)
        weights = 1.0 / train_class_counts[train_labels]
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(weights),
            num_samples=len(train_ds),
            replacement=True,
        )

        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size,
            sampler=sampler, num_workers=self.num_workers,
            pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
            pin_memory=True,
        )

        print(f"Dataset splits — Train: {len(train_ds)}, "
              f"Val: {len(val_ds)}, Test: {len(test_ds)}")
        print(f"Train class distribution: {dict(zip(['Real', 'Fake'], train_class_counts))}")

        return {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }


class _SplitDataset(Dataset):
    """Internal dataset built from pre-split file lists."""

    def __init__(self, paths, labels, transform, ela_quality, ela_scale):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.ela_quality = ela_quality
        self.ela_scale = ela_scale

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]

        image = Image.open(path).convert("RGB")
        image_np = np.array(image)

        ela_np = compute_ela(image_np, quality=self.ela_quality, scale=self.ela_scale)
        ela_pil = Image.fromarray(ela_np)

        rgb_tensor = self.transform(image)
        ela_tensor = self.transform(ela_pil)

        return {
            "rgb": rgb_tensor,
            "ela": ela_tensor,
            "label": label,
            "path": path,
        }
