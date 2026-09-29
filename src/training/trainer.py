# ============================================================
# ODD²F — Training Engine
# ============================================================
# Step 8: Full training loop with AMP, CutMix, early stopping,
# learning rate scheduling, and TensorBoard logging.
# ============================================================

from __future__ import annotations

import os
import time
import copy
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

from ..data.dataset import cutmix_data


class EarlyStopping:
    """Early stopping to terminate training when validation metric plateaus.

    Args:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as improvement.
        mode: 'min' for loss, 'max' for accuracy/AUC.
    """

    def __init__(self, patience: int = 7, min_delta: float = 0.001, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class Trainer:
    """Training engine for ODD²F and ablation variants.

    Handles the full training loop including mixed precision,
    CutMix augmentation, learning rate scheduling, early stopping,
    checkpointing, and TensorBoard logging.

    Args:
        model: The model to train (ODD2F or ablation variant).
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        config: Training configuration dict.
        device: Torch device ('cuda', 'cpu', etc.).
        save_dir: Directory for checkpoints and logs.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: dict,
        device: torch.device = None,
        save_dir: str = "runs",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get("learning_rate", 1e-3),
            weight_decay=config.get("weight_decay", 1e-5),
        )

        # Scheduler: linear warmup → cosine annealing
        total_epochs = config.get("epochs", 50)
        warmup_epochs = min(config.get("warmup_epochs", 3), max(0, total_epochs - 1))
        if warmup_epochs > 0 and total_epochs > warmup_epochs:
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, total_epochs - warmup_epochs),
                eta_min=1e-6,
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, total_epochs),
                eta_min=1e-6,
            )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Mixed precision
        self.use_amp = config.get("mixed_precision", True) and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        # CutMix
        cutmix_cfg = config.get("cutmix", {})
        self.use_cutmix = cutmix_cfg.get("enabled", True)
        self.cutmix_alpha = cutmix_cfg.get("alpha", 1.0)
        self.cutmix_p = cutmix_cfg.get("p", 0.5)

        # Early stopping
        es_cfg = config.get("early_stopping", {})
        self.early_stopping = EarlyStopping(
            patience=es_cfg.get("patience", 7),
            min_delta=es_cfg.get("min_delta", 0.001),
        ) if es_cfg.get("enabled", True) else None

        # Gradient clipping
        self.grad_clip = config.get("gradient_clip_max_norm", 1.0)

        # Logging
        self.writer = SummaryWriter(log_dir=str(self.save_dir / "tensorboard"))
        self.log_interval = config.get("log_interval", 50)

        # Best model tracking
        self.best_val_auc = 0.0
        self.best_model_state = None

    def train(self) -> dict:
        """Run the full training loop.

        Returns:
            dict with training history (losses, metrics per epoch).
        """
        epochs = self.config.get("epochs", 50)
        history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [], "val_auc": [],
            "lr": [],
        }

        print(f"\n{'='*60}")
        print(f"  ODD2F Training - {epochs} epochs")
        print(f"  Device: {self.device}")
        print(f"  AMP: {self.use_amp} | CutMix: {self.use_cutmix}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"{'='*60}\n")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # -- Train --
            train_loss, train_acc = self._train_epoch(epoch)

            # -- Validate --
            val_loss, val_acc, val_auc = self._validate_epoch()

            # -- LR step --
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            # -- Log --
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f} | "
                f"LR: {current_lr:.2e} | {elapsed:.1f}s"
            )

            self.writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
            self.writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)
            self.writer.add_scalar("AUC/val", val_auc, epoch)
            self.writer.add_scalar("LR", current_lr, epoch)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_auc"].append(val_auc)
            history["lr"].append(current_lr)

            # -- Best model checkpoint --
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self._save_checkpoint(epoch, val_auc, is_best=True)
                print(f"  [*] New best model saved (AUC: {val_auc:.4f})")

            # -- Early stopping --
            if self.early_stopping and self.early_stopping(val_auc):
                print(f"\n  [STOP] Early stopping triggered at epoch {epoch}")
                break

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\n  [OK] Restored best model (AUC: {self.best_val_auc:.4f})")

        self.writer.close()
        return history

    def _train_epoch(self, epoch: int) -> tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        pbar = tqdm(
            self.train_loader,
            desc=f"  Train Epoch {epoch}",
            leave=False,
            ncols=100,
        )

        for step, batch in enumerate(pbar):
            rgb = batch["rgb"].to(self.device, non_blocking=True)
            ela = batch["ela"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            # CutMix (Step 3)
            use_cutmix_this_step = (
                self.use_cutmix and np.random.rand() < self.cutmix_p
            )

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                if use_cutmix_this_step:
                    rgb, ela, targets_a, targets_b, lam = cutmix_data(
                        rgb, ela, labels, alpha=self.cutmix_alpha,
                    )
                    logits = self.model(rgb, ela)
                    loss = lam * self.criterion(logits, targets_a) + \
                           (1 - lam) * self.criterion(logits, targets_b)
                else:
                    logits = self.model(rgb, ela)
                    loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.grad_clip:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Metrics
            running_loss += loss.item() * rgb.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def _validate_epoch(self) -> tuple[float, float, float]:
        """Validate for one epoch."""
        self.model.eval()
        running_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        for batch in self.val_loader:
            rgb = batch["rgb"].to(self.device, non_blocking=True)
            ela = batch["ela"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                logits = self.model(rgb, ela)
                loss = self.criterion(logits, labels)

            running_loss += loss.item() * rgb.size(0)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

        epoch_loss = running_loss / len(self.val_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)

        try:
            epoch_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            epoch_auc = 0.0

        return epoch_loss, epoch_acc, epoch_auc

    def _save_checkpoint(self, epoch: int, val_auc: float, is_best: bool = False):
        """Save model checkpoint."""
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_auc": val_auc,
            "config": self.config,
        }
        filename = "best_model.pth" if is_best else f"checkpoint_epoch_{epoch}.pth"
        torch.save(ckpt, self.save_dir / filename)
