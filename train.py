#!/usr/bin/env python3
# ============================================================
# ODD²F — Training Script
# ============================================================
# Usage:
#   python train.py --config config/default.yaml
#   python train.py --data_root path/to/dataset --epochs 50
# ============================================================

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.odd2f import ODD2F
from src.data.dataset import MultiDatasetLoader
from src.training.trainer import Trainer


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ODD²F — Optimal Dual-Stream Deepfake Detection Training",
    )
    parser.add_argument(
        "--config", type=str, default="config/default.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--data_root", type=str, default=None,
        help="Override: single dataset root with real/ and fake/ subdirs.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override: number of training epochs.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Override: batch size.",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override: learning rate.",
    )
    parser.add_argument(
        "--save_dir", type=str, default=None,
        help="Override: checkpoint save directory.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'cuda', 'cpu', or 'auto'.",
    )
    parser.add_argument(
        "--no_pretrained", action="store_true",
        help="Disable ImageNet pretrained weights.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=None,
        help="Override: number of data loader workers.",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load config ──
    config = load_config(args.config)

    # ── Apply CLI overrides ──
    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})

    if args.epochs is not None:
        training_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        training_cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        training_cfg["learning_rate"] = args.lr
    if args.num_workers is not None:
        training_cfg["num_workers"] = args.num_workers
    if args.no_pretrained:
        model_cfg["pretrained"] = False

    save_dir = args.save_dir or config.get("logging", {}).get("save_dir", "runs")

    # ── Seed ──
    seed = training_cfg.get("seed", 42)
    set_seed(seed)

    # ── Device ──
    if args.device:
        device = torch.device(args.device if args.device != "auto" else
                              ("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  ODD2F Training")
    print(f"{'='*60}")
    print(f"  Device:     {device}")
    print(f"  Epochs:     {training_cfg.get('epochs', 50)}")
    print(f"  Batch size: {training_cfg.get('batch_size', 32)}")
    print(f"  LR:         {training_cfg.get('learning_rate', 1e-3)}")
    print(f"  Save dir:   {save_dir}")
    print(f"{'='*60}\n")

    # ── Dataset ──
    if args.data_root:
        dataset_configs = [{"name": "custom", "root": args.data_root}]
    else:
        dataset_configs = data_cfg.get("datasets", [])

    ela_cfg = config.get("ela", {})

    loader = MultiDatasetLoader(
        dataset_configs=dataset_configs,
        val_ratio=data_cfg.get("val_ratio", 0.15),
        test_ratio=data_cfg.get("test_ratio", 0.15),
        ela_quality=ela_cfg.get("quality", 95),
        ela_scale=ela_cfg.get("scale", 20),
        image_size=224,
        batch_size=training_cfg.get("batch_size", 32),
        num_workers=training_cfg.get("num_workers", 4),
        seed=seed,
    )

    splits = loader.prepare_splits()
    train_loader = splits["train"]
    val_loader = splits["val"]

    # ── Model ──
    model = ODD2F(
        pretrained=model_cfg.get("pretrained", True),
        feature_dim=model_cfg.get("feature_dim", 960),
        cbam_reduction=model_cfg.get("cbam_reduction", 16),
        cbam_kernel_size=model_cfg.get("cbam_kernel_size", 7),
        fusion_dim=model_cfg.get("fusion_dim", 512),
        classifier_hidden=model_cfg.get("classifier_hidden", 256),
        dropout=model_cfg.get("dropout", 0.3),
        num_classes=model_cfg.get("num_classes", 2),
    )

    param_info = model.count_parameters()
    print(f"  Model parameters: {param_info['total']:,} "
          f"({param_info['total_millions']}M)")

    # ── Resume from checkpoint ──
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Resumed from checkpoint: {args.resume}")

    # ── Merge augmentation + cutmix into training config ──
    aug_cfg = config.get("augmentation", {})
    cutmix_cfg = aug_cfg.get("cutmix", {})
    training_cfg["cutmix"] = cutmix_cfg

    # ── Train ──
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_cfg,
        device=device,
        save_dir=save_dir,
    )

    history = trainer.train()

    # ── Save final results ──
    import json
    results_path = Path(save_dir) / "training_history.json"
    with open(results_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n  Training history saved to {results_path}")

    # ── Plot training curves ──
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Loss
        axes[0].plot(history["train_loss"], label="Train", linewidth=2)
        axes[0].plot(history["val_loss"], label="Val", linewidth=2)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training & Validation Loss")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Accuracy
        axes[1].plot(history["train_acc"], label="Train", linewidth=2)
        axes[1].plot(history["val_acc"], label="Val", linewidth=2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Training & Validation Accuracy")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        # AUC
        axes[2].plot(history["val_auc"], label="Val AUC", linewidth=2, color="green")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("AUC")
        axes[2].set_title("Validation AUC")
        axes[2].legend()
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plot_path = Path(save_dir) / "training_curves.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Training curves saved to {plot_path}")
    except Exception as e:
        print(f"  [!] Could not plot training curves: {e}")

    print(f"\n  [OK] Training complete! Best model at {save_dir}/best_model.pth")


if __name__ == "__main__":
    main()
