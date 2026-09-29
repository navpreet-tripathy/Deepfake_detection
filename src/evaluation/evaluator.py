# ============================================================
# ODD²F — Evaluation & Ablation Studies
# ============================================================
# Step 9: Per-dataset evaluation, ablation studies, and
# robustness tests under various perturbations.
# ============================================================

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, precision_recall_fscore_support,
)
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from ..preprocessing.ela import compute_ela


class Evaluator:
    """Evaluation engine for ODD²F models.

    Runs inference, computes metrics, generates confusion matrices,
    and supports robustness testing under perturbations.

    Args:
        model: Trained model.
        device: Torch device.
        use_amp: Enable mixed precision for evaluation.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        use_amp: bool = True,
    ):
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()
        self.use_amp = use_amp and self.device.type == "cuda"

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        dataset_name: str = "test",
    ) -> dict:
        """Run evaluation on a dataset.

        Args:
            dataloader: Test DataLoader.
            dataset_name: Name for reporting.

        Returns:
            dict with accuracy, AUC, classification report, etc.
        """
        all_preds, all_labels, all_probs = [], [], []

        for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}", leave=False):
            rgb = batch["rgb"].to(self.device, non_blocking=True)
            ela = batch["ela"].to(self.device, non_blocking=True)
            labels = batch["label"]

            with autocast(enabled=self.use_amp):
                logits = self.model(rgb, ela)
                probs = F.softmax(logits, dim=1)

            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.0

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="binary",
        )
        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(
            all_labels, all_preds,
            target_names=["Real", "Fake"],
            output_dict=True,
        )

        results = {
            "dataset": dataset_name,
            "accuracy": accuracy,
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "num_samples": len(all_labels),
        }

        print(f"\n{'='*50}")
        print(f"  {dataset_name} Results")
        print(f"{'='*50}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  AUC:      {auc:.4f}")
        print(f"  F1:       {f1:.4f}")
        print(f"  Precision:{precision:.4f}")
        print(f"  Recall:   {recall:.4f}")
        print(f"{'='*50}\n")

        return results

    def plot_confusion_matrix(
        self,
        results: dict,
        save_path: Optional[str] = None,
    ):
        """Plot and optionally save a confusion matrix heatmap."""
        cm = np.array(results["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Real", "Fake"],
            yticklabels=["Real", "Fake"],
            ax=ax,
        )
        ax.set_xlabel("Predicted", fontsize=13)
        ax.set_ylabel("Actual", fontsize=13)
        ax.set_title(f"Confusion Matrix — {results['dataset']}", fontsize=15)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Confusion matrix saved to {save_path}")

        plt.close(fig)
        return fig


class AblationStudy:
    """Run ablation experiments comparing model variants.

    Compares:
        - Full ODD²F (RGB + ELA + CBAM)
        - RGB-only (no ELA stream)
        - ELA-only (no RGB stream)
        - No CBAM (dual-stream without attention)
        - No augmentation (full model, no data augmentation at test time)

    Args:
        models: dict mapping variant name → model instance.
        test_loader: Test DataLoader.
        device: Torch device.
    """

    def __init__(
        self,
        models: dict[str, nn.Module],
        test_loader: DataLoader,
        device: torch.device = None,
    ):
        self.models = models
        self.test_loader = test_loader
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def run(self, save_dir: Optional[str] = None) -> dict:
        """Execute all ablation experiments.

        Returns:
            dict mapping variant name → evaluation results.
        """
        results = {}

        print(f"\n{'='*60}")
        print("  ABLATION STUDY")
        print(f"{'='*60}")

        for name, model in self.models.items():
            print(f"\n  ▶ Evaluating: {name}")
            evaluator = Evaluator(model, device=self.device)
            result = evaluator.evaluate(self.test_loader, dataset_name=name)
            results[name] = result

            if save_dir:
                save_path = Path(save_dir)
                save_path.mkdir(parents=True, exist_ok=True)
                evaluator.plot_confusion_matrix(
                    result,
                    save_path=str(save_path / f"cm_{name}.png"),
                )

        # Summary table
        print(f"\n{'='*70}")
        print(f"  {'Variant':<25} {'Accuracy':>10} {'AUC':>10} {'F1':>10}")
        print(f"  {'-'*55}")
        for name, r in results.items():
            print(f"  {name:<25} {r['accuracy']:>10.4f} {r['auc']:>10.4f} {r['f1']:>10.4f}")
        print(f"{'='*70}\n")

        # Save results JSON
        if save_dir:
            with open(Path(save_dir) / "ablation_results.json", "w") as f:
                json.dump(results, f, indent=2)

        return results


class RobustnessTest:
    """Test model robustness under various perturbations (Step 9).

    Perturbations:
        - JPEG compression at various quality levels
        - Rotation at various degrees
        - Scaling at various factors
        - Gaussian noise at various standard deviations
        - Brightness shifts at various intensities

    Args:
        model: Trained model.
        test_loader: Test DataLoader.
        config: Robustness test config dict.
        device: Torch device.
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        config: dict,
        device: torch.device = None,
    ):
        self.model = model
        self.test_loader = test_loader
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()

    def run(self, save_dir: Optional[str] = None) -> dict:
        """Run all robustness tests.

        Returns:
            dict mapping perturbation_type → list of (param, AUC) tuples.
        """
        results = {}

        # Baseline
        evaluator = Evaluator(self.model, device=self.device)
        baseline = evaluator.evaluate(self.test_loader, "Baseline")
        baseline_auc = baseline["auc"]
        results["baseline_auc"] = baseline_auc

        # JPEG Compression
        jpeg_results = []
        for q in self.config.get("jpeg_qualities", [70, 50, 30]):
            auc = self._test_perturbation("jpeg", quality=q)
            degradation = baseline_auc - auc
            jpeg_results.append({"quality": q, "auc": auc, "degradation": degradation})
            print(f"  JPEG Q={q}: AUC={auc:.4f} (Δ={degradation:.4f})")
        results["jpeg_compression"] = jpeg_results

        # Rotation
        rotation_results = []
        for deg in self.config.get("rotation_degrees", [5, 10, 15, 20]):
            auc = self._test_perturbation("rotation", degrees=deg)
            degradation = baseline_auc - auc
            rotation_results.append({"degrees": deg, "auc": auc, "degradation": degradation})
            print(f"  Rotation {deg}°: AUC={auc:.4f} (Δ={degradation:.4f})")
        results["rotation"] = rotation_results

        # Scaling
        scale_results = []
        for s in self.config.get("scale_factors", [0.5, 0.75, 1.25, 1.5]):
            auc = self._test_perturbation("scale", factor=s)
            degradation = baseline_auc - auc
            scale_results.append({"factor": s, "auc": auc, "degradation": degradation})
            print(f"  Scale {s}x: AUC={auc:.4f} (Δ={degradation:.4f})")
        results["scaling"] = scale_results

        # Gaussian noise
        noise_results = []
        for std in self.config.get("gaussian_noise_std", [0.01, 0.02, 0.05]):
            auc = self._test_perturbation("noise", std=std)
            degradation = baseline_auc - auc
            noise_results.append({"std": std, "auc": auc, "degradation": degradation})
            print(f"  Noise σ={std}: AUC={auc:.4f} (Δ={degradation:.4f})")
        results["gaussian_noise"] = noise_results

        # Brightness
        brightness_results = []
        for shift in self.config.get("brightness_shifts", [-0.2, -0.1, 0.1, 0.2]):
            auc = self._test_perturbation("brightness", shift=shift)
            degradation = baseline_auc - auc
            brightness_results.append({"shift": shift, "auc": auc, "degradation": degradation})
            print(f"  Brightness Δ={shift}: AUC={auc:.4f} (Δ={degradation:.4f})")
        results["brightness"] = brightness_results

        # Save
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            with open(save_path / "robustness_results.json", "w") as f:
                json.dump(results, f, indent=2)
            self._plot_robustness(results, save_path)

        return results

    @torch.no_grad()
    def _test_perturbation(self, ptype: str, **kwargs) -> float:
        """Evaluate model under a specific perturbation.

        Applies the perturbation to the RGB tensor, re-computes ELA from
        the perturbed image, and runs inference.
        """
        all_probs, all_labels = [], []

        for batch in self.test_loader:
            rgb = batch["rgb"]  # (B, 3, H, W) — still normalized
            labels = batch["label"]

            # Apply perturbation to each image in the batch
            perturbed_rgb_list = []
            perturbed_ela_list = []

            for i in range(rgb.size(0)):
                # Denormalize
                img = self._denormalize(rgb[i])

                # Apply perturbation
                img_perturbed = self._apply_perturbation(img, ptype, **kwargs)

                # Recompute ELA on perturbed image
                ela_np = compute_ela(img_perturbed, quality=95, scale=20)

                # Normalize back
                from ..data.dataset import get_transform
                transform = get_transform(train=False)
                rgb_t = transform(Image.fromarray(img_perturbed))
                ela_t = transform(Image.fromarray(ela_np))

                perturbed_rgb_list.append(rgb_t)
                perturbed_ela_list.append(ela_t)

            rgb_batch = torch.stack(perturbed_rgb_list).to(self.device)
            ela_batch = torch.stack(perturbed_ela_list).to(self.device)

            with autocast(enabled=self.device.type == "cuda"):
                logits = self.model(rgb_batch, ela_batch)
                probs = F.softmax(logits, dim=1)

            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.numpy())

        try:
            return roc_auc_score(all_labels, all_probs)
        except ValueError:
            return 0.0

    @staticmethod
    def _denormalize(tensor: torch.Tensor) -> np.ndarray:
        """Reverse ImageNet normalization and convert to uint8 numpy."""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = tensor.cpu() * std + mean
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()
        return (img * 255).astype(np.uint8)

    @staticmethod
    def _apply_perturbation(
        image: np.ndarray,
        ptype: str,
        **kwargs,
    ) -> np.ndarray:
        """Apply a single perturbation to an image."""
        h, w = image.shape[:2]

        if ptype == "jpeg":
            # Re-encode at low quality
            quality = kwargs.get("quality", 50)
            pil_img = Image.fromarray(image)
            import io
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            return np.array(Image.open(buf).convert("RGB"))

        elif ptype == "rotation":
            degrees = kwargs.get("degrees", 10)
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, degrees, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h),
                                     borderMode=cv2.BORDER_REFLECT)
            return rotated

        elif ptype == "scale":
            factor = kwargs.get("factor", 0.75)
            new_w, new_h = int(w * factor), int(h * factor)
            scaled = cv2.resize(image, (new_w, new_h))
            # Resize back to original
            return cv2.resize(scaled, (w, h))

        elif ptype == "noise":
            std = kwargs.get("std", 0.02)
            noise = np.random.normal(0, std * 255, image.shape).astype(np.float32)
            noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
            return noisy.astype(np.uint8)

        elif ptype == "brightness":
            shift = kwargs.get("shift", 0.1)
            adjusted = image.astype(np.float32) + shift * 255
            return np.clip(adjusted, 0, 255).astype(np.uint8)

        return image

    @staticmethod
    def _plot_robustness(results: dict, save_dir: Path):
        """Generate robustness degradation plots."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Robustness Test Results — AUC Degradation", fontsize=16)

        baseline = results.get("baseline_auc", 1.0)

        # JPEG
        ax = axes[0, 0]
        if "jpeg_compression" in results:
            data = results["jpeg_compression"]
            x = [d["quality"] for d in data]
            y = [d["auc"] for d in data]
            ax.plot(x, y, "o-", color="#e74c3c", linewidth=2)
            ax.axhline(y=baseline, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel("JPEG Quality")
            ax.set_ylabel("AUC")
            ax.set_title("JPEG Compression")

        # Rotation
        ax = axes[0, 1]
        if "rotation" in results:
            data = results["rotation"]
            x = [d["degrees"] for d in data]
            y = [d["auc"] for d in data]
            ax.plot(x, y, "o-", color="#3498db", linewidth=2)
            ax.axhline(y=baseline, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel("Rotation (degrees)")
            ax.set_ylabel("AUC")
            ax.set_title("Rotation")

        # Scaling
        ax = axes[0, 2]
        if "scaling" in results:
            data = results["scaling"]
            x = [d["factor"] for d in data]
            y = [d["auc"] for d in data]
            ax.plot(x, y, "o-", color="#2ecc71", linewidth=2)
            ax.axhline(y=baseline, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel("Scale Factor")
            ax.set_ylabel("AUC")
            ax.set_title("Scaling")

        # Noise
        ax = axes[1, 0]
        if "gaussian_noise" in results:
            data = results["gaussian_noise"]
            x = [d["std"] for d in data]
            y = [d["auc"] for d in data]
            ax.plot(x, y, "o-", color="#9b59b6", linewidth=2)
            ax.axhline(y=baseline, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel("Noise σ")
            ax.set_ylabel("AUC")
            ax.set_title("Gaussian Noise")

        # Brightness
        ax = axes[1, 1]
        if "brightness" in results:
            data = results["brightness"]
            x = [d["shift"] for d in data]
            y = [d["auc"] for d in data]
            ax.plot(x, y, "o-", color="#f39c12", linewidth=2)
            ax.axhline(y=baseline, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel("Brightness Shift")
            ax.set_ylabel("AUC")
            ax.set_title("Brightness")

        # Summary bar chart
        ax = axes[1, 2]
        max_degradations = {}
        for key in ["jpeg_compression", "rotation", "scaling",
                     "gaussian_noise", "brightness"]:
            if key in results:
                max_deg = max(d["degradation"] for d in results[key])
                max_degradations[key] = max_deg

        if max_degradations:
            names = list(max_degradations.keys())
            values = list(max_degradations.values())
            colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]
            bars = ax.bar(range(len(names)), values, color=colors[:len(names)])
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(
                [n.replace("_", "\n") for n in names],
                fontsize=9,
            )
            ax.set_ylabel("Max AUC Degradation")
            ax.set_title("Worst-Case Degradation")
            ax.axhline(y=0.04, color="red", linestyle="--", alpha=0.5,
                       label="4% threshold")
            ax.legend()

        plt.tight_layout()
        fig.savefig(save_dir / "robustness_plots.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Robustness plots saved to {save_dir / 'robustness_plots.png'}")
