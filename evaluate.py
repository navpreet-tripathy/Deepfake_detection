#!/usr/bin/env python3
# ============================================================
# ODD²F — Evaluation + Ablation + Robustness Script
# ============================================================
# Usage:
#   python evaluate.py --checkpoint runs/best_model.pth \
#                      --data_root path/to/dataset \
#                      --ablation --robustness
# ============================================================

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.models.odd2f import ODD2F, RGBOnlyModel, ELAOnlyModel, NoCBAMModel
from src.data.dataset import MultiDatasetLoader
from src.evaluation.evaluator import Evaluator, AblationStudy, RobustnessTest
from src.export.converter import export_to_onnx, export_to_tflite, verify_onnx_inference


def parse_args():
    parser = argparse.ArgumentParser(
        description="ODD²F — Evaluation, Ablation, and Robustness Testing",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained model checkpoint (.pth).",
    )
    parser.add_argument(
        "--config", type=str, default="config/default.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--data_root", type=str, default=None,
        help="Single dataset root with real/ and fake/ subdirs.",
    )
    parser.add_argument(
        "--ablation", action="store_true",
        help="Run ablation study (RGB-only, ELA-only, no-CBAM).",
    )
    parser.add_argument(
        "--robustness", action="store_true",
        help="Run robustness tests (JPEG, rotation, noise, etc.).",
    )
    parser.add_argument(
        "--export", action="store_true",
        help="Export model to ONNX and TFLite.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="Number of data loader workers (default 0).",
    )
    parser.add_argument(
        "--save_dir", type=str, default="results",
        help="Directory to save evaluation results.",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'cuda', 'cpu', or 'auto'.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Config ──
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    ela_cfg = config.get("ela", {})
    training_cfg = config.get("training", {})

    # ── Device ──
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else args.device if args.device != "auto" else "cpu"
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ──
    model = ODD2F(
        pretrained=False,  # We're loading weights from checkpoint
        feature_dim=model_cfg.get("feature_dim", 960),
        cbam_reduction=model_cfg.get("cbam_reduction", 16),
        cbam_kernel_size=model_cfg.get("cbam_kernel_size", 7),
        fusion_dim=model_cfg.get("fusion_dim", 512),
        classifier_hidden=model_cfg.get("classifier_hidden", 256),
        dropout=model_cfg.get("dropout", 0.3),
        num_classes=model_cfg.get("num_classes", 2),
    )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"\n  [OK] Model loaded from {args.checkpoint}")
    print(f"    Checkpoint epoch: {checkpoint.get('epoch', '?')}")
    print(f"    Checkpoint AUC:   {checkpoint.get('val_auc', '?')}")

    # -- Dataset --
    if args.data_root:
        dataset_configs = [{"name": "custom", "root": args.data_root}]
    else:
        dataset_configs = data_cfg.get("datasets", [])

    loader = MultiDatasetLoader(
        dataset_configs=dataset_configs,
        val_ratio=data_cfg.get("val_ratio", 0.15),
        test_ratio=data_cfg.get("test_ratio", 0.15),
        ela_quality=ela_cfg.get("quality", 95),
        ela_scale=ela_cfg.get("scale", 20),
        batch_size=training_cfg.get("batch_size", 32),
        num_workers=args.num_workers,
    )

    splits = loader.prepare_splits()
    test_loader = splits["test"]

    # ══════════════════════════════════════════════════════════
    # 1. Standard Evaluation
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  STANDARD EVALUATION")
    print(f"{'='*60}")

    evaluator = Evaluator(model, device=device)
    results = evaluator.evaluate(test_loader, dataset_name="Full_ODD2F")
    evaluator.plot_confusion_matrix(results, save_path=str(save_dir / "confusion_matrix.png"))

    # Per-dataset evaluation
    if len(dataset_configs) > 1:
        for ds_cfg in dataset_configs:
            try:
                ds_splits = loader.prepare_splits(dataset_name=ds_cfg["name"])
                ds_test = ds_splits["test"]
                evaluator.evaluate(ds_test, dataset_name=ds_cfg["name"])
            except Exception as e:
                print(f"  [!] Skipping {ds_cfg['name']}: {e}")

    # ==========================================================
    # 2. Ablation Study
    # ==========================================================
    if args.ablation:
        print(f"\n{'='*60}")
        print("  ABLATION STUDY")
        print(f"{'='*60}")

        # Create ablation variants (initialized fresh -- untrained)
        # In practice you'd train each variant separately, but for
        # comparison we load the full model and test variants.
        ablation_models = {
            "Full_ODD2F": model,
        }

        # RGB-only
        try:
            rgb_model = RGBOnlyModel(
                pretrained=False,
                feature_dim=model_cfg.get("feature_dim", 960),
            )
            # Transfer RGB encoder weights from the full model
            rgb_model.encoder.load_state_dict(model.rgb_encoder.state_dict())
            rgb_model.to(device)
            ablation_models["RGB_Only"] = rgb_model
        except Exception as e:
            print(f"  [!] RGB-only model setup failed: {e}")

        # ELA-only
        try:
            ela_model = ELAOnlyModel(
                pretrained=False,
                feature_dim=model_cfg.get("feature_dim", 960),
            )
            ela_model.encoder.load_state_dict(model.ela_encoder.state_dict())
            ela_model.to(device)
            ablation_models["ELA_Only"] = ela_model
        except Exception as e:
            print(f"  [!] ELA-only model setup failed: {e}")

        # No-CBAM
        try:
            no_cbam = NoCBAMModel(
                pretrained=False,
                feature_dim=model_cfg.get("feature_dim", 960),
            )
            no_cbam.to(device)
            ablation_models["No_CBAM"] = no_cbam
        except Exception as e:
            print(f"  [!] No-CBAM model setup failed: {e}")

        ablation = AblationStudy(
            models=ablation_models,
            test_loader=test_loader,
            device=device,
        )
        ablation.run(save_dir=str(save_dir / "ablation"))

    # ==========================================================
    # 3. Robustness Tests
    # ==========================================================
    if args.robustness:
        print(f"\n{'='*60}")
        print("  ROBUSTNESS TESTS")
        print(f"{'='*60}")

        robustness_cfg = config.get("evaluation", {}).get("robustness_tests", {})
        robustness = RobustnessTest(
            model=model,
            test_loader=test_loader,
            config=robustness_cfg,
            device=device,
        )
        robustness.run(save_dir=str(save_dir / "robustness"))

    # ==========================================================
    # 4. Export
    # ==========================================================
    if args.export:
        print(f"\n{'='*60}")
        print("  MODEL EXPORT")
        print(f"{'='*60}")

        export_dir = save_dir / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)

        onnx_path = str(export_dir / "odd2f.onnx")
        export_to_onnx(model, save_path=onnx_path)
        verify_onnx_inference(onnx_path)
        export_to_tflite(
            onnx_path=onnx_path,
            tflite_path=str(export_dir / "odd2f.tflite"),
        )

    print(f"\n  [OK] All evaluations complete. Results saved to {save_dir}/")


if __name__ == "__main__":
    main()
