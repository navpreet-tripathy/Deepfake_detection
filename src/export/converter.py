# ============================================================
# ODD²F — Model Export (ONNX + TFLite)
# ============================================================
# Export trained PyTorch model to ONNX and TensorFlow Lite
# for edge deployment.
# ============================================================

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


def export_to_onnx(
    model: nn.Module,
    save_path: str = "exports/odd2f.onnx",
    input_size: tuple = (1, 3, 224, 224),
    device: str = "cpu",
    opset_version: int = 13,
) -> str:
    """Export the ODD²F model to ONNX format.

    The model takes two inputs (RGB and ELA), so we create two
    dummy tensors.

    Args:
        model: Trained PyTorch model.
        save_path: Output ONNX file path.
        input_size: Input tensor shape (B, C, H, W).
        device: Device for export.
        opset_version: ONNX opset version.

    Returns:
        Path to the saved ONNX model.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    model.eval()

    # Dummy inputs for both streams
    rgb_dummy = torch.randn(*input_size, device=device)
    ela_dummy = torch.randn(*input_size, device=device)

    torch.onnx.export(
        model,
        (rgb_dummy, ela_dummy),
        save_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["rgb_input", "ela_input"],
        output_names=["output"],
        dynamic_axes={
            "rgb_input": {0: "batch_size"},
            "ela_input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    print(f"  ✓ ONNX model saved to {save_path}")

    # Verify
    try:
        import onnx
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        print("  ✓ ONNX model verified successfully")
    except ImportError:
        print("  ⚠ onnx package not found — skipping verification")
    except Exception as e:
        print(f"  ⚠ ONNX verification warning: {e}")

    return save_path


def export_to_tflite(
    onnx_path: str = "exports/odd2f.onnx",
    tflite_path: str = "exports/odd2f.tflite",
    quantize: bool = False,
) -> str:
    """Convert ONNX model to TensorFlow Lite.

    Requires onnx2tf or tf2onnx to be installed. Falls back to
    ONNX Runtime if TFLite conversion is unavailable.

    Args:
        onnx_path: Path to the ONNX model.
        tflite_path: Output TFLite file path.
        quantize: Apply dynamic range quantization.

    Returns:
        Path to the saved TFLite model.
    """
    Path(tflite_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf

        # ONNX → TensorFlow SavedModel
        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)
        saved_model_dir = str(Path(tflite_path).parent / "tf_saved_model")
        tf_rep.export_graph(saved_model_dir)

        # SavedModel → TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model = converter.convert()

        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

        print(f"  ✓ TFLite model saved to {tflite_path}")
        return tflite_path

    except ImportError as e:
        print(f"  ⚠ TFLite conversion skipped — missing dependency: {e}")
        print("    Install with: pip install onnx-tf tensorflow")
        print("    The ONNX model can be used directly with ONNX Runtime.")
        return ""


def verify_onnx_inference(
    onnx_path: str,
    input_size: tuple = (1, 3, 224, 224),
) -> bool:
    """Verify ONNX model runs correctly with ONNX Runtime.

    Args:
        onnx_path: Path to the ONNX model.
        input_size: Input tensor shape.

    Returns:
        True if inference succeeds.
    """
    try:
        import onnxruntime as ort
        import numpy as np

        session = ort.InferenceSession(onnx_path)
        rgb_input = np.random.randn(*input_size).astype(np.float32)
        ela_input = np.random.randn(*input_size).astype(np.float32)

        outputs = session.run(
            None,
            {"rgb_input": rgb_input, "ela_input": ela_input},
        )

        print(f"  ✓ ONNX Runtime inference successful")
        print(f"    Output shape: {outputs[0].shape}")
        return True

    except Exception as e:
        print(f"  ✗ ONNX Runtime inference failed: {e}")
        return False
