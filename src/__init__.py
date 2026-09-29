# ============================================================
# ODD²F — Optimal Dual-Stream Deepfake Detection Framework
# ============================================================
#
# Package structure:
#   src/
#   ├── preprocessing/    ELA computation
#   ├── models/           ODD²F model + CBAM + ablation variants
#   ├── data/             Dataset loading + augmentation
#   ├── training/         Training engine
#   ├── evaluation/       Evaluation + ablation + robustness
#   └── export/           ONNX + TFLite conversion
# ============================================================

__version__ = "1.0.0"
__author__ = "ODD²F Team"
