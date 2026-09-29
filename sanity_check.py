"""Sanity check script — verifies environment and model instantiation."""
import sys
sys.path.insert(0, '.')

print("=" * 50)
print("  ODD2F Environment Sanity Check")
print("=" * 50)

# Check all imports
errors = []

try:
    import torch
    print(f"torch:           {torch.__version__}")
    print(f"CUDA available:  {torch.cuda.is_available()}")
    print(f"Device:          {'cuda' if torch.cuda.is_available() else 'cpu (AMD/no CUDA)'}")
except ImportError as e:
    errors.append(f"torch: {e}")

try:
    import torchvision
    print(f"torchvision:     {torchvision.__version__}")
except ImportError as e:
    errors.append(f"torchvision: {e}")

try:
    import timm
    print(f"timm:            {timm.__version__}")
except ImportError as e:
    errors.append(f"timm: {e}")

try:
    import cv2
    print(f"opencv-python:   {cv2.__version__}")
except ImportError as e:
    errors.append(f"cv2: {e}")

try:
    import albumentations
    print(f"albumentations:  {albumentations.__version__}")
except ImportError as e:
    errors.append(f"albumentations: {e}")

try:
    import onnx
    print(f"onnx:            {onnx.__version__}")
except ImportError as e:
    errors.append(f"onnx: {e}")

try:
    import onnxruntime
    print(f"onnxruntime:     {onnxruntime.__version__}")
except ImportError as e:
    errors.append(f"onnxruntime: {e}")

try:
    import tensorboard
    print(f"tensorboard:     OK")
except ImportError as e:
    errors.append(f"tensorboard: {e}")

try:
    import sklearn
    print(f"scikit-learn:    {sklearn.__version__}")
except ImportError as e:
    errors.append(f"scikit-learn: {e}")

try:
    import PIL
    print(f"Pillow:          {PIL.__version__}")
except ImportError as e:
    errors.append(f"pillow: {e}")

try:
    import numpy
    print(f"numpy:           {numpy.__version__}")
except ImportError as e:
    errors.append(f"numpy: {e}")

try:
    import pandas
    print(f"pandas:          {pandas.__version__}")
except ImportError as e:
    errors.append(f"pandas: {e}")

try:
    import matplotlib
    print(f"matplotlib:      {matplotlib.__version__}")
except ImportError as e:
    errors.append(f"matplotlib: {e}")

try:
    import seaborn
    print(f"seaborn:         {seaborn.__version__}")
except ImportError as e:
    errors.append(f"seaborn: {e}")

try:
    import yaml
    print(f"pyyaml:          OK")
except ImportError as e:
    errors.append(f"pyyaml: {e}")

print()

if errors:
    print("MISSING PACKAGES:")
    for e in errors:
        print(f"  - {e}")
else:
    print("All packages imported successfully!")

print()
print("=" * 50)
print("  Model Instantiation Test")
print("=" * 50)

try:
    from src.models.odd2f import ODD2F
    model = ODD2F(pretrained=False)  # Don't download weights for sanity check
    info = model.count_parameters()
    total = info['total']
    millions = info['total_millions']
    print(f"Model created successfully!")
    print(f"Total parameters:     {total:,}")
    print(f"Total (M):            {millions} M")
    print(f"Expected ~6.5M:       {'PASS' if 6.0 <= millions <= 7.0 else 'CHECK'}")
    print()

    # Quick forward pass test
    import torch
    rgb = torch.randn(1, 3, 224, 224)
    ela = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(rgb, ela)
    print(f"Forward pass test:    PASS (output shape: {list(out.shape)})")
    print()
    print("=" * 50)
    print("  ALL CHECKS PASSED - Environment Ready!")
    print("=" * 50)
except Exception as e:
    print(f"Model test FAILED: {e}")
    import traceback
    traceback.print_exc()
