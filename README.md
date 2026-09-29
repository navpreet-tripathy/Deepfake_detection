# ODD²F — Optimal Dual-Stream Deepfake Detection Framework

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b.svg)](https://streamlit.io/)

A binary (real/fake) image classifier that fuses **raw RGB analysis** with **Error Level Analysis (ELA)** forensic preprocessing, using twin MobileNetV3 backbones enhanced with **CBAM attention**, targeting **~98% accuracy** with **~6.5M parameters**.

---

## 🏗️ Architecture

```
RGB Image  ─→ MobileNetV3-Large ─→ CBAM ─→ 960-dim ─┐
                                                       ├─→ Concat (1920) ─→ FC(512) ─→ FC(256) ─→ FC(2) ─→ Real/Fake
ELA Image  ─→ MobileNetV3-Large ─→ CBAM ─→ 960-dim ─┘
```

### Key Components
| Component | Purpose |
|-----------|---------|
| **Dual-Stream** | RGB captures semantics; ELA captures compression anomalies |
| **MobileNetV3-Large** | Efficient backbone (~3.2M params each) for edge deployment |
| **CBAM Attention** | Focuses on tampered regions (channel + spatial) |
| **Late Fusion** | Independent feature learning before combining at 1920-dim |
| **ELA Preprocessing** | JPEG re-save → pixel diff exposes forgery artifacts |

---

## 📂 Project Structure

```
Deepfake_detection/
├── config/
│   └── default.yaml           # All hyperparameters and paths
├── src/
│   ├── preprocessing/
│   │   └── ela.py             # ELA computation + heatmaps
│   ├── models/
│   │   ├── cbam.py            # CBAM attention module
│   │   └── odd2f.py           # Dual-stream model + ablation variants
│   ├── data/
│   │   └── dataset.py         # Dataset, transforms, CutMix, splits
│   ├── training/
│   │   └── trainer.py         # Training loop + AMP + early stopping
│   ├── evaluation/
│   │   └── evaluator.py       # Evaluation + ablation + robustness
│   └── export/
│       └── converter.py       # ONNX + TFLite export
├── train.py                   # Training entry point
├── evaluate.py                # Evaluation entry point
├── app.py                     # Streamlit web application
├── requirements.txt           # Dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/navpreet-tripathy/Deepfake_detection.git
cd Deepfake_detection

# Create virtual environment (Python 3.11)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

Organize your dataset with this structure:
```
your_dataset/
├── real/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
└── fake/
    ├── image001.jpg
    ├── image002.jpg
    └── ...
```

Update the dataset paths in `config/default.yaml`:
```yaml
data:
  datasets:
    - name: "my_dataset"
      root: "path/to/your_dataset"
```

### 3. Train the Model

```bash
# Using config file
python train.py --config config/default.yaml

# Or with CLI overrides
python train.py --data_root path/to/dataset --epochs 50 --batch_size 32

# Resume training
python train.py --resume runs/best_model.pth --data_root path/to/dataset
```

### 4. Evaluate

```bash
# Standard evaluation
python evaluate.py --checkpoint runs/best_model.pth --data_root path/to/dataset

# Full evaluation + ablation + robustness + export
python evaluate.py --checkpoint runs/best_model.pth \
                   --data_root path/to/dataset \
                   --ablation --robustness --export
```

### 5. Launch the App

```bash
streamlit run app.py
```

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| **Datasets** | 140K Real & Fake Faces, CIFAKE, Celeb-DF |
| **Loss** | Cross-Entropy |
| **Optimizer** | Adam (lr=1e-3, weight_decay=1e-5) |
| **Scheduler** | Linear warmup (3 epochs) → Cosine annealing |
| **Batch size** | 32 |
| **Epochs** | 50 with early stopping (patience=7) |
| **Augmentation** | Flip, rotation, crop, color jitter, blur, CutMix |
| **Mixed Precision** | Enabled (AMP) |

---

## 📊 Evaluation

### Ablation Study

| Variant | Description |
|---------|-------------|
| **Full ODD²F** | RGB + ELA + CBAM (complete model) |
| **RGB-Only** | Single RGB stream, no ELA |
| **ELA-Only** | Single ELA stream, no RGB |
| **No-CBAM** | Dual-stream without attention |

### Robustness Tests

Tests AUC degradation (<4% expected) under:
- JPEG compression (Q=70, 50, 30)
- Rotation (5°, 10°, 15°, 20°)
- Scaling (0.5×, 0.75×, 1.25×, 1.5×)
- Gaussian noise (σ=0.01, 0.02, 0.05)
- Brightness shifts (±0.1, ±0.2)

---

## 🖥️ Streamlit App Features

- **🎯 Real/Fake prediction** with confidence gauge
- **🔬 ELA heatmap** showing compression anomalies
- **🧠 CBAM attention maps** for both RGB and ELA streams
- **📊 Channel attention distribution** charts
- **🔧 Configurable** ELA quality and amplification
- **📱 Responsive** dark-themed UI

---

## 📦 Export

The model can be exported for deployment:

```bash
python evaluate.py --checkpoint runs/best_model.pth --export
```

Produces:
- `exports/odd2f.onnx` — ONNX format (cross-platform)
- `exports/odd2f.tflite` — TensorFlow Lite (mobile/edge)

---

## 📜 License

MIT License
