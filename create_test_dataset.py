"""
Create a small synthetic test dataset with random noise images.
Used to verify the full pipeline (train → evaluate → app) without downloading real data.
Creates 50 fake 'real' images and 50 fake 'fake' images.
"""
import os
import numpy as np
from PIL import Image
from pathlib import Path

def create_test_dataset(root: str = "test_dataset", n_per_class: int = 60):
    root = Path(root)
    real_dir = root / "real"
    fake_dir = root / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating synthetic test dataset at: {root.absolute()}")
    print(f"  {n_per_class} images per class (real + fake)")

    # Real images: smooth gradients (natural-looking)
    for i in range(n_per_class):
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        # Smooth gradient + slight noise (mimics real photos)
        for c in range(3):
            base = np.linspace(80 + c * 20, 180 + c * 10, 224)
            img[:, :, c] = np.clip(
                np.outer(np.ones(224), base) + np.random.normal(0, 15, (224, 224)),
                0, 255
            ).astype(np.uint8)
        Image.fromarray(img).save(real_dir / f"real_{i:04d}.jpg", quality=95)

    # Fake images: high-frequency noise patterns (mimics deepfake artifacts)
    for i in range(n_per_class):
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        # Checkerboard + noise pattern (mimics compression artifacts in fakes)
        x = np.arange(224)
        y = np.arange(224)
        xx, yy = np.meshgrid(x, y)
        for c in range(3):
            pattern = ((xx // 8 + yy // 8) % 2) * 60 + 100
            img[:, :, c] = np.clip(
                pattern + np.random.normal(0, 25, (224, 224)),
                0, 255
            ).astype(np.uint8)
        Image.fromarray(img.astype(np.uint8)).save(fake_dir / f"fake_{i:04d}.jpg", quality=85)

    print(f"  Created {n_per_class} real images in {real_dir}")
    print(f"  Created {n_per_class} fake images in {fake_dir}")
    print(f"\nTo train on this test dataset:")
    print(f"  python train.py --data_root {root} --epochs 3 --batch_size 8")
    print(f"\nTo run the Streamlit app:")
    print(f"  streamlit run app.py")

if __name__ == "__main__":
    create_test_dataset("test_dataset", n_per_class=60)
