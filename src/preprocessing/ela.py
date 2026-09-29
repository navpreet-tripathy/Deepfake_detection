# ============================================================
# ODD²F — ELA (Error Level Analysis) Preprocessing
# ============================================================
# Step 2 of the framework: Generate forensic ELA images that
# expose compression inconsistencies in tampered regions.
# ============================================================

import io
import cv2
import numpy as np
from PIL import Image


def compute_ela(
    image: np.ndarray,
    quality: int = 95,
    scale: int = 20,
) -> np.ndarray:
    """Compute Error Level Analysis for a single image.

    Algorithm:
        1. Re-save the image at the given JPEG quality.
        2. Compute pixel-wise absolute difference between
           the original and re-compressed image.
        3. Amplify the difference by `scale` for visibility.

    Args:
        image: Input image as a NumPy array (H, W, 3) in RGB uint8.
        quality: JPEG re-compression quality (default 95).
        scale: Amplification factor for the difference map.

    Returns:
        ELA image as a NumPy array (H, W, 3), uint8, clipped to [0, 255].
    """
    # Convert to PIL for JPEG re-compression
    pil_image = Image.fromarray(image)

    # Re-save to in-memory JPEG buffer
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)

    # Reload the re-compressed image
    resaved = Image.open(buffer)
    resaved_np = np.array(resaved, dtype=np.float32)

    # Pixel-wise absolute difference
    original_float = image.astype(np.float32)
    ela = np.abs(original_float - resaved_np) * scale

    # Clip and convert back to uint8
    ela = np.clip(ela, 0, 255).astype(np.uint8)

    return ela


def compute_ela_pil(
    pil_image: Image.Image,
    quality: int = 95,
    scale: int = 20,
) -> Image.Image:
    """PIL-to-PIL convenience wrapper for ELA computation.

    Args:
        pil_image: PIL Image in RGB mode.
        quality: JPEG quality for re-compression.
        scale: Amplification factor.

    Returns:
        ELA result as a PIL Image (RGB).
    """
    image_np = np.array(pil_image.convert("RGB"))
    ela_np = compute_ela(image_np, quality=quality, scale=scale)
    return Image.fromarray(ela_np)


def compute_ela_heatmap(
    image: np.ndarray,
    quality: int = 95,
    scale: int = 20,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Compute a colorized ELA heatmap for visualization.

    Args:
        image: Input image (H, W, 3) RGB uint8.
        quality: JPEG quality for re-compression.
        scale: Amplification factor.
        colormap: OpenCV colormap constant.

    Returns:
        Colorized heatmap (H, W, 3) in RGB uint8.
    """
    ela = compute_ela(image, quality=quality, scale=scale)

    # Convert to grayscale for heatmap
    gray = cv2.cvtColor(ela, cv2.COLOR_RGB2GRAY)

    # Apply colormap (OpenCV uses BGR)
    heatmap_bgr = cv2.applyColorMap(gray, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    return heatmap_rgb


def overlay_ela_on_image(
    image: np.ndarray,
    quality: int = 95,
    scale: int = 20,
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay ELA heatmap on the original image for side-by-side forensics.

    Args:
        image: Original image (H, W, 3) RGB uint8.
        quality: JPEG quality.
        scale: Amplification factor.
        alpha: Blending weight for the heatmap.

    Returns:
        Blended overlay (H, W, 3) RGB uint8.
    """
    heatmap = compute_ela_heatmap(image, quality=quality, scale=scale)
    blended = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return blended
