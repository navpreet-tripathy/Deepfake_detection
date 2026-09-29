#!/usr/bin/env python3
# ============================================================
# ODD²F — Streamlit Deepfake Detection App
# ============================================================
# A premium forensic analysis interface with:
#   - Real/Fake prediction with confidence gauge
#   - ELA heatmap visualization
#   - CBAM attention map visualization
#   - Side-by-side forensic comparison
# ============================================================

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.odd2f import ODD2F
from src.preprocessing.ela import (
    compute_ela,
    compute_ela_heatmap,
    overlay_ela_on_image,
)


# ─── Page Config ──────────────────────────────────────────────

st.set_page_config(
    page_title="ODD²F — Deepfake Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Custom CSS ───────────────────────────────────────────────

st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: rgba(255, 255, 255, 0.7);
        font-size: 1.05rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }

    /* Result cards */
    .result-card {
        padding: 1.8rem;
        border-radius: 14px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .result-real {
        background: linear-gradient(135deg, #0a3d0a 0%, #1a5c1a 100%);
        border-left: 5px solid #00e676;
    }
    .result-fake {
        background: linear-gradient(135deg, #3d0a0a 0%, #5c1a1a 100%);
        border-left: 5px solid #ff1744;
    }
    .result-label {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: 1px;
    }
    .result-confidence {
        font-size: 1.1rem;
        margin-top: 0.3rem;
        opacity: 0.9;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .metric-label {
        color: rgba(255, 255, 255, 0.55);
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }
    .metric-value {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }

    /* Section headers */
    .section-header {
        color: #e8e8e8;
        font-size: 1.3rem;
        font-weight: 700;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(255, 255, 255, 0.1);
    }

    /* Sidebar styling */
    .sidebar-info {
        background: rgba(255, 255, 255, 0.03);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        margin-top: 1rem;
        font-size: 0.85rem;
        line-height: 1.6;
    }

    /* Image containers */
    .image-container {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─── Model Loading ────────────────────────────────────────────

@st.cache_resource
def load_model(checkpoint_path: str, device: str = "cpu"):
    """Load the trained ODD²F model."""
    model = ODD2F(pretrained=False)

    if Path(checkpoint_path).exists():
        checkpoint = torch.load(
            checkpoint_path,
            map_location=torch.device(device),
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model, True
    else:
        return model, False


def get_transform():
    """Standard inference transform (ImageNet normalization)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def generate_attention_heatmap(
    attention_map: torch.Tensor,
    original_size: tuple,
) -> np.ndarray:
    """Convert CBAM spatial attention map to a displayable heatmap.

    Args:
        attention_map: Spatial attention (1, 1, H, W) from CBAM.
        original_size: (width, height) for resizing.

    Returns:
        Colorized heatmap as (H, W, 3) RGB uint8.
    """
    att = attention_map.squeeze().cpu().numpy()  # (H, W)
    att = (att - att.min()) / (att.max() - att.min() + 1e-8)
    att = (att * 255).astype(np.uint8)
    att_resized = cv2.resize(att, original_size)
    heatmap_bgr = cv2.applyColorMap(att_resized, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return heatmap_rgb


def create_confidence_gauge(confidence: float, label: str) -> plt.Figure:
    """Create a semi-circular confidence gauge chart."""
    fig, ax = plt.subplots(figsize=(4, 2.5), subplot_kw={"projection": "polar"})

    # Setup
    ax.set_theta_offset(np.pi)
    ax.set_theta_direction(-1)
    ax.set_rlim(0, 1)
    ax.set_rticks([])
    ax.set_thetagrids([])
    ax.spines["polar"].set_visible(False)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # Background arc
    theta_bg = np.linspace(0, np.pi, 100)
    ax.fill_between(theta_bg, 0.6, 1.0, alpha=0.1, color="white")

    # Confidence arc
    theta_val = np.linspace(0, np.pi * confidence, 100)
    color = "#00e676" if label == "Real" else "#ff1744"
    ax.fill_between(theta_val, 0.6, 1.0, alpha=0.8, color=color)

    # Text
    ax.text(
        np.pi / 2, 0.25, f"{confidence * 100:.1f}%",
        ha="center", va="center", fontsize=22,
        fontweight="bold", color="white",
    )
    ax.text(
        np.pi / 2, -0.1, "Confidence",
        ha="center", va="center", fontsize=10,
        color="rgba(255,255,255,0.6)",
    )

    plt.tight_layout()
    return fig


# ─── Main App ─────────────────────────────────────────────────

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 ODD²F — Deepfake Detector</h1>
        <p>Optimal Dual-Stream Deepfake Detection Framework with CBAM Attention</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        checkpoint_path = st.text_input(
            "Model Checkpoint",
            value="runs/best_model.pth",
            help="Path to the trained .pth file",
        )

        device_option = st.selectbox(
            "Device",
            ["auto", "cpu", "cuda"],
            index=0,
        )
        if device_option == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = device_option

        st.markdown(f"**Using:** `{device}`")

        ela_quality = st.slider("ELA JPEG Quality", 50, 100, 95)
        ela_scale = st.slider("ELA Amplification", 1, 50, 20)

        st.markdown("---")
        st.markdown("""
        <div class="sidebar-info">
            <strong>How it works:</strong><br>
            1. Upload an image (face photo)<br>
            2. The model processes it through dual streams:<br>
            &nbsp;&nbsp;&nbsp;• <strong>RGB Stream</strong> — analyzes visual content<br>
            &nbsp;&nbsp;&nbsp;• <strong>ELA Stream</strong> — detects compression anomalies<br>
            3. CBAM attention focuses on suspicious regions<br>
            4. Features are fused for final prediction
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div class="sidebar-info">
            <strong>Architecture:</strong><br>
            • Twin MobileNetV3-Large backbones<br>
            • CBAM (Channel + Spatial) attention<br>
            • Late fusion at 1920-dim<br>
            • ~6.5M parameters<br>
            • Trained on 140K Faces + CIFAKE + Celeb-DF
        </div>
        """, unsafe_allow_html=True)

    # ── Load model ──
    model, model_loaded = load_model(checkpoint_path, device)

    if not model_loaded:
        st.warning(
            f"⚠️ No trained model found at `{checkpoint_path}`. "
            f"The app will run but predictions will be random. "
            f"Train the model first with `python train.py`."
        )

    # ── File uploader ──
    st.markdown('<p class="section-header">📤 Upload Image for Analysis</p>',
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        original_size = (image.width, image.height)

        # ── Show uploaded image ──
        col_img, col_info = st.columns([2, 1])
        with col_img:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        with col_info:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Dimensions</p>
                <p class="metric-value">{image.width} × {image.height}</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">File Size</p>
                <p class="metric-value">{uploaded_file.size / 1024:.1f} KB</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Format</p>
                <p class="metric-value">{uploaded_file.type.split('/')[-1].upper()}</p>
            </div>
            """, unsafe_allow_html=True)

        # ── Run analysis ──
        with st.spinner("🔬 Analyzing image through dual streams..."):
            # Prepare inputs
            transform = get_transform()
            rgb_tensor = transform(image).unsqueeze(0).to(device)

            ela_np = compute_ela(image_np, quality=ela_quality, scale=ela_scale)
            ela_pil = Image.fromarray(ela_np)
            ela_tensor = transform(ela_pil).unsqueeze(0).to(device)

            # Predict with attention maps
            result = model.predict_with_attention(rgb_tensor, ela_tensor)

        # ── Prediction Result ──
        st.markdown('<p class="section-header">🎯 Prediction Result</p>',
                    unsafe_allow_html=True)

        label = result["label"]
        confidence = result["confidence"]
        probs = result["probabilities"].squeeze().cpu().numpy()

        col_result, col_gauge, col_probs = st.columns([1.2, 1, 1])

        with col_result:
            result_class = "result-real" if label == "Real" else "result-fake"
            result_emoji = "✅" if label == "Real" else "🚨"
            st.markdown(f"""
            <div class="result-card {result_class}">
                <p class="result-label">{result_emoji} {label.upper()}</p>
                <p class="result-confidence">Confidence: {confidence * 100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

        with col_gauge:
            gauge_fig = create_confidence_gauge(confidence, label)
            st.pyplot(gauge_fig, use_container_width=True)
            plt.close(gauge_fig)

        with col_probs:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Real Probability</p>
                <p class="metric-value" style="color: #00e676;">{probs[0]*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Fake Probability</p>
                <p class="metric-value" style="color: #ff1744;">{probs[1]*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

        # ── ELA Analysis ──
        st.markdown('<p class="section-header">🔬 Error Level Analysis (ELA)</p>',
                    unsafe_allow_html=True)

        col_ela1, col_ela2, col_ela3 = st.columns(3)

        with col_ela1:
            st.image(image_np, caption="Original Image", use_container_width=True)

        with col_ela2:
            st.image(ela_np, caption="ELA Difference Map", use_container_width=True)

        with col_ela3:
            ela_heatmap = compute_ela_heatmap(image_np, quality=ela_quality, scale=ela_scale)
            overlay = overlay_ela_on_image(image_np, quality=ela_quality,
                                           scale=ela_scale, alpha=0.4)
            st.image(overlay, caption="ELA Overlay", use_container_width=True)

        # ── CBAM Attention Maps ──
        st.markdown('<p class="section-header">🧠 CBAM Attention Maps</p>',
                    unsafe_allow_html=True)

        col_att1, col_att2 = st.columns(2)

        with col_att1:
            st.markdown("**RGB Stream Attention**")
            rgb_spatial = result["rgb_attention"]["spatial_attention"]
            rgb_heatmap = generate_attention_heatmap(rgb_spatial, original_size)
            # Blend with original
            rgb_blend = cv2.addWeighted(image_np, 0.5, rgb_heatmap, 0.5, 0)
            st.image(rgb_blend, caption="RGB Stream — Where the model focuses",
                     use_container_width=True)

        with col_att2:
            st.markdown("**ELA Stream Attention**")
            ela_spatial = result["ela_attention"]["spatial_attention"]
            ela_heatmap_att = generate_attention_heatmap(ela_spatial, original_size)
            ela_blend = cv2.addWeighted(ela_np, 0.5, ela_heatmap_att, 0.5, 0)
            st.image(ela_blend, caption="ELA Stream — Compression anomaly focus",
                     use_container_width=True)

        # ── Channel Attention Visualization ──
        with st.expander("📊 Channel Attention Distribution"):
            col_ch1, col_ch2 = st.columns(2)

            with col_ch1:
                rgb_ch = result["rgb_attention"]["channel_attention"].squeeze().cpu().numpy()
                fig, ax = plt.subplots(figsize=(10, 3))
                fig.patch.set_alpha(0.0)
                ax.set_facecolor("none")
                ax.bar(range(len(rgb_ch)), rgb_ch, color="#3498db", alpha=0.7, width=1.0)
                ax.set_xlabel("Channel Index", color="white")
                ax.set_ylabel("Attention Weight", color="white")
                ax.set_title("RGB Stream — Channel Attention", color="white")
                ax.tick_params(colors="white")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            with col_ch2:
                ela_ch = result["ela_attention"]["channel_attention"].squeeze().cpu().numpy()
                fig, ax = plt.subplots(figsize=(10, 3))
                fig.patch.set_alpha(0.0)
                ax.set_facecolor("none")
                ax.bar(range(len(ela_ch)), ela_ch, color="#e74c3c", alpha=0.7, width=1.0)
                ax.set_xlabel("Channel Index", color="white")
                ax.set_ylabel("Attention Weight", color="white")
                ax.set_title("ELA Stream — Channel Attention", color="white")
                ax.tick_params(colors="white")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        # ── Technical Details ──
        with st.expander("🔧 Technical Details"):
            param_info = model.count_parameters()
            st.markdown(f"""
            | Property | Value |
            |----------|-------|
            | **Model** | ODD²F (Optimal Dual-Stream Deepfake Detection) |
            | **Parameters** | {param_info['total']:,} ({param_info['total_millions']}M) |
            | **Backbone** | MobileNetV3-Large × 2 |
            | **Attention** | CBAM (Channel + Spatial) |
            | **Fusion** | Late concatenation at 1920-dim |
            | **Device** | {device} |
            | **ELA Quality** | {ela_quality}% |
            | **ELA Scale** | {ela_scale}× |
            """)

    else:
        # Welcome state
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; opacity: 0.6;">
            <p style="font-size: 4rem; margin-bottom: 1rem;">📸</p>
            <p style="font-size: 1.2rem; font-weight: 500;">
                Upload an image to begin forensic analysis
            </p>
            <p style="font-size: 0.9rem; margin-top: 0.5rem;">
                Supports JPG, PNG, BMP, and WebP formats
            </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
