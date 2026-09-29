# ============================================================
# ODD²F — Dual-Stream Deepfake Detection Model
# ============================================================
# Steps 4–7: Twin MobileNetV3 backbones + CBAM attention +
# late fusion + classification head.
# ============================================================

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .cbam import CBAM


class StreamEncoder(nn.Module):
    """Single-stream encoder: MobileNetV3-Large backbone + CBAM.

    Extracts 7×7×960 feature maps from the backbone, refines them
    with CBAM attention, then produces a 960-dim feature vector
    via adaptive average pooling.

    Args:
        pretrained: Whether to load ImageNet-pretrained weights.
        feature_dim: Expected feature map channel count (960).
        cbam_reduction: CBAM channel reduction ratio.
        cbam_kernel_size: CBAM spatial conv kernel size.
    """

    def __init__(
        self,
        pretrained: bool = True,
        feature_dim: int = 960,
        cbam_reduction: int = 16,
        cbam_kernel_size: int = 7,
    ):
        super().__init__()

        # Load MobileNetV3-Large without classifier head
        self.backbone = timm.create_model(
            "mobilenetv3_large_100",
            pretrained=pretrained,
            features_only=False,
            num_classes=0,          # removes classifier, returns pooled features
            global_pool="",         # disable global pool — we want spatial maps
        )

        # CBAM attention on the feature maps
        self.cbam = CBAM(
            channels=feature_dim,
            reduction=cbam_reduction,
            kernel_size=cbam_kernel_size,
        )

        # Adaptive average pool → flatten
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor (B, 3, 224, 224).

        Returns:
            Feature vector (B, 960).
        """
        features = self.backbone(x)            # (B, 960, 7, 7)
        attended = self.cbam(features)          # (B, 960, 7, 7)
        pooled = self.pool(attended)            # (B, 960, 1, 1)
        return pooled.flatten(1)               # (B, 960)

    def forward_with_attention(self, x: torch.Tensor) -> dict:
        """Forward pass that also returns attention maps for visualization.

        Returns:
            dict with 'features', 'attention_maps', and 'pooled_vector'.
        """
        features = self.backbone(x)
        attention_info = self.cbam.get_attention_maps(features)
        pooled = self.pool(attention_info["output"])
        return {
            "raw_features": features,
            "attention_maps": attention_info,
            "pooled_vector": pooled.flatten(1),
        }


class ODD2F(nn.Module):
    """ODD²F — Optimal Dual-Stream Deepfake Detection Framework.

    Architecture:
        RGB image  ─→ StreamEncoder (MobileNetV3 + CBAM) ─→ 960-dim
        ELA image  ─→ StreamEncoder (MobileNetV3 + CBAM) ─→ 960-dim
                                                             │
                                    Concatenate ─────────→ 1920-dim
                                        │
                                Dense(512, ReLU, Drop 0.3)
                                        │
                                Dense(256, ReLU, Drop 0.3)
                                        │
                                Dense(2) ─→ softmax → prediction

    Args:
        pretrained: Use ImageNet weights for backbones.
        feature_dim: Feature map channels (960).
        cbam_reduction: CBAM reduction ratio.
        cbam_kernel_size: CBAM spatial kernel.
        fusion_dim: First fusion FC dimension.
        classifier_hidden: Second FC dimension.
        dropout: Dropout rate in the classifier.
        num_classes: Output classes (2 for real/fake).
    """

    def __init__(
        self,
        pretrained: bool = True,
        feature_dim: int = 960,
        cbam_reduction: int = 16,
        cbam_kernel_size: int = 7,
        fusion_dim: int = 512,
        classifier_hidden: int = 256,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()

        # Two independent stream encoders (no weight sharing)
        self.rgb_encoder = StreamEncoder(
            pretrained=pretrained,
            feature_dim=feature_dim,
            cbam_reduction=cbam_reduction,
            cbam_kernel_size=cbam_kernel_size,
        )
        self.ela_encoder = StreamEncoder(
            pretrained=pretrained,
            feature_dim=feature_dim,
            cbam_reduction=cbam_reduction,
            cbam_kernel_size=cbam_kernel_size,
        )

        # Fusion + classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, fusion_dim),       # 1920 → 512
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, classifier_hidden),      # 512 → 256
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes),     # 256 → 2
        )

        self.num_classes = num_classes

    def forward(
        self,
        rgb: torch.Tensor,
        ela: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            rgb: RGB image tensor (B, 3, 224, 224).
            ela: ELA image tensor (B, 3, 224, 224).

        Returns:
            Logits (B, num_classes).
        """
        rgb_feat = self.rgb_encoder(rgb)    # (B, 960)
        ela_feat = self.ela_encoder(ela)    # (B, 960)

        # Late fusion: concatenate
        fused = torch.cat([rgb_feat, ela_feat], dim=1)  # (B, 1920)

        # Classification
        logits = self.classifier(fused)     # (B, 2)
        return logits

    def predict(
        self,
        rgb: torch.Tensor,
        ela: torch.Tensor,
    ) -> dict:
        """Convenience method for inference — returns label + confidence.

        Returns:
            dict with 'logits', 'probabilities', 'predicted_class',
            'confidence', and 'label' ('Real' or 'Fake').
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(rgb, ela)
            probs = F.softmax(logits, dim=1)
            confidence, predicted = probs.max(dim=1)

        labels = ["Real", "Fake"]
        return {
            "logits": logits,
            "probabilities": probs,
            "predicted_class": predicted.item(),
            "confidence": confidence.item(),
            "label": labels[predicted.item()],
        }

    def predict_with_attention(
        self,
        rgb: torch.Tensor,
        ela: torch.Tensor,
    ) -> dict:
        """Full inference with attention maps for the Streamlit app.

        Returns:
            dict with prediction info + RGB and ELA attention maps.
        """
        self.eval()
        with torch.no_grad():
            rgb_info = self.rgb_encoder.forward_with_attention(rgb)
            ela_info = self.ela_encoder.forward_with_attention(ela)

            fused = torch.cat(
                [rgb_info["pooled_vector"], ela_info["pooled_vector"]],
                dim=1,
            )
            logits = self.classifier(fused)
            probs = F.softmax(logits, dim=1)
            confidence, predicted = probs.max(dim=1)

        labels = ["Real", "Fake"]
        return {
            "logits": logits,
            "probabilities": probs,
            "predicted_class": predicted.item(),
            "confidence": confidence.item(),
            "label": labels[predicted.item()],
            "rgb_attention": rgb_info["attention_maps"],
            "ela_attention": ela_info["attention_maps"],
        }

    def count_parameters(self) -> dict:
        """Count trainable and total parameters.

        Returns:
            dict with 'total', 'trainable', and 'non_trainable' counts.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "non_trainable": total - trainable,
            "total_millions": round(total / 1e6, 2),
        }


class RGBOnlyModel(nn.Module):
    """Ablation variant: RGB stream only (no ELA)."""

    def __init__(self, pretrained=True, feature_dim=960, cbam_reduction=16,
                 cbam_kernel_size=7, fusion_dim=512, classifier_hidden=256,
                 dropout=0.3, num_classes=2):
        super().__init__()
        self.encoder = StreamEncoder(pretrained, feature_dim, cbam_reduction, cbam_kernel_size)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, rgb, ela=None):
        feat = self.encoder(rgb)
        return self.classifier(feat)


class ELAOnlyModel(nn.Module):
    """Ablation variant: ELA stream only (no RGB)."""

    def __init__(self, pretrained=True, feature_dim=960, cbam_reduction=16,
                 cbam_kernel_size=7, fusion_dim=512, classifier_hidden=256,
                 dropout=0.3, num_classes=2):
        super().__init__()
        self.encoder = StreamEncoder(pretrained, feature_dim, cbam_reduction, cbam_kernel_size)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, rgb, ela=None):
        # Use ela if provided, otherwise assume rgb is actually the ELA input
        inp = ela if ela is not None else rgb
        feat = self.encoder(inp)
        return self.classifier(feat)


class NoCBAMModel(nn.Module):
    """Ablation variant: Dual-stream without CBAM attention."""

    def __init__(self, pretrained=True, feature_dim=960,
                 fusion_dim=512, classifier_hidden=256,
                 dropout=0.3, num_classes=2):
        super().__init__()

        # Backbones without CBAM
        self.rgb_backbone = timm.create_model(
            "mobilenetv3_large_100", pretrained=pretrained,
            num_classes=0, global_pool="",
        )
        self.ela_backbone = timm.create_model(
            "mobilenetv3_large_100", pretrained=pretrained,
            num_classes=0, global_pool="",
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, rgb, ela):
        rgb_feat = self.pool(self.rgb_backbone(rgb)).flatten(1)
        ela_feat = self.pool(self.ela_backbone(ela)).flatten(1)
        fused = torch.cat([rgb_feat, ela_feat], dim=1)
        return self.classifier(fused)
