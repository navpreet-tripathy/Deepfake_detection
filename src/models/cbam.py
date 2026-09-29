# ============================================================
# ODD²F — CBAM (Convolutional Block Attention Module)
# ============================================================
# Step 5 of the framework: Channel + Spatial attention applied
# independently to each stream's feature maps.
#
# Reference: Woo et al., "CBAM: Convolutional Block Attention
# Module", ECCV 2018.
# ============================================================

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel Attention sub-module of CBAM.

    Applies average-pooling and max-pooling across spatial dimensions,
    feeds each through a shared two-layer MLP, sums the results, and
    applies a sigmoid gate.

    Args:
        channels: Number of input channels (e.g., 960 for MobileNetV3).
        reduction: Channel reduction ratio for the bottleneck MLP.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map (B, C, H, W).

        Returns:
            Channel-attended feature map, same shape as input.
        """
        # Global average and max pooling → (B, C, 1, 1)
        avg_pool = x.mean(dim=(2, 3), keepdim=True)
        max_pool = x.amax(dim=(2, 3), keepdim=True)

        # Shared MLP
        avg_out = self.shared_mlp(avg_pool)
        max_out = self.shared_mlp(max_pool)

        # Combine and gate
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class SpatialAttention(nn.Module):
    """Spatial Attention sub-module of CBAM.

    Computes channel-wise average and max across the channel axis,
    concatenates them, and applies a 7×7 conv + sigmoid gate.

    Args:
        kernel_size: Convolution kernel size (default 7, as per paper).
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            2, 1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map (B, C, H, W).

        Returns:
            Spatially-attended feature map, same shape as input.
        """
        # Channel-wise pooling → (B, 1, H, W) each
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.amax(dim=1, keepdim=True)

        # Concatenate along channel axis → (B, 2, H, W)
        combined = torch.cat([avg_out, max_out], dim=1)

        # 7×7 conv → sigmoid gate
        attention = self.sigmoid(self.conv(combined))
        return x * attention


class CBAM(nn.Module):
    """Full CBAM module: Channel Attention → Spatial Attention.

    Args:
        channels: Number of input channels.
        reduction: Reduction ratio for channel attention.
        kernel_size: Kernel size for spatial attention conv.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        kernel_size: int = 7,
    ):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel then spatial attention sequentially."""
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

    def get_attention_maps(self, x: torch.Tensor):
        """Return intermediate attention maps for visualization.

        Returns:
            dict with keys:
                'channel_attention': (B, C, 1, 1) channel gate values
                'spatial_attention': (B, 1, H, W) spatial gate values
                'output': final attended feature map
        """
        # Channel attention
        avg_pool = x.mean(dim=(2, 3), keepdim=True)
        max_pool = x.amax(dim=(2, 3), keepdim=True)
        avg_out = self.channel_attention.shared_mlp(avg_pool)
        max_out = self.channel_attention.shared_mlp(max_pool)
        channel_gate = self.channel_attention.sigmoid(avg_out + max_out)
        x_channel = x * channel_gate

        # Spatial attention
        avg_s = x_channel.mean(dim=1, keepdim=True)
        max_s = x_channel.amax(dim=1, keepdim=True)
        combined = torch.cat([avg_s, max_s], dim=1)
        spatial_gate = self.spatial_attention.sigmoid(
            self.spatial_attention.conv(combined)
        )
        x_out = x_channel * spatial_gate

        return {
            "channel_attention": channel_gate,
            "spatial_attention": spatial_gate,
            "output": x_out,
        }
