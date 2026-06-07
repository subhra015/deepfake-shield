"""Plug-and-play attention modules for (B, C, H, W) feature maps."""

from __future__ import annotations

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel recalibration.

    Args:
        channels: Number of input channels.
        reduction: Channel reduction ratio.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SE attention to input feature map.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Recalibrated tensor of shape (B, C, H, W).
        """
        batch_size, channels, _, _ = x.size()
        squeeze = self.pool(x).view(batch_size, channels)
        excitation = self.fc(squeeze).view(batch_size, channels, 1, 1)
        return x * excitation.expand_as(x)


class ChannelAttention(nn.Module):
    """Channel attention submodule for CBAM."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute channel attention weights."""
        batch_size, channels, _, _ = x.size()
        avg_out = self.mlp(self.avg_pool(x).view(batch_size, channels))
        max_out = self.mlp(self.max_pool(x).view(batch_size, channels))
        return torch.sigmoid(avg_out + max_out).view(batch_size, channels, 1, 1)


class SpatialAttention(nn.Module):
    """Spatial attention submodule for CBAM."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute spatial attention weights."""
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(combined))


class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    Args:
        channels: Number of input channels.
        reduction: Channel reduction ratio.
        kernel_size: Spatial attention convolution kernel size.
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel then spatial attention.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Attended tensor of shape (B, C, H, W).
        """
        x = x * self.channel_attn(x)
        x = x * self.spatial_attn(x)
        return x
