"""Deepfake classifier with optional attention and CAM-friendly features."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.models.attention import CBAM, SEBlock
from src.models.backbone import BackboneFactory


class DeepfakeClassifier(nn.Module):
    """Backbone → optional CBAM/SE → GAP → Dropout → Linear classifier.

    Args:
        backbone_name: timm backbone identifier.
        num_classes: Number of output classes.
        dropout: Dropout probability before classifier head.
        use_attention: Whether to apply attention module.
        pretrained: Load ImageNet pretrained backbone weights.
        attention_type: 'cbam' or 'se'.
    """

    def __init__(
        self,
        backbone_name: str,
        num_classes: int = 2,
        dropout: float = 0.3,
        use_attention: bool = True,
        pretrained: bool = True,
        attention_type: str = "cbam",
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.use_attention = use_attention

        self.backbone, feature_dim = BackboneFactory.create(
            backbone_name, pretrained=pretrained
        )
        self.feature_dim = feature_dim

        self.attention: Optional[nn.Module] = None
        if use_attention:
            if attention_type.lower() == "se":
                self.attention = SEBlock(feature_dim)
            else:
                self.attention = CBAM(feature_dim)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(feature_dim, num_classes)

        self._last_features: Optional[torch.Tensor] = None

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial feature maps from backbone."""
        features = self.backbone.forward_features(x)
        if features.dim() != 4:
            raise RuntimeError(
                "Backbone forward_features() must return (B, C, H, W) for attention/CAM."
            )

        if self.attention is not None:
            features = self.attention(features)

        self._last_features = features
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning class logits.

        Args:
            x: Input batch of shape (B, 3, H, W).

        Returns:
            Logits of shape (B, num_classes).
        """
        features = self._forward_features(x)
        pooled = self.pool(features).flatten(1)
        pooled = self.dropout(pooled)
        return self.head(pooled)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return spatial feature maps (B, C, H, W) for Grad-CAM.

        Args:
            x: Input batch.

        Returns:
            Last spatial feature tensor.
        """
        self._forward_features(x)
        if self._last_features is None:
            raise RuntimeError("Features not computed")
        return self._last_features

    def get_cam_target_layer(self) -> nn.Module:
        """Return the layer to hook for Grad-CAM."""
        if self.attention is not None:
            return self.attention
        return self.backbone
