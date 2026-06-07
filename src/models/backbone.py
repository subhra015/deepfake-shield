"""Backbone factory using timm feature extractors."""

from __future__ import annotations

from typing import Dict, Tuple

import timm
import torch.nn as nn

from src.utils.logging import setup_logging

logger = setup_logging(name="deepfake.backbone")

SUPPORTED_BACKBONES: Dict[str, str] = {
    "resnet50": "resnet50",
    "convnext_tiny": "convnext_tiny",
    "efficientnetv2_s": "tf_efficientnetv2_s",
}

FEATURE_DIMS: Dict[str, int] = {
    "resnet50": 2048,
    "convnext_tiny": 768,
    "efficientnetv2_s": 1280,
}


class BackboneFactory:
    """Create timm backbones in feature-extraction mode (num_classes=0)."""

    @staticmethod
    def create(
        backbone_name: str,
        pretrained: bool = True,
    ) -> Tuple[nn.Module, int]:
        """Instantiate a backbone and return it with feature dimension.

        Args:
            backbone_name: One of resnet50, convnext_tiny, efficientnetv2_s.
            pretrained: Whether to load ImageNet pretrained weights.

        Returns:
            Tuple of (backbone module, feature dimension).

        Raises:
            ValueError: If backbone_name is unsupported.
        """
        if backbone_name not in SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone '{backbone_name}'. "
                f"Choose from: {list(SUPPORTED_BACKBONES.keys())}"
            )

        timm_name = SUPPORTED_BACKBONES[backbone_name]
        model = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=0,
        )
        feature_dim = FEATURE_DIMS[backbone_name]

        if hasattr(model, "num_features"):
            feature_dim = int(model.num_features)

        logger.info("Created backbone %s with feature_dim=%d", backbone_name, feature_dim)
        return model, feature_dim
