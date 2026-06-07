"""Tests for model backbone and classifier."""

from __future__ import annotations

import pytest
import torch

from src.models.attention import CBAM, SEBlock
from src.models.backbone import BackboneFactory, FEATURE_DIMS
from src.models.classifier import DeepfakeClassifier


@pytest.mark.parametrize("backbone_name", ["resnet50", "convnext_tiny", "efficientnetv2_s"])
def test_backbone_factory(backbone_name: str) -> None:
    """Verify backbone creation and feature dimensions."""
    model, feature_dim = BackboneFactory.create(backbone_name, pretrained=False)
    assert feature_dim == FEATURE_DIMS[backbone_name]
    x = torch.randn(2, 3, 224, 224)
    features = model.forward_features(x)
    assert features.dim() == 4
    assert features.shape[1] == feature_dim


def test_cbam_shape() -> None:
    """Verify CBAM preserves tensor shape."""
    attention = CBAM(channels=64)
    x = torch.randn(2, 64, 14, 14)
    out = attention(x)
    assert out.shape == x.shape


def test_se_block_shape() -> None:
    """Verify SEBlock preserves tensor shape."""
    block = SEBlock(channels=128)
    x = torch.randn(1, 128, 7, 7)
    out = block(x)
    assert out.shape == x.shape


def test_classifier_forward() -> None:
    """Verify classifier forward pass output shape."""
    model = DeepfakeClassifier(
        backbone_name="convnext_tiny",
        num_classes=2,
        use_attention=True,
        pretrained=False,
    )
    model.eval()
    x = torch.randn(4, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (4, 2)


def test_classifier_get_features() -> None:
    """Verify get_features returns spatial maps for CAM."""
    model = DeepfakeClassifier(
        backbone_name="convnext_tiny",
        use_attention=True,
        pretrained=False,
    )
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    features = model.get_features(x)
    assert features.dim() == 4
    assert features.shape[0] == 1
