"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.models.classifier import DeepfakeClassifier
from src.utils.config import load_config


@pytest.fixture
def sample_image_dir(tmp_path: Path) -> Path:
    """Create a temporary dataset with real/ and fake/ subfolders."""
    for _, name in [(0, "real"), (1, "fake")]:
        class_dir = tmp_path / name
        class_dir.mkdir()
        for index in range(3):
            image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            cv2.imwrite(str(class_dir / f"img_{index}.jpg"), image)
    return tmp_path


@pytest.fixture
def dummy_checkpoint(tmp_path: Path) -> Path:
    """Create a minimal checkpoint for API/inference tests."""
    config_path = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
    config = load_config(config_path)
    config["model"]["pretrained"] = False
    model = DeepfakeClassifier(
        backbone_name="convnext_tiny",
        num_classes=2,
        use_attention=True,
        pretrained=False,
    )
    checkpoint_path = tmp_path / "test_model.pth"
    torch.save(
        {"model_state_dict": model.state_dict(), "config": config},
        checkpoint_path,
    )
    return checkpoint_path


@pytest.fixture
def api_client(dummy_checkpoint: Path, monkeypatch: pytest.MonkeyPatch):
    """FastAPI TestClient with a loaded dummy model."""
    from fastapi.testclient import TestClient

    from src.api import dependencies

    monkeypatch.setenv("DEEPFAKE_CHECKPOINT", str(dummy_checkpoint))
    dependencies.DEFAULT_CHECKPOINT = dummy_checkpoint

    from src.api.main import app

    with TestClient(app) as client:
        yield client

    dependencies.app_state.predictor = None
    dependencies.app_state.grad_cam = None
    dependencies.app_state.inference_count = 0
    dependencies.app_state.total_latency_ms = 0.0
