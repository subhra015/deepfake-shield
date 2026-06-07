"""Tests for FastAPI endpoints."""

from __future__ import annotations

import io

import cv2
import numpy as np
from fastapi.testclient import TestClient


def _make_test_image_bytes() -> bytes:
    """Create a small JPEG image in memory."""
    image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", image)
    return encoded.tobytes()


def test_health_endpoint(api_client: TestClient) -> None:
    """Verify health endpoint returns model status."""
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert "device" in data


def test_predict_image_endpoint(api_client: TestClient) -> None:
    """Verify single image prediction endpoint."""
    image_bytes = _make_test_image_bytes()
    response = api_client.post(
        "/predict/image",
        files={"file": ("test.jpg", io.BytesIO(image_bytes), "image/jpeg")},
        headers={"X-Request-ID": "test-req-1"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["label"] in ("real", "fake")
    assert 0.0 <= data["confidence"] <= 1.0
    assert set(data["probabilities"].keys()) == {"real", "fake"}
    assert data["request_id"] == "test-req-1"


def test_metrics_endpoint(api_client: TestClient) -> None:
    """Verify Prometheus metrics endpoint."""
    response = api_client.get("/metrics")
    assert response.status_code == 200
    assert "deepfake_inference_total" in response.text
