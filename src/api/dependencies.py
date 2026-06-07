"""FastAPI dependency injection and application state."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.evaluation.explainability import GradCAM
from src.exceptions import ModelLoadError
from src.inference.predictor import DeepfakePredictor
from src.utils.logging import setup_logging

logger = setup_logging(name="deepfake.api")

APP_VERSION = "1.0.0"
DEFAULT_CHECKPOINT = Path(os.environ.get("DEEPFAKE_CHECKPOINT", "models/best_model.pth"))


@dataclass
class AppState:
    """Singleton application state for model and metrics."""

    predictor: Optional[DeepfakePredictor] = None
    grad_cam: Optional[GradCAM] = None
    executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=4)
    )
    inference_count: int = 0
    total_latency_ms: float = 0.0

    def load_model(self, checkpoint_path: Optional[Path] = None) -> None:
        """Load predictor and Grad-CAM from checkpoint.

        Args:
            checkpoint_path: Optional override path; uses env/default otherwise.
        """
        path = checkpoint_path or DEFAULT_CHECKPOINT
        if not path.exists():
            logger.warning("Checkpoint not found at %s; API will run in degraded mode", path)
            return
        try:
            self.predictor = DeepfakePredictor(path)
            model = self.predictor.get_model()
            self.grad_cam = GradCAM(model, model.get_cam_target_layer())
            logger.info("Model loaded from %s", path)
        except ModelLoadError as exc:
            logger.error("Failed to load model: %s", exc.message)

    def record_inference(self, latency_ms: float) -> None:
        """Update Prometheus-style inference counters."""
        self.inference_count += 1
        self.total_latency_ms += latency_ms

    def shutdown(self) -> None:
        """Release resources; executor is recreated on next startup if needed."""
        if self.grad_cam is not None:
            self.grad_cam.remove_hooks()
        self.predictor = None
        self.grad_cam = None


app_state = AppState()


def get_app_state() -> AppState:
    """Return the global application state singleton."""
    return app_state


def get_predictor() -> DeepfakePredictor:
    """Return loaded predictor or raise if unavailable."""
    if app_state.predictor is None:
        raise ModelLoadError("Model not loaded")
    return app_state.predictor


def get_grad_cam() -> Optional[GradCAM]:
    """Return Grad-CAM instance if model is loaded."""
    return app_state.grad_cam
