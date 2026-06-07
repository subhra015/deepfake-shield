"""Inference wrapper for single and batch predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.data.datamodule import build_eval_transform
from src.exceptions import InferenceError, ModelLoadError
from src.models.classifier import DeepfakeClassifier
from src.utils.device import get_device
from src.utils.logging import setup_logging

logger = setup_logging(name="deepfake.predictor")

LABEL_NAMES: Dict[int, str] = {0: "real", 1: "fake"}


class DeepfakePredictor:
    """Load a trained checkpoint and run image-level inference.

    Args:
        checkpoint_path: Path to ``.pth`` checkpoint.
        device: Optional device override.
        image_size: Input spatial size (must match training).
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[str] = None,
        image_size: int = 224,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = get_device(device or "cuda")
        self.image_size = image_size
        self.transform = build_eval_transform(image_size)

        if not self.checkpoint_path.exists():
            raise ModelLoadError(f"Checkpoint not found: {self.checkpoint_path}")

        try:
            checkpoint = torch.load(
                self.checkpoint_path, map_location=self.device, weights_only=False
            )
        except Exception as exc:
            raise ModelLoadError(f"Failed to load checkpoint: {exc}") from exc

        config = checkpoint.get("config", {})
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})
        self.image_size = int(data_cfg.get("image_size", image_size))
        self.transform = build_eval_transform(self.image_size)

        self.model = DeepfakeClassifier(
            backbone_name=model_cfg.get("backbone", "convnext_tiny"),
            num_classes=int(model_cfg.get("num_classes", 2)),
            dropout=float(model_cfg.get("dropout", 0.3)),
            use_attention=bool(model_cfg.get("use_attention", True)),
            pretrained=False,
        )
        state = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        self.config = config
        logger.info("Loaded model from %s on %s", self.checkpoint_path, self.device)

    def _load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Load an image from disk as RGB numpy array."""
        path = Path(image_path)
        image = cv2.imread(str(path))
        if image is None:
            raise InferenceError(f"Could not read image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _preprocess(self, image_rgb: np.ndarray) -> torch.Tensor:
        """Apply validation transforms and add batch dimension."""
        augmented = self.transform(image=image_rgb)
        tensor = augmented["image"].unsqueeze(0).to(self.device)
        return tensor

    @torch.no_grad()
    def predict_from_array(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        """Predict from an in-memory RGB image.

        Args:
            image_rgb: RGB numpy array (H, W, 3).

        Returns:
            Prediction dictionary with label, confidence, and probabilities.
        """
        try:
            tensor = self._preprocess(image_rgb)
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        except Exception as exc:
            raise InferenceError(f"Inference failed: {exc}") from exc

        pred_idx = int(np.argmax(probs))
        return {
            "label": LABEL_NAMES[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {"real": float(probs[0]), "fake": float(probs[1])},
        }

    @torch.no_grad()
    def predict(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Predict from an image file path.

        Args:
            image_path: Path to image file.

        Returns:
            Prediction dictionary.
        """
        image_rgb = self._load_image(image_path)
        return self.predict_from_array(image_rgb)

    @torch.no_grad()
    def predict_batch(self, image_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        """Run prediction on multiple image paths.

        Args:
            image_paths: List of image file paths.

        Returns:
            List of prediction dictionaries.
        """
        return [self.predict(path) for path in image_paths]

    def get_model(self) -> nn.Module:
        """Return the underlying PyTorch model."""
        return self.model
