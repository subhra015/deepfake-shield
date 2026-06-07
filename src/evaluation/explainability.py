"""Grad-CAM explainability for deepfake classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from src.utils.logging import setup_logging

logger = setup_logging(name="deepfake.explainability")


class GradCAM:
    """Gradient-weighted Class Activation Mapping.

    Args:
        model: Classification model.
        target_layer: Layer to register forward/backward hooks on.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[Tensor] = None
        self.gradients: Optional[Tensor] = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(
        self, module: nn.Module, inputs: tuple, output: Tensor
    ) -> None:
        """Forward hook to capture activations."""
        self.activations = output.detach()

    def _save_gradient(
        self, module: nn.Module, grad_input: tuple, grad_output: tuple
    ) -> None:
        """Backward hook to capture gradients."""
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        """Remove registered hooks."""
        for handle in self._handles:
            handle.remove()

    def __call__(
        self,
        input_tensor: Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Compute Grad-CAM heatmap for a single input.

        Args:
            input_tensor: Input batch (1, C, H, W).
            target_class: Target class index; uses argmax if None.

        Returns:
            Normalized heatmap as numpy array (H, W).
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        output = self.model(input_tensor)
        if target_class is None:
            target_class = int(output.argmax(dim=1).item())

        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                "Gradients or activations not captured. Check target_layer."
            )

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        heatmap = cam.squeeze().cpu().numpy()

        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap

    @staticmethod
    def overlay_heatmap(
        image_rgb: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.45,
    ) -> np.ndarray:
        """Blend heatmap overlay onto RGB image.

        Args:
            image_rgb: Original RGB image.
            heatmap: Normalized heatmap (H, W).
            alpha: Overlay blending weight.

        Returns:
            Blended RGB uint8 array.
        """
        height, width = image_rgb.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (width, height))
        heatmap_color = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        if image_rgb.dtype != np.uint8:
            image_rgb = (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8)
        blended = (alpha * heatmap_color + (1 - alpha) * image_rgb).astype(np.uint8)
        return blended

    def generate(
        self,
        input_tensor: Tensor,
        original_image_rgb: np.ndarray,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Generate visualization-ready overlay image.

        Args:
            input_tensor: Preprocessed input tensor.
            original_image_rgb: Original RGB image for overlay.
            target_class: Target class for CAM.

        Returns:
            Blended visualization array.
        """
        heatmap = self(input_tensor, target_class)
        return self.overlay_heatmap(original_image_rgb, heatmap)

    def save_visualization(
        self,
        input_tensor: Tensor,
        original_image_rgb: np.ndarray,
        save_path: Union[str, Path],
        target_class: Optional[int] = None,
    ) -> Path:
        """Generate and save Grad-CAM visualization.

        Args:
            input_tensor: Preprocessed input tensor.
            original_image_rgb: Original RGB image.
            save_path: Output file path.
            target_class: Target class index.

        Returns:
            Path to saved visualization.
        """
        overlay = self.generate(input_tensor, original_image_rgb, target_class)
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output), bgr)
        logger.info("Saved Grad-CAM visualization to %s", output)
        return output
