"""Robustness evaluation under common perturbations."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.logging import setup_logging

logger = setup_logging(name="deepfake.robustness")


def _accuracy_from_loader(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    transform_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> float:
    """Compute classification accuracy on a dataloader with optional transform."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"]
            labels = batch["label"]
            if transform_fn is not None:
                transformed = []
                for img in images:
                    array = img.permute(1, 2, 0).numpy()
                    array = ((array * np.array([0.229, 0.224, 0.225])) +
                             np.array([0.485, 0.456, 0.406]))
                    array = (array * 255).clip(0, 255).astype(np.uint8)
                    array = transform_fn(array)
                    transformed.append(torch.from_numpy(array.transpose(2, 0, 1)).float())
                images = torch.stack(transformed)
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


def test_jpeg_compression(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    quality: int = 50,
) -> float:
    """Measure accuracy after JPEG compression.

    Args:
        model: Classification model.
        dataloader: Evaluation dataloader.
        device: Inference device.
        quality: JPEG quality factor (1-100).

    Returns:
        Accuracy after perturbation.
    """

    def compress(image: np.ndarray) -> np.ndarray:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode(
            ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), encode_param
        )
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    return _accuracy_from_loader(model, dataloader, device, transform_fn=compress)


def test_gaussian_noise(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    var_limit: float = 25.0,
) -> float:
    """Measure accuracy after additive Gaussian noise.

    Args:
        model: Classification model.
        dataloader: Evaluation dataloader.
        device: Inference device.
        var_limit: Noise variance.

    Returns:
        Accuracy after perturbation.
    """
    noise_transform = A.GaussNoise(var_limit=(var_limit, var_limit), p=1.0)

    def add_noise(image: np.ndarray) -> np.ndarray:
        return noise_transform(image=image)["image"]

    return _accuracy_from_loader(model, dataloader, device, transform_fn=add_noise)


def test_gaussian_blur(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    blur_limit: int = 7,
) -> float:
    """Measure accuracy after Gaussian blur.

    Args:
        model: Classification model.
        dataloader: Evaluation dataloader.
        device: Inference device.
        blur_limit: Maximum blur kernel size.

    Returns:
        Accuracy after perturbation.
    """
    blur_transform = A.GaussianBlur(blur_limit=(3, blur_limit), p=1.0)

    def blur(image: np.ndarray) -> np.ndarray:
        return blur_transform(image=image)["image"]

    return _accuracy_from_loader(model, dataloader, device, transform_fn=blur)


def test_resize_distortion(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    scale: float = 0.5,
) -> float:
    """Measure accuracy after downscale-upscale distortion.

    Args:
        model: Classification model.
        dataloader: Evaluation dataloader.
        device: Inference device.
        scale: Downscale factor before upscaling back.

    Returns:
        Accuracy after perturbation.
    """

    def distort(image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        small = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_LINEAR,
        )
        return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)

    return _accuracy_from_loader(model, dataloader, device, transform_fn=distort)


def run_robustness_suite(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    baseline_accuracy: Optional[float] = None,
) -> Dict[str, Any]:
    """Run all robustness tests and return accuracy drop table.

    Args:
        model: Classification model.
        dataloader: Evaluation dataloader.
        device: Inference device.
        baseline_accuracy: Optional precomputed baseline; computed if None.

    Returns:
        Dictionary with per-test accuracy and accuracy drops.
    """
    if baseline_accuracy is None:
        baseline_accuracy = _accuracy_from_loader(model, dataloader, device)

    tests: List[Tuple[str, Callable[[], float]]] = [
        ("jpeg_compression", lambda: test_jpeg_compression(model, dataloader, device)),
        ("gaussian_noise", lambda: test_gaussian_noise(model, dataloader, device)),
        ("gaussian_blur", lambda: test_gaussian_blur(model, dataloader, device)),
        ("resize_distortion", lambda: test_resize_distortion(model, dataloader, device)),
    ]

    results: Dict[str, Any] = {"baseline_accuracy": baseline_accuracy, "tests": {}}
    for name, test_fn in tests:
        accuracy = test_fn()
        drop = baseline_accuracy - accuracy
        results["tests"][name] = {
            "accuracy": accuracy,
            "accuracy_drop": drop,
        }
        logger.info(
            "Robustness %s: accuracy=%.4f drop=%.4f", name, accuracy, drop
        )

    return results
