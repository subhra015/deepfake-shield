"""PyTorch Dataset for deepfake image classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset

from src.exceptions import DataError


class DeepfakeDataset(Dataset):
    """Image dataset with Albumentations transforms and metadata.

    Args:
        image_paths: List of image file paths.
        labels: Binary labels (0=real, 1=fake).
        transform: Optional Albumentations compose pipeline.
    """

    def __init__(
        self,
        image_paths: List[Union[str, Path]],
        labels: List[int],
        transform: Optional[A.Compose] = None,
    ) -> None:
        if len(image_paths) != len(labels):
            raise DataError(
                f"image_paths ({len(image_paths)}) and labels ({len(labels)}) must match"
            )
        self.image_paths: List[Path] = [Path(p) for p in image_paths]
        self.labels: List[int] = list(labels)
        self.transform = transform

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load and transform a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dict with keys ``image``, ``label``, and ``path``.
        """
        path = self.image_paths[idx]
        label = int(self.labels[idx])

        image = cv2.imread(str(path))
        if image is None:
            raise DataError(f"Could not read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
        if image.dtype != torch.float32:
            image = image.float()

        return {"image": image, "label": label, "path": str(path)}


def collect_images_from_dir(
    root_dir: Union[str, Path],
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
) -> Tuple[List[str], List[int]]:
    """Collect paths and binary labels from ``real/`` and ``fake/`` subfolders.

    Args:
        root_dir: Root directory containing class subfolders.
        extensions: Allowed image extensions.

    Returns:
        Tuple of (image_paths, labels).

    Raises:
        DataError: If no images are found.
    """
    root = Path(root_dir)
    paths: List[str] = []
    labels: List[int] = []

    for class_name, label in [("real", 0), ("fake", 1)]:
        class_dir = root / class_name
        if not class_dir.is_dir():
            continue
        for ext in extensions:
            for image_path in class_dir.rglob(f"*{ext}"):
                paths.append(str(image_path.resolve()))
                labels.append(label)

    if not paths:
        raise DataError(
            f"No images found under {root}. Expected real/ and fake/ subfolders."
        )
    return paths, labels


def stratified_train_val_split(
    image_paths: List[str],
    labels: List[int],
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], List[int], List[str], List[int]]:
    """Stratified train/validation split preserving class ratios.

    Args:
        image_paths: All image paths.
        labels: Corresponding labels.
        val_fraction: Fraction held out for validation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_paths, train_labels, val_paths, val_labels).
    """
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=seed
    )
    indices = np.arange(len(labels))
    train_idx, val_idx = next(splitter.split(indices, labels))

    train_paths = [image_paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_paths = [image_paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    return train_paths, train_labels, val_paths, val_labels
