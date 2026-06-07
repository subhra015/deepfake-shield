"""DataModule with Albumentations pipelines for train/val/test."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from src.data.dataset import (
    DeepfakeDataset,
    collect_images_from_dir,
    stratified_train_val_split,
)
from src.exceptions import DataError
from src.utils.logging import setup_logging

logger = setup_logging(name="deepfake.datamodule")


def _imagenet_normalize() -> A.Normalize:
    """Return ImageNet normalization transform."""
    return A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )


def build_train_transform(image_size: int = 224) -> A.Compose:
    """Build training augmentation pipeline.

    Args:
        image_size: Target spatial size.

    Returns:
        Albumentations compose pipeline.
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5,
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            _imagenet_normalize(),
            ToTensorV2(),
        ]
    )


def build_eval_transform(image_size: int = 224) -> A.Compose:
    """Build evaluation transform (resize + normalize only).

    Args:
        image_size: Target spatial size.

    Returns:
        Albumentations compose pipeline.
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            _imagenet_normalize(),
            ToTensorV2(),
        ]
    )


class DeepfakeDataModule:
    """Creates train/val/test datasets and dataloaders from YAML config.

    Args:
        config: Full project configuration dictionary.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        data_cfg = config.get("data", config)
        self.train_dir = Path(data_cfg.get("train_dir", "data/processed/train"))
        self.val_dir = Path(data_cfg.get("val_dir", "data/processed/val"))
        self.test_dir = Path(data_cfg.get("test_dir", "data/processed/test"))
        self.image_size = int(data_cfg.get("image_size", 224))
        self.batch_size = int(data_cfg.get("batch_size", 32))
        self.num_workers = int(data_cfg.get("num_workers", 4))
        self.seed = int(config.get("seed", 42))
        self.val_fraction = float(data_cfg.get("val_fraction", 0.0))

        self.train_dataset: Optional[DeepfakeDataset] = None
        self.val_dataset: Optional[DeepfakeDataset] = None
        self.test_dataset: Optional[DeepfakeDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize datasets for the given stage.

        Args:
            stage: Optional stage name ('fit', 'test', etc.).
        """
        train_transform = build_train_transform(self.image_size)
        eval_transform = build_eval_transform(self.image_size)

        train_paths, train_labels = self._load_split(self.train_dir)
        val_paths, val_labels = self._load_split(self.val_dir, optional=True)
        test_paths, test_labels = self._load_split(self.test_dir, optional=True)

        if not val_paths and self.val_fraction > 0:
            train_paths, train_labels, val_paths, val_labels = stratified_train_val_split(
                train_paths, train_labels, self.val_fraction, self.seed
            )

        self.train_dataset = DeepfakeDataset(train_paths, train_labels, train_transform)
        if not val_paths:
            raise DataError(
                f"No validation data found in {self.val_dir}. "
                "Provide val_dir with real/ and fake/ subfolders, "
                "or set data.val_fraction > 0."
            )
        self.val_dataset = DeepfakeDataset(val_paths, val_labels, eval_transform)

        if test_paths:
            self.test_dataset = DeepfakeDataset(test_paths, test_labels, eval_transform)
        else:
            self.test_dataset = self.val_dataset

        logger.info(
            "Setup complete: train=%d val=%d test=%d",
            len(self.train_dataset),
            len(self.val_dataset),
            len(self.test_dataset),
        )

    def _load_split(
        self, directory: Path, optional: bool = False
    ) -> Tuple[List[str], List[int]]:
        """Load image paths and labels from a directory split."""
        if not directory.exists():
            if optional:
                return [], []
            raise DataError(f"Data directory not found: {directory}")
        try:
            return collect_images_from_dir(directory)
        except DataError:
            if optional:
                return [], []
            raise

    def train_dataloader(self) -> DataLoader:
        """Return training DataLoader with augmentation."""
        if self.train_dataset is None:
            self.setup()
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=len(self.train_dataset) > self.batch_size,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation DataLoader."""
        if self.val_dataset is None:
            self.setup()
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test DataLoader."""
        if self.test_dataset is None:
            self.setup()
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
