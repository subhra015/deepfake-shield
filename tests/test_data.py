"""Tests for dataset loading and transforms."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.data.dataset import DeepfakeDataset, collect_images_from_dir, stratified_train_val_split
from src.data.datamodule import DeepfakeDataModule, build_eval_transform, build_train_transform
from src.exceptions import DataError


def test_collect_images_from_dir(sample_image_dir: Path) -> None:
    """Verify image collection from class subfolders."""
    paths, labels = collect_images_from_dir(sample_image_dir)
    assert len(paths) == 6
    assert set(labels) == {0, 1}
    assert all(Path(p).exists() for p in paths)


def test_collect_images_raises_on_empty(tmp_path: Path) -> None:
    """Verify DataError when no images are found."""
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(DataError):
        collect_images_from_dir(empty)


def test_deepfake_dataset_getitem(sample_image_dir: Path) -> None:
    """Verify dataset returns expected keys and tensor shapes."""
    paths, labels = collect_images_from_dir(sample_image_dir)
    transform = build_eval_transform(image_size=224)
    dataset = DeepfakeDataset(paths, labels, transform)

    sample = dataset[0]
    assert set(sample.keys()) == {"image", "label", "path"}
    assert isinstance(sample["image"], torch.Tensor)
    assert sample["image"].shape == (3, 224, 224)
    assert sample["label"] in (0, 1)


def test_train_transform_changes_shape(sample_image_dir: Path) -> None:
    """Verify training transform produces normalized tensors."""
    paths, labels = collect_images_from_dir(sample_image_dir)
    dataset = DeepfakeDataset(paths[:1], labels[:1], build_train_transform(224))
    tensor = dataset[0]["image"]
    assert tensor.dtype == torch.float32
    assert tensor.shape == (3, 224, 224)


def test_stratified_split_preserves_classes(sample_image_dir: Path) -> None:
    """Verify stratified split maintains both classes."""
    paths, labels = collect_images_from_dir(sample_image_dir)
    train_paths, train_labels, val_paths, val_labels = stratified_train_val_split(
        paths, labels, val_fraction=0.33, seed=42
    )
    assert len(train_paths) + len(val_paths) == len(paths)
    assert set(train_labels) == {0, 1}
    assert set(val_labels) == {0, 1}


def test_datamodule_setup(sample_image_dir: Path) -> None:
    """Verify DataModule creates dataloaders from config."""
    config = {
        "seed": 42,
        "data": {
            "train_dir": str(sample_image_dir),
            "val_dir": str(sample_image_dir),
            "test_dir": str(sample_image_dir),
            "image_size": 224,
            "batch_size": 2,
            "num_workers": 0,
        },
    }
    dm = DeepfakeDataModule(config)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    assert batch["image"].shape[0] <= 2
    assert batch["image"].shape[1:] == (3, 224, 224)
