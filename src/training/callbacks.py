"""Training callbacks: checkpointing, early stopping, W&B logging."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from src.utils.logging import setup_logging

logger = setup_logging(name="deepfake.callbacks")


class ModelCheckpoint:
    """Save top-k checkpoints based on a monitored metric.

    Args:
        save_dir: Directory for checkpoint files.
        monitor: Metric key to monitor.
        mode: 'max' or 'min'.
        top_k: Number of best checkpoints to retain.
        filename_template: Template for checkpoint filenames.
    """

    def __init__(
        self,
        save_dir: Union[str, Path],
        monitor: str = "val_auc",
        mode: str = "max",
        top_k: int = 3,
        filename_template: str = "epoch_{epoch:03d}_{monitor}_{value:.4f}.pth",
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.top_k = top_k
        self.filename_template = filename_template
        self.best_scores: List[tuple[float, Path]] = []

    def step(
        self,
        model: torch.nn.Module,
        metrics: Dict[str, float],
        epoch: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        """Save checkpoint if metric qualifies for top-k.

        Args:
            model: Model to serialize.
            metrics: Current epoch metrics.
            epoch: Current epoch number.
            extra: Additional data to store in checkpoint.

        Returns:
            Path to best_model.pth if this is a new best, else checkpoint path.
        """
        value = metrics.get(self.monitor)
        if value is None:
            return None

        is_better = (
            not self.best_scores
            or (self.mode == "max" and value > self.best_scores[0][0])
            or (self.mode == "min" and value < self.best_scores[0][0])
        )

        filename = self.filename_template.format(
            epoch=epoch, monitor=self.monitor, value=value
        )
        path = self.save_dir / filename

        checkpoint: Dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
        }
        if extra:
            checkpoint.update(extra)

        torch.save(checkpoint, path)
        logger.info("Saved checkpoint: %s", path)

        self.best_scores.append((value, path))
        self.best_scores.sort(key=lambda item: item[0], reverse=(self.mode == "max"))

        while len(self.best_scores) > self.top_k:
            _, old_path = self.best_scores.pop()
            if old_path.exists() and old_path not in [p for _, p in self.best_scores]:
                old_path.unlink(missing_ok=True)

        self.best_scores = self.best_scores[: self.top_k]

        if is_better:
            best_path = self.save_dir / "best_model.pth"
            shutil.copy(path, best_path)
            logger.info("New best %s=%.4f saved to %s", self.monitor, value, best_path)
            return best_path
        return path

    @property
    def best_metric(self) -> Optional[float]:
        """Return the best monitored metric value."""
        return self.best_scores[0][0] if self.best_scores else None


class EarlyStopping:
    """Stop training when monitored metric stops improving.

    Args:
        monitor: Metric key to monitor.
        patience: Epochs to wait without improvement.
        mode: 'max' or 'min'.
        min_delta: Minimum change to qualify as improvement.
    """

    def __init__(
        self,
        monitor: str = "val_auc",
        patience: int = 10,
        mode: str = "max",
        min_delta: float = 1e-4,
    ) -> None:
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best: Optional[float] = None
        self.should_stop = False

    def step(self, metrics: Dict[str, float]) -> bool:
        """Update early stopping state.

        Args:
            metrics: Current epoch metrics.

        Returns:
            True if training should stop.
        """
        value = metrics.get(self.monitor)
        if value is None:
            return False

        if self.best is None:
            self.best = value
            return False

        improved = (
            value > self.best + self.min_delta
            if self.mode == "max"
            else value < self.best - self.min_delta
        )

        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "Early stopping triggered after %d epochs without improvement",
                    self.patience,
                )
        return self.should_stop


class WandbLogger:
    """Log metrics, hyperparameters, and sample images to Weights & Biases.

    Args:
        project: W&B project name.
        config: Hyperparameter config dict.
        enabled: Whether W&B logging is active.
        run_name: Optional run display name.
    """

    def __init__(
        self,
        project: str,
        config: Dict[str, Any],
        enabled: bool = True,
        run_name: Optional[str] = None,
    ) -> None:
        self.enabled = enabled
        self.run = None
        self._wandb = None
        if not enabled:
            return
        try:
            import wandb

            self._wandb = wandb
            self.run = wandb.init(project=project, config=config, name=run_name)
            logger.info("Initialized W&B project: %s", project)
        except ImportError:
            logger.warning("wandb not installed; disabling W&B logger")
            self.enabled = False

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log scalar metrics to W&B."""
        if self.enabled and self.run is not None and self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def log_images(
        self, images: List[np.ndarray], captions: Optional[List[str]] = None
    ) -> None:
        """Log sample images to W&B."""
        if not self.enabled or self.run is None or self._wandb is None:
            return
        wandb_images = []
        for index, img in enumerate(images):
            caption = captions[index] if captions and index < len(captions) else None
            wandb_images.append(self._wandb.Image(img, caption=caption))
        self._wandb.log({"samples": wandb_images})

    def finish(self) -> None:
        """Finish the W&B run."""
        if self.enabled and self.run is not None and self._wandb is not None:
            self._wandb.finish()
