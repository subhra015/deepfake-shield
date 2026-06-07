"""Custom training loop with AMP, gradient clipping, and AUC-based checkpointing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import compute_auc
from src.training.callbacks import EarlyStopping, ModelCheckpoint, WandbLogger
from src.utils.device import get_device
from src.utils.logging import setup_logging

logger = setup_logging(name="deepfake.trainer")


class Trainer:
    """Custom trainer with mixed precision and validation AUC checkpointing.

    Args:
        model: Classification model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer instance.
        scheduler: Optional LR scheduler.
        config: Full project configuration.
        criterion: Loss function.
        device: Torch device override.
        checkpoint_callback: ModelCheckpoint instance.
        early_stopping: EarlyStopping instance.
        wandb_logger: Optional W&B logger.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler],
        config: Dict[str, Any],
        criterion: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        checkpoint_callback: Optional[ModelCheckpoint] = None,
        early_stopping: Optional[EarlyStopping] = None,
        wandb_logger: Optional[WandbLogger] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.criterion = criterion or nn.CrossEntropyLoss()

        train_cfg = config.get("training", config)
        self.epochs = int(train_cfg.get("epochs", 50))
        self.mixed_precision = bool(train_cfg.get("mixed_precision", True))
        self.grad_clip = float(train_cfg.get("grad_clip", 1.0))
        log_cfg = config.get("logging", {})
        self.log_interval = int(log_cfg.get("log_interval", 50))

        preferred = str(config.get("device", "cuda"))
        self.device = device or get_device(preferred)
        self.model.to(self.device)

        checkpoint_dir = Path(config.get("checkpoint_dir", "models/checkpoints"))
        self.checkpoint = checkpoint_callback or ModelCheckpoint(
            save_dir=checkpoint_dir,
            monitor="val_auc",
            mode="max",
        )
        self.early_stopping = early_stopping or EarlyStopping(
            monitor="val_auc",
            patience=int(train_cfg.get("patience", 10)),
            mode="max",
        )
        self.wandb = wandb_logger
        self.scaler = GradScaler(
            enabled=self.mixed_precision and self.device.type == "cuda"
        )
        self.history: List[Dict[str, float]] = []
        self.best_auc = 0.0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch with AMP and gradient clipping.

        Args:
            epoch: Current epoch number (1-indexed).

        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(self.train_loader, desc=f"Train Epoch {epoch}", leave=False)
        for step, batch in enumerate(progress):
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.scaler.is_enabled()):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()

            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

            progress.set_postfix(loss=loss.item(), acc=correct / max(total, 1))

            if self.wandb and step % self.log_interval == 0:
                self.wandb.log_metrics(
                    {
                        "train/loss_step": loss.item(),
                        "train/acc_step": correct / max(total, 1),
                    },
                    step=epoch * len(self.train_loader) + step,
                )

        if self.scheduler is not None:
            self.scheduler.step()

        return {
            "train_loss": total_loss / max(len(self.train_loader), 1),
            "train_acc": correct / max(total, 1),
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model and compute loss, accuracy, and AUC.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_labels: List[int] = []
        all_scores: List[float] = []

        for batch in tqdm(self.val_loader, desc="Validate", leave=False):
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            with autocast(enabled=self.scaler.is_enabled()):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)
            fake_scores = probs[:, 1].cpu().numpy()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

            all_labels.extend(labels.cpu().tolist())
            all_scores.extend(fake_scores.tolist())

        val_auc = compute_auc(np.array(all_labels), np.array(all_scores))
        return {
            "val_loss": total_loss / max(len(self.val_loader), 1),
            "val_acc": correct / max(total, 1),
            "val_auc": val_auc,
        }

    def fit(self) -> Dict[str, Any]:
        """Run the full training loop.

        Returns:
            Dictionary with best_auc and training history.
        """
        for epoch in range(1, self.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            metrics = {**train_metrics, **val_metrics, "epoch": float(epoch)}
            self.history.append(metrics)

            logger.info(
                "Epoch %d/%d | loss=%.4f acc=%.4f | val_loss=%.4f val_acc=%.4f val_auc=%.4f",
                epoch,
                self.epochs,
                metrics["train_loss"],
                metrics["train_acc"],
                metrics["val_loss"],
                metrics["val_acc"],
                metrics["val_auc"],
            )

            if self.wandb:
                wandb_metrics = {
                    k.replace("_", "/", 1) if "/" not in k else k: v
                    for k, v in metrics.items()
                }
                self.wandb.log_metrics(wandb_metrics, step=epoch)

            if metrics["val_auc"] > self.best_auc:
                self.best_auc = metrics["val_auc"]

            self.checkpoint.step(
                self.model,
                metrics,
                epoch,
                extra={"config": self.config},
            )

            if self.early_stopping.step(metrics):
                logger.info("Early stopping at epoch %d", epoch)
                break

        return {"best_auc": self.best_auc, "history": self.history}
