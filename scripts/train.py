"""Training CLI entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.data.datamodule import DeepfakeDataModule
from src.models.classifier import DeepfakeClassifier
from src.training.callbacks import WandbLogger
from src.training.losses import LabelSmoothingCrossEntropy
from src.training.trainer import Trainer
from src.utils.config import load_config, set_seed
from src.utils.logging import setup_logging

logger = setup_logging(name="deepfake.train")


def build_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """Build optimizer from config."""
    train_cfg = cfg.get("training", {})
    optim_cfg = cfg.get("optimizer", {})
    lr = float(train_cfg.get("lr", 1e-4))
    wd = float(train_cfg.get("weight_decay", 1e-2))

    name = str(optim_cfg.get("name", "adamw")).lower()
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: dict
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Build LR scheduler from config."""
    optim_cfg = cfg.get("optimizer", {})
    name = str(optim_cfg.get("scheduler", "cosine")).lower()
    epochs = int(cfg.get("training", {}).get("epochs", 50))

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name in ("none", "null", ""):
        return None
    raise ValueError(f"Unsupported scheduler: {name}")


def main() -> None:
    """Run training pipeline."""
    parser = argparse.ArgumentParser(description="Train Deepfake Shield model")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    dm = DeepfakeDataModule(cfg)
    dm.setup()

    model_cfg = cfg.get("model", {})
    model = DeepfakeClassifier(
        backbone_name=model_cfg.get("backbone", "convnext_tiny"),
        num_classes=int(model_cfg.get("num_classes", 2)),
        dropout=float(model_cfg.get("dropout", 0.3)),
        use_attention=bool(model_cfg.get("use_attention", True)),
        pretrained=bool(model_cfg.get("pretrained", True)),
    )

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    criterion = LabelSmoothingCrossEntropy(
        smoothing=float(cfg.get("training", {}).get("label_smoothing", 0.1))
    )

    wandb_logger = None
    log_cfg = cfg.get("logging", {})
    if bool(log_cfg.get("use_wandb", False)):
        wandb_logger = WandbLogger(
            project=str(log_cfg.get("project_name", cfg.get("project_name", "deepfake-shield"))),
            config=cfg,
            enabled=True,
        )

    trainer = Trainer(
        model=model,
        train_loader=dm.train_dataloader(),
        val_loader=dm.val_dataloader(),
        optimizer=optimizer,
        scheduler=scheduler,
        config=cfg,
        criterion=criterion,
        wandb_logger=wandb_logger,
    )
    result = trainer.fit()

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    final_path = models_dir / "final_model.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "best_auc": result["best_auc"],
        },
        final_path,
    )

    if wandb_logger:
        wandb_logger.finish()

    logger.info("Saved final model to %s (best_auc=%.4f)", final_path.resolve(), result["best_auc"])


if __name__ == "__main__":
    main()
