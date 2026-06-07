"""Configuration loading and reproducibility utilities."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch
import yaml

from src.utils.device import get_device
from src.utils.logging import setup_logging

__all__ = ["load_config", "set_seed", "get_device", "setup_logging"]


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML config.

    Returns:
        Parsed configuration dictionary.
    """
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
