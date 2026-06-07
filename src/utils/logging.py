"""Logging configuration for Deepfake Shield."""

from __future__ import annotations

import logging
import sys
from typing import Optional


def setup_logging(level: int = logging.INFO, name: Optional[str] = None) -> logging.Logger:
    """Configure and return a module logger with a consistent format.

    Args:
        level: Logging level (default INFO).
        name: Logger name; uses root logger when None.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
