"""Custom exceptions for Deepfake Shield."""

from __future__ import annotations


class DeepfakeShieldError(Exception):
    """Base exception for all Deepfake Shield errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ModelLoadError(DeepfakeShieldError):
    """Raised when a model checkpoint cannot be loaded."""


class InferenceError(DeepfakeShieldError):
    """Raised when inference fails on valid input."""


class DataError(DeepfakeShieldError):
    """Raised when dataset or preprocessing operations fail."""
