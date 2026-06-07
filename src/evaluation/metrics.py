"""Evaluation metrics and visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, f1_score, roc_curve

from src.utils.logging import setup_logging

logger = setup_logging(name="deepfake.metrics")


def compute_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute ROC-AUC score.

    Args:
        y_true: Binary ground-truth labels.
        y_scores: Predicted scores for the positive (fake) class.

    Returns:
        AUC value in [0, 1].
    """
    labels = np.asarray(y_true).astype(int)
    scores = np.asarray(y_scores).astype(float)
    if len(np.unique(labels)) < 2:
        return 0.5
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(auc(fpr, tpr))


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute Equal Error Rate (EER).

    Args:
        y_true: Binary ground-truth labels.
        y_scores: Predicted scores for the positive class.

    Returns:
        EER as a rate in [0, 1].
    """
    labels = np.asarray(y_true).astype(int)
    scores = np.asarray(y_scores).astype(float)
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)


def compute_f1_at_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute F1 score at a fixed decision threshold.

    Args:
        y_true: Binary ground-truth labels.
        y_scores: Predicted scores for the positive class.
        threshold: Classification threshold.

    Returns:
        F1 score.
    """
    labels = np.asarray(y_true).astype(int)
    predictions = (np.asarray(y_scores) >= threshold).astype(int)
    return float(f1_score(labels, predictions, zero_division=0))


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: Union[str, Path],
) -> Path:
    """Plot and save ROC curve.

    Args:
        y_true: Binary ground-truth labels.
        y_scores: Predicted scores.
        save_path: Output file path.

    Returns:
        Path to saved figure.
    """
    labels = np.asarray(y_true).astype(int)
    scores = np.asarray(y_scores).astype(float)
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = compute_auc(labels, scores)

    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved ROC curve to %s", output)
    return output


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Union[str, Path],
    labels: Optional[List[str]] = None,
) -> Path:
    """Plot and save confusion matrix heatmap.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        save_path: Output file path.
        labels: Class label names.

    Returns:
        Path to saved figure.
    """
    class_labels = labels or ["real", "fake"]
    matrix = confusion_matrix(np.asarray(y_true), np.asarray(y_pred))
    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved confusion matrix to %s", output)
    return output
