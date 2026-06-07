"""Evaluation metrics and visualization."""

from src.evaluation.metrics import (
    compute_auc,
    compute_eer,
    compute_f1_at_threshold,
    plot_confusion_matrix,
    plot_roc_curve,
)
from src.evaluation.explainability import GradCAM
from src.evaluation.robustness import (
    run_robustness_suite,
    test_gaussian_blur,
    test_gaussian_noise,
    test_jpeg_compression,
    test_resize_distortion,
)

__all__ = [
    "compute_auc",
    "compute_eer",
    "compute_f1_at_threshold",
    "plot_roc_curve",
    "plot_confusion_matrix",
    "GradCAM",
    "run_robustness_suite",
    "test_jpeg_compression",
    "test_gaussian_noise",
    "test_gaussian_blur",
    "test_resize_distortion",
]
