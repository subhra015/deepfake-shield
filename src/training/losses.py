"""Classification losses with label smoothing and focal loss."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing.

    Args:
        smoothing: Label smoothing factor in [0, 1).
    """

    def __init__(self, smoothing: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute smoothed cross-entropy loss.

        Args:
            logits: Model logits (B, C).
            targets: Ground-truth class indices (B,).

        Returns:
            Scalar loss tensor.
        """
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification.

    Args:
        gamma: Focusing parameter.
        alpha: Class balance weight for positive class.
    """

    def __init__(self, gamma: float = 2.0, alpha: float | None = 0.25) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Model logits (B, C).
            targets: Ground-truth class indices (B,).

        Returns:
            Scalar loss tensor.
        """
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            alpha_t = torch.where(
                targets == 1,
                torch.tensor(self.alpha, device=logits.device, dtype=logits.dtype),
                torch.tensor(1.0 - self.alpha, device=logits.device, dtype=logits.dtype),
            )
            focal = alpha_t * focal
        return focal.mean()
