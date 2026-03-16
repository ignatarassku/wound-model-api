"""
losses.py — Loss functions for wound segmentation.

For binary segmentation with class imbalance (wounds are much smaller than background),
pure BCE loss is insufficient. We combine BCE + Dice loss.

Dice Loss directly optimizes the Dice coefficient (= F1 score),
which is the primary metric for segmentation tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss: 1 - Dice coefficient.

    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Loss = 1 - Dice   (minimize → maximize Dice)

    Smooth factor prevents division by zero on empty masks.
    """

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : Raw model output [B, 1, H, W] — before sigmoid.
            targets : Binary ground truth [B, 1, H, W] — values 0.0 or 1.0.

        Returns:
            Scalar Dice loss.
        """
        probs = torch.sigmoid(logits)

        # Flatten spatial dimensions for easy computation
        probs   = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)

        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """
    Combined BCE + Dice Loss.

    BCE  handles per-pixel classification accuracy.
    Dice directly optimizes the overlap metric used for evaluation.

    Total = alpha * BCE + beta * Dice
    Default: equal weighting (0.5 + 0.5).
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5) -> None:
        super().__init__()
        self.alpha    = alpha
        self.beta     = beta
        self.bce      = nn.BCEWithLogitsLoss()   # numerically stable (includes sigmoid)
        self.dice     = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : Raw model output [B, 1, H, W].
            targets : Binary ground truth [B, 1, H, W].

        Returns:
            Scalar combined loss.
        """
        bce_loss  = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        return self.alpha * bce_loss + self.beta * dice_loss
