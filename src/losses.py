"""
losses.py — Loss functions for wound segmentation.

Binary segmentation losses:
    DiceLoss      — 1 - Dice coefficient
    BCEDiceLoss   — 0.5 * BCE + 0.5 * Dice  (default training loss)

Multi-class tissue segmentation losses:
    FocalLoss          — down-weights easy examples; strong against class imbalance
    MultiClassDiceLoss — mean Dice over all tissue classes
    TissueLoss         — 0.5 * Focal + 0.5 * MultiClassDice  (tissue training loss)
"""

from typing import Optional

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


# ── Multi-class tissue losses ─────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma > 0 reduces the relative loss for well-classified examples,
    forcing the model to focus on hard, misclassified tissue pixels.

    Args:
        gamma        : Focusing parameter (default 2.0 from Lin et al. 2017).
        weight       : Per-class weight tensor [C] — handles class frequency imbalance.
        ignore_index : Class index to exclude from loss (e.g. unknown pixels).
    """

    def __init__(
        self,
        gamma:        float             = 2.0,
        weight:       Optional[torch.Tensor] = None,
        ignore_index: int               = 255,
    ) -> None:
        super().__init__()
        self.gamma        = gamma
        self.weight       = weight
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : [B, C, H, W] — raw logits (no softmax).
            targets : [B, H, W]    — class indices (long), IGNORE_INDEX excluded.

        Returns:
            Scalar focal loss.
        """
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight       = self.weight.to(logits.device) if self.weight is not None else None,
            ignore_index = self.ignore_index,
            reduction    = "none",
        )
        pt      = torch.exp(-ce_loss)                       # probability of correct class
        focal   = (1 - pt) ** self.gamma * ce_loss

        # Only average over non-ignored pixels
        valid_mask = targets != self.ignore_index
        if valid_mask.sum() == 0:
            return focal.sum() * 0.0
        return focal[valid_mask].mean()


class MultiClassDiceLoss(nn.Module):
    """
    Mean Dice Loss over all tissue classes (macro-average).

    For each class c: Dice_c = 2 * |pred_c ∩ gt_c| / (|pred_c| + |gt_c|)
    Loss = 1 - mean(Dice_c)

    Ignored pixels (IGNORE_INDEX) are zero-ed out before computation.

    Args:
        num_classes  : Number of tissue classes.
        smooth       : Laplace smoothing to avoid division by zero.
        ignore_index : Class index to exclude from loss.
    """

    def __init__(
        self,
        num_classes:  int   = 4,
        smooth:       float = 1e-6,
        ignore_index: int   = 255,
    ) -> None:
        super().__init__()
        self.num_classes  = num_classes
        self.smooth       = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : [B, C, H, W]
            targets : [B, H, W] class indices

        Returns:
            Scalar Dice loss.
        """
        probs = F.softmax(logits, dim=1)              # [B, C, H, W]

        # Build valid pixel mask
        valid = (targets != self.ignore_index).float()     # [B, H, W]

        # One-hot encode targets; set IGNORE pixels to all-zero
        safe_targets = targets.clone()
        safe_targets[targets == self.ignore_index] = 0
        one_hot = F.one_hot(safe_targets, num_classes=self.num_classes)   # [B, H, W, C]
        one_hot = one_hot.permute(0, 3, 1, 2).float()                     # [B, C, H, W]

        # Apply valid mask per channel
        valid_expanded = valid.unsqueeze(1)               # [B, 1, H, W]
        probs    = probs    * valid_expanded
        one_hot  = one_hot  * valid_expanded

        # Flatten batch+spatial per class: [C, B*H*W]
        probs   = probs.reshape(self.num_classes, -1)
        one_hot = one_hot.reshape(self.num_classes, -1)

        intersection = (probs * one_hot).sum(dim=1)
        union        = probs.sum(dim=1) + one_hot.sum(dim=1)
        dice_per_cls = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice_per_cls.mean()


class TissueLoss(nn.Module):
    """
    Combined Focal + Multi-class Dice loss for tissue segmentation.

    Total = alpha * Focal + beta * MultiClassDice

    Args:
        gamma        : Focal loss gamma.
        alpha        : Weight for focal component (default 0.5).
        beta         : Weight for Dice component (default 0.5).
        class_weights: Optional per-class frequency weights for focal loss.
        ignore_index : Class index excluded from both losses.
    """

    def __init__(
        self,
        gamma:         float                    = 2.0,
        alpha:         float                    = 0.5,
        beta:          float                    = 0.5,
        num_classes:   int                      = 4,
        class_weights: Optional[torch.Tensor]   = None,
        ignore_index:  int                      = 255,
    ) -> None:
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.focal  = FocalLoss(
            gamma        = gamma,
            weight       = class_weights,
            ignore_index = ignore_index,
        )
        self.dice   = MultiClassDiceLoss(
            num_classes  = num_classes,
            ignore_index = ignore_index,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : [B, C, H, W]
            targets : [B, H, W]

        Returns:
            Scalar combined loss.
        """
        return self.alpha * self.focal(logits, targets) + self.beta * self.dice(logits, targets)
