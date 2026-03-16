"""
metrics.py — All classification metrics from classification_metrics.pdf.

In segmentation, EVERY PIXEL is a binary classification:
    - Positive class (1) = wound tissue
    - Negative class (0) = background / healthy skin

All metrics from the PDF therefore apply directly at the pixel level.

PDF metrics implemented here:
    #1  Confusion Matrix (TP, TN, FP, FN)
    #2  False Positive Rate  (FPR) — Type I error
    #3  False Negative Rate  (FNR) — Type II error
    #4  Specificity / True Negative Rate (TNR)
    #7  Recall / Sensitivity / True Positive Rate (TPR)
    #8  Precision / Positive Predictive Value (PPV)
    #9  Accuracy
    #11 F1 Score = Dice coefficient (primary metric)
    #13 Cohen Kappa
    #14 Matthews Correlation Coefficient (MCC)
    #16 ROC AUC
    #18 PR AUC / Average Precision

Segmentation-specific:
    IoU / Jaccard Index  ← standard benchmark for segmentation (target ≥ 0.75)
    Dice Coefficient     ← = F1 at pixel level (target ≥ 0.80)
"""

from typing import Dict, Tuple
import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    cohen_kappa_score,
)

from src.config import CFG


def compute_metrics(
    y_true_tensor: torch.Tensor,
    y_pred_tensor: torch.Tensor,
    threshold: float = CFG.THRESHOLD,
) -> Dict[str, float]:
    """
    Compute all classification metrics from the PDF on segmentation outputs.

    Args:
        y_true_tensor : Ground truth masks  [N, 1, H, W] — values 0.0 or 1.0.
        y_pred_tensor : Model probabilities [N, 1, H, W] — values in [0, 1] (after sigmoid).
        threshold     : Decision boundary for converting probs → binary predictions.

    Returns:
        Dictionary with all metric names and their float values.
    """
    # Move to CPU and flatten everything to 1D
    y_true = y_true_tensor.cpu().numpy().flatten().astype(int)
    y_prob = y_pred_tensor.cpu().numpy().flatten().astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    # ── Confusion matrix (PDF #1) ─────────────────────────────────────────
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # ── Per-formula metrics (matching PDF exactly) ────────────────────────

    # PDF #2 — False Positive Rate | Type I error
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # PDF #3 — False Negative Rate | Type II error
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    # PDF #4 — Specificity = True Negative Rate
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # PDF #7 — Recall = Sensitivity = True Positive Rate
    recall = recall_score(y_true, y_pred, zero_division=0)

    # PDF #8 — Precision = Positive Predictive Value
    precision = precision_score(y_true, y_pred, zero_division=0)

    # PDF #9 — Accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # PDF #11 — F1 Score (= Dice at pixel level)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # PDF #13 — Cohen Kappa
    kappa = cohen_kappa_score(y_true, y_pred)

    # PDF #14 — Matthews Correlation Coefficient
    mcc = matthews_corrcoef(y_true, y_pred)

    # PDF #16 — ROC AUC
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.0   # only one class present in batch

    # PDF #18 — PR AUC / Average Precision
    pr_auc = average_precision_score(y_true, y_prob)

    # ── Segmentation-specific metrics ────────────────────────────────────
    intersection = np.logical_and(y_true, y_pred).sum()
    union        = np.logical_or(y_true, y_pred).sum()

    iou  = float(intersection / union) if union > 0 else 0.0
    dice = float(2 * intersection / (y_true.sum() + y_pred.sum())) if (y_true.sum() + y_pred.sum()) > 0 else 0.0

    return {
        # Confusion matrix values
        "TP":          int(tp),
        "TN":          int(tn),
        "FP":          int(fp),
        "FN":          int(fn),
        # PDF metrics
        "Accuracy":    round(float(accuracy),    4),
        "Precision":   round(float(precision),   4),
        "Recall":      round(float(recall),      4),
        "Specificity": round(float(specificity), 4),
        "F1":          round(float(f1),          4),
        "Kappa":       round(float(kappa),       4),
        "MCC":         round(float(mcc),         4),
        "ROC_AUC":     round(float(roc_auc),     4),
        "PR_AUC":      round(float(pr_auc),      4),
        "FPR":         round(float(fpr),         4),
        "FNR":         round(float(fnr),         4),
        # Segmentation metrics
        "IoU":         round(iou,  4),
        "Dice":        round(dice, 4),
    }


def print_metrics(metrics: Dict[str, float], epoch: int = None) -> None:
    """
    Pretty-print all metrics to the console.

    Args:
        metrics : Dictionary returned by compute_metrics().
        epoch   : Optional epoch number to display in the header.
    """
    header = f"EPOCH {epoch} — " if epoch is not None else ""
    meets_iou  = "✅" if metrics["IoU"]     >= CFG.TARGET_IOU     else "❌"
    meets_dice = "✅" if metrics["Dice"]    >= CFG.TARGET_DICE    else "❌"
    meets_auc  = "✅" if metrics["ROC_AUC"] >= CFG.TARGET_ROC_AUC else "❌"

    print(f"\n{'='*58}")
    print(f"  {header}VERTINIMO METRIKOS")
    print(f"{'='*58}")
    print(f"  Confusion matrix:  TP={metrics['TP']}  TN={metrics['TN']}  "
          f"FP={metrics['FP']}  FN={metrics['FN']}")
    print(f"  {'─'*54}")
    print(f"  Accuracy      : {metrics['Accuracy']:.4f}")
    print(f"  Precision     : {metrics['Precision']:.4f}   (PDF #8)")
    print(f"  Recall        : {metrics['Recall']:.4f}   (PDF #7  — sensitivity)")
    print(f"  Specificity   : {metrics['Specificity']:.4f}   (PDF #4)")
    print(f"  FPR           : {metrics['FPR']:.4f}   (PDF #2  — Type I error)")
    print(f"  FNR           : {metrics['FNR']:.4f}   (PDF #3  — Type II error)")
    print(f"  {'─'*54}")
    print(f"  F1 / Dice     : {metrics['F1']:.4f}   (PDF #11) {meets_dice}")
    print(f"  Kappa         : {metrics['Kappa']:.4f}   (PDF #13)")
    print(f"  MCC           : {metrics['MCC']:.4f}   (PDF #14)")
    print(f"  ROC AUC       : {metrics['ROC_AUC']:.4f}   (PDF #16) {meets_auc}")
    print(f"  PR AUC        : {metrics['PR_AUC']:.4f}   (PDF #18)")
    print(f"  {'─'*54}")
    print(f"  IoU (Jaccard) : {metrics['IoU']:.4f}   ← segmentavimo etalonas {meets_iou}")
    print(f"  Dice coeff.   : {metrics['Dice']:.4f}   ← = F1 pikselių lygyje")
    print(f"{'='*58}")


def collect_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device = CFG.DEVICE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run model over an entire DataLoader and collect all predictions.

    Args:
        model  : Trained UNet model (in eval mode).
        loader : DataLoader (val or test).
        device : Torch device.

    Returns:
        Tuple of (all_masks, all_probs) — both as CPU tensors.
    """
    all_masks, all_probs = [], []

    model.eval()
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            logits = model(images)
            probs  = torch.sigmoid(logits)

            all_probs.append(probs.cpu())
            all_masks.append(masks.cpu())

    return torch.cat(all_masks), torch.cat(all_probs)
