"""
evaluate.py — Full evaluation with publication-quality plots for thesis.

Generates:
    - Confusion matrix heatmap     → results/plots/confusion_matrix.png
    - ROC curve                    → results/plots/roc_curve.png
    - Precision-Recall curve       → results/plots/pr_curve.png
    - Training history curves      → results/plots/training_history.png
    - Prediction samples grid      → results/plots/prediction_samples.png

Run from project root:
    python -m src.evaluate
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)

from src.config import CFG, ensure_dirs
from src.model import get_model
from src.dataset import get_dataloaders
from src.metrics import compute_metrics, print_metrics, collect_predictions


# ── Plot helpers ──────────────────────────────────────────────────────────

PLOT_STYLE = {
    "figure.facecolor": "white",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "font.family":      "DejaVu Sans",
}


def load_best_model() -> torch.nn.Module:
    """Load the best saved checkpoint."""
    model      = get_model()
    checkpoint = torch.load(CFG.BEST_MODEL_PATH, map_location=CFG.DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"[Evaluate] Loaded checkpoint from epoch {checkpoint['epoch']}  |  IoU = {checkpoint['best_iou']:.4f}")
    return model


# ── 1. Confusion matrix ───────────────────────────────────────────────────

def plot_confusion_matrix(metrics: dict, save_path: Path = CFG.PLOTS_DIR / "confusion_matrix.png") -> None:
    """
    Plot and save confusion matrix heatmap (PDF #1).
    """
    tp, tn, fp, fn = metrics["TP"], metrics["TN"], metrics["FP"], metrics["FN"]
    cm = np.array([[tn, fp], [fn, tp]])

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Predicted: Background", "Predicted: Wound"],
            yticklabels=["Actual: Background",    "Actual: Wound"],
            ax=ax, linewidths=0.5, linecolor="white",
            annot_kws={"size": 14, "weight": "bold"},
        )
        ax.set_title("Confusion Matrix (pixel-level)", fontsize=14, pad=12)
        ax.set_ylabel("Actual", fontsize=11)
        ax.set_xlabel("Predicted", fontsize=11)

        # Add metric annotations
        fig.text(0.5, -0.02,
                 f"Precision={metrics['Precision']:.3f}  |  "
                 f"Recall={metrics['Recall']:.3f}  |  "
                 f"F1/Dice={metrics['F1']:.3f}  |  "
                 f"MCC={metrics['MCC']:.3f}",
                 ha="center", fontsize=10, color="gray")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    print(f"  [Saved] {save_path}")


# ── 2. ROC curve ──────────────────────────────────────────────────────────

def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray,
                   save_path: Path = CFG.PLOTS_DIR / "roc_curve.png") -> None:
    """
    Plot ROC curve with AUC score (PDF #15, #16).
    """
    fpr_vals, tpr_vals, _ = roc_curve(y_true, y_prob)
    roc_auc_val            = auc(fpr_vals, tpr_vals)

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr_vals, tpr_vals, color="#2E8B74", lw=2,
                label=f"U-Net  (AUC = {roc_auc_val:.4f})")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random classifier")
        ax.fill_between(fpr_vals, tpr_vals, alpha=0.08, color="#2E8B74")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.02])
        ax.set_xlabel("False Positive Rate (FPR)", fontsize=12)
        ax.set_ylabel("True Positive Rate (TPR / Recall)", fontsize=12)
        ax.set_title("ROC Curve — Wound Segmentation", fontsize=14)
        ax.legend(loc="lower right", fontsize=11)
        ax.axhline(y=CFG.TARGET_ROC_AUC, color="orange", linestyle=":", lw=1,
                   label=f"Target AUC ≥ {CFG.TARGET_ROC_AUC}")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    print(f"  [Saved] {save_path}")


# ── 3. Precision-Recall curve ─────────────────────────────────────────────

def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray,
                  save_path: Path = CFG.PLOTS_DIR / "pr_curve.png") -> None:
    """
    Plot Precision-Recall curve with AP score (PDF #17, #18).
    Important when classes are imbalanced (wounds smaller than background).
    """
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(recall_vals, precision_vals, color="#1A6B5A", lw=2,
                label=f"U-Net  (AP = {ap:.4f})")
        ax.fill_between(recall_vals, precision_vals, alpha=0.08, color="#1A6B5A")

        baseline = y_true.mean()
        ax.axhline(y=baseline, color="gray", linestyle="--", lw=1,
                   label=f"Random baseline (= {baseline:.3f})")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.02])
        ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
        ax.set_ylabel("Precision (PPV)", fontsize=12)
        ax.set_title("Precision-Recall Curve — Wound Segmentation", fontsize=14)
        ax.legend(loc="upper right", fontsize=11)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    print(f"  [Saved] {save_path}")


# ── 4. Training history ───────────────────────────────────────────────────

def plot_training_history(csv_path: Path = CFG.METRICS_CSV,
                          save_path: Path = CFG.PLOTS_DIR / "training_history.png") -> None:
    """
    Plot IoU, Dice/F1, and loss over training epochs.
    """
    if not csv_path.exists():
        print(f"  [Skip] No CSV found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # IoU
        axes[0].plot(df["epoch"], df["IoU"], color="#2E8B74", lw=2)
        axes[0].axhline(CFG.TARGET_IOU, color="orange", linestyle="--", lw=1,
                        label=f"Target ≥ {CFG.TARGET_IOU}")
        axes[0].set_title("IoU per epoch", fontsize=13)
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("IoU")
        axes[0].legend(fontsize=9)

        # Dice / F1
        axes[1].plot(df["epoch"], df["Dice"], color="#1A6B5A", lw=2, label="Dice")
        axes[1].plot(df["epoch"], df["Recall"], color="#B5860D", lw=2, linestyle="--", label="Recall")
        axes[1].plot(df["epoch"], df["Precision"], color="#C0392B", lw=2, linestyle=":", label="Precision")
        axes[1].axhline(CFG.TARGET_DICE, color="orange", linestyle="--", lw=1,
                        label=f"Target Dice ≥ {CFG.TARGET_DICE}")
        axes[1].set_title("Dice / Recall / Precision", fontsize=13)
        axes[1].set_xlabel("Epoch"); axes[1].legend(fontsize=9)

        # Loss
        axes[2].plot(df["epoch"], df["train_loss"], color="#666666", lw=2)
        axes[2].set_title("Training Loss", fontsize=13)
        axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("BCE + Dice Loss")

        plt.suptitle("Training History — Wound Segmentation U-Net", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    print(f"  [Saved] {save_path}")


# ── 5. Prediction samples grid ────────────────────────────────────────────

def plot_prediction_samples(model: torch.nn.Module,
                             loader: torch.utils.data.DataLoader,
                             n_samples: int = 6,
                             save_path: Path = CFG.PLOTS_DIR / "prediction_samples.png") -> None:
    """
    Visual grid: original image | ground truth mask | predicted mask | overlay.
    """
    model.eval()
    images_list, masks_list, preds_list = [], [], []

    with torch.no_grad():
        for images, masks in loader:
            logits = model(images.to(CFG.DEVICE))
            preds  = torch.sigmoid(logits).cpu()
            images_list.append(images)
            masks_list.append(masks)
            preds_list.append(preds)
            if sum(len(x) for x in images_list) >= n_samples:
                break

    images_all = torch.cat(images_list)[:n_samples]
    masks_all  = torch.cat(masks_list)[:n_samples]
    preds_all  = torch.cat(preds_list)[:n_samples]

    # Denormalize images for display
    mean = torch.tensor(CFG.NORM_MEAN).view(3, 1, 1)
    std  = torch.tensor(CFG.NORM_STD).view(3, 1, 1)
    imgs_display = torch.clamp(images_all * std + mean, 0, 1).permute(0, 2, 3, 1).numpy()

    fig, axes = plt.subplots(n_samples, 4, figsize=(14, n_samples * 3))
    cols = ["Original", "Ground Truth", "Prediction", "Overlay"]

    for col_idx, col_name in enumerate(cols):
        axes[0, col_idx].set_title(col_name, fontsize=12, fontweight="bold")

    for i in range(n_samples):
        img  = imgs_display[i]
        gt   = masks_all[i, 0].numpy()
        pred = (preds_all[i, 0].numpy() >= CFG.THRESHOLD).astype(float)

        # Overlay: green = correct, red = FP, blue = FN
        overlay = img.copy()
        overlay[pred == 1,  1] = np.clip(overlay[pred == 1,  1] + 0.4, 0, 1)   # green FG
        overlay[(gt==1) & (pred==0), 2] = np.clip(overlay[(gt==1) & (pred==0), 2] + 0.4, 0, 1)  # blue FN

        axes[i, 0].imshow(img)
        axes[i, 1].imshow(gt,   cmap="Reds",   vmin=0, vmax=1)
        axes[i, 2].imshow(pred, cmap="Greens", vmin=0, vmax=1)
        axes[i, 3].imshow(overlay)

        for ax in axes[i]:
            ax.axis("off")

    plt.suptitle("Prediction Samples — Green=Correct  Blue=Missed", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {save_path}")


# ── Main evaluation runner ────────────────────────────────────────────────

def evaluate() -> None:
    """Run full evaluation on the test set and generate all thesis plots."""
    ensure_dirs()

    print("\n" + "="*58)
    print("  FULL EVALUATION + THESIS PLOTS")
    print("="*58)

    model = load_best_model()
    _, val_loader, test_loader = get_dataloaders()

    # Collect all test predictions
    print("\n[1/5] Computing metrics on test set...")
    all_masks, all_probs = collect_predictions(model, test_loader)
    metrics = compute_metrics(all_masks, all_probs)
    print_metrics(metrics)

    y_true = all_masks.numpy().flatten().astype(int)
    y_prob = all_probs.numpy().flatten()

    # Generate all plots
    print("\n[2/5] Plotting confusion matrix...")
    plot_confusion_matrix(metrics)

    print("[3/5] Plotting ROC curve...")
    plot_roc_curve(y_true, y_prob)

    print("[4/5] Plotting Precision-Recall curve...")
    plot_pr_curve(y_true, y_prob)

    print("[5/5] Plotting training history...")
    plot_training_history()

    print("\nPlotting prediction samples...")
    plot_prediction_samples(model, test_loader)

    print(f"\n✅ All plots saved to: {CFG.PLOTS_DIR}")


if __name__ == "__main__":
    evaluate()
