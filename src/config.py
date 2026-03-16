"""
config.py — Central configuration for the Wound Segmentation project.

ALL hyperparameters, paths, and constants live here.
Never hardcode values in other files — import from here instead.

Usage:
    from src.config import CFG
    print(CFG.LEARNING_RATE)
"""

from pathlib import Path
import torch


class CFG:
    # ── Project paths ──────────────────────────────────────────────────────
    ROOT_DIR       = Path(__file__).parent.parent          # project root
    DATA_DIR       = ROOT_DIR / "data"
    IMAGES_DIR     = DATA_DIR / "images"
    MASKS_DIR      = DATA_DIR / "masks"
    RAW_DIR        = DATA_DIR / "raw"                      # original downloads

    CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
    RESULTS_DIR     = ROOT_DIR / "results"
    METRICS_DIR     = RESULTS_DIR / "metrics"
    PLOTS_DIR       = RESULTS_DIR / "plots"
    PREDICTIONS_DIR = RESULTS_DIR / "predictions"

    # ── Model ──────────────────────────────────────────────────────────────
    IMG_SIZE        = 256          # input image size (square)
    IN_CHANNELS     = 3            # RGB
    OUT_CHANNELS    = 1            # binary segmentation
    ENCODER         = "resnet34"   # pretrained backbone
    ENCODER_WEIGHTS = "imagenet"   # pretrained weights source

    # ── Training ───────────────────────────────────────────────────────────
    EPOCHS          = 50
    BATCH_SIZE      = 8
    LEARNING_RATE   = 1e-4
    WEIGHT_DECAY    = 1e-5
    PATIENCE        = 8            # early stopping & scheduler patience

    # ── Data split ─────────────────────────────────────────────────────────
    TRAIN_SPLIT     = 0.70
    VAL_SPLIT       = 0.15
    TEST_SPLIT      = 0.15
    RANDOM_SEED     = 42

    # ── Prediction ─────────────────────────────────────────────────────────
    THRESHOLD       = 0.5          # sigmoid output threshold for binary mask

    # ── Normalization (ImageNet stats) ────────────────────────────────────
    NORM_MEAN       = [0.485, 0.456, 0.406]
    NORM_STD        = [0.229, 0.224, 0.225]

    # ── Device ─────────────────────────────────────────────────────────────
    DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Metrics targets (thesis benchmarks) ───────────────────────────────
    TARGET_IOU      = 0.75
    TARGET_DICE     = 0.80
    TARGET_ROC_AUC  = 0.90

    # ── Checkpoint filenames ───────────────────────────────────────────────
    BEST_MODEL_PATH = CHECKPOINTS_DIR / "best_model.pth"
    LAST_MODEL_PATH = CHECKPOINTS_DIR / "last_model.pth"
    METRICS_CSV     = METRICS_DIR / "history.csv"


def ensure_dirs() -> None:
    """Create all output directories if they don't exist yet."""
    for d in [
        CFG.CHECKPOINTS_DIR,
        CFG.METRICS_DIR,
        CFG.PLOTS_DIR,
        CFG.PREDICTIONS_DIR,
        CFG.IMAGES_DIR,
        CFG.MASKS_DIR,
        CFG.RAW_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
