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
    BEST_MODEL_PATH    = CHECKPOINTS_DIR / "best_model.pth"
    LAST_MODEL_PATH    = CHECKPOINTS_DIR / "last_model.pth"
    METRICS_CSV        = METRICS_DIR / "history.csv"

    # ── Tissue type classification (Feature 1) ────────────────────────────
    # Colour convention for tissue mask PNGs (RGB tuples):
    #   Granulation     (255,   0,   0)  → class 0  healthy red tissue
    #   Slough          (255, 255,   0)  → class 1  yellow necrotic tissue
    #   Eschar          ( 50,  50,  50)  → class 2  dark/black necrotic tissue
    #   Epithelialisation (255, 192, 203) → class 3  light pink healing skin
    TISSUE_CLASSES     = 4
    TISSUE_NAMES       = ["granulation", "slough", "eschar", "epithelialisation"]
    TISSUE_COLOURS     = {
        0: (255,   0,   0),   # granulation     — red
        1: (255, 255,   0),   # slough          — yellow
        2: ( 50,  50,  50),   # eschar          — near-black
        3: (255, 192, 203),   # epithelialisation — pink
    }
    TISSUE_MASKS_DIR   = DATA_DIR / "tissue_masks"
    TISSUE_MODEL_PATH  = CHECKPOINTS_DIR / "tissue_model.pth"
    TISSUE_METRICS_CSV = METRICS_DIR / "tissue_history.csv"

    # Focal loss gamma for tissue training (handles class imbalance)
    TISSUE_FOCAL_GAMMA = 2.0
    TISSUE_EPOCHS      = 30
    TISSUE_LR          = 5e-5   # lower LR — encoder is frozen

    # ── Healing Trajectory Prediction (Feature 3) ─────────────────────────
    HEALING_THRESHOLD   = 0.05   # weekly area decrease > 5% of mean → "healing"
    WORSENING_THRESHOLD = 0.05   # weekly area increase > 5% of mean → "worsening"

    # ── Wound Type Classification (Feature 4) ─────────────────────────────
    WOUND_TYPES       = [
        "diabetic_foot_ulcer",
        "venous_leg_ulcer",
        "pressure_injury",
        "surgical",
        "burn",
    ]
    WOUND_TYPES_DIR    = DATA_DIR / "wound_types"     # one sub-dir per class
    CLASSIFIER_PATH    = CHECKPOINTS_DIR / "classifier_model.pth"
    CLASSIFIER_METRICS = METRICS_DIR / "classifier_history.csv"
    CLASSIFIER_EPOCHS  = 20
    CLASSIFIER_LR      = 1e-4


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
        CFG.TISSUE_MASKS_DIR,
        CFG.WOUND_TYPES_DIR,
        *[CFG.WOUND_TYPES_DIR / wt for wt in CFG.WOUND_TYPES],
    ]:
        d.mkdir(parents=True, exist_ok=True)
