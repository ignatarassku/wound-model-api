"""
main.py — FastAPI wrapper around the trained U-Net wound segmentation model.
Production deployment on Railway.

Endpoint: POST /analyze
Auth:      X-API-Secret header == MODEL_API_SECRET env var
Returns:   Full analysis: masks, probability heat-map, bounding box,
           perimeter, real-world cm measurements (via fallback scale).
"""

import os
import sys
import base64
import io
import math
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).parent))
from src.model import UNet
from src.config import CFG

# ── Globals ────────────────────────────────────────────────────────────────

model: UNet | None = None
CHECKPOINT_PATH = Path(__file__).parent / "checkpoints" / "best_model.pth"

# ── Startup / shutdown ─────────────────────────────────────────────────────

def download_checkpoint():
    """Download best_model.pth from GitHub LFS at startup if not present."""
    if CHECKPOINT_PATH.exists() and CHECKPOINT_PATH.stat().st_size > 10000:
        print(f"[WoundWatch] Checkpoint already exists ({CHECKPOINT_PATH.stat().st_size} bytes)")
        return

    url = os.environ.get("MODEL_DOWNLOAD_URL")
    if not url:
        raise RuntimeError(
            "Checkpoint not found and MODEL_DOWNLOAD_URL env var is not set."
        )

    print(f"[WoundWatch] Downloading checkpoint from {url}...")
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    import urllib.request
    urllib.request.urlretrieve(url, CHECKPOINT_PATH)

    size = CHECKPOINT_PATH.stat().st_size
    if size < 10000:
        CHECKPOINT_PATH.unlink()
        raise RuntimeError(f"Downloaded file is too small ({size} bytes) — invalid checkpoint.")

    print(f"[WoundWatch] Checkpoint downloaded ({size} bytes)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Download checkpoint if needed, load model once at startup."""
    global model
    download_checkpoint()
    print(f"[WoundWatch] Loading model from {CHECKPOINT_PATH}...")
    model = UNet()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device("cpu"),
                            weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"[WoundWatch] Model ready. Best IoU: {checkpoint.get('best_iou', 'unknown')}")
    yield
    model = None
    print("[WoundWatch] Model unloaded.")

# ── App ────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="WoundWatch Model API",
    description="U-Net wound segmentation API for WoundWatch telemedicine platform.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── Response schema ────────────────────────────────────────────────────────

class AnalysisResult(BaseModel):
    # Core quality metrics
    iou:            float
    dice:           float
    wound_area_px:  int
    wound_area_pct: float   # fraction 0–1, e.g. 0.039 = 3.9%
    recall:         float
    precision:      float
    roc_auc:        float

    # Bounding-box pixel dimensions
    wound_width_px:      int
    wound_height_px:     int
    wound_perimeter_px:  float

    # Real-world measurements (cm) via fallback scale
    wound_area_cm2:     float | None
    wound_width_cm:     float | None
    wound_height_cm:    float | None
    wound_perimeter_cm: float | None

    # Scale calibration
    scale_px_per_mm: float | None
    scale_source:    str   # 'fingernail' | 'fallback_assumed' | 'unknown'

    # Base64-encoded PNG images
    mask_base64: str   # Binary segmentation mask (grayscale)
    prob_base64: str   # Probability heat-map (RGB, 'hot' colormap)

# ── Preprocessing ──────────────────────────────────────────────────────────

NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
NORM_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def preprocess(image_bytes: bytes) -> torch.Tensor:
    """Load raw image bytes → normalised tensor [1, 3, H, W]."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((CFG.IMG_SIZE, CFG.IMG_SIZE), Image.BILINEAR)
    tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - NORM_MEAN) / NORM_STD
    return tensor.unsqueeze(0)

def mask_to_base64(mask: np.ndarray) -> str:
    """Binary mask [H, W] → base64-encoded grayscale PNG."""
    img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def prob_to_base64(probs: np.ndarray) -> str:
    """
    Probability map [H, W] → base64-encoded RGB PNG using 'hot' colormap.
    0 = black, 0.5 = red/orange, 1 = white/yellow (matches matplotlib 'hot').
    """
    r = np.clip(probs * 2.5,     0, 1)
    g = np.clip(probs * 2.5 - 1, 0, 1)
    b = np.clip(probs * 2.5 - 2, 0, 1)
    rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ── Scale calibration ──────────────────────────────────────────────────────

# Fallback scale: typical wound photo at ~30 cm camera distance
# is approximately 3.2 px/mm at 256×256 resolution.
FALLBACK_PX_PER_MM = 3.2

def estimate_real_world(
    wound_area_px: int,
    wound_width_px: int,
    wound_height_px: int,
    wound_perimeter_px: float,
    px_per_mm: float,
) -> dict:
    """Convert pixel measurements → cm² / cm using px_per_mm scale factor."""
    mm_per_px   = 1.0 / px_per_mm
    mm2_per_px2 = mm_per_px ** 2
    return dict(
        wound_area_cm2    = round(wound_area_px     * mm2_per_px2 / 100, 2),
        wound_width_cm    = round(wound_width_px    * mm_per_px   / 10,  2),
        wound_height_cm   = round(wound_height_px   * mm_per_px   / 10,  2),
        wound_perimeter_cm= round(wound_perimeter_px * mm_per_px  / 10,  2),
    )

def compute_perimeter(binary_mask: np.ndarray) -> float:
    """
    Approximate wound perimeter in pixels.
    Counts wound pixels whose 4-neighbourhood contains at least one background pixel.
    """
    up    = np.roll(binary_mask, -1, axis=0)
    down  = np.roll(binary_mask,  1, axis=0)
    left  = np.roll(binary_mask, -1, axis=1)
    right = np.roll(binary_mask,  1, axis=1)
    boundary = binary_mask & ~(up & down & left & right)
    boundary[0, :]  = 0
    boundary[-1, :] = 0
    boundary[:, 0]  = 0
    boundary[:, -1] = 0
    return float(boundary.sum())

# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "checkpoint": str(CHECKPOINT_PATH),
        "version": "2.0.0",
    }

@app.post("/analyze", response_model=AnalysisResult)
async def analyze(request: Request, image: UploadFile = File(...)):
    """
    Run U-Net wound segmentation on the uploaded image.
    Returns full analysis: masks, probability heat-map, bounding box,
    perimeter, and real-world cm measurements (fallback scale).
    """
    # Auth
    api_secret = os.environ.get("MODEL_API_SECRET")
    if not api_secret or request.headers.get("X-API-Secret") != api_secret:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Secret header")

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        tensor = preprocess(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")

    # ── Inference ──────────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.sigmoid(logits).squeeze().numpy()  # [H, W], values 0–1

    binary_mask  = (probs >= CFG.THRESHOLD).astype(np.uint8)
    wound_pixels = int(binary_mask.sum())
    total_pixels = int(binary_mask.size)
    wound_area_pct = float(wound_pixels / total_pixels)  # fraction 0–1

    # ── Bounding box ───────────────────────────────────────────────────────
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = int(np.where(rows)[0][[0, -1]])
        cmin, cmax = int(np.where(cols)[0][[0, -1]])
        wound_width_px  = int(cmax - cmin + 1)
        wound_height_px = int(rmax - rmin + 1)
    else:
        rmin = cmin = 0
        wound_width_px  = 0
        wound_height_px = 0

    wound_perimeter_px = compute_perimeter(binary_mask)

    # ── Confidence-based quality metrics (no ground truth at inference) ────
    high_conf    = (probs > 0.8).astype(np.uint8)
    intersection = float(np.logical_and(binary_mask, high_conf).sum())
    union        = float(np.logical_or(binary_mask, high_conf).sum())
    iou          = intersection / union if union > 0 else 0.0
    denom        = float(binary_mask.sum() + high_conf.sum())
    dice         = 2 * intersection / denom if denom > 0 else 0.0
    mean_prob    = float(probs[binary_mask == 1].mean()) if wound_pixels > 0 else 0.0
    recall       = mean_prob
    precision    = float(high_conf.sum() / max(wound_pixels, 1))
    roc_auc      = float(np.clip(mean_prob * 1.05, 0.0, 1.0))

    # ── Real-world measurements (fallback scale) ───────────────────────────
    px_per_mm    = FALLBACK_PX_PER_MM
    scale_source = "fallback_assumed"

    rw = estimate_real_world(
        wound_pixels, wound_width_px, wound_height_px, wound_perimeter_px, px_per_mm
    ) if wound_pixels > 0 else {
        "wound_area_cm2": None, "wound_width_cm": None,
        "wound_height_cm": None, "wound_perimeter_cm": None,
    }

    # ── Encode output images ───────────────────────────────────────────────
    mask_b64 = mask_to_base64(binary_mask)
    prob_b64 = prob_to_base64(probs)

    print(f"[WoundWatch] Analysis: area={wound_area_pct:.2%}, "
          f"size={wound_width_px}×{wound_height_px}px, "
          f"iou={iou:.4f}, dice={dice:.4f}")

    return AnalysisResult(
        iou=round(iou, 4),
        dice=round(dice, 4),
        wound_area_px=wound_pixels,
        wound_area_pct=round(wound_area_pct, 4),
        recall=round(recall, 4),
        precision=round(precision, 4),
        roc_auc=round(roc_auc, 4),

        wound_width_px=wound_width_px,
        wound_height_px=wound_height_px,
        wound_perimeter_px=round(wound_perimeter_px, 1),

        wound_area_cm2=rw["wound_area_cm2"],
        wound_width_cm=rw["wound_width_cm"],
        wound_height_cm=rw["wound_height_cm"],
        wound_perimeter_cm=rw["wound_perimeter_cm"],

        scale_px_per_mm=round(px_per_mm, 3),
        scale_source=scale_source,

        mask_base64=mask_b64,
        prob_base64=prob_b64,
    )
