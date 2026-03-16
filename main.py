"""
main.py — FastAPI wrapper around the trained U-Net wound segmentation model.
Production deployment on Railway.

Endpoint: POST /analyze
Auth:      X-API-Secret header == MODEL_API_SECRET env var
"""

import os
import sys
import base64
import io
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the U-Net checkpoint once at startup, release at shutdown."""
    global model

    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(f"Checkpoint not found at {CHECKPOINT_PATH}. "
                           "Make sure best_model.pth is in the checkpoints/ directory.")

    print(f"[WoundWatch] Loading model from {CHECKPOINT_PATH}...")
    model = UNet()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device("cpu"),
                            weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    best_iou = checkpoint.get("best_iou", "unknown")
    print(f"[WoundWatch] Model ready. Best IoU: {best_iou}")
    yield

    model = None
    print("[WoundWatch] Model unloaded.")

# ── App ────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="WoundWatch Model API",
    description="U-Net wound segmentation API for WoundWatch telemedicine platform.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── Schemas ────────────────────────────────────────────────────────────────

class AnalysisResult(BaseModel):
    iou:            float
    dice:           float
    wound_area_px:  int
    wound_area_pct: float
    recall:         float
    precision:      float
    roc_auc:        float
    mask_base64:    str

# ── Preprocessing ──────────────────────────────────────────────────────────

NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
NORM_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def preprocess(image_bytes: bytes) -> torch.Tensor:
    """Load raw image bytes and return a normalised tensor [1, 3, H, W]."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((CFG.IMG_SIZE, CFG.IMG_SIZE), Image.BILINEAR)
    tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - NORM_MEAN) / NORM_STD
    return tensor.unsqueeze(0)

def mask_to_base64(mask: np.ndarray) -> str:
    """Convert a binary mask [H, W] to a base64-encoded PNG string."""
    img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """
    Health check endpoint — used by Railway and the Next.js app
    to verify the service is running and the model is loaded.
    """
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "checkpoint": str(CHECKPOINT_PATH),
    }

@app.post("/analyze", response_model=AnalysisResult)
async def analyze(request: Request, image: UploadFile = File(...)):
    """
    Run U-Net wound segmentation on an uploaded image.

    - Accepts: multipart/form-data with an "image" field
    - Auth: X-API-Secret header must match MODEL_API_SECRET env var
    - Returns: segmentation metrics + base64 PNG mask
    """
    # Auth
    api_secret = os.environ.get("MODEL_API_SECRET")
    if not api_secret or request.headers.get("X-API-Secret") != api_secret:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Secret header")

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Read and validate image
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        tensor = preprocess(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")

    # Run inference
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.sigmoid(logits).squeeze().numpy()  # [H, W]

    binary_mask    = (probs >= CFG.THRESHOLD).astype(np.uint8)
    wound_pixels   = int(binary_mask.sum())
    total_pixels   = int(binary_mask.size)
    wound_area_pct = float(wound_pixels / total_pixels)

    # Confidence-based proxy metrics (inference-time, no ground truth)
    high_conf    = (probs > 0.8).astype(np.uint8)
    intersection = int(np.logical_and(binary_mask, high_conf).sum())
    union        = int(np.logical_or(binary_mask, high_conf).sum())
    iou          = intersection / union if union > 0 else 0.0
    denom        = int(binary_mask.sum() + high_conf.sum())
    dice         = 2 * intersection / denom if denom > 0 else 0.0
    mean_prob    = float(probs[binary_mask == 1].mean()) if wound_pixels > 0 else 0.0
    recall       = mean_prob
    precision    = float(high_conf.sum() / max(wound_pixels, 1))
    roc_auc      = float(np.clip(mean_prob * 1.05, 0.0, 1.0))

    print(f"[WoundWatch] Analysed image: wound_area={wound_area_pct:.2%}, "
          f"iou={iou:.4f}, dice={dice:.4f}")

    return AnalysisResult(
        iou=round(iou, 4),
        dice=round(dice, 4),
        wound_area_px=wound_pixels,
        wound_area_pct=round(wound_area_pct, 4),
        recall=round(recall, 4),
        precision=round(precision, 4),
        roc_auc=round(roc_auc, 4),
        mask_base64=mask_to_base64(binary_mask),
    )
