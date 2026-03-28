"""
main.py — FastAPI wrapper around the trained U-Net wound segmentation model.
Production deployment on Railway.

Endpoints:
    GET  /health       — liveness check
    POST /analyze      — wound segmentation + measurement
    POST /push-score   — compute full PUSH score from analyze output + exudate level

Auth: X-API-Secret header == MODEL_API_SECRET env var
"""

import os
import sys
import asyncio
import base64
import io
import logging
import shutil
import threading
import urllib.request
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).parent))
from src.model import UNet, TissueUNet
from src.config import CFG
from src.measure import compute_measurements, compute_tissue_breakdown
from src.nail_detector import detect_nail
from src.photo_validator import PhotoQuality, validate_photo_basic, validate_wound_from_probs
from src.push_score import (
    compute_push_area_score_v2,
    compute_push_total,
    EXUDATE_LEVELS,
    PushResult,
)
from src.trajectory import compute_trend, VisitRecord, TrajectoryResult
from src.classifier import WoundClassifier, get_classifier, classify_wound

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if os.environ.get("SENTRY_DSN"):
    import sentry_sdk

    sentry_sdk.init(
        dsn=os.environ["SENTRY_DSN"],
        traces_sample_rate=0.1,
        send_default_pii=False,
    )

# ── Globals ────────────────────────────────────────────────────────────────────

model:             UNet             | None = None
tissue_model:      TissueUNet       | None = None
classifier_model:  WoundClassifier  | None = None
model_load_error:  str | None = None
CHECKPOINT_PATH           = Path(__file__).parent / "checkpoints" / "best_model.pth"
TISSUE_CHECKPOINT_PATH    = Path(__file__).parent / "checkpoints" / "tissue_model.pth"
CLASSIFIER_CHECKPOINT_PATH = Path(__file__).parent / "checkpoints" / "classifier_model.pth"

# ── Startup / shutdown ─────────────────────────────────────────────────────────


def _is_git_lfs_pointer(path: Path) -> bool:
    """True if `path` is a Git LFS pointer file (common when CI/deploy never fetched LFS blobs)."""
    try:
        head = path.read_bytes()[:200]
    except OSError:
        return False
    return head.startswith(b"version https://git-lfs.github.com/spec/v1\n")


def _reject_lfs_pointer(path: Path, label: str) -> None:
    if _is_git_lfs_pointer(path):
        raise RuntimeError(
            f"{label} at {path} is a Git LFS pointer, not real weights. "
            "Many hosts (including Railway) do not resolve LFS in the Docker build context. "
            "Set BINARY_CHECKPOINT_URL to a direct HTTPS URL of best_model.pth, "
            "or bake the real file into the image with a Dockerfile download step."
        )


def _maybe_download_binary_checkpoint() -> None:
    """If BINARY_CHECKPOINT_URL is set, download over a missing or LFS-pointer stub."""
    url = os.environ.get("BINARY_CHECKPOINT_URL", "").strip()
    if not url:
        return
    need = not CHECKPOINT_PATH.exists() or _is_git_lfs_pointer(CHECKPOINT_PATH)
    if not need:
        return
    logger.info("[WoundWatch] Downloading binary checkpoint from BINARY_CHECKPOINT_URL …")
    tmp = CHECKPOINT_PATH.with_suffix(".pth.part")
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "WoundWatch-model-api/1.0"})
        with urllib.request.urlopen(req, timeout=900) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
        if tmp.stat().st_size < 4096:
            raise RuntimeError("Downloaded checkpoint is too small to be valid.")
        if _is_git_lfs_pointer(tmp):
            raise RuntimeError("Downloaded file looks like a Git LFS pointer, not weights.")
        tmp.replace(CHECKPOINT_PATH)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def _load_models_sync() -> None:
    """Load checkpoints (runs in a background thread so /health can respond during load)."""
    global model, tissue_model, classifier_model

    _maybe_download_binary_checkpoint()

    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(
            f"Checkpoint not found at {CHECKPOINT_PATH}. "
            "Add best_model.pth under checkpoints/, or set BINARY_CHECKPOINT_URL to a direct download URL."
        )

    _reject_lfs_pointer(CHECKPOINT_PATH, "Binary segmentation checkpoint")

    logger.info("[WoundWatch] Loading binary segmentation model …")
    model = UNet()
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=CFG.DEVICE, weights_only=False)
    except Exception as e:
        if "invalid load key" in str(e).lower():
            raise RuntimeError(
                f"Failed to load {CHECKPOINT_PATH}: file is not a valid PyTorch checkpoint "
                "(often a Git LFS pointer or corrupt download). "
                "Set BINARY_CHECKPOINT_URL or ship the real .pth in the image."
            ) from e
        raise
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(CFG.DEVICE)
    model.eval()
    best_iou = checkpoint.get("best_iou", "unknown")
    logger.info("[WoundWatch] Binary model ready. Best IoU: %s", best_iou)

    if TISSUE_CHECKPOINT_PATH.exists():
        _reject_lfs_pointer(TISSUE_CHECKPOINT_PATH, "Tissue checkpoint")
        logger.info("[WoundWatch] Loading tissue classification model …")
        tissue_model = TissueUNet(num_classes=CFG.TISSUE_CLASSES)
        tissue_model.load_state_dict(torch.load(TISSUE_CHECKPOINT_PATH, map_location=CFG.DEVICE))
        tissue_model = tissue_model.to(CFG.DEVICE)
        tissue_model.eval()
        logger.info("[WoundWatch] Tissue model ready.")
    else:
        logger.info(
            "[WoundWatch] No tissue checkpoint found at %s — tissue classification disabled.",
            TISSUE_CHECKPOINT_PATH,
        )

    if CLASSIFIER_CHECKPOINT_PATH.exists():
        _reject_lfs_pointer(CLASSIFIER_CHECKPOINT_PATH, "Classifier checkpoint")
        logger.info("[WoundWatch] Loading wound type classifier …")
        classifier_model = get_classifier(
            num_classes=len(CFG.WOUND_TYPES),
            checkpoint=str(CLASSIFIER_CHECKPOINT_PATH),
        )
        logger.info("[WoundWatch] Classifier ready (%d classes).", len(CFG.WOUND_TYPES))
    else:
        logger.info(
            "[WoundWatch] No classifier checkpoint at %s — wound type classification disabled.",
            CLASSIFIER_CHECKPOINT_PATH,
        )


def _load_models_worker() -> None:
    global model_load_error
    try:
        _load_models_sync()
    except Exception as e:
        logger.exception("[WoundWatch] Model loading failed")
        # Surface the real error in /health and 503 responses (e.g. Git LFS pointer hint).
        msg = (str(e) or repr(e)).strip()
        if len(msg) > 1200:
            msg = msg[:1197] + "..."
        model_load_error = msg


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Start model loading in a background thread so Uvicorn binds immediately.
    Railway's HTTP healthcheck can hit /health while weights are still loading.
    """
    global model, tissue_model, classifier_model, model_load_error
    model_load_error = None
    t = threading.Thread(target=_load_models_worker, name="woundwatch-model-loader", daemon=True)
    t.start()
    yield
    model = tissue_model = classifier_model = None
    logger.info("[WoundWatch] Models unloaded.")

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="WoundWatch Model API",
    description="U-Net wound segmentation + measurement API for WoundWatch.",
    version="2.0.0",
    lifespan=lifespan,
)

_allowed_origins = os.environ.get(
    "ALLOWED_ORIGINS", "https://woundwatch.vercel.app,http://localhost:3000"
).split(",")
_allowed_origins = [o.strip() for o in _allowed_origins if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── Schemas ────────────────────────────────────────────────────────────────────

class PhotoQualityError(BaseModel):
    """Returned as HTTP 422 when photo quality checks fail."""
    error:    str = "photo_quality_failed"
    issues:   list[str]
    guidance: str


class AnalysisResult(BaseModel):
    # Photo quality
    photo_quality_passed: bool
    photo_issues:         list[str]
    photo_blur_score:     Optional[float]
    photo_brightness:     Optional[float]

    # Confidence-based segmentation quality estimates (not ground-truth metrics)
    confidence_iou:                float
    confidence_dice:               float
    confidence_recall:             float
    confidence_precision:          float
    confidence_roc_auc:            float

    # Pixel-space measurements (always present)
    wound_area_px:      int
    wound_area_pct:     float
    wound_width_px:     int
    wound_height_px:    int
    wound_perimeter_px: float

    # Bounding box pixel coordinates (for frontend overlay rendering)
    bbox_x_min:         int
    bbox_y_min:         int
    bbox_x_max:         int
    bbox_y_max:         int

    # Real-world measurements (None when scale unknown)
    wound_area_cm2:     Optional[float]
    wound_width_cm:     Optional[float]
    wound_height_cm:    Optional[float]
    wound_perimeter_cm: Optional[float]

    # Scale metadata
    scale_px_per_mm:    Optional[float]
    scale_source:       str   # "fingernail" | "fallback_assumed" | "unknown"

    # Tissue type breakdown (None when tissue model not available)
    # Fractions [0–1] that sum to ≈1.0 within the wound
    tissue_granulation:       Optional[float]
    tissue_slough:            Optional[float]
    tissue_eschar:            Optional[float]
    tissue_epithelialisation: Optional[float]
    tissue_model_available:   bool

    # Wound type classification (None when classifier not available)
    wound_type:            Optional[str]    # e.g. "diabetic_foot_ulcer"
    wound_type_confidence: Optional[float]  # softmax probability [0–1]
    wound_type_available:  bool

    # PUSH score — area sub-score is auto-computed; full score via /push-score
    push_score_area:    Optional[int]   # Sub-score A (0–10), None if area unknown

    # Base64-encoded PNG masks for overlay rendering (Next.js uploads to Storage)
    mask_base64:        str
    prob_heatmap_base64: Optional[str] = None
    tissue_mask_base64: Optional[str]   # colour-coded RGB PNG (None if no tissue model)

# ── Preprocessing helpers ──────────────────────────────────────────────────────

NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
NORM_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def preprocess(pil_image: Image.Image) -> torch.Tensor:
    """Resize + normalise a PIL image → model input tensor [1, 3, H, W]."""
    img    = pil_image.resize((CFG.IMG_SIZE, CFG.IMG_SIZE), Image.BILINEAR)
    tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - NORM_MEAN) / NORM_STD
    return tensor.unsqueeze(0)


def upsample_mask_to_image_size(binary_mask: np.ndarray, pil_image: Image.Image) -> np.ndarray:
    """
    Map 256×256 binary mask back to original PIL dimensions (nearest neighbour).
    Matches inverse of preprocess resize so measurements + stored PNG align with the photo.
    """
    w, h = pil_image.size
    m = (binary_mask * 255).astype(np.uint8)
    up = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return (up > 127).astype(np.uint8)


def resize_probs_to_image_size(probs: np.ndarray, pil_image: Image.Image) -> np.ndarray:
    """Bilinear resize of probability map [H, W] to original image size."""
    w, h = pil_image.size
    p = probs.astype(np.float32)
    return cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)


def upsample_class_map_to_image_size(class_map: np.ndarray, pil_image: Image.Image) -> np.ndarray:
    """Nearest-neighbour resize of integer class indices to original image size."""
    w, h = pil_image.size
    cm = class_map.astype(np.uint8)
    up = cv2.resize(cm, (w, h), interpolation=cv2.INTER_NEAREST)
    return up.astype(np.int64)


def mask_to_base64(mask: np.ndarray) -> str:
    """Convert a binary mask [H, W] uint8 to a base64-encoded PNG string."""
    img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def prob_heatmap_to_base64(probs: np.ndarray) -> str:
    """Encode probability map [H, W] float 0–1 as RGB PNG (approximate 'hot' colormap)."""
    p = np.clip(probs.astype(np.float64), 0.0, 1.0)
    h, w = p.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = (np.minimum(p * 3.0, 1.0) * 255.0).astype(np.uint8)
    rgb[..., 1] = (np.clip(p * 3.0 - 1.0, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgb[..., 2] = (np.clip(p * 3.0 - 2.0, 0.0, 1.0) * 255.0).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def tissue_colour_mask_to_base64(class_map: np.ndarray) -> str:
    """
    Convert a [H, W] int class map to an RGB colour-coded PNG (base64).
    Uses CFG.TISSUE_COLOURS for colouring. Ignored pixels → black.
    """
    h, w   = class_map.shape
    rgb    = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, colour in CFG.TISSUE_COLOURS.items():
        mask = class_map == cls_idx
        rgb[mask] = colour
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def run_tissue_inference(
    pil_image:  Image.Image,
    full_mask:  np.ndarray,
) -> tuple:
    """
    Run TissueUNet inference if the model is loaded.

    Args:
        full_mask: Binary wound mask at **original image resolution** (same as segmentation overlay).

    Returns:
        (breakdown_dict, tissue_class_map_full, tissue_mask_b64)
        All three are None if tissue_model is not loaded.
    """
    if tissue_model is None:
        return None, None, None

    try:
        tensor = preprocess(pil_image).to(CFG.DEVICE)
        with torch.no_grad():
            logits     = tissue_model(tensor)                    # [1, C, H, W]
            class_map  = logits.argmax(dim=1).squeeze().numpy()  # [H, W] int

        class_map_full = upsample_class_map_to_image_size(class_map, pil_image)
        breakdown = compute_tissue_breakdown(
            class_map_full,
            wound_mask   = full_mask,
            class_names  = CFG.TISSUE_NAMES,
        )
        colour_b64 = tissue_colour_mask_to_base64(class_map_full)
        return breakdown, class_map_full, colour_b64

    except Exception as exc:
        logger.warning("[WoundWatch] Tissue inference failed: %s", exc)
        return None, None, None

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness check — used by Railway and the frontend (returns 200 while models load)."""
    body: dict = {
        "status": "ok",
        "model_loaded": model is not None,
        "checkpoint": str(CHECKPOINT_PATH),
        "version": "2.0.0",
    }
    if model_load_error:
        body["error"] = model_load_error
    return body


class _WoundQualityReject(Exception):
    """Raised inside worker thread when photo / wound checks fail."""

    def __init__(self, issues: list[str], guidance: str):
        self.issues = issues
        self.guidance = guidance


MAX_IMAGE_BYTES = 15 * 1024 * 1024


def _analyze_sync(pil_image: Image.Image, basic_quality: PhotoQuality) -> AnalysisResult:
    """
    Single U-Net forward pass for wound presence + full metrics (no duplicate inference).
    Basic blur/resolution/brightness already validated in the request handler.
    """
    nail_result = detect_nail(pil_image)

    tensor = preprocess(pil_image).to(CFG.DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze().numpy()

    wq = validate_wound_from_probs(probs, CFG.THRESHOLD)
    if not wq.passed:
        raise _WoundQualityReject(wq.issues, wq.guidance)

    binary_mask = (probs >= CFG.THRESHOLD).astype(np.uint8)
    wound_pixels = int(binary_mask.sum())
    total_pixels = int(binary_mask.size)

    full_mask = upsample_mask_to_image_size(binary_mask, pil_image)
    w0, h0 = pil_image.size
    full_pixels = int(w0 * h0)
    wound_area_pct = float(full_mask.sum() / full_pixels) if full_pixels > 0 else 0.0

    measurements = compute_measurements(
        full_mask,
        px_per_mm=nail_result.px_per_mm,
        scale_source=nail_result.source,
    )

    high_conf = (probs > 0.8).astype(np.uint8)
    intersection = int(np.logical_and(binary_mask, high_conf).sum())
    union = int(np.logical_or(binary_mask, high_conf).sum())
    conf_iou = intersection / union if union > 0 else 0.0
    denom = int(binary_mask.sum() + high_conf.sum())
    conf_dice = 2 * intersection / denom if denom > 0 else 0.0
    mean_prob = float(probs[binary_mask == 1].mean()) if wound_pixels > 0 else 0.0
    conf_recall = mean_prob
    conf_precision = float(high_conf.sum() / max(wound_pixels, 1))
    conf_roc_auc = float(np.clip(mean_prob * 1.05, 0.0, 1.0))

    tissue_breakdown, _, tissue_mask_b64 = run_tissue_inference(pil_image, full_mask)

    wound_type_label = None
    wound_type_confidence = None
    if classifier_model is not None:
        try:
            wound_type_label, wound_type_confidence = classify_wound(pil_image, classifier_model)
            wound_type_confidence = round(wound_type_confidence, 4)
        except Exception as exc:
            logger.warning("[WoundWatch] Classifier inference failed: %s", exc)

    probs_full = resize_probs_to_image_size(probs, pil_image)
    prob_b64 = prob_heatmap_to_base64(probs_full)

    logger.info(
        "[WoundWatch] Analysed image: area=%.2f cm² (%.2f%%), scale=%s, tissue=%s",
        measurements.wound_area_cm2 or 0.0,
        wound_area_pct * 100,
        nail_result.source,
        "available" if tissue_breakdown else "unavailable",
    )

    return AnalysisResult(
        photo_quality_passed=basic_quality.passed,
        photo_issues=basic_quality.issues,
        photo_blur_score=basic_quality.blur_score,
        photo_brightness=basic_quality.brightness,
        confidence_iou=round(conf_iou, 4),
        confidence_dice=round(conf_dice, 4),
        confidence_recall=round(conf_recall, 4),
        confidence_precision=round(conf_precision, 4),
        confidence_roc_auc=round(conf_roc_auc, 4),
        wound_area_px=measurements.wound_area_px,
        wound_area_pct=round(wound_area_pct, 4),
        wound_width_px=measurements.wound_width_px,
        wound_height_px=measurements.wound_height_px,
        wound_perimeter_px=measurements.wound_perimeter_px,
        bbox_x_min=measurements.bbox_x_min,
        bbox_y_min=measurements.bbox_y_min,
        bbox_x_max=measurements.bbox_x_max,
        bbox_y_max=measurements.bbox_y_max,
        wound_area_cm2=measurements.wound_area_cm2,
        wound_width_cm=measurements.wound_width_cm,
        wound_height_cm=measurements.wound_height_cm,
        wound_perimeter_cm=measurements.wound_perimeter_cm,
        scale_px_per_mm=measurements.scale_px_per_mm,
        scale_source=measurements.scale_source,
        tissue_granulation=tissue_breakdown.get("granulation") if tissue_breakdown else None,
        tissue_slough=tissue_breakdown.get("slough") if tissue_breakdown else None,
        tissue_eschar=tissue_breakdown.get("eschar") if tissue_breakdown else None,
        tissue_epithelialisation=tissue_breakdown.get("epithelialisation") if tissue_breakdown else None,
        tissue_model_available=tissue_model is not None,
        wound_type=wound_type_label,
        wound_type_confidence=wound_type_confidence,
        wound_type_available=classifier_model is not None,
        push_score_area=compute_push_area_score_v2(measurements.wound_area_cm2),
        mask_base64=mask_to_base64(full_mask),
        prob_heatmap_base64=prob_b64,
        tissue_mask_base64=tissue_mask_b64,
    )


@app.post("/analyze", response_model=AnalysisResult)
async def analyze(request: Request, image: UploadFile = File(...)):
    """U-Net segmentation — one forward pass per request (see validate_wound_from_probs)."""
    api_secret = os.environ.get("MODEL_API_SECRET")
    if not api_secret or request.headers.get("X-API-Secret") != api_secret:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Secret header")

    if model is None:
        if model_load_error:
            raise HTTPException(status_code=503, detail=model_load_error)
        raise HTTPException(status_code=503, detail="Model is still loading; retry in a few seconds")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Image larger than 15 MB")

    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not open image: {exc}")

    basic = validate_photo_basic(pil_image)
    if not basic.passed:
        logger.info("[WoundWatch] Photo rejected (basic): %s", basic.issues)
        raise HTTPException(
            status_code=422,
            detail=PhotoQualityError(
                issues=basic.issues,
                guidance=basic.guidance,
            ).model_dump(),
        )

    try:
        result = await asyncio.wait_for(asyncio.to_thread(_analyze_sync, pil_image, basic), timeout=30.0)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Inference timed out") from None
    except _WoundQualityReject as rej:
        logger.info("[WoundWatch] Photo rejected (wound): %s", rej.issues)
        raise HTTPException(
            status_code=422,
            detail=PhotoQualityError(
                issues=rej.issues,
                guidance=rej.guidance,
            ).model_dump(),
        ) from None

    return result


# ── /push-score endpoint ────────────────────────────────────────────────────────

class PushScoreRequest(BaseModel):
    """
    Input for the /push-score endpoint.

    Paste the values from a previous /analyze response plus the doctor's
    exudate level observation. All tissue fields are optional — if absent,
    tissue sub-score C defaults to 0.
    """
    wound_area_cm2:           Optional[float] = None
    exudate_level:            str             = "none"   # none | light | moderate | heavy
    tissue_granulation:       Optional[float] = None
    tissue_slough:            Optional[float] = None
    tissue_eschar:            Optional[float] = None
    tissue_epithelialisation: Optional[float] = None


class PushScoreResponse(BaseModel):
    area_score:      int
    exudate_score:   int
    tissue_score:    int
    total_score:     int
    interpretation:  str
    wound_area_cm2:  Optional[float]
    tissue_dominant: Optional[str]
    exudate_level:   str


@app.post("/push-score", response_model=PushScoreResponse)
async def push_score(request: Request, body: PushScoreRequest):
    """
    Compute a full PUSH score from wound analysis output + doctor's exudate
    level input.

    - Sub-score A (area):    auto-computed from wound_area_cm2
    - Sub-score B (exudate): taken from exudate_level (doctor input)
    - Sub-score C (tissue):  auto-computed from tissue_* fractions

    Intended workflow:
      1. Doctor uploads wound photo → POST /analyze → get wound_area_cm2,
         tissue_granulation, tissue_slough, tissue_eschar, tissue_epithelialisation
      2. Doctor observes exudate level and enters it in the app
      3. App sends both to POST /push-score → gets full PUSH score + interpretation

    Auth: same X-API-Secret header as /analyze.
    """
    api_secret = os.environ.get("MODEL_API_SECRET")
    if not api_secret or request.headers.get("X-API-Secret") != api_secret:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Secret header")

    if body.exudate_level.strip().lower() not in EXUDATE_LEVELS:
        raise HTTPException(
            status_code=422,
            detail=f"exudate_level must be one of: {list(EXUDATE_LEVELS.keys())}"
        )

    tissue_breakdown = None
    if any(v is not None for v in [
        body.tissue_granulation, body.tissue_slough,
        body.tissue_eschar, body.tissue_epithelialisation
    ]):
        tissue_breakdown = {
            "granulation":       body.tissue_granulation       or 0.0,
            "slough":            body.tissue_slough            or 0.0,
            "eschar":            body.tissue_eschar            or 0.0,
            "epithelialisation": body.tissue_epithelialisation or 0.0,
        }

    result: PushResult = compute_push_total(
        wound_area_cm2   = body.wound_area_cm2,
        exudate_level    = body.exudate_level,
        tissue_breakdown = tissue_breakdown,
    )

    logger.info(
        "[WoundWatch] PUSH score: A=%d B=%d C=%d total=%d (%s)",
        result.area_score, result.exudate_score, result.tissue_score,
        result.total_score, result.interpretation,
    )

    return PushScoreResponse(
        area_score      = result.area_score,
        exudate_score   = result.exudate_score,
        tissue_score    = result.tissue_score,
        total_score     = result.total_score,
        interpretation  = result.interpretation,
        wound_area_cm2  = result.wound_area_cm2,
        tissue_dominant = result.tissue_dominant,
        exudate_level   = body.exudate_level,
    )


# ── /trajectory endpoint ───────────────────────────────────────────────────────

class VisitRecordRequest(BaseModel):
    """One wound visit entry for trajectory input."""
    visit_date:       str
    wound_area_cm2:   Optional[float] = None
    tissue_granulation:       Optional[float] = None
    tissue_slough:            Optional[float] = None
    tissue_eschar:            Optional[float] = None
    tissue_epithelialisation: Optional[float] = None
    push_score:       Optional[int]   = None
    notes:            Optional[str]   = None


class TrajectoryRequest(BaseModel):
    """List of past visit records for healing trajectory prediction."""
    visits: list[VisitRecordRequest]


class TrajectoryResponse(BaseModel):
    trend:             str
    weekly_rate_cm2:   Optional[float]
    weekly_rate_pct:   Optional[float]
    estimated_closure: Optional[str]
    weeks_to_closure:  Optional[float]
    r_squared:         Optional[float]
    visits_used:       int
    first_area_cm2:    Optional[float]
    latest_area_cm2:   Optional[float]
    total_change_cm2:  Optional[float]
    total_change_pct:  Optional[float]
    interpretation:    str


@app.post("/trajectory", response_model=TrajectoryResponse)
async def trajectory(request: Request, body: TrajectoryRequest):
    """
    Predict healing trajectory from a time-series of wound visit measurements.

    Intended workflow:
      1. Each time the doctor uploads a photo, the frontend stores the
         wound_area_cm2 from /analyze along with the visit date.
      2. After ≥ 2 visits, the frontend calls POST /trajectory with the
         full list of past measurements.
      3. The API returns a trend label and estimated closure date.

    Input: list of visit records with at minimum visit_date + wound_area_cm2.
    Output: trend ("healing" | "stable" | "worsening" | "insufficient_data"),
            weekly_rate_cm2, estimated_closure date, and plain-language interpretation.

    Auth: same X-API-Secret header as /analyze.
    """
    api_secret = os.environ.get("MODEL_API_SECRET")
    if not api_secret or request.headers.get("X-API-Secret") != api_secret:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Secret header")

    if not body.visits:
        raise HTTPException(status_code=422, detail="visits list must not be empty")

    # Convert request objects → domain VisitRecord objects
    visit_records: list[VisitRecord] = []
    for v in body.visits:
        tissue = None
        if any(x is not None for x in [
            v.tissue_granulation, v.tissue_slough,
            v.tissue_eschar, v.tissue_epithelialisation,
        ]):
            tissue = {
                "granulation":       v.tissue_granulation       or 0.0,
                "slough":            v.tissue_slough            or 0.0,
                "eschar":            v.tissue_eschar            or 0.0,
                "epithelialisation": v.tissue_epithelialisation or 0.0,
            }
        try:
            visit_records.append(VisitRecord(
                visit_date       = v.visit_date,
                wound_area_cm2   = v.wound_area_cm2,
                tissue_breakdown = tissue,
                push_score       = v.push_score,
                notes            = v.notes,
            ))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"Invalid visit_date: {exc}")

    result: TrajectoryResult = compute_trend(visit_records)

    logger.info(
        "[WoundWatch] Trajectory: trend=%s rate=%.3f cm²/week visits=%d closure=%s",
        result.trend,
        result.weekly_rate_cm2 or 0.0,
        result.visits_used,
        result.estimated_closure or "N/A",
    )

    return TrajectoryResponse(
        trend             = result.trend,
        weekly_rate_cm2   = result.weekly_rate_cm2,
        weekly_rate_pct   = result.weekly_rate_pct,
        estimated_closure = result.estimated_closure,
        weeks_to_closure  = result.weeks_to_closure,
        r_squared         = result.r_squared,
        visits_used       = result.visits_used,
        first_area_cm2    = result.first_area_cm2,
        latest_area_cm2   = result.latest_area_cm2,
        total_change_cm2  = result.total_change_cm2,
        total_change_pct  = result.total_change_pct,
        interpretation    = result.interpretation,
    )
