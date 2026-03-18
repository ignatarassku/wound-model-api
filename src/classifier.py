"""
classifier.py — Wound Type Classification (Feature 4).

Classifies a wound image into one of 5 clinical categories using a
fine-tuned ResNet50 backbone with a custom classification head.

Classes:
    0  diabetic_foot_ulcer   — DFU: neuropathic, plantar surface
    1  venous_leg_ulcer      — VLU: medial lower leg, irregular edges
    2  pressure_injury       — NPUAP stage I–IV, bony prominences
    3  surgical              — post-operative incision / dehiscence
    4  burn                  — thermal / chemical / radiation injury

Usage:
    model = get_classifier()                          # load architecture
    model.load_state_dict(torch.load(CFG.CLASSIFIER_PATH))
    label, confidence = classify_wound(pil_image, model)

Training:
    python -m src.train_classifier
    Requires dataset in data/wound_types/<class_name>/*.jpg
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
import numpy as np

from src.config import CFG


# ── Model ─────────────────────────────────────────────────────────────────────

class WoundClassifier(nn.Module):
    """
    ResNet50 fine-tuned for 5-class wound type classification.

    Architecture:
        - ResNet50 backbone (ImageNet pretrained)
        - Global average pooling (built into ResNet)
        - Dropout (0.3) for regularisation
        - Linear head: 2048 → num_classes
        - No sigmoid/softmax — raw logits returned; use softmax at inference
    """

    def __init__(self, num_classes: int = len(CFG.WOUND_TYPES), dropout: float = 0.3) -> None:
        super().__init__()
        from torchvision import models
        backbone        = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features     = backbone.fc.in_features
        backbone.fc     = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        self.backbone   = backbone
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, 3, H, W] normalised image tensor
        Returns:
            logits [B, num_classes] — apply softmax for probabilities
        """
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        """Freeze all layers except the final FC head (for warmup training)."""
        for name, param in self.named_parameters():
            if "backbone.fc" not in name:
                param.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters for fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True


def get_classifier(
    num_classes: int = len(CFG.WOUND_TYPES),
    checkpoint:  Optional[str] = None,
) -> WoundClassifier:
    """
    Build a WoundClassifier and optionally load a saved checkpoint.

    Args:
        num_classes : number of wound type classes
        checkpoint  : path to a .pth state dict (None = random init)

    Returns:
        WoundClassifier in eval mode if checkpoint provided, else train mode.
    """
    model = WoundClassifier(num_classes=num_classes)
    if checkpoint and Path(checkpoint).exists():
        state = torch.load(checkpoint, map_location=CFG.DEVICE, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        print(f"[WoundClassifier] Loaded checkpoint: {checkpoint}")
    else:
        print(f"[WoundClassifier] {num_classes} classes | random weights (untrained)")
    model = model.to(CFG.DEVICE)
    return model


# ── Preprocessing ─────────────────────────────────────────────────────────────

_CLASSIFIER_SIZE = 224   # ResNet50 expects 224×224

import torchvision.transforms as T

_CLASSIFIER_TRANSFORM = T.Compose([
    T.Resize((_CLASSIFIER_SIZE, _CLASSIFIER_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=CFG.NORM_MEAN, std=CFG.NORM_STD),
])


def preprocess_for_classifier(pil_image: Image.Image) -> torch.Tensor:
    """
    Resize and normalise a PIL image for WoundClassifier input.

    Returns:
        Tensor [1, 3, 224, 224] on CFG.DEVICE.
    """
    tensor = _CLASSIFIER_TRANSFORM(pil_image.convert("RGB"))
    return tensor.unsqueeze(0).to(CFG.DEVICE)


# ── Inference ─────────────────────────────────────────────────────────────────

def classify_wound(
    pil_image: Image.Image,
    model:     WoundClassifier,
) -> Tuple[str, float]:
    """
    Classify a wound image into one of CFG.WOUND_TYPES.

    Args:
        pil_image : PIL RGB image (any size — will be resized internally)
        model     : loaded WoundClassifier in eval mode

    Returns:
        (label, confidence)
            label      : class name string from CFG.WOUND_TYPES
            confidence : softmax probability of the predicted class [0–1]
    """
    model.eval()
    tensor = preprocess_for_classifier(pil_image)

    with torch.no_grad():
        logits = model(tensor)                          # [1, C]
        probs  = torch.softmax(logits, dim=1).squeeze() # [C]

    class_idx  = int(probs.argmax().item())
    confidence = float(probs[class_idx].item())
    label      = CFG.WOUND_TYPES[class_idx]

    return label, confidence


def classify_wound_full(
    pil_image: Image.Image,
    model:     WoundClassifier,
) -> dict:
    """
    Classify a wound image and return all class probabilities.

    Returns:
        dict with keys:
            label          : top-1 class name
            confidence     : top-1 softmax probability
            all_probs      : {class_name: probability} for all classes
    """
    model.eval()
    tensor = preprocess_for_classifier(pil_image)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    class_idx = int(np.argmax(probs))
    return {
        "label":      CFG.WOUND_TYPES[class_idx],
        "confidence": float(probs[class_idx]),
        "all_probs":  {name: float(p) for name, p in zip(CFG.WOUND_TYPES, probs)},
    }
