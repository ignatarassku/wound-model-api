"""
model.py — U-Net architectures for wound segmentation.

Architecture:
    Encoder: ResNet34 pretrained on ImageNet (transfer learning)
    Decoder: U-Net upsampling path with skip connections

Two model variants:
    UNet           — binary segmentation (wound vs background)
                     Output: [B, 1, H, W] raw logits → sigmoid for probability.
    TissueUNet     — multi-class tissue segmentation (4 classes)
                     Shares the frozen ResNet34 encoder with UNet.
                     Output: [B, N_CLASSES, H, W] raw logits → softmax for class probs.

Usage:
    from src.model import UNet, TissueUNet, get_model, get_tissue_model
    seg_model    = get_model()
    tissue_model = get_tissue_model(seg_checkpoint="results/checkpoints/best_model.pth")
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from src.config import CFG


# ── Building blocks ───────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """
    Two consecutive Conv2d → BatchNorm → ReLU blocks.
    Used in both the encoder (custom) and decoder paths.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    One upsampling step in the decoder:
        Upsample × 2  →  concatenate skip connection  →  DoubleConv
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample  = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv      = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        # Pad if spatial sizes differ (can happen with odd input dimensions)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

        x = torch.cat([skip, x], dim=1)   # channel-wise concatenation (skip connection)
        return self.conv(x)


# ── Main U-Net ────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    U-Net with pretrained ResNet34 encoder for wound segmentation.

    Input:  [B, 3, 256, 256]  — normalized RGB image
    Output: [B, 1, 256, 256]  — raw logits (apply sigmoid for probabilities)

    Encoder feature map channels (ResNet34):
        Layer 0 (stem)  : 64
        Layer 1 (layer1): 64
        Layer 2 (layer2): 128
        Layer 3 (layer3): 256
        Layer 4 (layer4): 512  ← bottleneck
    """

    def __init__(self, out_channels: int = CFG.OUT_CHANNELS) -> None:
        super().__init__()

        # Load pretrained ResNet34 encoder
        backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # ── Encoder (ResNet34 layers) ──────────────────────────────────────
        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # 64 ch, /2
        self.pool = backbone.maxpool                                              # /2
        self.enc1 = backbone.layer1    # 64 ch,  stride /4  total
        self.enc2 = backbone.layer2    # 128 ch, stride /8  total
        self.enc3 = backbone.layer3    # 256 ch, stride /16 total
        self.enc4 = backbone.layer4    # 512 ch, stride /32 total (bottleneck)

        # ── Decoder (U-Net upsampling path) ───────────────────────────────
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64,  64)
        self.dec1 = DecoderBlock(64,  64,  32)

        # Final upsampling to original resolution + output layer
        self.final_upsample = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.final_conv     = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            Logits tensor [B, 1, H, W] — apply sigmoid to get probabilities.
        """
        # ── Encoder ───────────────────────────────────────────────────────
        s0 = self.enc0(x)        # [B, 64,  H/2,  W/2]   — skip 0
        e  = self.pool(s0)       # [B, 64,  H/4,  W/4]
        s1 = self.enc1(e)        # [B, 64,  H/4,  W/4]   — skip 1
        s2 = self.enc2(s1)       # [B, 128, H/8,  W/8]   — skip 2
        s3 = self.enc3(s2)       # [B, 256, H/16, W/16]  — skip 3
        s4 = self.enc4(s3)       # [B, 512, H/32, W/32]  — bottleneck

        # ── Decoder ───────────────────────────────────────────────────────
        d4 = self.dec4(s4, s3)   # [B, 256, H/16, W/16]
        d3 = self.dec3(d4, s2)   # [B, 128, H/8,  W/8]
        d2 = self.dec2(d3, s1)   # [B, 64,  H/4,  W/4]
        d1 = self.dec1(d2, s0)   # [B, 32,  H/2,  W/2]

        # ── Final output ──────────────────────────────────────────────────
        out = self.final_upsample(d1)   # [B, 16, H, W]
        out = self.final_conv(out)      # [B, 1,  H, W] — logits

        return out


# ── Tissue U-Net (multi-class head on shared encoder) ─────────────────────

class TissueUNet(nn.Module):
    """
    Tissue type segmentation model built on top of the binary UNet.

    Strategy:
        1. Load the pretrained binary UNet encoder (enc0-enc4).
        2. Freeze the encoder — tissue features are learned only in the decoder.
        3. Add a fresh lightweight decoder head that outputs N_CLASSES logits.

    Input:  [B, 3, 256, 256]
    Output: [B, TISSUE_CLASSES, 256, 256] — raw logits; apply softmax for probs.

    The binary segmentation mask is used as input gating to focus predictions
    inside the wound region during inference (optional).
    """

    def __init__(
        self,
        num_classes:       int  = CFG.TISSUE_CLASSES,
        freeze_encoder:    bool = True,
    ) -> None:
        super().__init__()

        backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.enc1 = backbone.layer1
        self.enc2 = backbone.layer2
        self.enc3 = backbone.layer3
        self.enc4 = backbone.layer4

        if freeze_encoder:
            for p in self.enc0.parameters(): p.requires_grad = False
            for p in self.enc1.parameters(): p.requires_grad = False
            for p in self.enc2.parameters(): p.requires_grad = False
            for p in self.enc3.parameters(): p.requires_grad = False
            for p in self.enc4.parameters(): p.requires_grad = False

        # Fresh tissue decoder — same shape as the binary decoder
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128,  64,  64)
        self.dec1 = DecoderBlock( 64,  64,  32)

        self.final_upsample = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dropout        = nn.Dropout2d(p=0.2)
        self.final_conv     = nn.Conv2d(16, num_classes, kernel_size=1)

    def load_encoder_weights(self, checkpoint_path: str) -> None:
        """
        Bootstrap encoder from a trained binary UNet checkpoint.
        Only encoder keys are copied; mismatched decoder keys are ignored.
        """
        state = torch.load(checkpoint_path, map_location="cpu")
        encoder_keys = {k: v for k, v in state.items()
                        if k.startswith(("enc", "pool"))}
        missing, unexpected = self.load_state_dict(encoder_keys, strict=False)
        print(f"[TissueUNet] Loaded {len(encoder_keys)} encoder weights "
              f"| Missing: {len(missing)} | Unexpected: {len(unexpected)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, 3, H, W]
        Returns:
            Logits tensor [B, TISSUE_CLASSES, H, W]
        """
        s0 = self.enc0(x)
        e  = self.pool(s0)
        s1 = self.enc1(e)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        d4  = self.dec4(s4, s3)
        d3  = self.dec3(d4, s2)
        d2  = self.dec2(d3, s1)
        d1  = self.dec1(d2, s0)

        out = self.final_upsample(d1)
        out = self.dropout(out)
        out = self.final_conv(out)
        return out


def get_tissue_model(
    seg_checkpoint: Optional[str] = None,
    freeze_encoder: bool          = True,
) -> "TissueUNet":
    """
    Instantiate TissueUNet, optionally bootstrapping encoder from a binary
    UNet checkpoint, and move to the configured device.

    Args:
        seg_checkpoint : Path to binary UNet .pth file (optional).
        freeze_encoder : Freeze ResNet34 encoder layers (recommended).

    Returns:
        TissueUNet on CFG.DEVICE.
    """
    model = TissueUNet(
        num_classes    = CFG.TISSUE_CLASSES,
        freeze_encoder = freeze_encoder,
    ).to(CFG.DEVICE)

    if seg_checkpoint:
        model.load_encoder_weights(seg_checkpoint)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(
        f"[TissueUNet] {CFG.TISSUE_CLASSES} classes | "
        f"Trainable params: {trainable:,} / {total:,} | "
        f"Device: {CFG.DEVICE}"
    )
    return model


def get_model() -> UNet:
    """
    Instantiate model and move to the configured device.

    Returns:
        UNet model on CFG.DEVICE, ready for training.
    """
    model = UNet(out_channels=CFG.OUT_CHANNELS).to(CFG.DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] UNet with ResNet34 encoder | Trainable params: {n_params:,}")
    print(f"[Model] Running on: {CFG.DEVICE}")

    return model
