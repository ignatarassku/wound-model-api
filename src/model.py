"""
model.py — U-Net architecture for binary wound segmentation.

Architecture:
    Encoder: ResNet34 pretrained on ImageNet (transfer learning)
    Decoder: U-Net upsampling path with skip connections
    Output:  Single-channel logits → apply sigmoid for probabilities

The model outputs RAW LOGITS (no sigmoid applied).
Apply torch.sigmoid() during inference, or use BCEWithLogitsLoss during training.

Usage:
    from src.model import UNet
    model = UNet()
    logits = model(images)          # [B, 1, H, W] — raw logits
    probs  = torch.sigmoid(logits)  # [B, 1, H, W] — probabilities [0, 1]
"""

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
