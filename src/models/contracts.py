"""
Dermatological Model Architecture Contracts and Feature Extraction Interfaces

Establishes structural and dimensional contracts for model backbones (ResNet-50,
EfficientNet-B4, Vision Transformer) to guarantee feature extraction alignment
for domain adaptation and Grad-CAM interpretability.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class DermatologyModelContract:
    """
    Validates structural contracts, tensor dimensional shapes, and feature extraction
    interfaces across dermatological neural network backbones.
    """

    SUPPORTED_BACKBONES: Dict[str, Dict[str, Any]] = {
        "resnet50": {
            "feature_dim": 2048,
            "default_image_size": (224, 224),
            "expected_channels": 3,
            "target_layer_gradcam": "layer4",
        },
        "efficientnet_b4": {
            "feature_dim": 1792,
            "default_image_size": (224, 224),
            "expected_channels": 3,
            "target_layer_gradcam": "conv_head",
        },
        "vit_base_patch16_224": {
            "feature_dim": 768,
            "default_image_size": (224, 224),
            "expected_channels": 3,
            "target_layer_gradcam": "encoder_layer_11",
        },
    }

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 2,
    ) -> None:
        """
        Initialize the model contract.

        Args:
            backbone: Architecture identifier.
            num_classes: Number of diagnostic target classes (e.g., 2, 8, 114).

        Raises:
            ValueError: If backbone is not in SUPPORTED_BACKBONES or num_classes < 2.
        """
        if backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone '{backbone}'. Supported: {list(self.SUPPORTED_BACKBONES.keys())}"
            )
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2; got {num_classes}")

        self.backbone = backbone
        self.num_classes = num_classes
        self.spec = self.SUPPORTED_BACKBONES[backbone]
        self.feature_dim = self.spec["feature_dim"]
        self.expected_image_size = self.spec["default_image_size"]
        self.expected_channels = self.spec["expected_channels"]
        self.eval_mode = False

    def eval(self) -> "DermatologyModelContract":
        """Sets evaluation mode (disables dropout)."""
        self.eval_mode = True
        return self

    def train(self) -> "DermatologyModelContract":
        """Sets training mode."""
        self.eval_mode = False
        return self

    def validate_input_tensor_shape(self, shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
        """
        Validates that input tensor conforms to [B, C, H, W] contract.

        Args:
            shape: Tensor dimension tuple.

        Returns:
            Validated 4-tuple (batch_size, channels, height, width).

        Raises:
            ValueError: If input tensor does not match expected channels or dimensions.
        """
        if len(shape) != 4:
            raise ValueError(f"Expected 4D input tensor [B, C, H, W]; got shape {shape}")

        b, c, h, w = shape
        if b <= 0:
            raise ValueError(f"Batch dimension must be positive; got {b}")
        if c != self.expected_channels:
            raise ValueError(
                f"Expected {self.expected_channels} input channels; got {c} in shape {shape}"
            )
        if (h, w) != self.expected_image_size:
            raise ValueError(
                f"Expected spatial resolution {self.expected_image_size}; got ({h}, {w}) in shape {shape}"
            )
        return b, c, h, w

    def extract_features(self, input_shape: Tuple[int, ...]) -> Tuple[int, int]:
        """
        Simulates unprojected bottleneck feature representation shape [B, feature_dim].

        Args:
            input_shape: Shape tuple (B, C, H, W).

        Returns:
            Tuple of (batch_size, feature_dim).
        """
        b, _, _, _ = self.validate_input_tensor_shape(input_shape)
        return b, self.feature_dim

    def forward(self, input_shape: Tuple[int, ...]) -> Tuple[int, int]:
        """
        Simulates forward classification pass returning logit shape [B, num_classes].

        Args:
            input_shape: Shape tuple (B, C, H, W).

        Returns:
            Tuple of (batch_size, num_classes).
        """
        b, _ = self.extract_features(input_shape)
        return b, self.num_classes
