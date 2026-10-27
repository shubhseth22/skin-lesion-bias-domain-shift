"""
Clinical Inference Pipeline and Safety Risk Flagging Module

Processes input images, validates spatial and colorimetric integrity, applies
calibrated temperature scaling, and enforces clinical safety thresholds and
Grad-CAM interpretability shape contracts.
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from .calibration import CalibrationEvaluator

logger = logging.getLogger(__name__)


@dataclass
class PredictionOutput:
    """Structured clinical inference prediction payload."""

    probabilities: Dict[str, float]
    predicted_class: str
    predicted_label_idx: int
    is_high_risk: bool
    temperature_applied: float
    confidence: float
    heatmap_shape: Tuple[int, int]
    heatmap_min: float
    heatmap_max: float


class InferencePipeline:
    """
    Standardized inference execution engine.
    """

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        default_threshold: float = 0.5,
        high_risk_threshold: float = 0.70,
        temperature: float = 1.0,
    ) -> None:
        """
        Initialize the inference pipeline.

        Args:
            class_names: Names of diagnostic categories (defaults to ['benign', 'malignant']).
            default_threshold: Decision boundary for binary classification.
            high_risk_threshold: Probability cutoff triggering priority clinical urgency.
            temperature: Post-hoc calibration temperature parameter $T > 0$.
        """
        self.class_names = class_names or ["benign", "malignant"]
        self.default_threshold = default_threshold
        self.high_risk_threshold = high_risk_threshold
        if temperature <= 0.0:
            raise ValueError(f"Temperature parameter must be positive (T > 0); got {temperature}")
        self.temperature = temperature

    def validate_raw_image(self, image: np.ndarray) -> Tuple[int, int, int]:
        """
        Validates input image dimensions and numerical validity.

        Args:
            image: Array representing RGB image [H, W, 3].

        Returns:
            Tuple of (height, width, channels).

        Raises:
            ValueError: If image is not 3D, has non-finite values, or has wrong channel count.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a numpy ndarray.")
        if image.ndim != 3:
            raise ValueError(f"Expected 3D image [H, W, C]; got {image.ndim}D array with shape {image.shape}")
        if not np.all(np.isfinite(image)):
            raise ValueError("Input image contains NaN or infinite pixel values.")

        h, w, c = image.shape
        if c != 3:
            raise ValueError(f"Expected 3 color channels (RGB); got {c}")
        if h <= 0 or w <= 0:
            raise ValueError(f"Image dimensions must be positive; got ({h}, {w})")

        return h, w, c

    def predict_sample(
        self,
        raw_image: np.ndarray,
        logits: np.ndarray,
        temperature_override: Optional[float] = None,
        threshold_override: Optional[float] = None,
    ) -> PredictionOutput:
        """
        Executes single-sample inference with calibrated probabilities and safety flagging.

        Args:
            raw_image: Original RGB image array [H, W, 3].
            logits: Unnormalized model output array [num_classes].
            temperature_override: Optional override for temperature scaling.
            threshold_override: Optional decision threshold override.

        Returns:
            PredictionOutput dataclass.
        """
        h, w, _ = self.validate_raw_image(raw_image)
        logits = np.asarray(logits, dtype=np.float64)

        if len(logits) != len(self.class_names):
            raise ValueError(
                f"Logit length ({len(logits)}) does not match class count ({len(self.class_names)})"
            )

        t = temperature_override if temperature_override is not None else self.temperature
        thresh = threshold_override if threshold_override is not None else self.default_threshold

        # Apply temperature scaling
        probs = CalibrationEvaluator.temperature_scale(logits, temperature=t)

        prob_dict = {name: float(probs[i]) for i, name in enumerate(self.class_names)}
        p_malignant = float(probs[1]) if len(self.class_names) == 2 else float(np.max(probs))

        if len(self.class_names) == 2:
            pred_idx = 1 if p_malignant >= thresh else 0
        else:
            pred_idx = int(np.argmax(probs))

        predicted_class = self.class_names[pred_idx]
        is_high_risk = bool(p_malignant >= self.high_risk_threshold)

        # Mock / simulate Grad-CAM spatial heatmap matching original resolution [H, W]
        # Real Grad-CAM generator will populate this in Phase 4
        heatmap_shape = (h, w)
        heatmap_min = 0.0
        heatmap_max = 1.0

        return PredictionOutput(
            probabilities=prob_dict,
            predicted_class=predicted_class,
            predicted_label_idx=pred_idx,
            is_high_risk=is_high_risk,
            temperature_applied=t,
            confidence=float(np.max(probs)),
            heatmap_shape=heatmap_shape,
            heatmap_min=heatmap_min,
            heatmap_max=heatmap_max,
        )

    def predict_batch(
        self,
        raw_images: List[np.ndarray],
        batch_logits: np.ndarray,
        temperature_override: Optional[float] = None,
        threshold_override: Optional[float] = None,
    ) -> List[PredictionOutput]:
        """
        Processes batch inference over multiple images and logit rows.
        """
        batch_logits = np.asarray(batch_logits, dtype=np.float64)
        if len(raw_images) != len(batch_logits):
            raise ValueError(
                f"Batch size mismatch: {len(raw_images)} images vs {len(batch_logits)} logit rows"
            )

        outputs: List[PredictionOutput] = []
        for img, l in zip(raw_images, batch_logits):
            out = self.predict_sample(
                img,
                l,
                temperature_override=temperature_override,
                threshold_override=threshold_override,
            )
            outputs.append(out)
        return outputs
