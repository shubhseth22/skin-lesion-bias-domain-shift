"""
Calibration Evaluation, Reliability Diagrams, and Temperature Scaling Module

Quantifies predictive probability calibration, Expected Calibration Error (ECE),
Maximum Calibration Error (MCE), and Brier Score across patient subgroups and domain shifts.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


class CalibrationEvaluator:
    """
    Evaluates probability calibration and post-hoc temperature scaling.
    """

    def __init__(self, n_bins: int = 10) -> None:
        """
        Initialize calibration evaluator.

        Args:
            n_bins: Number of equal-width probability bins in [0, 1].
        """
        if n_bins < 2:
            raise ValueError("n_bins must be at least 2.")
        self.n_bins = n_bins

    def get_reliability_diagram_bins(
        self,
        confidences: np.ndarray,
        accuracies: np.ndarray,
    ) -> List[Dict[str, float]]:
        """
        Partitions predictions into confidence bins and computes empirical accuracy per bin.

        Args:
            confidences: Array of predicted confidence scores in [0, 1].
            accuracies: Binary array indicating correct predictions (1) or incorrect (0).

        Returns:
            List of bin dictionaries with bin boundaries, average confidence,
            empirical accuracy, and sample count.
        """
        confidences = np.asarray(confidences, dtype=np.float64)
        accuracies = np.asarray(accuracies, dtype=np.float64)

        if len(confidences) != len(accuracies):
            raise ValueError("Length mismatch between confidences and accuracies.")

        if len(confidences) == 0:
            return []

        bin_boundaries = np.linspace(0.0, 1.0, self.n_bins + 1)
        bin_data: List[Dict[str, float]] = []

        for i in range(self.n_bins):
            b_lower = bin_boundaries[i]
            b_upper = bin_boundaries[i + 1]

            if i == 0:
                in_bin = (confidences >= b_lower) & (confidences <= b_upper)
            else:
                in_bin = (confidences > b_lower) & (confidences <= b_upper)

            count = int(np.sum(in_bin))
            if count > 0:
                avg_conf = float(np.mean(confidences[in_bin]))
                avg_acc = float(np.mean(accuracies[in_bin]))
            else:
                avg_conf = float((b_lower + b_upper) / 2.0)
                avg_acc = 0.0

            bin_data.append({
                "bin_idx": i,
                "lower": float(b_lower),
                "upper": float(b_upper),
                "count": count,
                "avg_confidence": avg_conf,
                "accuracy": avg_acc,
            })

        return bin_data

    def compute_ece(
        self,
        confidences: np.ndarray,
        accuracies: np.ndarray,
    ) -> float:
        """
        Computes Expected Calibration Error (ECE) across bins:
        $$\\text{ECE} = \\sum_{m=1}^{M} \\frac{|B_m|}{N} |\\text{acc}(B_m) - \\text{conf}(B_m)|$$

        Args:
            confidences: Array of predicted confidence scores in [0, 1].
            accuracies: Binary array indicating correct predictions (1) or incorrect (0).

        Returns:
            Float representing ECE in [0.0, 1.0].
        """
        confidences = np.asarray(confidences, dtype=np.float64)
        accuracies = np.asarray(accuracies, dtype=np.float64)
        n = len(confidences)
        if n == 0:
            return 0.0

        bin_data = self.get_reliability_diagram_bins(confidences, accuracies)
        ece = 0.0

        for b in bin_data:
            if b["count"] > 0:
                bin_weight = b["count"] / n
                gap = abs(b["accuracy"] - b["avg_confidence"])
                ece += bin_weight * gap

        return float(np.clip(ece, 0.0, 1.0))

    def compute_mce(
        self,
        confidences: np.ndarray,
        accuracies: np.ndarray,
    ) -> float:
        """
        Computes Maximum Calibration Error (MCE) across populated bins:
        $$\\text{MCE} = \\max_{m \\in \\{1..M\\}} |\\text{acc}(B_m) - \\text{conf}(B_m)|$$
        """
        bin_data = self.get_reliability_diagram_bins(confidences, accuracies)
        populated = [b for b in bin_data if b["count"] > 0]
        if not populated:
            return 0.0

        mce = max(abs(b["accuracy"] - b["avg_confidence"]) for b in populated)
        return float(np.clip(mce, 0.0, 1.0))

    @staticmethod
    def compute_brier_score(
        probabilities: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """
        Computes the Brier Score:
        $$\\text{BS} = \\frac{1}{N} \\sum_{i=1}^N (p_i - y_i)^2$$
        """
        p = np.asarray(probabilities, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        if len(p) != len(y):
            raise ValueError("Length mismatch between probabilities and labels.")
        if len(p) == 0:
            return 0.0
        return float(np.mean((p - y) ** 2))

    @staticmethod
    def temperature_scale(
        logits: np.ndarray,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """
        Applies temperature scaling to raw logits and computes calibrated probabilities:
        $$\\hat{p}_i = \\frac{\\exp(z_i / T)}{\\sum_j \\exp(z_j / T)}$$

        Args:
            logits: Array of shape [N, num_classes] or [num_classes].
            temperature: Scalar temperature parameter $T > 0$.

        Returns:
            Calibrated probability distribution array matching logits shape.

        Raises:
            ValueError: If temperature is <= 0.
        """
        if temperature <= 0.0:
            raise ValueError(f"Temperature parameter must be positive (T > 0); got {temperature}")

        z = np.asarray(logits, dtype=np.float64) / temperature
        # Numerically stable softmax
        if z.ndim == 1:
            exp_z = np.exp(z - np.max(z))
            return exp_z / np.sum(exp_z)
        else:
            exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
            return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
