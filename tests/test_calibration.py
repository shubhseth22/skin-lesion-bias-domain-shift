"""
Unit tests for CalibrationEvaluator, ECE, MCE, and Temperature Scaling
Exercises actual source code in src/evaluation/calibration.py
"""

import numpy as np
import pytest
from src.evaluation.calibration import CalibrationEvaluator


class TestCalibrationEvaluator:
    """Validates calibration error computations, reliability diagrams, and temperature scaling."""

    def test_perfectly_calibrated_predictions(self):
        # 10 samples with conf=0.1, 1 is positive (10% acc)
        # 10 samples with conf=0.9, 9 are positive (90% acc)
        confidences = np.array([0.1] * 10 + [0.9] * 10)
        accuracies = np.array([1] + [0] * 9 + [1] * 9 + [0] * 1)

        evaluator = CalibrationEvaluator(n_bins=10)
        ece = evaluator.compute_ece(confidences, accuracies)
        assert np.isclose(ece, 0.0, atol=1e-4)

    def test_overconfident_wrong_predictions_high_ece(self):
        # 100% confidence, 0% accuracy
        confidences = np.ones(50) * 0.99
        accuracies = np.zeros(50)

        evaluator = CalibrationEvaluator(n_bins=10)
        ece = evaluator.compute_ece(confidences, accuracies)
        mce = evaluator.compute_mce(confidences, accuracies)

        assert ece > 0.95
        assert mce > 0.95

    def test_reliability_diagram_bins_structure(self):
        confidences = np.array([0.15, 0.25, 0.85, 0.95])
        accuracies = np.array([0, 1, 1, 1])

        evaluator = CalibrationEvaluator(n_bins=5)
        bins = evaluator.get_reliability_diagram_bins(confidences, accuracies)

        assert len(bins) == 5
        for b in bins:
            assert "bin_idx" in b
            assert "lower" in b
            assert "upper" in b
            assert "count" in b
            assert "avg_confidence" in b
            assert "accuracy" in b

    def test_brier_score_computation(self):
        probs = np.array([0.9, 0.1, 0.8])
        labels = np.array([1, 0, 1])
        # (0.1^2 + 0.1^2 + 0.2^2) / 3 = (0.01 + 0.01 + 0.04) / 3 = 0.06 / 3 = 0.02
        bs = CalibrationEvaluator.compute_brier_score(probs, labels)
        assert np.isclose(bs, 0.02)

    def test_temperature_scaling_logits_smoothing(self):
        logits = np.array([3.0, -1.0])
        # Cool temperature (T < 1) sharpens distribution
        # Warm temperature (T > 1) softens distribution
        p_sharp = CalibrationEvaluator.temperature_scale(logits, temperature=0.5)
        p_orig = CalibrationEvaluator.temperature_scale(logits, temperature=1.0)
        p_smooth = CalibrationEvaluator.temperature_scale(logits, temperature=2.0)

        assert p_sharp[0] > p_orig[0] > p_smooth[0]
        assert np.isclose(np.sum(p_smooth), 1.0)
        assert np.isclose(np.sum(p_sharp), 1.0)

    def test_invalid_temperature_raises_value_error(self):
        logits = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="Temperature parameter must be positive"):
            CalibrationEvaluator.temperature_scale(logits, temperature=0.0)

        with pytest.raises(ValueError, match="Temperature parameter must be positive"):
            CalibrationEvaluator.temperature_scale(logits, temperature=-1.5)

    def test_mismatched_lengths_raises_value_error(self):
        evaluator = CalibrationEvaluator()
        with pytest.raises(ValueError, match="Length mismatch"):
            evaluator.compute_ece(np.array([0.5, 0.8]), np.array([1]))
