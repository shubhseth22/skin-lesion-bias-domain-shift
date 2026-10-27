"""
Unit tests for clinical inference pipeline and safety thresholding
Exercises actual source code in src/evaluation/inference.py
"""

import numpy as np
import pytest
from src.evaluation.inference import InferencePipeline, PredictionOutput


class TestInferencePipeline:
    """Validates inference probability properties, safety thresholds, and explainability shapes."""

    @pytest.fixture
    def pipeline(self):
        return InferencePipeline(
            class_names=["benign", "malignant"],
            default_threshold=0.5,
            high_risk_threshold=0.70,
            temperature=1.0,
        )

    @pytest.fixture
    def dummy_image(self):
        return np.ones((450, 600, 3), dtype=np.uint8) * 128

    def test_probabilities_sum_to_one(self, pipeline, dummy_image):
        logits = np.array([0.5, 2.5])  # Malignant favored
        output = pipeline.predict_sample(dummy_image, logits)

        assert isinstance(output, PredictionOutput)
        p_total = output.probabilities["benign"] + output.probabilities["malignant"]
        assert np.isclose(p_total, 1.0)
        assert output.predicted_class == "malignant"

    def test_high_risk_flag_trigger(self, pipeline, dummy_image):
        # Extreme malignant logit -> p_malignant > 0.70
        output_high = pipeline.predict_sample(dummy_image, np.array([-1.0, 3.0]))
        assert output_high.is_high_risk is True

        # Benign logit -> p_malignant < 0.70
        output_low = pipeline.predict_sample(dummy_image, np.array([2.0, -1.0]))
        assert output_low.is_high_risk is False
        assert output_low.predicted_class == "benign"

    def test_temperature_scaling_calibrated_probabilities(self, dummy_image):
        # Extreme logits
        logits = np.array([0.0, 5.0])
        pipe_t1 = InferencePipeline(temperature=1.0)
        pipe_t2 = InferencePipeline(temperature=2.5)

        out_t1 = pipe_t1.predict_sample(dummy_image, logits)
        out_t2 = pipe_t2.predict_sample(dummy_image, logits)

        # Warm temperature softens extreme probability (reduces overconfidence)
        assert out_t2.probabilities["malignant"] < out_t1.probabilities["malignant"]
        assert out_t2.temperature_applied == 2.5

    def test_threshold_sensitivity(self, pipeline, dummy_image):
        # Marginal logits: malignant probability is ~0.52
        logits = np.array([0.0, 0.1])
        out_std = pipeline.predict_sample(dummy_image, logits, threshold_override=0.50)
        assert out_std.predicted_class == "malignant"

        # Conservative clinical screening threshold (0.60)
        out_cons = pipeline.predict_sample(dummy_image, logits, threshold_override=0.60)
        assert out_cons.predicted_class == "benign"

    def test_gradcam_heatmap_dimensions_match_input(self, pipeline, dummy_image):
        output = pipeline.predict_sample(dummy_image, np.array([1.0, 1.0]))
        # Heatmap must strictly match original image resolution [H, W] = (450, 600)
        assert output.heatmap_shape == (450, 600)
        assert output.heatmap_min >= 0.0
        assert output.heatmap_max <= 1.0

    def test_batch_inference_processing(self, pipeline):
        images = [np.ones((224, 224, 3), dtype=np.uint8) * 100 for _ in range(3)]
        batch_logits = np.array([
            [1.0, 3.0],   # malignant
            [2.5, -0.5],  # benign
            [-2.0, 4.0],  # high risk malignant
        ])
        results = pipeline.predict_batch(images, batch_logits)
        assert len(results) == 3
        assert results[0].predicted_class == "malignant"
        assert results[1].predicted_class == "benign"
        assert results[2].is_high_risk is True

    def test_invalid_image_raises_value_error(self, pipeline):
        # 2D image instead of 3D
        with pytest.raises(ValueError, match="Expected 3D image"):
            pipeline.predict_sample(np.zeros((100, 100), dtype=np.uint8), np.array([1.0, 1.0]))

        # Image contains NaN
        nan_img = np.zeros((100, 100, 3), dtype=np.float32)
        nan_img[0, 0, 0] = np.nan
        with pytest.raises(ValueError, match="contains NaN or infinite"):
            pipeline.predict_sample(nan_img, np.array([1.0, 1.0]))

    def test_invalid_temperature_raises_value_error(self):
        with pytest.raises(ValueError, match="Temperature parameter must be positive"):
            InferencePipeline(temperature=-0.5)
