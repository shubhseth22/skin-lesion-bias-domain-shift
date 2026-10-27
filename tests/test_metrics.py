"""
Unit tests for SubgroupEvaluator and fairness metrics in src/research/subgroup_analysis.py
Exercises actual source code and asserts behavior across demographic subgroups.
"""

import numpy as np
import pandas as pd
import pytest
from src.research.subgroup_analysis import SubgroupEvaluator


@pytest.fixture
def sample_evaluation_data():
    """Generates synthetic predictions across 3 skin tone subgroups."""
    np.random.seed(42)
    n = 150
    return pd.DataFrame({
        "image_id": [f"IMG_{i:04d}" for i in range(n)],
        "fitzpatrick_tone": ["I-II"] * 60 + ["III-IV"] * 50 + ["V-VI"] * 40,
        "label": np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
        "pred_prob": np.random.uniform(0.0, 1.0, size=n),
    })


class TestSubgroupEvaluator:
    """Validates subgroup metric disaggregation, equalized odds, and demographic parity."""

    def test_compute_subgroup_metrics_output_structure(self, sample_evaluation_data):
        evaluator = SubgroupEvaluator(
            subgroup_col="fitzpatrick_tone",
            target_col="label",
            pred_col="pred_prob",
        )
        results = evaluator.compute_subgroup_metrics(sample_evaluation_data)

        assert "I-II" in results
        assert "III-IV" in results
        assert "V-VI" in results

        for group, m in results.items():
            assert "sample_count" in m
            assert "positive_count" in m
            assert "negative_count" in m
            assert "auroc" in m
            assert "sensitivity" in m
            assert "specificity" in m
            assert "f1" in m
            assert 0.0 <= m["auroc"] <= 1.0 or np.isnan(m["auroc"])

    def test_zero_disparity_for_identical_subgroup_distributions(self):
        """When subgroups have identical predictions and labels, disparity must be 0."""
        df = pd.DataFrame({
            "fitzpatrick_tone": ["GroupA"] * 4 + ["GroupB"] * 4,
            "label": [1, 0, 1, 0] * 2,
            "pred_prob": [0.9, 0.1, 0.85, 0.2] * 2,
        })
        evaluator = SubgroupEvaluator(
            subgroup_col="fitzpatrick_tone",
            target_col="label",
            pred_col="pred_prob",
        )
        res = evaluator.compute_equalized_odds_disparity(df, threshold=0.5)
        assert np.isclose(res["equalized_odds_disparity"], 0.0)
        assert np.isclose(res["max_tpr_gap"], 0.0)
        assert np.isclose(res["max_fpr_gap"], 0.0)

    def test_maximum_disparity_divergent_groups(self):
        """Group 1: 100% TPR, 0% FPR. Group 2: 0% TPR, 100% FPR -> EOD = 2.0."""
        df = pd.DataFrame({
            "fitzpatrick_tone": ["Group1"] * 4 + ["Group2"] * 4,
            "label": [1, 1, 0, 0] + [1, 1, 0, 0],
            # Group 1: perfect predictions
            # Group 2: completely inverted predictions
            "pred_prob": [0.9, 0.8, 0.1, 0.2] + [0.1, 0.2, 0.9, 0.8],
        })
        evaluator = SubgroupEvaluator()
        res = evaluator.compute_equalized_odds_disparity(df, threshold=0.5)
        assert np.isclose(res["equalized_odds_disparity"], 2.0)
        assert np.isclose(res["max_tpr_gap"], 1.0)
        assert np.isclose(res["max_fpr_gap"], 1.0)

    def test_demographic_parity_difference(self):
        df = pd.DataFrame({
            "fitzpatrick_tone": ["Light"] * 4 + ["Dark"] * 4,
            "label": [1, 0, 1, 0] * 2,
            # Light: 100% selected at threshold 0.5
            # Dark: 0% selected at threshold 0.5
            "pred_prob": [0.8, 0.9, 0.7, 0.6] + [0.1, 0.2, 0.3, 0.4],
        })
        evaluator = SubgroupEvaluator()
        dp_diff = evaluator.compute_demographic_parity_difference(df, threshold=0.5)
        assert np.isclose(dp_diff, 1.0)

    def test_small_subgroup_sample_size_guard(self):
        """Subgroup with < 5 samples must gracefully set metrics to NaN without crashing."""
        df = pd.DataFrame({
            "fitzpatrick_tone": ["Common"] * 20 + ["TinyRare"] * 2,
            "label": [1, 0] * 10 + [1, 0],
            "pred_prob": [0.6] * 22,
        })
        evaluator = SubgroupEvaluator()
        res = evaluator.compute_subgroup_metrics(df, min_samples_per_subgroup=5)
        assert "TinyRare" in res
        assert np.isnan(res["TinyRare"]["auroc"])
        assert res["TinyRare"]["sample_count"] == 2

    def test_missing_column_raises_key_error(self):
        df = pd.DataFrame({"wrong_col": [1, 2, 3]})
        evaluator = SubgroupEvaluator()
        with pytest.raises(KeyError, match="DataFrame is missing required columns"):
            evaluator.compute_subgroup_metrics(df)
