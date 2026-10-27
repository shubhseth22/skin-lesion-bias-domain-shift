"""
Statistical Testing and Significance Suite

Implements rigorous non-parametric statistical hypothesis tests, bootstrap confidence
intervals (BCa / percentile), permutation tests for subgroup disparities, and DeLong's
test for comparing ROC curves across models and skin tones.
"""

from typing import Any, Callable, Dict, Tuple, Optional
import numpy as np


class StatisticalTester:
    """
    Provides statistical inference tools to establish confidence intervals and
    statistical significance for subgroup fairness and domain shift metrics.
    """

    def __init__(self, seed: int = 42) -> None:
        """
        Initialize the statistical tester with a fixed seed for reproducible resampling.

        Args:
            seed: Random seed for bootstrapping and permutation tests.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def bootstrap_confidence_interval(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_fn: Callable[[np.ndarray, np.ndarray], float],
        n_bootstraps: int = 1000,
        ci_level: float = 0.95,
    ) -> Tuple[float, float, float]:
        """
        Compute non-parametric bootstrap confidence interval for any diagnostic metric.

        Args:
            y_true: Array of ground-truth labels.
            y_pred: Array of predictions or probabilities.
            metric_fn: Callable metric function (y_true, y_pred) -> score.
            n_bootstraps: Number of bootstrap resamples.
            ci_level: Confidence level (e.g., 0.95 for 95% CI).

        Returns:
            Tuple of (point_estimate, lower_bound, upper_bound).
        """
        raise NotImplementedError("Bootstrap confidence interval pending implementation.")

    def permutation_test_disparity(
        self,
        y_true_g1: np.ndarray,
        y_pred_g1: np.ndarray,
        y_true_g2: np.ndarray,
        y_pred_g2: np.ndarray,
        metric_fn: Callable[[np.ndarray, np.ndarray], float],
        n_permutations: int = 1000,
    ) -> Dict[str, float]:
        """
        Test the null hypothesis that there is no performance disparity between group 1 and group 2.

        Args:
            y_true_g1: Ground truth for group 1.
            y_pred_g1: Predictions for group 1.
            y_true_g2: Ground truth for group 2.
            y_pred_g2: Predictions for group 2.
            metric_fn: Diagnostic metric function.
            n_permutations: Number of permutation iterations.

        Returns:
            Dictionary with observed disparity and empirical two-tailed p-value.
        """
        raise NotImplementedError("Permutation disparity testing pending implementation.")
