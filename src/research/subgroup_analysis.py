"""
Subgroup Disparity Analysis Module

Evaluates dermatological classification models across demographic subgroups
defined by Fitzpatrick skin-tone types (I-VI) and Individual Typology Angle (ITA)
brackets. Computes disaggregated performance metrics and fairness disparities.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

logger = logging.getLogger(__name__)


class SubgroupEvaluator:
    """
    Evaluates model predictions across skin-tone subgroups to detect and quantify
    diagnostic bias and disparate impact.
    """

    def __init__(
        self,
        subgroup_col: str = "fitzpatrick_tone",
        target_col: str = "label",
        pred_col: str = "pred_prob",
    ) -> None:
        """
        Initialize the subgroup evaluator.

        Args:
            subgroup_col: Column name identifying skin-tone subgroup (e.g. Fitzpatrick scale I-VI).
            target_col: Column name containing ground-truth diagnostic labels.
            pred_col: Column name containing predicted probabilities.
        """
        self.subgroup_col = subgroup_col
        self.target_col = target_col
        self.pred_col = pred_col

    def _validate_input_df(self, df: pd.DataFrame) -> None:
        required_cols = [self.subgroup_col, self.target_col, self.pred_col]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise KeyError(f"DataFrame is missing required columns: {missing}")

    def compute_subgroup_metrics(
        self,
        df: pd.DataFrame,
        metrics: Optional[List[str]] = None,
        threshold: float = 0.5,
        min_samples_per_subgroup: int = 5,
    ) -> Dict[str, Dict[str, Union[float, int]]]:
        """
        Compute performance metrics disaggregated across each subgroup.

        Args:
            df: DataFrame containing ground truth, predictions, and demographic subgroup labels.
            metrics: Optional list of metrics to compute (defaults to AUROC, Sensitivity, Specificity, F1).
            threshold: Probability threshold for binary classification metrics.
            min_samples_per_subgroup: Minimum subgroup size required to compute metrics.

        Returns:
            Dictionary mapping each subgroup label to its computed metric dictionary.
        """
        self._validate_input_df(df)
        requested_metrics = metrics or ["auroc", "sensitivity", "specificity", "f1"]
        results: Dict[str, Dict[str, Union[float, int]]] = {}

        for group_name, g_df in df.groupby(self.subgroup_col):
            g_str = str(group_name)
            n_samples = len(g_df)
            y_true = g_df[self.target_col].to_numpy()
            y_pred = g_df[self.pred_col].to_numpy()
            y_bin = (y_pred >= threshold).astype(int)

            n_pos = int(np.sum(y_true == 1))
            n_neg = int(np.sum(y_true == 0))

            g_metrics: Dict[str, Union[float, int]] = {
                "sample_count": n_samples,
                "positive_count": n_pos,
                "negative_count": n_neg,
            }

            if n_samples < min_samples_per_subgroup:
                logger.warning(
                    "Subgroup '%s' has only %d samples (< %d). Setting metrics to NaN.",
                    g_str,
                    n_samples,
                    min_samples_per_subgroup,
                )
                for m in requested_metrics:
                    g_metrics[m] = float(np.nan)
                results[g_str] = g_metrics
                continue

            # AUROC: Requires at least 1 positive and 1 negative sample
            if "auroc" in requested_metrics:
                if n_pos >= 1 and n_neg >= 1:
                    try:
                        g_metrics["auroc"] = float(roc_auc_score(y_true, y_pred))
                    except ValueError:
                        g_metrics["auroc"] = float(np.nan)
                else:
                    g_metrics["auroc"] = float(np.nan)

            # Sensitivity / Recall / TPR
            if "sensitivity" in requested_metrics:
                if n_pos > 0:
                    tpr = np.sum((y_bin == 1) & (y_true == 1)) / n_pos
                    g_metrics["sensitivity"] = float(tpr)
                else:
                    g_metrics["sensitivity"] = float(np.nan)

            # Specificity / TNR
            if "specificity" in requested_metrics:
                if n_neg > 0:
                    tnr = np.sum((y_bin == 0) & (y_true == 0)) / n_neg
                    g_metrics["specificity"] = float(tnr)
                else:
                    g_metrics["specificity"] = float(np.nan)

            # F1 Score
            if "f1" in requested_metrics:
                if n_pos > 0:
                    g_metrics["f1"] = float(f1_score(y_true, y_bin, zero_division=0))
                else:
                    g_metrics["f1"] = float(np.nan)

            results[g_str] = g_metrics

        return results

    def compute_equalized_odds_disparity(
        self,
        df: pd.DataFrame,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Compute True Positive Rate (TPR) and False Positive Rate (FPR) parity gaps
        between privileged and unprivileged skin-tone subgroups.

        Mathematical definition:
        $$\\Delta_{\\text{EOD}} = \\max_{g_1, g_2} (|\\text{TPR}_{g_1} - \\text{TPR}_{g_2}| + |\\text{FPR}_{g_1} - \\text{FPR}_{g_2}|)$$

        Args:
            df: DataFrame containing predictions and ground truth.
            threshold: Decision threshold for binarizing probabilities.

        Returns:
            Dictionary with max TPR gap, max FPR gap, overall Equalized Odds disparity,
            and individual subgroup rates.
        """
        self._validate_input_df(df)

        subgroup_rates: Dict[str, Dict[str, float]] = {}
        for group_name, g_df in df.groupby(self.subgroup_col):
            g_str = str(group_name)
            y_true = g_df[self.target_col].to_numpy()
            y_pred = g_df[self.pred_col].to_numpy()
            y_bin = (y_pred >= threshold).astype(int)

            n_pos = np.sum(y_true == 1)
            n_neg = np.sum(y_true == 0)

            tpr = float(np.sum((y_bin == 1) & (y_true == 1)) / n_pos) if n_pos > 0 else float(np.nan)
            fpr = float(np.sum((y_bin == 1) & (y_true == 0)) / n_neg) if n_neg > 0 else float(np.nan)

            subgroup_rates[g_str] = {"tpr": tpr, "fpr": fpr}

        valid_groups = [g for g, r in subgroup_rates.items() if np.isfinite(r["tpr"]) and np.isfinite(r["fpr"])]
        if len(valid_groups) < 2:
            return {
                "max_tpr_gap": 0.0,
                "max_fpr_gap": 0.0,
                "equalized_odds_disparity": 0.0,
                "subgroup_rates": subgroup_rates,
            }

        max_tpr_gap = 0.0
        max_fpr_gap = 0.0
        max_eod = 0.0

        for i, g1 in enumerate(valid_groups):
            for g2 in valid_groups[i + 1 :]:
                tpr_gap = abs(subgroup_rates[g1]["tpr"] - subgroup_rates[g2]["tpr"])
                fpr_gap = abs(subgroup_rates[g1]["fpr"] - subgroup_rates[g2]["fpr"])
                eod = tpr_gap + fpr_gap

                if tpr_gap > max_tpr_gap:
                    max_tpr_gap = tpr_gap
                if fpr_gap > max_fpr_gap:
                    max_fpr_gap = fpr_gap
                if eod > max_eod:
                    max_eod = eod

        return {
            "max_tpr_gap": float(max_tpr_gap),
            "max_fpr_gap": float(max_fpr_gap),
            "equalized_odds_disparity": float(max_eod),
            "subgroup_rates": subgroup_rates,
        }

    def compute_demographic_parity_difference(
        self,
        df: pd.DataFrame,
        threshold: float = 0.5,
    ) -> float:
        """
        Compute maximum difference in selection rate across demographic subgroups.

        $$\\Delta_{\\text{DP}} = \\max_g P(\\hat{Y}=1 \\mid G=g) - \\min_g P(\\hat{Y}=1 \\mid G=g)$$

        Args:
            df: DataFrame containing predictions.
            threshold: Decision threshold.

        Returns:
            Float representing demographic parity difference in [0.0, 1.0].
        """
        self._validate_input_df(df)
        selection_rates: List[float] = []

        for _, g_df in df.groupby(self.subgroup_col):
            if len(g_df) == 0:
                continue
            y_pred = g_df[self.pred_col].to_numpy()
            rate = float(np.mean(y_pred >= threshold))
            selection_rates.append(rate)

        if len(selection_rates) < 2:
            return 0.0

        return float(max(selection_rates) - min(selection_rates))
