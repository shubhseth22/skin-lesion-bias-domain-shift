"""
Unit tests for Stratified Group Splitting and Domain Adaptation Splitting in src/datasets/splits.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.datasets.splits import DatasetSplitter


@pytest.fixture
def sample_metadata():
    """Generates synthetic metadata with labels, patient IDs, and skin-tone subgroups."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "image_id": [f"IMG_{i:04d}" for i in range(n)],
        "patient_id": [f"PAT_{i // 2:03d}" for i in range(n)],  # ~2 images per patient
        "label": np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
        "fitzpatrick_tone": np.random.choice(["I-II", "III-IV", "V-VI"], size=n, p=[0.5, 0.3, 0.2]),
        "dataset_domain": np.random.choice(["isic", "fitzpatrick17k"], size=n, p=[0.7, 0.3]),
    })


# ==============================================================================
# Critical Fix 1: Stratified Group Splitting Tests
# ==============================================================================

class TestStratifiedGroupSplitting:
    """Validates patient-level group isolation, class stratification, and edge case handling."""

    def test_stratified_group_split_zero_leakage(self, sample_metadata):
        splitter = DatasetSplitter(seed=42)
        train_df, val_df, test_df, meta = splitter.create_stratified_group_split(
            sample_metadata,
            target_col="label",
            group_col="patient_id",
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
        )

        p_train = set(train_df["patient_id"])
        p_val = set(val_df["patient_id"])
        p_test = set(test_df["patient_id"])

        # Strict group disjointness assertions
        assert len(p_train & p_val) == 0, "Patient leakage between train and val"
        assert len(p_train & p_test) == 0, "Patient leakage between train and test"
        assert len(p_val & p_test) == 0, "Patient leakage between val and test"

        # Verify metadata audit
        assert meta["zero_leakage_verified"] is True
        assert meta["method"] == "stratified_group_split"
        assert meta["seed"] == 42
        assert len(train_df) + len(val_df) + len(test_df) == len(sample_metadata)

    def test_stratified_group_split_preserves_class_proportions(self, sample_metadata):
        splitter = DatasetSplitter(seed=42)
        train_df, val_df, test_df, meta = splitter.create_stratified_group_split(
            sample_metadata,
            target_col="label",
            group_col="patient_id",
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
        )
        total_p_positive = (sample_metadata["label"] == 1).mean()
        train_p_positive = (train_df["label"] == 1).mean()
        val_p_positive = (val_df["label"] == 1).mean()
        test_p_positive = (test_df["label"] == 1).mean()

        # Both classes represented in all partitions
        assert train_p_positive > 0
        assert val_p_positive > 0
        assert test_p_positive > 0
        # Positive proportion remains bounded around the empirical mean
        assert abs(train_p_positive - total_p_positive) < 0.10
        assert abs(test_p_positive - total_p_positive) < 0.15

    def test_insufficient_group_class_handling(self):
        """Tests handling when a rare class appears in fewer groups than partitions (< 3)."""
        df = pd.DataFrame({
            "image_id": [f"IMG_{i}" for i in range(30)],
            "patient_id": [f"P_{i // 2}" for i in range(30)],  # 15 patients
            "label": [0] * 28 + [1, 1],  # Class 1 exists in only 1 patient (P_14)
        })
        splitter = DatasetSplitter(seed=42)

        # Mode A: allow_unrepresented_classes=False raises explicit ValueError
        with pytest.raises(ValueError, match="appear in fewer unique groups than partitions"):
            splitter.create_stratified_group_split(
                df,
                target_col="label",
                group_col="patient_id",
                allow_unrepresented_classes=False,
            )

        # Mode B: allow_unrepresented_classes=True documents in metadata without leakage
        train_df, val_df, test_df, meta = splitter.create_stratified_group_split(
            df,
            target_col="label",
            group_col="patient_id",
            allow_unrepresented_classes=True,
        )
        assert 1 in meta["insufficient_group_classes"]
        assert meta["insufficient_group_classes"][1] == 1
        # Patient P_14 is in train, zero leakage
        assert "P_14" in train_df["patient_id"].values
        assert "P_14" not in val_df["patient_id"].values
        assert "P_14" not in test_df["patient_id"].values

    def test_missing_group_identifiers_imputed_as_singletons(self):
        """Verifies missing patient IDs are treated as independent singletons, not lumped together."""
        df = pd.DataFrame({
            "image_id": [f"IMG_{i}" for i in range(40)],
            "patient_id": [f"P_{i}" for i in range(30)] + [None] * 5 + [""] * 5,
            "label": [0] * 20 + [1] * 20,
        })
        splitter = DatasetSplitter(seed=42)
        train_df, val_df, test_df, meta = splitter.create_stratified_group_split(
            df,
            target_col="label",
            group_col="patient_id",
        )
        assert meta["missing_group_imputations"] == 10
        assert len(train_df) + len(val_df) + len(test_df) == 40
        assert meta["zero_leakage_verified"] is True

    def test_reproducibility_with_seed(self, sample_metadata):
        splitter_a = DatasetSplitter(seed=999)
        t_a, v_a, te_a, _ = splitter_a.create_stratified_group_split(sample_metadata)

        splitter_b = DatasetSplitter(seed=999)
        t_b, v_b, te_b, _ = splitter_b.create_stratified_group_split(sample_metadata)

        assert t_a["image_id"].tolist() == t_b["image_id"].tolist()
        assert te_a["image_id"].tolist() == te_b["image_id"].tolist()


# ==============================================================================
# Critical Fix 3: Target Domain Adaptation Splitting Tests
# ==============================================================================

class TestTargetDomainAdaptationSplitting:
    """Validates complete target test isolation and adaptation split properties."""

    @pytest.fixture
    def multi_domain_data(self):
        np.random.seed(42)
        n_source = 150
        n_target = 100
        source = pd.DataFrame({
            "image_id": [f"SRC_{i}" for i in range(n_source)],
            "patient_id": [f"P_SRC_{i // 2}" for i in range(n_source)],
            "dataset_domain": ["isic"] * n_source,
            "label": np.random.choice([0, 1], size=n_source, p=[0.7, 0.3]),
        })
        target = pd.DataFrame({
            "image_id": [f"TGT_{i}" for i in range(n_target)],
            "patient_id": [f"P_TGT_{i // 2}" for i in range(n_target)],
            "dataset_domain": ["fitzpatrick17k"] * n_target,
            "label": np.random.choice([0, 1], size=n_target, p=[0.6, 0.4]),
        })
        return pd.concat([source, target], ignore_index=True)

    def test_unsupervised_adaptation_masks_target_labels(self, multi_domain_data):
        splitter = DatasetSplitter(seed=42)
        splits, meta = splitter.create_domain_adaptation_split(
            multi_domain_data,
            source_domain="isic",
            target_domain="fitzpatrick17k",
            adaptation_mode="unsupervised",
            target_adaptation_ratio=0.50,
            mask_unsupervised_target_labels=True,
        )

        assert meta["adaptation_mode"] == "unsupervised"
        assert meta["target_test_completely_isolated"] is True
        assert meta["target_adaptation_labels_masked"] is True

        # Target adaptation training labels MUST be masked (NaN) to prevent accidental loss calculation
        t_adapt = splits["target_adaptation_train"]
        assert t_adapt["label"].isna().all()
        assert "original_unsupervised_label" in t_adapt.columns

        # Target test data MUST preserve labels for evaluation
        t_test = splits["target_test"]
        assert not t_test["label"].isna().any()

    def test_supervised_adaptation_preserves_target_labels(self, multi_domain_data):
        splitter = DatasetSplitter(seed=42)
        splits, meta = splitter.create_domain_adaptation_split(
            multi_domain_data,
            source_domain="isic",
            target_domain="fitzpatrick17k",
            adaptation_mode="supervised",
            target_adaptation_ratio=0.30,
        )
        assert meta["adaptation_mode"] == "supervised"
        assert not splits["target_adaptation_train"]["label"].isna().any()
        assert not splits["target_test"]["label"].isna().any()

    def test_target_patient_isolation_zero_leakage(self, multi_domain_data):
        splitter = DatasetSplitter(seed=42)
        splits, _ = splitter.create_domain_adaptation_split(
            multi_domain_data,
            source_domain="isic",
            target_domain="fitzpatrick17k",
            group_col="patient_id",
            target_adaptation_ratio=0.50,
        )
        adapt_patients = set(splits["target_adaptation_train"]["patient_id"])
        test_patients = set(splits["target_test"]["patient_id"])

        # Target test patients must NEVER appear in target adaptation training
        assert len(adapt_patients & test_patients) == 0, "Patient leakage into target test set!"

    def test_missing_target_domain_raises_error(self, multi_domain_data):
        splitter = DatasetSplitter(seed=42)
        with pytest.raises(ValueError, match="Target domain 'non_existent' contains 0 records"):
            splitter.create_domain_adaptation_split(
                multi_domain_data,
                source_domain="isic",
                target_domain="non_existent",
            )
