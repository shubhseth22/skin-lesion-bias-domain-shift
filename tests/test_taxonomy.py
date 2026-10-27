"""
Unit tests for Label Harmonizer and Taxonomy Module (src/datasets/taxonomy.py)
"""

import numpy as np
import pandas as pd
import pytest
from src.datasets.taxonomy import (
    FITZPATRICK17K_THREE_TIER_MAPPING,
    ISIC2019_MAPPING,
    PAD_UFES_20_MAPPING,
    HarmonizedCategory,
    LabelHarmonizer,
)


class TestLabelHarmonizer:
    """Validates clinical label mappings, unmapped preservation, and binary policies."""

    def test_isic2019_harmonization(self):
        df = pd.DataFrame({
            "image_id": ["ISIC_01", "ISIC_02", "ISIC_03", "ISIC_04", "ISIC_05"],
            "label": ["MEL", "NV", "BCC", "AK", "BKL"],
        })
        harmonizer = LabelHarmonizer()
        harmonized_df, report = harmonizer.harmonize_dataset(df, "isic2019")

        assert harmonized_df["harmonized_label"].tolist() == [
            HarmonizedCategory.MALIGNANT.value,
            HarmonizedCategory.BENIGN.value,
            HarmonizedCategory.MALIGNANT.value,
            HarmonizedCategory.PRE_MALIGNANT.value,
            HarmonizedCategory.BENIGN.value,
        ]
        # Preserves original labels unmodified
        assert harmonized_df["label"].tolist() == ["MEL", "NV", "BCC", "AK", "BKL"]
        assert report["unmapped_count"] == 0

    def test_pad_ufes_20_harmonization(self):
        df = pd.DataFrame({
            "img_id": ["PAD_01", "PAD_02", "PAD_03", "PAD_04"],
            "diagnostic": ["MEL", "NEV", "SEK", "ACK"],
        })
        harmonizer = LabelHarmonizer()
        harmonized_df, report = harmonizer.harmonize_dataset(
            df, "pad_ufes_20", source_col="diagnostic"
        )

        assert harmonized_df["harmonized_label"].tolist() == [
            HarmonizedCategory.MALIGNANT.value,
            HarmonizedCategory.BENIGN.value,
            HarmonizedCategory.BENIGN.value,
            HarmonizedCategory.PRE_MALIGNANT.value,
        ]
        # Verify original column preserved
        assert harmonized_df["diagnostic"].tolist() == ["MEL", "NEV", "SEK", "ACK"]
        assert report["unmapped_count"] == 0

    def test_fitzpatrick17k_preserves_non_neoplastic(self):
        df = pd.DataFrame({
            "md5hash": ["hash1", "hash2", "hash3"],
            "three_partition_label": ["malignant", "benign", "non-neoplastic"],
        })
        harmonizer = LabelHarmonizer()
        harmonized_df, report = harmonizer.harmonize_dataset(
            df, "fitzpatrick17k", source_col="three_partition_label"
        )

        assert harmonized_df["harmonized_label"].tolist() == [
            HarmonizedCategory.MALIGNANT.value,
            HarmonizedCategory.BENIGN.value,
            HarmonizedCategory.NON_NEOPLASTIC.value,
        ]
        # Crucial: non-neoplastic is NOT converted to benign
        assert harmonized_df["harmonized_label"].iloc[2] == "non_neoplastic"

    def test_unmapped_labels_are_preserved_not_discarded(self):
        df = pd.DataFrame({
            "image_id": ["img1", "img2", "img3"],
            "label": ["MEL", "UNKNOWN_NEW_DIAGNOSIS", "NV"],
        })
        harmonizer = LabelHarmonizer()
        harmonized_df, report = harmonizer.harmonize_dataset(df, "isic2019")

        # Zero rows dropped
        assert len(harmonized_df) == 3
        # Explicit unmapped category
        assert harmonized_df["harmonized_label"].iloc[1] == HarmonizedCategory.UNMAPPED.value
        assert bool(harmonized_df["is_unmapped"].iloc[1]) is True
        assert report["unmapped_count"] == 1
        assert "UNKNOWN_NEW_DIAGNOSIS" in report["unmapped_labels"]

    def test_mapping_table_validation_catches_invalid_category(self):
        invalid_mapping = {
            "TEST": {
                "category": "not_an_enum",  # Invalid
                "clinical_name": "Test",
                "rationale": "Some rationale",
            }
        }
        with pytest.raises(ValueError, match="Must be a HarmonizedCategory"):
            LabelHarmonizer(custom_mappings={"test_ds": invalid_mapping})

    def test_mapping_table_validation_catches_missing_rationale(self):
        invalid_mapping = {
            "TEST": {
                "category": HarmonizedCategory.BENIGN,
                "clinical_name": "Test",
                "rationale": "",  # Empty rationale
            }
        }
        with pytest.raises(ValueError, match="Missing clinical 'rationale'"):
            LabelHarmonizer(custom_mappings={"test_ds": invalid_mapping})

    def test_to_binary_target_pre_malignant_policies(self):
        df = pd.DataFrame({
            "harmonized_label": [
                HarmonizedCategory.MALIGNANT.value,
                HarmonizedCategory.BENIGN.value,
                HarmonizedCategory.PRE_MALIGNANT.value,
            ]
        })
        harmonizer = LabelHarmonizer()

        # Policy 1: Exclude pre-malignant (sets to NaN)
        df_ex = harmonizer.to_binary_target(df, pre_malignant_policy="exclude")
        assert df_ex["binary_label"].iloc[0] == 1.0
        assert df_ex["binary_label"].iloc[1] == 0.0
        assert np.isnan(df_ex["binary_label"].iloc[2])

        # Policy 2: Treat pre-malignant as malignant (high-sensitivity screening)
        df_mal = harmonizer.to_binary_target(df, pre_malignant_policy="malignant")
        assert df_mal["binary_label"].iloc[2] == 1.0

        # Policy 3: Treat pre-malignant as benign
        df_ben = harmonizer.to_binary_target(df, pre_malignant_policy="benign")
        assert df_ben["binary_label"].iloc[2] == 0.0
