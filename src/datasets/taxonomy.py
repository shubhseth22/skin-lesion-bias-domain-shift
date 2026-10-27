"""
Dermatological Label Taxonomy and Clinical Harmonization Module

Provides explicit, clinically grounded label mappings across heterogeneous
dermatological benchmark datasets (ISIC 2019/2020, PAD-UFES-20, Fitzpatrick 17k)
into a standardized diagnostic taxonomy without inventing clinical equivalences.
"""

from enum import Enum
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HarmonizedCategory(str, Enum):
    """
    Standardized dermatological diagnostic categories.
    Preserves distinctions between neoplasms and inflammatory/non-neoplastic diseases.
    """

    MALIGNANT = "malignant"
    BENIGN = "benign"
    PRE_MALIGNANT = "pre_malignant"
    NON_NEOPLASTIC = "non_neoplastic"
    AMBIGUOUS = "ambiguous"
    UNMAPPED = "unmapped"


# ==============================================================================
# Clinical Mapping Tables with Explicit Rationales
# ==============================================================================

ISIC2019_MAPPING: Dict[str, Dict[str, Any]] = {
    "MEL": {
        "category": HarmonizedCategory.MALIGNANT,
        "clinical_name": "Melanoma",
        "rationale": "Invasive or in-situ malignant neoplasm of transformed epidermal melanocytes.",
    },
    "BCC": {
        "category": HarmonizedCategory.MALIGNANT,
        "clinical_name": "Basal Cell Carcinoma",
        "rationale": "Malignant cutaneous epithelial neoplasm arising from basal keratinocytes.",
    },
    "SCC": {
        "category": HarmonizedCategory.MALIGNANT,
        "clinical_name": "Squamous Cell Carcinoma",
        "rationale": "Malignant invasive carcinoma of epidermal keratinocytes.",
    },
    "NV": {
        "category": HarmonizedCategory.BENIGN,
        "clinical_name": "Melanocytic Nevus",
        "rationale": "Common benign proliferation of cutaneous melanocytes (mole).",
    },
    "BKL": {
        "category": HarmonizedCategory.BENIGN,
        "clinical_name": "Benign Keratosis",
        "rationale": "Heterogeneous category encompassing seborrheic keratosis, solar lentigo, and lichen-planus-like keratosis.",
    },
    "DF": {
        "category": HarmonizedCategory.BENIGN,
        "clinical_name": "Dermatofibroma",
        "rationale": "Benign fibrohistiocytic dermal nodule.",
    },
    "VASC": {
        "category": HarmonizedCategory.BENIGN,
        "clinical_name": "Vascular Lesion",
        "rationale": "Benign vascular proliferations including cherry angiomas, angiokeratomas, and pyogenic granulomas.",
    },
    "AK": {
        "category": HarmonizedCategory.PRE_MALIGNANT,
        "clinical_name": "Actinic Keratosis",
        "rationale": "Dysplastic epidermal keratinocytic neoplasia with malignant potential; clinically pre-malignant.",
    },
}

PAD_UFES_20_MAPPING: Dict[str, Dict[str, Any]] = {
    "MEL": {
        "category": HarmonizedCategory.MALIGNANT,
        "clinical_name": "Melanoma",
        "rationale": "Biopsy-confirmed malignant melanoma.",
    },
    "BCC": {
        "category": HarmonizedCategory.MALIGNANT,
        "clinical_name": "Basal Cell Carcinoma",
        "rationale": "Biopsy-confirmed basal cell carcinoma.",
    },
    "SCC": {
        "category": HarmonizedCategory.MALIGNANT,
        "clinical_name": "Squamous Cell Carcinoma",
        "rationale": "Biopsy-confirmed squamous cell carcinoma.",
    },
    "NEV": {
        "category": HarmonizedCategory.BENIGN,
        "clinical_name": "Nevus",
        "rationale": "Clinically or histopathologically confirmed benign melanocytic nevus (equivalent to ISIC 'NV').",
    },
    "SEK": {
        "category": HarmonizedCategory.BENIGN,
        "clinical_name": "Seborrheic Keratosis",
        "rationale": "Specific benign epidermal tumor. Clinically represents a subset of ISIC 'BKL'.",
    },
    "ACK": {
        "category": HarmonizedCategory.PRE_MALIGNANT,
        "clinical_name": "Actinic Keratosis",
        "rationale": "Portuguese abbreviation (ceratose actínica) for actinic keratosis; pre-malignant lesion.",
    },
}

FITZPATRICK17K_THREE_TIER_MAPPING: Dict[str, Dict[str, Any]] = {
    "malignant": {
        "category": HarmonizedCategory.MALIGNANT,
        "clinical_name": "Malignant Skin Neoplasm",
        "rationale": "Aggregated tier of cutaneous malignant lesions (Melanoma, BCC, SCC, cutaneous T-cell lymphoma).",
    },
    "benign": {
        "category": HarmonizedCategory.BENIGN,
        "clinical_name": "Benign Skin Neoplasm",
        "rationale": "Aggregated tier of benign neoplasms (Nevi, Seborrheic Keratoses, Hemangiomas).",
    },
    "non-neoplastic": {
        "category": HarmonizedCategory.NON_NEOPLASTIC,
        "clinical_name": "Non-Neoplastic Dermatosis",
        "rationale": "Inflammatory, autoimmune, genetic, and infectious skin diseases (e.g., eczema, psoriasis, acne, alopecia). Must not be conflated with benign neoplasms.",
    },
}


class LabelHarmonizer:
    """
    Harmonizes dataset-specific labels into standardized taxonomic categories.
    Guarantees that original labels are preserved and unmapped/ambiguous cases
    are explicitly surfaced rather than silently discarded or forced into wrong classes.
    """

    DATASET_MAPPINGS = {
        "isic2019": ISIC2019_MAPPING,
        "isic2020": ISIC2019_MAPPING,
        "pad_ufes_20": PAD_UFES_20_MAPPING,
        "fitzpatrick17k": FITZPATRICK17K_THREE_TIER_MAPPING,
    }

    def __init__(self, custom_mappings: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None) -> None:
        """
        Initialize the harmonizer with default and optional custom mappings.

        Args:
            custom_mappings: Optional dictionary extending or overriding dataset mappings.
        """
        self.mappings = dict(self.DATASET_MAPPINGS)
        if custom_mappings:
            for ds, mapping in custom_mappings.items():
                self.validate_mapping_table(mapping, ds)
                self.mappings[ds] = mapping

    @staticmethod
    def validate_mapping_table(mapping: Dict[str, Dict[str, Any]], dataset_name: str) -> None:
        """
        Validates mapping table integrity, requiring clinical rationale and valid categories.

        Args:
            mapping: Dictionary mapping source label strings to metadata dictionaries.
            dataset_name: Name of the dataset being validated.

        Raises:
            ValueError: If invalid categories, missing rationales, or empty mappings are found.
        """
        if not mapping:
            raise ValueError(f"Mapping table for '{dataset_name}' cannot be empty.")

        seen_keys: Set[str] = set()
        for key, entry in mapping.items():
            if key in seen_keys:
                raise ValueError(f"Duplicate mapping key '{key}' detected in '{dataset_name}'.")
            seen_keys.add(key)

            if not isinstance(entry, dict):
                raise ValueError(f"Entry for '{key}' in '{dataset_name}' must be a dictionary.")

            if "category" not in entry:
                raise ValueError(f"Missing required 'category' key for '{key}' in '{dataset_name}'.")

            if not isinstance(entry["category"], HarmonizedCategory):
                raise ValueError(
                    f"Invalid category '{entry['category']}' for '{key}'. Must be a HarmonizedCategory."
                )

            if "rationale" not in entry or not entry["rationale"].strip():
                raise ValueError(
                    f"Missing clinical 'rationale' for mapping '{key}' in '{dataset_name}'."
                )

    def harmonize_dataset(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        source_col: str = "label",
        target_col: str = "harmonized_label",
        preserve_original: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Harmonizes source labels into the standardized taxonomy.

        Args:
            df: Input DataFrame containing clinical metadata.
            dataset_name: Identifier of the source dataset (e.g. 'isic2019', 'pad_ufes_20').
            source_col: Column containing raw source labels.
            target_col: Column name to write harmonized category values.
            preserve_original: If True, guarantees source_col is kept unmodified.

        Returns:
            Tuple of (harmonized_df, report_dict).

        Raises:
            KeyError: If source_col is not present in df.
            ValueError: If dataset_name has no registered mapping table.
        """
        if source_col not in df.columns:
            raise KeyError(f"Source label column '{source_col}' not found in DataFrame.")

        canonical_name = dataset_name.lower().replace("-", "_")
        if canonical_name not in self.mappings:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. Available: {list(self.mappings.keys())}"
            )

        mapping_table = self.mappings[canonical_name]
        out_df = df.copy()

        if preserve_original and source_col == target_col:
            raise ValueError("target_col cannot overwrite source_col when preserve_original=True.")

        harmonized_values: List[str] = []
        unmapped_labels: Set[str] = set()
        category_counts: Dict[str, int] = {cat.value: 0 for cat in HarmonizedCategory}

        for val in out_df[source_col]:
            str_val = str(val).strip()
            if str_val in mapping_table:
                cat = mapping_table[str_val]["category"].value
            else:
                cat = HarmonizedCategory.UNMAPPED.value
                unmapped_labels.add(str_val)

            harmonized_values.append(cat)
            category_counts[cat] += 1

        out_df[target_col] = harmonized_values
        out_df["is_unmapped"] = out_df[target_col] == HarmonizedCategory.UNMAPPED.value
        out_df["is_ambiguous"] = out_df[target_col] == HarmonizedCategory.AMBIGUOUS.value

        if unmapped_labels:
            logger.warning(
                "Dataset '%s' contains %d unmapped labels: %s",
                dataset_name,
                len(unmapped_labels),
                unmapped_labels,
            )

        report = {
            "dataset_name": dataset_name,
            "total_samples": len(out_df),
            "category_counts": category_counts,
            "unmapped_labels": list(unmapped_labels),
            "unmapped_count": category_counts[HarmonizedCategory.UNMAPPED.value],
            "ambiguous_count": category_counts[HarmonizedCategory.AMBIGUOUS.value],
        }

        return out_df, report

    def to_binary_target(
        self,
        df: pd.DataFrame,
        harmonized_col: str = "harmonized_label",
        binary_col: str = "binary_label",
        pre_malignant_policy: str = "exclude",
        non_neoplastic_policy: str = "exclude",
    ) -> pd.DataFrame:
        """
        Converts harmonized categories to a binary screening target (0 = benign, 1 = malignant).
        Provides explicit, configurable clinical policies for pre-malignant and non-neoplastic samples.

        Args:
            df: Input DataFrame with harmonized labels.
            harmonized_col: Column containing harmonized category strings.
            binary_col: Column name to write binary target (0, 1, or NaN).
            pre_malignant_policy: How to treat pre-malignant lesions:
                - 'malignant': High-sensitivity triage (positive screen).
                - 'benign': Conservative lesion classification.
                - 'exclude': Exclude from binary evaluation (set to NaN).
            non_neoplastic_policy: How to treat non-neoplastic conditions:
                - 'exclude': Exclude from neoplastic benign/malignant triage (set to NaN).
                - 'benign': Treat non-cancerous inflammatory dermatoses as negative.

        Returns:
            DataFrame with binary_col appended.
        """
        if harmonized_col not in df.columns:
            raise KeyError(f"Harmonized column '{harmonized_col}' missing from DataFrame.")

        valid_policies = {"malignant", "benign", "exclude"}
        if pre_malignant_policy not in valid_policies:
            raise ValueError(f"Invalid pre_malignant_policy: {pre_malignant_policy}")
        if non_neoplastic_policy not in {"benign", "exclude"}:
            raise ValueError(f"Invalid non_neoplastic_policy: {non_neoplastic_policy}")

        out_df = df.copy()
        binary_vals = []

        for cat in out_df[harmonized_col]:
            if cat == HarmonizedCategory.MALIGNANT.value:
                binary_vals.append(1)
            elif cat == HarmonizedCategory.BENIGN.value:
                binary_vals.append(0)
            elif cat == HarmonizedCategory.PRE_MALIGNANT.value:
                if pre_malignant_policy == "malignant":
                    binary_vals.append(1)
                elif pre_malignant_policy == "benign":
                    binary_vals.append(0)
                else:
                    binary_vals.append(np.nan)
            elif cat == HarmonizedCategory.NON_NEOPLASTIC.value:
                if non_neoplastic_policy == "benign":
                    binary_vals.append(0)
                else:
                    binary_vals.append(np.nan)
            else:
                # AMBIGUOUS or UNMAPPED
                binary_vals.append(np.nan)

        out_df[binary_col] = binary_vals
        return out_df
