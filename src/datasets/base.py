"""
Abstract Base Dataset Contract for Dermatological Image Modalities

Defines standardized interfaces for image retrieval, metadata parsing, unlesioned
skin extraction, and Individual Typology Angle (ITA) colorimetry.
"""

from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseDermatologyDataset(ABC):
    """
    Abstract base class for all dermatological datasets (ISIC, Fitzpatrick17k, PAD-UFES-20).
    Enforces unified interfaces for tone metadata and multi-modal image evaluation.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        image_dir: Union[str, Path],
        transform: Optional[Any] = None,
        image_col: str = "image_id",
        label_col: str = "label",
        tone_col: Optional[str] = "fitzpatrick_tone",
        domain_name: str = "generic",
    ) -> None:
        """
        Initialize the dataset contract.

        Args:
            metadata_df: DataFrame containing sample identifiers and clinical labels.
            image_dir: Path to directory containing raw image files.
            transform: Optional transformation or augmentation pipeline.
            image_col: Column containing image file basenames or IDs.
            label_col: Column containing ground-truth diagnostic labels.
            tone_col: Optional column containing Fitzpatrick or ITA tone categories.
            domain_name: Modality/dataset origin identifier (e.g. 'dermoscopy', 'clinical').
        """
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_col = image_col
        self.label_col = label_col
        self.tone_col = tone_col
        self.domain_name = domain_name

    def __len__(self) -> int:
        return len(self.metadata_df)

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves sample by index.

        Must return a dictionary containing:
        - 'image': Transformed tensor or array [C, H, W]
        - 'label': Diagnostic target class (integer or float)
        - 'image_id': Sample identifier
        - 'skin_tone': Fitzpatrick type or ITA bucket (if present)
        - 'domain': Modality/dataset domain tag
        """
        pass

    @staticmethod
    def calculate_ita(l_star: float, b_star: float, eps: float = 1e-6) -> float:
        """
        Computes Individual Typology Angle (ITA) from CIE L* (luminance) and b* (yellow-blue).

        Mathematical formulation:
        $$\\text{ITA} = \\arctan\\left(\\frac{L^* - 50}{b^*}\\right) \\times \\frac{180}{\\pi}$$

        Mathematical & Numerical Stability Corrections:
        1. Non-finite values: Returns np.nan for NaN or infinite L* or b*.
        2. Biological bounds: Validates L* within physical CIE limits [0, 100]. Returns np.nan if out of bounds.
        3. Epsilon sign guard: When |b*| < eps and b* != 0, preserves the mathematical sign
           using copysign(eps, b*). Naively using +eps flips the sign by 180 degrees when b* is negative.
        4. Exact zero denominator:
           - If L* == 50.0 and b* == 0.0, the quotient is 0/0 (indeterminate) -> returns np.nan.
           - If L* > 50.0 and b* == 0.0, limit is +90.0 degrees.
           - If L* < 50.0 and b* == 0.0, limit is -90.0 degrees.

        Note: ITA is an objective colorimetric measurement in degrees (-90 to +90) of local skin
        reflectance and pigmentation; it is NOT clinically or biologically interchangeable with
        Fitzpatrick phototype (I-VI), which measures genetic ancestry and sunburn/tanning response.

        Args:
            l_star: Mean luminance across unlesioned skin patch [0, 100].
            b_star: Mean yellow-blue chromaticity across unlesioned skin patch.
            eps: Epsilon guard for near-zero denominators.

        Returns:
            ITA angle in degrees (-90.0 to +90.0), or np.nan if invalid/indeterminate.
        """
        if not np.isfinite(l_star) or not np.isfinite(b_star):
            return float(np.nan)

        # Validate physical CIE L* bounds
        if l_star < 0.0 or l_star > 100.0:
            logger.warning("CIE L* value %.2f out of physical bounds [0, 100]. Returning NaN.", l_star)
            return float(np.nan)

        delta_l = l_star - 50.0

        if abs(b_star) >= eps:
            denominator = b_star
        elif b_star == 0.0:
            if delta_l == 0.0:
                # 0 / 0 indeterminate
                return float(np.nan)
            return 90.0 if delta_l > 0.0 else -90.0
        else:
            # Near-zero: preserve mathematical sign of b* to prevent 180-degree sign flip
            denominator = float(np.copysign(eps, b_star))

        return float(np.arctan(delta_l / denominator) * (180.0 / np.pi))

    @staticmethod
    def map_ita_to_bucket(ita: float) -> str:
        """
        Maps continuous ITA degree measurement to one of the 6 standard dermatological buckets.
        Guarantees that NaN or indeterminate measurements are explicitly returned as
        'Unknown / Indeterminate' rather than silently defaulting to 'Dark'.

        Args:
            ita: Angle in degrees (-90.0 to +90.0).

        Returns:
            Dermatological skin-tone bucket string.
        """
        if ita is None or not np.isfinite(ita):
            return "Unknown / Indeterminate"

        if ita > 55.0:
            return "Very Light"
        elif 41.0 < ita <= 55.0:
            return "Light"
        elif 28.0 < ita <= 41.0:
            return "Intermediate"
        elif 10.0 < ita <= 28.0:
            return "Tan"
        elif -30.0 < ita <= 10.0:
            return "Brown"
        else:
            return "Dark"

    def get_subgroup_distribution(self) -> Dict[str, int]:
        """Returns frequency counts of samples across skin-tone subgroups."""
        if not self.tone_col or self.tone_col not in self.metadata_df.columns:
            return {"unannotated": len(self.metadata_df)}
        return self.metadata_df[self.tone_col].value_counts().to_dict()
