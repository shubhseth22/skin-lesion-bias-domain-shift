"""
Unit tests for BaseDermatologyDataset, ITA Calculation, and PeripheralSkinExtractor
Exercises actual source code in src/datasets/base.py and src/datasets/skin_extractor.py
"""

import numpy as np
import pytest
from src.datasets.base import BaseDermatologyDataset
from src.datasets.skin_extractor import PeripheralSkinExtractor, SkinExtractionResult


class TestITACalculationAndStability:
    """Validates mathematical correctness, boundary limits, and epsilon sign stability."""

    def test_standard_ita_light_skin(self):
        # L* = 70 (high lightness), b* = 15 (moderate yellow)
        ita = BaseDermatologyDataset.calculate_ita(l_star=70.0, b_star=15.0)
        assert np.isfinite(ita)
        assert ita > 41.0
        assert BaseDermatologyDataset.map_ita_to_bucket(ita) in ["Light", "Very Light"]

    def test_standard_ita_dark_skin(self):
        # L* = 30 (dark pigmentation), b* = 20
        ita = BaseDermatologyDataset.calculate_ita(l_star=30.0, b_star=20.0)
        assert np.isfinite(ita)
        assert ita < -30.0
        assert BaseDermatologyDataset.map_ita_to_bucket(ita) == "Dark"

    def test_epsilon_sign_guard_negative_near_zero(self):
        """
        Critical Fix 5: When b* is negative and near zero (e.g. -1e-8),
        naive +eps incorrectly flips the angle to +90 degrees.
        The corrected guard preserves the negative sign (yielding -90 degrees).
        """
        l_star = 70.0  # L* > 50 -> (L* - 50) > 0
        b_neg_small = -1e-8

        ita = BaseDermatologyDataset.calculate_ita(l_star=l_star, b_star=b_neg_small)
        # Positive numerator / negative near-zero denominator -> -infinity -> -90 degrees
        assert np.isclose(ita, -90.0, atol=1e-2)
        assert ita < 0.0

    def test_epsilon_sign_guard_positive_near_zero(self):
        l_star = 70.0
        b_pos_small = 1e-8

        ita = BaseDermatologyDataset.calculate_ita(l_star=l_star, b_star=b_pos_small)
        # Positive numerator / positive near-zero denominator -> +infinity -> +90 degrees
        assert np.isclose(ita, 90.0, atol=1e-2)
        assert ita > 0.0

    def test_exact_zero_denominator_indeterminate(self):
        """When L* == 50.0 and b* == 0.0, (0/0) is mathematically indeterminate -> NaN."""
        ita = BaseDermatologyDataset.calculate_ita(l_star=50.0, b_star=0.0)
        assert np.isnan(ita)
        assert BaseDermatologyDataset.map_ita_to_bucket(ita) == "Unknown / Indeterminate"

    def test_out_of_bounds_luminance_returns_nan(self):
        """CIE L* must be within physical limits [0, 100]."""
        ita_neg = BaseDermatologyDataset.calculate_ita(l_star=-10.0, b_star=15.0)
        assert np.isnan(ita_neg)

        ita_high = BaseDermatologyDataset.calculate_ita(l_star=150.0, b_star=15.0)
        assert np.isnan(ita_high)

    def test_non_finite_inputs_return_nan(self):
        assert np.isnan(BaseDermatologyDataset.calculate_ita(l_star=float("nan"), b_star=15.0))
        assert np.isnan(BaseDermatologyDataset.calculate_ita(l_star=60.0, b_star=float("inf")))

    def test_map_ita_to_bucket_never_defaults_nan_to_dark(self):
        """Critical Fix 5: NaN must NEVER be silently converted to 'Dark'."""
        assert BaseDermatologyDataset.map_ita_to_bucket(np.nan) == "Unknown / Indeterminate"
        assert BaseDermatologyDataset.map_ita_to_bucket(None) == "Unknown / Indeterminate"
        assert BaseDermatologyDataset.map_ita_to_bucket(float("inf")) == "Unknown / Indeterminate"


class TestPeripheralSkinExtractor:
    """Validates unlesioned peripheral skin patch extraction and artifact rejection."""

    @pytest.fixture
    def synthetic_lesion_image(self):
        """
        Generates a synthetic clinical image (200x200 RGB):
        - Outer border (vignette): black (0, 0, 0)
        - Peripheral skin: warm peach (220, 180, 140)
        - Center lesion: dark brown circle (60, 30, 20)
        """
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        # Fill with peripheral skin
        img[:, :] = [220, 180, 140]

        # Draw black border (outer 8%)
        img[:16, :] = [0, 0, 0]
        img[-16:, :] = [0, 0, 0]
        img[:, :16] = [0, 0, 0]
        img[:, -16:] = [0, 0, 0]

        # Draw central dark lesion (radius 30 pixels)
        y, x = np.ogrid[:200, :200]
        mask = (x - 100) ** 2 + (y - 100) ** 2 <= 30**2
        img[mask] = [60, 30, 20]

        return img

    def test_extract_skin_valid_image(self, synthetic_lesion_image):
        extractor = PeripheralSkinExtractor()
        res = extractor.extract_skin(synthetic_lesion_image, return_mask=True)

        assert res.is_valid is True
        assert res.rejection_reason is None
        assert res.valid_pixel_count > 500
        assert np.isfinite(res.ita)
        assert res.ita_bucket != "Unknown / Indeterminate"
        assert 0.0 <= res.quality_score <= 1.0
        assert res.peripheral_mask is not None

    def test_extract_skin_pure_artifact_image_fails_safely(self):
        """All-black image (or extreme vignette) must be rejected with explicit reason."""
        all_black = np.zeros((200, 200, 3), dtype=np.uint8)
        extractor = PeripheralSkinExtractor()
        res = extractor.extract_skin(all_black)

        assert res.is_valid is False
        assert res.rejection_reason == "insufficient_peripheral_skin_pixels"
        assert np.isnan(res.ita)
        assert res.ita_bucket == "Unknown / Indeterminate"

    def test_extract_skin_blue_marker_rejected(self):
        """Blue surgical marker ink must be rejected by chromaticity filter."""
        blue_img = np.zeros((200, 200, 3), dtype=np.uint8)
        blue_img[:, :] = [20, 30, 230]  # Saturated blue ink
        extractor = PeripheralSkinExtractor()
        res = extractor.extract_skin(blue_img)

        assert res.is_valid is False
        assert np.isnan(res.ita)

    def test_extract_skin_invalid_dimensions(self):
        extractor = PeripheralSkinExtractor()
        # 2D grayscale
        res_2d = extractor.extract_skin(np.zeros((100, 100), dtype=np.uint8))
        assert res_2d.is_valid is False
        assert res_2d.rejection_reason == "invalid_image_shape"
