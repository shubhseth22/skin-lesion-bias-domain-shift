"""
Peripheral Skin Extraction and Colorimetric Estimation Heuristic

Extracts peripheral, unlesioned skin patches from clinical and dermoscopic images
to compute local CIE L*a*b* colorimetry and Individual Typology Angle (ITA) while
rejecting borders, vignettes, central lesions, hair, rulers, and surgical ink.

Scientific Limitations & Clinical Caution:
1. Peripheral skin color does NOT represent a patient's true or constitutional
   phototype (sun-exposed sites, tanning, and erythema alter local measurements).
2. Image acquisition differences (camera sensor white balance, polarized vs.
   non-polarized lighting, immersion gel, vignetting) introduce severe covariate
   shifts that can distort colorimetric readings.
3. This module NEVER synthesizes or fabricates skin-tone labels; if no reliable
   peripheral skin patch can be isolated, it marks the sample as indeterminate.
"""

from dataclasses import dataclass
import logging
from typing import Optional, Tuple, Union
import numpy as np

from .base import BaseDermatologyDataset

logger = logging.getLogger(__name__)


@dataclass
class SkinExtractionResult:
    """Encapsulates peripheral skin extraction metrics and diagnostic quality."""

    is_valid: bool
    l_mean: float
    a_mean: float
    b_mean: float
    ita: float
    ita_bucket: str
    quality_score: float  # [0.0, 1.0]
    valid_pixel_count: int
    rejection_reason: Optional[str] = None
    peripheral_mask: Optional[np.ndarray] = None


class PeripheralSkinExtractor:
    """
    Automated heuristic to locate unlesioned peripheral skin patches in dermatological imagery.
    """

    def __init__(
        self,
        border_margin_ratio: float = 0.08,
        central_exclusion_radius_ratio: float = 0.35,
        min_valid_pixels: int = 100,
        min_valid_fraction: float = 0.005,
    ) -> None:
        """
        Initialize the peripheral skin extractor.

        Args:
            border_margin_ratio: Fraction of outer image boundary to crop (rejects vignetting/borders).
            central_exclusion_radius_ratio: Normalized radius from center to exclude (avoids central lesion).
            min_valid_pixels: Minimum count of candidate skin pixels required for valid estimation.
            min_valid_fraction: Minimum fraction of total image pixels required.
        """
        self.border_margin_ratio = border_margin_ratio
        self.central_exclusion_radius_ratio = central_exclusion_radius_ratio
        self.min_valid_pixels = min_valid_pixels
        self.min_valid_fraction = min_valid_fraction

    @staticmethod
    def rgb_to_cielab(rgb_image: np.ndarray) -> np.ndarray:
        """
        Converts standard sRGB image to CIE L*a*b* using standard D65 illuminant conversion.

        Args:
            rgb_image: Float array [H, W, 3] in range [0, 1] or uint8 [0, 255].

        Returns:
            Float array [H, W, 3] where L* in [0, 100], a* in [-128, 127], b* in [-128, 127].
        """
        img = rgb_image.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0

        # sRGB gamma companding to linear RGB
        linear = np.where(
            img > 0.04045,
            np.power((img + 0.055) / 1.055, 2.4),
            img / 12.92,
        )

        # Linear RGB to CIE XYZ (Observer = 2°, Illuminant = D65)
        r, g, b = linear[..., 0], linear[..., 1], linear[..., 2]
        x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
        y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
        z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

        # D65 reference white normalization
        xn, yn, zn = 0.95047, 1.00000, 1.08883
        xr, yr, zr = x / xn, y / yn, z / zn

        epsilon = 0.008856
        kappa = 903.3

        fx = np.where(xr > epsilon, np.cbrt(xr), (kappa * xr + 16.0) / 116.0)
        fy = np.where(yr > epsilon, np.cbrt(yr), (kappa * yr + 16.0) / 116.0)
        fz = np.where(zr > epsilon, np.cbrt(zr), (kappa * zr + 16.0) / 116.0)

        l_star = np.clip(116.0 * fy - 16.0, 0.0, 100.0)
        a_star = 500.0 * (fx - fy)
        b_star = 200.0 * (fy - fz)

        return np.stack([l_star, a_star, b_star], axis=-1)

    def extract_skin(
        self,
        rgb_image: np.ndarray,
        return_mask: bool = False,
    ) -> SkinExtractionResult:
        """
        Executes the peripheral skin extraction heuristic on an RGB image.

        Args:
            rgb_image: Array [H, W, 3] in RGB order.
            return_mask: If True, includes the boolean peripheral mask in the result.

        Returns:
            SkinExtractionResult containing colorimetry and validity flags.
        """
        if not isinstance(rgb_image, np.ndarray) or rgb_image.ndim != 3 or rgb_image.shape[-1] != 3:
            return SkinExtractionResult(
                is_valid=False,
                l_mean=float(np.nan),
                a_mean=float(np.nan),
                b_mean=float(np.nan),
                ita=float(np.nan),
                ita_bucket="Unknown / Indeterminate",
                quality_score=0.0,
                valid_pixel_count=0,
                rejection_reason="invalid_image_shape",
            )

        h, w, _ = rgb_image.shape
        total_pixels = h * w
        if total_pixels == 0:
            return SkinExtractionResult(
                is_valid=False,
                l_mean=float(np.nan),
                a_mean=float(np.nan),
                b_mean=float(np.nan),
                ita=float(np.nan),
                ita_bucket="Unknown / Indeterminate",
                quality_score=0.0,
                valid_pixel_count=0,
                rejection_reason="empty_image",
            )

        # Convert to float [0, 1] for color space operations
        img_norm = rgb_image.astype(np.float32)
        if img_norm.max() > 1.0:
            img_norm = img_norm / 255.0

        # Convert to CIE L*a*b*
        lab = self.rgb_to_cielab(img_norm)
        l_chan = lab[..., 0]
        a_chan = lab[..., 1]
        b_chan = lab[..., 2]

        # ----------------------------------------------------------------------
        # Heuristic 1: Geometric Spatial Mask (Peripheral Annulus)
        # ----------------------------------------------------------------------
        y_coords, x_coords = np.ogrid[:h, :w]
        center_y, center_x = h / 2.0, w / 2.0
        norm_dist = np.sqrt(((x_coords - center_x) / (w / 2.0)) ** 2 + ((y_coords - center_y) / (h / 2.0)) ** 2)

        # Exclude central lesion (normalized radius < central_exclusion_radius_ratio)
        peripheral_spatial = norm_dist > self.central_exclusion_radius_ratio

        # Exclude outer borders/vignettes
        b_margin_y = int(h * self.border_margin_ratio)
        b_margin_x = int(w * self.border_margin_ratio)
        border_mask = np.ones((h, w), dtype=bool)
        if b_margin_y > 0:
            border_mask[:b_margin_y, :] = False
            border_mask[-b_margin_y:, :] = False
        if b_margin_x > 0:
            border_mask[:, :b_margin_x] = False
            border_mask[:, -b_margin_x:] = False

        candidate_mask = peripheral_spatial & border_mask

        # ----------------------------------------------------------------------
        # Heuristic 2: Artifact & Non-Skin Filtering
        # ----------------------------------------------------------------------
        # A. Vignette / deep shadow / dark hair: L* < 15
        non_dark = l_chan >= 15.0

        # B. Specular reflection / white borders / ruler glare: L* > 95
        non_glare = l_chan <= 95.0

        # C. Biological chromaticity filter:
        # In human skin, oxygenated hemoglobin and melanin yield positive a* and b* (warm/yellowish-red).
        # Saturated blues, greens, or cyan (surgical markers, backgrounds) have b* <= 0 or a* <= -5.
        biological_chroma = (b_chan > 1.0) & (a_chan > 0.0)

        # Combine filters
        skin_mask = candidate_mask & non_dark & non_glare & biological_chroma
        valid_pixel_count = int(np.sum(skin_mask))

        min_required = max(self.min_valid_pixels, int(total_pixels * self.min_valid_fraction))
        if valid_pixel_count < min_required:
            logger.warning(
                "Insufficient valid peripheral skin pixels (%d < %d required). Marking indeterminate.",
                valid_pixel_count,
                min_required,
            )
            return SkinExtractionResult(
                is_valid=False,
                l_mean=float(np.nan),
                a_mean=float(np.nan),
                b_mean=float(np.nan),
                ita=float(np.nan),
                ita_bucket="Unknown / Indeterminate",
                quality_score=0.0,
                valid_pixel_count=valid_pixel_count,
                rejection_reason="insufficient_peripheral_skin_pixels",
                peripheral_mask=skin_mask if return_mask else None,
            )

        # ----------------------------------------------------------------------
        # Heuristic 3: Compute Regional Colorimetry and Quality
        # ----------------------------------------------------------------------
        l_pixels = l_chan[skin_mask]
        a_pixels = a_chan[skin_mask]
        b_pixels = b_chan[skin_mask]

        l_mean = float(np.mean(l_pixels))
        a_mean = float(np.mean(a_pixels))
        b_mean = float(np.mean(b_pixels))

        l_std = float(np.std(l_pixels))

        # Quality penalizes excessive lighting non-uniformity or shadow gradients
        spatial_coverage = min(1.0, valid_pixel_count / (total_pixels * 0.10))
        uniformity_score = np.exp(-l_std / 20.0)
        quality_score = float(np.clip(spatial_coverage * uniformity_score, 0.0, 1.0))

        # Compute ITA using mathematically corrected epsilon guard
        ita = BaseDermatologyDataset.calculate_ita(l_mean, b_mean)
        ita_bucket = BaseDermatologyDataset.map_ita_to_bucket(ita)

        return SkinExtractionResult(
            is_valid=True,
            l_mean=l_mean,
            a_mean=a_mean,
            b_mean=b_mean,
            ita=ita,
            ita_bucket=ita_bucket,
            quality_score=quality_score,
            valid_pixel_count=valid_pixel_count,
            rejection_reason=None,
            peripheral_mask=skin_mask if return_mask else None,
        )
