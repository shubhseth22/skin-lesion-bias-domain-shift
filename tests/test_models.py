"""
Unit tests for model architecture contracts and feature dimensions
Exercises actual source code in src/models/contracts.py
"""

import pytest
from src.models.contracts import DermatologyModelContract


class TestModelArchitectureContracts:
    """Validates structural interfaces and dimension requirements for models."""

    @pytest.mark.parametrize(
        "backbone, expected_dim",
        [
            ("resnet50", 2048),
            ("efficientnet_b4", 1792),
            ("vit_base_patch16_224", 768),
        ],
    )
    def test_supported_backbones_feature_dimensions(self, backbone: str, expected_dim: int):
        contract = DermatologyModelContract(backbone=backbone, num_classes=2)
        assert contract.feature_dim == expected_dim
        assert contract.backbone == backbone

    def test_unsupported_backbone_raises_error(self):
        with pytest.raises(ValueError, match="Unsupported backbone"):
            DermatologyModelContract(backbone="unsupported_vgg16")

    def test_invalid_num_classes_raises_error(self):
        with pytest.raises(ValueError, match="num_classes must be >= 2"):
            DermatologyModelContract(backbone="resnet50", num_classes=1)

    def test_feature_extraction_contract(self):
        contract = DermatologyModelContract(backbone="resnet50")
        feat_shape = contract.extract_features((8, 3, 224, 224))
        assert feat_shape == (8, 2048)

    @pytest.mark.parametrize("num_classes", [2, 8, 114])
    def test_classification_head_dimensions(self, num_classes: int):
        contract = DermatologyModelContract(backbone="resnet50", num_classes=num_classes)
        out_shape = contract.forward((4, 3, 224, 224))
        assert out_shape == (4, num_classes)

    def test_invalid_input_dimensions_raises_error(self):
        contract = DermatologyModelContract()
        # Channels != 3
        with pytest.raises(ValueError, match="Expected 3 input channels"):
            contract.forward((4, 1, 224, 224))

        # Spatial dimensions != 224x224
        with pytest.raises(ValueError, match="Expected spatial resolution"):
            contract.forward((4, 3, 256, 256))

        # Not 4D tensor
        with pytest.raises(ValueError, match="Expected 4D input tensor"):
            contract.forward((3, 224, 224))

    def test_mode_toggling(self):
        contract = DermatologyModelContract()
        assert contract.eval_mode is False
        contract.eval()
        assert contract.eval_mode is True
        contract.train()
        assert contract.eval_mode is False
