"""
Unit tests for utility functions in src/utils/seed.py
"""

import random
import numpy as np
import pytest
from src.utils.seed import get_device, seed_everything


class TestReproducibilityUtilities:
    """Validates deterministic seed setting and hardware device detection."""

    def test_seed_everything_determinism(self):
        seed_everything(1234)
        np_seq_1 = np.random.uniform(0, 1, size=5)
        py_seq_1 = [random.random() for _ in range(5)]

        # Re-seed with identical value
        seed_everything(1234)
        np_seq_2 = np.random.uniform(0, 1, size=5)
        py_seq_2 = [random.random() for _ in range(5)]

        assert np.allclose(np_seq_1, np_seq_2)
        assert py_seq_1 == py_seq_2

    def test_get_device_explicit_override(self):
        assert get_device("cpu") == "cpu"
        assert get_device("cuda") == "cuda"
        assert get_device("mps") == "mps"

    def test_get_device_fallback(self):
        dev = get_device()
        assert dev in ["cpu", "cuda", "mps"]
