import unittest
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import patch
from generator_package import GaussianDataGenerator, config


class TestGaussianDataGenerator(unittest.TestCase):
    """Comprehensive tests for the GaussianDataGenerator"""

    @classmethod
    def setUpClass(cls):
        """set up class level test fixtures"""
        cls.default_params = {
            "feature_0": {"mean": 0, "std": 1},
            "feature_1": {"mean": 5, "std": 2},
            "feature_2": {"mean": -2, "std": 0.5},
        }

        cls.de
