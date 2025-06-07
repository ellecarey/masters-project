"""
Comprehensive tests for GaussianDataGenerator class.
Contains all the core functionality tests including initialisation,
feature generation, perturbations, target creation, and visualisation.
"""

import pytest
import numpy as pd
import pandas as pd
from src.generator_package.gaussian_data_generator import GaussianDataGenerator


class TestGaussianDataGeneratorInit:
    def test_valid_initialisation(self):
        """test valid initialisation parameters"""
        generator = GaussianDataGenerator(n_samples=100, n_features=5, random_state=42)
        assert generator.n_samples == 100
        assert generator.n_features == 5
        assert generator.random_state == 42
        assert generator.data is None
        assert generator.feature_types == {}
        assert generator.feature_parameters == {}

    def test_default_random_state_only(self):
        """test that only random_state has a default value of 42"""
        generator = GaussianDataGenerator(n_samples=100, n_features=5)
        assert generator.random_state == 42

    def test_invalid_n_samples(self):
        """test validation catches invalid n_samples"""
        with pytest.raises(ValueError, match="Number of samples must be positive"):
            GaussianDataGenerator(n_samples=0, n_features=5)

    def test_invalid_n_features(self):
        """test validation catches invalid n_features"""
        with pytest.raises(ValueError, match="Number of features must be positive"):
            GaussianDataGenerator(n_samples=0, n_features=5)


# class TestFeatureGeneration:

# class TestPerturbations:

# class TestTargetVariable:

# class TestVisualisation:

# class TestEdgeCases
