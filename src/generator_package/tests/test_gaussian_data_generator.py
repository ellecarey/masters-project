"""
Comprehensive tests for GaussianDataGenerator class.
Contains all the core functionality tests including initialisation,
feature generation, perturbations, target creation, and visualisation.
"""

import pytest
from generator_package.gaussian_data_generator import GaussianDataGenerator
from .fixtures.sample_data import DATASET_CONFIGS, ERROR_PATTERNS, REPRODUCIBILITY_SEEDS


class TestGaussianDataGeneratorInit:
    def test_valid_initialisation(self):
        """test valid initialisation parameters"""
        config = DATASET_CONFIGS["unit_test_standard"]
        generator = GaussianDataGenerator(
            n_samples=config["samples"],
            n_features=config["features"],
            random_state=REPRODUCIBILITY_SEEDS[0],
        )
        assert generator.n_samples == config["samples"]
        assert generator.n_features == config["features"]
        assert generator.random_state == 42
        assert generator.data is None
        assert generator.feature_types == {}
        assert generator.feature_parameters == {}

    def test_default_random_state_only(self):
        """test that only random_state has a default value of 42"""
        config = DATASET_CONFIGS["unit_test_standard"]
        generator = GaussianDataGenerator(
            n_samples=config["samples"],
            n_features=config["features"],
            random_state=REPRODUCIBILITY_SEEDS[0],
        )
        assert generator.random_state == 42

    def test_invalid_n_samples(self):
        """test validation catches invalid n_samples"""
        with pytest.raises(ValueError, match=ERROR_PATTERNS["negative_samples"]):
            GaussianDataGenerator(
                n_samples=0,
                n_features=DATASET_CONFIGS["minimal_valid_input"]["features"],
                random_state=REPRODUCIBILITY_SEEDS[0],
            )

    def test_invalid_n_features(self):
        """test validation catches invalid n_features"""
        with pytest.raises(ValueError, match=ERROR_PATTERNS["negative_features"]):
            GaussianDataGenerator(
                n_samples=DATASET_CONFIGS["minimal_valid_input"]["samples"],
                n_features=0,
                random_state=REPRODUCIBILITY_SEEDS[0],
            )
