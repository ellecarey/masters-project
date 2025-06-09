"""
Tests for feature generation functionality with specific parameter configurations.
"""

import pytest
import numpy as np
import pandas as pd

from generator_package.gaussian_data_generator import GaussianDataGenerator
from .fixtures.sample_data import (
    SAMPLE_FEATURE_PARAMS,
    SAMPLE_FEATURE_TYPES,
    REPRODUCIBILITY_SEEDS,
    DATASET_CONFIGS,
    ERROR_PATTERNS,
)


class TestFeatureGeneration:
    def validate_basic_structure(
        self, generator, expected_samples, expected_features, expected_columns
    ):
        """Helper method to validate basic data structure"""
        assert generator.data is not None
        assert generator.data.shape == (expected_samples, expected_features)
        assert list(generator.data.columns) == expected_columns

    def validate_parameter_storage(self, generator):
        """Helper method to validate parameter storage"""
        assert generator.feature_parameters == SAMPLE_FEATURE_PARAMS
        assert generator.feature_types == SAMPLE_FEATURE_TYPES

    def test_generate_features_with_sample_params(
        self, generator_with_sample_data, standard_test_config
    ):
        """Test feature generation using predefined sample parameters with statistical confidence"""

        # Validate basic structure
        self.validate_basic_structure(
            generator_with_sample_data,
            100,  # From generator_with_sample_data fixture
            len(SAMPLE_FEATURE_PARAMS),
            list(SAMPLE_FEATURE_PARAMS.keys()),
        )

        # Validate parameter storage
        self.validate_parameter_storage(generator_with_sample_data)

        # Check feature distributions with statistical confidence intervals
        for feature_name, params in SAMPLE_FEATURE_PARAMS.items():
            feature_data = generator_with_sample_data.data[feature_name]
            n_samples = len(feature_data)

            # Calculate standard errors for mean and std
            se_mean = params["std"] / np.sqrt(n_samples)
            se_std = params["std"] / np.sqrt(2 * n_samples)

            # Use 3 standard errors for 99.7% confidence
            assert abs(feature_data.mean() - params["mean"]) < 3 * se_mean, (
                f"Mean for {feature_name} outside expected range: {feature_data.mean():.3f} vs {params['mean']:.3f}"
            )
            assert abs(feature_data.std() - params["std"]) < 3 * se_std, (
                f"Std for {feature_name} outside expected range: {feature_data.std():.3f} vs {params['std']:.3f}"
            )

    def test_continuous_vs_discrete_features_from_fixtures(
        self, generator_with_sample_data
    ):
        """Test continuous vs discrete handling using SAMPLE_FEATURE_TYPES"""

        # Test continuous features from actual fixture data
        continuous_features = [
            name
            for name, ftype in SAMPLE_FEATURE_TYPES.items()
            if ftype == "continuous"
        ]

        # Ensure testing actual continuous features
        assert len(continuous_features) > 0, (
            "No continuous features found in SAMPLE_FEATURE_TYPES"
        )

        for feature in continuous_features:
            feature_data = generator_with_sample_data.data[feature]
            # Continuous features should have decimal places
            has_decimals = any(not float(val).is_integer() for val in feature_data)
            assert has_decimals, (
                f"Continuous feature {feature} should contain decimal values"
            )

        # Test discrete features from actual fixture data
        discrete_features = [
            name for name, ftype in SAMPLE_FEATURE_TYPES.items() if ftype == "discrete"
        ]

        for feature in discrete_features:
            feature_data = generator_with_sample_data.data[feature]
            # Discrete features should be integers after generation
            all_integers = all(float(val).is_integer() for val in feature_data)
            assert all_integers, (
                f"Discrete feature {feature} contains non-integer values"
            )

    def test_reproducibility_with_specific_seeds(self, standard_test_config):
        """Test reproducibility using exact REPRODUCIBILITY_SEEDS"""
        seed = REPRODUCIBILITY_SEEDS[1]

        # Create generators as needed
        gen1 = GaussianDataGenerator(
            n_samples=standard_test_config["samples"],
            n_features=len(SAMPLE_FEATURE_PARAMS),
            random_state=seed,
        )
        gen2 = GaussianDataGenerator(
            n_samples=standard_test_config["samples"],
            n_features=len(SAMPLE_FEATURE_PARAMS),
            random_state=seed,
        )

        # Generate data
        gen1.generate_features(SAMPLE_FEATURE_PARAMS, SAMPLE_FEATURE_TYPES)
        gen2.generate_features(SAMPLE_FEATURE_PARAMS, SAMPLE_FEATURE_TYPES)

        # check data is identical
        pd.testing.assert_frame_equal(gen1.data, gen2.data)

    def test_different_seeds_from_fixtures_produce_different_data(
        self, different_seed_generators
    ):
        """Test different seeds from REPRODUCIBILITY_SEEDS produce different datasets"""
        gen1, gen2 = different_seed_generators

        # Generate data with different seeds
        gen1.generate_features(SAMPLE_FEATURE_PARAMS, SAMPLE_FEATURE_TYPES)
        gen2.generate_features(SAMPLE_FEATURE_PARAMS, SAMPLE_FEATURE_TYPES)

        # Data should be different with different seeds
        assert not gen1.data.equals(gen2.data), (
            "Different seeds should produce different data"
        )

        # check tructure and parameters are the same
        assert gen1.data.shape == gen2.data.shape
        assert list(gen1.data.columns) == list(gen2.data.columns)
        assert gen1.feature_parameters == gen2.feature_parameters
        assert gen1.feature_types == gen2.feature_types

        # Verify some values are different
        differences = (gen1.data != gen2.data).sum().sum()
        assert differences > 0, (
            "Expected some differences between datasets with different seeds"
        )

    def test_basic_generator_without_data(self, basic_generator):
        """Test basic generator instance before data generation"""
        # check initialised with no data
        assert basic_generator.data is None
        assert basic_generator.n_samples == 100
        assert basic_generator.n_features == 3
        assert basic_generator.random_state == REPRODUCIBILITY_SEEDS[0]
        assert basic_generator.feature_parameters == {}
        assert basic_generator.feature_types == {}

    def test_feature_generation_validates_parameter_consistency(
        self, generator_factory
    ):
        """Test that generate_features handles parameter tupe mismatch between feature params and type dictionaries"""
        generator = generator_factory(
            n_samples=100, n_features=len(SAMPLE_FEATURE_PARAMS)
        )

        # create an inconsistency "extra_feature" that doesn't exist in SAMPLE_FEATURE_PARAM
        mismatched_types = SAMPLE_FEATURE_TYPES.copy()
        mismatched_types["extra_feature"] = "continuous"

        # graceful handling: method ignores extra "extra_feature" type, only generates features that exist in both dictionaries
        try:
            generator.generate_features(SAMPLE_FEATURE_PARAMS, mismatched_types)
            # Validate that generated columns match the parameter keys
            assert all(
                col in SAMPLE_FEATURE_PARAMS.keys() for col in generator.data.columns
            )
        except (ValueError, KeyError):
            # raising an error instead is acceptable
            pass

    def test_generator_factory_flexibility(self, generator_factory):
        """Test  generator factory fixture with different configurations"""
        # default parameters
        gen_default = generator_factory()
        assert gen_default.n_samples == 100
        assert gen_default.n_features == 5
        assert gen_default.random_state == REPRODUCIBILITY_SEEDS[0]

        # custom parameters
        gen_custom = generator_factory(
            n_samples=200, n_features=10, random_state=REPRODUCIBILITY_SEEDS[1]
        )
        assert gen_custom.n_samples == 200
        assert gen_custom.n_features == 10
        assert gen_custom.random_state == REPRODUCIBILITY_SEEDS[1]

    def test_sample_data_consistency(self, sample_feature_params, sample_feature_types):
        """Test that sample data fixtures return expected structures"""
        # Test feature parameters structure
        assert isinstance(sample_feature_params, dict)
        assert len(sample_feature_params) > 0

        for feature_name, params in sample_feature_params.items():
            assert "mean" in params
            assert "std" in params
            assert isinstance(params["mean"], (int, float))
            assert isinstance(params["std"], (int, float))
            assert params["std"] > 0

        # Test feature types structure
        assert isinstance(sample_feature_types, dict)
        assert len(sample_feature_types) > 0

        for feature_name, ftype in sample_feature_types.items():
            assert ftype in ["continuous", "discrete"]

        # Test consistency between parameters and types
        assert set(sample_feature_params.keys()) == set(sample_feature_types.keys())

    def test_seeds_consistency(self, test_seeds):
        """Test that REPRODUCIBILITY_SEEDS fixture is properly structured"""
        assert isinstance(test_seeds, list)
        assert len(test_seeds) >= 2  # at least 2 seeds for different tests
        assert all(isinstance(seed, int) for seed in test_seeds)
        assert len(set(test_seeds)) == len(test_seeds)  # seeds should be unique

    @pytest.mark.slow
    def test_large_dataset_performance(self, large_dataset):
        """Test performance with larger dataset using fixture"""
        # Should have been generated successfully by the fixture
        assert large_dataset.data is not None
        assert large_dataset.data.shape[0] >= 1000  # At least 1000 samples
        assert large_dataset.data.shape[1] >= 5  # At least 5 features

        # Basic statistical checks
        for col in large_dataset.data.columns:
            feature_data = large_dataset.data[col]
            assert abs(feature_data.mean()) < 0.5  # roughly centered
            assert 0.5 < feature_data.std() < 2.0  # Reasonable variance
