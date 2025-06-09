"""
Tests for feature generation functionality with specific parameter configurations.
"""

import pytest
import numpy as np
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
