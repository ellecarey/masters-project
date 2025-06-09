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
    def test_generate_features_with_sample_params(self):
        """Test feature generation using predefined sample parameters with statistical confidence"""
        config = DATASET_CONFIGS["unit_test_standard"]
        generator = GaussianDataGenerator(
            n_samples=config["samples"],
            n_features=len(SAMPLE_FEATURE_PARAMS),
            random_state=REPRODUCIBILITY_SEEDS[0],
        )

        generator.generate_features(SAMPLE_FEATURE_PARAMS, SAMPLE_FEATURE_TYPES)

        # Verify data structure matches fixture configurations
        assert generator.data is not None
        assert generator.data.shape == (config["samples"], len(SAMPLE_FEATURE_PARAMS))
        assert list(generator.data.columns) == list(SAMPLE_FEATURE_PARAMS.keys())

        # Verify feature parameters are stored exactly as provided
        assert generator.feature_parameters == SAMPLE_FEATURE_PARAMS
        assert generator.feature_types == SAMPLE_FEATURE_TYPES

        # Check feature distributions with statistical confidence intervals
        for feature_name, params in SAMPLE_FEATURE_PARAMS.items():
            feature_data = generator.data[feature_name]
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
