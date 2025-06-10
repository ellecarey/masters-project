"""
Tests for data validation functionality.
"""

import pytest
import pandas as pd
import numpy as np
from generator_package.validators import DataGeneratorValidators
from generator_package.gaussian_data_generator import GaussianDataGenerator
from .fixtures.sample_data import (
    SAMPLE_FEATURE_PARAMS,
    SAMPLE_FEATURE_TYPES,
    INVALID_FEATURE_PARAMS,
    TARGET_WEIGHTS,
    FUNCTION_TYPES,
    PERTURBATION_LEVELS,
    REPRODUCIBILITY_SEEDS,
    DATASET_CONFIGS,
)


class TestDataGeneratorValidators:
    def test_validate_feature_parameters_valid(self):
        """Test validation passes with valid sample feature parameters"""
        DataGeneratorValidators.validate_feature_parameters(SAMPLE_FEATURE_PARAMS)

    @pytest.mark.parametrize(
        "invalid_params,expected_error",
        [
            ("not_a_dict", "Feature parameters must be a dictionary"),
            (
                {"feature_1": "not_a_dict"},
                "Parameters for feature_1 must be a dictionary",
            ),
            (
                {"feature_1": {"std": 1.0}},
                "Feature feature_1 must have 'mean' and 'std' parameters",
            ),
            (
                {"feature_1": {"mean": 0.0}},
                "Feature feature_1 must have 'mean' and 'std' parameters",
            ),
            (
                {"feature_1": {"mean": 0.0, "std": -1.0}},
                "Standard deviation for feature_1 must be positive",
            ),
            (
                {"feature_1": {"mean": 0.0, "std": 0.0}},
                "Standard deviation for feature_1 must be positive",
            ),
        ],
    )
    def test_validate_feature_parameters_invalid(self, invalid_params, expected_error):
        """Test feature parameter validation with various invalid inputs"""
        with pytest.raises(ValueError, match=expected_error):
            DataGeneratorValidators.validate_feature_parameters(invalid_params)

    @pytest.mark.parametrize(
        "n_samples,n_features,should_pass",
        [
            (100, 5, True),
            (1000, 10, True),
            (0, 5, False),
            (-1, 5, False),
            (100, 0, False),
            (100, -1, False),
        ],
    )
    def test_validate_init_parameters(self, n_samples, n_features, should_pass):
        """Test initialisation parameter validation"""
        if should_pass:
            DataGeneratorValidators.validate_init_parameters(n_samples, n_features)
        else:
            with pytest.raises(ValueError):
                DataGeneratorValidators.validate_init_parameters(n_samples, n_features)

    def test_validate_feature_types_valid(self):
        """Test feature type validation with valid sample types"""
        DataGeneratorValidators.validate_feature_types(SAMPLE_FEATURE_TYPES)

    @pytest.mark.parametrize(
        "invalid_types",
        [
            "not_a_dict",
            {"feature1": "invalid_type"},
            {"feature1": "categorical"},
            {"feature1": 123},
            {"feature1": None},
        ],
    )
    def test_validate_feature_types_invalid(self, invalid_types):
        """Test feature type validation with invalid inputs"""
        with pytest.raises(ValueError):
            DataGeneratorValidators.validate_feature_types(invalid_types)

    def test_validate_perturbation_parameters_valid(self, generator_with_sample_data):
        """Test perturbation validation with valid parameters"""
        valid_features = list(SAMPLE_FEATURE_PARAMS.keys())[:2]

        # Test valid cases
        for perturbation_type in ["gaussian", "uniform"]:
            for scale in [0.0, 0.1, 0.5]:
                DataGeneratorValidators.validate_perturbation_parameters(
                    perturbation_type,
                    valid_features,
                    scale,
                    generator_with_sample_data.data,
                )

    @pytest.mark.parametrize(
        "perturbation_type,features,scale,data,expected_error",
        [
            (
                "invalid_type",
                ["test_feature_1"],
                0.1,
                "dummy_data",
                "perturbation_type must be 'gaussian' or 'uniform'",
            ),
            (
                "gaussian",
                ["test_feature_1"],
                -0.1,
                "dummy_data",
                "Scale must be non-negative",
            ),
            (
                "gaussian",
                ["nonexistent"],
                0.1,
                "dummy_data",
                "Feature 'nonexistent' not found in data",
            ),
            ("gaussian", ["test_feature_1"], 0.1, None, "No data generated"),
        ],
    )
    def test_validate_perturbation_parameters_invalid(
        self,
        perturbation_type,
        features,
        scale,
        data,
        expected_error,
        generator_with_sample_data,
    ):
        """Test perturbation parameter validation with invalid inputs"""
        # Use real data for valid cases, None for the no-data test
        test_data = generator_with_sample_data.data if data != None else None

        with pytest.raises(ValueError, match=expected_error):
            DataGeneratorValidators.validate_perturbation_parameters(
                perturbation_type, features, scale, test_data
            )

    def test_validate_target_parameters_valid(self, generator_with_sample_data):
        """Test target parameter validation with valid configurations"""
        features_to_use = list(SAMPLE_FEATURE_PARAMS.keys())[:2]
        weights = TARGET_WEIGHTS[: len(features_to_use)]

        for function_type in FUNCTION_TYPES:
            DataGeneratorValidators.validate_target_parameters(
                features_to_use, weights, function_type, generator_with_sample_data.data
            )

    @pytest.mark.parametrize(
        "features,weights,function_type,expected_error",
        [
            (["nonexistent_feature"], [1.0], "linear", "Features not found in data"),
            (
                ["test_feature_1", "test_feature_2"],
                [1.0],
                "linear",
                "Number of weights must match number of features",
            ),
            (
                ["test_feature_1"],
                [1.0],
                "invalid_function",
                "function_type must be 'linear', 'polynomial', or 'logistic'",
            ),
        ],
    )
    def test_validate_target_parameters_invalid(
        self,
        features,
        weights,
        function_type,
        expected_error,
        generator_with_sample_data,
    ):
        """Test target parameter validation with invalid inputs"""
        with pytest.raises(ValueError, match=expected_error):
            DataGeneratorValidators.validate_target_parameters(
                features, weights, function_type, generator_with_sample_data.data
            )

    def test_validate_target_parameters_no_data(self):
        """Test target parameter validation fails with no data"""
        with pytest.raises(ValueError, match="No data generated"):
            DataGeneratorValidators.validate_target_parameters(
                ["test_feature_1"], [1.0], "linear", None
            )

    def test_validate_visualisation_parameters_valid(self, generator_with_sample_data):
        """Test visualisation parameter validation with valid inputs"""
        features = list(SAMPLE_FEATURE_PARAMS.keys())
        DataGeneratorValidators.validate_visualisation_parameters(
            features,
            max_features_to_show=3,
            n_bins=20,
            data=generator_with_sample_data.data,
        )

    @pytest.mark.parametrize(
        "features,max_features,n_bins,data,expected_error",
        [
            (["test_feature_1"], 3, 20, None, "No data generated to visualise"),
            (["nonexistent"], 3, 20, "dummy_data", "Features not found in data"),
            (
                ["test_feature_1"],
                0,
                20,
                "dummy_data",
                "max_features_to_show must be positive",
            ),
            (["test_feature_1"], 3, 0, "dummy_data", "n_bins must be positive"),
            (
                ["test_feature_1"],
                -1,
                20,
                "dummy_data",
                "max_features_to_show must be positive",
            ),
            (["test_feature_1"], 3, -5, "dummy_data", "n_bins must be positive"),
        ],
    )
    def test_validate_visualisation_parameters_invalid(
        self,
        features,
        max_features,
        n_bins,
        data,
        expected_error,
        generator_with_sample_data,
    ):
        """Test visualisation parameter validation with invalid inputs"""
        test_data = generator_with_sample_data.data if data != None else None

        with pytest.raises(ValueError, match=expected_error):
            DataGeneratorValidators.validate_visualisation_parameters(
                features,
                max_features_to_show=max_features,
                n_bins=n_bins,
                data=test_data,
            )

    def test_integration_with_actual_fixtures(self, generator_factory):
        """Test validators work with actual fixture data throughout pipeline"""
        # Test complete pipeline validation
        config = DATASET_CONFIGS["unit_test_standard"]

        # Validate init
        DataGeneratorValidators.validate_init_parameters(
            config["samples"], config["features"]
        )

        # Create and validate features
        gen = generator_factory(
            n_samples=config["samples"], n_features=len(SAMPLE_FEATURE_PARAMS)
        )
        DataGeneratorValidators.validate_feature_parameters(SAMPLE_FEATURE_PARAMS)
        DataGeneratorValidators.validate_feature_types(SAMPLE_FEATURE_TYPES)

        # Generate data and validate operations
        gen.generate_features(SAMPLE_FEATURE_PARAMS, SAMPLE_FEATURE_TYPES)

        # Validate perturbation
        DataGeneratorValidators.validate_perturbation_parameters(
            "gaussian", list(SAMPLE_FEATURE_PARAMS.keys())[:1], 0.1, gen.data
        )

        # Validate target creation
        DataGeneratorValidators.validate_target_parameters(
            list(SAMPLE_FEATURE_PARAMS.keys())[:2],
            TARGET_WEIGHTS[:2],
            "linear",
            gen.data,
        )

        # Validate visualisation
        DataGeneratorValidators.validate_visualisation_parameters(
            list(SAMPLE_FEATURE_PARAMS.keys()),
            max_features_to_show=2,
            n_bins=10,
            data=gen.data,
        )
