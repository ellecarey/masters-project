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
    INVALID_FEATURE_TYPES,
    TARGET_WEIGHTS,
    FUNCTION_TYPES,
    TARGET_NOISE_LEVELS,
    PERTURBATION_LEVELS,
    ERROR_PATTERNS,
    REPRODUCIBILITY_SEEDS,
    DATASET_CONFIGS,
)


class TestDataGeneratorValidators:
    def test_validate_feature_parameters_valid_samples(self):
        """Test validation passes with valid sample feature parameters"""
        DataGeneratorValidators.validate_feature_parameters(SAMPLE_FEATURE_PARAMS)

    def test_validate_feature_parameters_invalid_patterns(self):
        """Test validation fails with invalid feature parameters from fixtures"""
        for invalid_params in INVALID_FEATURE_PARAMS:
            with pytest.raises(ValueError):
                DataGeneratorValidators.validate_feature_parameters(invalid_params)

    def test_validate_feature_parameters_missing_mean(self):
        """Test validation fails when mean is missing"""
        invalid_params = {"test_feature_1": {"std": 1.0}}
        with pytest.raises(
            ValueError,
            match="Feature test_feature_1 must have 'mean' and 'std' parameters",
        ):
            DataGeneratorValidators.validate_feature_parameters(invalid_params)

    def test_validate_feature_parameters_missing_std(self):
        """Test validation fails when std is missing"""
        invalid_params = {"test_feature_1": {"mean": 0.0}}
        with pytest.raises(
            ValueError,
            match="Feature test_feature_1 must have 'mean' and 'std' parameters",
        ):
            DataGeneratorValidators.validate_feature_parameters(invalid_params)

    def test_validate_feature_parameters_negative_std(self):
        """Test validation fails with negative standard deviation"""
        invalid_params = {"test_feature_1": {"mean": 0.0, "std": -1.0}}
        with pytest.raises(
            ValueError, match="Standard deviation for test_feature_1 must be positive"
        ):
            DataGeneratorValidators.validate_feature_parameters(invalid_params)

    def test_validate_feature_parameters_zero_std(self):
        """Test validation fails with zero standard deviation"""
        invalid_params = {"test_feature_1": {"mean": 0.0, "std": 0.0}}
        with pytest.raises(
            ValueError, match="Standard deviation for test_feature_1 must be positive"
        ):
            DataGeneratorValidators.validate_feature_parameters(invalid_params)

    def test_validate_feature_parameters_not_dict(self):
        """Test validation fails when parameter is not a dictionary"""
        invalid_params = {"test_feature_1": "not_a_dict"}
        with pytest.raises(
            ValueError, match="Parameters for test_feature_1 must be a dictionary"
        ):
            DataGeneratorValidators.validate_feature_parameters(invalid_params)

    def test_validate_init_parameters_valid_configs(self):
        """Test initialisation parameter validation with valid dataset configs"""
        for config_name, config in DATASET_CONFIGS.items():
            DataGeneratorValidators.validate_init_parameters(
                config["samples"], config["features"]
            )

    def test_validate_init_parameters_invalid_samples(self):
        """Test initialisation fails with invalid n_samples"""
        with pytest.raises(ValueError, match="Number of samples must be positive"):
            DataGeneratorValidators.validate_init_parameters(0, 5)

        with pytest.raises(ValueError, match="Number of samples must be positive"):
            DataGeneratorValidators.validate_init_parameters(-1, 5)

    def test_validate_init_parameters_invalid_features(self):
        """Test initialisation fails with invalid n_features"""
        with pytest.raises(ValueError, match="Number of features must be positive"):
            DataGeneratorValidators.validate_init_parameters(100, 0)

        with pytest.raises(ValueError, match="Number of features must be positive"):
            DataGeneratorValidators.validate_init_parameters(100, -1)

    def test_validate_feature_types_valid_samples(self):
        """Test feature type validation with valid sample types"""
        DataGeneratorValidators.validate_feature_types(SAMPLE_FEATURE_TYPES)

    def test_validate_feature_types_non_dict(self):
        """Test feature type validation fails with non-dictionary input"""
        non_dict_inputs = ["not_a_dict", 123, None, []]
        for invalid_input in non_dict_inputs:
            with pytest.raises(ValueError, match="Feature types must be a dictionary"):
                DataGeneratorValidators.validate_feature_types(invalid_input)

    def test_validate_feature_types_invalid_values(self):
        """Test feature type validation fails with invalid feature type values"""
        invalid_type_dicts = [
            {"feature1": "invalid_type"},
            {"feature1": "categorical"},
            {"feature1": 123},
            {"feature1": None},
        ]
        for invalid_dict in invalid_type_dicts:
            with pytest.raises(ValueError, match="Invalid feature type"):
                DataGeneratorValidators.validate_feature_types(invalid_dict)

    def test_validate_perturbation_parameters_valid_types(
        self, generator_with_sample_data
    ):
        """Test perturbation validation with valid parameters"""
        valid_types = ["gaussian", "uniform"]
        valid_features = list(SAMPLE_FEATURE_PARAMS.keys())[:2]

        for perturbation_type in valid_types:
            for scale_name, scale_value in PERTURBATION_LEVELS.items():
                DataGeneratorValidators.validate_perturbation_parameters(
                    perturbation_type,
                    valid_features,
                    scale_value,
                    generator_with_sample_data.data,
                )

    def test_validate_perturbation_parameters_invalid_type(
        self, generator_with_sample_data
    ):
        """Test perturbation validation with invalid perturbation type"""
        valid_features = list(SAMPLE_FEATURE_PARAMS.keys())[:1]

        with pytest.raises(
            ValueError, match="perturbation_type must be 'gaussian' or 'uniform'"
        ):
            DataGeneratorValidators.validate_perturbation_parameters(
                "invalid_type", valid_features, 0.1, generator_with_sample_data.data
            )

    def test_validate_perturbation_parameters_negative_scale(
        self, generator_with_sample_data
    ):
        """Test perturbation validation with negative scale"""
        valid_features = list(SAMPLE_FEATURE_PARAMS.keys())[:1]

        with pytest.raises(ValueError, match="Scale must be non-negative"):
            DataGeneratorValidators.validate_perturbation_parameters(
                "gaussian", valid_features, -0.1, generator_with_sample_data.data
            )

    def test_validate_perturbation_parameters_zero_scale(
        self, generator_with_sample_data
    ):
        """Test perturbation validation allows zero scale (noise-free)"""
        valid_features = list(SAMPLE_FEATURE_PARAMS.keys())[:1]

        DataGeneratorValidators.validate_perturbation_parameters(
            "gaussian",
            valid_features,
            PERTURBATION_LEVELS["noise_free"],
            generator_with_sample_data.data,
        )

    def test_validate_perturbation_parameters_no_data(self):
        """Test perturbation validation fails with no data"""
        valid_features = list(SAMPLE_FEATURE_PARAMS.keys())[:1]

        with pytest.raises(
            ValueError, match="No data generated. Call generate_features\\(\\) first."
        ):
            DataGeneratorValidators.validate_perturbation_parameters(
                "gaussian", valid_features, 0.1, None
            )

    def test_validate_perturbation_parameters_nonexistent_features(
        self, generator_with_sample_data
    ):
        """Test perturbation validation fails with non-existent features"""
        nonexistent_features = ["nonexistent_feature", "another_fake_feature"]

        for feature in nonexistent_features:
            with pytest.raises(
                ValueError, match=f"Feature '{feature}' not found in data"
            ):
                DataGeneratorValidators.validate_perturbation_parameters(
                    "gaussian", [feature], 0.1, generator_with_sample_data.data
                )

    def test_validate_target_parameters_valid_configs(self, generator_with_sample_data):
        """Test target parameter validation with valid configurations"""
        features_to_use = list(SAMPLE_FEATURE_PARAMS.keys())[:2]

        # Handle TARGET_WEIGHTS as either list or dict
        if isinstance(TARGET_WEIGHTS, dict):
            weights = list(TARGET_WEIGHTS.values())[: len(features_to_use)]
        else:
            weights = TARGET_WEIGHTS[: len(features_to_use)]

        for function_type in FUNCTION_TYPES:
            DataGeneratorValidators.validate_target_parameters(
                features_to_use, weights, function_type, generator_with_sample_data.data
            )

    def test_validate_target_parameters_no_data(self):
        """Test target parameter validation fails with no data"""
        features_to_use = list(SAMPLE_FEATURE_PARAMS.keys())[:2]
        weights = TARGET_WEIGHTS[: len(features_to_use)]

        with pytest.raises(
            ValueError, match="No data generated. Call generate_features\\(\\) first."
        ):
            DataGeneratorValidators.validate_target_parameters(
                features_to_use, weights, "linear", None
            )

    def test_validate_target_parameters_missing_features(
        self, generator_with_sample_data
    ):
        """Test target parameter validation fails with missing features"""
        missing_features = ["nonexistent_feature_1", "nonexistent_feature_2"]
        weights = TARGET_WEIGHTS[: len(missing_features)]

        with pytest.raises(ValueError, match="Features not found in data"):
            DataGeneratorValidators.validate_target_parameters(
                missing_features, weights, "linear", generator_with_sample_data.data
            )

    def test_validate_target_parameters_weight_mismatch(
        self, generator_with_sample_data
    ):
        """Test target parameter validation fails with mismatched weights"""
        features_to_use = list(SAMPLE_FEATURE_PARAMS.keys())[:3]
        wrong_weights = TARGET_WEIGHTS[:2]  # Only 2 weights for 3 features

        with pytest.raises(
            ValueError, match="Number of weights must match number of features to use."
        ):
            DataGeneratorValidators.validate_target_parameters(
                features_to_use,
                wrong_weights,
                "linear",
                generator_with_sample_data.data,
            )

    def test_validate_target_parameters_invalid_function_type(
        self, generator_with_sample_data
    ):
        """Test target parameter validation fails with invalid function type"""
        features_to_use = list(SAMPLE_FEATURE_PARAMS.keys())[:2]
        weights = TARGET_WEIGHTS[: len(features_to_use)]

        with pytest.raises(
            ValueError,
            match="function_type must be 'linear', 'polynomial', or 'logistic'",
        ):
            DataGeneratorValidators.validate_target_parameters(
                features_to_use,
                weights,
                "invalid_function",
                generator_with_sample_data.data,
            )

    def test_validate_visualisation_parameters_valid_configs(
        self, generator_with_sample_data
    ):
        """Test visualisation parameter validation with valid configurations"""
        features = list(SAMPLE_FEATURE_PARAMS.keys())

        DataGeneratorValidators.validate_visualisation_parameters(
            features,
            max_features_to_show=3,
            n_bins=20,
            data=generator_with_sample_data.data,
        )

    def test_validate_visualisation_parameters_no_data(self):
        """Test visualisation parameter validation fails with no data"""
        features = list(SAMPLE_FEATURE_PARAMS.keys())

        with pytest.raises(ValueError, match="No data generated to visualise."):
            DataGeneratorValidators.validate_visualisation_parameters(
                features, max_features_to_show=3, n_bins=20, data=None
            )

    def test_validate_visualisation_parameters_invalid_features(
        self, generator_with_sample_data
    ):
        """Test visualisation parameter validation fails with invalid features"""
        invalid_features = ["nonexistent_feature"]

        with pytest.raises(ValueError, match="Features not found in data"):
            DataGeneratorValidators.validate_visualisation_parameters(
                invalid_features,
                max_features_to_show=3,
                n_bins=20,
                data=generator_with_sample_data.data,
            )

    def test_validate_visualisation_parameters_invalid_max_features(
        self, generator_with_sample_data
    ):
        """Test visualisation parameter validation fails with invalid max_features_to_show"""
        features = list(SAMPLE_FEATURE_PARAMS.keys())

        with pytest.raises(ValueError, match="max_features_to_show must be positive"):
            DataGeneratorValidators.validate_visualisation_parameters(
                features,
                max_features_to_show=0,
                n_bins=20,
                data=generator_with_sample_data.data,
            )

        with pytest.raises(ValueError, match="max_features_to_show must be positive"):
            DataGeneratorValidators.validate_visualisation_parameters(
                features,
                max_features_to_show=-1,
                n_bins=20,
                data=generator_with_sample_data.data,
            )

    def test_validate_visualisation_parameters_invalid_bins(
        self, generator_with_sample_data
    ):
        """Test visualisation parameter validation fails with invalid n_bins"""
        features = list(SAMPLE_FEATURE_PARAMS.keys())

        with pytest.raises(ValueError, match="n_bins must be positive"):
            DataGeneratorValidators.validate_visualisation_parameters(
                features,
                max_features_to_show=3,
                n_bins=0,
                data=generator_with_sample_data.data,
            )

        with pytest.raises(ValueError, match="n_bins must be positive"):
            DataGeneratorValidators.validate_visualisation_parameters(
                features,
                max_features_to_show=3,
                n_bins=-5,
                data=generator_with_sample_data.data,
            )

    def test_comprehensive_validation_workflow(self, generator_factory):
        """Test comprehensive validation workflow using actual sample data"""
        # Test complete workflow with valid data
        config = DATASET_CONFIGS["unit_test_standard"]

        # Validate initialisation parameters
        DataGeneratorValidators.validate_init_parameters(
            config["samples"], config["features"]
        )

        # Create generator and validate feature generation
        gen = generator_factory(
            n_samples=config["samples"],
            n_features=len(SAMPLE_FEATURE_PARAMS),
            random_state=REPRODUCIBILITY_SEEDS[0],
        )

        # Validate feature parameters and types
        DataGeneratorValidators.validate_feature_parameters(SAMPLE_FEATURE_PARAMS)
        DataGeneratorValidators.validate_feature_types(SAMPLE_FEATURE_TYPES)

        # Generate features
        gen.generate_features(SAMPLE_FEATURE_PARAMS, SAMPLE_FEATURE_TYPES)

        # Validate perturbation parameters
        features_to_perturb = list(SAMPLE_FEATURE_PARAMS.keys())[:2]
        DataGeneratorValidators.validate_perturbation_parameters(
            "gaussian",
            features_to_perturb,
            PERTURBATION_LEVELS["realistic_noise"],
            gen.data,
        )

        # Validate target parameters
        features_for_target = list(SAMPLE_FEATURE_PARAMS.keys())[:3]
        weights_for_target = TARGET_WEIGHTS[: len(features_for_target)]
        DataGeneratorValidators.validate_target_parameters(
            features_for_target, weights_for_target, "linear", gen.data
        )

        # Validate visualisation parameters
        DataGeneratorValidators.validate_visualisation_parameters(
            list(SAMPLE_FEATURE_PARAMS.keys()),
            max_features_to_show=3,
            n_bins=20,
            data=gen.data,
        )

    def test_edge_cases_with_actual_sample_data(self, generator_with_sample_data):
        """Test edge cases using your specific sample data structure"""
        # Test with all sample features
        all_features = list(SAMPLE_FEATURE_PARAMS.keys())

        # Test perturbation with all features
        DataGeneratorValidators.validate_perturbation_parameters(
            "uniform",
            all_features,
            PERTURBATION_LEVELS["extreme_noise"],
            generator_with_sample_data.data,
        )

        # Test target creation with all features and matching weights
        all_weights = TARGET_WEIGHTS[: len(all_features)]
        if len(all_weights) == len(all_features):
            DataGeneratorValidators.validate_target_parameters(
                all_features, all_weights, "polynomial", generator_with_sample_data.data
            )

        # Test visualisation with maximum features
        DataGeneratorValidators.validate_visualisation_parameters(
            all_features,
            max_features_to_show=len(all_features),
            n_bins=50,
            data=generator_with_sample_data.data,
        )

    def test_error_message_patterns_match_fixtures(self, generator_with_sample_data):
        """Test that error messages match patterns defined in ERROR_PATTERNS"""
        # Test specific error pattern for invalid perturbation type
        with pytest.raises(
            ValueError, match="perturbation_type must be 'gaussian' or 'uniform'"
        ):
            DataGeneratorValidators.validate_perturbation_parameters(
                "invalid_type", ["test_feature_1"], 0.1, generator_with_sample_data.data
            )

        # Test specific error pattern for missing features
        with pytest.raises(ValueError, match="Features not found in data"):
            DataGeneratorValidators.validate_target_parameters(
                ["nonexistent"], [1.0], "linear", generator_with_sample_data.data
            )

        # Test specific error pattern for weight mismatch
        with pytest.raises(
            ValueError, match="Number of weights must match number of features to use"
        ):
            DataGeneratorValidators.validate_target_parameters(
                list(SAMPLE_FEATURE_PARAMS.keys())[:2],
                [1.0],
                "linear",
                generator_with_sample_data.data,
            )
