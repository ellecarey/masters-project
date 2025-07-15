"""
Tests for validation functionality - Updated for current use case.
"""

import pytest
import pandas as pd
from data_generator_module.validators import DataGeneratorValidators


class TestDataGeneratorValidators:
    def test_validate_feature_parameters_valid(self):
        """Test valid feature parameter validation"""
        valid_params = {
            "feature_0": {"mean": 0.0, "std": 1.0},
            "feature_1": {"mean": 10.0, "std": 2.0},
        }

        # Should not raise exception
        DataGeneratorValidators.validate_feature_parameters(valid_params)

    def test_validate_feature_parameters_invalid(self):
        """Test invalid feature parameter validation"""
        invalid_cases = [
            ({"feature_0": {"mean": 0.0, "std": -1.0}}, "must be positive"),
            ({"feature_0": {"mean": 0.0}}, "must have 'mean' and 'std'"),
            ({"feature_0": "not_dict"}, "must be a dictionary"),
            ("not_dict", "must be a dictionary"),
        ]

        for invalid_params, expected_error in invalid_cases:
            with pytest.raises(ValueError, match=expected_error):
                DataGeneratorValidators.validate_feature_parameters(invalid_params)

    def test_validate_init_parameters(self):
        """Test initialization parameter validation"""
        # Valid cases
        DataGeneratorValidators.validate_init_parameters(100, 5)
        DataGeneratorValidators.validate_init_parameters(1, 1)

        # Invalid cases
        with pytest.raises(ValueError, match="Number of samples must be positive"):
            DataGeneratorValidators.validate_init_parameters(0, 5)

        with pytest.raises(ValueError, match="Number of features must be positive"):
            DataGeneratorValidators.validate_init_parameters(100, 0)

    def test_validate_feature_types(self):
        """Test feature type validation"""
        # Valid cases
        valid_types = {"feature_0": "continuous", "feature_1": "discrete"}
        DataGeneratorValidators.validate_feature_types(valid_types)

        # Invalid cases
        invalid_cases = [
            ({"feature_0": "invalid_type"}, "Invalid feature type"),
            ({"feature_0": "categorical"}, "Invalid feature type"),
            ({"feature_0": 123}, "Invalid feature type"),
            ("not_dict", "must be a dictionary"),
        ]

        for invalid_types, expected_error in invalid_cases:
            with pytest.raises(ValueError, match=expected_error):
                DataGeneratorValidators.validate_feature_types(invalid_types)

    def test_validate_visualisation_parameters(self):
        """Test visualization parameter validation"""
        # Create sample data
        data = pd.DataFrame({"feature_0": [1, 2, 3], "feature_1": [4, 5, 6]})

        # Valid case
        DataGeneratorValidators.validate_visualisation_parameters(
            features=["feature_0", "feature_1"],
            max_features_to_show=2,
            n_bins=20,
            data=data,
        )

        # Invalid cases
        with pytest.raises(ValueError, match="No data generated"):
            DataGeneratorValidators.validate_visualisation_parameters(
                features=["feature_0"], max_features_to_show=1, n_bins=20, data=None
            )

        with pytest.raises(ValueError, match="Features not found"):
            DataGeneratorValidators.validate_visualisation_parameters(
                features=["nonexistent"], max_features_to_show=1, n_bins=20, data=data
            )

    def test_validate_signal_noise_parameters(self):
        """Test signal/noise parameter validation"""
        data = pd.DataFrame({"feature_0": [1, 2, 3], "feature_1": [4, 5, 6]})

        # Valid case
        DataGeneratorValidators.validate_signal_noise_parameters(
            signal_features=["feature_0", "feature_1"],
            signal_weights=[0.5, -0.5],
            data=data,
        )

        # Invalid cases
        with pytest.raises(ValueError, match="No data generated"):
            DataGeneratorValidators.validate_signal_noise_parameters(
                signal_features=["feature_0"], signal_weights=[0.5], data=None
            )

        with pytest.raises(ValueError, match="Number of weights must match"):
            DataGeneratorValidators.validate_signal_noise_parameters(
                signal_features=["feature_0", "feature_1"],
                signal_weights=[0.5],
                data=data,
            )
