"""
Tests for target variable creation functionality - Condensed Version
"""

import pytest
import numpy as np
import pandas as pd
from generator_package.gaussian_data_generator import GaussianDataGenerator


class TestTargetCreation:
    def test_core_function_types_and_linear_validation(
        self, generator_factory, sample_feature_params, sample_feature_types
    ):
        """Test all function types with exact linear validation - consolidated test"""
        from .fixtures.sample_data import (
            TARGET_WEIGHTS,
            TARGET_NOISE_LEVELS,
            FUNCTION_TYPES,
        )

        gen = generator_factory(n_samples=100, n_features=len(sample_feature_params))
        gen.generate_features(sample_feature_params, sample_feature_types)

        features_to_use = list(sample_feature_params.keys())[:2]
        weights = TARGET_WEIGHTS[: len(features_to_use)]

        # Test linear with exact validation
        gen.create_target_variable(
            features_to_use=features_to_use,
            weights=weights,
            function_type="linear",
            noise_level=TARGET_NOISE_LEVELS["no_noise"],
        )

        # Validate exact linear combination
        expected_target = (
            gen.data[features_to_use[0]] * weights[0]
            + gen.data[features_to_use[1]] * weights[1]
        )
        np.testing.assert_array_almost_equal(
            gen.data["target"].values, expected_target.values, decimal=10
        )
        linear_target = gen.data["target"].copy()

        # Test polynomial
        gen.create_target_variable(
            features_to_use=features_to_use,
            weights=weights,
            function_type="polynomial",
            noise_level=TARGET_NOISE_LEVELS["no_noise"],
        )
        polynomial_target = gen.data["target"].copy()

        # Test logistic
        gen.create_target_variable(
            features_to_use=features_to_use,
            weights=weights,
            function_type="logistic",
            noise_level=TARGET_NOISE_LEVELS["no_noise"],
        )

        # Logistic should be bounded [0,1] and different from others
        assert all(0 <= val <= 1 for val in gen.data["target"])
        assert gen.data["target"].std() > 0.01
        assert not np.array_equal(polynomial_target.values, gen.data["target"].values)
        assert not np.array_equal(linear_target.values, gen.data["target"].values)

    def test_noise_levels_and_reproducibility(
        self, generator_factory, sample_feature_params, sample_feature_types, test_seeds
    ):
        """Test noise impact and reproducibility - consolidated test"""
        from .fixtures.sample_data import TARGET_WEIGHTS, TARGET_NOISE_LEVELS

        features_to_use = list(sample_feature_params.keys())[:2]
        weights = TARGET_WEIGHTS[: len(features_to_use)]
        seed = test_seeds[0]

        # Test different noise levels
        targets = {}
        for noise_name in ["no_noise", "minimal_noise", "high_noise"]:
            gen = generator_factory(
                n_samples=100, n_features=len(sample_feature_params), random_state=seed
            )
            gen.generate_features(sample_feature_params, sample_feature_types)
            gen.create_target_variable(
                features_to_use=features_to_use,
                weights=weights,
                function_type="linear",
                noise_level=TARGET_NOISE_LEVELS[noise_name],
            )
            targets[noise_name] = gen.data["target"].copy()

        # Higher noise should increase variance
        assert targets["no_noise"].std() < targets["high_noise"].std()
        assert not np.array_equal(
            targets["no_noise"].values, targets["minimal_noise"].values
        )

        # Test reproducibility
        gen1 = generator_factory(
            n_samples=100, n_features=len(sample_feature_params), random_state=seed
        )
        gen2 = generator_factory(
            n_samples=100, n_features=len(sample_feature_params), random_state=seed
        )

        gen1.generate_features(sample_feature_params, sample_feature_types)
        gen2.generate_features(sample_feature_params, sample_feature_types)

        target_params = {
            "features_to_use": features_to_use,
            "weights": weights,
            "function_type": "linear",
            "noise_level": TARGET_NOISE_LEVELS["minimal_noise"],
        }

        gen1.create_target_variable(**target_params)
        gen2.create_target_variable(**target_params)

        np.testing.assert_array_equal(
            gen1.data["target"].values, gen2.data["target"].values
        )

    def test_comprehensive_validation_and_structure(
        self, generator_with_sample_data, sample_feature_params, sample_feature_types
    ):
        """Test validation errors, structure preservation, and mixed features - consolidated test"""
        from .fixtures.sample_data import TARGET_WEIGHTS, TARGET_NOISE_LEVELS

        valid_features = list(sample_feature_params.keys())[:2]
        valid_weights = TARGET_WEIGHTS[: len(valid_features)]

        # Test validation errors
        with pytest.raises(
            ValueError,
            match="function_type must be 'linear', 'polynomial', or 'logistic'",
        ):
            generator_with_sample_data.create_target_variable(
                features_to_use=valid_features,
                weights=valid_weights,
                function_type="invalid_function",
                noise_level=TARGET_NOISE_LEVELS["no_noise"],
            )

        with pytest.raises(
            ValueError, match="Number of weights must match number of features"
        ):
            generator_with_sample_data.create_target_variable(
                features_to_use=valid_features,
                weights=[1.0],  # Wrong length
                function_type="linear",
                noise_level=TARGET_NOISE_LEVELS["no_noise"],
            )

        with pytest.raises(ValueError, match="Features not found in data"):
            generator_with_sample_data.create_target_variable(
                features_to_use=["nonexistent_feature"],
                weights=[1.0],
                function_type="linear",
                noise_level=TARGET_NOISE_LEVELS["no_noise"],
            )

        # Test structure preservation and mixed features
        original_shape = generator_with_sample_data.data.shape
        original_columns = generator_with_sample_data.data.columns.tolist()

        # Use mix of continuous and discrete features
        continuous_features = [
            name
            for name, ftype in sample_feature_types.items()
            if ftype == "continuous"
        ][:1]
        discrete_features = [
            name for name, ftype in sample_feature_types.items() if ftype == "discrete"
        ][:1]
        mixed_features = continuous_features + discrete_features
        mixed_weights = TARGET_WEIGHTS[: len(mixed_features)]

        generator_with_sample_data.create_target_variable(
            features_to_use=mixed_features,
            weights=mixed_weights,
            function_type="linear",
            noise_level=TARGET_NOISE_LEVELS["minimal_noise"],
        )

        # Validate structure preservation
        assert generator_with_sample_data.data.shape[0] == original_shape[0]
        assert generator_with_sample_data.data.shape[1] == original_shape[1] + 1
        assert "target" in generator_with_sample_data.data.columns
        assert not generator_with_sample_data.data.isnull().any().any()
        assert generator_with_sample_data.data["target"].std() > 0

    def test_neural_network_scenarios(
        self,
        generator_factory,
        sample_feature_params,
        sample_feature_types,
        standard_test_config,
    ):
        """Test comprehensive neural network error propagation research scenarios"""
        from .fixtures.sample_data import TARGET_WEIGHTS, TARGET_NOISE_LEVELS

        # Test multiple scenarios for neural network research
        research_scenarios = [
            ("linear", "minimal_noise", "Simple regression for baseline NN training"),
            (
                "polynomial",
                "standard_noise",
                "Complex non-linear for advanced NN testing",
            ),
            ("logistic", "standard_noise", "Classification scenarios for NN research"),
        ]

        results = {}
        for func_type, noise_level, description in research_scenarios:
            gen = generator_factory(
                n_samples=standard_test_config["samples"],
                n_features=len(sample_feature_params),
            )
            gen.generate_features(sample_feature_params, sample_feature_types)

            features_to_use = list(sample_feature_params.keys())[:3]
            weights = TARGET_WEIGHTS[: len(features_to_use)]

            gen.create_target_variable(
                features_to_use=features_to_use,
                weights=weights,
                function_type=func_type,
                noise_level=TARGET_NOISE_LEVELS[noise_level],
            )

            results[func_type] = {
                "target_std": gen.data["target"].std(),
                "target_mean": gen.data["target"].mean(),
                "description": description,
            }

            # Validate specific properties
            if func_type == "logistic":
                assert all(0 <= val <= 1 for val in gen.data["target"])
                target_range = gen.data["target"].max() - gen.data["target"].min()
                # More realistic expectation for logistic range with typical data
                assert target_range > 0.01, (
                    f"Logistic target range too small: {target_range}"
                )
                # Ensure there's some variation (not all the same value)
                assert gen.data["target"].std() > 0.001, (
                    f"Logistic target has insufficient variation: {gen.data['target'].std()}"
                )
            else:
                assert gen.data["target"].std() > 0.1, (
                    f"{func_type} target should have variation"
                )

        # All should be valid for NN training
        for scenario, stats in results.items():
            assert 0.001 < stats["target_std"] < 50, (
                f"Invalid std for {scenario}: {stats['target_std']}"
            )
            if scenario != "logistic":  # Logistic has bounded mean around 0.5
                assert abs(stats["target_mean"]) < 100, (
                    f"Invalid mean for {scenario}: {stats['target_mean']}"
                )
