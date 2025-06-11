"""
Tests for data validation functionality.
"""

import pytest
from generator_package.validators import DataGeneratorValidators
from .fixtures.sample_data import (
    SAMPLE_FEATURE_PARAMS,
    SAMPLE_FEATURE_TYPES,
    INVALID_FEATURE_PARAMS,
    TARGET_WEIGHTS,
    FUNCTION_TYPES,
    FEATURE_NOISE_LEVELS,
    REPRODUCIBILITY_SEEDS,
    DATASET_CONFIGS,
)


class TestDataGeneratorValidators:
    def test_validate_feature_parameters_comprehensive(self, sample_feature_params):
        """Test feature parameter validation with valid and invalid cases"""
        # Test valid parameters
        DataGeneratorValidators.validate_feature_parameters(sample_feature_params)

        # Test invalid cases
        invalid_cases = [
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
        ]

        for invalid_params, expected_error in invalid_cases:
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
    def test_validate_init_parameters_comprehensive(
        self, n_samples, n_features, should_pass, standard_test_config
    ):
        """Test initialisation parameter validation with multiple scenarios"""
        if should_pass:
            DataGeneratorValidators.validate_init_parameters(n_samples, n_features)
            # Also verify with actual config values
            DataGeneratorValidators.validate_init_parameters(
                standard_test_config["samples"], standard_test_config["features"]
            )
        else:
            with pytest.raises(ValueError):
                DataGeneratorValidators.validate_init_parameters(n_samples, n_features)

    def test_validate_feature_types_comprehensive(self, sample_feature_types):
        """Test feature type validation with valid and invalid cases"""
        # Test valid types
        DataGeneratorValidators.validate_feature_types(sample_feature_types)

        # Test invalid cases
        invalid_cases = [
            "not_a_dict",
            {"feature1": "invalid_type"},
            {"feature1": "categorical"},
            {"feature1": 123},
            {"feature1": None},
        ]

        for invalid_types in invalid_cases:
            with pytest.raises(ValueError):
                DataGeneratorValidators.validate_feature_types(invalid_types)

    def test_validate_perturbation_parameters_comprehensive(
        self, generator_with_sample_data, sample_feature_params
    ):
        """Test comprehensive perturbation parameter validation"""

        valid_features = list(sample_feature_params.keys())[:2]

        # Test valid cases using fixture data
        for perturbation_type in ["gaussian", "uniform"]:
            for scale_name, scale_value in FEATURE_NOISE_LEVELS.items():
                DataGeneratorValidators.validate_perturbation_parameters(
                    perturbation_type,
                    valid_features,
                    scale_value,
                    generator_with_sample_data.data,
                )

        # Test invalid cases
        invalid_cases = [
            (
                "invalid_type",
                valid_features,
                0.1,
                generator_with_sample_data.data,
                "perturbation_type must be 'gaussian' or 'uniform'",
            ),
            (
                "gaussian",
                valid_features,
                -0.1,
                generator_with_sample_data.data,
                "Scale must be non-negative",
            ),
            (
                "gaussian",
                ["nonexistent"],
                0.1,
                generator_with_sample_data.data,
                "Feature 'nonexistent' not found in data",
            ),
            ("gaussian", valid_features, 0.1, None, "No data generated"),
        ]

        for perturbation_type, features, scale, data, expected_error in invalid_cases:
            with pytest.raises(ValueError, match=expected_error):
                DataGeneratorValidators.validate_perturbation_parameters(
                    perturbation_type, features, scale, data
                )

    def test_validate_target_parameters_comprehensive(
        self, generator_with_sample_data, sample_feature_params
    ):
        """Test comprehensive target parameter validation"""

        features_to_use = list(sample_feature_params.keys())[:2]
        weights = TARGET_WEIGHTS[: len(features_to_use)]

        # Test valid cases
        for function_type in FUNCTION_TYPES:
            DataGeneratorValidators.validate_target_parameters(
                features_to_use, weights, function_type, generator_with_sample_data.data
            )

        # Test invalid cases
        invalid_cases = [
            (features_to_use, weights, "linear", None, "No data generated"),
            (
                ["nonexistent_feature_1"],
                [1.0],
                "linear",
                generator_with_sample_data.data,
                "Features not found in data",
            ),
            (
                features_to_use,
                [1.0],
                "linear",
                generator_with_sample_data.data,
                "Number of weights must match number of features",
            ),
            (
                features_to_use,
                weights,
                "invalid_function",
                generator_with_sample_data.data,
                "function_type must be 'linear', 'polynomial', or 'logistic'",
            ),
        ]

        for (
            features,
            test_weights,
            function_type,
            data,
            expected_error,
        ) in invalid_cases:
            with pytest.raises(ValueError, match=expected_error):
                DataGeneratorValidators.validate_target_parameters(
                    features, test_weights, function_type, data
                )

    def test_validate_visualisation_parameters_comprehensive(
        self, generator_with_sample_data, sample_feature_params
    ):
        """Test comprehensive visualisation parameter validation"""
        features = list(sample_feature_params.keys())

        # Test valid case
        DataGeneratorValidators.validate_visualisation_parameters(
            features,
            max_features_to_show=3,
            n_bins=20,
            data=generator_with_sample_data.data,
        )

        # Test invalid cases
        invalid_cases = [
            (features, 3, 20, None, "No data generated to visualise"),
            (
                ["nonexistent_feature"],
                3,
                20,
                generator_with_sample_data.data,
                "Features not found in data",
            ),
            (
                features,
                0,
                20,
                generator_with_sample_data.data,
                "max_features_to_show must be positive",
            ),
            (
                features,
                -1,
                20,
                generator_with_sample_data.data,
                "max_features_to_show must be positive",
            ),
            (
                features,
                3,
                0,
                generator_with_sample_data.data,
                "n_bins must be positive",
            ),
            (
                features,
                3,
                -5,
                generator_with_sample_data.data,
                "n_bins must be positive",
            ),
        ]

        for test_features, max_features, n_bins, data, expected_error in invalid_cases:
            with pytest.raises(ValueError, match=expected_error):
                DataGeneratorValidators.validate_visualisation_parameters(
                    test_features,
                    max_features_to_show=max_features,
                    n_bins=n_bins,
                    data=data,
                )

    def test_complete_pipeline_integration(
        self,
        generator_factory,
        sample_feature_params,
        sample_feature_types,
        standard_test_config,
        test_seeds,
    ):
        """Test validators work throughout complete pipeline using fixtures"""

        # Test complete pipeline validation using all fixtures
        DataGeneratorValidators.validate_init_parameters(
            standard_test_config["samples"], standard_test_config["features"]
        )

        # Create and validate features using factory
        gen = generator_factory(
            n_samples=standard_test_config["samples"],
            n_features=len(sample_feature_params),
            random_state=test_seeds[0],
        )
        DataGeneratorValidators.validate_feature_parameters(sample_feature_params)
        DataGeneratorValidators.validate_feature_types(sample_feature_types)

        # Generate data and validate operations
        gen.generate_features(sample_feature_params, sample_feature_types)

        # Validate perturbation using fixtures
        test_features = list(sample_feature_params.keys())[:1]
        DataGeneratorValidators.validate_perturbation_parameters(
            "gaussian", test_features, FEATURE_NOISE_LEVELS["realistic_noise"], gen.data
        )

        # Validate target creation using fixtures
        target_features = list(sample_feature_params.keys())[:2]
        target_weights = TARGET_WEIGHTS[: len(target_features)]
        DataGeneratorValidators.validate_target_parameters(
            target_features, target_weights, "linear", gen.data
        )

        # Validate visualisation using fixtures
        DataGeneratorValidators.validate_visualisation_parameters(
            list(sample_feature_params.keys()),
            max_features_to_show=2,
            n_bins=10,
            data=gen.data,
        )

    def test_fixture_consistency_and_cross_compatibility(
        self,
        sample_feature_params,
        sample_feature_types,
        test_seeds,
        standard_test_config,
        generator_with_sample_data,
        different_seed_generators,
    ):
        """Test fixture consistency and cross-fixture compatibility"""
        # Test sample data fixture consistency
        assert (
            isinstance(sample_feature_params, dict) and len(sample_feature_params) > 0
        )
        for feature_name, params in sample_feature_params.items():
            assert "mean" in params and "std" in params
            assert isinstance(params["mean"], (int, float))
            assert isinstance(params["std"], (int, float))
            assert params["std"] > 0

        # Test feature types fixture consistency
        assert isinstance(sample_feature_types, dict) and len(sample_feature_types) > 0
        for feature_name, ftype in sample_feature_types.items():
            assert ftype in ["continuous", "discrete"]

        # Test consistency between parameters and types
        assert set(sample_feature_params.keys()) == set(sample_feature_types.keys())

        # Test seeds fixture consistency
        assert isinstance(test_seeds, list) and len(test_seeds) >= 2
        assert all(isinstance(seed, int) for seed in test_seeds)
        assert len(set(test_seeds)) == len(test_seeds)  # All unique

        # Test standard config fixture consistency
        assert isinstance(standard_test_config, dict)
        assert "samples" in standard_test_config and "features" in standard_test_config
        assert (
            standard_test_config["samples"] > 0 and standard_test_config["features"] > 0
        )

        # Test cross-fixture compatibility
        assert generator_with_sample_data.feature_parameters == sample_feature_params
        assert generator_with_sample_data.feature_types == sample_feature_types

        # Test different_seed_generators fixture
        gen1, gen2 = different_seed_generators
        gen1.generate_features(sample_feature_params, sample_feature_types)
        gen2.generate_features(sample_feature_params, sample_feature_types)

        assert gen1.data.shape == gen2.data.shape
        assert list(gen1.data.columns) == list(gen2.data.columns)
        assert gen1.random_state != gen2.random_state

    @pytest.mark.slow
    def test_validators_with_large_dataset_edge_cases(
        self, large_dataset, sample_feature_params
    ):
        """Test validators with large dataset and edge cases"""
        # Test that validators can handle large dataset from fixture
        features = large_dataset.data.columns[:3].tolist()

        DataGeneratorValidators.validate_visualisation_parameters(
            features,
            max_features_to_show=len(features),
            n_bins=50,
            data=large_dataset.data,
        )

        # Verify large dataset characteristics
        assert large_dataset.data.shape[0] >= 1000
        assert large_dataset.data.shape[1] >= 5
        assert not large_dataset.data.isnull().any().any()

        # Test edge cases with extreme values from sample data
        extreme_feature_params = {
            name: {"mean": params["mean"] * 100, "std": params["std"] * 10}
            for name, params in sample_feature_params.items()
        }
        DataGeneratorValidators.validate_feature_parameters(extreme_feature_params)

        # Test with single feature
        single_feature_params = {
            list(sample_feature_params.keys())[0]: list(sample_feature_params.values())[
                0
            ]
        }
        DataGeneratorValidators.validate_feature_parameters(single_feature_params)
