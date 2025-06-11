"""
Tests for feature generation functionality with specific parameter configurations.
"""

import numpy as np
import pandas as pd

from data_generator_module.gaussian_data_generator import GaussianDataGenerator


class TestFeatureGeneration:
    def test_core_feature_generation_functionality(
        self, generator_with_sample_data, sample_feature_params, sample_feature_types
    ):
        """Test core feature generation with statistical validation and structure checks"""

        # Validate basic structure
        assert generator_with_sample_data.data is not None
        assert generator_with_sample_data.data.shape == (
            100,
            len(sample_feature_params),
        )
        assert list(generator_with_sample_data.data.columns) == list(
            sample_feature_params.keys()
        )

        # Validate parameter storage
        assert generator_with_sample_data.feature_parameters == sample_feature_params
        assert generator_with_sample_data.feature_types == sample_feature_types

        # Check feature distributions with statistical confidence intervals
        for feature_name, params in sample_feature_params.items():
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

    def test_feature_types_and_data_integrity(
        self, generator_with_sample_data, sample_feature_types
    ):
        """Test continuous vs discrete feature handling and data integrity"""

        # Test continuous features
        continuous_features = [
            name
            for name, ftype in sample_feature_types.items()
            if ftype == "continuous"
        ]
        assert len(continuous_features) > 0, "No continuous features found in fixture"

        for feature in continuous_features:
            feature_data = generator_with_sample_data.data[feature]
            # Continuous features should have decimal places
            has_decimals = any(not float(val).is_integer() for val in feature_data)
            assert has_decimals, (
                f"Continuous feature {feature} should contain decimal values"
            )

        # Test discrete features
        discrete_features = [
            name for name, ftype in sample_feature_types.items() if ftype == "discrete"
        ]
        for feature in discrete_features:
            feature_data = generator_with_sample_data.data[feature]
            # Discrete features should be integers after generation
            all_integers = all(float(val).is_integer() for val in feature_data)
            assert all_integers, (
                f"Discrete feature {feature} contains non-integer values"
            )

        # Validate no missing values
        assert not generator_with_sample_data.data.isnull().any().any(), (
            "Data should not contain NaN values"
        )

    def test_reproducibility_and_seed_behavior(
        self,
        reproducible_generators,
        different_seed_generators,
        sample_feature_params,
        sample_feature_types,
        test_seeds,
    ):
        """Test comprehensive reproducibility behavior across different scenarios"""

        # Test identical seeds produce identical results
        gen1, gen2 = reproducible_generators
        gen1.generate_features(sample_feature_params, sample_feature_types)
        gen2.generate_features(sample_feature_params, sample_feature_types)

        pd.testing.assert_frame_equal(
            gen1.data,
            gen2.data,
            check_dtype=True,
        )

        # Test different seeds produce different results
        gen3, gen4 = different_seed_generators
        gen3.generate_features(sample_feature_params, sample_feature_types)
        gen4.generate_features(sample_feature_params, sample_feature_types)

        assert not gen3.data.equals(gen4.data), (
            "Different seeds should produce different data"
        )
        assert gen3.data.shape == gen4.data.shape, "Data shape should be consistent"
        assert list(gen3.data.columns) == list(gen4.data.columns), (
            "Columns should be consistent"
        )

        # Test statistical consistency across multiple seeds
        results = []
        for seed in test_seeds[:3]:  # Use first 3 seeds
            gen = GaussianDataGenerator(
                n_samples=100, n_features=len(sample_feature_params), random_state=seed
            )
            gen.generate_features(sample_feature_params, sample_feature_types)

            # Collect statistics for each feature
            feature_stats = {}
            for feature_name in sample_feature_params.keys():
                feature_data = gen.data[feature_name]
                feature_stats[feature_name] = {
                    "mean": feature_data.mean(),
                    "std": feature_data.std(),
                }
            results.append(feature_stats)

        # Verify statistical consistency across seeds
        for feature_name, expected_params in sample_feature_params.items():
            means = [result[feature_name]["mean"] for result in results]
            stds = [result[feature_name]["std"] for result in results]

            # All means should be close to expected mean
            for mean in means:
                assert abs(mean - expected_params["mean"]) < 0.5, (
                    f"Mean for {feature_name} deviates too much: {mean} vs {expected_params['mean']}"
                )

            # All stds should be close to expected std
            for std in stds:
                assert abs(std - expected_params["std"]) < 0.3, (
                    f"Std for {feature_name} deviates too much: {std} vs {expected_params['std']}"
                )

    def test_initialisation_and_validation(
        self,
        basic_generator,
        generator_factory,
        test_seeds,
        sample_feature_params,
        sample_feature_types,
    ):
        """Test generator initialisation, validation, and parameter handling"""

        # Test basic generator state before data generation
        assert basic_generator.data is None
        assert basic_generator.n_samples == 100
        assert basic_generator.n_features == 3
        assert basic_generator.random_state == test_seeds[0]
        assert basic_generator.feature_parameters == {}
        assert basic_generator.feature_types == {}

        # Test generator factory flexibility
        gen_default = generator_factory()
        assert gen_default.n_samples == 100
        assert gen_default.n_features == 5
        assert gen_default.random_state == test_seeds[0]

        gen_custom = generator_factory(
            n_samples=200, n_features=10, random_state=test_seeds[1]
        )
        assert gen_custom.n_samples == 200
        assert gen_custom.n_features == 10
        assert gen_custom.random_state == test_seeds[1]

        # Test parameter consistency validation
        generator = generator_factory(
            n_samples=100, n_features=len(sample_feature_params)
        )

        # Create an inconsistency
        mismatched_types = sample_feature_types.copy()
        mismatched_types["extra_feature"] = "continuous"

        # Test graceful handling: method ignores extra "extra_feature" type
        try:
            generator.generate_features(sample_feature_params, mismatched_types)
            # Validate that generated columns match the parameter keys
            assert all(
                col in sample_feature_params.keys() for col in generator.data.columns
            )
        except (ValueError, KeyError):
            # Raising an error instead is acceptable
            pass

    def test_comprehensive_dataset_configurations(
        self,
        generator_factory,
        sample_feature_params,
        sample_feature_types,
        test_seeds,
        large_dataset,
    ):
        """Test feature generation across all dataset configurations including performance"""
        from .fixtures.sample_data import DATASET_CONFIGS

        # Test all standard dataset configurations
        for config_name, config in DATASET_CONFIGS.items():
            if config_name == "performance_benchmark":
                continue  # Skip performance test here

            gen = generator_factory(
                n_samples=config["samples"],
                n_features=len(sample_feature_params),
                random_state=test_seeds[0],
            )

            gen.generate_features(sample_feature_params, sample_feature_types)

            # Validate structure
            assert gen.data.shape[0] == config["samples"]
            assert gen.data.shape[1] == len(sample_feature_params)
            assert list(gen.data.columns) == list(sample_feature_params.keys())

            # Basic statistical checks for larger datasets
            if config["samples"] >= 100:
                for feature_name, params in sample_feature_params.items():
                    feature_data = gen.data[feature_name]
                    # Allow wider tolerance for different sample sizes
                    tolerance = max(0.5, 2.0 / np.sqrt(config["samples"]))
                    assert abs(feature_data.mean() - params["mean"]) < tolerance

        # Test large dataset performance
        assert large_dataset.data is not None
        assert large_dataset.data.shape[0] >= 1000
        assert large_dataset.data.shape[1] >= 5

        # Basic statistical checks for large dataset
        for col in large_dataset.data.columns:
            feature_data = large_dataset.data[col]
            assert abs(feature_data.mean()) < 0.5  # Roughly centered
            assert 0.5 < feature_data.std() < 2.0  # Reasonable variance
            assert not feature_data.isnull().any()  # No missing values

    def test_fixture_consistency_and_edge_cases(
        self, sample_feature_params, sample_feature_types, test_seeds
    ):
        """Test fixture consistency and edge cases"""

        # Test sample data fixture consistency
        assert isinstance(sample_feature_params, dict)
        assert len(sample_feature_params) > 0

        for feature_name, params in sample_feature_params.items():
            assert "mean" in params and "std" in params
            assert isinstance(params["mean"], (int, float))
            assert isinstance(params["std"], (int, float))
            assert params["std"] > 0

        # Test feature types fixture consistency
        assert isinstance(sample_feature_types, dict)
        assert len(sample_feature_types) > 0

        for feature_name, ftype in sample_feature_types.items():
            assert ftype in ["continuous", "discrete"]

        # Test consistency between parameters and types
        assert set(sample_feature_params.keys()) == set(sample_feature_types.keys())

        # Test seeds fixture consistency
        assert isinstance(test_seeds, list)
        assert len(test_seeds) >= 2  # At least 2 seeds for different tests
        assert all(isinstance(seed, int) for seed in test_seeds)
        assert len(set(test_seeds)) == len(test_seeds)  # Seeds should be unique
