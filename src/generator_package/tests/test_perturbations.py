"""
Tests for data perturbation functionality.
"""

import pytest
import numpy as np
from .fixtures.sample_data import (
    PERTURBATION_LEVELS,
)


class TestPerturbations:
    @pytest.mark.parametrize("noise_level", PERTURBATION_LEVELS.keys())
    def test_all_noise_levels_comprehensive(
        self,
        generator_factory,
        sample_feature_params,
        sample_feature_types,
        test_seeds,
        noise_level,
    ):
        """Test all noise levels from fixtures including noise_free"""
        scale = PERTURBATION_LEVELS[noise_level]

        gen = generator_factory(
            n_samples=100,
            n_features=len(sample_feature_params),
            random_state=test_seeds[0],
        )
        gen.generate_features(sample_feature_params, sample_feature_types)
        original_data = gen.data.copy()

        continuous_features = [
            name
            for name, ftype in sample_feature_types.items()
            if ftype == "continuous"
        ]

        gen.add_perturbations(
            features=continuous_features, perturbation_type="gaussian", scale=scale
        )

        for feature in continuous_features:
            diff = gen.data[feature] - original_data[feature]

            if noise_level == "noise_free":
                # Should be exactly zero
                np.testing.assert_array_equal(
                    original_data[feature].values,
                    gen.data[feature].values,
                    err_msg=f"Noise-free should produce no changes for {feature}",
                )
            else:
                # check for changes
                assert not np.array_equal(
                    original_data[feature].values,
                    gen.data[feature].values,
                ), f"Feature {feature} should change with {noise_level}"

                # Magnitude should correlate with scale from fixtures
                expected_std = scale
                actual_std = diff.std()
                tolerance = max(
                    0.05, expected_std * 0.4
                )  # Allow for statistical variation

                assert abs(actual_std - expected_std) < tolerance, (
                    f"{noise_level} (scale={scale}) produced std={actual_std:.3f}, "
                    f"expected ~{expected_std:.3f} for {feature}"
                )

    @pytest.mark.parametrize("perturbation_type", ["gaussian", "uniform"])
    def test_perturbation_types_by_feature_type(
        self,
        generator_factory,
        sample_feature_params,
        sample_feature_types,
        test_seeds,
        perturbation_type,
    ):
        """Test both perturbation types on continuous and discrete features"""
        gen = generator_factory(
            n_samples=100,
            n_features=len(sample_feature_params),
            random_state=test_seeds[0],
        )
        gen.generate_features(sample_feature_params, sample_feature_types)
        original_data = gen.data.copy()

        # Test continuous features with realistic noise
        continuous_features = [
            name
            for name, ftype in sample_feature_types.items()
            if ftype == "continuous"
        ]
        if continuous_features:
            continuous_scale = PERTURBATION_LEVELS["realistic_noise"]
            gen.add_perturbations(
                features=continuous_features,
                perturbation_type=perturbation_type,
                scale=continuous_scale,
            )

            for feature in continuous_features:
                assert not np.array_equal(
                    original_data[feature].values,
                    gen.data[feature].values,
                ), (
                    f"Continuous feature {feature} not perturbed with {perturbation_type}"
                )

                if perturbation_type == "uniform":
                    diff = gen.data[feature] - original_data[feature]
                    assert all(abs(d) <= continuous_scale for d in diff), (
                        f"Uniform perturbation exceeds bounds for {feature}"
                    )

        # Test discrete features with higher noise
        discrete_features = [
            name for name, ftype in sample_feature_types.items() if ftype == "discrete"
        ]
        if discrete_features:
            # Create fresh generator for discrete test
            fresh_gen = generator_factory(
                n_samples=100,
                n_features=len(sample_feature_params),
                random_state=test_seeds[0],
            )
            fresh_gen.generate_features(sample_feature_params, sample_feature_types)
            original_discrete = fresh_gen.data.copy()

            # Use extreme noise for discrete features
            discrete_scale = PERTURBATION_LEVELS["extreme_noise"]
            fresh_gen.add_perturbations(
                features=discrete_features,
                perturbation_type=perturbation_type,
                scale=discrete_scale,
            )

            for feature in discrete_features:
                changes = (original_discrete[feature] != fresh_gen.data[feature]).sum()
                assert changes > 0, (
                    f"Discrete feature {feature} should change with {perturbation_type}"
                )

                # Verify discrete features remain integers
                all_integers = all(
                    float(val).is_integer() for val in fresh_gen.data[feature]
                )
                assert all_integers, (
                    f"Discrete feature {feature} should remain integers"
                )

    def test_selective_and_sequential_perturbations(
        self, generator_with_sample_data, sample_feature_params
    ):
        """Test selective feature perturbation and sequential application"""
        original_data = generator_with_sample_data.data.copy()
        all_features = list(sample_feature_params.keys())
        features_to_perturb = [all_features[0]]  # Only perturb first feature
        scale = PERTURBATION_LEVELS["realistic_noise"]

        # Test selective perturbation
        generator_with_sample_data.add_perturbations(
            features=features_to_perturb, perturbation_type="gaussian", scale=scale
        )

        # Check selected feature changed
        for feature in features_to_perturb:
            assert not np.array_equal(
                original_data[feature].values,
                generator_with_sample_data.data[feature].values,
            ), f"Selected feature {feature} was not perturbed"

        # Check other features unchanged
        unchanged_features = [f for f in all_features if f not in features_to_perturb]
        for feature in unchanged_features:
            (
                np.testing.assert_array_equal(
                    original_data[feature].values,
                    generator_with_sample_data.data[feature].values,
                ),
                f"Unselected feature {feature} should not have changed",
            )

        # Test sequential perturbation on the same feature
        first_perturbation = generator_with_sample_data.data.copy()
        low_noise_scale = PERTURBATION_LEVELS["low_noise"]

        # Apply second perturbation
        generator_with_sample_data.add_perturbations(
            features=[features_to_perturb[0]],
            perturbation_type="uniform",
            scale=low_noise_scale,
        )

        # Verify cumulative changes
        assert not np.array_equal(
            first_perturbation[features_to_perturb[0]].values,
            generator_with_sample_data.data[features_to_perturb[0]].values,
        ), "Sequential perturbation should modify data further"

    def test_perturbation_validation_and_structure(
        self, generator_with_sample_data, sample_feature_params
    ):
        """Test comprehensive validation and data structure preservation"""
        actual_feature = list(sample_feature_params.keys())[0]
        valid_scale = PERTURBATION_LEVELS["realistic_noise"]

        # Test validation errors
        with pytest.raises(
            ValueError, match="perturbation_type must be 'gaussian' or 'uniform'"
        ):
            generator_with_sample_data.add_perturbations(
                features=[actual_feature],
                perturbation_type="invalid_type",
                scale=valid_scale,
            )

        with pytest.raises(ValueError, match="Scale must be non-negative"):
            generator_with_sample_data.add_perturbations(
                features=[actual_feature], perturbation_type="gaussian", scale=-0.1
            )

        with pytest.raises(ValueError, match="Feature.*not found"):
            generator_with_sample_data.add_perturbations(
                features=["nonexistent_feature"],
                perturbation_type="gaussian",
                scale=valid_scale,
            )

        # Test structure preservation
        original_shape = generator_with_sample_data.data.shape
        original_columns = generator_with_sample_data.data.columns.tolist()

        generator_with_sample_data.add_perturbations(
            features=list(sample_feature_params.keys()),
            perturbation_type="gaussian",
            scale=valid_scale,
        )

        # Verify structure preservation
        assert generator_with_sample_data.data.shape == original_shape
        assert generator_with_sample_data.data.columns.tolist() == original_columns
        assert not generator_with_sample_data.data.isnull().any().any()

    def test_reproducibility_and_seed_differences(
        self,
        reproducible_generators,
        different_seed_generators,
        sample_feature_params,
        sample_feature_types,
    ):
        """Test both reproducible behavior and seed differences"""
        test_feature = list(sample_feature_params.keys())[0]
        realistic_scale = PERTURBATION_LEVELS["realistic_noise"]

        # Test reproducible behavior
        gen1, gen2 = reproducible_generators
        gen1.generate_features(sample_feature_params, sample_feature_types)
        gen2.generate_features(sample_feature_params, sample_feature_types)

        perturbation_params = {
            "features": [test_feature],
            "perturbation_type": "gaussian",
            "scale": realistic_scale,
        }

        gen1.add_perturbations(**perturbation_params)
        gen2.add_perturbations(**perturbation_params)

        # Results should be identical
        np.testing.assert_array_equal(
            gen1.data[test_feature].values,
            gen2.data[test_feature].values,
            err_msg="Perturbations should be reproducible with same seed",
        )

        # Test different seeds produce different results
        gen3, gen4 = different_seed_generators
        gen3.generate_features(sample_feature_params, sample_feature_types)
        gen4.generate_features(sample_feature_params, sample_feature_types)

        orig3 = gen3.data[test_feature].copy()
        orig4 = gen4.data[test_feature].copy()

        gen3.add_perturbations(**perturbation_params)
        gen4.add_perturbations(**perturbation_params)

        # The perturbations should be different due to different seeds
        diff3 = gen3.data[test_feature] - orig3
        diff4 = gen4.data[test_feature] - orig4

        assert not np.array_equal(diff3.values, diff4.values), (
            "Different seeds should produce different perturbations"
        )

    def test_neural_network_research_scenarios_comprehensive(
        self,
        generator_factory,
        sample_feature_params,
        sample_feature_types,
        standard_test_config,
    ):
        """Test complete neural network research scenarios including mixed features"""
        # Test research scenarios with different noise levels
        research_scenarios = [
            ("low_noise", "Low noise for baseline NN training"),
            ("realistic_noise", "Realistic noise for standard NN research"),
            ("high_noise", "High noise for robustness testing"),
        ]

        for noise_level, description in research_scenarios:
            gen = generator_factory(
                n_samples=standard_test_config["samples"],
                n_features=len(sample_feature_params),
            )
            gen.generate_features(sample_feature_params, sample_feature_types)
            original_std = gen.data.std().mean()

            # Apply perturbation using fixture values
            scale = PERTURBATION_LEVELS[noise_level]
            gen.add_perturbations(
                features=list(sample_feature_params.keys()),
                perturbation_type="gaussian",
                scale=scale,
            )

            # Verify complexity
            perturbed_std = gen.data.std().mean()

            if noise_level != "low_noise":
                assert perturbed_std > original_std, (
                    f"{description}: Should increase data complexity"
                )

            # Should maintain reasonable bounds for NN training
            feature_means = gen.data.mean()
            assert all(abs(mean) < 50 for mean in feature_means), (
                f"{description}: Data should remain in reasonable range for NN training"
            )

        # Test mixed feature types handling
        continuous_features = [
            name
            for name, ftype in sample_feature_types.items()
            if ftype == "continuous"
        ][:1]
        discrete_features = [
            name for name, ftype in sample_feature_types.items() if ftype == "discrete"
        ][:1]

        if continuous_features and discrete_features:
            mixed_gen = generator_factory(
                n_samples=100, n_features=len(sample_feature_params)
            )
            mixed_gen.generate_features(sample_feature_params, sample_feature_types)
            original_mixed = mixed_gen.data.copy()

            # Apply different appropriate scales for each feature type
            mixed_gen.add_perturbations(
                features=continuous_features,
                perturbation_type="gaussian",
                scale=PERTURBATION_LEVELS["realistic_noise"],
            )
            mixed_gen.add_perturbations(
                features=discrete_features,
                perturbation_type="gaussian",
                scale=PERTURBATION_LEVELS["extreme_noise"],
            )

            # Verify mixed feature handling
            for feature in continuous_features + discrete_features:
                assert not np.array_equal(
                    original_mixed[feature].values,
                    mixed_gen.data[feature].values,
                ), f"Mixed feature {feature} should be perturbed"

    @pytest.mark.slow
    def test_performance_and_edge_cases(
        self,
        large_dataset,
        basic_generator,
        sample_feature_params,
        sample_feature_types,
    ):
        """Test performance with large dataset and edge cases"""
        # Test large dataset performance
        original_data = large_dataset.data.copy()
        features_to_test = large_dataset.data.columns[:3].tolist()

        large_dataset.add_perturbations(
            features=features_to_test,
            perturbation_type="gaussian",
            scale=PERTURBATION_LEVELS["realistic_noise"],
        )

        # Verify performance and correctness
        assert large_dataset.data.shape[0] >= 1000  # From large_dataset fixture

        for feature in features_to_test:
            assert not np.array_equal(
                original_data[feature].values, large_dataset.data[feature].values
            ), f"Large dataset feature {feature} should be perturbed"

        # Test edge case: perturbation before data generation
        with pytest.raises(ValueError, match="No data generated"):
            basic_generator.add_perturbations(
                features=["feature_0"],
                perturbation_type="gaussian",
                scale=PERTURBATION_LEVELS["realistic_noise"],
            )
