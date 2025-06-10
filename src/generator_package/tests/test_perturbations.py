"""
Tests for data perturbation functionality.
"""

import pytest
import numpy as np
from generator_package.gaussian_data_generator import GaussianDataGenerator
from .fixtures.sample_data import (
    SAMPLE_FEATURE_PARAMS,
    SAMPLE_FEATURE_TYPES,
    PERTURBATION_LEVELS,
    REPRODUCIBILITY_SEEDS,
    DATASET_CONFIGS,
    ERROR_PATTERNS,
)


class TestPerturbations:
    def test_realistic_noise_continuous_features(self, generator_with_sample_data):
        """Test realistic noise on continuous features"""
        original_data = generator_with_sample_data.data.copy()
        continuous_features = [
            name
            for name, ftype in SAMPLE_FEATURE_TYPES.items()
            if ftype == "continuous"
        ]
        scale = PERTURBATION_LEVELS["realistic_noise"]

        generator_with_sample_data.add_perturbations(
            features=continuous_features, perturbation_type="gaussian", scale=scale
        )

        for feature in continuous_features:
            assert not np.array_equal(
                original_data[feature].values,
                generator_with_sample_data.data[feature].values,
            ), f"Continuous feature {feature} was not perturbed"

    def test_minimal_noise_continuous_features(self, generator_with_sample_data):
        """Test minimal noise on continuous features"""
        original_data = generator_with_sample_data.data.copy()
        continuous_features = [
            name
            for name, ftype in SAMPLE_FEATURE_TYPES.items()
            if ftype == "continuous"
        ]
        scale = PERTURBATION_LEVELS["minimal_noise"]

        generator_with_sample_data.add_perturbations(
            features=continuous_features, perturbation_type="gaussian", scale=scale
        )

        for feature in continuous_features:
            assert not np.array_equal(
                original_data[feature].values,
                generator_with_sample_data.data[feature].values,
            ), f"Continuous feature {feature} was not perturbed with minimal noise"

            # Check that changes are small for minimal noise
            diff = generator_with_sample_data.data[feature] - original_data[feature]
            assert diff.std() < 0.05, (
                f"Minimal noise should produce small changes for {feature}"
            )

    def test_extreme_noise_continuous_features(self, generator_with_sample_data):
        """Test extreme noise on continuous features"""
        original_data = generator_with_sample_data.data.copy()
        continuous_features = [
            name
            for name, ftype in SAMPLE_FEATURE_TYPES.items()
            if ftype == "continuous"
        ]
        scale = PERTURBATION_LEVELS["extreme_noise"]

        generator_with_sample_data.add_perturbations(
            features=continuous_features, perturbation_type="gaussian", scale=scale
        )

        for feature in continuous_features:
            assert not np.array_equal(
                original_data[feature].values,
                generator_with_sample_data.data[feature].values,
            ), f"Continuous feature {feature} was not perturbed with extreme noise"

            # Check that changes are substantial for extreme noise
            diff = generator_with_sample_data.data[feature] - original_data[feature]
            assert diff.std() > 0.3, (
                f"Extreme noise should produce large changes for {feature}"
            )

    def test_uniform_perturbation_continuous_features(self, generator_with_sample_data):
        """Test uniform perturbation on continuous features"""
        original_data = generator_with_sample_data.data.copy()
        continuous_features = [
            name
            for name, ftype in SAMPLE_FEATURE_TYPES.items()
            if ftype == "continuous"
        ]
        scale = PERTURBATION_LEVELS["realistic_noise"]

        generator_with_sample_data.add_perturbations(
            features=continuous_features, perturbation_type="uniform", scale=scale
        )

        for feature in continuous_features:
            assert not np.array_equal(
                original_data[feature].values,
                generator_with_sample_data.data[feature].values,
            ), f"Continuous feature {feature} was not perturbed with uniform noise"

            # For uniform distribution, changes should be within [-scale, scale]
            diff = generator_with_sample_data.data[feature] - original_data[feature]
            assert all(abs(d) <= scale for d in diff), (
                f"Uniform perturbation exceeds bounds for {feature}"
            )

    def test_realistic_noise_discrete_features(self, generator_with_sample_data):
        """Test realistic noise on discrete features with appropriate scale"""
        original_data = generator_with_sample_data.data.copy()
        discrete_features = [
            name for name, ftype in SAMPLE_FEATURE_TYPES.items() if ftype == "discrete"
        ]
        # higher scale for discrete features
        scale = PERTURBATION_LEVELS["high_noise"]  # 0.3

        if discrete_features:
            generator_with_sample_data.add_perturbations(
                features=discrete_features, perturbation_type="gaussian", scale=scale
            )

            for feature in discrete_features:
                changes = (
                    original_data[feature] != generator_with_sample_data.data[feature]
                ).sum()
                assert changes > 0, (
                    f"Discrete feature {feature} should have some changes"
                )

    def test_uniform_perturbation_discrete_features(self, generator_with_sample_data):
        """Test uniform perturbation on discrete features with higher scale"""
        original_data = generator_with_sample_data.data.copy()
        discrete_features = [
            name for name, ftype in SAMPLE_FEATURE_TYPES.items() if ftype == "discrete"
        ]
        scale = PERTURBATION_LEVELS["extreme_noise"]

        if discrete_features:
            generator_with_sample_data.add_perturbations(
                features=discrete_features, perturbation_type="uniform", scale=scale
            )

            for feature in discrete_features:
                changes = (
                    original_data[feature] != generator_with_sample_data.data[feature]
                ).sum()
                assert changes > 0, (
                    f"Discrete feature {feature} should have changes with uniform noise"
                )

                # Check that discrete features remain integers
                all_integers = all(
                    float(val).is_integer()
                    for val in generator_with_sample_data.data[feature]
                )
                assert all_integers, (
                    f"Discrete feature {feature} should remain integers after perturbation"
                )

    def test_selective_feature_perturbation(self, generator_with_sample_data):
        """Test that only specified features are perturbed"""
        original_data = generator_with_sample_data.data.copy()
        all_features = list(SAMPLE_FEATURE_PARAMS.keys())
        features_to_perturb = [all_features[0]]  # Only perturb first feature
        scale = PERTURBATION_LEVELS["realistic_noise"]

        generator_with_sample_data.add_perturbations(
            features=features_to_perturb, perturbation_type="gaussian", scale=scale
        )

        # Check perturbed feature changed
        for feature in features_to_perturb:
            assert not np.array_equal(
                original_data[feature].values,
                generator_with_sample_data.data[feature].values,
            ), f"Selected feature {feature} was not perturbed"

        # Check other features remained unchanged
        unchanged_features = [f for f in all_features if f not in features_to_perturb]
        for feature in unchanged_features:
            (
                np.testing.assert_array_equal(
                    original_data[feature].values,
                    generator_with_sample_data.data[feature].values,
                ),
                f"Unselected feature {feature} should not have changed",
            )

    def test_multiple_perturbations_sequential(self, generator_with_sample_data):
        """Test applying multiple perturbations sequentially"""
        original_data = generator_with_sample_data.data.copy()
        continuous_features = [
            name
            for name, ftype in SAMPLE_FEATURE_TYPES.items()
            if ftype == "continuous"
        ]
        test_feature = continuous_features[0]

        # Apply first perturbation
        generator_with_sample_data.add_perturbations(
            features=[test_feature],
            perturbation_type="gaussian",
            scale=PERTURBATION_LEVELS["low_noise"],
        )
        first_perturbation = generator_with_sample_data.data.copy()

        # Apply second perturbation
        generator_with_sample_data.add_perturbations(
            features=[test_feature],
            perturbation_type="uniform",
            scale=PERTURBATION_LEVELS["low_noise"],
        )

        # Check data is different from both original and first perturbation
        assert not np.array_equal(
            original_data[test_feature].values,
            generator_with_sample_data.data[test_feature].values,
        ), "Final data should differ from original"

        assert not np.array_equal(
            first_perturbation[test_feature].values,
            generator_with_sample_data.data[test_feature].values,
        ), "Final data should differ from first perturbation"

    def test_perturbation_validation_errors(self, generator_with_sample_data):
        """Test perturbation parameter validation using exact error patterns"""
        actual_feature = list(SAMPLE_FEATURE_PARAMS.keys())[0]

        # Test invalid perturbation type
        with pytest.raises(
            ValueError, match="perturbation_type must be 'gaussian' or 'uniform'"
        ):
            generator_with_sample_data.add_perturbations(
                features=[actual_feature], perturbation_type="invalid_type", scale=0.1
            )

        # Test negative scale
        with pytest.raises(ValueError, match="Scale must be positive"):
            generator_with_sample_data.add_perturbations(
                features=[actual_feature], perturbation_type="gaussian", scale=-0.1
            )

        # Test non-existent feature
        with pytest.raises(ValueError, match="Feature.*not found"):
            generator_with_sample_data.add_perturbations(
                features=["nonexistent_feature"],
                perturbation_type="gaussian",
                scale=0.1,
            )

    def test_perturbation_preserves_data_structure(self, generator_with_sample_data):
        """Test that perturbation preserves DataFrame structure and types"""
        original_shape = generator_with_sample_data.data.shape
        original_columns = generator_with_sample_data.data.columns.tolist()
        all_features = list(SAMPLE_FEATURE_PARAMS.keys())

        generator_with_sample_data.add_perturbations(
            features=all_features,
            perturbation_type="gaussian",
            scale=PERTURBATION_LEVELS["realistic_noise"],
        )

        # Structure should remain the same
        assert generator_with_sample_data.data.shape == original_shape
        assert generator_with_sample_data.data.columns.tolist() == original_columns
        assert not generator_with_sample_data.data.isnull().any().any()

    def test_noise_free_perturbation(self, generator_with_sample_data):
        """Test that noise-free perturbation makes no changes"""
        original_data = generator_with_sample_data.data.copy()
        all_features = list(SAMPLE_FEATURE_PARAMS.keys())
        scale = PERTURBATION_LEVELS["noise_free"]  # 0.0

        generator_with_sample_data.add_perturbations(
            features=all_features, perturbation_type="gaussian", scale=scale
        )

        # Data should remain unchanged
        for feature in all_features:
            (
                np.testing.assert_array_equal(
                    original_data[feature].values,
                    generator_with_sample_data.data[feature].values,
                ),
                f"Feature {feature} should not change with noise-free perturbation",
            )

    def test_all_perturbation_levels_continuous(self, generator_factory):
        """Test all perturbation levels on continuous features"""
        for level_name, scale_value in PERTURBATION_LEVELS.items():
            if level_name == "noise_free":  # Skip zero noise
                continue

            gen = generator_factory(n_samples=50, n_features=len(SAMPLE_FEATURE_PARAMS))
            gen.generate_features(SAMPLE_FEATURE_PARAMS, SAMPLE_FEATURE_TYPES)
            original_data = gen.data.copy()

            continuous_features = [
                name
                for name, ftype in SAMPLE_FEATURE_TYPES.items()
                if ftype == "continuous"
            ]

            gen.add_perturbations(
                features=continuous_features,
                perturbation_type="gaussian",
                scale=scale_value,
            )

            # Verify perturbation was applied
            for feature in continuous_features:
                assert not np.array_equal(
                    original_data[feature].values,
                    gen.data[feature].values,
                ), f"No perturbation applied for {level_name} level on {feature}"

                # Check magnitude correlation with level
                diff = gen.data[feature] - original_data[feature]
                if level_name == "minimal_noise":
                    assert diff.std() < 0.05, (
                        f"Minimal noise too large: {diff.std():.4f}"
                    )
                elif level_name == "extreme_noise":
                    assert diff.std() > 0.2, (
                        f"Extreme noise too small: {diff.std():.4f}"
                    )

    def test_perturbation_reproducibility_with_seeds(
        self, generator_factory, test_seeds
    ):
        """Test perturbation reproducibility"""
        seed = test_seeds[0]
        continuous_features = [
            name
            for name, ftype in SAMPLE_FEATURE_TYPES.items()
            if ftype == "continuous"
        ]
        test_feature = continuous_features[0]

        # Create two identical generators
        gen1 = generator_factory(
            n_samples=100, n_features=len(SAMPLE_FEATURE_PARAMS), random_state=seed
        )
        gen2 = generator_factory(
            n_samples=100, n_features=len(SAMPLE_FEATURE_PARAMS), random_state=seed
        )

        # Generate identical base data
        gen1.generate_features(SAMPLE_FEATURE_PARAMS, SAMPLE_FEATURE_TYPES)
        gen2.generate_features(SAMPLE_FEATURE_PARAMS, SAMPLE_FEATURE_TYPES)

        # Apply same perturbation
        perturbation_params = {
            "features": [test_feature],
            "perturbation_type": "gaussian",
            "scale": PERTURBATION_LEVELS["realistic_noise"],
        }

        gen1.add_perturbations(**perturbation_params)
        gen2.add_perturbations(**perturbation_params)

        # Results should be identical if implementation is deterministic
        np.testing.assert_array_equal(
            gen1.data[test_feature].values,
            gen2.data[test_feature].values,
            err_msg="Perturbations should be reproducible with same seed",
        )
