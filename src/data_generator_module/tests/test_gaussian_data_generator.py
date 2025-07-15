"""
Tests for GaussianDataGenerator class - Updated for feature-based classification only.
"""

import pytest
import numpy as np
import pandas as pd
from data_generator_module.gaussian_data_generator import GaussianDataGenerator


class TestGaussianDataGeneratorInit:
    def test_valid_initialisation(self):
        """Test valid initialisation parameters"""
        generator = GaussianDataGenerator(n_samples=100, n_features=5, random_state=42)

        assert generator.n_samples == 100
        assert generator.n_features == 5
        assert generator.random_state == 42
        assert generator.data is None
        assert generator.feature_types == {}
        assert generator.feature_parameters == {}

    def test_invalid_n_samples(self):
        """Test validation catches invalid n_samples"""
        with pytest.raises(ValueError, match="Number of samples must be positive"):
            GaussianDataGenerator(n_samples=0, n_features=5, random_state=42)

    def test_invalid_n_features(self):
        """Test validation catches invalid n_features"""
        with pytest.raises(ValueError, match="Number of features must be positive"):
            GaussianDataGenerator(n_samples=100, n_features=0, random_state=42)


class TestFeatureGeneration:
    def test_generate_features_basic(
        self, generator_factory, sample_feature_params, sample_feature_types
    ):
        """Test basic feature generation"""
        gen = generator_factory(n_samples=100, n_features=len(sample_feature_params))
        gen.generate_features(sample_feature_params, sample_feature_types)

        assert gen.data is not None
        assert gen.data.shape == (100, len(sample_feature_params))
        assert list(gen.data.columns) == list(sample_feature_params.keys())

        # Test feature types
        for feature_name, expected_type in sample_feature_types.items():
            if expected_type == "discrete":
                assert all(float(val).is_integer() for val in gen.data[feature_name])
            else:  # continuous
                assert any(
                    not float(val).is_integer() for val in gen.data[feature_name]
                )

    def test_feature_statistical_properties(
        self, generator_with_sample_data, sample_feature_params
    ):
        """Test that generated features match expected statistical properties"""
        for feature_name, params in sample_feature_params.items():
            feature_data = generator_with_sample_data.data[feature_name]

            # Allow for statistical variation in small samples
            mean_tolerance = 3 * params["std"] / np.sqrt(len(feature_data))
            std_tolerance = params["std"] / np.sqrt(2 * len(feature_data))

            assert abs(feature_data.mean() - params["mean"]) < mean_tolerance
            assert abs(feature_data.std() - params["std"]) < std_tolerance


class TestFeatureBasedSignalNoiseClassification:
    def test_fixed_signal_ratio(self, generator_with_sample_data):
        """Test that signal ratio is always 0.5"""
        signal_params = {"mean": 2.0, "std": 0.8}
        noise_params = {"mean": -1.0, "std": 1.2}

        generator_with_sample_data.create_feature_based_signal_noise_classification(
            signal_distribution_params=signal_params,
            noise_distribution_params=noise_params,
        )

        # Signal ratio should always be 0.5
        actual_ratio = generator_with_sample_data.data["target"].mean()
        assert abs(actual_ratio - 0.5) < 0.05  # Allow small statistical variation

        # Verify metadata
        assert generator_with_sample_data.feature_based_metadata["signal_ratio"] == 0.5

    def test_signal_noise_labels(self, generator_with_sample_data):
        """Test that labels are binary (0 and 1)"""
        signal_params = {"mean": 2.0, "std": 0.8}
        noise_params = {"mean": -1.0, "std": 1.2}

        generator_with_sample_data.create_feature_based_signal_noise_classification(
            signal_distribution_params=signal_params,
            noise_distribution_params=noise_params,
        )

        targets = generator_with_sample_data.data["target"]
        assert set(targets.unique()) == {0, 1}
        assert targets.dtype == int

    def test_store_for_visualization(self, generator_with_sample_data):
        """Test temporary observation storage for visualization"""
        signal_params = {"mean": 2.0, "std": 0.8}
        noise_params = {"mean": -1.0, "std": 1.2}

        generator_with_sample_data.create_feature_based_signal_noise_classification(
            signal_distribution_params=signal_params,
            noise_distribution_params=noise_params,
            store_for_visualization=True,
        )

        assert "_temp_observations" in generator_with_sample_data.data.columns
        assert (
            generator_with_sample_data.feature_based_metadata["has_temp_observations"]
            == True
        )

    def test_no_observation_feature_by_default(self, generator_with_sample_data):
        """Test that no observation feature is stored by default"""
        signal_params = {"mean": 2.0, "std": 0.8}
        noise_params = {"mean": -1.0, "std": 1.2}

        generator_with_sample_data.create_feature_based_signal_noise_classification(
            signal_distribution_params=signal_params,
            noise_distribution_params=noise_params,
        )

        assert "_temp_observations" not in generator_with_sample_data.data.columns
        assert (
            generator_with_sample_data.feature_based_metadata["has_temp_observations"]
            == False
        )

    def test_reproducibility(
        self, reproducible_generators, sample_feature_params, sample_feature_types
    ):
        """Test reproducibility with same seeds"""
        gen1, gen2 = reproducible_generators

        gen1.generate_features(sample_feature_params, sample_feature_types)
        gen2.generate_features(sample_feature_params, sample_feature_types)

        signal_params = {"mean": 2.0, "std": 0.8}
        noise_params = {"mean": -1.0, "std": 1.2}

        gen1.create_feature_based_signal_noise_classification(
            signal_distribution_params=signal_params,
            noise_distribution_params=noise_params,
        )
        gen2.create_feature_based_signal_noise_classification(
            signal_distribution_params=signal_params,
            noise_distribution_params=noise_params,
        )

        pd.testing.assert_frame_equal(gen1.data, gen2.data)


class TestVisualization:
    def test_signal_noise_visualization_with_custom_titles(
        self, generator_with_sample_data
    ):
        """Test signal vs noise visualization with custom titles"""
        signal_params = {"mean": 2.0, "std": 0.8}
        noise_params = {"mean": -1.0, "std": 1.2}

        generator_with_sample_data.create_feature_based_signal_noise_classification(
            signal_distribution_params=signal_params,
            noise_distribution_params=noise_params,
        )

        # Test that method accepts custom titles without error
        generator_with_sample_data.visualise_signal_noise_by_features(
            title="Custom Main Title", subtitle="Custom Subtitle"
        )

        # Method should return self for chaining
        result = generator_with_sample_data.visualise_signal_noise_by_features()
        assert result == generator_with_sample_data

    def test_visualization_error_without_target(self, generator_with_sample_data):
        """Test that visualization raises error without target data"""
        with pytest.raises(ValueError, match="No signal/noise data generated"):
            generator_with_sample_data.visualise_signal_noise_by_features()


class TestDataManagement:
    def test_save_data(self, generator_with_sample_data, tmp_path):
        """Test data saving functionality"""
        signal_params = {"mean": 2.0, "std": 0.8}
        noise_params = {"mean": -1.0, "std": 1.2}

        generator_with_sample_data.create_feature_based_signal_noise_classification(
            signal_distribution_params=signal_params,
            noise_distribution_params=noise_params,
        )

        save_path = tmp_path / "test_data.csv"
        generator_with_sample_data.save_data(str(save_path))

        assert save_path.exists()

        # Verify saved data
        saved_data = pd.read_csv(save_path)
        pd.testing.assert_frame_equal(saved_data, generator_with_sample_data.data)

    def test_get_feature_information(self, generator_with_sample_data):
        """Test feature information retrieval"""
        signal_params = {"mean": 2.0, "std": 0.8}
        noise_params = {"mean": -1.0, "std": 1.2}

        generator_with_sample_data.create_feature_based_signal_noise_classification(
            signal_distribution_params=signal_params,
            noise_distribution_params=noise_params,
        )

        info = generator_with_sample_data.get_feature_information()

        assert "target" in info
        assert info["target"]["mean"] == 0.5  # With 50/50 split

        for feature_name in generator_with_sample_data.feature_parameters.keys():
            assert feature_name in info
            assert "mean" in info[feature_name]
            assert "std" in info[feature_name]
