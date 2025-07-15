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
            std_tolerance = params["std"] / np.sqrt(len(feature_data))

            assert abs(feature_data.mean() - params["mean"]) < mean_tolerance
            assert abs(feature_data.std() - params["std"]) < std_tolerance


class TestFeatureBasedSignalNoiseClassification:
    def test_fixed_signal_ratio(self, basic_generator, signal_noise_config):
        """Test that signal ratio is always 0.5"""
        basic_generator.create_feature_based_signal_noise_classification(
            **signal_noise_config
        )

        actual_ratio = basic_generator.data["target"].mean()
        assert abs(actual_ratio - 0.5) < 0.05
        assert basic_generator.feature_based_metadata["signal_ratio"] == 0.5

    def test_signal_noise_labels(self, basic_generator, signal_noise_config):
        """Test that labels are binary (0 and 1)"""
        basic_generator.create_feature_based_signal_noise_classification(
            **signal_noise_config
        )

        targets = basic_generator.data["target"]
        assert set(targets.unique()) == {0, 1}
        assert targets.dtype == int

    def test_store_for_visualisation(self, basic_generator, signal_noise_config):
        """Test temporary observation storage for visualisation"""
        config = signal_noise_config.copy()
        config["store_for_visualisation"] = True

        basic_generator.create_feature_based_signal_noise_classification(**config)

        assert "_temp_observations" in basic_generator.data.columns
        assert basic_generator.feature_based_metadata["has_temp_observations"] == True

    def test_no_observation_feature_by_default(
        self, basic_generator, signal_noise_config
    ):
        """Test that no observation feature is stored by default"""
        basic_generator.create_feature_based_signal_noise_classification(
            **signal_noise_config
        )

        assert "_temp_observations" not in basic_generator.data.columns
        assert basic_generator.feature_based_metadata["has_temp_observations"] == False

    def test_reproducibility(self, reproducible_generators, signal_noise_config):
        """Test reproducibility with same seeds"""
        gen1, gen2 = reproducible_generators

        gen1.create_feature_based_signal_noise_classification(**signal_noise_config)
        gen2.create_feature_based_signal_noise_classification(**signal_noise_config)

        pd.testing.assert_frame_equal(gen1.data, gen2.data)


class Testvisualisation:
    def test_signal_noise_visualisation_with_custom_titles(
        self, basic_generator, signal_noise_config
    ):
        """Test signal vs noise visualisation with custom titles"""
        basic_generator.create_feature_based_signal_noise_classification(
            **signal_noise_config
        )

        # Test that method accepts custom titles without error
        basic_generator.visualise_signal_noise_by_features(
            title="Custom Main Title", subtitle="Custom Subtitle"
        )

        # Method should return self for chaining
        result = basic_generator.visualise_signal_noise_by_features()
        assert result == basic_generator


class TestDataManagement:
    def test_save_data(self, basic_generator, signal_noise_config, tmp_path):
        """Test data saving functionality"""
        basic_generator.create_feature_based_signal_noise_classification(
            **signal_noise_config
        )

        save_path = tmp_path / "test_data.csv"
        basic_generator.save_data(str(save_path))

        assert save_path.exists()

        # Verify saved data
        saved_data = pd.read_csv(save_path)
        pd.testing.assert_frame_equal(saved_data, basic_generator.data)

    def test_get_feature_information(self, basic_generator, signal_noise_config):
        """Test feature information retrieval"""
        basic_generator.create_feature_based_signal_noise_classification(
            **signal_noise_config
        )

        info = basic_generator.get_feature_information()

        assert "target" in info
        assert info["target"]["mean"] == 0.5  # With 50/50 split

        for feature_name in signal_noise_config["signal_features"].keys():
            assert feature_name in info
            assert "mean" in info[feature_name]
            assert "std" in info[feature_name]
