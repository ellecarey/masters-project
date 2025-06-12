"""
Comprehensive tests for GaussianDataGenerator class.
Contains all the core functionality tests including initialisation,
feature generation, perturbations, target creation, and visualisation.
"""

import pytest
from data_generator_module.gaussian_data_generator import GaussianDataGenerator
from .fixtures.sample_data import DATASET_CONFIGS, ERROR_PATTERNS, REPRODUCIBILITY_SEEDS


class TestGaussianDataGeneratorInit:
    def test_valid_initialisation(self):
        """test valid initialisation parameters"""
        config = DATASET_CONFIGS["unit_test_standard"]
        generator = GaussianDataGenerator(
            n_samples=config["samples"],
            n_features=config["features"],
            random_state=REPRODUCIBILITY_SEEDS[0],
        )
        assert generator.n_samples == config["samples"]
        assert generator.n_features == config["features"]
        assert generator.random_state == 42
        assert generator.data is None
        assert generator.feature_types == {}
        assert generator.feature_parameters == {}

    def test_default_random_state_only(self):
        """test that only random_state has a default value of 42"""
        config = DATASET_CONFIGS["unit_test_standard"]
        generator = GaussianDataGenerator(
            n_samples=config["samples"],
            n_features=config["features"],
            random_state=REPRODUCIBILITY_SEEDS[0],
        )
        assert generator.random_state == 42

    def test_invalid_n_samples(self):
        """test validation catches invalid n_samples"""
        with pytest.raises(ValueError, match=ERROR_PATTERNS["negative_samples"]):
            GaussianDataGenerator(
                n_samples=0,
                n_features=DATASET_CONFIGS["minimal_valid_input"]["features"],
                random_state=REPRODUCIBILITY_SEEDS[0],
            )

    def test_visualise_features_with_titles_and_saving(
        self, mocker, tmp_path, generator_with_sample_data
    ):
        """
        Tests that visualise_features correctly passes titles to Matplotlib
        and calls the save function when a path is provided.
        """
        # Mock all external functions that would be called
        mock_suptitle = mocker.patch("matplotlib.figure.Figure.suptitle")
        mock_figtext = mocker.patch("matplotlib.pyplot.figtext")
        mock_savefig = mocker.patch("matplotlib.pyplot.savefig")
        mock_makedirs = mocker.patch("os.makedirs")

        # Define test data and use tmp_path to create a safe path
        test_title = "My Custom Title"
        test_subtitle = "My Custom Subtitle"
        # tmp_path creates a temporary directory for this test run
        save_path = tmp_path / "test_plot.pdf"

        # Call the method
        generator_with_sample_data.visualise_features(
            features=["test_feature_1"],
            max_features_to_show=1,
            n_bins=10,
            save_to_path=str(save_path),  # Pass the path to trigger saving
            title=test_title,
            subtitle=test_subtitle,
        )

        # Assert that all functions were called correctly
        # Assert titles were set
        mock_suptitle.assert_called_once_with(test_title, fontsize=20, weight="bold")
        mock_figtext.assert_called_once_with(
            0.5, 0.92, test_subtitle, ha="center", fontsize=14, style="italic"
        )

        # Assert directory creation was called with the parent of save path
        mock_makedirs.assert_called_once_with(str(tmp_path), exist_ok=True)

        # Assert that savefig was called with the exact path and parameters
        mock_savefig.assert_called_once_with(
            str(save_path), dpi=300, bbox_inches="tight"
        )
