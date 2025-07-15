"""
Tests for utility functions - Updated for simplified naming.
"""

import pytest
from data_generator_module.utils import (
    create_filename_from_config,
    create_plot_title_from_config,
)


class TestFilenameGeneration:
    def test_create_filename_basic(self):
        """Test basic filename generation"""
        config = {
            "dataset_settings": {"n_samples": 100000, "n_initial_features": 5},
            "feature_generation": {
                "feature_types": {
                    "feature_0": "discrete",
                    "feature_1": "discrete",
                    "feature_2": "discrete",
                    "feature_3": "discrete",
                    "feature_4": "discrete",
                }
            },
        }

        expected = "n100000_f_init5_cont0_disc5"
        result = create_filename_from_config(config)
        assert result == expected

    def test_create_filename_mixed_types(self):
        """Test filename generation with mixed feature types"""
        config = {
            "dataset_settings": {"n_samples": 50000, "n_initial_features": 4},
            "feature_generation": {
                "feature_types": {
                    "feature_0": "continuous",
                    "feature_1": "continuous",
                    "feature_2": "discrete",
                    "feature_3": "discrete",
                }
            },
        }

        expected = "n50000_f_init4_cont2_disc2"
        result = create_filename_from_config(config)
        assert result == expected

    def test_create_filename_missing_keys(self):
        """Test filename generation with missing keys uses defaults"""
        config = {"dataset_settings": {"n_samples": 1000}}

        expected = "n1000_f_init5_cont0_disc0"
        result = create_filename_from_config(config)
        assert result == expected


class TestPlotTitleGeneration:
    def test_create_plot_title_basic(self):
        """Test basic plot title generation"""
        config = {
            "dataset_settings": {"n_samples": 100000, "n_initial_features": 5},
            "create_feature_based_signal_noise_classification": {
                "signal_distribution_params": {"mean": 2.0, "std": 0.8},
                "noise_distribution_params": {"mean": -1.0, "std": 1.2},
            },
        }

        title, subtitle = create_plot_title_from_config(config)

        assert title == "Distribution of Generated Features"
        assert "100,000 Samples" in subtitle
        assert "5 Features" in subtitle
        assert "Feature-based Classification" in subtitle

    def test_create_plot_title_fallback(self):
        """Test plot title generation fallback"""
        config = {}  # Empty config

        title, subtitle = create_plot_title_from_config(config)

        assert title == "Feature Distribution"
        assert subtitle == "Configuration details unavailable"
