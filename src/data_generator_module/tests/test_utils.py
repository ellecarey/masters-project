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
        """Test discrete-feature naming with default separation and seed."""
        config = {
            "dataset_settings": {"n_samples": 100000, "n_initial_features": 5},
            "create_feature_based_signal_noise_classification": {
                "feature_types": {
                    "feature_0": "discrete",
                    "feature_1": "discrete",
                    "feature_2": "discrete",
                    "feature_3": "discrete",
                    "feature_4": "discrete",
                }
            },
        }

        expected = "n100000_f_init5_cont0_disc5_sep0p0_seed42"
        result = create_filename_from_config(config)
        assert result == expected

    def test_create_filename_mixed_types(self):
        """Test mixed-feature naming with default separation and seed."""
        config = {
            "dataset_settings": {"n_samples": 50000, "n_initial_features": 4},
            "create_feature_based_signal_noise_classification": {
                "feature_types": {
                    "feature_0": "continuous",
                    "feature_1": "continuous",
                    "feature_2": "discrete",
                    "feature_3": "discrete",
                }
            },
        }

        expected = "n50000_f_init4_cont2_disc2_sep0p0_seed42"
        result = create_filename_from_config(config)
        assert result == expected

    def test_create_filename_missing_keys(self):
        """Test filename defaults when optional naming details are absent."""
        config = {"dataset_settings": {"n_samples": 1000}}

        expected = "n1000_f_init0_cont0_disc0_sep0p0_seed42"
        result = create_filename_from_config(config)
        assert result == expected


class TestPlotTitleGeneration:
    def test_create_plot_title_basic(self):
        """Test that the plot title describes the configured dataset."""
        config = {
            "dataset_settings": {"n_samples": 100000, "n_initial_features": 5},
            "create_feature_based_signal_noise_classification": {
                "feature_types": {
                    "feature_0": "discrete",
                    "feature_1": "discrete",
                    "feature_2": "discrete",
                    "feature_3": "discrete",
                    "feature_4": "discrete",
                }
            },
        }

        title, subtitle = create_plot_title_from_config(config)

        assert title == "Distribution of Generated Features"
        assert "100,000 Samples" in subtitle
        assert "5 Features (0 Cont, 5 Disc)" in subtitle
        assert "No Perturbations" in subtitle
        assert "Std. Separation: 0.00" in subtitle

    def test_create_plot_title_fallback(self):
        """Test that an empty configuration produces a readable title."""
        title, subtitle = create_plot_title_from_config({})

        assert title == "Distribution of Generated Features"
        assert subtitle == (
            "Dataset: N/A Samples, 0 Features (0 Cont, 0 Disc)\n"
            "No Perturbations | Std. Separation: 0.00"
        )
