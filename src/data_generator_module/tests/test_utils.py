from data_generator_module.utils import (
    create_filename_from_config,
    create_plot_title_from_config,
)


def test_create_filename_from_config(sample_config):
    """Tests that the filename is generated correctly from a standard config."""
    expected_name = "n1000_f_init5_add2_pert-gaussian_scl0p15_func-polynomial"
    filename = create_filename_from_config(sample_config)
    assert filename == expected_name


def test_create_plot_title_from_config(sample_config):
    """Tests that the plot title and subtitle are generated correctly."""
    expected_title = "Distribution of Generated Features"
    expected_subtitle = (
        "Dataset: 1000 Samples, 7 Features | "
        "Perturbation: Gaussian (Scale: 0.15) | "
        "Target: Polynomial Relationship"
    )

    title, subtitle = create_plot_title_from_config(sample_config)

    assert title == expected_title
    assert subtitle == expected_subtitle


def test_naming_handles_missing_keys():
    """Tests that the utility functions are robust to missing config keys."""
    minimal_config = {
        "dataset_settings": {"n_samples": 500, "n_initial_features": 3},
        "create_target": {"function_type": "linear"},
    }

    # Test filename fallback logic
    expected_filename = "n500_f_init3_add0_pert-none_scl0_func-linear"
    assert create_filename_from_config(minimal_config) == expected_filename

    # Test title fallback logic
    expected_subtitle = (
        "Dataset: 500 Samples, 3 Features | "
        "No Perturbations | "
        "Target: Linear Relationship"
    )
    _, subtitle = create_plot_title_from_config(minimal_config)
    assert subtitle == expected_subtitle
