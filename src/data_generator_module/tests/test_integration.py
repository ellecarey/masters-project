"""
Integration tests for complete workflow
"""

import subprocess
import yaml
import sys


def test_full_pipeline_with_dynamic_naming(tmp_path):
    """
    Tests the full pipeline from data generation to model training,
    verifying dynamic file creation and usage with an isolated config.
    """
    # Define a temporary config and save it in a temporary directory
    # This test uses temporary paths for all outputs.
    plots_dir = tmp_path / "plots"
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"

    config_data = {
        "global_settings": {"random_seed": 42},
        "dataset_settings": {"n_samples": 50, "n_initial_features": 3},
        "feature_generation": {
            "feature_parameters": {
                "f1": {"mean": 0, "std": 1},
                "f2": {"mean": 1, "std": 1},
                "f3": {"mean": 2, "std": 1},
            },
            "feature_types": {
                "f1": "continuous",
                "f2": "continuous",
                "f3": "continuous",
            },
        },
        "create_target": {
            "features_to_use": ["f1", "f2"],
            "weights": [0.5, -0.5],
            "noise_level": 0.1,
            "function_type": "linear",
        },
        "visualisation": {
            "features": ["f1", "f2"],
            "max_features_to_show": 2,
            "n_bins": 10,
            "save_to_dir": str(plots_dir),
        },
        "training_settings": {
            "output_data_dir": str(data_dir),
            "model_output_dir": str(models_dir),
            "target_column": "target",
            "validation_set_ratio": 0.5,
            "test_set_ratio": 0.25,
            "hyperparameters": {
                "epochs": 1,
                "hidden_size": 4,
                "output_size": 1,
                "learning_rate": 0.01,
                "batch_size": 16,
            },
        },
    }

    config_path = tmp_path / "test_config.yml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    # Run the data generation script using the temporary config
    subprocess.run(
        [
            sys.executable,
            "run_generator.py",
            "--config",
            str(config_path),
        ],  # Use sys.executable
        check=True,
        capture_output=True,
        text=True,
    )

    # Assert that the output files were created with the correct dynamic name
    expected_name_base = "n50_f_init3_add0_pert-none_scl0_func-linear"

    dataset_file = data_dir / f"{expected_name_base}_dataset.csv"
    plot_file = plots_dir / f"{expected_name_base}_plot.pdf"

    assert dataset_file.exists(), "Dataset file was not created with the dynamic name."
    assert plot_file.exists(), "Plot file was not created with the dynamic name."

    # Run the training script
    subprocess.run(
        [
            sys.executable,
            "run_training.py",
            "--config",
            str(config_path),
        ],  # Use sys.executable
        check=True,
        capture_output=True,
        text=True,
    )

    # Assert that the model was saved with the correct dynamic name
    model_file = models_dir / f"{expected_name_base}_model.pt"
    assert model_file.exists(), "Model file was not saved with the dynamic name."
