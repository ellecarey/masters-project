import pytest
import yaml

from src.training_module import training_cli


class StopBeforeReadingCSV(Exception):
    """Stop the workflow once it reaches the CSV-loading step."""


@pytest.mark.parametrize("workflow", ["train", "evaluate"])
@pytest.mark.parametrize(
    "directory_case",
    ["default", "empty_settings", "relative", "absolute"],
)
def test_workflow_uses_configured_data_directory(
    tmp_path, monkeypatch, mocker, workflow, directory_case
):
    project_root = tmp_path / "project"
    project_root.mkdir()

    data_config = {
        "dataset_settings": {
            "n_samples": 1000,
            "n_initial_features": 2,
        },
        "global_settings": {
            "random_seed": 99 if workflow == "train" else 42,
        },
    }

    if directory_case == "default":
        expected_dir = project_root / "data"
    elif directory_case == "empty_settings":
        data_config["output_settings"] = {}
        expected_dir = project_root / "data"
    elif directory_case == "relative":
        data_config["output_settings"] = {"data_dir": "data/custom"}
        expected_dir = project_root / "data/custom"
    else:
        expected_dir = tmp_path / "external_data"
        data_config["output_settings"] = {
            "data_dir": str(expected_dir),
        }

    training_config = {
        "training_settings": {
            "model_name": "mlp_001",
            "model_output_dir": "models",
            "hyperparameters": {},
        }
    }

    config_name = (
        "example_training_config.yml"
        if workflow == "train"
        else "example_seed42_config.yml"
    )
    data_config_path = project_root / config_name
    training_config_path = project_root / "model_config.yml"

    data_config_path.write_text(yaml.safe_dump(data_config), encoding="utf-8")
    training_config_path.write_text(yaml.safe_dump(training_config), encoding="utf-8")

    monkeypatch.setattr(
        training_cli.data_utils,
        "find_project_root",
        lambda: str(project_root),
    )
    monkeypatch.setattr(training_cli, "apply_custom_plot_style", lambda: None)
    monkeypatch.setattr(training_cli.torch.cuda, "is_available", lambda: False)

    read_csv = mocker.patch.object(
        training_cli.pd,
        "read_csv",
        side_effect=StopBeforeReadingCSV,
    )

    with pytest.raises(StopBeforeReadingCSV):
        if workflow == "train":
            training_cli.train_single_config(
                str(data_config_path), str(training_config_path)
            )
        else:
            training_cli.evaluate_single_config(
                str(project_root / "unused_model.pt"),
                str(data_config_path),
                str(training_config_path),
            )

    filename_base = training_cli.data_utils.create_filename_from_config(data_config)
    expected_path = expected_dir / f"{filename_base}_dataset.csv"
    read_csv.assert_called_once_with(expected_path)
