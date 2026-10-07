import pytest
import yaml

from src.training_module import training_cli


@pytest.mark.parametrize(
    "missing_input",
    ["data_config", "training_config", "dataset"],
)
def test_single_evaluation_raises_for_missing_input(
    tmp_path, monkeypatch, mocker, missing_input
):
    monkeypatch.setattr(
        training_cli.data_utils,
        "find_project_root",
        lambda: str(tmp_path),
    )
    monkeypatch.setattr(training_cli.torch.cuda, "is_available", lambda: False)

    data_config_path = tmp_path / "example_seed42_config.yml"
    training_config_path = tmp_path / "model_config.yml"

    data_config = {
        "dataset_settings": {
            "n_samples": 4,
            "n_initial_features": 1,
        },
        "global_settings": {"random_seed": 42},
        "create_feature_based_signal_noise_classification": {
            "feature_types": {"feature_0": "continuous"},
        },
    }
    training_config = {
        "training_settings": {
            "model_name": "mlp_001",
            "model_output_dir": "models",
            "target_column": "target",
            "hyperparameters": {},
        }
    }

    if missing_input != "data_config":
        data_config_path.write_text(yaml.safe_dump(data_config), encoding="utf-8")

    if missing_input != "training_config":
        training_config_path.write_text(
            yaml.safe_dump(training_config), encoding="utf-8"
        )

    # No CSV is created: the dataset case must fail at data loading.
    get_model = mocker.patch.object(training_cli, "get_model")
    load_model = mocker.patch.object(training_cli.torch, "load")

    with pytest.raises(FileNotFoundError):
        training_cli.evaluate_single_config(
            str(tmp_path / "unused_model.pt"),
            str(data_config_path),
            str(training_config_path),
        )

    get_model.assert_not_called()
    load_model.assert_not_called()
    assert not list((tmp_path / "models").glob("*.json"))


@pytest.mark.parametrize(
    "missing_input",
    ["optimal_config", "evaluation_family"],
)
def test_multiseed_evaluation_raises_for_missing_input(
    tmp_path, monkeypatch, mocker, missing_input
):
    monkeypatch.setattr(
        training_cli.data_utils,
        "find_project_root",
        lambda: str(tmp_path),
    )

    data_config_dir = tmp_path / "configs" / "data_generation"
    data_config_dir.mkdir(parents=True)

    # The supplied base config exists, but the searched family directory
    # deliberately contains no evaluation configurations.
    base_config_path = tmp_path / "example_seed0_config.yml"
    base_config_path.write_text(
        yaml.safe_dump({"global_settings": {"random_seed": 0}}),
        encoding="utf-8",
    )

    optimal_config_path = tmp_path / "optimal.yml"
    if missing_input != "optimal_config":
        optimal_config_path.write_text(
            yaml.safe_dump(
                {
                    "training_settings": {
                        "model_name": "mlp_001",
                        "model_output_dir": "models",
                    }
                }
            ),
            encoding="utf-8",
        )

    evaluate_single = mocker.patch.object(training_cli, "evaluate_single_config")

    with pytest.raises(FileNotFoundError):
        training_cli.evaluate_multi_seed(
            trained_model_path=str(tmp_path / "unused_model.pt"),
            data_config_base=str(base_config_path),
            optimal_config=str(optimal_config_path),
        )

    evaluate_single.assert_not_called()
