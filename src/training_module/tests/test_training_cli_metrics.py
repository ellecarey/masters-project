import json

import pandas as pd
import pytest
import torch

from src.training_module import training_cli


class FixedScoreModel(torch.nn.Module):
    """Convert known input probabilities to logits."""

    def forward(self, features):
        return torch.logit(features[:, :1])


def test_evaluation_auc_uses_probabilities(tmp_path, monkeypatch, mocker):
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
            "hyperparameters": {
                "hidden_size": 2,
                "output_size": 1,
                "batch_size": 4,
            },
        }
    }
    data = pd.DataFrame(
        {
            "feature_0": [0.1, 0.2, 0.3, 0.4],
            "target": [0, 0, 1, 1],
        }
    )

    monkeypatch.setattr(
        training_cli.data_utils,
        "find_project_root",
        lambda: str(tmp_path),
    )
    monkeypatch.setattr(training_cli.torch.cuda, "is_available", lambda: False)

    mocker.patch.object(
        training_cli.data_utils,
        "load_yaml_config",
        side_effect=[data_config, training_config],
    )
    mocker.patch.object(training_cli.pd, "read_csv", return_value=data)
    mocker.patch.object(training_cli, "get_model", return_value=FixedScoreModel())
    mocker.patch.object(training_cli.torch, "load", return_value={})

    training_cli.evaluate_single_config(
        str(tmp_path / "unused_model.pt"),
        str(tmp_path / "example_seed42_config.yml"),
        str(tmp_path / "model_config.yml"),
    )

    metrics_files = list((tmp_path / "models").glob("*.json"))
    assert len(metrics_files) == 1

    metrics = json.loads(metrics_files[0].read_text(encoding="utf-8"))

    assert metrics["Accuracy"] == pytest.approx(0.5)
    assert metrics["AUC"] == pytest.approx(1.0)


def test_training_auc_uses_probabilities(tmp_path, monkeypatch, mocker):
    data_config = {
        "dataset_settings": {
            "n_samples": 40,
            "n_initial_features": 1,
        },
        "global_settings": {"random_seed": 99},
        "create_feature_based_signal_noise_classification": {
            "feature_types": {"feature_0": "continuous"},
        },
    }
    training_config = {
        "training_settings": {
            "model_name": "mlp_001",
            "model_output_dir": "models",
            "target_column": "target",
            "validation_set_ratio": 0.2,
            "test_set_ratio": 0.2,
            "hyperparameters": {
                "hidden_size": 2,
                "output_size": 1,
                "batch_size": 8,
                "learning_rate": 0.001,
                "epochs": 1,
            },
        }
    }
    data = pd.DataFrame(
        {
            "feature_0": [0.1, 0.2, 0.3, 0.4] * 10,
            "target": [0, 0, 1, 1] * 10,
        }
    )

    # The optimiser requires a parameter, although training is mocked.
    model = FixedScoreModel()
    model.register_parameter(
        "unused_parameter",
        torch.nn.Parameter(torch.zeros(1)),
    )

    monkeypatch.setattr(
        training_cli.data_utils,
        "find_project_root",
        lambda: str(tmp_path),
    )
    monkeypatch.setattr(training_cli.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(training_cli, "apply_custom_plot_style", lambda: None)

    mocker.patch.object(
        training_cli.data_utils,
        "load_yaml_config",
        side_effect=[data_config, training_config],
    )
    mocker.patch.object(training_cli.pd, "read_csv", return_value=data)
    mocker.patch.object(training_cli, "get_model", return_value=model)
    mocker.patch.object(
        training_cli,
        "train_model",
        return_value=(model, {}, 1),
    )
    mocker.patch.object(training_cli.train_utils, "plot_training_history")
    mocker.patch.object(training_cli.train_utils, "plot_final_metrics")

    training_cli.train_single_config(
        str(tmp_path / "example_training_config.yml"),
        str(tmp_path / "model_config.yml"),
    )

    metrics_files = list((tmp_path / "models").glob("*.json"))
    assert len(metrics_files) == 1

    metrics = json.loads(metrics_files[0].read_text(encoding="utf-8"))

    assert metrics["Accuracy"] == pytest.approx(0.5)
    assert metrics["AUC"] == pytest.approx(1.0)
