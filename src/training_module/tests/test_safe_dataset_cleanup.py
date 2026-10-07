"""Disk safety tests: all real writes/deletions are restricted to tmp_path."""

from pathlib import Path
import json

import pandas as pd
import pytest
import torch
import yaml

import experiment_manager as manager
from src.data_generator_module import generator_cli
from src.training_module import training_cli
from src.utils import dataset_lifecycle as lifecycle


METRICS = {
    "Test Loss (BCE)": 0.5,
    "Accuracy": 0.8,
    "F1-Score": 0.8,
    "Precision": 0.8,
    "Recall": 0.8,
    "AUC": 0.9,
}


def config(seed=0):
    return {
        "dataset_settings": {"n_samples": 20, "n_initial_features": 1},
        "global_settings": {"random_seed": seed},
        "output_settings": {"data_dir": "custom-data"},
        "create_feature_based_signal_noise_classification": {
            "feature_types": {"feature_0": "continuous"},
            "signal_features": {"feature_0": {"mean": 2.0, "std": 1.0}},
            "noise_features": {"feature_0": {"mean": 0.0, "std": 1.0}},
        },
    }


def write_yaml(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(content), encoding="utf-8")
    return path


@pytest.fixture
def roots(tmp_path, monkeypatch):
    monkeypatch.setattr(manager, "find_project_root", lambda: str(tmp_path))
    monkeypatch.setattr(generator_cli, "find_project_root", lambda: str(tmp_path))
    monkeypatch.setattr(generator_cli.utils, "find_project_root", lambda: str(tmp_path))
    monkeypatch.setattr(training_cli.torch.cuda, "is_available", lambda: False)
    return tmp_path


def owned_file(root, name="example_seed0_dataset.csv"):
    return lifecycle.save_new_dataset(
        pd.DataFrame({"feature_0": [0.2], "target": [0]}),
        root / name,
        root / "example_config.yml",
    )


def receipt(item, metrics_path):
    metrics_path.write_text(json.dumps(METRICS), encoding="utf-8")
    return lifecycle.EvaluatedDataset(
        item.csv_path, item.stamp, metrics_path, lifecycle.file_stamp(metrics_path)
    )


def test_cleanup_requires_both_creation_and_fresh_evaluation(tmp_path):
    created = owned_file(tmp_path)
    training = owned_file(tmp_path, "example_training_dataset.csv")
    cached = owned_file(tmp_path, "cached_seed1_dataset.csv")
    unrelated = tmp_path / "unrelated.csv"
    unrelated.write_text("keep", encoding="utf-8")
    confirmed = receipt(created, tmp_path / "fresh.json")
    # A receipt alone must never authorize deleting an unowned CSV.
    unowned = lifecycle.EvaluatedDataset(
        unrelated, lifecycle.file_stamp(unrelated),
        confirmed.metrics_path, confirmed.metrics_stamp,
    )
    removed = lifecycle.cleanup_evaluated_datasets(
        [created, training, cached],
        {created.csv_path: confirmed, unrelated: unowned},
    )
    assert removed == [created.csv_path]
    assert training.csv_path.exists() and cached.csv_path.exists()
    assert unrelated.read_text(encoding="utf-8") == "keep"
    assert confirmed.metrics_path.exists()


@pytest.mark.parametrize("changed", ["dataset", "metrics", "invalid_metrics"])
def test_cleanup_fails_closed_before_any_deletion(tmp_path, changed):
    first = owned_file(tmp_path, "first_seed0_dataset.csv")
    second = owned_file(tmp_path, "second_seed0_dataset.csv")
    one = receipt(first, tmp_path / "first.json")
    two = receipt(second, tmp_path / "second.json")
    if changed == "dataset":
        second.csv_path.write_text("changed", encoding="utf-8")
    else:
        two.metrics_path.write_text("{}", encoding="utf-8")
        if changed == "invalid_metrics":
            two = lifecycle.EvaluatedDataset(
                second.csv_path, second.stamp, two.metrics_path,
                lifecycle.file_stamp(two.metrics_path),
            )
    with pytest.raises((RuntimeError, ValueError)):
        lifecycle.cleanup_evaluated_datasets(
            [first, second], {first.csv_path: one, second.csv_path: two}
        )
    assert first.csv_path.exists() and second.csv_path.exists()


def test_generation_preserves_existing_csv_and_different_config(roots):
    path = write_yaml(roots / "input.yml", config())
    item = generator_cli.generate_from_config(str(path), keep_original_name=True)
    original = item.csv_path.read_bytes()
    with pytest.raises(FileExistsError):
        generator_cli.generate_from_config(str(path), keep_original_name=True)
    assert item.csv_path.read_bytes() == original
    with pytest.raises(FileExistsError):
        lifecycle.write_config_without_overwrite(path, config(1))
    assert yaml.safe_load(path.read_text()) == config()


def test_perturbation_requires_all_inputs_and_never_overwrites(roots):
    base = write_yaml(roots / "base.yml", config())
    generated = generator_cli.generate_multi_seed(str(base), num_seeds=2)
    perturb = write_yaml(roots / "shift.yml", {
        "perturbation_settings": [{
            "feature": "feature_0", "class_label": 0, "sigma_shift": 1.0,
        }]
    })
    paths = lifecycle.evaluation_config_paths(generated)
    outputs = generator_cli.perturb_multi_seed(str(base), str(perturb), config_paths=paths)
    assert len(outputs) == 2
    assert all(item.csv_path.parent == roots / "custom-data" for item in outputs)
    original = outputs[0].csv_path.read_bytes()
    with pytest.raises(FileExistsError):
        generator_cli.perturb_multi_seed(str(base), str(perturb), config_paths=paths)
    assert outputs[0].csv_path.read_bytes() == original
    generated[1].csv_path.unlink()  # Test-owned temporary input only.
    with pytest.raises(FileNotFoundError):
        generator_cli.perturb_multi_seed(str(base), str(perturb), config_paths=[paths[1]])


def test_real_evaluation_issues_receipts_but_cached_metrics_do_not(roots):
    base = write_yaml(roots / "base.yml", config())
    item = generator_cli.generate_from_config(str(base), keep_original_name=True)
    # A real one-input model, using an explicitly saved temporary checkpoint.
    model = training_cli.get_model(
        "mlp_001", {"input_size": 1, "hidden_size": 2, "output_size": 1}
    )
    checkpoint = roots / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    optimal = write_yaml(roots / "optimal.yml", {"training_settings": {
        "model_name": "mlp_001", "model_output_dir": "models",
        "target_column": "target",
        "hyperparameters": {"hidden_size": 2, "output_size": 1, "batch_size": 4},
    }})
    result = training_cli.evaluate_multi_seed(
        str(checkpoint), str(base), str(optimal), config_paths=[base]
    )
    assert set(result) == {item.csv_path}
    metrics_path = result[item.csv_path].metrics_path
    before = metrics_path.read_bytes()
    cached = training_cli.evaluate_multi_seed(
        str(checkpoint), str(base), str(optimal), config_paths=[base]
    )
    assert cached == {}
    assert lifecycle.cleanup_evaluated_datasets([item], cached) == []
    assert metrics_path.read_bytes() == before and item.csv_path.exists()
    with pytest.raises(FileExistsError):
        training_cli.evaluate_single_config(str(checkpoint), str(base), str(optimal))
    assert metrics_path.read_bytes() == before


@pytest.mark.parametrize("mode", ["default", "cleanup", "failure"])
@pytest.mark.parametrize("workflow", ["study", "full"])
def test_workflow_cleanup_is_opt_in_and_never_runs_after_evaluation_failure(
    roots, monkeypatch, mode, workflow
):
    base = write_yaml(roots / "base.yml", config())
    perturb = write_yaml(roots / "shift.yml", {"perturbation_settings": [{
        "feature": "feature_0", "class_label": 0, "sigma_shift": 1.0,
    }]})
    write_yaml(roots / "configs/experiments.yml", {"tuning_jobs": {"tiny": {
        "base_training_config": "configs/training/mlp_001.yml",
        "sample_fraction": 0.1,
    }}})
    # Keep real generation, evaluation discovery and cleanup, but no tuning.
    real_generate = generator_cli.generate_multi_seed
    monkeypatch.setattr(manager, "generate_multi_seed", lambda base_config_path:
        real_generate(base_config_path, num_seeds=2))
    monkeypatch.setattr(manager, "run_experiments", lambda **kw: None)
    monkeypatch.setattr(manager, "run_tuning_analysis", lambda **kw: None)
    monkeypatch.setattr(manager, "aggregate_all_families", lambda **kw: None)
    unrelated = roots / "data/unrelated.csv"
    unrelated.parent.mkdir()
    unrelated.write_text("must survive", encoding="utf-8")
    evaluation_calls = []

    def evaluate(**kwargs):
        paths = kwargs["config_paths"]
        evaluation_calls.append(paths)
        if mode == "failure" and len(evaluation_calls) == 2:
            raise RuntimeError("evaluation failed")
        result = {}
        for path in paths:
            contents = yaml.safe_load(Path(path).read_text())
            name = generator_cli.create_filename_from_config(contents)
            csv = lifecycle.dataset_path(roots, contents, name)
            item = lifecycle.GeneratedDataset(Path(path), csv, lifecycle.file_stamp(csv))
            result[csv] = receipt(item, roots / f"{name}.json")
        return result

    monkeypatch.setattr(manager, "evaluate_multi_seed", evaluate)
    arguments = dict(base_data_config=str(base), tuning_job="tiny")
    if workflow == "study":
        run = manager.run_perturbation_study
        arguments["perturb_configs"] = [str(perturb)]
    else:
        run = manager.run_full_pipeline
        arguments["perturb_config"] = str(perturb)
    if mode != "default":
        arguments["cleanup_generated"] = True
    if mode == "failure":
        with pytest.raises(RuntimeError, match="evaluation failed"):
            run(**arguments)
    else:
        run(**arguments)
    remaining = list((roots / "custom-data").glob("*.csv"))
    assert len(remaining) == (1 if mode == "cleanup" else 5)
    assert unrelated.read_text() == "must survive"
    assert any(p.name.endswith("_training_dataset.csv") for p in remaining)


def test_full_pipeline_has_no_startup_deletion_and_batch_stops_on_failure(roots, monkeypatch):
    unrelated = roots / "data/unrelated.csv"
    unrelated.parent.mkdir()
    unrelated.write_text("keep", encoding="utf-8")

    def stop(**kwargs):
        raise RuntimeError("generation stopped")

    monkeypatch.setattr(manager, "generate_multi_seed", stop)
    with pytest.raises(RuntimeError, match="generation stopped"):
        manager.run_full_pipeline("not_read.yml", "tiny")
    assert unrelated.exists()
    calls = []

    def fail(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("batch failed")

    monkeypatch.setattr(manager, "run_full_pipeline", fail)
    with pytest.raises(RuntimeError, match="batch failed"):
        manager.run_pipeline_batch(["first", "second"], "tiny", cleanup_generated=True)
    assert len(calls) == 1 and calls[0]["cleanup_generated"] is True


def test_legacy_cleanup_entry_points_refuse_deletion(roots):
    unrelated = roots / "data/unrelated.csv"
    unrelated.parent.mkdir()
    unrelated.write_text("keep", encoding="utf-8")
    with pytest.raises(RuntimeError):
        manager.clean_data_directory()
    with pytest.raises(RuntimeError):
        manager.clean_specific_family_data("unrelated")
    assert unrelated.exists()
