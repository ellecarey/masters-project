import pytest
import yaml

import experiment_manager as manager


def test_missing_evaluation_config_stops_before_cleanup(tmp_path, monkeypatch, mocker):
    monkeypatch.setattr(manager, "find_project_root", lambda: str(tmp_path))

    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    base_config_path = tmp_path / "base.yml"
    base_config_path.write_text(
        yaml.safe_dump({"global_settings": {"random_seed": 0}}),
        encoding="utf-8",
    )

    perturb_config_path = tmp_path / "perturbation.yml"
    perturb_config_path.write_text(
        yaml.safe_dump({"perturbation_settings": []}),
        encoding="utf-8",
    )

    experiments = {
        "tuning_jobs": {
            "test_job": {
                "base_training_config": "configs/training/mlp_001.yml",
                "sample_fraction": 0.1,
            }
        }
    }
    (config_dir / "experiments.yml").write_text(
        yaml.safe_dump(experiments),
        encoding="utf-8",
    )

    # Predictable names; no evaluation config is created.
    mocker.patch.object(
        manager,
        "create_filename_from_config",
        side_effect=["example_training", "example_pert_seed0"],
    )

    # Prevent all real cleanup, generation, tuning and evaluation.
    mocker.patch.object(manager, "clean_data_directory")
    cleanup = mocker.patch.object(manager, "clean_specific_family_data")
    mocker.patch.object(manager, "generate_multi_seed")
    mocker.patch.object(manager, "perturb_multi_seed")
    mocker.patch.object(manager, "run_experiments")
    mocker.patch.object(manager, "run_tuning_analysis")
    evaluate = mocker.patch.object(manager, "evaluate_multi_seed")
    aggregate = mocker.patch.object(manager, "aggregate_all_families")

    with pytest.raises(FileNotFoundError, match="Evaluation config not found"):
        manager.run_perturbation_study(
            base_data_config=str(base_config_path),
            tuning_job="test_job",
            perturb_configs=[str(perturb_config_path)],
        )

    # Only baseline evaluation should have been attempted.
    evaluate.assert_called_once()
    cleanup.assert_not_called()
    aggregate.assert_not_called()
