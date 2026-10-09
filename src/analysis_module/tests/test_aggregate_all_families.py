from src.analysis_module import analysis_cli


def test_comparison_uses_the_training_optimal_config(tmp_path, monkeypatch):
    project = tmp_path / "project"
    models = project / "models"
    models.mkdir(parents=True)
    optimal = (
        project
        / "configs"
        / "training"
        / "generated"
        / "n8000_f_init1_cont1_disc0_sep0p0_training_mlp_001_optimal.yml"
    )
    optimal.parent.mkdir(parents=True)
    optimal.write_text("training_settings: {}\n", encoding="utf-8")

    prefix = "n8000_f_init1_cont1_disc0_sep0p0"
    (models / f"{prefix}_seed0_mlp_001_metrics.json").write_text("{}", encoding="utf-8")
    (models / f"{prefix}_pert_f0n_by1p0s_seed0_mlp_001_metrics.json").write_text(
        "{}", encoding="utf-8"
    )

    monkeypatch.setattr(analysis_cli, "find_project_root", lambda: str(project))
    monkeypatch.setattr(analysis_cli, "aggregate", lambda **kwargs: None)
    monkeypatch.setattr(analysis_cli, "generate_global_tracking_sheet", lambda: None)
    calls = []

    def record_compare(original_optimal_config, perturbation_tag):
        calls.append((original_optimal_config, perturbation_tag))

    monkeypatch.setattr(analysis_cli, "compare_families", record_compare)

    analysis_cli.aggregate_all_families(str(optimal))

    assert calls == [(str(optimal), "_pert_f0n_by1p0s")]
