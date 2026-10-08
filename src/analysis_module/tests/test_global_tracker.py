import pandas as pd
import pytest
import yaml

from data_generator_module.gaussian_data_generator import GaussianDataGenerator
from data_generator_module.utils import create_filename_from_config
from src.analysis_module import global_tracker

MEASURED_AUC = 0.1234


def _empirical_separability(frame):
    signal = frame[frame["target"] == 1]
    noise = frame[frame["target"] == 0]
    gaps = []
    for feature_name in frame.columns.drop("target"):
        signal_std = signal[feature_name].std(ddof=0)
        noise_std = noise[feature_name].std(ddof=0)
        denominator = (signal_std**2 + noise_std**2) ** 0.5
        if denominator <= 0:
            continue
        gaps.append(
            abs(signal[feature_name].mean() - noise[feature_name].mean()) / denominator
        )
    if not gaps:
        return 0.0
    return (sum(gap**2 for gap in gaps)) ** 0.5


def _classification(signal_features, noise_features):
    return {
        "signal_features": signal_features,
        "noise_features": noise_features,
        "feature_types": {name: "continuous" for name in signal_features},
    }


def _write_config(
    path, signal_features, noise_features, perturbation_settings=None, n_samples=100_000
):
    config = {
        "dataset_settings": {
            "n_samples": n_samples,
            "n_initial_features": len(signal_features),
        },
        "create_feature_based_signal_noise_classification": _classification(
            signal_features, noise_features
        ),
    }
    if perturbation_settings is not None:
        config["perturbation_settings"] = perturbation_settings
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def _write_summary(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"mean": [MEASURED_AUC], "std": [0.01]},
        index=pd.Index(["AUC"], name="metric"),
    ).to_csv(path)


def _tracking_row(tmp_path, monkeypatch, family_name):
    project_root = tmp_path / "project"
    monkeypatch.setattr(global_tracker, "find_project_root", lambda: str(project_root))
    global_tracker.generate_global_tracking_sheet()
    sheet = pd.read_csv(
        project_root / "reports" / "global_experiment_tracking.csv",
        keep_default_na=False,
    )
    return sheet.loc[sheet["experiment_family"] == family_name].iloc[0]


def _family(signal, noise, perturbation_settings, n_samples):
    config = {
        "dataset_settings": {
            "n_samples": n_samples,
            "n_initial_features": len(signal),
        },
        "create_feature_based_signal_noise_classification": _classification(
            signal, noise
        ),
        "global_settings": {"random_seed": 0},
    }
    if perturbation_settings is not None:
        config["perturbation_settings"] = perturbation_settings
    return create_filename_from_config(config).rsplit("_seed", 1)[0]


def _generated_scores(signal, noise, perturbation_settings, n_samples):
    generator = GaussianDataGenerator(
        n_samples=n_samples, n_features=len(signal), random_state=0
    )
    generator.create_feature_based_signal_noise_classification(
        signal_features=signal,
        noise_features=noise,
        feature_types={name: "continuous" for name in signal},
    )
    baseline = _empirical_separability(generator.data)
    for entry in perturbation_settings:
        if entry.get("type", "individual") == "correlated":
            for feature_name in entry["features"]:
                generator.perturb_feature(
                    feature_name,
                    entry["class_label"],
                    sigma_shift=entry.get("sigma_shift"),
                    scale_factor=entry.get("scale_factor"),
                )
        else:
            generator.apply_perturbation_from_config(entry)
    return baseline, _empirical_separability(generator.data)


def test_perturbed_separation_follows_generator_and_auc_stays_measured(
    tmp_path, monkeypatch
):
    signal = {"feature_0": {"mean": 2.0, "std": 2.0}}
    noise = {"feature_0": {"mean": 1.0, "std": 1.5}}
    perturbation_settings = [
        {"feature": "feature_0", "class_label": 0, "scale_factor": -1.5}
    ]
    original_family = "n100000_f_init1_cont1_disc0_sep0p4"
    family_name = f"{original_family}_pert_f0n_scale-1p5"

    project_root = tmp_path / "project"
    data_dir = project_root / "configs" / "data_generation"
    data_dir.mkdir(parents=True)
    (project_root / "configs" / "training" / "generated").mkdir(parents=True)
    _write_config(data_dir / f"{original_family}_seed0_config.yml", signal, noise)
    _write_config(
        data_dir / f"{family_name}_seed0_config.yml",
        signal,
        noise,
        perturbation_settings,
    )
    _write_summary(
        project_root
        / "reports"
        / "spreadsheets"
        / family_name
        / f"{family_name}_summary.csv"
    )

    generator = GaussianDataGenerator(n_samples=100_000, n_features=1, random_state=0)
    generator.create_feature_based_signal_noise_classification(
        signal_features=signal,
        noise_features=noise,
        feature_types={"feature_0": "continuous"},
    )
    baseline_score = _empirical_separability(generator.data)
    generator.perturb_feature("feature_0", class_label=0, scale_factor=-1.5)
    perturbed_score = _empirical_separability(generator.data)

    row = _tracking_row(tmp_path, monkeypatch, family_name)

    assert float(row["separation"]) == pytest.approx(baseline_score, abs=0.05)
    assert float(row["separation"]) + float(row["separation_delta"]) == pytest.approx(
        perturbed_score, abs=0.05
    )
    assert row["AUC_mean"] == pytest.approx(MEASURED_AUC)


def test_unperturbed_family_has_no_delta(tmp_path, monkeypatch):
    signal = {"feature_0": {"mean": 2.0, "std": 2.0}}
    noise = {"feature_0": {"mean": 1.0, "std": 1.5}}
    family_name = "n100000_f_init1_cont1_disc0_sep0p4"

    project_root = tmp_path / "project"
    data_dir = project_root / "configs" / "data_generation"
    data_dir.mkdir(parents=True)
    (project_root / "configs" / "training" / "generated").mkdir(parents=True)
    _write_config(data_dir / f"{family_name}_seed0_config.yml", signal, noise)
    _write_summary(
        project_root
        / "reports"
        / "spreadsheets"
        / family_name
        / f"{family_name}_summary.csv"
    )

    generator = GaussianDataGenerator(n_samples=100_000, n_features=1, random_state=0)
    generator.create_feature_based_signal_noise_classification(
        signal_features=signal,
        noise_features=noise,
        feature_types={"feature_0": "continuous"},
    )

    row = _tracking_row(tmp_path, monkeypatch, family_name)

    assert float(row["separation"]) == pytest.approx(
        _empirical_separability(generator.data), abs=0.05
    )
    assert row["separation_delta"] == "N/A"
    assert row["AUC_mean"] == pytest.approx(MEASURED_AUC)


def test_missing_original_config_leaves_delta_blank(tmp_path, monkeypatch):
    signal = {"feature_0": {"mean": 2.0, "std": 2.0}}
    noise = {"feature_0": {"mean": 1.0, "std": 1.5}}
    family_name = "n100000_f_init1_cont1_disc0_sep0p4_pert_f0n_scale-1p5"

    project_root = tmp_path / "project"
    data_dir = project_root / "configs" / "data_generation"
    data_dir.mkdir(parents=True)
    (project_root / "configs" / "training" / "generated").mkdir(parents=True)
    _write_config(
        data_dir / f"{family_name}_seed0_config.yml",
        signal,
        noise,
        [{"feature": "feature_0", "class_label": 0, "scale_factor": -1.5}],
    )
    _write_summary(
        project_root
        / "reports"
        / "spreadsheets"
        / family_name
        / f"{family_name}_summary.csv"
    )

    generator = GaussianDataGenerator(n_samples=100_000, n_features=1, random_state=0)
    generator.create_feature_based_signal_noise_classification(
        signal_features=signal,
        noise_features=noise,
        feature_types={"feature_0": "continuous"},
    )

    row = _tracking_row(tmp_path, monkeypatch, family_name)

    assert float(row["separation"]) == pytest.approx(
        _empirical_separability(generator.data), abs=0.05
    )
    assert row["separation_delta"] == "N/A"
    assert row["AUC_mean"] == pytest.approx(MEASURED_AUC)


@pytest.mark.parametrize(
    "signal, noise, perturbation_settings",
    [
        pytest.param(
            {"feature_0": {"mean": 0.0, "std": 2.0}},
            {"feature_0": {"mean": 0.0, "std": 2.0}},
            [{"feature": "feature_0", "class_label": 0, "sigma_shift": 1.0}],
            id="individual-shift",
        ),
        pytest.param(
            {
                "feature_0": {"mean": 0.0, "std": 1.0},
                "feature_1": {"mean": 0.0, "std": 1.0},
            },
            {
                "feature_0": {"mean": 2.0, "std": 1.0},
                "feature_1": {"mean": 2.0, "std": 1.0},
            },
            [
                {
                    "type": "correlated",
                    "class_label": 0,
                    "features": ["feature_0", "feature_1"],
                    "correlation_matrix": [[1.0, 0.5], [0.5, 1.0]],
                    "scale_factor": -1.5,
                }
            ],
            id="correlated-scale",
        ),
        pytest.param(
            {
                "feature_0": {"mean": 0.0, "std": 2.0},
                "feature_1": {"mean": 2.0, "std": 2.0},
            },
            {
                "feature_0": {"mean": 0.0, "std": 2.0},
                "feature_1": {"mean": 1.0, "std": 1.5},
            },
            [
                {"feature": "feature_0", "class_label": 0, "sigma_shift": 1.0},
                {"feature": "feature_1", "class_label": 0, "scale_factor": -1.5},
            ],
            id="combined",
        ),
    ],
)
def test_separation_matches_generated_frame_and_auc_stays_measured(
    tmp_path, monkeypatch, signal, noise, perturbation_settings
):
    n_samples = 8_000
    baseline_family = _family(signal, noise, None, n_samples)
    pert_suffix = _family(signal, noise, perturbation_settings, n_samples).split(
        "_pert_", 1
    )[1]
    family_name = f"{baseline_family}_pert_{pert_suffix}"

    project_root = tmp_path / "project"
    data_dir = project_root / "configs" / "data_generation"
    data_dir.mkdir(parents=True)
    (project_root / "configs" / "training" / "generated").mkdir(parents=True)
    _write_config(
        data_dir / f"{baseline_family}_seed0_config.yml",
        signal,
        noise,
        n_samples=n_samples,
    )
    _write_config(
        data_dir / f"{family_name}_seed0_config.yml",
        signal,
        noise,
        perturbation_settings,
        n_samples=n_samples,
    )
    _write_summary(
        project_root
        / "reports"
        / "spreadsheets"
        / family_name
        / f"{family_name}_summary.csv"
    )

    baseline_score, perturbed_score = _generated_scores(
        signal, noise, perturbation_settings, n_samples
    )
    row = _tracking_row(tmp_path, monkeypatch, family_name)

    assert float(row["separation"]) == pytest.approx(baseline_score, abs=0.05)
    assert float(row["separation"]) + float(row["separation_delta"]) == pytest.approx(
        perturbed_score, abs=0.05
    )
    assert row["AUC_mean"] == pytest.approx(MEASURED_AUC)
