import pytest

from data_generator_module.utils import create_filename_from_config
from src.utils.filenames import parse_experiment_name, parse_optimal_config_name

RECIPES = [
    pytest.param(
        {"feature_0": {"mean": 0.0, "std": 2.0}},
        {"feature_0": {"mean": 0.0, "std": 2.0}},
        None,
        id="unperturbed",
    ),
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
]


def _config(signal, noise, perturbation_settings, seed):
    config = {
        "dataset_settings": {
            "n_samples": 8000,
            "n_initial_features": len(signal),
        },
        "create_feature_based_signal_noise_classification": {
            "signal_features": signal,
            "noise_features": noise,
            "feature_types": {name: "continuous" for name in signal},
        },
        "global_settings": {"random_seed": seed},
    }
    if perturbation_settings is not None:
        config["perturbation_settings"] = perturbation_settings
    return config


def _tail(parsed):
    if parsed.seed is None:
        return "_training"
    return f"_seed{parsed.seed}"


@pytest.mark.parametrize("signal, noise, perturbation_settings", RECIPES)
@pytest.mark.parametrize("seed", [0, 99])
def test_parser_splits_names_from_create_filename(
    signal, noise, perturbation_settings, seed
):
    name = create_filename_from_config(
        _config(signal, noise, perturbation_settings, seed)
    )
    parsed = parse_experiment_name(name)

    if perturbation_settings:
        assert parsed.perturbation_tag is not None
        assert parsed.perturbation_tag.count("pert_") == len(perturbation_settings)
        assert f"{parsed.base}_{parsed.perturbation_tag}{_tail(parsed)}" == name
    else:
        assert parsed.perturbation_tag is None
        assert f"{parsed.base}{_tail(parsed)}" == name

    if seed == 99:
        assert parsed.seed is None
    else:
        assert parsed.seed == seed
    assert parsed.model_name is None

    optimal = parse_experiment_name(f"{name}_mlp_001_optimal")
    assert optimal.base == parsed.base
    assert optimal.perturbation_tag == parsed.perturbation_tag
    assert optimal.seed == parsed.seed
    assert optimal.model_name == "mlp_001"

    base, model_name, parsed_seed, tag = parse_optimal_config_name(
        f"{name}_mlp_001_optimal.yml"
    )
    assert (base, model_name, parsed_seed, tag) == (
        parsed.base,
        "mlp_001",
        parsed.seed,
        parsed.perturbation_tag,
    )
