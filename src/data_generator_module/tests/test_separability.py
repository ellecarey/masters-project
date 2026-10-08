import copy

import pytest

from data_generator_module.gaussian_data_generator import GaussianDataGenerator
from data_generator_module.separability import calculate_separability


def _config(signal_features, noise_features, perturbation_settings=None):
    config = {
        "create_feature_based_signal_noise_classification": {
            "signal_features": signal_features,
            "noise_features": noise_features,
        }
    }
    if perturbation_settings is not None:
        config["perturbation_settings"] = perturbation_settings
    return config


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


class TestSeparabilityScore:
    def test_equal_means_score_zero(self):
        config = _config(
            {"feature_0": {"mean": 0, "std": 2}},
            {"feature_0": {"mean": 0, "std": 2}},
        )
        assert calculate_separability(config) == 0.0

    def test_single_feature_gap(self):
        config = _config(
            {"feature_0": {"mean": 3, "std": 0}},
            {"feature_0": {"mean": 0, "std": 4}},
        )
        assert calculate_separability(config) == pytest.approx(0.75)

    def test_root_sum_square_of_two_unit_gaps(self):
        config = _config(
            {
                "feature_0": {"mean": 1, "std": 1},
                "feature_1": {"mean": 1, "std": 1},
            },
            {
                "feature_0": {"mean": 0, "std": 0},
                "feature_1": {"mean": 0, "std": 0},
            },
        )
        assert calculate_separability(config) == pytest.approx(2**0.5)

    def test_feature_on_one_class_is_ignored(self):
        config = _config(
            {
                "feature_0": {"mean": 3, "std": 0},
                "feature_1": {"mean": 100, "std": 1},
            },
            {"feature_0": {"mean": 0, "std": 4}},
        )
        assert calculate_separability(config) == pytest.approx(0.75)

    def test_both_standard_deviations_zero(self):
        config = _config(
            {"feature_0": {"mean": 5, "std": 0}},
            {"feature_0": {"mean": 0, "std": 0}},
        )
        assert calculate_separability(config) == 0.0

    def test_empty_config(self):
        assert calculate_separability({}) == 0.0


class TestPerturbationMarginals:
    def test_noise_sigma_shift(self):
        config = _config(
            {"feature_0": {"mean": 0, "std": 2}},
            {"feature_0": {"mean": 0, "std": 2}},
            [{"feature": "feature_0", "class_label": 0, "sigma_shift": 1}],
        )
        assert calculate_separability(config) == pytest.approx(2 / (8**0.5))

    def test_negative_scale_flips_the_mean(self):
        config = _config(
            {"feature_0": {"mean": 2, "std": 2}},
            {"feature_0": {"mean": 2, "std": 2}},
            [{"feature": "feature_0", "class_label": 0, "scale_factor": -1.5}],
        )
        assert calculate_separability(config) == pytest.approx(5 / (13**0.5))

    def test_later_shift_uses_scaled_standard_deviation(self):
        config = _config(
            {"feature_0": {"mean": 0, "std": 2}},
            {"feature_0": {"mean": 1, "std": 2}},
            [
                {"feature": "feature_0", "class_label": 0, "scale_factor": 2},
                {"feature": "feature_0", "class_label": 0, "sigma_shift": 1},
            ],
        )
        assert calculate_separability(config) == pytest.approx(6 / (20**0.5))

    def test_correlated_shift_updates_every_listed_mean(self):
        signal = {
            "feature_1": {"mean": 0, "std": 1},
            "feature_2": {"mean": 0, "std": 2},
            "feature_3": {"mean": 0, "std": 4},
        }
        noise = copy.deepcopy(signal)
        config = _config(
            signal,
            noise,
            [
                {
                    "type": "correlated",
                    "class_label": 0,
                    "features": ["feature_1", "feature_2", "feature_3"],
                    "correlation_matrix": [
                        [1.0, 0.7, 0.5],
                        [0.7, 1.0, 0.6],
                        [0.5, 0.6, 1.0],
                    ],
                    "sigma_shift": 1.2,
                }
            ],
        )
        assert calculate_separability(config) == pytest.approx(1.2 * (1.5**0.5))

    def test_config_is_not_mutated(self):
        config = _config(
            {"feature_0": {"mean": 0, "std": 2}},
            {"feature_0": {"mean": 0, "std": 2}},
            [{"feature": "feature_0", "class_label": 0, "sigma_shift": 1}],
        )
        original = copy.deepcopy(config)
        calculate_separability(config)
        assert config == original

    def test_shift_and_scale_on_one_entry_raises(self):
        config = _config(
            {"feature_0": {"mean": 0, "std": 1}},
            {"feature_0": {"mean": 0, "std": 1}},
            [
                {
                    "feature": "feature_0",
                    "class_label": 0,
                    "sigma_shift": 1,
                    "scale_factor": 2,
                }
            ],
        )
        with pytest.raises(ValueError, match="sigma_shift and scale_factor"):
            calculate_separability(config)

    def test_additive_noise_raises(self):
        config = _config(
            {"feature_0": {"mean": 0, "std": 1}},
            {"feature_0": {"mean": 0, "std": 1}},
            [{"feature": "feature_0", "class_label": 0, "additive_noise": 0.5}],
        )
        with pytest.raises(NotImplementedError):
            calculate_separability(config)


class TestAgainstGenerator:
    def _generate(self, signal_features, noise_features):
        feature_types = {name: "continuous" for name in signal_features}
        generator = GaussianDataGenerator(
            n_samples=100_000,
            n_features=len(signal_features),
            random_state=0,
        )
        generator.create_feature_based_signal_noise_classification(
            signal_features=signal_features,
            noise_features=noise_features,
            feature_types=feature_types,
        )
        return generator

    def test_individual_sigma_shift_matches_generated_sample(self):
        signal = {"feature_0": {"mean": 1.0, "std": 2.0}}
        noise = {"feature_0": {"mean": 0.0, "std": 2.0}}
        generator = self._generate(signal, noise)
        generator.perturb_feature("feature_0", class_label=0, sigma_shift=1.0)
        config = _config(
            signal,
            noise,
            [{"feature": "feature_0", "class_label": 0, "sigma_shift": 1.0}],
        )
        assert calculate_separability(config) == pytest.approx(
            _empirical_separability(generator.data),
            abs=0.03,
        )

    def test_individual_scale_matches_generated_sample(self):
        signal = {"feature_0": {"mean": 2.0, "std": 2.0}}
        noise = {"feature_0": {"mean": 1.0, "std": 1.5}}
        generator = self._generate(signal, noise)
        generator.perturb_feature("feature_0", class_label=0, scale_factor=-1.5)
        config = _config(
            signal,
            noise,
            [{"feature": "feature_0", "class_label": 0, "scale_factor": -1.5}],
        )
        assert calculate_separability(config) == pytest.approx(
            _empirical_separability(generator.data),
            abs=0.03,
        )
