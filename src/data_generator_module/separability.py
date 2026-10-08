import copy


def _rss_from_marginals(signal_features: dict, noise_features: dict) -> float:
    """Root-sum-square of per-feature standardized mean gaps."""
    gaps = []
    for feature_name, signal_params in signal_features.items():
        if feature_name not in noise_features:
            continue
        noise_params = noise_features[feature_name]
        signal_mean = signal_params.get("mean", 0)
        noise_mean = noise_params.get("mean", 0)
        signal_std = signal_params.get("std", 1)
        noise_std = noise_params.get("std", 1)
        denominator = (signal_std**2 + noise_std**2) ** 0.5
        if denominator <= 0:
            continue
        gaps.append(abs(signal_mean - noise_mean) / denominator)
    if not gaps:
        return 0.0
    return (sum(gap**2 for gap in gaps)) ** 0.5


def _apply_marginal_update(params: dict, perturbation: dict) -> None:
    """Update one feature's mean and std the way the generator updates that column."""
    sigma_shift = perturbation.get("sigma_shift")
    scale_factor = perturbation.get("scale_factor")
    if (
        perturbation.get("additive_noise") is not None
        or perturbation.get("multiplicative_factor") is not None
    ):
        raise NotImplementedError(
            "additive_noise and multiplicative_factor are out of scope."
        )
    if sigma_shift is not None and scale_factor is not None:
        raise ValueError(
            "Cannot apply both sigma_shift and scale_factor simultaneously."
        )
    if sigma_shift is not None:
        std = params.get("std", 1)
        params["mean"] = params.get("mean", 0) + sigma_shift * std
    elif scale_factor is not None:
        params["mean"] = params.get("mean", 0) * scale_factor
        params["std"] = params.get("std", 1) * scale_factor


def _marginals_after_perturbation(config: dict) -> tuple[dict, dict]:
    """Return signal and noise marginals after each perturbation_settings entry."""
    class_config = config.get("create_feature_based_signal_noise_classification", {})
    signal_features = copy.deepcopy(class_config.get("signal_features", {}))
    noise_features = copy.deepcopy(class_config.get("noise_features", {}))
    for perturbation in config.get("perturbation_settings") or []:
        target = (
            signal_features if perturbation.get("class_label") == 1 else noise_features
        )
        if perturbation.get("type", "individual") == "correlated":
            feature_names = perturbation.get("features", [])
        else:
            feature_name = perturbation.get("feature")
            feature_names = [feature_name] if feature_name else []
        for feature_name in feature_names:
            if feature_name in target:
                _apply_marginal_update(target[feature_name], perturbation)
    return signal_features, noise_features


def calculate_separability(config: dict) -> float:
    """Root-sum-square of per-feature standardized mean gaps for a data config."""
    signal_features, noise_features = _marginals_after_perturbation(config)
    return _rss_from_marginals(signal_features, noise_features)
