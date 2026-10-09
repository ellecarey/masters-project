from pathlib import Path
from typing import NamedTuple
import re


class ParsedExperimentName(NamedTuple):
    base: str
    perturbation_tag: str | None
    seed: int | None
    model_name: str | None


_EXPERIMENT_NAME = re.compile(
    r"^(?P<base>n\d+_f_init\d+_cont\d+_disc\d+_sep\d+p\d+)"
    r"(?:_(?P<pert>pert_.+?))?"
    r"_(?:seed(?P<seed>\d+)|training)"
    r"(?:_(?P<model>.+)_optimal)?$"
)


def parse_experiment_name(name: str) -> ParsedExperimentName:
    """Split a dataset stem or an optimal-config stem into base, pert tag, and role."""
    match = _EXPERIMENT_NAME.match(Path(name).stem)
    if not match:
        raise ValueError(f"Could not parse experiment name: {name}")
    seed_str = match.group("seed")
    return ParsedExperimentName(
        base=match.group("base"),
        perturbation_tag=match.group("pert"),
        seed=int(seed_str) if seed_str else None,
        model_name=match.group("model"),
    )


def parse_optimal_config_name(opt_config_path):
    """
    Parse the optimal config filename to extract dataset base, model name, seed (int), or perturbation_tag (or None).
    """
    parsed = parse_experiment_name(opt_config_path)
    if parsed.model_name is None:
        raise ValueError(f"Could not parse config filename: {opt_config_path}")
    return parsed.base, parsed.model_name, parsed.seed, parsed.perturbation_tag


def experiment_name(
    dataset_base_name: str,
    model_name: str,
    seed: int = None,
    perturbation_tag: str = None,
    optimized: bool = True,
) -> str:
    name = dataset_base_name
    if perturbation_tag:
        name += f"_{perturbation_tag}"
    if seed is not None:
        name += f"_seed{seed}"
    name += f"_{model_name}"
    if optimized:
        name += "_optimal"
    return name


def metrics_filename(*args, **kwargs):
    return experiment_name(*args, **kwargs) + "_metrics.json"


def model_filename(*args, **kwargs):
    return experiment_name(*args, **kwargs) + "_model.pt"


def config_filename(*args, **kwargs):
    return experiment_name(*args, **kwargs) + ".yml"


if __name__ == "__main__":
    print(
        parse_optimal_config_name(
            "n1000_f_init5_cont0_disc5_sep5p1_seed0_mlp_001_optimal"
        )
    )
    print(
        parse_optimal_config_name(
            "n1000_f_init5_cont0_disc5_sep5p1_pert_f4n_by1p0s_seed1_mlp_001_optimal"
        )
    )
