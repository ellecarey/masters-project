from pathlib import Path
import re

def parse_optimal_config_name(opt_config_path):
    """
    Parse the optimal config filename to extract dataset base, model name, seed (int), or perturbation_tag (or None).
    """
    basename = Path(opt_config_path).stem
    m = re.match(
        r"(?P<base>.+?)(?:_(?P<pert>pert_[^_]+))?(?:_seed(?P<seed>\d+)|_training)_(?P<model>[\w]+)_optimal$",
        basename
    )

    if not m:
        raise ValueError(f"Could not parse config filename: {basename}")

    dataset_base = m.group("base")
    perturbation_tag = m.group("pert")
    seed_str = m.group("seed")
    seed = int(seed_str) if seed_str else None
    model_name = m.group("model")

    return dataset_base, model_name, seed, perturbation_tag


def experiment_name(
    dataset_base_name: str,
    model_name: str,
    seed: int = None,
    perturbation_tag: str = None,
    optimized: bool = True
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
    print(parse_optimal_config_name("n1000_f_init5_cont0_disc5_sep5p1_seed0_mlp_001_optimal"))
    print(parse_optimal_config_name("n1000_f_init5_cont0_disc5_sep5p1_pert_f4n_by1p0s_seed1_mlp_001_optimal"))
