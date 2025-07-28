import argparse
import yaml
from pathlib import Path
import pandas as pd
import sys

try:
    from src.data_generator_module.utils import find_project_root, create_filename_from_config
    from src.data_generator_module.gaussian_data_generator import GaussianDataGenerator
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src.data_generator_module.utils import find_project_root, create_filename_from_config
    from src.data_generator_module.gaussian_data_generator import GaussianDataGenerator

def main():
    parser = argparse.ArgumentParser(description="Apply perturbations to a family of existing datasets.")
    parser.add_argument("--data-config-base", "-dcb", type=str, required=True, help="Path to one of the data configs from the desired dataset family.")
    parser.add_argument("--perturb-config", "-pc", type=str, required=True, help="Path to the perturbation YAML config file.")
    args = parser.parse_args()

    project_root = Path(find_project_root())
    
    perturb_config_path = project_root / args.perturb_config
    with open(perturb_config_path, 'r') as f:
        perturb_config = yaml.safe_load(f)
    print(f"Loaded perturbation config: {perturb_config.get('description', 'N/A')}")

    base_data_config_path = project_root / args.data_config_base
    family_name = base_data_config_path.stem.split('_seed')[0]
    data_config_dir = project_root / "configs" / "data_generation"
    all_data_configs = sorted(list(data_config_dir.glob(f"{family_name}_seed*_config.yml")))
    
    print(f"\nFound {len(all_data_configs)} datasets in family '{family_name}' to perturb.")

    for data_config_path in all_data_configs:
        with open(data_config_path, 'r') as f:
            data_config = yaml.safe_load(f)

        dataset_base_name = create_filename_from_config(data_config)
        dataset_path = project_root / "data" / f"{dataset_base_name}_dataset.csv"
        
        if not dataset_path.exists():
            print(f" - Skipping: Original dataset not found at {dataset_path}")
            continue

        print(f"\n--- Processing: {dataset_path.name} ---")
        
        generator = GaussianDataGenerator(
            n_samples=data_config['dataset_settings']['n_samples'],
            n_features=data_config['dataset_settings']['n_initial_features'],
            random_state=data_config['global_settings']['random_seed']
        )
        generator.data = pd.read_csv(dataset_path)
        generator.feature_based_metadata = {
            'signal_features': data_config['create_feature_based_signal_noise_classification']['signal_features'],
            'noise_features': data_config['create_feature_based_signal_noise_classification']['noise_features'],
            'perturbations': []
        }

        for p_conf in perturb_config['perturbation_settings']:
            generator.perturb_feature(
                feature_name=p_conf['feature'],
                class_label=p_conf['class_label'],
                sigma_shift=p_conf['sigma_shift']
            )

        perturbed_config = data_config.copy()
        perturbed_config['perturbation_settings'] = perturb_config['perturbation_settings']
        new_filename_base = create_filename_from_config(perturbed_config)
        new_dataset_path = project_root / "data" / f"{new_filename_base}_dataset.csv"

        generator.save_data(str(new_dataset_path))
        
        new_config_path = data_config_dir / f"{new_filename_base}_config.yml"
        with open(new_config_path, 'w') as f:
            yaml.dump(perturbed_config, f, default_flow_style=False)
        print(f"Saved new config to: {new_config_path.name}")

    print("\n" + "="*60)
    print("Multi-seed perturbation complete.")
    print("Run `uv run run_update_dataset_tracking.py` to update your registry.")
    print("="*60)

if __name__ == "__main__":
    main()