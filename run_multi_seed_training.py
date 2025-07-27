import argparse
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm

try:
    from src.data_generator_module.utils import find_project_root, load_yaml_config, create_filename_from_config
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src.data_generator_module.utils import find_project_root, load_yaml_config, create_filename_from_config

def main():
    """
    Automates training a model with a single optimal hyperparameter configuration
    across multiple datasets that differ only by their random seed.
    """
    parser = argparse.ArgumentParser(
        description="Run final training on multiple datasets with different seeds."
    )
    parser.add_argument(
        "--data-config-base",
        "-dcb",
        type=str,
        required=True,
        help="Path to one of the data generation config files from the desired dataset family (e.g., one with a specific seed)."
    )
    parser.add_argument(
        "--optimal-config",
        "-oc",
        type=str,
        required=True,
        help="Path to the single '_optimal.yml' file containing the best hyperparameters."
    )
    args = parser.parse_args()

    try:
        project_root = Path(find_project_root())
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # --- 1. Identify the Dataset Family ---
    base_data_config_path = project_root / args.data_config_base
    optimal_config_path = project_root / args.optimal_config

    if not base_data_config_path.exists():
        print(f"Error: Base data config not found at '{base_data_config_path}'")
        return
    if not optimal_config_path.exists():
        print(f"Error: Optimal training config not found at '{optimal_config_path}'")
        return

    # Derive the "family name" of the datasets
    base_name_with_seed = base_data_config_path.stem
    if "_config" in base_name_with_seed:
        base_name_with_seed = base_name_with_seed.replace('_config', '')
    
    dataset_family_name = base_name_with_seed.split('_seed')[0]
    print(f"Identified dataset family: '{dataset_family_name}'")

    # --- 2. Find all relevant data configs for that family ---
    data_config_dir = project_root / "configs" / "data_generation"
    all_data_configs = sorted(
        list(data_config_dir.glob(f"{dataset_family_name}_seed*_config.yml"))
    )

    if not all_data_configs:
        print(f"Error: No data configuration files found for the family '{dataset_family_name}' in '{data_config_dir}'")
        return

    print(f"\nFound {len(all_data_configs)} datasets to train on:")
    for path in all_data_configs:
        print(f" - {path.name}")
    print("-" * 60)

    # --- 3. Loop through and run final training for each ---
    for data_config in all_data_configs:
        seed_info = data_config.stem.split('_seed')[-1].split('_')[0]
        print(f"\n--- Starting training for seed: {seed_info} ---")

        # Construct the command to call the existing final training script
        command = [
            "uv",
            "run",
            "run_training.py",
            "--data-config",
            str(data_config),
            "--training-config",
            str(optimal_config_path)
        ]

        # Run the training script as a subprocess
        process = subprocess.run(
            command,
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False
        )
        
        # Print the output from the subprocess in real-time
        print(process.stdout)
        if process.returncode != 0:
            print(f"--- ERROR: Training failed for seed {seed_info} ---")
            print(process.stderr)
            print("-" * 50)
        else:
            print(f"--- Successfully completed training for seed: {seed_info} ---")

    print("\n" + "=" * 60)
    print("Multi-seed final training complete.")
    print("\n--- Next Steps ---")
    print("1. Aggregate the results from all seeds into a summary:")
    print(f"   uv run run_results_aggregator.py -oc {args.optimal_config}")
    print("\n2. (Optional) Update the main registry with all the individual model files:")
    print("   uv run run_experiment_tracking.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
