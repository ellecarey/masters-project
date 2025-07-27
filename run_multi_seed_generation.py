# run_multi_seed_generation.py

import argparse
import yaml
import subprocess
import os
from pathlib import Path


from src.data_generator_module.utils import find_project_root, load_yaml_config, create_filename_from_config

def main():
    """
    Automates the generation of multiple datasets from a single base
    configuration.
    """
    parser = argparse.ArgumentParser(
        description="Generate multiple datasets with different random seeds."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the base data generation YAML config file.",
    )
    parser.add_argument(
        "--num-seeds",
        "-n",
        type=int,
        default=5,
        help="The number of different seeds to generate datasets for.",
    )
    parser.add_argument(
        "--start-seed",
        "-s",
        type=int,
        default=0,
        help="The starting random seed. Subsequent seeds will be incremented from this value.",
    )
    args = parser.parse_args()

    try:
        project_root = Path(find_project_root())
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    base_config_path = Path(args.config)
    if not base_config_path.is_absolute():
        base_config_path = project_root / base_config_path

    try:
        base_config = load_yaml_config(str(base_config_path))
        print(f"Loaded base configuration from: {base_config_path.name}")
    except FileNotFoundError:
        print(f"Error: Base configuration file not found at '{base_config_path}'")
        return

    print(f"\nWill generate {args.num_seeds} datasets using seeds from {args.start_seed} to {args.start_seed + args.num_seeds - 1}.")
    print("-" * 60)
    
    config_dir = project_root / "configs" / "data_generation"

    for i in range(args.num_seeds):
        current_seed = args.start_seed + i
        print(f"\n--- Generating dataset for seed: {current_seed} ---")

        # Create a new config dictionary with the updated seed
        new_config = base_config.copy()
        if "global_settings" not in new_config:
            new_config["global_settings"] = {}
        new_config["global_settings"]["random_seed"] = current_seed

        # --- THIS IS THE FIX ---
        # 1. Create a permanent filename for the new config
        new_config_base_name = create_filename_from_config(new_config)
        new_config_filename = f"{new_config_base_name}_config.yml"
        new_config_path = config_dir / new_config_filename
        
        # 2. Save the new permanent config file
        with open(new_config_path, 'w') as f:
            yaml.dump(new_config, f, default_flow_style=False)
        print(f"Saved new configuration to: {new_config_path.relative_to(project_root)}")

        # 3. Construct the command to call the existing data generator script
        command = [
            "uv",
            "run",
            "run_data_generator.py",
            "--config",
            str(new_config_path),
            "--keep-original-name" 
        ]

        # 4. Run the data generator script as a subprocess
        process = subprocess.run(
            command,
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False 
        )
        
        print(process.stdout)
        if process.returncode != 0:
            print(f"--- ERROR generating dataset for seed {current_seed} ---")
            print(process.stderr)

    print("\n" + "=" * 60)
    print("Multi-seed data generation complete.")
    print("Run `uv run run_update_dataset_tracking_spreadsheet.py` to update your dataset registry.")
    print("=" * 60)

if __name__ == "__main__":
    main()

