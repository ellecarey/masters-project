import os
import argparse
from src.data_generator_module import GaussianDataGenerator, utils
from src.training_module.utils import set_global_seed
from src.data_generator_module.utils import (
    create_filename_from_config,
    create_plot_title_from_config,
    rename_config_file,
)


def main():
    """
    Runs the data generation pipeline using settings from a YAML file.
    This script generates a dataset and associated plots with names
    derived from the experiment's configuration.
    """
    parser = argparse.ArgumentParser(description="Run the data generation pipeline.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.yml",
        help="Path to the configuration YAML file (default: config.yml)",
    )
    parser.add_argument(
        "--keep-original-name",
        action="store_true",
        help="Keep original config filename (don't rename)",
    )
    args = parser.parse_args()

    # Load configuration and set up reproducibility
    try:
        config_path = args.config
        config = utils.load_yaml_config(config_path)
        print(f"Successfully loaded configuration from {config_path}")
    except FileNotFoundError:
        print(
            f"Error: {config_path} not found. Please ensure it is in the project root."
        )
        return
    except Exception as e:
        print(f"Error loading or parsing {config_path}: {e}")
        return

    # Set the global random seed for the entire pipeline
    global_seed = config["global_settings"]["random_seed"]
    set_global_seed(global_seed)

    # Generate a unique name for this experiment from the config
    experiment_name = create_filename_from_config(config)
    print(f"Generated experiment name: {experiment_name}")

    # Initialise the data generator
    dataset_settings = config["dataset_settings"]
    generator = GaussianDataGenerator(
        n_samples=dataset_settings["n_samples"],
        n_features=dataset_settings["n_initial_features"],
        random_state=global_seed,
    )

    # Execute the data generation pipeline steps from the config
    print("\nStarting data generation pipeline...")
    generator.generate_features(**config["feature_generation"])

    if "add_features" in config:
        generator.add_features(**config["add_features"])

    if "perturbation" in config:
        generator.add_perturbations(**config["perturbation"])

    if "create_target" in config:
        generator.create_target_variable(**config["create_target"])

    print("Pipeline finished generating data.")

    # Save the generated dataset with the dynamic filename
    # Use output_settings from data generation config, or fallback to default
    output_settings = config.get("output_settings", {"data_dir": "data/"})
    output_data_dir = output_settings.get("data_dir", "data/")
    dataset_filepath = os.path.join(output_data_dir, f"{experiment_name}_dataset.csv")
    generator.save_data(file_path=dataset_filepath)

    # Visualise features and save the plot with the dynamic filename
    if "visualisation" in config:
        vis_config = config["visualisation"]

        main_title, subtitle = create_plot_title_from_config(config)

        # Get the output directory from the config
        plot_dir = vis_config.get("save_to_dir")

        if plot_dir:
            # Construct the full, unique path for the plot
            plot_filepath = os.path.join(plot_dir, f"{experiment_name}_plot.pdf")

            # Get features for visualisation - handle missing 'features' key
            if "features" in vis_config:
                features_to_show = vis_config["features"]
            else:
                # Auto-generate features list from feature_generation section
                feature_generation = config.get("feature_generation", {})
                all_features = list(
                    feature_generation.get("feature_parameters", {}).keys()
                )
                max_features = vis_config.get("max_features_to_show", 6)
                features_to_show = all_features[:max_features]

            # visualise features
            generator.visualise_features(
                features=features_to_show,
                max_features_to_show=vis_config["max_features_to_show"],
                n_bins=vis_config["n_bins"],
                save_to_path=plot_filepath,
                title=main_title,
                subtitle=subtitle,
            )

    # Rename configuration file to match generated files (unless disabled)
    if not args.keep_original_name:
        renamed_config_path = rename_config_file(config_path, experiment_name)
        print(f"Configuration file available at: {renamed_config_path}")
    else:
        print(f"Configuration file kept at original location: {config_path}")

    print("\nData generation pipeline finished successfully.")


if __name__ == "__main__":
    main()
