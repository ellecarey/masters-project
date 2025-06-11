import os
from src.generator_package import GaussianDataGenerator, utils, config as path_config


def main():
    """
    Runs the data generation pipeline using settings from a YAML file.
    """
    # 1. Load the generation recipe from the YAML configuration file
    try:
        config_path = "config.yml"
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

    # Initialise the generator using dataset-wide settings
    dataset_settings = config["dataset_settings"]
    generator = GaussianDataGenerator(
        n_samples=dataset_settings["n_samples"],
        n_features=dataset_settings["n_initial_features"],
        random_state=dataset_settings["random_state"],
    )

    # Execute the data generation pipeline by unpacking config dictionaries
    print("Starting data generation pipeline...")
    generator.generate_features(**config["feature_generation"])

    if "add_features" in config:
        generator.add_features(**config["add_features"])

    generator.add_perturbations(**config["perturbation"]).create_target_variable(
        **config["create_target"]
    )

    print("Pipeline finished generating data.")

    # Save the generated data using the new method
    generator.save_data(path_config.OUTPUT_DATA_PATH)

    # Visualise features and save the plot
    if "visualisation" in config:
        vis_config = config["visualisation"]
        # Set the save directory from path_config, ensuring it exists
        os.makedirs(path_config.FIGURES_OUTPUT_DIR, exist_ok=True)
        vis_config["save_to_dir"] = path_config.FIGURES_OUTPUT_DIR
        generator.visualise_features(**vis_config)


if __name__ == "__main__":
    main()
