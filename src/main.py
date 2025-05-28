import random
import numpy as np
import torch
import os
from generator_package import config
from generator_package.gaussian_data_generator import GuassianDataGenerator
from generator_package.plotting_style import apply_custom_plot_style


def run_data_generation():
    """
    Runs the data generation process based on settings in config.py.
    """
    apply_custom_plot_style()

    # --- Initial Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Global random seeding for reproducibility across libraries
    random.seed(config.RANDOM_STATE)
    np.random.seed(config.RANDOM_STATE)
    torch.manual_seed(config.RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_STATE)

    print("--- Initializing GuassianDataGenerator ---")
    generator = GuassianDataGenerator(
        n_samples=config.N_SAMPLES,
        n_features=config.N_INITIAL_FEATURES,
        random_state=config.RANDOM_STATE,
    )

    # --- 1. Generate Initial Features ---
    print("\n--- Generating Initial Features ---")
    try:
        generator.generate_features(
            feature_parameters=config.FEATURE_PARAMETERS_OVERRIDE,
            feature_types=config.FEATURE_TYPES_OVERRIDE,
        )
        print("Initial data head:")
        if generator.get_data() is not None:
            print(generator.get_data().head())
    except Exception as e:
        print(f"Error generating initial features: {e}")
        return

    # --- 2. Add Perturbations (Optional) ---
    if hasattr(config, "PERTURBATION_SETTINGS") and config.PERTURBATION_SETTINGS:
        print("\n--- Adding Perturbations ---")
        try:
            # Validate required parameters exist
            required_keys = ["perturbation_type", "features_to_perturb", "scale"]
            if all(key in config.PERTURBATION_SETTINGS for key in required_keys):
                generator.add_perturbations(
                    perturbation_type=config.PERTURBATION_SETTINGS["perturbation_type"],
                    features=config.PERTURBATION_SETTINGS["features_to_perturb"],
                    scale=config.PERTURBATION_SETTINGS["scale"],
                )
                print("Data head after perturbations:")
                if generator.get_data() is not None:
                    print(generator.get_data().head())
            else:
                print("Skipping perturbations: Missing required parameters in config")
        except Exception as e:
            print(f"Error adding perturbations: {e}")

    # --- 3. Add More Features (Optional) ---
    if hasattr(config, "ADD_FEATURES_SETTINGS") and config.ADD_FEATURES_SETTINGS:
        print("\n--- Adding New Features ---")
        try:
            # Validate required parameters exist
            required_keys = ["n_new_features", "feature_parameters", "feature_types"]
            if all(key in config.ADD_FEATURES_SETTINGS for key in required_keys):
                generator.add_features(
                    n_new_features=config.ADD_FEATURES_SETTINGS["n_new_features"],
                    feature_parameters=config.ADD_FEATURES_SETTINGS[
                        "feature_parameters"
                    ],
                    feature_types=config.ADD_FEATURES_SETTINGS["feature_types"],
                )
                print("Data head after adding new features:")
                if generator.get_data() is not None:
                    print(generator.get_data().head())
            else:
                print("Skipping add features: Missing required parameters in config")
        except Exception as e:
            print(f"Error adding new features: {e}")

    # --- 4. Create Target Variable (Optional) ---
    if hasattr(config, "CREATE_TARGET_SETTINGS") and config.CREATE_TARGET_SETTINGS:
        print("\n--- Creating Target Variable ---")
        try:
            # Validate required parameters exist
            required_keys = [
                "features_to_use",
                "weights",
                "noise_level",
                "function_type",
            ]
            if all(key in config.CREATE_TARGET_SETTINGS for key in required_keys):
                features_for_target = config.CREATE_TARGET_SETTINGS["features_to_use"]

                # Robust check for features_for_target
                current_data = generator.get_data()
                if current_data is not None:
                    available_features = [
                        col for col in current_data.columns if col != "target"
                    ]

                    # Validate that specified features exist
                    valid_features = [
                        f for f in features_for_target if f in available_features
                    ]

                    if valid_features and len(valid_features) == len(
                        features_for_target
                    ):
                        generator.create_target_variable(
                            features_to_use=config.CREATE_TARGET_SETTINGS[
                                "features_to_use"
                            ],
                            weights=config.CREATE_TARGET_SETTINGS["weights"],
                            noise_level=config.CREATE_TARGET_SETTINGS["noise_level"],
                            function_type=config.CREATE_TARGET_SETTINGS[
                                "function_type"
                            ],
                        )
                        print("Data head after creating target variable:")
                        print(generator.get_data().head())
                    else:
                        print(
                            f"Skipping target creation: Invalid features specified. Available: {available_features}"
                        )
                else:
                    print("Skipping target creation: No data available.")
            else:
                print("Skipping target creation: Missing required parameters in config")
        except Exception as e:
            print(f"Error creating target variable: {e}")

    # --- 5. Visualize Features ---
    print("\n--- Visualizing Features ---")
    if generator.get_data() is not None:
        try:
            # Validate required visualization parameters
            if (
                hasattr(config, "VISUALIZATION_SETTINGS")
                and config.VISUALIZATION_SETTINGS
                and all(
                    key in config.VISUALIZATION_SETTINGS
                    for key in [
                        "features_to_visualise",
                        "max_features_to_show",
                        "n_bins",
                    ]
                )
            ):
                figures_dir_to_save = config.VISUALIZATION_SETTINGS.get("save_to_dir")
                if hasattr(config, "FIGURES_OUTPUT_DIR") and config.FIGURES_OUTPUT_DIR:
                    figures_dir_to_save = config.FIGURES_OUTPUT_DIR
                    print(f"Saving figures to directory: {figures_dir_to_save}")

                # Validate that specified features exist
                current_data = generator.get_data()
                available_features = list(current_data.columns)
                specified_features = config.VISUALIZATION_SETTINGS[
                    "features_to_visualise"
                ]
                valid_features = [
                    f for f in specified_features if f in available_features
                ]

                if valid_features:
                    generator.visualise_features(
                        features=valid_features,
                        max_features_to_show=config.VISUALIZATION_SETTINGS[
                            "max_features_to_show"
                        ],
                        n_bins=config.VISUALIZATION_SETTINGS["n_bins"],
                        save_to_dir=figures_dir_to_save,
                    )
                else:
                    print(
                        f"No valid features to visualize. Available: {available_features}"
                    )
            else:
                print("Skipping visualization: Missing required parameters in config")
        except Exception as e:
            print(f"Error during visualization: {e}")
    else:
        print("No data to visualize.")

    # --- 6. Display Feature Information ---
    print("\n--- Final Feature Information ---")
    try:
        print(generator.get_feature_information())
    except Exception as e:
        print(f"Error getting feature information: {e}")

    # --- Save Data ---
    if generator.get_data() is not None:
        try:
            # Debug: Print the paths
            print(f"Config PROJECT_ROOT: {config.PROJECT_ROOT}")
            print(f"Config OUTPUT_DATA_PATH: {config.OUTPUT_DATA_PATH}")

            # Use the path from config
            output_path = config.OUTPUT_DATA_PATH

            # Create directory if it doesn't exist
            import os

            output_dir = os.path.dirname(output_path)
            if output_dir:
                print(f"Creating directory: {output_dir}")
                os.makedirs(output_dir, exist_ok=True)
                print(f"Directory created/exists: {output_dir}")

            generator.get_data().to_csv(output_path, index=False)
            print(f"\nGenerated data saved to {output_path}")
        except Exception as e:
            print(f"Error saving data: {e}")


if __name__ == "__main__":
    run_data_generation()
