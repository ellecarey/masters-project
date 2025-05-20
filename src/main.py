import random
import numpy as np
import torch

from generator_package import config
from generator_package.gaussian_data_generator import GuassianDataGenerator
from generator_package.plotting_style import (
    apply_custom_plot_style,
)


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
        n_features=config.N_INITIAL_FEATURES,  # Initial number of features for the first call
        random_state=config.RANDOM_STATE,
    )

    # --- 1. Generate Initial Features ---
    print("\n--- Generating Initial Features ---")
    generator.generate_features(
        feature_parameters=config.FEATURE_PARAMETERS_OVERRIDE,
        feature_types=config.FEATURE_TYPES_OVERRIDE,
    )
    print("Initial data head:")
    if generator.get_data() is not None:
        print(generator.get_data().head())

    # --- 2. Add Perturbations (Optional) ---
    if config.PERTURBATION_SETTINGS:
        print("\n--- Adding Perturbations ---")
        generator.add_perturbations(
            perturbation_type=config.PERTURBATION_SETTINGS["perturbation_type"],
            features=config.PERTURBATION_SETTINGS.get(
                "features_to_perturb"
            ),  # Use .get for safety
            scale=config.PERTURBATION_SETTINGS["scale"],
        )
        print("Data head after perturbations:")
        if generator.get_data() is not None:
            print(generator.get_data().head())

    # --- 3. Add More Features (Optional) ---
    if config.ADD_FEATURES_SETTINGS:
        print("\n--- Adding New Features ---")
        generator.add_features(
            n_new_features=config.ADD_FEATURES_SETTINGS.get("n_new_features", 0),
            feature_parameters=config.ADD_FEATURES_SETTINGS.get("feature_parameters"),
            feature_types=config.ADD_FEATURES_SETTINGS.get("feature_types"),
        )
        print("Data head after adding new features:")
        if generator.get_data() is not None:
            print(generator.get_data().head())

    # --- 4. Create Target Variable (Optional) ---
    if config.CREATE_TARGET_SETTINGS:
        print("\n--- Creating Target Variable ---")
        features_for_target = config.CREATE_TARGET_SETTINGS.get("features_to_use")

        # Robust check for features_for_target
        current_data = generator.get_data()
        if current_data is not None:
            available_features = [
                col for col in current_data.columns if col != "target"
            ]
            if features_for_target is None and available_features:
                features_for_target = available_features[
                    : min(3, len(available_features))
                ]  # Default to first 3
            elif features_for_target:  # if specified, ensure they exist
                features_for_target = [
                    f for f in features_for_target if f in available_features
                ]

            if features_for_target:  # Proceed only if we have valid features for target
                generator.create_target_variable(
                    features_to_use=features_for_target,
                    weights=config.CREATE_TARGET_SETTINGS.get("weights"),
                    noise_level=config.CREATE_TARGET_SETTINGS.get("noise_level", 0.1),
                    function_type=config.CREATE_TARGET_SETTINGS.get(
                        "function_type", "linear"
                    ),
                )
                print("Data head after creating target variable:")
                print(generator.get_data().head())
            else:
                print("Skipping target creation: No valid features_to_use for target.")
        else:
            print("Skipping target creation: No data available.")

    # --- 5. Visualize Features ---
    print("\n--- Visualizing Features ---")
    if generator.get_data() is not None:
        figures_dir_to_save = None
        if hasattr(config, "FIGURES_OUTPUT_DIR") and config.FIGURES_OUTPUT_DIR:
            figures_dir_to_save = config.FIGURES_OUTPUT_DIR

        print(f"Saving figures to directory: {figures_dir_to_save}")
        generator.visusalise_features(
            features=config.VISUALIZATION_SETTINGS.get("features_to_visualise"),
            max_features_to_show=config.VISUALIZATION_SETTINGS.get(
                "max_features_to_show", 10
            ),
            n_bins=config.VISUALIZATION_SETTINGS.get("n_bins", 30),
            save_to_dir=figures_dir_to_save,
        )
    else:
        print("No data to visualize.")

    # --- 6. Display Feature Information ---
    print("\n--- Final Feature Information ---")
    print(generator.get_feature_information())

    # --- Save Data ---
    if hasattr(config, "OUTPUT_DATA_PATH") and config.OUTPUT_DATA_PATH:
        if generator.get_data() is not None:
            try:
                generator.get_data().to_csv(config.OUTPUT_DATA_PATH, index=False)
                print(f"\nGenerated data saved to {config.OUTPUT_DATA_PATH}")
            except Exception as e:
                print(f"Error saving data: {e}")


if __name__ == "__main__":
    run_data_generation()
