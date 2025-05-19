import random
import numpy as np
import torch

from generator_package.gaussian_data_generator import GuassianDataGenerator
from generator_package import config


def run_data_generation():
    """
    Runs the data generation process based on settings in config.py.
    """
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
        n_features=config.N_INITIAL_FEATURES,  # This is the initial number of features to generate
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
    else:
        print("No data generated yet.")

    # --- 2. Add Perturbations (Optional) ---
    if config.PERTURBATION_SETTINGS:
        print("\n--- Adding Perturbations ---")
        generator.add_perturbations(
            perturbation_type=config.PERTURBATION_SETTINGS["perturbation_type"],
            features=config.PERTURBATION_SETTINGS["features_to_perturb"],
            scale=config.PERTURBATION_SETTINGS["scale"],
        )
        print("Data head after perturbations:")
        print(generator.get_data().head())

    # --- 3. Add More Features (Optional) ---
    if config.ADD_FEATURES_SETTINGS:
        print("\n--- Adding New Features ---")
        generator.add_features(
            n_new_features=config.ADD_FEATURES_SETTINGS["n_new_features"],
            feature_parameters=config.ADD_FEATURES_SETTINGS.get("feature_parameters"),
            feature_types=config.ADD_FEATURES_SETTINGS.get("feature_types"),
        )
        print("Data head after adding new features:")
        print(generator.get_data().head())

    # --- 4. Create Target Variable (Optional) ---
    if config.CREATE_TARGET_SETTINGS:
        print("\n--- Creating Target Variable ---")
        # Ensure features_to_use for target exist, or select from available if None in config
        features_for_target = config.CREATE_TARGET_SETTINGS["features_to_use"]
        if features_for_target is None and generator.get_data() is not None:
            # Use up to first 3 available features if not specified
            features_for_target = generator.get_data().columns.tolist()
            features_for_target = [f for f in features_for_target if f != "target"][:3]

        generator.create_target_variable(
            features_to_use=features_for_target,
            weights=config.CREATE_TARGET_SETTINGS["weights"],
            noise_level=config.CREATE_TARGET_SETTINGS["noise_level"],
            function_type=config.CREATE_TARGET_SETTINGS["function_type"],
        )
        print("Data head after creating target variable:")
        print(generator.get_data().head())

    # --- 5. Visualize Features ---
    print("\n--- Visualizing Features ---")
    if generator.get_data() is not None:
        generator.visusalise_features(
            features=config.VISUALIZATION_SETTINGS["features_to_visualise"],
            max_features_to_show=config.VISUALIZATION_SETTINGS["max_features_to_show"],
            n_bins=config.VISUALIZATION_SETTINGS["n_bins"],
        )
    else:
        print("No data to visualize.")

    # --- 6. Display Feature Information ---
    print("\n--- Final Feature Information ---")
    print(generator.get_feature_information())

    # --- Save Data ---
    if (
        hasattr(config, "OUTPUT_DATA_PATH")
        and config.OUTPUT_DATA_PATH
        and generator.get_data() is not None
    ):
        try:
            generator.get_data().to_csv(config.OUTPUT_DATA_PATH, index=False)
            print(f"\nGenerated data saved to {config.OUTPUT_DATA_PATH}")
        except Exception as e:
            print(f"Error saving data: {e}")


if __name__ == "__main__":
    run_data_generation()
