import os
import argparse
from src.data_generator_module import GaussianDataGenerator, utils
from src.data_generator_module.utils import (
    create_filename_from_config,
    create_plot_title_from_config,
    rename_config_file,
    set_global_seed,
)


def main():
    """
    Runs the feature-based signal vs noise data generation pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Run feature-based signal vs noise data generation pipeline."
    )
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
        print(f"Error: {config_path} not found. Please ensure it exists.")
        return
    except Exception as e:
        print(f"Error loading or parsing {config_path}: {e}")
        return

    # Validate configuration structure
    if "create_feature_based_signal_noise_classification" not in config:
        print(
            "Error: Configuration must include 'create_feature_based_signal_noise_classification' section"
        )
        print("This pipeline supports feature-based signal vs noise classification")
        return

    # Set the global random seed for reproducibility
    global_seed = config["global_settings"]["random_seed"]
    utils.set_global_seed(global_seed)

    # Generate unique experiment name from configuration
    experiment_name = create_filename_from_config(config)
    print(f"Generated experiment name: {experiment_name}")

    # Initialise the data generator
    dataset_settings = config["dataset_settings"]
    generator = GaussianDataGenerator(
        n_samples=dataset_settings["n_samples"],
        n_features=dataset_settings["n_initial_features"],
        random_state=global_seed,
    )

    # Execute the data generation pipeline
    print("\nStarting feature-based signal vs noise data generation...")

    # Step 2: Create feature-based signal/noise classification target
    if "create_feature_based_signal_noise_classification" in config:
        feature_config = config["create_feature_based_signal_noise_classification"]
        generator.create_feature_based_signal_noise_classification(
            signal_features=feature_config["signal_features"],
            noise_features=feature_config["noise_features"],
            feature_types=feature_config["feature_types"],
            store_for_visualisation=feature_config.get(
                "store_for_visualisation", False
            ),
        )
        
    if "perturbation_settings" in config:
        print("\nApplying perturbations...")
        perturbations = config["perturbation_settings"]
        for p_config in perturbations:
            generator.perturb_feature(
                feature_name=p_config['feature'],
                class_label=p_config['class_label'],
                sigma_shift=p_config['sigma_shift']
            )
            
    # Step 3: Save the generated dataset
    output_settings = config.get("output_settings", {"data_dir": "data/"})
    output_data_dir = output_settings.get("data_dir", "data/")
    dataset_filepath = os.path.join(output_data_dir, f"{experiment_name}_dataset.csv")
    generator.save_data(file_path=dataset_filepath)

    # Step 4: Generate visualisations
    if "visualisation" in config:
        vis_config = config["visualisation"]
        main_title, subtitle = create_plot_title_from_config(config)

        base_plot_dir = vis_config.get("save_to_dir", "reports/figures")
        experiment_plot_dir = os.path.join(base_plot_dir, experiment_name)
        os.makedirs(experiment_plot_dir, exist_ok=True)

        if "create_feature_based_signal_noise_classification" in config:
            feature_wise_plot_path = os.path.join(
                experiment_plot_dir, f"feature_wise_signal_noise_{experiment_name}.pdf"
            )
            generator.visualise_signal_noise_by_features(
                save_path=feature_wise_plot_path,
                title=main_title,
                subtitle=subtitle,
            )

    # Step 5: Configuration management
    if not args.keep_original_name:
        renamed_config_path = utils.rename_config_file(config_path, experiment_name)
        print(f"Configuration file available at: {renamed_config_path}")
    else:
        print(f"Configuration file kept at original location: {config_path}")

    # Step 6: Generate summary report
    print("\n" + "=" * 60)
    print("FEATURE-BASED SIGNAL VS NOISE GENERATION SUMMARY")
    print("=" * 60)

    # Display dataset information
    data_summary = generator.get_data_summary()
    if hasattr(generator, "feature_based_metadata"):
        metadata = generator.feature_based_metadata
        print(f"Signal Ratio: {metadata['signal_ratio']:.1%}")
        print(f"Actual Signal Ratio: {metadata['actual_signal_ratio']:.1%}")
        print(f"Signal Features: {metadata.get('signal_features', 'N/A')}")
        print(f"Signal Coefficients: {metadata.get('signal_coefficients', 'N/A')}")
        print(f"Approach: {metadata.get('approach', 'feature_based_learning')}")

    print("\nGenerated Files:")
    print(f"  Dataset: {dataset_filepath}")
    print(
        f"  Feature Plots: {plot_filepath if 'plot_filepath' in locals() else 'Not generated'}"
    )

    print("\nDataset Structure:")
    if generator.data is not None:
        print(f"  Shape: {generator.data.shape}")
        print(f"  Columns: {list(generator.data.columns)}")
        if "target" in generator.data.columns:
            signal_count = (generator.data["target"] == 1).sum()
            noise_count = (generator.data["target"] == 0).sum()
            print(f"  Signal samples (target=1): {signal_count}")
            print(f"  Noise samples (target=0): {noise_count}")

    print("\nFeature-based signal vs noise data generation completed successfully!")
    print(
        "Neural networks will learn to classify samples based on feature combinations only."
    )


if __name__ == "__main__":
    main()
