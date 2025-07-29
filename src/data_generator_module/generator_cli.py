import os
from pathlib import Path
import pandas as pd
from src.data_generator_module import utils
from src.data_generator_module.gaussian_data_generator import GaussianDataGenerator

def generate_from_config(config_path: str, keep_original_name: bool = False):
    """
    Generate a dataset from a YAML config file, with support for
    signal/noise separation, perturbations, and visualisation.
    Arguments:
        config_path: Path to your YAML config.
        keep_original_name: If True, retains config filename.
    """

    # Load configuration and set up reproducibility
    try:
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
        print("Error: Configuration must include 'create_feature_based_signal_noise_classification' section")
        print("This pipeline supports feature-based signal vs noise classification")
        return

    # Set the global random seed for reproducibility
    global_seed = config["global_settings"]["random_seed"]
    utils.set_global_seed(global_seed)

    # Generate unique experiment name from configuration
    experiment_name = utils.create_filename_from_config(config)
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
    feature_config = config["create_feature_based_signal_noise_classification"]
    generator.create_feature_based_signal_noise_classification(
        signal_features=feature_config["signal_features"],
        noise_features=feature_config["noise_features"],
        feature_types=feature_config["feature_types"],
        store_for_visualisation=feature_config.get("store_for_visualisation", False)
    )

    # Apply perturbations if any
    if "perturbation_settings" in config:
        print("\nApplying perturbations...")
        for p_config in config["perturbation_settings"]:
            generator.perturb_feature(
                feature_name=p_config['feature'],
                class_label=p_config['class_label'],
                sigma_shift=p_config['sigma_shift']
            )

    # Save the generated dataset
    output_settings = config.get("output_settings", {"data_dir": "data/"})
    output_data_dir = output_settings.get("data_dir", "data/")
    dataset_filepath = os.path.join(output_data_dir, f"{experiment_name}_dataset.csv")
    generator.save_data(file_path=dataset_filepath)

    # Generate visualisations if requested
    if "visualisation" in config:
        vis_config = config["visualisation"]
        main_title, subtitle = utils.create_plot_title_from_config(config)
        
        # Use family-based path structure - put plot directly in the experiment subfolder
        from src.utils.report_paths import experiment_family_path
        feature_wise_plot_path = experiment_family_path(
            full_experiment_name=experiment_name,
            art_type="figure",
            subfolder=experiment_name,  # Use full experiment name as subfolder
            filename=f"feature_wise_signal_noise.pdf"
        )
        
        generator.visualise_signal_noise_by_features(
            save_path=str(feature_wise_plot_path),
            title=main_title,
            subtitle=subtitle,
        )
        
        print(f"Generated visualization: {feature_wise_plot_path}")

    # Configuration management
    if not keep_original_name:
        renamed_config_path = utils.rename_config_file(config_path, experiment_name)
        print(f"Configuration file available at: {renamed_config_path}")
    else:
        print(f"Configuration file kept at original location: {config_path}")

    # Print summary (optional)
    print("\n" + "=" * 60)
    print("FEATURE-BASED SIGNAL VS NOISE GENERATION SUMMARY")
    print("=" * 60)
    data_summary = generator.get_data_summary()
    if hasattr(generator, "feature_based_metadata"):
        metadata = generator.feature_based_metadata
        print(f"Signal Ratio: {metadata['signal_ratio']:.1%}")
        print(f"Actual Signal Ratio: {metadata['actual_signal_ratio']:.1%}")
        print(f"Signal Features: {metadata.get('signal_features', 'N/A')}")
        print(f"Signal Coefficients: {metadata.get('signal_coefficients', 'N/A')}")
        print(f"Approach: {metadata.get('approach', 'feature_based_learning')}")
    print("\nGenerated Files:")
    print(f" Dataset: {dataset_filepath}")
    print("\nDataset Structure:")
    if generator.data is not None:
        print(f" Shape: {generator.data.shape}")
        print(f" Columns: {list(generator.data.columns)}")
        if "target" in generator.data.columns:
            signal_count = (generator.data["target"] == 1).sum()
            noise_count = (generator.data["target"] == 0).sum()
            print(f" Signal samples (target=1): {signal_count}")
            print(f" Noise samples (target=0): {noise_count}")
    print("\nFeature-based signal vs noise data generation completed successfully!")
    print("Neural networks will learn to classify samples based on feature combinations only.\n")

def generate_multi_seed(base_config_path: str, num_seeds: int = 5, start_seed: int = 0):
    """
    Generate multiple datasets from a base config, varying random_seed.
    """
    import yaml
    from pathlib import Path
    from src.data_generator_module.utils import find_project_root, create_filename_from_config

    project_root = Path(find_project_root())
    config_dir = project_root / "configs" / "data_generation"
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)
    for i in range(num_seeds):
        current_seed = start_seed + i
        new_config = base_config.copy()
        new_config["global_settings"]["random_seed"] = current_seed
        new_config_base_name = create_filename_from_config(new_config)
        new_config_filename = f"{new_config_base_name}_config.yml"
        new_config_path = config_dir / new_config_filename
        with open(new_config_path, 'w') as f:
            yaml.dump(new_config, f, default_flow_style=False)
        generate_from_config(str(new_config_path), keep_original_name=True)

def perturb_multi_seed(data_config_base: str, perturb_config: str):
    """
    Apply perturbations to a family of datasets (multi-seed).
    """
    import yaml
    import pandas as pd
    from pathlib import Path
    from src.data_generator_module.gaussian_data_generator import GaussianDataGenerator
    from src.data_generator_module.utils import (
        find_project_root,
        create_filename_from_config,
        create_plot_title_from_config
    )

    project_root = Path(find_project_root())
    perturb_config_path = project_root / perturb_config
    with open(perturb_config_path, 'r') as f:
        perturb_data = yaml.safe_load(f)

    base_data_config_path = project_root / data_config_base
    family_name = base_data_config_path.stem.split('_seed')[0]
    data_config_dir = project_root / "configs" / "data_generation"
    all_data_configs = sorted(list(data_config_dir.glob(f"{family_name}_seed*_config.yml")))
    for data_config_path in all_data_configs:
        with open(data_config_path, 'r') as f:
            data_config = yaml.safe_load(f)
        dataset_base_name = create_filename_from_config(data_config)
        dataset_path = project_root / "data" / f"{dataset_base_name}_dataset.csv"
        if not dataset_path.exists():
            print(f" - Skipping: Original dataset not found at {dataset_path}")
            continue
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
        for p_conf in perturb_data['perturbation_settings']:
            generator.perturb_feature(
                feature_name=p_conf['feature'],
                class_label=p_conf['class_label'],
                sigma_shift=p_conf['sigma_shift']
            )
        perturbed_config = data_config.copy()
        perturbed_config['perturbation_settings'] = perturb_data['perturbation_settings']
        new_filename_base = create_filename_from_config(perturbed_config)
        new_dataset_path = project_root / "data" / f"{new_filename_base}_dataset.csv"
        generator.save_data(str(new_dataset_path))
        new_config_path = data_config_dir / f"{new_filename_base}_config.yml"
        with open(new_config_path, 'w') as f:
            yaml.dump(perturbed_config, f, default_flow_style=False)
        print(f"Saved new config to: {new_config_path.name}")
        if "visualisation" in data_config:
            vis_config = data_config["visualisation"]
            main_title, subtitle = create_plot_title_from_config(perturbed_config)
            
            subfolder = new_filename_base
            
            # Use family-based path structure
            from src.utils.report_paths import experiment_family_path
            feature_wise_plot_path = experiment_family_path(
                full_experiment_name=new_filename_base,
                art_type="figure",
                subfolder=subfolder,
                filename=f"feature_wise_signal_noise_{new_filename_base}.pdf"
            )
        
            generator.visualise_signal_noise_by_features(
                save_path=str(feature_wise_plot_path),
                title=main_title,
                subtitle=subtitle,
            )
        
            print(f"Generated visualisation: {feature_wise_plot_path}")
    
        print("\nMulti-seed perturbation complete.")
