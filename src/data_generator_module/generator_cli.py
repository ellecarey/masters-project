import os
from pathlib import Path
import pandas as pd
from src.data_generator_module import utils
from src.data_generator_module.gaussian_data_generator import GaussianDataGenerator
import yaml
from copy import deepcopy
from src.utils.dataset_lifecycle import (
    dataset_path,
    save_new_dataset,
    write_config_without_overwrite,
)
from src.data_generator_module.utils import (
    find_project_root,
    create_filename_from_config,
    create_plot_title_from_config
)


TRAINING_SEED = 99

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
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Generation config not found: {config_path}") from e

    # Validate configuration structure
    if "create_feature_based_signal_noise_classification" not in config:
        raise ValueError("Configuration must include 'create_feature_based_signal_noise_classification'")

    # Set the global random seed for reproducibility
    global_seed = config["global_settings"]["random_seed"]
    utils.set_global_seed(global_seed)

    # Generate unique experiment name from configuration
    experiment_name = utils.create_filename_from_config(config)
    print(f"Generated experiment name: {experiment_name}")
    dataset_filepath = dataset_path(Path(find_project_root()), config, experiment_name)
    if dataset_filepath.exists() or dataset_filepath.is_symlink():
        raise FileExistsError(f"Refusing to overwrite dataset: {dataset_filepath}")

    # Initialise the data generator
    dataset_settings = config["dataset_settings"]
    generator = GaussianDataGenerator(
        n_samples=dataset_settings["n_samples"],
        n_features=dataset_settings["n_initial_features"],
        random_state=global_seed,
        dataset_settings=dataset_settings # &lt;-- Pass the settings here
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
            generator.apply_perturbation_from_config(p_config)

    # Save the generated dataset
    generated = save_new_dataset(generator.data, dataset_filepath, Path(config_path))
    print(f"Data successfully saved to {dataset_filepath}")

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
    return generated

def generate_multi_seed(base_config_path: str, num_seeds: int = 10, start_seed: int = 0, generate_training_seed: bool = True):
    """
    Generate multiple datasets from a base config, varying random_seed.
    Can optionally generate a dedicated training/tuning dataset.
    """
    import yaml
    from pathlib import Path
    from src.data_generator_module.utils import find_project_root, create_filename_from_config
    
    project_root = Path(find_project_root())
    config_dir = project_root / "configs" / "data_generation"
    generated = []

    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    # Generate evaluation seeds (e.g., 0 through 4)
    print(f"--- Generating {num_seeds} evaluation datasets (seeds {start_seed} to {start_seed + num_seeds - 1}) ---")
    for i in range(num_seeds):
        current_seed = start_seed + i
        new_config = deepcopy(base_config)
        new_config["global_settings"]["random_seed"] = current_seed
        
        new_config_base_name = create_filename_from_config(new_config)
        new_config_filename = f"{new_config_base_name}_config.yml"
        new_config_path = config_dir / new_config_filename

        write_config_without_overwrite(new_config_path, new_config)
        generated.append(generate_from_config(str(new_config_path), keep_original_name=True))

    # Generate the dedicated training seed if requested
    if generate_training_seed:
        print(f"\n--- Generating dedicated training dataset (seed {TRAINING_SEED}) ---")
        training_config = deepcopy(base_config)
        training_config["global_settings"]["random_seed"] = TRAINING_SEED
    
        temp_base_name = create_filename_from_config(training_config)
        training_base_name = temp_base_name.replace(f"_seed{TRAINING_SEED}", "_training")
        
        training_config_filename = f"{training_base_name}_config.yml" 
        training_config_path = config_dir / training_config_filename

        write_config_without_overwrite(training_config_path, training_config)
        generated.append(generate_from_config(str(training_config_path), keep_original_name=True))
    return generated
        
def perturb_multi_seed(data_config_base: str, perturb_config: str, *, config_paths=None):
    """
    Apply perturbations to a family of datasets (multi-seed).
    """

    project_root = Path(find_project_root())
    perturb_config_path = project_root / perturb_config
    with open(perturb_config_path, 'r') as f:
        perturb_data = yaml.safe_load(f)

    base_data_config_path = project_root / data_config_base
    family_name = base_data_config_path.stem.split('_seed')[0]
    data_config_dir = project_root / "configs" / "data_generation"
    
    all_data_configs = sorted(list(data_config_dir.glob(f"{family_name}_seed*_config.yml")))
    
    # Exclude the training seed from perturbation
    evaluation_configs = (
        [Path(p) for p in config_paths]
        if config_paths is not None
        else [p for p in all_data_configs if "_training" not in p.name]
    )
    print(f"Found {len(evaluation_configs)} evaluation datasets to perturb ('_training' dataset will be skipped).")
    if not evaluation_configs:
        raise FileNotFoundError(f"No evaluation configs found for family: {family_name}")
    generated = []
    
    for data_config_path in evaluation_configs:
        with open(data_config_path, 'r') as f:
            data_config = yaml.safe_load(f)
        dataset_base_name = create_filename_from_config(data_config)
        input_dataset_path = dataset_path(project_root, data_config, dataset_base_name)
        if not input_dataset_path.is_file():
            raise FileNotFoundError(f"Original dataset not found: {input_dataset_path}")
        generator = GaussianDataGenerator(
            n_samples=data_config['dataset_settings']['n_samples'],
            n_features=data_config['dataset_settings']['n_initial_features'],
            random_state=data_config['global_settings']['random_seed']
        )
        generator.data = pd.read_csv(input_dataset_path)
        generator.feature_based_metadata = {
            'signal_features': data_config['create_feature_based_signal_noise_classification']['signal_features'],
            'noise_features': data_config['create_feature_based_signal_noise_classification']['noise_features'],
            'perturbations': []
        }

        for p_conf in perturb_data['perturbation_settings']:
            generator.apply_perturbation_from_config(p_conf)

        perturbed_config = deepcopy(data_config)
        perturbed_config['perturbation_settings'] = perturb_data['perturbation_settings']
        new_filename_base = create_filename_from_config(perturbed_config)
        new_dataset_path = dataset_path(project_root, perturbed_config, new_filename_base)
        new_config_path = data_config_dir / f"{new_filename_base}_config.yml"
        write_config_without_overwrite(new_config_path, perturbed_config)
        generated.append(save_new_dataset(generator.data, new_dataset_path, new_config_path))
        print(f"Saved new config to: {new_config_path.name}")
        current_seed = data_config.get("global_settings", {}).get("random_seed", -1)
        if current_seed == 0 and "visualisation" in data_config:
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
    return generated
