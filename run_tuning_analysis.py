import os
import argparse
from pathlib import Path
import yaml
import optuna
import matplotlib.pyplot as plt

# Import utility functions
from src.data_generator_module import utils as data_utils
from optuna.visualization.matplotlib import plot_pareto_front, plot_optimization_history, plot_param_importances

def create_and_save_optimal_config(best_params, final_data_config_path, base_training_config_path, tuning_config):
    """
    Creates a new training config file populated with the best hyperparameters.
    """
    with open(base_training_config_path, 'r') as f:
        config_template = yaml.safe_load(f)
    if 'model_name' in tuning_config:
        config_template['training_settings']['model_name'] = tuning_config['model_name']
    config_template['training_settings']['hyperparameters'].update(best_params)
    final_data_config = data_utils.load_yaml_config(final_data_config_path)
    dataset_base_name = data_utils.create_filename_from_config(final_data_config)
    training_suffix = Path(base_training_config_path).stem
    new_config_filename = f"{dataset_base_name}_{training_suffix}_optimal.yml"
    save_dir = "configs/training/generated"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, new_config_filename)
    with open(save_path, 'w') as f:
        yaml.dump(config_template, f, default_flow_style=False, sort_keys=False)
    print(f"\nOptimal training configuration saved to: {save_path}")
    return save_path

def main():
    """
    Analyses a completed Optuna study, allows the user to select the best
    trial, and generates the final training configuration.
    """
    parser = argparse.ArgumentParser(description="Analyse a completed hyperparameter tuning study.")
    parser.add_argument("--data-config", "-dc", required=True, help="Path to the data config file used for the study.")
    parser.add_argument("--base-training-config", "-btc", required=True, help="Path to the base training config template used for the study.")
    args = parser.parse_args()

    try:
        project_root = Path(data_utils.find_project_root())
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # --- 1. Reconstruct Study Name and Storage Path ---
    data_config_path = Path(args.data_config)
    base_training_config_path = Path(args.base_training_config)
    data_config = data_utils.load_yaml_config(data_config_path)
    tuning_config_path = project_root / "configs" / "tuning" / f"{base_training_config_path.stem}.yml"
    
    try:
        tuning_config = data_utils.load_yaml_config(tuning_config_path)
    except FileNotFoundError:
        print(f"\nError: Tuning configuration not found at '{tuning_config_path}'")
        return

    dataset_base_name = data_utils.create_filename_from_config(data_config)
    study_name = f"{dataset_base_name}_{base_training_config_path.stem}"
    storage_name = f"sqlite:///reports/{dataset_base_name}_tuning.db"

    print(f"--- Analysing Study: {study_name} ---")
    print(f"Loading results from: {storage_name}")

    # --- 2. Load the Completed Study ---
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_name)
    except KeyError:
        print(f"\nError: Study '{study_name}' not found in the database.")
        print("Please ensure you have run the hyperparameter tuning first.")
        return

    # --- 3. Print Pareto Front Results ---
    pareto_front = study.best_trials
    if not pareto_front:
        print("No successful trials found in the study. Exiting.")
        return
    
    print(f"\nFound {len(pareto_front)} Pareto optimal trials.")
    print("\nOptimal trials (Objective 0: AUC, Objective 1: Training Time):")
    for trial in pareto_front:
        print(f"  Trial {trial.number}:")
        print(f"    Values: AUC={trial.values[0]:.4f}, Time={trial.values[1]:.2f}s")
        print(f"    Params: {trial.params}")

    # --- 4. Generate and Save Tuning Visualisations ---
    print("\n--- Generating Tuning Visualisations (using Matplotlib) ---")
    model_name = tuning_config["model_name"]
    output_plot_dir = project_root / "reports" / "figures" / dataset_base_name / model_name
    os.makedirs(output_plot_dir, exist_ok=True)
    print(f"Saving plots to: {output_plot_dir}")

    plot_pareto_front(study, target_names=["AUC", "Training Time (s)"])
    plt.savefig(output_plot_dir / "tuning_pareto_front.pdf", bbox_inches='tight')
    plt.close()

    plot_optimization_history(study, target=lambda t: t.values[0], target_name="AUC")
    plt.savefig(output_plot_dir / "tuning_optimisation_history_auc.pdf", bbox_inches='tight')
    plt.close()

    plot_optimization_history(study, target=lambda t: t.values[1], target_name="Training Time (s)")
    plt.savefig(output_plot_dir / "tuning_optimisation_history_time.pdf", bbox_inches='tight')
    plt.close()

    plot_param_importances(study, target=lambda t: t.values[0], target_name="AUC")
    plt.savefig(output_plot_dir / "tuning_param_importances_auc.pdf", bbox_inches='tight')
    plt.close()

    # --- 5. User Selects the Best Trial ---
    selected_trial = None
    while not selected_trial:
        try:
            choice = input("\nEnter the number of the trial you want to use for the final config: ")
            trial_number = int(choice)
            selected_trial = next((t for t in study.trials if t.number == trial_number), None)
            if selected_trial and selected_trial in pareto_front:
                print(f"You selected Trial {selected_trial.number}.")
            else:
                print("Invalid trial number or trial is not on the Pareto front. Please choose from the list above.")
                selected_trial = None
        except (ValueError, IndexError):
            print("Invalid input. Please enter a valid trial number.")
        except (KeyboardInterrupt, EOFError):
            print("\nSelection cancelled. Exiting.")
            return

    # --- 6. Generate Final Configuration File ---
    print("\n--- Generating Final Configuration File ---")
    optimal_config_path = create_and_save_optimal_config(
        best_params=selected_trial.params,
        final_data_config_path=str(data_config_path),
        base_training_config_path=str(base_training_config_path),
        tuning_config=tuning_config,
    )

    # --- 7. Print Next Steps ---
    print("\n--- Next Step: Final Training on Full Dataset ---")
    print("To train your final model, run the following command:")
    print("\n" + "=" * 80)
    print(f"uv run run_training.py -dc {args.data_config} -tc {optimal_config_path}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
