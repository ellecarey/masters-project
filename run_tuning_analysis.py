import os
import argparse
from pathlib import Path
import yaml
import optuna
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import warnings

from src.data_generator_module import utils as data_utils
from optuna.trial import TrialState
from optuna.visualization.matplotlib import plot_pareto_front, plot_optimization_history, plot_param_importances
from optuna.exceptions import ExperimentalWarning
warnings.filterwarnings("ignore", category=ExperimentalWarning)

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
    model_name_suffix = base_training_config_path.stem

    study_name = f"{dataset_base_name}_{base_training_config_path.stem}"
    storage_name = f"sqlite:///reports/{dataset_base_name}_{model_name_suffix}_tuning.db"

    print(f"--- Analysing Study: {study_name} ---")
    print(f"Loading results from: {storage_name}")

    # --- 2. Load the Completed Study ---
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_name)
    except KeyError:
        print(f"\nError: Study '{study_name}' not found in the database.")
        return
    completed_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    if len(completed_trials) == 0:
        print("\nError: No trials in the study completed successfully.")
        print("This may be because all trials were pruned, failed, or the job was interrupted.")
        print("Cannot determine the best hyperparameters. Please check the tuning logs.")
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        failed_trials = study.get_trials(deepcopy=False, states=[TrialState.FAIL])
        print(f"Summary: {len(pruned_trials)} trials pruned, {len(failed_trials)} trials failed.")
        return
        
    # --- 3. Print Best Trial Result (Single-Objective) ---
    best_trial = study.best_trial
    if not best_trial:
        print("No successful trials found in the study. Exiting.")
        return

    print("\n--- Best Trial Found ---")
    print(f" Trial Number: {best_trial.number}")
    print(f" Objective Value (AUC): {best_trial.value:.4f}")
    # Retrieve training time from user attributes
    training_time = best_trial.user_attrs.get("training_time", "N/A")
    if isinstance(training_time, float):
        print(f" Training Time: {training_time:.2f}s")
    else:
        print(f" Training Time: {training_time}")
    print(f" Hyperparameters: {best_trial.params}")
    
    # --- 4. Generate and Save Tuning Visualisations ---
    print("\n--- Generating Tuning Visualisations (using Matplotlib) ---")
    model_name = tuning_config["model_name"]
    output_plot_dir = project_root / "reports" / "figures" / dataset_base_name / model_name
    os.makedirs(output_plot_dir, exist_ok=True)
    print(f"Saving plots to: {output_plot_dir}")

    plot_optimization_history(study, target_name="AUC")
    plt.savefig(output_plot_dir / "tuning_optimisation_history_auc.pdf", bbox_inches='tight')
    plt.close()

    try:
        plot_param_importances(study, target_name="AUC")
        plt.savefig(output_plot_dir / "tuning_param_importances_auc.pdf", bbox_inches='tight')
        plt.close()
    except RuntimeError as e:
        if "Encountered zero total variance" in str(e):
            print("\nWarning: Could not generate parameter importance plot.")
            print("This typically happens when all trials have a perfect score (e.g., AUC=1.0),")
            print("making it impossible to determine which parameter is more important.")
        else:
            # Re-raise any other runtime errors
            raise e
            
    # --- 5. Generate Final Configuration File ---
    print("\n--- Generating Final Configuration File ---")
    optimal_config_path = create_and_save_optimal_config(
        best_params=best_trial.params,
        final_data_config_path=str(data_config_path),
        base_training_config_path=str(base_training_config_path),
        tuning_config=tuning_config,
    )

    # --- 6. Print Next Steps ---
    print("\n--- Next Step: Final Training on Full Dataset ---")
    print("To train your final model, run the following command:")
    print("\n" + "=" * 80)
    print(f"uv run run_training.py -dc {args.data_config} -tc {optimal_config_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

