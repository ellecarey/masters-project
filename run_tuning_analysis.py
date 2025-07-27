import os
import argparse
from pathlib import Path
import yaml
import optuna
import matplotlib.pyplot as plt
import warnings
import json

from src.data_generator_module import utils as data_utils
from optuna.trial import TrialState
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
from optuna.exceptions import ExperimentalWarning

# Ignore experimental warnings from Optuna's visualization module
warnings.filterwarnings("ignore", category=ExperimentalWarning)

def create_and_save_optimal_config(best_params, final_data_config_path, base_training_config_path, tuning_config):
    """
    Creates a new training config file populated with the best hyperparameters.
    (This function remains unchanged).
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
    Analyses a completed Optuna study, allows the user to select from the
    top 5 trials, and generates the final training configuration.
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
    study_name = f"{dataset_base_name}_{model_name_suffix}"
    storage_name = f"sqlite:///reports/{dataset_base_name}_{model_name_suffix}_tuning.db"

    print(f"--- Analysing Study: {study_name} ---")
    print(f"Loading results from: {storage_name}")

    # --- 2. Load and Validate the Study ---
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_name)
    except KeyError:
        print(f"\nError: Study '{study_name}' not found in the database.")
        return

    completed_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    if len(completed_trials) == 0:
        print("\nError: No trials in the study completed successfully.")
        return

    # --- 3. Display Top 5 Trials for User Selection ---
    completed_trials.sort(key=lambda t: t.value, reverse=True)
    top_trials = completed_trials[:5]

    print("\n--- Top 5 Completed Trials (by AUC) ---")
    table_data = []
    for i, trial in enumerate(top_trials):
        training_time = trial.user_attrs.get('training_time', 'N/A')
        time_str = f"{training_time:.2f}" if isinstance(training_time, float) else "N/A"
        table_data.append({
            "Rank": i + 1,
            "Trial #": trial.number,
            "AUC": f"{trial.value:.4f}",
            "Time (s)": time_str,
            "Params": json.dumps(trial.params)
        })
    
    print(f"{'Rank':<6} | {'Trial #':<8} | {'AUC':<10} | {'Time (s)':<10} | {'Hyperparameters'}")
    print("-" * 120)
    for row in table_data:
        print(f"{row['Rank']:<6} | {row['Trial #']:<8} | {row['AUC']:<10} | {row['Time (s)']:<10} | {row['Params']}")
    
    # --- 4. Prompt User for Selection ---
    selected_trial = None
    while not selected_trial:
        try:
            choice_str = input(f"\nEnter the Rank of the trial to use (1-{len(top_trials)}): ")
            choice_idx = int(choice_str) - 1
            if 0 <= choice_idx < len(top_trials):
                selected_trial = top_trials[choice_idx]
                print(f"You selected Rank {choice_str} (Trial #{selected_trial.number}).")
            else:
                print(f"Invalid rank. Please enter a number between 1 and {len(top_trials)}.")
        except (ValueError, IndexError):
            print("Invalid input. Please enter a number from the list.")
        except (KeyboardInterrupt, EOFError):
            print("\nSelection cancelled. Exiting.")
            return

    # --- 5. Generate Visualisations (based on the whole study) ---
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
    except Exception:
        # Gracefully handle cases where this plot fails (e.g., perfect scores)
        print("\nWarning: Could not generate parameter importance plot.")
        
    # --- 6. Generate Final Configuration File from User's Choice ---
    print("\n--- Generating Final Configuration File ---")
    optimal_config_path = create_and_save_optimal_config(
        best_params=selected_trial.params,
        final_data_config_path=str(data_config_path),
        base_training_config_path=str(base_training_config_path),
        tuning_config=tuning_config,
    )

    # --- 7. Print Next Steps ---
    print("\n" + "="*80)
    print("Next Step: Choose a final training method:")
    print("="*80)

    print("\nOPTION 1: Train a single model on this specific dataset")
    print("-" * 55)
    print(f"uv run run_training.py -dc {args.data_config} -tc {optimal_config_path}")

    print("\nOPTION 2: Train models on ALL datasets in this family (all seeds)")
    print("-" * 65)
    print(f"uv run run_multi_seed_training.py -dcb {args.data_config} -oc {optimal_config_path}")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()