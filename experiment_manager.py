import argparse
import os
import torch.multiprocessing as mp
import subprocess
import yaml
import shutil
from pathlib import Path
import re
from src.data_generator_module.generator_cli import (
    generate_from_config,
    generate_multi_seed,
    perturb_multi_seed,
)
from src.training_module.training_cli import evaluate_multi_seed, train_single_config
from src.training_module.tuning_cli import (
    run_experiments,
    run_hyperparameter_tuning,
    run_tuning_analysis,
)
from src.analysis_module.global_tracker import generate_global_tracking_sheet
from src.analysis_module.analysis_cli import (
    aggregate,
    aggregate_all_families
)
from src.analysis_module.comparison import compare_families
from src.data_generator_module.utils import find_project_root, create_filename_from_config


os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

TRAINING_SEED = 99

def run_full_pipeline(base_data_config: str, tuning_job: str, perturb_config: str = None):
    """
    Orchestrates the entire ML pipeline from data generation to final comparison
    by calling functions directly for improved efficiency and robustness.
    """
    project_root = Path(find_project_root())
    print("="*80)
    print("🚀 STARTING FULL AUTOMATED PIPELINE 🚀")
    print("="*80)

    # --- Step 1: Generate multi-seed datasets ---
    print("\n[STEP 1/6] Generating multi-seed datasets...")
    generate_multi_seed(base_config_path=base_data_config)
    print("✅ Datasets generated successfully.")

    # --- Step 2: Apply perturbations (if specified) ---
    if perturb_config:
        print("\n[STEP 2/6] Applying perturbations...")
        perturb_multi_seed(data_config_base=base_data_config, perturb_config=perturb_config)
        print("✅ Perturbations applied successfully.")

    # --- Determine paths for downstream tasks ---
    with open(base_data_config, "r") as f:
        data_conf_dict = yaml.safe_load(f)
    
    # Create a temporary in-memory config to find the training dataset name
    training_data_conf_dict = data_conf_dict.copy()
    training_data_conf_dict["global_settings"]["random_seed"] = TRAINING_SEED
    temp_base_name = create_filename_from_config(training_data_conf_dict)
    training_base_name = temp_base_name.replace(f"_seed{TRAINING_SEED}", "_training")
    training_data_config_path = project_root / "configs" / "data_generation" / f"{training_base_name}_config.yml"
    
    with open(project_root / "configs" / "experiments.yml", "r") as f:
        job_params = yaml.safe_load(f)["tuning_jobs"][tuning_job]
        base_training_config_path = project_root / job_params["base_training_config"]

    model_name_suffix = base_training_config_path.stem
    optimal_config_path = project_root / "configs" / "training" / "generated" / f"{training_base_name}_{model_name_suffix}_optimal.yml"
    final_model_path = project_root / "models" / f"{training_base_name}_{model_name_suffix}_optimal_model.pt"

    # --- Check for existing tuning results ---
    if optimal_config_path.exists() and final_model_path.exists():
        print("\n[INFO] Found existing optimal configuration and model.")
        print(f"Skipping hyperparameter tuning (Step 3) and analysis (Step 4).")
        print(f"Using existing optimal config: {optimal_config_path.name}")
        print(f"Using existing model: {final_model_path.name}")
    else:
        # --- Step 3: Run hyperparameter tuning ---
        print(f"\n[STEP 3/6] Launching hyperparameter tuning job: '{tuning_job}'...")
        run_experiments(job=tuning_job, data_config_path=str(training_data_config_path))
        print("✅ Hyperparameter tuning complete.")

        # --- Step 4: Analyze tuning results and save optimal model ---
        print("\n[STEP 4/6] Analyzing tuning results (non-interactive)...")
        run_tuning_analysis(
            data_config=str(training_data_config_path),
            base_training_config=str(base_training_config_path),
            sample_fraction=job_params["sample_fraction"],
            non_interactive=True # Ensure non-interactive execution
        )
        print("✅ Tuning analysis complete.")

    # --- Step 5: Evaluate the optimal model on all dataset families ---
    print("\n[STEP 5/6] Evaluating optimal model on all seeds...")
    
    # Evaluate the original family first
    print(f"--- Evaluating on original family from: {Path(base_data_config).name} ---")
    evaluate_multi_seed(
        trained_model_path=str(final_model_path),
        data_config_base=str(base_data_config),
        optimal_config=str(optimal_config_path)
    )

    # Evaluate the perturbed family if it exists
    if perturb_config:
        pert_tag_match = re.search(r'pert_.*', Path(perturb_config).stem)
        if pert_tag_match:
            pert_tag = pert_tag_match.group(0)
            orig_family_base = Path(base_data_config).stem.split('_seed')[0]
            pert_family_base_config = project_root / "configs/data_generation" / f"{orig_family_base}_{pert_tag}_seed0_config.yml"
            if pert_family_base_config.exists():
                print(f"--- Evaluating on perturbed family from: {pert_family_base_config.name} ---")
                evaluate_multi_seed(
                    trained_model_path=str(final_model_path),
                    data_config_base=str(pert_family_base_config),
                    optimal_config=str(optimal_config_path)
                )

    print("✅ Multi-seed evaluation complete.")

    # --- Step 6: Aggregate and Compare All Results ---
    print("\n[STEP 6/6] Aggregating all results and generating final comparison...")
    aggregate_all_families(optimal_config=str(optimal_config_path))
    print("✅ Aggregation and comparison complete.")

    print("\n🎉 PIPELINE FINISHED SUCCESSFULLY! 🎉")

    
def main():
    """
    Main function to parse arguments and execute the corresponding workflow command.
    """
    # Set the multiprocessing start method to 'spawn' for CUDA compatibility.
    # This must be done at the beginning of the main execution block.
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # The start method can only be set once. If it's already set, we can ignore the error.
        pass

    parser = argparse.ArgumentParser(description="Unified Experiment Workflow Manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- CLI command definitions ---
    # Data generation
    gen = subparsers.add_parser("generate", help="Generate dataset from config")
    gen.add_argument("--config", required=True, type=str, help="Dataset config YAML")
    gen.add_argument("--keep-original-name", action="store_true", help="Do not rename the config after generation")

    gen_multi = subparsers.add_parser("generate-multiseed", help="Multi-seed generation. Generates N evaluation seeds and one training seed by default.")
    gen_multi.add_argument("--config", required=True, type=str, help="Base YAML config file for the dataset family")
    gen_multi.add_argument("--num-seeds",type=int,default=10, help="Number of evaluation seeds to generate (e.g., 0-9). Default: 10")
    gen_multi.add_argument("--start-seed", type=int, default=0, help="Starting random seed for evaluation sets. Default: 0")
    gen_multi.add_argument( "--no-training-seed",action="store_true",help="Do NOT generate the dedicated training dataset (seed 99).")

    pert_multi = subparsers.add_parser("perturb-multiseed", help="Multi-seed perturbation (add perturbation across a family)")
    pert_multi.add_argument("--data-config-base", required=True, type=str, help="Example config file from the family to perturb")
    pert_multi.add_argument("--perturb-config", required=True, type=str, help="Perturbation YAML config")

    # Training
    train_single = subparsers.add_parser("train-single", help="Train a single model on a specific dataset.")
    train_single.add_argument("--data-config", required=True, type=str, help="Path to the specific data config YAML file.")
    train_single.add_argument("--optimal-config", required=True, type=str, help="The optimal config YAML for the training run.")

    # Evaluation
    eval_multi = subparsers.add_parser("evaluate-multiseed", help="Evaluate one model on a multi-seed dataset family")
    eval_multi.add_argument("--trained-model", required=True, type=str, help="Path to the pre-trained model (.pt) file")
    eval_multi.add_argument("--data-config-base", required=True, type=str, help="An example config file from the family to evaluate")
    eval_multi.add_argument("--optimal-config", required=True, type=str, help="The optimal config YAML used to create the model (for architecture info)")

    # Aggregation
    agg = subparsers.add_parser("aggregate", help="Aggregate multi-seed results")
    agg.add_argument("--optimal-config", required=True, type=str, help="Path to one of the final optimal config files used for the runs")

    agg_all = subparsers.add_parser("aggregate-all", help="Aggregate original and all detected perturbed experiment families automatically.")
    agg_all.add_argument("--optimal-config", required=True, type=str, help="Path to the optimal config file of the ORIGINAL family.")

    # Comparison
    comp = subparsers.add_parser("compare-families", help="Compare experiment families (original vs perturbed)")
    comp.add_argument("--original-optimal-config", required=True, type=str, help="Original (unperturbed) optimal config")
    comp.add_argument("--perturbation-tag", required=True, type=str, help="Perturbation tag used in filename for perturbed family")

    # Tuning
    tuneexp = subparsers.add_parser("tune-experiment", help="Launch Optuna hyperparameter tuning job (multi-worker)")
    tuneexp.add_argument("--job", required=True, type=str, help="Job name from configs/experiments.yml")
    tuneexp.add_argument("--data-config", required=False, type=str, help="Path to the training data config file to use (skips interactive prompt).")

    tune_worker = subparsers.add_parser("tune-worker", help="Worker for distributed Optuna tuning")
    tune_worker.add_argument("--data-config", required=True, type=str)
    tune_worker.add_argument("--tuning-config", required=True, type=str)
    tune_worker.add_argument("--base-training-config", required=True, type=str)
    tune_worker.add_argument("--sample-fraction", type=float, default=0.1)
    tune_worker.add_argument("--n-trials", type=int, default=50)

    tune_anal = subparsers.add_parser("tune-analysis", help="Analyse/finalise hyperparameter tuning results")
    tune_anal.add_argument("--data-config", required=True, type=str)
    tune_anal.add_argument("--base-training-config", required=True, type=str)
    tune_anal.add_argument("--sample-fraction", type=float, default=0.1, help="The fraction of data used during tuning, for result reproducibility.")
    tune_anal.add_argument("--non-interactive", action="store_true", help="Automatically select the best trial without user input.")

    # FULL PIPELINE
    pipe = subparsers.add_parser("run-full-pipeline", help="Run the full pipeline automatically.")
    pipe.add_argument("--base-data-config", required=True, type=str, help="Path to the base data config for the experiment.")
    pipe.add_argument("--tuning-job", required=True, type=str, help="Name of the tuning job from experiments.yml.")
    pipe.add_argument("--perturb-config", type=str, default=None, help="Optional: Path to a perturbation config YAML.")

    args = parser.parse_args()

    if args.command == "generate":
        generate_from_config(
            args.config, 
            keep_original_name=args.keep_original_name)
    elif args.command == "generate-multiseed":
        generate_multi_seed(
            args.config, 
            num_seeds=args.num_seeds, 
            start_seed=args.start_seed, 
            generate_training_seed=(not args.no_training_seed))
    elif args.command == "perturb-multiseed":
        perturb_multi_seed(
            args.data_config_base, 
            args.perturb_config)
    elif args.command == "train-single":
        train_single_config(
            args.data_config, 
            args.optimal_config)
    elif args.command == "evaluate-multiseed":
        evaluate_multi_seed(
            args.trained_model, 
            args.data_config_base, 
            args.optimal_config)
    elif args.command == "aggregate":
        aggregate(args.optimal_config)
    elif args.command == "aggregate-all":
        aggregate_all_families(args.optimal_config)
    elif args.command == "compare-families":
        compare_families(
            args.original_optimal_config, 
            args.perturbation_tag)
    elif args.command == "tune-experiment":
        run_experiments(
            args.job, 
            data_config_path=args.data_config)
    elif args.command == "tune-worker":
        run_hyperparameter_tuning(
            args.data_config,
            args.tuning_config,
            args.base_training_config,
            args.sample_fraction,
            args.n_trials
        )
    elif args.command == "tune-analysis":
        run_tuning_analysis(
            args.data_config, 
            args.base_training_config, 
            args.sample_fraction, 
            non_interactive=args.non_interactive)
    elif args.command == "run-full-pipeline":
        run_full_pipeline(
            args.base_data_config,
            args.tuning_job,
            args.perturb_config
        )

if __name__ == "__main__":
    main()
