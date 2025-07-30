import argparse
from src.data_generator_module.generator_cli import (
    generate_from_config,
    generate_multi_seed,
    perturb_multi_seed
)
from src.training_module.training_cli import train_multi_seed, evaluate_multi_seed, train_single_config
from src.training_module.tuning_cli import (
    run_experiments,
    run_hyperparameter_tuning,
    run_tuning_analysis,
)
from src.analysis_module.analysis_cli import (
    aggregate,
    compare_families,
    aggregate_all_families
)


def main():
    parser = argparse.ArgumentParser(description="Unified Experiment Workflow Manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

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

    train_multi = subparsers.add_parser("train-multiseed", help="Multi-seed training")
    train_multi.add_argument("--data-config-base", required=True, type=str, help="Example config file from family for training")
    train_multi.add_argument("--optimal-config", required=True, type=str, help="Optimal config YAML for the run")

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

    tune_worker = subparsers.add_parser("tune-worker", help="Worker for distributed Optuna tuning")
    tune_worker.add_argument("--data-config", required=True, type=str)
    tune_worker.add_argument("--tuning-config", required=True, type=str)
    tune_worker.add_argument("--base-training-config", required=True, type=str)
    tune_worker.add_argument("--sample-fraction", type=float, default=0.8)
    tune_worker.add_argument("--n-trials", type=int, default=50)

    tune_anal = subparsers.add_parser("tune-analysis", help="Analyse/finalise hyperparameter tuning results")
    tune_anal.add_argument("--data-config", required=True, type=str)
    tune_anal.add_argument("--base-training-config", required=True, type=str)


    args = parser.parse_args()

    if args.command == "generate":
        generate_from_config(args.config, keep_original_name=args.keep_original_name)
    elif args.command == "generate-multiseed":
        generate_multi_seed(args.config, num_seeds=args.num_seeds, start_seed=args.start_seed, generate_training_seed=(not args.no_training_seed))
    elif args.command == "perturb-multiseed":
        perturb_multi_seed(args.data_config_base, args.perturb_config)
    elif args.command == "train-single":
        train_single_config(args.data_config, args.optimal_config)
    elif args.command == "train-multiseed":
        train_multi_seed(args.data_config_base, args.optimal_config)
    elif args.command == "evaluate-multiseed":
        evaluate_multi_seed(args.trained_model, args.data_config_base, args.optimal_config)
    elif args.command == "aggregate":
        aggregate(args.optimal_config)
    elif args.command == "aggregate-all":
        aggregate_all_families(args.optimal_config)
    elif args.command == "compare-families":
        compare_families(args.original_optimal_config, args.perturbation_tag)
    elif args.command == "tune-experiment":
        run_experiments(args.job)
    elif args.command == "tune-worker":
        run_hyperparameter_tuning(
            args.data_config,
            args.tuning_config,
            args.base_training_config,
            args.sample_fraction,
            args.n_trials
        )
    elif args.command == "tune-analysis":
        run_tuning_analysis(args.data_config, args.base_training_config)


if __name__ == "__main__":
    main()
