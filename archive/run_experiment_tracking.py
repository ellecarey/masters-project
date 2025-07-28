import os
import pandas as pd
from pathlib import Path
import optuna
from datetime import datetime
import json
import sys

try:
    from src.data_generator_module.utils import find_project_root
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src.data_generator_module.utils import find_project_root
    
def parse_filename(filepath: Path):
    """Parses dataset and model names from a standardized filename."""
    # Example filename: n10000_f_init5_..._seed42_mlp_001_tuning.db
    # Or: n10000_f_init5_..._seed42_mlp_001_final_model.pt
    parts = filepath.stem.split('_')
    
    # Find the model name (e.g., 'mlp_001')
    model_name = None
    for part in parts:
        if part.startswith("mlp"): # Or a more generic model prefix
            model_name = part
            break
            
    if not model_name:
        return None, None

    dataset_name = filepath.stem.split(f"_{model_name}")[0]
    
    return dataset_name, model_name

def update_experiment_registry():
    """
    Scans for completed experiments and updates the tracking spreadsheet,
    now with auditing to remove deleted experiments.
    """
    project_root = Path(find_project_root())
    output_file = project_root / "experiment_tracking.xlsx"

    # --- 1. Load existing registry or create a new one ---
    if output_file.exists():
        registry_df = pd.read_excel(output_file, sheet_name="Experiment Registry")
    else:
        registry_df = pd.DataFrame(columns=[
            "Experiment ID", "Dataset Name", "Model Name", "Experiment Type",
            "Status", "Date Completed", "Key Metric (AUC)", "Path to Config",
            "Path to Artifact", "Path to Figures"
        ])

    # --- 2. Audit and Remove Deleted Experiments ---
    if not registry_df.empty:
        print("\n--- Auditing existing experiment registry ---")
        initial_count = len(registry_df)
        
        # Check if artifact paths still exist
        registry_df['artifact_exists'] = registry_df['Path to Artifact'].apply(
            lambda p: (project_root / p).exists()
        )
        
        deleted_artifacts = registry_df[~registry_df['artifact_exists']]
        
        if not deleted_artifacts.empty:
            print(f"Found {len(deleted_artifacts)} deleted artifacts to remove from the registry:")
            for path in deleted_artifacts['Path to Artifact']:
                print(f" - Removing record for: {path}")
            
            # Filter the registry to keep only existing artifacts
            registry_df = registry_df[registry_df['artifact_exists']].copy()
        else:
            print("No deleted artifacts found. Registry is clean.")

        # Clean up the helper column
        registry_df.drop(columns=['artifact_exists'], inplace=True)
        
        if len(registry_df) < initial_count:
            # Re-index the Experiment IDs to keep them sequential
            registry_df['Experiment ID'] = [f"EXP{i+1:03d}" for i in range(len(registry_df))]

    existing_artifacts = set(registry_df["Path to Artifact"])
    print(f"Found {len(existing_artifacts)} valid experiments in registry after audit.")

    new_experiments = []

    # --- 3. Scan for Tuning Experiments (from .db files) ---
    print("\n--- Scanning for new Tuning Experiments ---")
    reports_dir = project_root / "reports"
    for db_file in reports_dir.glob("*_tuning.db"):
        artifact_path = str(db_file.relative_to(project_root))
        if artifact_path in existing_artifacts:
            continue
        
        print(f" + Found new tuning run: {db_file.name}")
        dataset_name, model_name = parse_filename(db_file)
        if not dataset_name or not model_name:
            continue
            
        try:
            study = optuna.load_study(
                study_name=db_file.stem.replace('_tuning', ''),
                storage=f"sqlite:///{db_file}"
            )
            best_value = study.best_value if study.best_trial else None
            
            new_experiments.append({
                "Dataset Name": dataset_name, "Model Name": model_name,
                "Experiment Type": "Tuning", "Status": "Completed",
                "Date Completed": datetime.fromtimestamp(db_file.stat().st_mtime).strftime('%Y-%m-%d'),
                "Key Metric (AUC)": best_value,
                "Path to Config": f"configs/tuning/{model_name}.yml",
                "Path to Artifact": artifact_path,
                "Path to Figures": f"reports/figures/{dataset_name}/{model_name}"
            })
        except Exception as e:
            print(f" - Could not process {db_file.name}: {e}")

    # --- 4. Scan for Final Training Experiments (from ..._metrics.json files) ---
    print("\n--- Scanning for new Final Training Experiments ---")
    models_dir = project_root / "models"
    for metrics_file in models_dir.glob("*_metrics.json"):
        # Use the metrics file as the primary artifact to track
        artifact_path = str(metrics_file.relative_to(project_root))
        if artifact_path in existing_artifacts:
            continue

        print(f" + Found new final training run: {metrics_file.name}")
        
        # Infer names from the metrics file
        dataset_name, model_name = parse_filename(metrics_file)
        if not dataset_name or not model_name:
            continue
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        test_auc = metrics.get("AUC")
        
        config_path = project_root / "configs/training/generated" / f"{metrics_file.stem.replace('_metrics', '')}.yml"
        
        new_experiments.append({
            "Dataset Name": dataset_name, "Model Name": model_name,
            "Experiment Type": "Final Training", "Status": "Completed",
            "Date Completed": datetime.fromtimestamp(metrics_file.stat().st_mtime).strftime('%Y-%m-%d'),
            "Key Metric (AUC)": test_auc,
            "Path to Config": str(config_path.relative_to(project_root)) if config_path.exists() else "N/A",
            "Path to Artifact": artifact_path,
            "Path to Figures": f"reports/figures/{dataset_name}/{model_name}_final"
        })

    # --- 5. Update and Save the Spreadsheet ---
    if new_experiments:
        new_df = pd.DataFrame(new_experiments)
        updated_df = pd.concat([registry_df, new_df], ignore_index=True)
        updated_df["Experiment ID"] = [f"EXP{i+1:03d}" for i in range(len(updated_df))]
        
        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            updated_df.to_excel(writer, sheet_name="Experiment Registry", index=False)
        print(f"\n{len(new_experiments)} new experiment(s) added. Registry updated: {output_file}")
    else:
        print("\nNo new experiments found. Registry is up-to-date.")

if __name__ == "__main__":
    update_experiment_registry()
