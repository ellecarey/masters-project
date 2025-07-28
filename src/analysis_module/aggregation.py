import pandas as pd
import json
from pathlib import Path
from src.utils.filenames import metrics_filename, model_filename, config_filename

def aggregate_family_results(optimal_config_path: str):
    """Aggregates multi-seed experiment results into a DataFrame and summary stats."""
    optimal_config = Path(optimal_config_path)
    project_root = optimal_config.parents[2]  # Adjust as needed
    base_name = optimal_config.stem.replace('_config', '')
    models_dir = project_root / "models"
    reports_dir = project_root / "reports"
    pattern = f"{base_experiment_name.rsplit('_seed', 1)[0]}*_metrics.json"
    metric_files = list(models_dir.glob(pattern))
    all_metrics = []
    for f in metric_files:
        with open(f, 'r') as file:
            metrics = json.load(file)
            seed = f.stem.split('_seed')[-1].split('_')[0]
            metrics['seed'] = int(seed)
            all_metrics.append(metrics)
    if not all_metrics:
        return pd.DataFrame(), pd.DataFrame(), base_name
    results_df = pd.DataFrame(all_metrics)
    summary_stats = results_df.drop(columns=['seed']).agg(['mean', 'std']).T
    return results_df, summary_stats, base_name
