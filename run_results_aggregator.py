import argparse
import json
import pandas as pd
from pathlib import Path
import sys
import openpyxl
import matplotlib.pyplot as plt

# Ensure the src directory is in the Python path for utils
try:
    from src.data_generator_module.utils import find_project_root
    from src.data_generator_module.plotting_style import apply_custom_plot_style
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src.data_generator_module.utils import find_project_root
    from src.data_generator_module.plotting_style import apply_custom_plot_style

def find_experiment_family(models_dir: Path, base_experiment_name: str):
    """
    Finds all metric files belonging to a specific experiment family.
    """
    pattern = f"{base_experiment_name.split('_seed')[0]}*_metrics.json"
    metric_files = list(models_dir.glob(pattern))
    return metric_files

def main():
    """
    Aggregates results from multi-seed training runs, saves them to a
    spreadsheet, and generates summary plots.
    """
    parser = argparse.ArgumentParser(
        description="Aggregate multi-seed experiment results and generate plots."
    )
    parser.add_argument(
        "--optimal-config",
        "-oc",
        type=str,
        required=True,
        help="Path to one of the final optimal config files used for the runs (e.g., ..._optimal.yml)."
    )
    args = parser.parse_args()

    apply_custom_plot_style()
    project_root = Path(find_project_root())
    models_dir = project_root / "models"
    reports_dir = project_root / "reports"
    spreadsheet_path = reports_dir / "experiment_tracking.xlsx"
    
    # --- 1. Identify the Experiment Family ---
    optimal_config_path = Path(args.optimal_config)
    if not optimal_config_path.is_absolute():
        optimal_config_path = project_root / optimal_config_path

    if not optimal_config_path.exists():
        print(f"Error: Optimal config file not found at '{optimal_config_path}'")
        return

    base_experiment_name = optimal_config_path.stem.replace('_config', '')
    
    print(f"--- Aggregating results for experiment family: {base_experiment_name} ---")

    metric_files = find_experiment_family(models_dir, base_experiment_name)

    if not metric_files:
        print(f"Error: No metric files found for this experiment family in '{models_dir}'.")
        return
        
    print(f"Found {len(metric_files)} metric files to aggregate.")

    # --- 2. Load and Aggregate Metrics ---
    all_metrics = []
    for f in metric_files:
        with open(f, 'r') as file:
            metrics = json.load(file)
            seed = f.stem.split('_seed')[-1].split('_')[0]
            metrics['seed'] = int(seed)
            all_metrics.append(metrics)

    results_df = pd.DataFrame(all_metrics)
    results_df = results_df[['seed'] + [col for col in results_df.columns if col != 'seed']]
    
    # --- 3. Calculate Summary Statistics ---
    summary_stats = results_df.drop(columns=['seed']).agg(['mean', 'std']).T
    summary_stats['mean'] = summary_stats['mean'].round(4)
    summary_stats['std'] = summary_stats['std'].round(4)
    
    print("\n--- Aggregated Results ---")
    print(results_df.round(4).to_string(index=False))
    
    print("\n--- Summary Statistics (Mean & Std Dev) ---")
    print(summary_stats)
    
    # --- 4. Save to Spreadsheet ---
    sheet_name = f"{base_experiment_name}_summary"
    print(f"\nSaving aggregated results to sheet '{sheet_name[:31]}' in '{spreadsheet_path}'...")
    
    try:
        with pd.ExcelWriter(spreadsheet_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            results_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
            summary_stats.to_excel(writer, sheet_name=sheet_name[:31], startrow=len(results_df) + 2)
        print("Successfully saved results to spreadsheet.")
    except Exception as e:
        print(f"An error occurred while writing to the Excel file: {e}")

    # --- 5. Generate Visualisations ---
    plot_dir = reports_dir / "figures" / "aggregated_summaries"
    plot_dir.mkdir(exist_ok=True)
    
    # A. Generate Faceted Line Plots
    plot_data = summary_stats.reset_index().rename(columns={'index': 'Metric'})
    
    # Plot 1: Main metrics 
    main_metrics_df = plot_data[plot_data['Metric'] != 'Test Loss (BCE)']
    plt.errorbar(main_metrics_df['Metric'], main_metrics_df['mean'], yerr=main_metrics_df['std'], fmt='-o', capsize=5, label='Mean ± Std Dev')
    for _, row in main_metrics_df.iterrows():
        plt.text(
            row['Metric'], 
            row['mean'], 
            f" {row['mean']:.4f} ", 
            ha='center', 
            va='bottom', # Position text above the point
            fontsize=10, 
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1) # Add background
        )
        
    plt.title(f'Aggregated Performance Metrics\n({base_experiment_name})', fontsize=16)
    plt.ylabel('Score')
    plt.xlabel('Metric')
    plt.xticks(rotation=45, ha="right")
    
    # Adjust y-axis limits to give labels space
    min_y = main_metrics_df['mean'].min() - main_metrics_df['std'].max() * 2
    max_y = main_metrics_df['mean'].max() + main_metrics_df['std'].max() * 2
    plt.ylim(min_y, max_y)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    faceted_plot_path_main = plot_dir / f"{base_experiment_name}_summary_faceted_main.pdf"
    plt.savefig(faceted_plot_path_main)
    print(f"\nSaved main metrics plot to: {faceted_plot_path_main}")
    plt.close()

    print("\n--- Aggregation complete. ---")

if __name__ == "__main__":
    main()
