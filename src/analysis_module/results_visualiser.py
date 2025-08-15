import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import re

from src.data_generator_module.plotting_style import apply_custom_plot_style
from src.data_generator_module.utils import find_project_root

def categorize_perturbation(perturbation_string: str) -> str:
    """
    Categorizes a detailed perturbation filename string into a broad category.
    
    Args:
        perturbation_string: The 'perturbation' value from the tracking sheet.

    Returns:
        A string representing the general perturbation category.
    """
    if "original" in perturbation_string:
        return "Baseline"
    
    # Use regex to count the number of individual perturbation definitions
    # e.g., 'pert_f0n_scale-1p5_pert_f1n_by0p5s' has two definitions.
    num_perts = len(re.findall(r'pert_f\d+', perturbation_string))

    if num_perts > 1:
        return "Complex Combined"

    is_corr = 'corr' in perturbation_string.lower()
    is_scale = 'scale' in perturbation_string.lower()
    is_shift = 'by' in perturbation_string.lower() and 's' in perturbation_string.lower()

    if is_corr and is_scale:
        return "Correlated Scale"
    if is_corr and is_shift:
        return "Correlated Shift"
    if not is_corr and is_scale:
        return "Individual Scale"
    if not is_corr and is_shift:
        return "Individual Shift"
    
    return "Other" # Fallback for any uncategorized perturbations

def plot_baseline_performance(df: pd.DataFrame, save_path: Path):
    """
    Generates and saves a bar plot for the baseline model performance.
    
    Args:
        df: The global experiment tracking DataFrame.
        save_path: The path to save the generated plot.
    """
    baseline_df = df[df['perturbation'] == 'original'].copy()
    
    fig, ax = plt.subplots()
    
    # Create a more descriptive label for the x-axis from existing columns
    baseline_df['Experiment Label'] = baseline_df.apply(
        lambda row: f"{row['n_samples']/1000000 if row['n_samples'] >= 1000000 else row['n_samples']/1000}{'M' if row['n_samples'] >= 1000000 else 'K'} samples, {row['n_features']} features, sep={row['separation']}",
        axis=1
    )
    
    sns.barplot(x='Experiment Label', y='AUC_mean', data=baseline_df, palette="viridis", ax=ax)
    
    # Add error bars for standard deviation
    ax.errorbar(x=range(len(baseline_df.index)), y=baseline_df['AUC_mean'], yerr=baseline_df['AUC_std'], fmt='none', c='black', capsize=5)

    ax.set_title('Figure 1: Baseline Model Performance on Unperturbed Datasets', fontweight='bold')
    ax.set_ylabel('Mean AUC Score')
    ax.set_xlabel('Dataset Configuration')
    plt.xticks(rotation=15, ha='right')
    ax.set_ylim(0.9, 1.0) # Set y-axis limit to better visualize differences
    
    fig.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to: {save_path}")

def plot_perturbation_modality_impact(df: pd.DataFrame, save_path: Path):
    """
    Generates and saves a bar plot comparing the impact of different perturbation modalities.

    Args:
        df: The global experiment tracking DataFrame.
        save_path: The path to save the generated plot.
    """
    # 1. Get baseline AUC for each family
    baselines = df[df['perturbation'] == 'original'].set_index('experiment_family')['AUC_mean']
    
    # 2. Filter for perturbed runs
    perturbed_df = df[df['perturbation'] != 'original'].copy()
    
    # 3. Extract base family name to map baseline AUCs
    perturbed_df['base_family'] = perturbed_df['experiment_family'].apply(lambda x: x.split('_pert_')[0])
    perturbed_df['baseline_auc'] = perturbed_df['base_family'].map(baselines)
    
    # 4. Calculate ΔAUC and categorize
    perturbed_df.dropna(subset=['baseline_auc'], inplace=True) # Drop runs without a baseline
    perturbed_df['delta_auc'] = perturbed_df['AUC_mean'] - perturbed_df['baseline_auc']
    perturbed_df['perturbation_category'] = perturbed_df['perturbation'].apply(categorize_perturbation)
    
    # 5. Aggregate results
    summary = perturbed_df.groupby('perturbation_category')['delta_auc'].mean().reset_index()
    summary = summary.rename(columns={'delta_auc': 'Average ΔAUC (Mean)'})
    
    # 6. Plotting
    fig, ax = plt.subplots()
    sns.barplot(x='perturbation_category', y='Average ΔAUC (Mean)', data=summary, palette="plasma", ax=ax)
    
    # Add labels to the bars
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.4f'), 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'center', 
                       xytext = (0, -15 if p.get_height() < 0 else 9), 
                       textcoords = 'offset points')

    ax.set_title('Figure 2: Impact of Perturbation Modality on Model Performance', fontweight='bold')
    ax.set_ylabel('Change in Mean AUC (ΔAUC)')
    ax.set_xlabel('Perturbation Type')
    ax.axhline(0, color='grey', lw=1.5, linestyle='--')
    
    fig.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to: {save_path}")

def plot_dataset_characteristics_robustness(df: pd.DataFrame, save_path: Path, perturbation_category: str = "Correlated Scale"):
    """
    Generates and saves a bar plot showing the influence of dataset characteristics on robustness.
    
    Args:
        df: The global experiment tracking DataFrame.
        save_path: The path to save the generated plot.
        perturbation_category: The category of perturbation to analyze.
    """
    # 1. Get baselines
    baselines = df[df['perturbation'] == 'original'].set_index('experiment_family')[['AUC_mean', 'n_samples', 'n_features', 'separation']]
    
    # 2. Filter for perturbed runs and categorize
    perturbed_df = df[df['perturbation'] != 'original'].copy()
    perturbed_df['perturbation_category'] = perturbed_df['perturbation'].apply(categorize_perturbation)
    
    # 3. Select only the runs matching the desired category
    target_df = perturbed_df[perturbed_df['perturbation_category'] == perturbation_category].copy()
    
    # 4. Calculate ΔAUC
    target_df['base_family'] = target_df['experiment_family'].apply(lambda x: x.split('_pert_')[0])
    target_df['baseline_auc'] = target_df['base_family'].map(baselines['AUC_mean'])
    target_df.dropna(subset=['baseline_auc'], inplace=True)
    target_df['delta_auc'] = target_df['AUC_mean'] - target_df['baseline_auc']
    
    # 5. Aggregate results (average ΔAUC for each base family)
    summary = target_df.groupby('base_family').agg(
        delta_auc_mean=('delta_auc', 'mean'),
    ).reset_index()
    
    # Merge back original characteristics for labeling
    summary = summary.merge(baselines.reset_index(), left_on='base_family', right_on='experiment_family')

    # 6. Plotting
    fig, ax = plt.subplots()
    summary['Experiment Label'] = summary.apply(
        lambda row: f"{row['n_samples']/1000000 if row['n_samples'] >= 1000000 else row['n_samples']/1000}{'M' if row['n_samples'] >= 1000000 else 'K'} samples, {row['n_features']} features, sep={row['separation']}",
        axis=1
    )
    
    sns.barplot(x='Experiment Label', y='delta_auc_mean', data=summary, palette="cividis", ax=ax)

    ax.set_title(f'Figure 3: Influence of Dataset Characteristics on Robustness (Perturbation: {perturbation_category})', fontweight='bold')
    ax.set_ylabel('Change in Mean AUC (ΔAUC) from Perturbation')
    ax.set_xlabel('Dataset Configuration')
    plt.xticks(rotation=15, ha='right')
    ax.axhline(0, color='grey', lw=1.5, linestyle='--')
    
    fig.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to: {save_path}")

def main(tracking_sheet_path: Path):
    """Main function to load data and generate all plots."""
    apply_custom_plot_style()
    
    try:
        df = pd.read_csv(tracking_sheet_path)
        # Convert relevant columns to numeric, coercing errors
        for col in ['AUC_mean', 'AUC_std', 'n_samples', 'n_features', 'separation']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['AUC_mean', 'AUC_std'], inplace=True) # Drop rows where key metrics are missing
    except FileNotFoundError:
        print(f"Error: Tracking sheet not found at {tracking_sheet_path}")
        return

    # Create a dedicated directory for these final plots
    project_root = Path(find_project_root())
    save_dir = project_root / "reports" / "figures" / "final_results"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nPlots will be saved in: {save_dir}")

    print("\nGenerating Figure 1: Baseline Performance...")
    plot_baseline_performance(df, save_dir / "figure_1_baseline_performance.pdf")
    
    print("Generating Figure 2: Perturbation Modality Impact...")
    plot_perturbation_modality_impact(df, save_dir / "figure_2_perturbation_impact.pdf")
    
    print("Generating Figure 3: Dataset Robustness (for Correlated Scale)...")
    plot_dataset_characteristics_robustness(df, save_dir / "figure_3_robustness_vs_characteristics.pdf", perturbation_category="Correlated Scale")
    
    print("\nAll plots generated and saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate final result plots from the global tracking sheet.")
    parser.add_argument(
        "--sheet",
        type=str,
        default="reports/global_experiment_tracking.csv",
        help="Path to the global experiment tracking CSV file relative to project root."
    )
    args = parser.parse_args()
    
    try:
        project_root = Path(find_project_root())
        full_sheet_path = project_root / args.sheet
        main(full_sheet_path)
    except Exception as e:
        print(f"An error occurred: {e}")