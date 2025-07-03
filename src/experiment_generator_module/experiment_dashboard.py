"""
Dashboard to monitor experiment progress and results.
"""

import yaml
import pandas as pd
from pathlib import Path
import os


def analyze_experiments():
    """Analyze completed experiments."""

    # Load experiment index
    index_path = Path("configs/systematic_experiments/experiment_index.yaml")
    with open(index_path, "r") as f:
        index = yaml.safe_load(f)

    experiments = index["experiments"]

    # Check which experiments have been completed
    data_dir = Path("data")
    results = []

    for exp in experiments:
        # Check if dataset exists
        dataset_pattern = f"{exp['name']}_dataset.csv"
        dataset_exists = (data_dir / dataset_pattern).exists()

        if dataset_exists:
            dataset_size = (data_dir / dataset_pattern).stat().st_size / (
                1024 * 1024
            )  # MB
        else:
            dataset_size = 0

        results.append(
            {
                "experiment_id": exp["experiment_id"],
                "name": exp["name"],
                "n_features": exp["n_features"],
                "n_samples": exp["n_samples"],
                "continuous": exp["continuous"],
                "discrete": exp["discrete"],
                "completed": dataset_exists,
                "dataset_size_mb": round(dataset_size, 2),
            }
        )

    df = pd.DataFrame(results)

    # Print summary
    print("EXPERIMENT DASHBOARD")
    print("=" * 60)
    print(f"Total experiments: {len(df)}")
    print(f"Completed: {df['completed'].sum()}")
    print(f"Remaining: {(~df['completed']).sum()}")
    print(f"Total data size: {df['dataset_size_mb'].sum():.2f} MB")

    # Completion by feature count
    print("\nCompletion by feature count:")
    completion_by_features = df.groupby("n_features")["completed"].agg(["count", "sum"])
    completion_by_features["percentage"] = (
        completion_by_features["sum"] / completion_by_features["count"] * 100
    ).round(1)
    print(completion_by_features)

    # Completion by sample size
    print("\nCompletion by sample size:")
    completion_by_samples = df.groupby("n_samples")["completed"].agg(["count", "sum"])
    completion_by_samples["percentage"] = (
        completion_by_samples["sum"] / completion_by_samples["count"] * 100
    ).round(1)
    print(completion_by_samples)

    # Show incomplete experiments
    incomplete = df[~df["completed"]]
    if len(incomplete) > 0:
        print("\nIncomplete experiments:")
        for _, exp in incomplete.head(10).iterrows():
            print(f"  {exp['name']}")
        if len(incomplete) > 10:
            print(f"  ... and {len(incomplete) - 10} more")

    return df


if __name__ == "__main__":
    analyze_experiments()
