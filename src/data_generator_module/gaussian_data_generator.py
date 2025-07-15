"""
Gaussian Data Generator for Pure Signal vs Noise Classification

This module provides functionality to generate synthetic datasets where samples
are classified based purely on their originating Gaussian distribution.
"""

from typing import Dict, Optional, List
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from .validators import DataGeneratorValidators


class GaussianDataGenerator:
    """
    A class for generating datasets with pure Gaussian signal vs noise classification.

    Samples are labeled based purely on which Gaussian distribution they originate from:
    - Signal samples: drawn from signal Gaussian distribution (target = 1)
    - Noise samples: drawn from noise Gaussian distribution (target = 0)
    """

    def __init__(self, n_samples: int, n_features: int, random_state: int):
        """
        Initialize the GaussianDataGenerator.

        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        n_features : int
            Number of features to generate
        random_state : int
            Random seed for reproducibility
        """
        DataGeneratorValidators.validate_init_parameters(n_samples, n_features)

        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state
        self.data: Optional[pd.DataFrame] = None
        self.feature_types: Dict[str, str] = {}
        self.feature_parameters: Dict[str, Dict] = {}

    def generate_features(
        self,
        feature_parameters: Dict[str, Dict],
        feature_types: Dict[str, str],
        n_features_to_generate: Optional[int] = None,
    ):
        """
        Generate features based on specified parameters.

        Parameters:
        -----------
        feature_parameters : Dict[str, Dict]
            Dictionary containing feature parameters with 'mean' and 'std' keys
        feature_types : Dict[str, str]
            Dictionary mapping feature names to their types ('continuous' or 'discrete')
        n_features_to_generate : Optional[int]
            Number of features to generate (defaults to all provided parameters)

        Returns:
        --------
        self : GaussianDataGenerator
            Returns self for method chaining
        """
        DataGeneratorValidators.validate_feature_parameters(feature_parameters)
        DataGeneratorValidators.validate_feature_types(feature_types)

        rng = np.random.RandomState(self.random_state)

        if n_features_to_generate is None:
            n_features_to_generate = len(feature_parameters)

        generated_data = {}
        for i, (feature_name, params) in enumerate(feature_parameters.items()):
            if i >= n_features_to_generate:
                break

            feature_data = rng.normal(params["mean"], params["std"], self.n_samples)

            feature_type = feature_types.get(feature_name, "continuous")
            if feature_type == "discrete":
                feature_data = np.round(feature_data)

            generated_data[feature_name] = feature_data

        self.data = pd.DataFrame(generated_data)
        self.feature_parameters.update(feature_parameters)
        self.feature_types.update(feature_types)

        print(
            f"Generated {len(generated_data)} features with {self.n_samples} samples each."
        )
        return self

    def create_feature_based_signal_noise_classification(
        self,
        signal_distribution_params: Dict[str, float],
        noise_distribution_params: Dict[str, float],
        store_for_visualization: bool = False,
    ):
        """
        Create binary classification where labels are assigned based on originating
        Gaussian distribution, but observation values are NOT stored as a feature.

        Parameters:
        -----------
        signal_distribution_params : Dict[str, float]
            Parameters for signal Gaussian: {"mean": float, "std": float}
        noise_distribution_params : Dict[str, float]
            Parameters for noise Gaussian: {"mean": float, "std": float}
        signal_ratio : float
            Proportion of samples that should be signal class
        store_for_visualization : bool
            Whether to temporarily store Gaussian observations for visualization

        Returns:
        --------
        self : GaussianDataGenerator
            Returns self for method chaining
        """

        if self.data is None:
            raise ValueError("No data generated. Call generate_features() first.")

        signal_ratio = 0.5
        if not 0 <= signal_ratio <= 1:
            raise ValueError("signal_ratio must be between 0 and 1.")

        n_signal_samples = int(self.n_samples * signal_ratio)
        n_noise_samples = self.n_samples - n_signal_samples

        rng = np.random.RandomState(self.random_state)

        # Generate signal and noise observations (internal use only)
        signal_observations = rng.normal(
            signal_distribution_params["mean"],
            signal_distribution_params["std"],
            n_signal_samples,
        )

        noise_observations = rng.normal(
            noise_distribution_params["mean"],
            noise_distribution_params["std"],
            n_noise_samples,
        )

        # Create labels based purely on originating distribution
        signal_labels = np.ones(n_signal_samples, dtype=int)  # Signal = 1
        noise_labels = np.zeros(n_noise_samples, dtype=int)  # Noise = 0

        # Combine labels only (observations are discarded by default)
        all_labels = np.concatenate([signal_labels, noise_labels])

        # Shuffle to randomize order
        indices = rng.permutation(self.n_samples)
        all_labels = all_labels[indices]

        # Store ONLY the binary target - no observation feature by default
        self.data["target"] = all_labels

        # Optionally store observations for visualization
        if store_for_visualization:
            all_observations = np.concatenate([signal_observations, noise_observations])
            all_observations = all_observations[indices]
            self.data["_temp_observations"] = all_observations

        # Store metadata for analysis
        self.feature_based_metadata = {
            "signal_distribution": signal_distribution_params,
            "noise_distribution": noise_distribution_params,
            "signal_ratio": signal_ratio,
            "actual_signal_ratio": all_labels.mean(),
            "approach": "feature_based_learning",
            "has_temp_observations": store_for_visualization,
        }

        print("Created feature-based signal/noise classification:")
        print(f"  Signal samples: {(all_labels == 1).sum()} (target = 1)")
        print(f"  Noise samples: {(all_labels == 0).sum()} (target = 0)")

        if store_for_visualization:
            print("  Temporary observations stored for visualization")

        return self

    def visualise_signal_noise_by_features(self, save_path: Optional[str] = None):
        """
        visualise signal vs noise distributions for each individual feature.
        Shows overlapping histograms of signal and noise samples for each feature.
        """
        if self.data is None or "target" not in self.data.columns:
            raise ValueError("No signal/noise data generated.")

        # Get feature columns (exclude target)
        feature_columns = [
            col for col in self.data.columns if col.startswith("feature_")
        ]

        if not feature_columns:
            raise ValueError("No features found in dataset.")

        # Separate signal and noise samples
        signal_data = self.data[self.data["target"] == 1]
        noise_data = self.data[self.data["target"] == 0]

        # Create subplots
        n_features = len(feature_columns)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

        # Handle single subplot case
        if n_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        # Plot each feature
        for i, feature in enumerate(feature_columns):
            ax = axes[i]

            # Plot overlapping histograms
            ax.hist(
                noise_data[feature],
                bins=30,
                alpha=0.7,
                label=f"Noise (n={len(noise_data)})",
                color="red",
                density=True,
                edgecolor="black",
                linewidth=0.5,
            )

            ax.hist(
                signal_data[feature],
                bins=30,
                alpha=0.7,
                label=f"Signal (n={len(signal_data)})",
                color="blue",
                density=True,
                edgecolor="black",
                linewidth=0.5,
            )

            # Customize subplot
            ax.set_title(f"Distribution of {feature}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add statistics text box
            signal_mean = signal_data[feature].mean()
            signal_std = signal_data[feature].std()
            noise_mean = noise_data[feature].mean()
            noise_std = noise_data[feature].std()

            stats_text = f"""Signal: mean={signal_mean:.2f}, std={signal_std:.2f}
    Noise: mean={noise_mean:.2f}, std={noise_std:.2f}
    Separation: {abs(signal_mean - noise_mean):.2f}"""

            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        # Overall title and layout
        fig.suptitle(
            "Signal vs Noise Distributions by Feature", fontsize=16, weight="bold"
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Feature-wise signal/noise visualization saved to {save_path}")
        else:
            plt.show()

        return self

    def get_data(self) -> Optional[pd.DataFrame]:
        """Get the generated data."""
        return self.data

    def get_feature_information(self):
        """Get detailed information about the features in the dataset."""
        if self.data is None:
            raise ValueError("No data generated yet.")

        info = {}
        for feature in self.data.columns:
            if feature.startswith("feature_") or feature in ["observation", "target"]:
                stats = {
                    "mean": self.data[feature].mean(),
                    "std": self.data[feature].std(),
                    "min": self.data[feature].min(),
                    "max": self.data[feature].max(),
                    "type": self.feature_types.get(feature, "unknown"),
                }
                info[feature] = stats

        return info

    def save_data(self, file_path: str):
        """Save the generated DataFrame to a CSV file."""
        if self.data is None:
            print("Warning: No data to save. Please generate data first.")
            return self

        try:
            output_dir = os.path.dirname(file_path)
            os.makedirs(output_dir, exist_ok=True)

            self.data.to_csv(file_path, index=False)
            print(f"Data successfully saved to {file_path}")
        except Exception as e:
            print(f"Error: Could not save data to {file_path}. Reason: {e}")

        return self

    def get_data_summary(self):
        """Get a summary of the generated data."""
        if self.data is None:
            print("No data generated yet.")
            return None

        print(f"Dataset shape: {self.data.shape}")
        print(f"Features: {list(self.data.columns)}")
        print(f"Feature types: {self.feature_types}")
        print("\nSummary statistics:")
        return self.data.describe()

    def visualise_features(
        self,
        features: List[str],
        max_features_to_show: int,
        n_bins: int,
        save_to_path: Optional[str],
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
    ):
        """
        visualise the distribution of selected features.

        Parameters:
        -----------
        features : List[str]
            List of feature names to visualise
        max_features_to_show : int
            Maximum number of features to display
        n_bins : int
            Number of bins for histograms
        save_to_path : Optional[str]
            Path to save the plot (if None, displays the plot)
        title : Optional[str]
            Main title for the plot
        subtitle : Optional[str]
            Subtitle for the plot

        Returns:
        --------
        self : GaussianDataGenerator
            Returns self for method chaining
        """
        # Add validation call
        DataGeneratorValidators.validate_visualisation_parameters(
            features, max_features_to_show, n_bins, self.data
        )

        # Limit the number of features to show
        features_to_plot = features[:max_features_to_show]

        # Create subplots
        n_features = len(features_to_plot)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows + 1))

        # Handle case where there's only one subplot
        if n_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        if title:
            fig.suptitle(title, fontsize=20, weight="bold")

        if subtitle:
            plt.figtext(0.5, 0.92, subtitle, ha="center", fontsize=14, style="italic")

        # Plot histograms
        for i, feature in enumerate(features_to_plot):
            ax = axes[i]

            # Special handling for observation feature to show signal/noise separation
            if feature == "observation" and "target" in self.data.columns:
                # Plot signal and noise separately with different colors
                signal_data = self.data[self.data["target"] == 1][feature]
                noise_data = self.data[self.data["target"] == 0][feature]

                ax.hist(
                    noise_data,
                    bins=n_bins,
                    alpha=0.7,
                    color="red",
                    label=f"Noise (n={len(noise_data)})",
                    edgecolor="black",
                    linewidth=0.5,
                )
                ax.hist(
                    signal_data,
                    bins=n_bins,
                    alpha=0.7,
                    color="blue",
                    label=f"Signal (n={len(signal_data)})",
                    edgecolor="black",
                    linewidth=0.5,
                )
                ax.legend()
            else:
                # Standard histogram for regular features
                ax.hist(self.data[feature], bins=n_bins, alpha=0.7, edgecolor="black")

            ax.set_title(f"Distribution of {feature}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        fig.tight_layout(rect=[0, 0.03, 1, 0.92])

        if save_to_path:
            output_dir = os.path.dirname(save_to_path)
            os.makedirs(output_dir, exist_ok=True)

            plt.savefig(
                save_to_path,
                dpi=300,
                bbox_inches="tight",
            )
            print(f"Feature visualisation successfully saved to {save_to_path}")
        else:
            plt.show()

        return self
