from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .validators import DataGeneratorValidators


class GaussianDataGenerator:
    """
    A class for generating synthetic datasets with Gaussian-distributed features.

    This class provides functionality to create datasets with specified statistical
    properties, add perturbations, create target variables, and visualise the data.
    """

    def __init__(self, n_samples: int, n_features: int, random_state: int):
        """
        Initialise the GaussianDataGenerator.
        """
        # Add validation call
        DataGeneratorValidators.validate_init_parameters(n_samples, n_features)

        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state
        self.data: Optional[pd.DataFrame] = None
        self.feature_types: Dict[str, str] = {}
        self.feature_parameters: Dict[str, Dict] = {}

        np.random.seed(self.random_state)

    def generate_features(
        self,
        feature_parameters: Dict[str, Dict],
        feature_types: Dict[str, str],
        n_features_to_generate: Optional[int] = None,
    ):
        """Generate features based on specified parameters."""

        # validation calls
        DataGeneratorValidators.validate_feature_parameters(feature_parameters)
        DataGeneratorValidators.validate_feature_types(feature_types)

        # Create random number generator from the instance's random state
        rng = np.random.RandomState(self.random_state)

        # Count existing feature columns
        current_feature_count = 0
        if self.data is not None:
            feature_columns = [
                col for col in self.data.columns if col.startswith("feature_")
            ]
            current_feature_count = len(feature_columns)

        # Determine how many features to generate
        if n_features_to_generate is None:
            n_features_to_generate = len(feature_parameters)

        # Generate the data
        generated_data = {}
        for i, (feature_name, params) in enumerate(feature_parameters.items()):
            if i >= n_features_to_generate:
                break

            # Generate feature data
            feature_data = rng.normal(params["mean"], params["std"], self.n_samples)

            # Apply feature type
            feature_type = feature_types.get(feature_name, "continuous")
            if feature_type == "discrete":
                feature_data = np.round(feature_data)

            generated_data[feature_name] = feature_data

        # Create or update DataFrame
        new_df = pd.DataFrame(generated_data)

        if self.data is None:
            self.data = new_df
        else:
            self.data = pd.concat([self.data, new_df], axis=1)

        # Store parameters and types
        self.feature_parameters.update(feature_parameters)
        self.feature_types.update(feature_types)

        print(
            f"Generated {len(generated_data)} features with {self.n_samples} samples each."
        )
        return self

    def add_features(
        self,
        n_new_features: int,
        feature_parameters: Dict[str, Dict],
        feature_types: Dict[str, str],
    ):
        """Add new features to the existing dataset."""

        # Add validation calls
        DataGeneratorValidators.validate_feature_parameters(feature_parameters)
        DataGeneratorValidators.validate_feature_types(feature_types)

        if self.data is None and n_new_features > 0:
            self.n_features = n_new_features
            return self.generate_features(feature_parameters, feature_types)

        # Create new feature parameters for the specified number of features
        new_feature_params = {}
        feature_names = list(feature_parameters.keys())[:n_new_features]

        for feature_name in feature_names:
            new_feature_params[feature_name] = feature_parameters[feature_name]

        # Generate new features
        new_feature_data = {}
        for feature_name, params in new_feature_params.items():
            feature_data = np.random.normal(
                params["mean"], params["std"], self.n_samples
            )

            # Apply feature type
            feature_type = feature_types.get(feature_name, "continuous")
            if feature_type == "discrete":
                feature_data = np.round(feature_data)

            new_feature_data[feature_name] = feature_data

        # Add to existing data
        new_df = pd.DataFrame(new_feature_data)
        self.data = pd.concat([self.data, new_df], axis=1)

        # Update stored parameters
        self.feature_parameters.update(new_feature_params)
        for feature_name in feature_names:
            self.feature_types[feature_name] = feature_types.get(
                feature_name, "continuous"
            )

        print(f"Added {len(new_feature_data)} new features to the dataset.")
        return self

    def add_perturbations(
        self,
        perturbation_type: str,
        features: List[str],
        scale: float,
    ):
        """Add controlled perturbations to the data."""

        # validation
        DataGeneratorValidators.validate_perturbation_parameters(
            perturbation_type, features, scale, self.data
        )
        rng = np.random.RandomState(self.random_state)

        for feature in features:
            if perturbation_type == "gaussian":
                noise = rng.normal(0, scale, self.n_samples)
            elif perturbation_type == "uniform":
                noise = rng.uniform(-scale, scale, self.n_samples)

            self.data[feature] += noise

            # Apply feature type if it's discrete
            if self.feature_types.get(feature) == "discrete":
                self.data[feature] = np.round(self.data[feature])

        print(f"Added {perturbation_type} perturbations to {len(features)} features.")
        return self

    def get_data(self) -> Optional[pd.DataFrame]:
        """
        Get the generated data.

        Returns:
        --------
        Optional[pd.DataFrame]
            The generated dataset, or None if no data has been generated
        """
        return self.data

    def get_feature_information(self):
        """
        Get detailed information about the features in the dataset.

        Returns:
        --------
        dict
            Dictionary with feature names as keys and their statistics as values
        """
        if self.data is None:
            raise ValueError("No data generated yet.")

        info = {}
        for feature in self.data.columns:
            if feature.startswith("feature_"):
                stats = {
                    "mean": self.data[feature].mean(),
                    "std": self.data[feature].std(),
                    "min": self.data[feature].min(),
                    "max": self.data[feature].max(),
                    "type": self.feature_types.get(feature, "unknown"),
                }
                info[feature] = stats
        return info

    def create_target_variable(
        self,
        features_to_use: List[str],
        weights: List[float],
        noise_level: float,
        function_type: str,
    ):
        """Create a target variable based on the selected features."""

        # Add validation call
        DataGeneratorValidators.validate_target_parameters(
            features_to_use, weights, function_type, self.data
        )

        # Create the target variable
        target = np.zeros(self.n_samples)

        for feature, weight in zip(features_to_use, weights):
            if function_type == "linear":
                target += weight * self.data[feature]
            elif function_type == "polynomial":
                target += weight * (self.data[feature] ** 2)
            elif function_type == "logistic":
                target += weight * self.data[feature]

        # Add noise
        if noise_level > 0:
            np.random.seed(self.random_state)  # Ensure consistent random state
            noise = np.random.normal(0, noise_level, self.n_samples)
            target += noise

        # Apply logistic transformation if specified
        if function_type == "logistic":
            target = 1 / (1 + np.exp(-target))

        self.data["target"] = target
        print(f"Created target variable using {function_type} function.")
        return self

    def change_feature_type(self, feature_name: str, new_type: str):
        """Change the type of a feature (continuous or discrete)."""

        if self.data is None or feature_name not in self.data.columns:
            raise ValueError(f"Feature {feature_name} not found in the data.")

        # Add validation call
        temp_types = {feature_name: new_type}
        DataGeneratorValidators.validate_feature_types(temp_types)

        self.feature_types[feature_name] = new_type

        if new_type == "discrete":
            self.data[feature_name] = np.round(self.data[feature_name])

        print(f"Changed type of '{feature_name}' to '{new_type}'.")
        return self

    def visualise_features(
        self,
        features: List[str],
        max_features_to_show: int,
        n_bins: int,
        save_to_dir: Optional[str],
    ):
        """Visualise the distribution of selected features."""

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

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

        # Handle case where there's only one subplot
        if n_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        # Plot histograms
        for i, feature in enumerate(features_to_plot):
            ax = axes[i]
            ax.hist(self.data[feature], bins=n_bins, alpha=0.7, edgecolor="black")
            ax.set_title(f"Distribution of {feature}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        if save_to_dir:
            plt.savefig(
                f"{save_to_dir}/feature_distributions_plot.pdf",
                dpi=300,
                bbox_inches="tight",
            )
            print(f"Plot saved to {save_to_dir}/feature_distributions.pdf")
        else:
            plt.show()

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
