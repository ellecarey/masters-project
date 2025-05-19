import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Union, Tuple, Optional
import random


class GuassianDataGenerator:
    """
    A class for generating random toy data with Gaussian distributions and controlled perturbations.

    This class:
    - generates data with a specified number of features
    - controls wheter features are discrete or continuous
    - adds controlled perturbations to the data
    - visualises the generated data
    """

    def __init__(
        self, n_samples: int = 1000, n_features: int = 10, random_state: int = 42
    ):
        """
        Initialise the GaussianDataGenerator.

        Parameters:
        -----------
        n_samples (int): Number of samples to generate.
        n_features (int): Number of features in each sample.
        random_state (int): Seed for the random number generator.
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state
        self.data: Optional[pd.DataFrame] = None
        self.feature_types: Dict[str, str] = {}
        self.feature_parameters: Dict[str, Dict] = {}

        # Seed numpy for operations within this class instance
        np.random.seed(self.random_state)
        # Note: Global random.seed and torch.manual_seed should be set in the main script if needed elsewhere

    def generate_features(
        self,
        feature_parameters: Optional[Dict[str, Dict]] = None,
        feature_types: Optional[Dict[str, str]] = None,
    ):
        """
        Generate features based on specified parameters.
        """
        current_feature_count = 0
        if self.data is not None:
            current_feature_count = len(self.data.columns)

        # If no parameters provided, generate default
        if feature_parameters is None:
            feature_parameters = {}
            for i in range(self.n_features):
                feature_name = f"feature_{current_feature_count + i}"
                feature_parameters[feature_name] = {
                    "mean": np.random.uniform(-5, 5),
                    "std": np.random.uniform(0.5, 3),
                }

        # If no types provided, default all to continuous
        if feature_types is None:
            feature_types = {}
            for feature_name in feature_parameters.keys():
                feature_types[feature_name] = "continuous"

        self.feature_parameters.update(feature_parameters)
        self.feature_types.update(feature_types)

        data_dictionary = {}
        for feature_name, params in feature_parameters.items():
            raw_data = np.random.normal(params["mean"], params["std"], self.n_samples)
            if self.feature_types.get(feature_name) == "discrete":
                raw_data = np.round(raw_data)
            data_dictionary[feature_name] = raw_data

        new_data_df = pd.DataFrame(data_dictionary)

        if self.data is None:
            self.data = new_data_df
        else:
            self.data = pd.concat([self.data, new_data_df], axis=1)

        self.n_features = len(self.data.columns)  # Update total number of features
        print(f"Generated/added features. Total features: {self.n_features}")
        print(f"Current feature parameters: {self.feature_parameters}")
        return self

    def add_perturbations(
        self,
        perturbation_type: str = "gaussian",
        features: Optional[List[str]] = None,
        scale: float = 0.1,
    ):
        """
        Add controlled perturbations to the data.
        """
        if self.data is None:
            raise ValueError("No data generated. Call generate_features() first.")

        if features is None:
            features = self.data.columns.tolist()

        perturbed_data = self.data.copy()
        for feature in features:
            if feature not in self.data.columns:
                print(
                    f"Warning: Feature '{feature}' not found for perturbation. Skipping."
                )
                continue

            if perturbation_type == "gaussian":
                # Corrected from .stdd() to .std()
                noise = np.random.normal(
                    0, scale * self.data[feature].std(), self.n_samples
                )
                perturbed_data[feature] += noise
            elif perturbation_type == "uniform":
                noise = (
                    np.random.uniform(-scale, scale, self.n_samples)
                    * self.data[feature].std()
                )
                perturbed_data[feature] += noise
            else:
                raise ValueError(f"Unknown perturbation type: {perturbation_type}")

            if self.feature_types.get(feature) == "discrete":
                perturbed_data[feature] = np.round(perturbed_data[feature])

        self.data = perturbed_data
        print(
            f"Added {perturbation_type} perturbations with scale {scale} to features: {features}"
        )
        return self

    def add_features(
        self,
        n_new_features: int = 1,
        feature_parameters: Optional[Dict[str, Dict]] = None,
        feature_types: Optional[Dict[str, str]] = None,
    ):
        """
        Add new features to the existing dataset.
        """
        if (
            self.data is None and n_new_features > 0
        ):  # If no data, treat this as initial generation
            self.n_features = n_new_features  # Set total features for initial call
            return self.generate_features(feature_parameters, feature_types)

        if n_new_features <= 0:
            return self

        start_idx = len(self.data.columns) if self.data is not None else 0

        # Generate default parameters for new features if not provided
        if feature_parameters is None:
            feature_parameters = {}
            for i in range(n_new_features):
                feature_name = f"feature_{start_idx + i}"
                feature_parameters[feature_name] = {
                    "mean": np.random.uniform(-5, 5),
                    "std": np.random.uniform(0.5, 3),
                }

        # Generate default types for new features if not provided
        if feature_types is None:
            feature_types = {}
            for feature_name in feature_parameters.keys():
                feature_types[feature_name] = "continuous"

        # Call generate_features to actually add them by passing only the new ones
        self.generate_features(feature_parameters, feature_types)
        print(f"Added {n_new_features} new features.")
        return self

    def change_feature_type(self, feature_name: str, new_type: str):
        """
        Change the type of a feature (continuous or discrete).
        """
        if self.data is None or feature_name not in self.data.columns:
            raise ValueError(f"Feature {feature_name} not found in the data.")
        if new_type not in ["continuous", "discrete"]:
            raise ValueError("Invalid feature type. Use 'continuous' or 'discrete'.")

        self.feature_types[feature_name] = new_type
        if new_type == "discrete":
            self.data[feature_name] = np.round(self.data[feature_name])
        # If 'continuous', no change needed to data values unless they were previously discrete (which is fine)
        print(f"Changed type of '{feature_name}' to '{new_type}'.")
        return self

    def visusalise_features(
        self,
        features: Optional[List[str]] = None,
        max_features_to_show: Optional[int] = 10,
        n_bins: int = 30,
    ):
        """
        Visualise the distribution of selected features.
        """
        if self.data is None:
            raise ValueError("No data generated. Call generate_features() first.")

        if features is None:
            features_to_plot = self.data.columns.tolist()
            if (
                max_features_to_show is not None
                and len(features_to_plot) > max_features_to_show
            ):
                print(
                    f"Too many features to visualise. Showing first {max_features_to_show} out of {len(features_to_plot)}."
                )
                features_to_plot = features_to_plot[:max_features_to_show]
        else:
            features_to_plot = features

        if not features_to_plot:
            print("No features to visualise.")
            return

        n_features_plot = len(features_to_plot)
        n_cols = min(3, n_features_plot)
        n_rows = (n_features_plot + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(15, 4 * n_rows), squeeze=False
        )
        axes = axes.flatten()

        for i, feature_name in enumerate(features_to_plot):
            if feature_name not in self.data.columns:
                print(
                    f"Warning: Feature '{feature_name}' for visualization not found in data. Skipping."
                )
                axes[i].set_visible(False)  # Hide axis if feature doesn't exist
                continue

            ax = axes[i]
            self.data[feature_name].hist(bins=n_bins, ax=ax, alpha=0.7)
            ax.set_title(
                f"{feature_name} ({self.feature_types.get(feature_name, 'unknown')})"
            )
            ax.axvline(
                self.data[feature_name].mean(),
                color="red",
                linestyle="--",
                label=f"Mean: {self.data[feature_name].mean():.2f}",
            )
            ax.axvline(
                self.data[feature_name].median(),
                color="green",
                linestyle="-",
                label=f"Median: {self.data[feature_name].median():.2f}",
            )
            ax.legend()

        for i in range(n_features_plot, len(axes)):  # Hide unused subplots
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.show()

    def get_data(self) -> Optional[pd.DataFrame]:
        """Get the generated data."""
        return self.data

    def get_feature_information(self) -> Dict:
        """Get information about the features."""
        return {
            "feature_parameters": self.feature_parameters,
            "feature_types": self.feature_types,
        }

    def create_target_variable(
        self,
        features_to_use: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
        noise_level: float = 0.1,
        function_type: str = "linear",
    ):
        """
        Create a target variable based on the selected features.
        """
        if self.data is None:
            raise ValueError("No data generated. Call generate_features() first.")

        if features_to_use is None:
            features_to_use = [
                col for col in self.data.columns if col != "target"
            ]  # Exclude existing target

        # Ensure all selected features exist
        missing_features = [f for f in features_to_use if f not in self.data.columns]
        if missing_features:
            raise ValueError(f"Features not found in data: {missing_features}")

        if not features_to_use:
            raise ValueError(
                "No features specified or available to create a target variable."
            )

        if weights is None:
            weights = np.random.uniform(-1, 1, len(features_to_use))
        elif len(weights) != len(features_to_use):
            raise ValueError("Number of weights must match number of features to use.")

        X = self.data[features_to_use].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if function_type == "linear":
            y = np.dot(X_scaled, weights)
        elif function_type == "polynomial":
            y = np.dot(X_scaled, weights) + np.sum(
                0.5 * np.array(weights) * X_scaled**2, axis=1
            )
        elif function_type == "logistic":
            logits = np.dot(X_scaled, weights)
            y = 1 / (1 + np.exp(-logits))
            y = (y > 0.5).astype(int)  # Binary target
        else:
            raise ValueError(
                "Invalid function type. Use 'linear', 'polynomial', or 'logistic'."
            )

        if (
            function_type != "logistic"
        ):  # Add noise only if not logistic (which is already binary)
            noise = np.random.normal(0, noise_level, self.n_samples)
            y += noise

        self.data["target"] = y
        print(
            f"Created 'target' variable using function '{function_type}' based on features: {features_to_use}"
        )
        return self
