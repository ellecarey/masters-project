import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Optional
import os


class GaussianDataGenerator:
    """
    A class for generating random toy data with Gaussian distributions and controlled perturbations.
    All parameters must be explicitly specified by the user.
    """

    def __init__(self, n_samples: int, n_features: int, random_state: int):
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

        np.random.seed(self.random_state)

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
        feature_parameters (Dict[str, Dict]): Parameters for each feature
        feature_types (Dict[str, str]): Type specification for each feature
        """
        # Count existing feature columns
        current_feature_count = 0
        if self.data is not None:
            feature_columns = [
                col for col in self.data.columns if col.startswith("feature_")
            ]
            current_feature_count = len(feature_columns)

        # Use number of features provided in parameters if n_features_to_generate not specified
        if n_features_to_generate is None:
            n_features_to_generate = len(feature_parameters)

        # Validate that parameters are provided for all features
        expected_features = [
            f"feature_{current_feature_count + i}"
            for i in range(n_features_to_generate)
        ]

        for feature_name in expected_features:
            if feature_name not in feature_parameters:
                raise ValueError(
                    f"Parameters must be provided for feature: {feature_name}"
                )
            if feature_name not in feature_types:
                raise ValueError(f"Type must be specified for feature: {feature_name}")

        self.feature_parameters.update(feature_parameters)
        self.feature_types.update(feature_types)

        data_dictionary = {}
        for feature_name, params in feature_parameters.items():
            if "mean" not in params or "std" not in params:
                raise ValueError(
                    f"Both 'mean' and 'std' must be specified for {feature_name}"
                )

            raw_data = np.random.normal(params["mean"], params["std"], self.n_samples)

            if self.feature_types.get(feature_name) == "discrete":
                raw_data = np.round(raw_data)
            elif self.feature_types.get(feature_name) != "continuous":
                raise ValueError(
                    f"Feature type must be 'continuous' or 'discrete' for {feature_name}"
                )

            data_dictionary[feature_name] = raw_data

        new_data_df = pd.DataFrame(data_dictionary)

        if self.data is None:
            self.data = new_data_df
        else:
            self.data = pd.concat([self.data, new_data_df], axis=1)

        self.n_features = len(self.data.columns)
        print(f"Generated/added features. Total features: {self.n_features}")
        return self

    def add_perturbations(
        self,
        perturbation_type: str,
        features: List[str],
        scale: float,
    ):
        """
        Add controlled perturbations to the data.

        Parameters:
        -----------
        perturbation_type (str): Type of perturbation ('gaussian' or 'uniform')
        features (List[str]): List of feature names to perturb
        scale (float): Scale of the perturbation
        """
        if self.data is None:
            raise ValueError("No data generated. Call generate_features() first.")

        if perturbation_type not in ["gaussian", "uniform"]:
            raise ValueError("perturbation_type must be 'gaussian' or 'uniform'")

        perturbed_data = self.data.copy()

        for feature in features:
            if feature not in self.data.columns:
                raise ValueError(f"Feature '{feature}' not found in data")

            if perturbation_type == "gaussian":
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

            if self.feature_types.get(feature) == "discrete":
                perturbed_data[feature] = np.round(perturbed_data[feature])

        self.data = perturbed_data
        print(
            f"Added {perturbation_type} perturbations with scale {scale} to features: {features}"
        )
        return self

    def add_features(
        self,
        n_new_features: int,
        feature_parameters: Dict[str, Dict],
        feature_types: Dict[str, str],
    ):
        """
        Add new features to the existing dataset.

        Parameters:
        -----------
        n_new_features (int): Number of new features to add
        feature_parameters (Dict[str, Dict]): Parameters for new features
        feature_types (Dict[str, str]): Types for new features
        """
        if self.data is None and n_new_features > 0:
            self.n_features = n_new_features
            return self.generate_features(feature_parameters, feature_types)

        if n_new_features <= 0:
            raise ValueError("n_new_features must be greater than 0")

        # Calculate the correct starting index based on existing feature columns
        existing_feature_columns = [
            col for col in self.data.columns if col.startswith("feature_")
        ]
        start_idx = len(existing_feature_columns)

        expected_features = [f"feature_{start_idx + i}" for i in range(n_new_features)]

        # Validate that all required parameters are provided
        for feature_name in expected_features:
            if feature_name not in feature_parameters:
                raise ValueError(
                    f"Parameters must be provided for new feature: {feature_name}"
                )
            if feature_name not in feature_types:
                raise ValueError(
                    f"Type must be specified for new feature: {feature_name}"
                )

        # Extract only the parameters for the new features
        new_feature_params = {
            name: feature_parameters[name] for name in expected_features
        }
        new_feature_types = {name: feature_types[name] for name in expected_features}

        # Generate new features directly using existing data structure
        data_dictionary = {}
        for feature_name, params in new_feature_params.items():
            if "mean" not in params or "std" not in params:
                raise ValueError(
                    f"Both 'mean' and 'std' must be specified for {feature_name}"
                )

            raw_data = np.random.normal(params["mean"], params["std"], self.n_samples)

            if new_feature_types.get(feature_name) == "discrete":
                raw_data = np.round(raw_data)
            elif new_feature_types.get(feature_name) != "continuous":
                raise ValueError(
                    f"Feature type must be 'continuous' or 'discrete' for {feature_name}"
                )

            data_dictionary[feature_name] = raw_data

        # Add new features to existing data
        for feature_name, feature_data in data_dictionary.items():
            self.data[feature_name] = feature_data

        # Update stored parameters and types
        self.feature_parameters.update(new_feature_params)
        self.feature_types.update(new_feature_types)

        print(f"Added {n_new_features} new features.")
        return self

    def change_feature_type(self, feature_name: str, new_type: str):
        """
        Change the type of a feature (continuous or discrete).
        """
        if self.data is None or feature_name not in self.data.columns:
            raise ValueError(f"Feature {feature_name} not found in the data.")

        if new_type not in ["continuous", "discrete"]:
            raise ValueError("new_type must be 'continuous' or 'discrete'")

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
        """
        Visualise the distribution of selected features.

        Parameters:
        -----------
        features (List[str]): List of feature names to visualize
        max_features_to_show (int): Maximum number of features to display
        n_bins (int): Number of bins for histograms
        save_to_dir (Optional[str]): Directory to save figure (None to not save)
        """
        if self.data is None:
            raise ValueError("No data generated to visualize.")

        # Validate features exist
        invalid_features = [f for f in features if f not in self.data.columns]
        if invalid_features:
            raise ValueError(f"Features not found in data: {invalid_features}")

        features_to_plot = features[:max_features_to_show]

        n_features_plot = len(features_to_plot)
        n_cols = min(3, n_features_plot)
        n_rows = (n_features_plot + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=plt.rcParams.get("figure.figsize"), squeeze=False
        )
        axes = axes.flatten()

        for i, feature_name in enumerate(features_to_plot):
            ax = axes[i]
            self.data[feature_name].hist(
                bins=n_bins, ax=ax, alpha=0.7, color="skyblue", edgecolor="black"
            )
            ax.set_title(
                f"{feature_name} ({self.feature_types.get(feature_name, 'N/A')})"
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

        for i in range(n_features_plot, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        if save_to_dir is not None:
            try:
                absolute_save_dir = os.path.abspath(save_to_dir)
                os.makedirs(absolute_save_dir, exist_ok=True)
                filename = "feature_distributions_plot.pdf"
                full_save_path = os.path.join(absolute_save_dir, filename)
                fig.savefig(full_save_path, dpi=plt.rcParams.get("figure.dpi"))
                print(f"Figure saved to: {full_save_path}")
            except Exception as e:
                print(f"Error saving figure: {e}")

        plt.show()

    def create_target_variable(
        self,
        features_to_use: List[str],
        weights: List[float],
        noise_level: float,
        function_type: str,
    ):
        """
        Create a target variable based on the selected features.

        Parameters:
        -----------
        features_to_use (List[str]): List of feature names to use
        weights (List[float]): Weights for each feature
        noise_level (float): Level of noise to add
        function_type (str): Type of function ('linear', 'polynomial', or 'logistic')
        """
        if self.data is None:
            raise ValueError("No data generated. Call generate_features() first.")

        # Validate features exist
        missing_features = [f for f in features_to_use if f not in self.data.columns]
        if missing_features:
            raise ValueError(f"Features not found in data: {missing_features}")

        if len(weights) != len(features_to_use):
            raise ValueError("Number of weights must match number of features to use.")

        if function_type not in ["linear", "polynomial", "logistic"]:
            raise ValueError(
                "function_type must be 'linear', 'polynomial', or 'logistic'"
            )

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
            y = (y > 0.5).astype(int)

        if function_type != "logistic":
            noise = np.random.normal(0, noise_level, self.n_samples)
            y += noise

        self.data["target"] = y
        print(
            f"Created 'target' variable using function '{function_type}' based on features: {features_to_use}"
        )
        return self

    def get_data(self) -> Optional[pd.DataFrame]:
        """Get the generated data."""
        return self.data

    def get_feature_information(self) -> Dict:
        """Get information about the features."""
        return {
            "feature_parameters": self.feature_parameters,
            "feature_types": self.feature_types,
        }
