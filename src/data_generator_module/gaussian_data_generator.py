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

    def __init__(self, n_samples: int, n_features: int, random_state: int, dataset_settings: dict = None):
        """
        Initialize the GaussianDataGenerator.
        """
        DataGeneratorValidators.validate_init_parameters(n_samples, n_features)
        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state
        self.dataset_settings = dataset_settings if dataset_settings is not None else {}
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
        signal_features: Dict[str, Dict[str, float]],
        noise_features: Dict[str, Dict[str, float]],
        feature_types: Dict[str, str],
        store_for_visualisation: bool = False,
    ):
        """
        Create binary classification with separate feature distributions for signal and noise.

        Parameters:
        -----------
        signal_features : Dict[str, Dict[str, float]]
            Feature parameters for signal samples: {"feature_0": {"mean": 15.0, "std": 1.0}, ...}
        noise_features : Dict[str, Dict[str, float]]
            Feature parameters for noise samples: {"feature_0": {"mean": 5.0, "std": 2.0}, ...}
        feature_types : Dict[str, str]
            Feature types for all features: {"feature_0": "discrete", ...}
        store_for_visualisation : bool
            Whether to store observations for visualisation
        """

        # Fixed 50/50 split
        signal_ratio = 0.5
        n_signal_samples = int(self.n_samples * signal_ratio)
        n_noise_samples = self.n_samples - n_signal_samples

        rng = np.random.RandomState(self.random_state)

        # Get the number of decimal places from settings, with a fallback to None
        decimal_places = self.dataset_settings.get('continuous_decimal_places')
    
        # Generate signal features
        signal_data = {}
        for feature_name, params in signal_features.items():
            feature_data = rng.normal(params["mean"], params["std"], n_signal_samples)
            feature_type = feature_types.get(feature_name, "continuous")
            
            if feature_type == "discrete":
                feature_data = np.round(feature_data)
            elif feature_type == "continuous" and decimal_places is not None:
                feature_data = np.round(feature_data, decimal_places) # &lt;-- Round continuous data
                
            signal_data[feature_name] = feature_data
    
        # Generate noise features
        noise_data = {}
        for feature_name, params in noise_features.items():
            feature_data = rng.normal(params["mean"], params["std"], n_noise_samples)
            feature_type = feature_types.get(feature_name, "continuous")
    
            if feature_type == "discrete":
                feature_data = np.round(feature_data)
            elif feature_type == "continuous" and decimal_places is not None:
                feature_data = np.round(feature_data, decimal_places) # &lt;-- Round continuous data
    
            noise_data[feature_name] = feature_data

        # Combine all features
        combined_data = {}
        for feature_name in signal_features.keys():
            combined_data[feature_name] = np.concatenate(
                [signal_data[feature_name], noise_data[feature_name]]
            )

        # Create labels
        signal_labels = np.ones(n_signal_samples, dtype=int)
        noise_labels = np.zeros(n_noise_samples, dtype=int)
        all_labels = np.concatenate([signal_labels, noise_labels])

        # Shuffle everything together
        indices = rng.permutation(self.n_samples)
        for feature_name in combined_data:
            combined_data[feature_name] = combined_data[feature_name][indices]
        all_labels = all_labels[indices]

        # Store as DataFrame
        combined_data["target"] = all_labels

        # Add temporary observations if requested
        if store_for_visualisation:
            # Generate separate observations for visualization
            signal_obs = rng.normal(2.0, 0.8, n_signal_samples)  # Example values
            noise_obs = rng.normal(-1.0, 1.2, n_noise_samples)  # Example values
            all_obs = np.concatenate([signal_obs, noise_obs])
            combined_data["_temp_observations"] = all_obs[indices]

        self.data = pd.DataFrame(combined_data)

        # Store metadata
        self.feature_based_metadata = {
            "signal_features": signal_features,
            "noise_features": noise_features,
            "feature_types": feature_types,
            "signal_ratio": signal_ratio,
            "actual_signal_ratio": all_labels.mean(),
            "approach": "separate_distributions",
            "has_temp_observations": store_for_visualisation,
        }

        print(
            "Created feature-based signal/noise classification with separate distributions:"
        )
        print(f" Signal samples: {(all_labels == 1).sum()} (target = 1)")
        print(f" Noise samples: {(all_labels == 0).sum()} (target = 0)")
        print(" Features have different distributions for signal vs noise samples")

        return self

    def visualise_signal_noise_by_features(
        self,
        save_path: Optional[str] = None,
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
    ):
        """
        Visualise signal vs noise distributions for each individual feature.
        Shows overlapping histograms of signal and noise samples for each feature.

        Parameters:
        -----------
        save_path : Optional[str]
            Path to save the plot
        title : Optional[str]
            Custom main title for the plot
        subtitle : Optional[str]
            Custom subtitle for the plot
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

        # Add custom titles if provided
        if title:
            fig.suptitle(title, fontsize=20, weight="bold")

        if subtitle:
            plt.figtext(0.5, 0.92, subtitle, ha="center", fontsize=14, style="italic")

        # Plot each feature (rest of the plotting code remains the same)
        for i, feature in enumerate(feature_columns):
            ax = axes[i]

            # Plot overlapping histograms
            ax.hist(
                noise_data[feature],
                bins=30,
                alpha=0.7,
                label="Noise",
                color="red",
                density=True,
                edgecolor="black",
                linewidth=0.5,
            )

            ax.hist(
                signal_data[feature],
                bins=30,
                alpha=0.7,
                label="Signal",
                color="blue",
                density=True,
                edgecolor="black",
                linewidth=0.5,
            )

            # Customize subplot
            ax.set_title(f"Distribution of {feature}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.legend(loc="upper right")
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

        # Adjust layout to accommodate custom titles
        if title or subtitle:
            plt.tight_layout(rect=[0, 0.03, 1, 0.92])
        else:
            plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Feature-wise signal/noise visualisation saved to {save_path}")
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

    def perturb_feature(self, feature_name: str, class_label: int, sigma_shift: float=None, scale_factor: float=None, additive_noise: float=None, multiplicative_factor: float=None):
        """
        Perturbs a specific feature for a given class by either an additive shift
        or a multiplicative scale factor.
    
        Parameters:
        -----------
        feature_name : str
            The name of the feature to perturb.
        class_label : int
            The class to apply the perturbation to (1 for signal, 0 for noise).
        sigma_shift : float, optional
            The additive amount to shift the feature's mean, in multiples of its standard deviation.
        scale_factor : float, optional
            The multiplicative factor to scale the feature's values.
        additive_noise : float, optional
            The standard deviation of Gaussian noise to add to the feature.
        multiplicative_factor : float, optional
            The multiplicative factor to apply to the feature values.
        """
        if self.data is None:
            raise ValueError("Data has not been generated yet. Call a generation method first.")
    
        if feature_name not in self.data.columns:
            raise ValueError(f"Feature '{feature_name}' not found in the dataset.")
    
        perturbation_params = [sigma_shift, scale_factor, additive_noise, multiplicative_factor]
        active_params = [p for p in perturbation_params if p is not None]
        if len(active_params) > 1:
            raise ValueError("Cannot apply multiple perturbation types simultaneously.")
        
        if len(active_params) == 0:
            raise ValueError("At least one perturbation parameter must be specified.")
    
        class_name = "Signal" if class_label == 1 else "Noise"
        class_indices = self.data['target'] == class_label
    
        # Apply the selected perturbation
        if scale_factor is not None:
            self.data.loc[class_indices, feature_name] *= scale_factor
            perturbation_desc = f"scaled by {scale_factor}x"
            perturbation_type = 'scale'
            perturbation_value = scale_factor
    
        elif sigma_shift is not None:
            # Get original parameters for this feature and class
            if hasattr(self, 'feature_based_metadata'):
                feature_params = self.feature_based_metadata[
                    'signal_features' if class_label == 1 else 'noise_features'
                ].get(feature_name)
                if feature_params:
                    std_dev = feature_params['std']
                    shift_amount = sigma_shift * std_dev
                    self.data.loc[class_indices, feature_name] += shift_amount
                    perturbation_desc = f"shifted by {sigma_shift}σ ({shift_amount:.3f})"
                else:
                    # Fallback: use empirical std of the class
                    empirical_std = self.data.loc[class_indices, feature_name].std()
                    shift_amount = sigma_shift * empirical_std
                    self.data.loc[class_indices, feature_name] += shift_amount
                    perturbation_desc = f"shifted by {sigma_shift}σ ({shift_amount:.3f}) [empirical]"
            else:
                raise ValueError("Cannot apply sigma_shift: feature metadata not available.")
            
            perturbation_type = 'sigma_shift'
            perturbation_value = sigma_shift
    
        elif additive_noise is not None:
            noise = np.random.normal(0, additive_noise, sum(class_indices))
            self.data.loc[class_indices, feature_name] += noise
            perturbation_desc = f"added noise (σ={additive_noise})"
            perturbation_type = 'additive_noise'
            perturbation_value = additive_noise
    
        elif multiplicative_factor is not None:
            self.data.loc[class_indices, feature_name] *= multiplicative_factor
            perturbation_desc = f"multiplied by {multiplicative_factor}"
            perturbation_type = 'multiplicative'
            perturbation_value = multiplicative_factor
    
        else:
            # This should never be reached due to validation above, but kept as safety
            print(f"Warning: No perturbation specified for {feature_name}. No changes made.")
            return self
    
        print(f"Perturbed '{feature_name}' for {class_name} class: {perturbation_desc}.")
    
        # Store metadata about the perturbation
        if 'perturbations' not in self.feature_based_metadata:
            self.feature_based_metadata['perturbations'] = []
    
        self.feature_based_metadata['perturbations'].append({
            'feature': feature_name,
            'class': class_name,
            'type': perturbation_type,
            'value': perturbation_value
        })
    
        return self
        
    def perturb_correlated_features(
        self, 
        feature_names: List[str], 
        class_label: int, 
        correlation_matrix: List[List[float]],
        sigma_shift: float = None, 
        scale_factor: float = None,
        description: str = None
    ):
        """
        Perturbs multiple features simultaneously with specified correlations.
        
        Parameters:
        -----------
        feature_names : List[str]
            List of feature names to perturb together
        class_label : int
            The class to apply the perturbation to (1 for signal, 0 for noise)
        correlation_matrix : List[List[float]]
            Correlation matrix for the features (must be positive definite)
        sigma_shift : float, optional
            The additive shift amount in multiples of standard deviation
        scale_factor : float, optional
            The multiplicative scaling factor
        description : str, optional
            Human-readable description of the perturbation
        """
        if self.data is None:
            raise ValueError("Data has not been generated yet. Call a generation method first.")
        
        if sigma_shift is not None and scale_factor is not None:
            raise ValueError("Cannot apply both sigma_shift and scale_factor simultaneously.")
        
        # Validate inputs
        for feature_name in feature_names:
            if feature_name not in self.data.columns:
                raise ValueError(f"Feature '{feature_name}' not found in the dataset.")
        
        correlation_matrix = np.array(correlation_matrix)
        if correlation_matrix.shape != (len(feature_names), len(feature_names)):
            raise ValueError("Correlation matrix dimensions must match number of features.")
        
        # Check if correlation matrix is positive definite
        try:
            np.linalg.cholesky(correlation_matrix)
        except np.linalg.LinAlgError:
            raise ValueError("Correlation matrix must be positive definite.")
        
        class_name = "Signal" if class_label == 1 else "Noise"
        class_indices = self.data['target'] == class_label
        n_samples = class_indices.sum()
        
        if n_samples == 0:
            print(f"Warning: No samples found for {class_name} class. No perturbation applied.")
            return self
        
        # Generate correlated perturbations
        rng = np.random.RandomState(self.random_state + hash(tuple(feature_names)) % 2**31)
        
        if scale_factor is not None:
            # Handle negative scale factors by separating sign from magnitude
            sign = np.sign(scale_factor)
            magnitude = abs(scale_factor)
        
            if magnitude == 0:
                # Scaling by zero is a valid edge case, just set features to zero
                multipliers = np.zeros((n_samples, len(feature_names)))
            else:
                # Use log-normal distribution for the magnitude to ensure positive multipliers
                mean_log = np.log(magnitude) # This is now safe from math domain errors
                cov_log = correlation_matrix * 0.1 # Small variance in log space
                log_multipliers = rng.multivariate_normal(
                    mean=[mean_log] * len(feature_names),
                    cov=cov_log,
                    size=n_samples
                )
                multipliers = np.exp(log_multipliers)
        
            # Apply correlated scaling with the correct sign
            for i, feature_name in enumerate(feature_names):
                self.data.loc[class_indices, feature_name] *= (sign * multipliers[:, i])
        
            perturbation_desc = f"correlated scaling by ~{scale_factor}x"
            perturbation_type = 'correlated_scale'
            perturbation_value = scale_factor
            
        elif sigma_shift is not None:
            # Get original parameters for each feature
            feature_params = []
            for feature_name in feature_names:
                params = self.feature_based_metadata[
                    'signal_features' if class_label == 1 else 'noise_features'
                ].get(feature_name)
                if not params:
                    raise ValueError(f"Could not find original parameters for '{feature_name}' and class '{class_name}'.")
                feature_params.append(params)
            
            # Create covariance matrix from correlation matrix and standard deviations
            std_devs = np.array([params['std'] for params in feature_params])
            cov_matrix = correlation_matrix * np.outer(std_devs, std_devs)
            
            # Generate correlated shifts
            shifts = rng.multivariate_normal(
                mean=[0] * len(feature_names),
                cov=cov_matrix,
                size=n_samples
            ) * sigma_shift
            
            # Apply correlated shifts
            for i, feature_name in enumerate(feature_names):
                self.data.loc[class_indices, feature_name] += shifts[:, i]
            
            perturbation_desc = f"correlated shift by {sigma_shift} sigma"
            perturbation_type = 'correlated_shift'
            perturbation_value = sigma_shift
        else:
            print(f"Warning: No perturbation specified for correlated features. No changes made.")
            return self
        
        print(f"Applied {perturbation_desc} to {feature_names} for {class_name} class.")
        
        # Store metadata about the perturbation
        if 'perturbations' not in self.feature_based_metadata:
            self.feature_based_metadata['perturbations'] = []
        
        self.feature_based_metadata['perturbations'].append({
            'type': 'correlated',
            'features': feature_names,
            'class': class_name,
            'perturbation_type': perturbation_type,
            'value': perturbation_value,
            'correlation_matrix': correlation_matrix.tolist(),
            'description': description
        })
        
        return self

    

    def apply_perturbation_from_config(self, perturbation_config: Dict):
        """
        Apply perturbation from configuration dictionary.
        Handles individual, correlated, and all-feature perturbations.
        """
        pert_type = perturbation_config.get('type', 'individual')
    
        if pert_type == 'correlated':
            self.perturb_correlated_features(
                feature_names=perturbation_config['features'],
                class_label=perturbation_config['class_label'],
                correlation_matrix=perturbation_config['correlation_matrix'],
                sigma_shift=perturbation_config.get('sigma_shift'),
                scale_factor=perturbation_config.get('scale_factor'),
                description=perturbation_config.get('description')
            )
        elif pert_type == 'all_features': 
            self.perturb_all_features(
                class_label=perturbation_config['class_label'],
                sigma_shift=perturbation_config.get('sigma_shift'),
                scale_factor=perturbation_config.get('scale_factor'),
                description=perturbation_config.get('description')
            )
        else:  # Handle individual perturbations
            self.perturb_feature(
                feature_name=perturbation_config['feature'],
                class_label=perturbation_config['class_label'],
                sigma_shift=perturbation_config.get('sigma_shift'),
                scale_factor=perturbation_config.get('scale_factor'),
                additive_noise=perturbation_config.get('additive_noise'),
                multiplicative_factor=perturbation_config.get('multiplicative_factor')
            )