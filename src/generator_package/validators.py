from typing import Dict, List
import pandas as pd


class DataGeneratorValidators:
    """Validation methods for GaussianDataGenerator parameters and data."""

    @staticmethod
    def validate_feature_parameters(feature_parameters: Dict[str, Dict]) -> None:
        """
        Validate feature parameters structure and values.

        Parameters:
        -----------
        feature_parameters : Dict[str, Dict]
            Dictionary containing feature parameters with 'mean' and 'std' keys

        Raises:
        -------
        ValueError
            If parameters are invalid or missing required keys
        """
        for feature_name, params in feature_parameters.items():
            if not isinstance(params, dict):
                raise ValueError(f"Parameters for {feature_name} must be a dictionary")
            if "mean" not in params or "std" not in params:
                raise ValueError(
                    f"Feature {feature_name} must have 'mean' and 'std' parameters"
                )
            if params["std"] <= 0:
                raise ValueError(
                    f"Standard deviation for {feature_name} must be positive"
                )

    @staticmethod
    def validate_init_parameters(n_samples: int, n_features: int) -> None:
        """
        Validate initialisation parameters.

        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        n_features : int
            Number of features to generate

        Raises:
        -------
        ValueError
            If parameters are not positive integers
        """
        if n_samples <= 0:
            raise ValueError("Number of samples must be positive")
        if n_features <= 0:
            raise ValueError("Number of features must be positive")

    @staticmethod
    def validate_feature_types(feature_types: Dict[str, str]) -> None:
        """
        Validate feature types dictionary.

        Parameters:
        -----------
        feature_types : Dict[str, str]
            Dictionary mapping feature names to their types

        Raises:
        -------
        ValueError
            If feature types are not 'continuous' or 'discrete'
        """
        valid_types = {"continuous", "discrete"}
        for feature_name, feature_type in feature_types.items():
            if feature_type not in valid_types:
                raise ValueError(
                    f"Invalid feature type '{feature_type}' for {feature_name}. "
                    f"Must be one of {valid_types}"
                )

    @staticmethod
    def validate_perturbation_parameters(
        perturbation_type: str, features: List[str], scale: float, data: pd.DataFrame
    ) -> None:
        """
        Validate perturbation parameters.

        Parameters:
        -----------
        perturbation_type : str
            Type of perturbation ('gaussian' or 'uniform')
        features : List[str]
            List of feature names to perturb
        scale : float
            Scale of perturbation
        data : pd.DataFrame
            The data containing the features

        Raises:
        -------
        ValueError
            If parameters are invalid
        """
        if perturbation_type not in ["gaussian", "uniform"]:
            raise ValueError("perturbation_type must be 'gaussian' or 'uniform'")

        if scale < 0:
            raise ValueError("Scale must be positive")

        if data is None:
            raise ValueError("No data generated. Call generate_features() first.")

        for feature in features:
            if feature not in data.columns:
                raise ValueError(f"Feature '{feature}' not found in data")

    @staticmethod
    def validate_target_parameters(
        features_to_use: List[str],
        weights: List[float],
        function_type: str,
        data: pd.DataFrame,
    ) -> None:
        """
        Validate target variable creation parameters.

        Parameters:
        -----------
        features_to_use : List[str]
            List of feature names to use for target creation
        weights : List[float]
            Weights for each feature
        function_type : str
            Type of function to use
        data : pd.DataFrame
            The data containing the features

        Raises:
        -------
        ValueError
            If parameters are invalid
        """
        if data is None:
            raise ValueError("No data generated. Call generate_features() first.")

        missing_features = [f for f in features_to_use if f not in data.columns]
        if missing_features:
            raise ValueError(f"Features not found in data: {missing_features}")

        if len(weights) != len(features_to_use):
            raise ValueError("Number of weights must match number of features to use.")

        if function_type not in ["linear", "polynomial", "logistic"]:
            raise ValueError(
                "function_type must be 'linear', 'polynomial', or 'logistic'"
            )

    @staticmethod
    def validate_visualisation_parameters(
        features: List[str], max_features_to_show: int, n_bins: int, data: pd.DataFrame
    ) -> None:
        """
        Validate visualisation parameters.

        Parameters:
        -----------
        features : List[str]
            List of feature names to visualise
        max_features_to_show : int
            Maximum number of features to display
        n_bins : int
            Number of bins for histograms
        data : pd.DataFrame
            The data containing the features

        Raises:
        -------
        ValueError
            If parameters are invalid
        """
        if data is None:
            raise ValueError("No data generated to visualise.")

        invalid_features = [f for f in features if f not in data.columns]
        if invalid_features:
            raise ValueError(f"Features not found in data: {invalid_features}")

        if max_features_to_show <= 0:
            raise ValueError("max_features_to_show must be positive")

        if n_bins <= 0:
            raise ValueError("n_bins must be positive")
