"""
Integration tests for complete workflows
"""

import pytest
import pandas as pd
from data_generator_module.config import (
    FEATURE_GENERATION_SETTINGS,
    PERTURBATION_SETTINGS,
    CREATE_TARGET_SETTINGS,
    DATASET_SETTINGS,
)
from .fixtures.sample_data import (
    TARGET_WEIGHTS,
    TARGET_NOISE_LEVELS,
    FEATURE_NOISE_LEVELS,
)


class TestIntegration:
    def test_complete_workflow_with_config_and_reproducibility(
        self,
        generator_factory,
        sample_feature_params,
        sample_feature_types,
        test_seeds,
        standard_test_config,
    ):
        """Test complete workflow, config integration, and reproducibility"""
        # Test basic workflow with fixture parameters
        generator = generator_factory(
            n_samples=standard_test_config["samples"],
            n_features=len(sample_feature_params),
            random_state=test_seeds[0],
        )

        generator.generate_features(sample_feature_params, sample_feature_types)

        # Add perturbations using fixture data
        continuous_features = [
            name
            for name, ftype in sample_feature_types.items()
            if ftype == "continuous"
        ][:2]

        if continuous_features:
            generator.add_perturbations(
                features=continuous_features,
                perturbation_type="gaussian",
                scale=FEATURE_NOISE_LEVELS["medium_noise"],
            )

        # Create target variable using fixtures
        features_for_target = list(sample_feature_params.keys())[:3]
        weights_for_target = TARGET_WEIGHTS[: len(features_for_target)]

        generator.create_target_variable(
            features_to_use=features_for_target,
            weights=weights_for_target,
            noise_level=TARGET_NOISE_LEVELS["medium_noise"],
            function_type="linear",
        )

        # Verify final dataset structure
        expected_columns = list(sample_feature_params.keys()) + ["target"]
        assert list(generator.data.columns) == expected_columns
        assert generator.data.shape == (
            standard_test_config["samples"],
            len(expected_columns),
        )
        assert not generator.data.isnull().any().any()

        # Verify target has reasonable values for neural network training
        target_std = generator.data["target"].std()
        target_mean = generator.data["target"].mean()
        assert 0.1 < target_std < 50, (
            f"Target std should be reasonable for NN training: {target_std}"
        )
        assert abs(target_mean) < 100, (
            f"Target mean should be reasonable for NN training: {target_mean}"
        )

        # Test reproducibility with same seed
        seed = test_seeds[0]
        gen1 = generator_factory(
            n_samples=standard_test_config["samples"],
            n_features=len(sample_feature_params),
            random_state=seed,
        )
        gen2 = generator_factory(
            n_samples=standard_test_config["samples"],
            n_features=len(sample_feature_params),
            random_state=seed,
        )

        # Apply identical workflows
        for gen in [gen1, gen2]:
            gen.generate_features(sample_feature_params, sample_feature_types)

            if continuous_features:
                gen.add_perturbations(
                    features=continuous_features,
                    perturbation_type="gaussian",
                    scale=FEATURE_NOISE_LEVELS["medium_noise"],
                )

            gen.create_target_variable(
                features_to_use=features_for_target,
                weights=weights_for_target,
                noise_level=TARGET_NOISE_LEVELS["medium_noise"],
                function_type="linear",
            )

        # Results should be identical
        pd.testing.assert_frame_equal(
            gen1.data,
            gen2.data,
            obj="Complete workflow should be reproducible with same seed",
        )

        # Test config integration if available
        try:
            # Create generator using actual config
            config_gen = generator_factory(
                n_samples=DATASET_SETTINGS["n_samples"],
                n_features=DATASET_SETTINGS["n_features"],
                random_state=test_seeds[0],
            )

            # Use actual config parameters
            config_gen.generate_features(
                FEATURE_GENERATION_SETTINGS["feature_parameters"],
                FEATURE_GENERATION_SETTINGS["feature_types"],
            )

            config_gen.add_perturbations(
                features=PERTURBATION_SETTINGS["features"],
                perturbation_type=PERTURBATION_SETTINGS["perturbation_type"],
                scale=PERTURBATION_SETTINGS["scale"],
            )

            config_gen.create_target_variable(
                features_to_use=CREATE_TARGET_SETTINGS["features_to_use"],
                weights=CREATE_TARGET_SETTINGS["weights"],
                noise_level=CREATE_TARGET_SETTINGS["noise_level"],
                function_type=CREATE_TARGET_SETTINGS["function_type"],
            )

            # Verify workflow completed successfully
            expected_total_features = (
                DATASET_SETTINGS["n_features"] + 1
            )  # +1 for target
            assert config_gen.data.shape == (
                DATASET_SETTINGS["n_samples"],
                expected_total_features,
            )
            assert not config_gen.data.isnull().any().any()

        except ImportError:
            # If config structure is different, continue without this validation
            pass

    def test_neural_network_research_pipeline(
        self, generator_factory, sample_feature_params, sample_feature_types, test_seeds
    ):
        """Test complete pipeline for neural network error propagation"""
        research_configs = [
            {
                "samples": 100,
                "noise": "low_noise",
                "desc": "Small dataset for development",
            },
            {
                "samples": 500,
                "noise": "medium_noise",
                "desc": "Medium dataset for training",
            },
            {
                "samples": 1000,
                "noise": "high_noise",
                "desc": "Large dataset for robustness testing",
            },
        ]

        for config in research_configs:
            # Create generator for each scenario
            generator = generator_factory(
                n_samples=config["samples"],
                n_features=len(sample_feature_params),
                random_state=test_seeds[0],
            )

            # Generate base features
            generator.generate_features(sample_feature_params, sample_feature_types)

            # Add perturbations for error propagation study
            all_features = list(sample_feature_params.keys())
            generator.add_perturbations(
                features=all_features,
                perturbation_type="gaussian",
                scale=FEATURE_NOISE_LEVELS[config["noise"]],
            )

            # Create target with polynomial complexity for neural network training
            target_features = all_features[:3]
            target_weights = TARGET_WEIGHTS[: len(target_features)]
            generator.create_target_variable(
                features_to_use=target_features,
                weights=target_weights,
                noise_level=TARGET_NOISE_LEVELS[config["noise"]],
                function_type="polynomial",
            )

            # Verify dataset is suitable for neural network research
            assert generator.data.shape[0] == config["samples"]
            assert "target" in generator.data.columns

            # Check that perturbations increased complexity appropriately
            if config["noise"] != "low_noise":
                feature_complexity = generator.data[all_features].std().mean()
                assert feature_complexity > 0.1, (
                    f"Features should have sufficient complexity for {config['desc']}"
                )

            # Verify target is suitable for neural network training
            target_range = (
                generator.data["target"].max() - generator.data["target"].min()
            )
            assert target_range > 0.5, (
                f"Target should have sufficient range for NN training: {target_range}"
            )

        # Test multiple function types for different NN scenarios
        function_scenarios = [
            ("linear", "Simple regression for baseline NN training"),
            ("polynomial", "Complex non-linear for advanced NN testing"),
            ("logistic", "Classification scenarios for NN research"),
        ]

        for func_type, description in function_scenarios:
            gen = generator_factory(
                n_samples=100, n_features=len(sample_feature_params)
            )
            gen.generate_features(sample_feature_params, sample_feature_types)

            gen.create_target_variable(
                features_to_use=list(sample_feature_params.keys())[:3],
                weights=TARGET_WEIGHTS[:3],
                noise_level=TARGET_NOISE_LEVELS["medium_noise"],
                function_type=func_type,
            )

            if func_type == "logistic":
                # Logistic should be bounded [0,1] for classification
                assert all(0 <= val <= 1 for val in gen.data["target"])
                target_range = gen.data["target"].max() - gen.data["target"].min()
                assert target_range > 0.01, (
                    f"Logistic target range insufficient: {target_range}"
                )
            else:
                # Linear and polynomial should have reasonable variance
                assert gen.data["target"].std() > 0.1, (
                    f"{func_type} target should have variation"
                )

    def test_data_export_and_visualisation_integration(
        self, generator_with_sample_data, sample_feature_params, tmp_path
    ):
        """Test data export and visualisation functionality"""
        # Test data export functionality
        test_file_path = tmp_path / "test_export.csv"

        # Export data using actual method name
        generator_with_sample_data.data.to_csv(str(test_file_path))

        # Verify file exists
        assert test_file_path.exists()

        # Import and verify using pandas
        imported_data = pd.read_csv(test_file_path, index_col=0)

        # Should match original data structure
        assert imported_data.shape == generator_with_sample_data.data.shape
        assert list(imported_data.columns) == list(
            generator_with_sample_data.data.columns
        )

        # Verify data integrity (allowing for minor floating point differences)
        pd.testing.assert_frame_equal(
            imported_data,
            generator_with_sample_data.data,
            check_dtype=False,
            atol=1e-10,
        )

        # Test visualisation integration
        features_to_visualise = list(sample_feature_params.keys())[:3]

        # This should not raise an error
        try:
            generator_with_sample_data.visualise_features(
                features=features_to_visualise,
                max_features_to_show=3,
                n_bins=20,
                save_to_dir=str(tmp_path),
            )
            # If we get here, visualisation worked
            visualisation_works = True
        except Exception as e:
            pytest.fail(f"Visualisation failed: {e}")
            visualisation_works = False

        assert visualisation_works, "Visualisation should work with generated data"

    @pytest.mark.slow
    def test_performance_with_large_dataset(self, large_dataset):
        """Test performance with larger dataset using large_dataset fixture"""
        # Verify the large dataset was created successfully
        assert large_dataset.data is not None
        assert large_dataset.data.shape[0] >= 1000  # At least 1000 samples
        assert large_dataset.data.shape[1] >= 5  # At least 5 features

        # Basic statistical checks for neural network compatibility
        for col in large_dataset.data.columns:
            feature_data = large_dataset.data[col]
            assert abs(feature_data.mean()) < 2.0, (
                f"Feature {col} mean should be reasonable: {feature_data.mean()}"
            )
            assert 0.1 < feature_data.std() < 5.0, (
                f"Feature {col} std should be reasonable: {feature_data.std()}"
            )
            assert not feature_data.isnull().any(), (
                f"Feature {col} should not have null values"
            )

        # Test that large dataset can handle additional operations

        # Should be able to add perturbations to large dataset
        available_features = large_dataset.data.columns.tolist()[:3]
        large_dataset.add_perturbations(
            features=available_features,
            perturbation_type="gaussian",
            scale=FEATURE_NOISE_LEVELS["medium_noise"],
        )

        # Verify perturbations were applied successfully
        assert large_dataset.data.shape[0] >= 1000  # Still has same number of samples
        assert not large_dataset.data.isnull().any().any()  # No null values introduced

    def test_comprehensive_error_handling_and_edge_cases(
        self, generator_factory, sample_feature_params, sample_feature_types
    ):
        """Test comprehensive error handling and edge cases throughout the workflow"""
        generator = generator_factory(
            n_samples=50, n_features=len(sample_feature_params)
        )

        # Test error when creating target before generating features
        with pytest.raises(ValueError, match="No data generated"):
            generator.create_target_variable(
                features_to_use=["feature_1"],
                weights=[1.0],
                noise_level=0.1,
                function_type="linear",
            )

        # Generate features first
        generator.generate_features(sample_feature_params, sample_feature_types)

        # Test error with invalid perturbation parameters
        with pytest.raises(ValueError, match="Feature.*not found"):
            generator.add_perturbations(
                features=["nonexistent_feature"],
                perturbation_type="gaussian",
                scale=0.1,
            )

        # Test error with invalid target parameters
        with pytest.raises(ValueError, match="Features not found"):
            generator.create_target_variable(
                features_to_use=["nonexistent_feature"],
                weights=[1.0],
                noise_level=0.1,
                function_type="linear",
            )

        # Test edge cases that should work
        single_feature = [list(sample_feature_params.keys())[0]]

        # Should work with single feature
        generator.create_target_variable(
            features_to_use=single_feature,
            weights=[1.0],
            noise_level=0.0,  # No noise
            function_type="linear",
        )

        assert "target" in generator.data.columns
        assert generator.data["target"].notna().all()

        # Test multiple target creation calls (should overwrite)
        original_target = generator.data["target"].copy()

        generator.create_target_variable(
            features_to_use=single_feature,
            weights=[2.0],  # Different weight
            noise_level=0.0,
            function_type="linear",
        )

        # Should have overwritten the target
        assert not generator.data["target"].equals(original_target)
        assert len([col for col in generator.data.columns if col == "target"]) == 1
