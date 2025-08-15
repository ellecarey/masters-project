from config_helpers import *
from itertools import combinations
import copy

# --- Configuration for Combinations ---
COMBO_SIZE = 10

def main():
    print(f"\n--- Generating Combined SCALE Perturbations (Same-Value, Size {COMBO_SIZE}, Threshold: {SEPARATION_REDUCTION_THRESHOLD*100}%) ---")
    print(f"Original Dataset Separability (d_D): {ORIGINAL_SEPARATION:.4f}")

    generated_count = 0
    checked_count = 0

    # Iterate through each specific scale factor
    for scale in SCALE_FACTORS:
        print(f"\n--- Processing Scale Factor: {scale} ---")
        # Generate combinations of features to apply this scale to
        for feature_combo in combinations(FEATURES, COMBO_SIZE):
            checked_count += 1
            
            perturbed_means = copy.deepcopy(ORIGINAL_NOISE_MEANS)
            perturbed_vars = copy.deepcopy(ORIGINAL_NOISE_VARS)

            for feature in feature_combo:
                perturbed_means[feature] *= scale
                perturbed_vars[feature] *= (scale**2)
            
            new_separation = calculate_separability(perturbed_means, perturbed_vars)

            # Check against threshold
            if new_separation < (ORIGINAL_SEPARATION * SEPARATION_REDUCTION_THRESHOLD):
                generated_count += 1
                
                pert_settings = [{'feature': f, 'class_label': CLASS_LABEL_TO_PERTURB, 'scale_factor': scale} for f in sorted(feature_combo)]
                pert_names = [f"pert_f{FEATURES.index(p['feature'])}{CLASS_NAME_IN_FILENAME}_scale{format_val(scale)}" for p in pert_settings]
                filename = "_".join(pert_names) + ".yml"
                config = {'perturbation_settings': pert_settings}
                write_yaml_file(config, filename)

    rejected_count = checked_count - generated_count
    
    print(f"\n--- Generation Complete ---")
    print(f"Checked {checked_count} total combinations.")
    print(f"Generated {generated_count} configurations (passing threshold).")
    print(f"Rejected {rejected_count} configurations (not passing threshold).")

if __name__ == "__main__":
    main()