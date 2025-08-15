from config_helpers import *
from itertools import combinations
import copy

# --- Configuration for Combinations ---
COMBO_SIZE = 10

from config_helpers import *
from itertools import combinations
import copy

# --- Configuration for Combinations ---
COMBO_SIZE = 10

def main():
    print(f"\n--- Generating Combined SHIFT Perturbations (Same-Value, Size {COMBO_SIZE}, Threshold: {SEPARATION_REDUCTION_THRESHOLD*100}%) ---")
    print(f"Original Dataset Separability (d_D): {ORIGINAL_SEPARATION:.4f}")

    # Determine which features are eligible for positive or negative shifts
    positive_shift_features = [f for f in FEATURES if FEATURE_MEANS[f]['signal'] > FEATURE_MEANS[f]['noise']]
    negative_shift_features = [f for f in FEATURES if FEATURE_MEANS[f]['signal'] <= FEATURE_MEANS[f]['noise']]
    
    generated_count = 0
    checked_count = 0

    # --- Process POSITIVE shifts ---
    print("\n--- Processing POSITIVE Shifts ---")
    for shift in POSITIVE_SIGMA_SHIFTS:
        # Only check combinations if there are enough eligible features
        if len(positive_shift_features) >= COMBO_SIZE:
            for feature_combo in combinations(positive_shift_features, COMBO_SIZE):
                checked_count += 1
                
                perturbed_means = copy.deepcopy(ORIGINAL_NOISE_MEANS)
                for feature in feature_combo:
                    shift_val = shift * FEATURE_STD_DEVS[feature]['noise']
                    perturbed_means[feature] += shift_val
                
                new_separation = calculate_separability(perturbed_means, ORIGINAL_NOISE_VARS)

                if new_separation < (ORIGINAL_SEPARATION * SEPARATION_REDUCTION_THRESHOLD):
                    generated_count += 1
                    pert_settings = [{'feature': f, 'class_label': CLASS_LABEL_TO_PERTURB, 'sigma_shift': shift} for f in sorted(feature_combo)]
                    pert_names = [f"pert_f{FEATURES.index(p['feature'])}{CLASS_NAME_IN_FILENAME}_by{format_val(shift)}s" for p in pert_settings]
                    filename = "_".join(pert_names) + ".yml"
                    config = {'perturbation_settings': pert_settings}
                    write_yaml_file(config, filename)

    # --- Process NEGATIVE shifts ---
    print("\n--- Processing NEGATIVE Shifts ---")
    for shift in NEGATIVE_SIGMA_SHIFTS:
        if len(negative_shift_features) >= COMBO_SIZE:
            for feature_combo in combinations(negative_shift_features, COMBO_SIZE):
                checked_count += 1

                perturbed_means = copy.deepcopy(ORIGINAL_NOISE_MEANS)
                for feature in feature_combo:
                    shift_val = shift * FEATURE_STD_DEVS[feature]['noise']
                    perturbed_means[feature] += shift_val
                
                new_separation = calculate_separability(perturbed_means, ORIGINAL_NOISE_VARS)

                if new_separation < (ORIGINAL_SEPARATION * SEPARATION_REDUCTION_THRESHOLD):
                    generated_count += 1
                    pert_settings = [{'feature': f, 'class_label': CLASS_LABEL_TO_PERTURB, 'sigma_shift': shift} for f in sorted(feature_combo)]
                    pert_names = [f"pert_f{FEATURES.index(p['feature'])}{CLASS_NAME_IN_FILENAME}_by{format_val(shift)}s" for p in pert_settings]
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
