from config_helpers import *
from itertools import combinations
import copy 

# --- Configuration for Combinations ---
COMBO_SIZE = 5

def main():
    print(f"\n--- Generating Combined Perturbations (Size {COMBO_SIZE}, Threshold: {SEPARATION_REDUCTION_THRESHOLD*100}%) ---")
    print(f"Original Dataset Separability (d_D): {ORIGINAL_SEPARATION:.4f}")

    base_perts = []
    # ... (code to build base_perts list is the same) ...
    for i, feature in enumerate(FEATURES):
        shifts_to_add = POSITIVE_SIGMA_SHIFTS if FEATURE_MEANS[feature]['signal'] > FEATURE_MEANS[feature]['noise'] else NEGATIVE_SIGMA_SHIFTS
        for shift in shifts_to_add:
            base_perts.append({'name': f"pert_f{i}{CLASS_NAME_IN_FILENAME}_by{format_val(shift)}s", 'cfg': {'feature': feature, 'class_label': CLASS_LABEL_TO_PERTURB, 'sigma_shift': shift}})
        for scale in SCALE_FACTORS:
            base_perts.append({'name': f"pert_f{i}{CLASS_NAME_IN_FILENAME}_scale{format_val(scale)}", 'cfg': {'feature': feature, 'class_label': CLASS_LABEL_TO_PERTURB, 'scale_factor': scale}})

    generated_count = 0
    checked_count = 0
    for combo in combinations(base_perts, COMBO_SIZE):
        checked_count += 1
        features_in_combo = [p['cfg']['feature'] for p in combo]
        if len(set(features_in_combo)) != COMBO_SIZE:
            continue

        perturbed_means = copy.deepcopy(ORIGINAL_NOISE_MEANS)
        perturbed_vars = copy.deepcopy(ORIGINAL_NOISE_VARS)
        # ... (code to calculate new means and vars is the same) ...
        for p in combo:
            feature = p['cfg']['feature']
            if 'sigma_shift' in p['cfg']:
                shift_val = p['cfg']['sigma_shift'] * FEATURE_STD_DEVS[feature]['noise']
                perturbed_means[feature] += shift_val
            if 'scale_factor' in p['cfg']:
                scale_val = p['cfg']['scale_factor']
                perturbed_means[feature] *= scale_val
                perturbed_vars[feature] *= (scale_val**2)

        new_separation = calculate_separability(perturbed_means, perturbed_vars)

        # --- Check against the threshold ---
        if new_separation < (ORIGINAL_SEPARATION * SEPARATION_REDUCTION_THRESHOLD):
            generated_count += 1
            sorted_combo = sorted(combo, key=lambda x: x['name'])
            filename = "_".join([p['name'] for p in sorted_combo]) + ".yml"
            config = {'perturbation_settings': [p['cfg'] for p in sorted_combo]}
            write_yaml_file(config, filename)

    rejected_count = checked_count - generated_count
    print(f"\n--- Generation Complete ---")
    print(f"Checked {checked_count} total combinations.")
    print(f"Generated {generated_count} configurations (passing threshold).")
    print(f"Rejected {rejected_count} configurations (not passing threshold or invalid).")

if __name__ == "__main__":
    main()
