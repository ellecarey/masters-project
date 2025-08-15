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

def process_and_write_scale_combo(combo):
    """
    Checks a scale combination, calculates its separability, and writes the config
    if it passes the threshold. Returns 1 if a file was generated, 0 otherwise.
    """
    # Ensure each feature is perturbed only once in the combination
    features_in_combo = [p['cfg']['feature'] for p in combo]
    if len(set(features_in_combo)) != COMBO_SIZE:
        return 0 # Invalid: same feature perturbed twice

    # --- Apply scale perturbations ---
    perturbed_means = copy.deepcopy(ORIGINAL_NOISE_MEANS)
    perturbed_vars = copy.deepcopy(ORIGINAL_NOISE_VARS)

    for p in combo:
        feature = p['cfg']['feature']
        scale_val = p['cfg']['scale_factor']
        perturbed_means[feature] *= scale_val
        perturbed_vars[feature] *= (scale_val**2)

    new_separation = calculate_separability(perturbed_means, perturbed_vars)

    # --- Check against the threshold and write file ---
    if new_separation < (ORIGINAL_SEPARATION * SEPARATION_REDUCTION_THRESHOLD):
        sorted_combo = sorted(combo, key=lambda x: x['name'])
        filename = "_".join([p['name'] for p in sorted_combo]) + ".yml"
        config = {'perturbation_settings': [p['cfg'] for p in sorted_combo]}
        write_yaml_file(config, filename)
        return 1
    
    return 0

def main():
    print(f"\n--- Generating Combined SCALE Perturbations (Size {COMBO_SIZE}, Threshold: {SEPARATION_REDUCTION_THRESHOLD*100}%) ---")
    print(f"Original Dataset Separability (d_D): {ORIGINAL_SEPARATION:.4f}")

    # --- Build a list of base scale perturbations ---
    base_scales = []
    for i, feature in enumerate(FEATURES):
        for scale in SCALE_FACTORS:
            base_scales.append({
                'name': f"pert_f{i}{CLASS_NAME_IN_FILENAME}_scale{format_val(scale)}", 
                'cfg': {'feature': feature, 'class_label': CLASS_LABEL_TO_PERTURB, 'scale_factor': scale}
            })

    generated_count = 0
    checked_count = 0
    
    # --- Process combinations of SCALES only ---
    print(f"\n--- Checking {COMBO_SIZE}-feature combinations of SCALES ---")
    for combo in combinations(base_scales, COMBO_SIZE):
        checked_count += 1
        generated_count += process_and_write_scale_combo(combo)

    rejected_count = checked_count - generated_count
    
    print(f"\n--- Generation Complete ---")
    print(f"Checked {checked_count} total combinations.")
    print(f"Generated {generated_count} configurations (passing threshold).")
    print(f"Rejected {rejected_count} configurations (not passing threshold or invalid).")

if __name__ == "__main__":
    main()
