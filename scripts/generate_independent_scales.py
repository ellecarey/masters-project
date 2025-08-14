from config_helpers import *

def main():
    print(f"\n--- Generating Independent Scales (w/ Threshold: {SEPARATION_REDUCTION_THRESHOLD*100}%) ---")
    print(f"Original Dataset Separability (d_D): {ORIGINAL_SEPARATION:.4f}")
    generated_count = 0
    rejected_count = 0

    for i, feature in enumerate(FEATURES):
        for scale in SCALE_FACTORS:
            perturbed_means = copy.deepcopy(ORIGINAL_NOISE_MEANS)
            perturbed_vars = copy.deepcopy(ORIGINAL_NOISE_VARS)
            
            perturbed_means[feature] *= scale
            perturbed_vars[feature] *= (scale**2)

            new_separation = calculate_separability(perturbed_means, perturbed_vars)

            # --- Check against the threshold ---
            if new_separation < (ORIGINAL_SEPARATION * SEPARATION_REDUCTION_THRESHOLD):
                config = {'perturbation_settings': [{'feature': feature, 'class_label': CLASS_LABEL_TO_PERTURB, 'scale_factor': scale}]}
                filename = f"pert_f{i}{CLASS_NAME_IN_FILENAME}_scale{format_val(scale)}.yml"
                write_yaml_file(config, filename)
                generated_count += 1
            else:
                rejected_count += 1
    
    print(f"\n--- Generation Complete ---")
    print(f"Generated {generated_count} configurations (passing threshold).")
    print(f"Rejected {rejected_count} configurations (not passing threshold).")

if __name__ == "__main__":
    main()
