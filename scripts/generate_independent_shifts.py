from config_helpers import *

def main():
    print(f"\n--- Generating Independent Shifts (w/ Threshold: {SEPARATION_REDUCTION_THRESHOLD*100}%) ---")
    print(f"Original Dataset Separability (d_D): {ORIGINAL_SEPARATION:.4f}")
    generated_count = 0
    rejected_count = 0
    
    for i, feature in enumerate(FEATURES):
        noise_mean = FEATURE_MEANS[feature]['noise']
        signal_mean = FEATURE_MEANS[feature]['signal']
        shifts_to_use = POSITIVE_SIGMA_SHIFTS if signal_mean > noise_mean else NEGATIVE_SIGMA_SHIFTS
        
        for shift in shifts_to_use:
            perturbed_means = copy.deepcopy(ORIGINAL_NOISE_MEANS)
            shift_val = shift * FEATURE_STD_DEVS[feature]['noise']
            perturbed_means[feature] += shift_val
            
            new_separation = calculate_separability(perturbed_means, ORIGINAL_NOISE_VARS)
            
            # --- Check against the threshold ---
            if new_separation < (ORIGINAL_SEPARATION * SEPARATION_REDUCTION_THRESHOLD):
                config = {'perturbation_settings': [{'feature': feature, 'class_label': CLASS_LABEL_TO_PERTURB, 'sigma_shift': shift}]}
                filename = f"pert_f{i}{CLASS_NAME_IN_FILENAME}_by{format_val(shift)}s.yml"
                write_yaml_file(config, filename)
                generated_count += 1
            else:
                rejected_count += 1

    print(f"\n--- Generation Complete ---")
    print(f"Generated {generated_count} configurations (passing threshold).")
    print(f"Rejected {rejected_count} configurations (not passing threshold).")

if __name__ == "__main__":
    main()
