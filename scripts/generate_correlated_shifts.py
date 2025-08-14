from config_helpers import *
from itertools import combinations
import copy

def main():
    print(f"\n--- Generating Correlated Shifts (w/ Threshold: {SEPARATION_REDUCTION_THRESHOLD*100}%) ---")
    print(f"Original Dataset Separability (d_D): {ORIGINAL_SEPARATION:.4f}")
    generated_count = 0
    rejected_count = 0

    for size in range(2, len(FEATURES) + 1):
        matrix = CORRELATION_MATRICES.get(size)
        if not matrix: continue

        for indices in combinations(range(len(FEATURES)), size):
            names = [FEATURES[i] for i in indices]
            positive_needed = sum(1 for f in names if FEATURE_MEANS[f]['signal'] > FEATURE_MEANS[f]['noise'])
            shifts_to_use = POSITIVE_SIGMA_SHIFTS if positive_needed >= (len(names) / 2) else NEGATIVE_SIGMA_SHIFTS

            for shift in shifts_to_use:
                perturbed_means = copy.deepcopy(ORIGINAL_NOISE_MEANS)
                for feature in names:
                    shift_val = shift * FEATURE_STD_DEVS[feature]['noise']
                    perturbed_means[feature] += shift_val

                new_separation = calculate_separability(perturbed_means, ORIGINAL_NOISE_VARS)
                
                # --- Check against the threshold ---
                if new_separation < (ORIGINAL_SEPARATION * SEPARATION_REDUCTION_THRESHOLD):
                    f_str = "".join(map(str, indices))
                    config = {'perturbation_settings': [{'type': 'correlated', 'class_label': CLASS_LABEL_TO_PERTURB, 'features': names, 'correlation_matrix': matrix, 'sigma_shift': shift}]}
                    filename = f"pert_corr_f{f_str}{CLASS_NAME_IN_FILENAME}_by{format_val(shift)}s.yml"
                    write_yaml_file(config, filename)
                    generated_count += 1
                else:
                    rejected_count += 1
    
    print(f"\n--- Generation Complete ---")
    print(f"Generated {generated_count} configurations (passing threshold).")
    print(f"Rejected {rejected_count} configurations (not passing threshold).")

if __name__ == "__main__":
    main()
