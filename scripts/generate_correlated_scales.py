from config_helpers import *
from itertools import combinations
import copy

def main():
    print(f"\n--- Generating Correlated Scales (w/ Threshold: {SEPARATION_REDUCTION_THRESHOLD*100}%) ---")
    print(f"Original Dataset Separability (d_D): {ORIGINAL_SEPARATION:.4f}")

    generated_count = 0
    rejected_count = 0

    for size in range(2, len(FEATURES) + 1):
        matrix = CORRELATION_MATRICES.get(size)
        if not matrix:
            continue

        for indices in combinations(range(len(FEATURES)), size):
            names = [FEATURES[i] for i in indices]
            for scale in SCALE_FACTORS:
                perturbed_means = copy.deepcopy(ORIGINAL_NOISE_MEANS)
                perturbed_vars = copy.deepcopy(ORIGINAL_NOISE_VARS)
                for feature in names:
                    perturbed_means[feature] *= scale
                    perturbed_vars[feature] *= (scale**2)
                new_separation = calculate_separability(perturbed_means, perturbed_vars)
                # --- Check against the threshold ---
                if new_separation < (ORIGINAL_SEPARATION * SEPARATION_REDUCTION_THRESHOLD):
                    # E.g.: pert_corr04s_scale1p5.yml for features 0, 4, scale 1.5
                    f_str = "".join(str(i) for i in indices)
                    filename = f"pert_corr{f_str}{CLASS_NAME_IN_FILENAME}_scale{format_val(scale)}.yml"
                    config = {
                        'perturbation_settings': [{
                            'type': 'correlated',
                            'class_label': CLASS_LABEL_TO_PERTURB,
                            'features': names,
                            'correlation_matrix': matrix,
                            'scale_factor': scale,
                            # Add description optionally:
                            'description': "Sensor drift correlation" if len(names)==2 else ""
                        }]
                    }
                    write_yaml_file(config, filename)
                    generated_count += 1
                else:
                    rejected_count += 1

    print(f"\n--- Generation Complete ---")
    print(f"Generated {generated_count} configurations (passing threshold).")
    print(f"Rejected {rejected_count} configurations (not passing threshold).")

if __name__ == "__main__":
    main()

