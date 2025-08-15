from config_helpers import *
import copy

# ------------------------------------------------------------
# Helper
# ------------------------------------------------------------
def chunk_name(feature: str, ptype: str, value: float) -> str:
    idx = FEATURES.index(feature)
    if ptype == "sigma_shift":
        return f"pert_f{idx}{CLASS_NAME_IN_FILENAME}_by{format_val(value)}s"
    return f"pert_f{idx}{CLASS_NAME_IN_FILENAME}_scale{format_val(value)}"

def build_cfg_and_name(ptype: str, value: float):
    """
    Returns (cfg_list, filename) for a uniform perturbation that
    touches every feature with the same value.
    """
    cfg, name_chunks = [], []
    for f in FEATURES:
        entry = {"feature": f, "class_label": CLASS_LABEL_TO_PERTURB}
        if ptype == "sigma_shift":
            entry["sigma_shift"] = value
        else:
            entry["scale_factor"] = value
        cfg.append(entry)
        name_chunks.append(chunk_name(f, ptype, value))
    filename = "_".join(sorted(name_chunks)) + ".yml"
    return cfg, filename

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print(f"\n--- Generating uniform perturbations for ALL features "
          f"(threshold {SEPARATION_REDUCTION_THRESHOLD*100:.0f} %) ---")
    print(f"Original separability d_D = {ORIGINAL_SEPARATION:.4f}")

    checked = generated = 0

    # 1) σ-shifts (positive ⊕ negative)
    for shift in POSITIVE_SIGMA_SHIFTS + NEGATIVE_SIGMA_SHIFTS:
        checked += 1
        cfg, fname = build_cfg_and_name("sigma_shift", shift)

        means = copy.deepcopy(ORIGINAL_NOISE_MEANS)
        vars_ = copy.deepcopy(ORIGINAL_NOISE_VARS)
        for f in FEATURES:
            means[f] += shift * FEATURE_STD_DEVS[f]["noise"]

        if calculate_separability(means, vars_) \
           < ORIGINAL_SEPARATION * SEPARATION_REDUCTION_THRESHOLD:
            generated += 1
            write_yaml_file({"perturbation_settings": cfg}, fname)

    # 2) scale factors
    for scale in SCALE_FACTORS:
        checked += 1
        cfg, fname = build_cfg_and_name("scale_factor", scale)

        means = {f: ORIGINAL_NOISE_MEANS[f] * scale for f in FEATURES}
        vars_  = {f: ORIGINAL_NOISE_VARS[f]  * scale**2 for f in FEATURES}

        if calculate_separability(means, vars_) \
           < ORIGINAL_SEPARATION * SEPARATION_REDUCTION_THRESHOLD:
            generated += 1
            write_yaml_file({"perturbation_settings": cfg}, fname)

    # --------------------------------------------------------
    print("\n--- Generation complete ---")
    print(f"Checked   : {checked} candidate perturbations")
    print(f"Generated : {generated} YAML files (passed threshold)")
    print(f"Rejected  : {checked - generated} (above threshold)")

if __name__ == "__main__":
    main()