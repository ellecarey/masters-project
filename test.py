from pathlib import Path
import json

originals = [
    "models/n1000_f_init5_cont0_disc5_sep5p1_seed0_mlp_001_optimal_metrics.json",
    "models/n1000_f_init5_cont0_disc5_sep5p1_seed1_mlp_001_optimal_metrics.json",
    "models/n1000_f_init5_cont0_disc5_sep5p1_seed2_mlp_001_optimal_metrics.json",
    "models/n1000_f_init5_cont0_disc5_sep5p1_seed3_mlp_001_optimal_metrics.json",
    "models/n1000_f_init5_cont0_disc5_sep5p1_seed4_mlp_001_optimal_metrics.json"
]
perturbed = [
    "models/n1000_f_init5_cont0_disc5_sep5p1_pert_f4n_by1p0s_seed0_mlp_001_optimal_metrics.json",
    "models/n1000_f_init5_cont0_disc5_sep5p1_pert_f4n_by1p0s_seed1_mlp_001_optimal_metrics.json",
    "models/n1000_f_init5_cont0_disc5_sep5p1_pert_f4n_by1p0s_seed2_mlp_001_optimal_metrics.json",
    "models/n1000_f_init5_cont0_disc5_sep5p1_pert_f4n_by1p0s_seed3_mlp_001_optimal_metrics.json",
    "models/n1000_f_init5_cont0_disc5_sep5p1_pert_f4n_by1p0s_seed4_mlp_001_optimal_metrics.json"
]

for fname in originals + perturbed:
    try:
        with open(fname, 'r') as f:
            data = json.load(f)
        print(fname, "OK", list(data.keys()))
    except Exception as e:
        print(fname, "FAILED", e)