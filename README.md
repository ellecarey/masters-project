# Masters project: robustness of neural nets under synthetic distribution shift

This repository supports an MSc project on how classification error changes when a neural network, trained on clean synthetic data, is evaluated on controlled perturbations of that data.

The working system is a YAML-driven **generate → tune/train → evaluate → aggregate** loop. The unified CLI is [`experiment_manager.py`](experiment_manager.py). Labels are not linear or logistic functions of the features: each row is drawn from a **signal** Gaussian (`target = 1`) or a **noise** Gaussian (`target = 0`). Perturbations then shift or scale feature distributions (independently or with correlation) so class separability changes.

---

## Requirements

- **Python 3.12+**
- Dependencies declared in [`pyproject.toml`](pyproject.toml) (PyTorch, Optuna, scikit-learn, pandas, matplotlib, and others)

> **Note:** There is no `requirements.txt`.

---

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ellecarey/masters-project.git
   cd masters-project
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   ```
   - **Windows:** `.venv\Scripts\activate`
   - **macOS/Linux:** `source .venv/bin/activate`

3. **Install dependencies:**
   ```bash
   pip install -e .
   ```
   *(If you use `uv`, `uv sync` from the lockfile is equivalent.)*

4. **Run tests:**
   ```bash
   pytest
   ```
   *Tests live under `src/data_generator_module/tests` and `src/training_module/tests`.*

---

## Pipeline

1. **Generate** a dataset family from a base data YAML: evaluation seeds (default 0–9) plus a dedicated training draw (seed 99, filenames use `_training`).
2. **Optionally perturb** evaluation CSVs with a perturbation YAML (the `_training` set is left unperturbed).
3. **Tune** `mlp_001` with Optuna on a subsample of the training CSV (`configs/experiments.yml`), then write an optimal training config and save the model.
4. **Evaluate** that frozen model on every evaluation seed (full CSV; no re-split).
5. **Aggregate** per-family mean ± std, write a global tracking sheet, and compare original vs. perturbed families.

> If `configs/training/generated/*_optimal.yml` and the matching `models/*_optimal_model.pt` already exist, `run-full-pipeline` and `run-perturbation-study` skip tuning.

```text
base data YAML
    → generate-multiseed          data/*_dataset.csv
    → perturb-multiseed (optional)
    → tune-experiment / tune-analysis
    → models/*_optimal_model.pt
    → evaluate-multiseed          models/*_metrics.json
    → aggregate-all               reports/...
```

---

## Project Structure

| Path | Role |
| --- | --- |
| `experiment_manager.py` | CLI for every workflow |
| `src/data_generator_module/` | `GaussianDataGenerator`, validators, multi-seed generate/perturb |
| `src/training_module/` | `mlp_001`, trainer, train/eval CLI, Optuna workers |
| `src/analysis_module/` | Aggregation, family comparison, global tracking, result plots |
| `src/utils/` | Filenames, report paths, conservative CSV cleanup |
| `scripts/` | Helpers that emit perturbation YAML combinatorics |
| `configs/data_generation/` | Dataset YAMLs (base families plus generated seed/pert copies) |
| `configs/perturbation/` | Perturbation recipes (`sigma_shift`, `scale_factor`, correlated blocks) |
| `configs/training/templates/` | Base training YAML for `mlp_001` |
| `configs/training/generated/` | Optimal hyperparameters after tuning |
| `configs/tuning/` | Optuna search space |
| `configs/experiments.yml` | Tuning job definitions (`mlp_001`) |
| `data/` | Generated CSVs (`*_dataset.csv`) |
| `models/` | `.pt` weights and `*_metrics.json` |
| `reports/` | Figures, spreadsheets, comparisons, `global_experiment_tracking.csv` |
| `db/` | Optuna SQLite studies |

> Dataset filenames are derived from the data YAML: `n{samples}_f_init{n}_cont{c}_disc{d}_sep{rss}_[pert_...]_seed{k}`.

---

## CLI Usage

View help for any command:
```bash
python experiment_manager.py <command> --help
```

### Typical Study (One Base, Many Perturbations)
```bash
python experiment_manager.py run-perturbation-study \
  --base-data-config configs/data_generation/n1000_f_init5_cont0_disc5_sep1p8_seed0_config.yml \
  --tuning-job mlp_001 \
  --perturb-configs configs/perturbation/base/pert_f4n_by1p0s.yml
```
- Pass several perturbation files after `--perturb-configs`.
- `--all-perturbations` only loads `*.yml` directly under `configs/perturbation/` (not nested folders such as `disc5_sep1p8/`). Prefer explicit `--perturb-configs` for nested files.
- `--cleanup-generated` deletes evaluation CSVs created by this run after a fresh successful evaluation. Training CSVs and failed runs are kept.

### One-Shot Pipeline (Optional Single Perturbation)
```bash
python experiment_manager.py run-full-pipeline \
  --base-data-config configs/data_generation/n1000_f_init5_cont0_disc5_sep1p8_seed0_config.yml \
  --tuning-job mlp_001 \
  --perturb-config configs/perturbation/base/pert_f4n_by1p0s.yml
```
*Note: `run-pipeline-batch` takes `--base-data-configs` (several bases) and one optional `--perturb-config`.*

---

## Step-by-Step Commands

| Command | Purpose |
| --- | --- |
| `generate` | One dataset from `--config` |
| `generate-multiseed` | Eval seeds + training seed (`--no-training-seed` to skip training) |
| `perturb-multiseed` | Apply `--perturb-config` to a family (`--data-config-base`) |
| `tune-experiment` | Launch Optuna job named in `configs/experiments.yml` |
| `tune-analysis` | Pick a trial and write optimal config + model (`--non-interactive` to take the best trial) |
| `train-single` | Train only on a `*_training_config.yml` with `--optimal-config` |
| `evaluate-multiseed` | Frozen `--trained-model` on a family |
| `aggregate` / `aggregate-all` | Summaries (`aggregate-all` also builds the global sheet and comparisons) |
| `compare-families` | Original vs. one `--perturbation-tag` |
| `visualise-results` | Plots from `reports/global_experiment_tracking.csv` |

> `tune-experiment` / `run-full-pipeline` analysis of trials is interactive unless an optimal model already exists. For unattended selection, run `tune-analysis --non-interactive` explicitly.
> 
> Tuning job `mlp_001` (see `configs/experiments.yml`) uses **18 workers, 20 trials**, and a **10% subsample** of the training CSV. Optuna storage is `db/{study}_tuning.db`.

---

## Key Modules

- **`GaussianDataGenerator`**: Gaussian features, 50/50 signal vs. noise, individual and correlated perturbations, optional plots.
- **`mlp_001`**: One hidden layer; trained with BCE-with-logits, early stopping, and a learning-rate scheduler.
- **Lifecycle Helpers**: Refuse to overwrite existing CSVs/configs; cleanup is receipt-based, not glob-delete.

---

## License

MIT — see [LICENSE.md](LICENSE.md).

## Acknowledgements

Matplotlib, NumPy, Pandas, scikit-learn, PyTorch, and Optuna. With thanks to MSc supervisor Professor Adrian Bevan.