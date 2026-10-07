# Project Maintenance Notes

This document records the project-health review undertaken after returning to the repository. Its purpose is to distinguish observed failures, implementation decisions, and verified fixes while restoring a reliable test and execution baseline.

## Current Baseline

**Environment**

- Operating system: Windows
- Python: 3.12.10
- Environment/package tool: uv 0.7.5
- Test command used: `uv run python -m pytest`

**Repository state**

- The project uses a `src/`-based structure with modules for synthetic data generation, training, analysis, experiment management, and shared utilities.
- The current data-generation configuration schema uses:
  - `dataset_settings`
  - `create_feature_based_signal_noise_classification`
  - `global_settings`
  - optional `perturbation_settings`
- The current MLP implementation is in `src/training_module/models.py`.
- The currently available model class is `mlp_001`, with constructor parameters `input_size`, `hidden_size`, and `output_size`.

**Test baseline — 2026-10-07**

```text
Command:
uv run python -m pytest src\data_generator_module\tests -q

Result:
24 tests collected
19 passed
5 failed
```

The passing tests cover Gaussian data-generator behaviour and parameter validation.

The five current failures are in:

```text
src/data_generator_module/tests/test_utils.py
```

They concern filename generation and plot-title generation.

## Known Failures

### Training-test import is stale

The training tests import:

```python
from training_module.mlp_model import MLP
```

However, the current model implementation is:

```python
from training_module.models import mlp_001
```

This prevents the full test suite from completing collection.

**Status:** Identified; repair not yet verified.

### Utility tests use an outdated configuration schema

The utility tests use the old configuration structure:

```python
feature_generation:
  feature_types:
```

The current implementation expects:

```python
create_feature_based_signal_noise_classification:
  feature_types:
```

As a result, current utility functions see zero configured continuous/discrete features when the old test dictionaries are used.

**Status:** Identified; tests need migration to the current schema.

### Filename expectations are outdated

The current filename-generation function includes dataset separability and random seed, for example:

```text
n100000_f_init5_cont0_disc5_sep0p0_seed7
```

Older tests expect filenames without separation and seed components.

**Status:** Identified; expected outputs need updating after explicit current-schema test configurations are defined.

### Plot-title expectation is outdated

The current implementation returns:

```text
Distribution of Generated Features
```

An older test expects:

```text
Feature Distribution
```

**Status:** Identified; confirm the newer wording is intended and update the test accordingly.

### Generated report paths are too long for Windows

Some tracked generated report paths exceed the Windows path-length limit because directory names and filenames encode long dataset and perturbation descriptions.

**Status:** Identified; defer refactor until unit tests are stable.

## Decisions Made

### Use the current implementation and example configuration as the source of truth

Tests will be updated only after comparing them with:

- Current source implementation.
- Current example YAML configurations.
- Current model API.
- Current command-line entry points.

The project will not be changed merely to recreate obsolete test imports, old configuration structures, or old filename conventions.

### Repair tests in small layers

Test repair order:

1. Utility tests in `src/data_generator_module/tests/test_utils.py`.
2. Full data-generator test suite.
3. Training-module test imports and expectations.
4. Full test suite.
5. Small end-to-end smoke test.

Each repair will be run and committed separately.

### Use `uv run python -m pytest`

Use:

```powershell
uv run python -m pytest
```

as the standard test command during this maintenance work.

This invocation correctly identifies the project root and is currently the most informative way to run the test suite.

### Do not run full historical experiments yet

Do not run million-sample datasets, broad perturbation sweeps, or full Optuna tuning during the initial repair work.

The first integration target will be a small, deterministic smoke test that verifies:

```text
configuration → dataset generation → model training → metrics/output artefact
```

### Preserve historical work

Existing Git stashes will remain untouched during project-health work. They may contain older experiments or unfinished work and should be reviewed separately after the current branch has a passing baseline.