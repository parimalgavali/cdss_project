# Session Handoff

Date: 2026-05-27

This file is for continuing the project from a new account or new assistant
session. Read this file first, then read `AGENTS.md`, `docs/project_memory.md`,
and the decision/milestone files listed below.

## Current Git State

Current branch:

```text
codex-sliding-window-milestone
```

Important: the latest windowing/modeling work is currently **local and
uncommitted**. Do not delete the folder, reset Git, switch branches, or reclone
before saving this work.

Current uncommitted tracked/created files include:

```text
docs/decisions/README.md
docs/project_memory.md
docs/decisions/0005-windowing-direction.md
docs/decisions/0006-per-hour-modeling-evaluation.md
docs/milestones/0002-windowing.md
docs/milestones/0003-modeling-and-evaluation-plan.md
scripts/build_windows.py
scripts/run_experiments.py
scripts/summarize_experiments.py
src/cdss_project/experiments.py
src/cdss_project/windowing.py
tests/test_experiments.py
tests/test_windowing.py
```

There is also a local compatibility edit in:

```text
src/cdss_project/preprocessing.py
```

That edit preserves the old function name
`transform_with_professor_imputation` while supporting the newer
`transform_imputation` name.

## Files To Read First

Read these in order:

```text
AGENTS.md
docs/project_memory.md
docs/decisions/README.md
docs/decisions/0005-windowing-direction.md
docs/decisions/0006-per-hour-modeling-evaluation.md
docs/milestones/0002-windowing.md
docs/milestones/0003-modeling-and-evaluation-plan.md
```

## Current Project Direction

The project is a CDSS/sepsis early prediction project using ICU hourly
time-series data.

Completed or implemented locally:

```text
1. Repository scaffold and memory docs
2. Data audit and label policy
3. Professor-style preprocessing
4. Sliding/expanding window generation
5. Per-hour padded window generation
6. Experiment utilities for Logistic Regression, Random Forest, LSTM, and GRU
7. Overall and per-hour metric generation
8. Comparison table and plot summarization scripts
```

The current goal is to generate professor-facing comparisons across:

```text
windowing technique x model type x prediction hour
```

Windowing techniques:

```text
fixed_3h_padded
fixed_6h_padded
expanding_min_1h
```

Models:

```text
logistic_regression
random_forest
lstm
gru
```

Total planned experiments:

```text
3 windowing techniques x 4 models = 12 experiments
```

## Data And Generated Files

The raw and generated data files are intentionally ignored by Git.

Raw dataset:

```text
data/mimic_sepsis_onset_prediction.csv
```

Preprocessed CSV:

```text
data/processed/mimic_sepsis_preprocessed.csv
```

Older generated window files currently present locally may include:

```text
data/windows/fixed_3h_windows.npz
data/windows/fixed_6h_windows.npz
data/windows/expanding_min_3h_windows.npz
```

The newer per-hour padded window files still need to be generated if they are
not present:

```text
data/windows/fixed_3h_padded_windows.npz
data/windows/fixed_6h_padded_windows.npz
data/windows/expanding_min_1h_windows.npz
```

Experiment outputs will be generated under:

```text
outputs/experiments/
```

Do not commit `data/`, `outputs/`, generated model files, generated plots, or
clinical data.

## Core Policy Decisions

Label policy:

```text
sepsis NaN values are not filled.
stay_has_sepsis = max(observed sepsis values for the stay)
```

Window label policy:

```text
Each window currently inherits stay_has_sepsis.
```

This is a v1 modeling policy. If the professor later gives a stricter onset-time
definition, horizon-specific labels can be added later.

Train/test policy:

```text
Split by stay_id, not by individual windows.
```

This avoids putting windows from the same patient stay into both train and test.

Current deterministic split:

```text
train: 64%
validation: 16%
test: 20%
random_state: 42
```

Oversampling/undersampling:

```text
Do not oversample or undersample the test set.
Do not oversample or undersample in v1 training.
```

Class imbalance is handled with model class weights / positive loss weights
where applicable.

## Main Scripts

### 1. Build Window Files

Generate per-hour padded fixed 3-hour windows:

```bash
python scripts/build_windows.py --window-type fixed --window-length 3 --include-partial --output data/windows/fixed_3h_padded_windows.npz --overwrite
```

Generate per-hour padded fixed 6-hour windows:

```bash
python scripts/build_windows.py --window-type fixed --window-length 6 --include-partial --output data/windows/fixed_6h_padded_windows.npz --overwrite
```

Generate expanding windows from the first hour:

```bash
python scripts/build_windows.py --window-type expanding --min-window-length 1 --output data/windows/expanding_min_1h_windows.npz --overwrite
```

Dry-run checks:

```bash
python scripts/build_windows.py --window-type fixed --window-length 3 --include-partial --dry-run
python scripts/build_windows.py --window-type fixed --window-length 6 --include-partial --dry-run
python scripts/build_windows.py --window-type expanding --min-window-length 1 --dry-run
```

Expected dry-run count for each per-hour method:

```text
Total windows: 1,711,200
Positive windows: 289,710
Negative windows: 1,421,490
```

### 2. Run Experiments

Smoke test all 12 experiments first:

```bash
python scripts/run_experiments.py --all --smoke --smoke-stays 40 --epochs 1
```

Run all full experiments:

```bash
python scripts/run_experiments.py --all
```

Run one windowing technique at a time:

```bash
python scripts/run_experiments.py --windows fixed_3h_padded
python scripts/run_experiments.py --windows fixed_6h_padded
python scripts/run_experiments.py --windows expanding_min_1h
```

Run one model for one windowing technique:

```bash
python scripts/run_experiments.py --windows fixed_3h_padded --models logistic_regression
python scripts/run_experiments.py --windows fixed_3h_padded --models random_forest
python scripts/run_experiments.py --windows fixed_3h_padded --models lstm
python scripts/run_experiments.py --windows fixed_3h_padded --models gru
```

Notes:

```text
fixed_3h_padded is expected to be fastest.
fixed_6h_padded is expected to be slower.
expanding_min_1h is expected to be slowest, especially for LSTM/GRU.
```

### 3. Summarize Results

After experiments finish:

```bash
python scripts/summarize_experiments.py
```

Main outputs:

```text
outputs/experiments/comparison/all_metrics_overall.csv
outputs/experiments/comparison/all_metrics_by_hour.csv
outputs/experiments/comparison/model_window_comparison.csv
outputs/experiments/comparison/plots/
```

## Output Structure

Experiment outputs are organized like this:

```text
outputs/experiments/
  fixed_3h_padded/
    logistic_regression/
      model_config.json
      metrics_overall.csv
      metrics_by_hour.csv
      predictions_by_hour/
    random_forest/
    lstm/
    gru/

  fixed_6h_padded/
    logistic_regression/
    random_forest/
    lstm/
    gru/

  expanding_min_1h/
    logistic_regression/
    random_forest/
    lstm/
    gru/

  comparison/
    all_metrics_overall.csv
    all_metrics_by_hour.csv
    model_window_comparison.csv
    plots/
```

## Verification Status

The most recent full test verification before this handoff was:

```text
35 passed
```

A synthetic smoke run of all 12 model/window combinations also completed, and
comparison CSV/plot outputs were generated under `/private/tmp`.

Before continuing work, run:

```bash
python -m pytest
```

## Recommended Next Steps

1. Activate the virtual environment.
2. Run tests.
3. Generate the three new per-hour window files.
4. Run one smoke experiment.
5. Run one real experiment first, preferably:

```bash
python scripts/run_experiments.py --windows fixed_3h_padded --models logistic_regression
```

6. Run `scripts/summarize_experiments.py`.
7. Inspect the generated comparison CSVs.
8. Continue with the remaining models/windowing techniques.

## Transition Warning

This repository currently contains important uncommitted work. A new session can
continue safely if it uses this same local folder, but GitHub will not contain
the latest work until it is committed and pushed.
