# Milestone 0003: Per-Hour Modeling and Evaluation Plan

Date: 2026-05-26

## Goal

Update windowing for per-hour prediction, train baseline and sequence models on
all selected windowing techniques, and generate comparison tables and plots that
can be shown to the professor.

## Inputs

Use the preprocessed CSV as the source table:

```text
data/processed/mimic_sepsis_preprocessed.csv
```

Do not modify the original CSV or the preprocessed CSV.

## Windowing Techniques

Generate one prediction opportunity per patient hour, from `hour_before = 30`
down to `hour_before = 1`.

### 1. Fixed 3-Hour Padded Windows

Use a maximum window length of 3 hours.

Early predictions are padded:

```text
prediction hour 30: [30, PAD, PAD]
prediction hour 29: [30, 29, PAD]
prediction hour 28: [30, 29, 28]
prediction hour 27: [29, 28, 27]
...
```

Expected output:

```text
data/windows/fixed_3h_padded_windows.npz
```

### 2. Fixed 6-Hour Padded Windows

Use a maximum window length of 6 hours.

Early predictions are padded:

```text
prediction hour 30: [30, PAD, PAD, PAD, PAD, PAD]
prediction hour 29: [30, 29, PAD, PAD, PAD, PAD]
prediction hour 28: [30, 29, 28, PAD, PAD, PAD]
...
prediction hour 25: [30, 29, 28, 27, 26, 25]
prediction hour 24: [29, 28, 27, 26, 25, 24]
...
```

Expected output:

```text
data/windows/fixed_6h_padded_windows.npz
```

### 3. Expanding Per-Hour Windows

Use all available history so far, padded to the full 30-hour length.

```text
prediction hour 30: [30, PAD, PAD, ..., PAD]
prediction hour 29: [30, 29, PAD, ..., PAD]
prediction hour 28: [30, 29, 28, PAD, ..., PAD]
...
prediction hour 1:  [30, 29, 28, ..., 1]
```

Expected output:

```text
data/windows/expanding_min_1h_windows.npz
```

## Label Policy

Use the existing derived stay-level label:

```text
label = stay_has_sepsis
```

Do not fill, overwrite, or directly train on the raw `sepsis` column.

Each generated window inherits the stay-level label. This is a v1 modeling
policy. Horizon-specific relabeling can be added later if the professor provides
a stricter onset-time definition.

## Models

Train the following models on each of the three windowing techniques:

```text
Logistic Regression
Random Forest
LSTM
GRU
```

This creates 12 model/window experiments:

```text
3 windowing techniques x 4 models = 12 experiments
```

## Train/Validation/Test Splits

Split by `stay_id`, never by individual windows.

Use a deterministic split:

```text
train: 64%
validation: 16%
test: 20%
random_state: 42
```

This keeps all windows from the same patient stay in only one split and avoids
patient leakage.

Do not oversample or undersample the test set. Do not oversample or undersample
in v1 training. If class weighting is used, record it in the model result file.

## Model Input Handling

### Logistic Regression and Random Forest

Flatten each window:

```text
time_steps x features -> one long feature vector
```

Examples:

```text
fixed 3h: 3 x 27 = 81 features
fixed 6h: 6 x 27 = 162 features
expanding: 30 x 27 = 810 features
```

For padded windows, append mask values to the flattened features so classical
models can distinguish real observations from padding.

### LSTM and GRU

Keep the sequence shape:

```text
windows x time_steps x features
```

Use masks so padded time steps do not behave like real observations.

## Evaluation Metrics

Calculate metrics overall and per prediction hour.

Overall metrics:

```text
AUROC
AUPRC
F1 score
precision
recall / sensitivity
specificity
accuracy
confusion matrix values
```

Per-hour metrics:

```text
hour_before = 30
hour_before = 29
...
hour_before = 1
```

For each hour, calculate:

```text
AUROC
AUPRC
F1 score
precision
sensitivity
specificity
accuracy
positive window count
negative window count
```

If a metric cannot be calculated for an hour because only one class is present,
write `NaN` for that metric instead of failing silently.

Use validation predictions to choose a classification threshold that maximizes
F1. Apply that threshold to the test set for F1, precision, sensitivity,
specificity, accuracy, and confusion-matrix calculations. AUROC and AUPRC use
predicted probabilities directly.

## Output Folder Structure

Generated experiment outputs should stay out of Git under:

```text
outputs/experiments/
```

Use this structure:

```text
outputs/experiments/
  fixed_3h_padded/
    logistic_regression/
      model_config.json
      metrics_overall.csv
      metrics_by_hour.csv
      predictions_by_hour/
        hour_30.csv
        hour_29.csv
        ...
        hour_01.csv
      plots/
        auroc_by_hour.png
        auprc_by_hour.png
        f1_by_hour.png
    random_forest/
      ...
    lstm/
      ...
    gru/
      ...

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
      auroc_comparison.png
      auprc_comparison.png
      f1_comparison.png
      sensitivity_comparison.png
      specificity_comparison.png
```

## Implementation Steps

1. Update windowing code to support fixed padded per-hour windows.
2. Generate fixed 3h padded, fixed 6h padded, and expanding min 1h window files.
3. Add utilities for loading `.npz` window datasets and metadata.
4. Add deterministic stay-level train/validation/test splitting.
5. Add metric utilities for overall and per-hour evaluation.
6. Add classical model training for Logistic Regression and Random Forest.
7. Add PyTorch training for LSTM and GRU.
8. Add experiment runner that trains all 12 experiments.
9. Add result aggregation and plot generation.
10. Update project memory, decision docs, and milestone summary.

## Expected Commands

Window generation:

```bash
python scripts/build_windows.py --window-type fixed --window-length 3 --include-partial --output data/windows/fixed_3h_padded_windows.npz
python scripts/build_windows.py --window-type fixed --window-length 6 --include-partial --output data/windows/fixed_6h_padded_windows.npz
python scripts/build_windows.py --window-type expanding --min-window-length 1 --output data/windows/expanding_min_1h_windows.npz
```

Dry-run the same window generation without writing large files:

```bash
python scripts/build_windows.py --window-type fixed --window-length 3 --include-partial --dry-run
python scripts/build_windows.py --window-type fixed --window-length 6 --include-partial --dry-run
python scripts/build_windows.py --window-type expanding --min-window-length 1 --dry-run
```

Run all experiments:

```bash
python scripts/run_experiments.py --all
```

Run a small smoke experiment before full training:

```bash
python scripts/run_experiments.py --all --smoke --smoke-stays 40 --epochs 1
```

Aggregate comparison outputs:

```bash
python scripts/summarize_experiments.py
```

## Current Implementation Status

Implemented:

- fixed padded per-hour windows using `--include-partial`;
- expanding per-hour windows with `--min-window-length 1`;
- `.npz` window loading utilities;
- deterministic stay-level train/validation/test splitting;
- overall and per-hour metrics;
- Logistic Regression and Random Forest training;
- LSTM and GRU training;
- experiment runner and comparison summarizer scripts.

Current full-dataset dry-run counts:

```text
Fixed 3-hour padded windows:
Total windows: 1,711,200
Positive windows: 289,710
Negative windows: 1,421,490

Fixed 6-hour padded windows:
Total windows: 1,711,200
Positive windows: 289,710
Negative windows: 1,421,490

Expanding windows from 1 hour:
Total windows: 1,711,200
Positive windows: 289,710
Negative windows: 1,421,490
```

## Notes For Professor Explanation

The next milestone will compare how model performance changes by:

```text
windowing technique
model type
prediction hour
```

This lets us answer:

```text
Which windowing strategy performs best?
Which model performs best?
How early can the model make useful predictions?
How do metrics change from hour_before 30 down to hour_before 1?
```

The main deliverables are comparison tables and plots, not only trained model
files.
