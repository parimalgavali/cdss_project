# Milestone 0002: Sliding and Expanding Window Generation

Date: 2026-05-12

## Purpose

This milestone converts the preprocessed hourly patient-stay table into
model-ready time-series windows. The original CSV and the preprocessed CSV are
not modified.

Default input:

```text
data/processed/mimic_sepsis_preprocessed.csv
```

Default generated outputs:

```text
data/windows/fixed_6h_windows.npz
data/windows/fixed_3h_windows.npz
data/windows/expanding_min_3h_windows.npz
```

Generated window files stay local and ignored by Git.

## What The Pipeline Does

1. Reads the preprocessed CSV.
2. Orders each stay chronologically as `hour_before` `30 -> 29 -> ... -> 1`.
3. Builds fixed-length sliding windows:
   - 6-hour windows as the primary baseline;
   - 3-hour windows as the shorter-context comparison.
4. Builds expanding windows from a configurable minimum length.
5. Uses `stay_has_sepsis` as the v1 window label.
6. Excludes identifier and label columns from the model input features.
7. Stores feature tensors as `float32` arrays for ML compatibility and lower
   memory usage.
8. Stores window metadata for explainability and debugging.

## Window Metadata

Each generated window records:

```text
stay_id
window_type
window_length
window_start_hour_before
window_end_hour_before
prediction_hour_before
label
```

For fixed windows, the prediction hour is the final hour inside the window. For
example:

```text
[30, 29, 28, 27, 26, 25] -> prediction_hour_before = 25
```

## Label Policy

Window labels use the derived stay-level outcome:

```text
label = stay_has_sepsis
```

The raw `sepsis` column is not filled, overwritten, or used directly as a
window label in this milestone.

## How To Run

Activate the local environment:

```bash
source .venv/bin/activate
```

Dry-run fixed 6-hour windows:

```bash
python scripts/build_windows.py --window-type fixed --window-length 6 --dry-run
```

Dry-run fixed 3-hour windows:

```bash
python scripts/build_windows.py --window-type fixed --window-length 3 --dry-run
```

Dry-run expanding windows from 3 hours:

```bash
python scripts/build_windows.py --window-type expanding --min-window-length 3 --dry-run
```

Write fixed 6-hour windows:

```bash
python scripts/build_windows.py --window-type fixed --window-length 6
```

If an output already exists and should be replaced:

```bash
python scripts/build_windows.py --window-type fixed --window-length 6 --overwrite
```

## Current Full-Dataset Dry-Run Results

```text
Fixed 6-hour windows:
Total windows: 1,426,000
Positive windows: 241,425
Negative windows: 1,184,575

Fixed 3-hour windows:
Total windows: 1,597,120
Positive windows: 270,396
Negative windows: 1,326,724

Expanding windows from 3 hours:
Total windows: 1,597,120
Positive windows: 270,396
Negative windows: 1,326,724
```

## What This Milestone Does Not Do Yet

- It does not train Logistic Regression, Random Forest, LSTM, or GRU models.
- It does not perform oversampling or undersampling.
- It does not create horizon-specific labels beyond storing
  `prediction_hour_before` metadata.
- It does not evaluate AUROC, AUPRC, F1, sensitivity, specificity, or Early
  Detection Advantage.
