# Milestone 0001: Data Audit and Professor-Style Preprocessing

Date: 2026-04-27

## Purpose

This milestone prepares the original ICU time-series CSV for later sliding
window construction and model training while preserving the original file.

The original dataset remains at:

```text
data/mimic_sepsis_onset_prediction.csv
```

The processed copy is written separately to:

```text
data/processed/mimic_sepsis_preprocessed.csv
```

Both files stay local and ignored by Git.

## What The Pipeline Does

1. Loads the original CSV.
2. Removes exact duplicate `stay_id + hour_before` rows.
3. Fails if duplicate stay-hour rows contain conflicting values.
4. Splits stays into train/test groups by `stay_id` using an 80/20 split.
5. Fits global feature means using training stays only.
6. Applies professor-style feature imputation:
   - linear interpolation within each patient's hourly time series;
   - train-set mean fallback for remaining missing feature values.
7. Leaves the `sepsis` column untouched.
8. Adds stay-level derived label columns:
   - `stay_has_sepsis`
   - `observed_label_count`
9. Writes a new processed CSV without modifying the original CSV.

## Important Label Policy

The `sepsis` column is not statistically imputed.

Rows with `sepsis == NaN` and no clinical feature values are treated as
structural empty rows. The patient/stay outcome is derived only from observed
labels:

```text
stay_has_sepsis = max(observed sepsis values for the stay)
```

## Why This Matches The Professor's Direction

The preprocessing follows the professor's 2025 paper:

- aggregate features into hourly time windows;
- retain the most recent measurement when multiple values occur in an hour;
- use linear interpolation within each patient time series;
- use a global mean when a feature remains unavailable for a patient.

In this implementation, global means are fit on training stays only to reduce
train/test leakage.

## How To Run

Activate the local environment:

```bash
source .venv/bin/activate
```

Run a dry run without writing files:

```bash
python scripts/preprocess_data.py --dry-run
```

Create the processed CSV copy:

```bash
python scripts/preprocess_data.py
```

If the output file already exists and should be replaced:

```bash
python scripts/preprocess_data.py --overwrite
```

## Current Full-Dataset Dry-Run Result

```text
Rows after deduplication: 1,711,200
Unique stays: 57,040
Train stays: 45,632
Test stays: 11,408
Feature columns imputed: 27
Missing feature values before: 34,516,021
Missing feature values after: 0
Stay-level outcomes: positive=9,657, negative=47,383
```

## Current Output Verification

The processed CSV was created at:

```text
data/processed/mimic_sepsis_preprocessed.csv
```

Verification results:

- processed shape check: `1,711,200` rows;
- unique stays: `57,040`;
- duplicate `stay_id + hour_before` rows after preprocessing: `0`;
- derived columns present: `stay_has_sepsis`, `observed_label_count`;
- original CSV SHA-256 before and after preprocessing:
  `7a13cf25bc2deea011a148a9d34895b937d1eef34005939d1190d3dff797ac75`.

## What This Milestone Does Not Do Yet

- It does not build 6-hour sliding windows.
- It does not create validation splits.
- It does not train baseline or neural-network models.
- It does not oversample or undersample train, validation, or test data.
- It does not modify the original CSV.
