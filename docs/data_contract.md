# Data Contract

The project expects a local CSV at:

```text
data/mimic_sepsis_onset_prediction.csv
```

This file is intentionally ignored by Git because it is a large MIMIC-derived
clinical dataset.

## Required Identifier and Label Columns

- `stay_id`: unique ICU stay identifier.
- `hour_before`: integer hour index in the local CSV. The project outline uses
  the name `hours_before`, but the available CSV uses `hour_before`.
- `sepsis`: binary label where `1` indicates sepsis and `0` indicates no sepsis.
  Current inspection shows many `NaN` values, so label handling must be explicit
  before modeling.

## Required Feature Columns

General features:

- `gender`
- `ethnicity`
- `admission_age`
- `height`
- `weight`

Diagnosis features:

- `diabetes`
- `sofa_24hours`

Vital signs and labs:

- `platelet`
- `leukocytes`
- `po2`
- `fio2`
- `lactate`
- `creatinine`
- `bilirubin_total`
- `gcs`
- `crp`
- `dbp`
- `glucose`
- `heart_rate`
- `mbp`
- `resp_rate`
- `sbp`
- `temperature`
- `spo2`
- `urineoutput`

Therapy and unit context:

- `ventilation_status`
- `last_careunit`

## Current Label Policy

Treat `sepsis == NaN` rows as structural padding/no-observation rows. Do not
statistically impute `sepsis`, and do not convert label NaNs to `0`.

Derive stay-level outcome from observed labels only:

```text
stay_has_sepsis = max(observed sepsis values for the stay)
```

This policy is recorded in `docs/decisions/0002-sepsis-label-nan-policy.md`.

Numeric clinical feature columns should be imputed during preprocessing
according to the professor's 2025 paper: patient-level linear interpolation,
with global training-set mean fallback for features entirely missing within a
patient. Categorical columns (`gender`, `ethnicity`, `last_careunit`) are filled
within the stay and fall back to the training-set mode. Label columns must not
be statistically imputed.

## Processed Output Columns

`data/processed/mimic_sepsis_preprocessed.csv` contains all input columns plus:

- `is_padding`: `1` when the row was a structural empty row before imputation.
- `stay_has_sepsis`, `observed_label_count`: stay-level derived labels.
- `fold`: stay-level fold (stratified by `stay_has_sepsis`).
- `split`: `test` for the chosen test fold, otherwise `train`. Imputation
  statistics are fit on `train` only.
