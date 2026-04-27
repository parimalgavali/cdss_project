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

For the first baseline sanity checks, labeled rows can be filtered to
`sepsis in {0, 1}`. Before final modeling, the team should decide whether
`NaN` labels represent unlabeled horizons, censored examples, or values that
should be derived from patient-level onset information.
