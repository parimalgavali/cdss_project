# Project Memory

This is the continuity file for future sessions. Read it before planning,
coding, reviewing, or changing project direction.

## Current Objective

Build a reproducible CDSS research project for early sepsis onset prediction
using ICU hourly time-series data. The core research question is how predictive
performance changes across horizons from 25 hours before the decision point
down to 1 hour, comparing classical baselines with LSTM and GRU models.

## Current Repository State

- GitHub remote: `git@github.com:parimalgavali/cdss_project.git`
- Main branch: `main`
- Environment style: local `.venv` plus `requirements.txt`
- Dataset path: `data/mimic_sepsis_onset_prediction.csv`
- Dataset is intentionally ignored by Git.
- Reusable code lives in `src/cdss_project/`.
- Exploratory notebooks live in `notebooks/`.
- Documentation lives in `docs/`.
- Tests live in `tests/`.

## Durable Decisions

- Keep `data/` out of Git.
- Keep generated artifacts out of Git unless explicitly requested.
- Keep professor-provided/reference PDFs local unless the user explicitly
  approves publishing them.
- Use scripts plus notebooks: reusable logic in `src/`, notebooks for
  exploration and figures.
- Use `venv + requirements.txt` as the first environment approach.
- Use parallel agents selectively later, once preprocessing/windowing decisions
  are stable. Do not split tightly coupled label/windowing work across agents.
- Treat `sepsis == NaN` as structural padding/no-observation rows, not as an
  unknown clinical outcome and not as a value to statistically impute.
- Derive stay-level sepsis outcome from observed labels only:
  `stay_has_sepsis = max(observed sepsis)`.
- For clinical feature columns, follow the professor's 2025 paper as the
  primary preprocessing style: aggregate to 1-hour windows, keep the most recent
  measurement when multiple values occur in an hour, linearly interpolate
  missing values within each patient's time series, and replace features that
  remain missing after interpolation with a global training-set mean. This
  covers leading/trailing missing values and features entirely missing for a
  patient.
- Categorical columns (`gender`, `ethnicity`, `last_careunit`) are never
  interpolated or averaged: fill within the stay (ffill/bfill), then use the
  train-set mode. Interpolation and mean fallback apply to numeric columns only
  (`INTERPOLATED_FEATURE_COLUMNS`). See `0006`.
- The processed table flags structural empty rows as `is_padding` (computed
  before imputation) and records a persisted, `stay_has_sepsis`-stratified
  `fold` (default 5 folds, seed 42) plus `split` for the chosen test fold.
- Label columns must not be statistically imputed.
- Sliding windows (decision `0007`): 6-hour windows, 1-hour stride, 25 per
  stay; horizon = `hour_before` of the window's last row (25..1); window label
  = `stay_has_sepsis`; windows never cross stays; no windows dropped by
  default (`--min-observed-hours` only for sensitivity analyses).
- Fit global imputation, scaling, and encoding statistics on training stays
  only to avoid train/test leakage.
- Keep the repository threat model in `SECURITY_MODEL.md` and update it when
  the attack surface changes (new dependencies, model loading, deployment).

## Current Data Findings

- CSV shape: `1,711,228` rows by `30` columns.
- Unique stays: `57,040`.
- Expected hour column in the actual CSV is `hour_before`, not `hours_before`.
- `hour_before` ranges from `1` to `30`.
- There are `28` duplicate `stay_id + hour_before` groups (`56` rows involved,
  `28` surplus rows), all from stay `34934165`, which has `58` rows. Plan A
  inspection found these are exact duplicate rows; exact duplicates may be
  dropped, while conflicting duplicates must raise an error.
- Rows with `sepsis == NaN` have zero non-missing clinical feature values in the
  inspected CSV. Current count: `908,798` structural empty rows.
- Stay-level outcomes derived from observed labels: `9,657` positive stays and
  `47,383` negative stays.
- Professor-style preprocessing dry run (before `0006`) deduplicated to
  `1,711,200` rows, imputed `27` feature columns, and reduces missing feature values from
  `34,516,021` to `0` without writing output files.
- Preprocessing can now write a generated processed copy to
  `data/processed/mimic_sepsis_preprocessed.csv`. The original CSV is never
  overwritten. The generated processed copy has `1,711,200` rows, `57,040`
  stays, no duplicate `stay_id + hour_before` rows, and derived
  `stay_has_sepsis` / `observed_label_count` columns.
- Very sparse features include `crp`, `fio2`, `bilirubin_total`, `lactate`,
  `po2`, `leukocytes`, `platelet`, and `creatinine`.

## Next Milestones

### Milestone 2A: Data Audit and Label Policy

- Confirm and document structural empty rows.
- Resolve duplicate `stay_id + hour_before` rows.
- Derive `stay_has_sepsis` from observed labels.
- Define static, dynamic numeric, categorical, label, and identifier columns.
- Add tests for the label policy and duplicate handling.

### Milestone 2B: Preprocessing Transform

- Implement professor-style clinical feature imputation:
  patient-level linear interpolation plus train-set global mean fallback for
  leading/trailing missing values and features entirely missing within a
  patient.
- Fit train-only preprocessing statistics before transforming validation/test
  stays.
- Produce a clean hourly table ready for sliding windows.
- Status: implemented as a dry-run pipeline and as an output-writing pipeline
  that creates a separate processed CSV under `data/processed/`.
- Refined by `0006`: categorical mode fill, `is_padding`, persisted stratified
  `fold` / `split`. The processed CSV must be regenerated on the real data to
  pick up these columns.

### Milestone Documentation

- Each completed milestone should get its own Markdown summary under
  `docs/milestones/`.
- Milestone summaries: `docs/milestones/0001-preprocessing.md`,
  `docs/milestones/0002-windowing.md`.

### Milestone 2: Sliding Windows

- Implemented in `src/cdss_project/windows.py` and
  `scripts/build_windows.py`; output `data/processed/windows_w6.npz`.
- Each window stores `X (6, 27)`, `y`, `horizon`, `stay_id`,
  `observed_hours`, `fold`, `split`.
- Not yet run on the real CSV; expected 1,426,000 windows.

### Later Milestones

- Build Logistic Regression and Random Forest baselines.
- Build PyTorch LSTM and GRU models.
- Evaluate AUROC, AUPRC, F1, sensitivity, specificity, and Early Detection
  Advantage.
- Define the EDA alert rule concretely (for example, the largest horizon at
  which a stay's prediction first crosses the decision threshold).

## Open Questions

- Confirm with the professor whether the provided CSV was derived from
  PhysioNet 2019, MIMIC-IV, or another preprocessing pipeline.
- Decide whether to keep missingness masks as auxiliary features in addition to
  professor-style interpolation, or reserve them for a later sensitivity check.
- Where does padding sit in the 30-hour sequence? Run
  `python scripts/inspect_data.py` on the real CSV and read the
  "Structural empty rows by hour_before" table. If padding concentrates in the
  early hours, long-horizon windows are mostly imputed and filtering by
  observed hours should be considered.
- Confirm whether categorical columns in the real CSV are numeric codes or
  text (the inspection report prints this).
