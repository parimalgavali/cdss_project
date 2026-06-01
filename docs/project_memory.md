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
- Label columns must not be statistically imputed.
- Fit global imputation, scaling, and encoding statistics on training stays
  only to avoid train/test leakage.
- Build windowed modeling inputs from the processed CSV, not from the original
  raw CSV.
- For local neural model training on the user's Apple M1 laptop, prefer
  PyTorch's Apple Metal/MPS backend when available, then CUDA if present, then
  CPU as the fallback.
- Use fixed 6-hour sliding windows as the primary windowing baseline, fixed
  3-hour windows as a shorter-context comparison, and expanding windows as the
  professor-requested all-history comparison.
- Window chronology follows `hour_before` descending from `30` to `1`.
- Use `stay_has_sepsis` as the v1 window label. Keep the original `sepsis`
  column unchanged.

## Current Data Findings

- CSV shape: `1,711,228` rows by `30` columns.
- Unique stays: `57,040`.
- Expected hour column in the actual CSV is `hour_before`, not `hours_before`.
- `hour_before` ranges from `1` to `30`.
- There are `28` duplicate `stay_id + hour_before` rows, all from one stay with
  `58` rows. Plan A inspection found these are exact duplicate rows for stay
  `34934165`; exact duplicates may be dropped, while conflicting duplicates
  must raise an error.
- Rows with `sepsis == NaN` have zero non-missing clinical feature values in the
  inspected CSV. Current count: `908,798` structural empty rows.
- Stay-level outcomes derived from observed labels: `9,657` positive stays and
  `47,383` negative stays.
- Professor-style preprocessing dry run deduplicates to `1,711,200` rows,
  imputes `27` feature columns, and reduces missing feature values from
  `34,516,021` to `0` without writing output files.
- Preprocessing can now write a generated processed copy to
  `data/processed/mimic_sepsis_preprocessed.csv`. The original CSV is never
  overwritten. The generated processed copy has `1,711,200` rows, `57,040`
  stays, no duplicate `stay_id + hour_before` rows, and derived
  `stay_has_sepsis` / `observed_label_count` columns.
- Very sparse features include `crp`, `fio2`, `bilirubin_total`, `lactate`,
  `po2`, `leukocytes`, `platelet`, and `creatinine`.
- Windowing dry runs against the processed CSV produced:
  - fixed 6-hour windows: `1,426,000` windows, positive=`241,425`,
    negative=`1,184,575`;
  - fixed 3-hour windows: `1,597,120` windows, positive=`270,396`,
    negative=`1,326,724`;
  - expanding windows from 3 hours: `1,597,120` windows, positive=`270,396`,
    negative=`1,326,724`.
- Window feature arrays are saved as `float32` tensors to reduce memory usage
  and match common ML framework input types.
- Per-hour padded window dry runs against the processed CSV produced one window
  per patient-hour for each method: `1,711,200` windows with positive=`289,710`
  and negative=`1,421,490`.

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

### Milestone Documentation

- Each completed milestone should get its own Markdown summary under
  `docs/milestones/`.
- Current milestone summary: `docs/milestones/0001-preprocessing.md`.

### Later Milestones

- Implement per-hour padded window generation for fixed 3-hour, fixed 6-hour,
  and expanding windows starting from the first available hour.
- Train Logistic Regression, Random Forest, LSTM, and GRU across the three
  per-hour windowing techniques.
- Evaluate overall and per-hour AUROC, AUPRC, F1, precision, sensitivity,
  specificity, accuracy, and confusion-matrix values.
- Generate professor-facing comparison tables and plots under
  `outputs/experiments/`.
- Current plan file:
  `docs/milestones/0003-modeling-and-evaluation-plan.md`.
- Current implementation includes `scripts/run_experiments.py` for model
  training and `scripts/summarize_experiments.py` for comparison tables/plots.
- Full expanding-window training is memory-heavy on the user's 8GB Apple M1
  laptop. If local runs are killed, run the same committed code on stronger
  approved hardware while keeping the dataset local and out of Git.

## Open Questions

- Confirm with the professor whether the provided CSV was derived from
  PhysioNet 2019, MIMIC-IV, or another preprocessing pipeline.
- Decide whether to keep missingness masks as auxiliary features in addition to
  professor-style interpolation, or reserve them for a later sensitivity check.
