# 0006: Categorical Handling, Padding Flag, and Persisted Fold Policy

Date: 2026-10-28

## Decision

Refine the professor-style preprocessing from `0003` in three ways:

1. **Categoricals are not interpolated or averaged.** Linear interpolation and
   train-set mean fallback apply only to numeric columns
   (`INTERPOLATED_FEATURE_COLUMNS`: static numeric plus dynamic numeric).
   `gender`, `ethnicity`, and `last_careunit` are filled from the same stay
   (forward then backward fill; they are stay-static) and fall back to the
   train-set mode.
2. **Padding rows are flagged.** The processed table carries `is_padding`
   (1 for structural empty rows, computed before imputation) so later steps can
   tell observed hours from fully imputed padding.
3. **Stay folds are persisted and stratified.** Stays are assigned to
   `n_folds` folds (default 5, seed 42), stratified by `stay_has_sepsis`. The
   processed table carries `fold` and `split` (`test` for `--test-fold`,
   default 0, otherwise `train`). Imputation means and modes are fit on the
   `train` split only. This replaces the previous unstratified `--test-size`
   80/20 split that was not saved in the output.

`scripts/inspect_data.py` now reports structural empty rows per `hour_before`
and the value type of each categorical column, so the position of padding in
the 30-hour sequence can be checked on the real CSV.

## Rationale

- Interpolating or averaging category codes can produce values that are not a
  valid category (for example `last_careunit = 2.7`). The professor's paper
  describes interpolation for measurements, not categories.
- About 53% of rows are structural padding. After imputation they look like
  real hours, so windowing and evaluation need an explicit flag to measure how
  much of each window is observed.
- The outline plans 5-fold cross-validation. Recording the fold of every stay
  lets later milestones reuse exactly the split the imputation statistics were
  fit on, which prevents train/test leakage. Stratification keeps the ~17%
  positive rate stable across folds.

## Consequences

- Processed CSV gains `is_padding`, `fold`, and `split` columns.
- The CLI uses `--n-folds` / `--test-fold` instead of `--test-size`.
- The processed CSV reflects one test fold. For full 5-fold CV, either rerun
  preprocessing per `--test-fold` or refit imputation per fold in memory with
  `assign_stay_folds`, `fit_feature_means`, and `fit_categorical_modes`.
- Whether windows dominated by padding should be filtered is still open and
  depends on the padding-position report from the real CSV.
