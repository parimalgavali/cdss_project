# 0013: Missingness Shortcut and Missingness-Only Control Model

Date: 2026-10-28

## Context

The first real-data run (test fold 0, on the user's machine) showed that the
amount of recorded data, not the measurements, carries most of the label:

- Recorded (non-padding) hours per stay: mean 3.4 for positive stays vs 16.2
  for negative stays. 23,043 of 57,040 stays have at most 5 recorded hours.
- Padding is not only leading: 49,860 stays start with padding, 22,140 end
  with it, and most stays switch between recorded and padding at least once.
- A stay-level Logistic Regression on the 30 recorded/padding flags alone
  (stratified 5-fold, aggregate check) reached AUROC 0.943, AUPRC 0.793
  (prevalence 0.169). Per 6-hour window it reached AUROC 0.83-0.85.
- Fold 0 baselines on mean/min/max features: LR AUROC 0.910 / AUPRC 0.594,
  RF 0.920 / 0.636. Per-horizon AUROC peaks near horizon 15 and is lowest at
  horizon 1 (RF 0.797), the opposite of a physiological early-warning signal.
  With the first-crossing EDA rule, about 95% of positive stays alert at a
  mean of about 23.5 hours, so EDA saturates.
- LSTM fold 0 stopped after 11 epochs with best epoch 1.

Likely cause (not confirmed): the CSV is aligned to onset, so positive stays
have fewer recorded hours before the decision point. Professor-style imputation
turns padding into recognisable mean/interpolated values, so models can detect
it.

## Decision

- Windows carry `observed_mask` (windows x 6, 1 = recorded hour), saved in the
  windows file next to `observed_hours`.
- New baseline `missingness_only` (`--models miss`): class-weighted Logistic
  Regression on the 6 `observed_mask` flags of each window, with the same
  threshold, metrics and EDA as the other baselines. `run_cv.py` runs it by
  default (`--baseline-models lr rf miss`), so every summary table and plot
  shows how much of a model's score the recording pattern alone explains.
- No change to preprocessing or labels: the shortcut is reported, not hidden.
  Whether to restrict the analysis (for example `--min-observed-hours 6`, or
  stays with enough recorded hours) is a question for the professor.

## Rationale

A model can only be credited with early physiological detection to the extent
it beats `missingness_only` at the same horizon. A control model in the
existing pipeline makes this visible in every fold and figure without a
separate analysis.

## Consequences

- Windows files built before this decision lack `observed_mask`;
  `missingness_only` raises an error asking to rebuild them (`run_cv.py`
  rebuilds them for every fold anyway).
- The RNN EDA comparison also lists `missingness_only` as a reference.
