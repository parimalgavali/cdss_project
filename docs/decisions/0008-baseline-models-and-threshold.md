# 0008: Baseline Models, Threshold Selection, and Early-Detection Definition

Date: 2026-10-28

## Decision

Baselines (outline 3.3) are trained on aggregated window features:

- **Features:** mean, min and max over the 6-hour window for every numeric
  feature (static and dynamic, 24 columns, giving 72). The categorical codes
  `gender`, `ethnicity`, `last_careunit` are taken from the window's last hour
  and one-hot encoded, not averaged. That makes 75 input columns before
  one-hot expansion.
- **Models:** `LogisticRegression(class_weight="balanced", max_iter=2000)`
  after `StandardScaler`, and `RandomForestClassifier(n_estimators=200,
  min_samples_leaf=20, class_weight="balanced_subsample")`. All seeds are 42.
- **Class imbalance:** class weights, the outline's first strategy (3.1.3).
  SMOTE is not used. The outline lists it as "if needed", and oversampling
  overlapping windows of the same stay would duplicate near-identical samples.
- **Decision threshold:** tuned on the precision-recall curve (outline
  3.1.3), never on the test fold. The validation fold is the training fold
  after the test fold, cyclically (fold 1 when fold 0 is the test fold). The
  model is fit on the other training folds and the threshold picked on the
  validation fold: max F1 by default, or with `--target-sensitivity` the
  highest threshold that reaches the target. The model is then refit on all
  training folds and used with that threshold. One threshold is used for all
  horizons.
- **Metrics:** AUROC, AUPRC, F1, sensitivity, specificity on the test fold,
  overall and per horizon (25..1).
- **Early detection (provisional):** a stay's warning time is the largest
  horizon whose window score reaches the threshold, i.e. the earliest alert,
  or 0 if no window alerts. EDA between a model and a reference is the mean
  difference in warning time over positive test stays, paired by stay
  (`early_detection_advantage`).
- **Outputs** go to the git-ignored `outputs/baselines/test_fold_<k>/`:
  `metrics_by_horizon.csv`, per-window `<model>_predictions.csv.gz`, per-stay
  `<model>_stay_alerts.csv`, `run_summary.json`. Fitted models are not
  persisted.

## Rationale

The outline asks for mean/min/max aggregates. Averaging categorical codes
would repeat the bug fixed in decision `0006`. Tuning the threshold on a
persisted training fold keeps the test fold untouched and keeps stays
stratified and whole. Saving per-window predictions lets the later LSTM/GRU
milestone compute EDA against the baselines without retraining them. Not
saving models avoids pickle/joblib loading, which would change the threat
model.

## Consequences

- Full 5-fold CV means repeating preprocessing (`--test-fold k`), windowing
  and baseline training for k = 0..4. Each run writes its own
  `test_fold_<k>` directory, and the outline wants mean ± std across folds.
- The first-alert EDA rule should be confirmed with the professor. A
  persistence rule (alert must stay on for N windows) is a possible
  alternative.
- Amendment (2026-10-28): `--alert-persistence N` (train_baselines.py and
  train_rnn.py, default 1 = first crossing, unchanged) requires the score to
  reach the threshold in N consecutive windows (horizons h+N-1 .. h, i.e. N
  consecutive hours). The alert is raised at the last of them, h, so warning
  time is the horizon at which the alert is confirmed. Non-contiguous horizons
  (windows dropped by `--min-observed-hours`) break a run. `train_rnn.py`
  refuses to compute EDA against baselines run with a different N. Either
  professor answer can now be reported without code changes.
- RF hyperparameters are fixed defaults, not tuned. On ~1.14M training
  windows the random forest is the slowest step.
