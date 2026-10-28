# 0015: Five-Fold Real-Data Results

Date: 2026-10-29

## Context

The full 5-fold run (`run_cv.py --device cpu --explain`, CPU on an 8 GB M1,
25-42 min per fold) finished on the real CSV. Fold 0 started before
`missingness_only` existed, so that control was added afterwards with
`run_cv.py --folds 0 --skip-rnn --baseline-models miss` into a separate
output root. Mean ± std over the 5 test folds:

| model | AUROC all windows | AUPRC all windows | AUROC 6 recorded hours | AUPRC 6 recorded hours |
| --- | --- | --- | --- | --- |
| logistic_regression | 0.910 ± 0.003 | 0.594 ± 0.006 | 0.723 ± 0.017 | 0.060 ± 0.009 |
| random_forest | 0.921 ± 0.003 | 0.638 ± 0.010 | 0.840 ± 0.016 | 0.169 ± 0.034 |
| missingness_only | 0.803 ± 0.003 | 0.360 ± 0.004 | 0.500 | 0.015 ± 0.001 |
| lstm | 0.915 ± 0.002 | 0.625 ± 0.009 | 0.756 ± 0.011 | 0.056 ± 0.008 |
| gru | 0.917 ± 0.003 | 0.629 ± 0.008 | 0.753 ± 0.015 | 0.057 ± 0.004 |

- Fully observed windows: about 111,500 per fold, about 1,630 positive
  (prevalence 1.5%).
- At the all-window thresholds, sensitivity on fully observed windows is
  ≤ 0.03 for every model (specificity ≈ 1.0).
- Minimum reliable horizon (AUROC ≥ 0.80): all windows LR/RF 0, LSTM/GRU 25,
  because the baselines dip just below 0.80 at horizon 1. Fully observed
  windows: RF 25, every other model 0.
- EDA (hours, mean ± std): LSTM vs LR 0.08 ± 0.17, GRU vs LR 0.01 ± 0.28,
  LSTM vs RF -0.02 ± 0.09, GRU vs RF -0.09 ± 0.25 (vs `missingness_only`:
  LSTM -0.84, GRU -0.93, folds 1-4 only, since the fold-0 RNN run predates it).
- RNNs early-stop at epoch 1-2 in every fold.
- Mean |SHAP| feature ranking across folds is led by `ethnicity`
  (LSTM 0.216, GRU 0.122), then `last_careunit`. Static and
  recording-related features outrank vital signs.

## Decision

- These are the reference results of the professor-aligned method. No
  method change is made here.
- In the written report, the all-window tables must be shown next to the
  `missingness_only` row and the fully observed-window table. The all-window
  scores must not be read as physiological early warning.
- The minimum-reliable-horizon contrast (RNN 25 vs baselines 0) is an
  artifact of the 0.80 cut-off at horizon 1 and must not be reported as an
  RNN advantage. EDA shows no RNN advantage.

## Proposed Follow-Ups (pending professor)

1. Make fully observed windows (or stays with enough recorded hours) the
   primary evaluation set, and select thresholds on that subset.
2. Drop `ethnicity`, or the code-`1` stays, and rerun.
3. Confirm the CSV's onset alignment and provenance.

## Consequences

A full real-data run takes about 2.5 h on CPU. Outputs and figures stay
local and git-ignored.
