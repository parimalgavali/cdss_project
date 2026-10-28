# 0014: Observed-Window Evaluation and Ethnicity-Code Artifact

Date: 2026-10-28

## Context

Decision `0013` showed that recorded-hour counts nearly determine the label.
Fold 0 of the real run (all models trained on all windows) then gave:

| model | AUROC all windows | AUROC windows with 6 recorded hours |
| --- | --- | --- |
| logistic_regression | 0.910 | 0.743 |
| random_forest | 0.920 | 0.850 |
| lstm | 0.914 | 0.766 |
| gru | 0.916 | 0.745 |

Fully observed windows are 39% of test windows but hold only 189 of 1,931
positive stays (prevalence 1.4% vs 16.9%). Both RNNs early-stopped at epoch 1
(validation loss rises from epoch 2) and EDA vs the baselines is about 0 h.
The minimum reliable horizon flips between 0 and 25 on h=1 AUROC 0.797 vs
0.804, so it is fragile under this shortcut.

Explanations rank `ethnicity` first for both RNNs. An aggregate check found
that ethnicity code `1` covers 9,794 stays (17%) and has no sepsis-positive
stay at all; other codes range 13-36% positive. This is almost certainly a
cohort-construction artifact, not clinical signal.

## Decision

- `summarize_results.py --min-observed-hours N` (default 6, 0 disables)
  rescores every model's saved predictions on windows with at least N
  recorded hours, using that model's original threshold. It writes
  `metrics_mean_std_observed_ge<N>.csv`,
  `minimum_reliable_horizon_observed_ge<N>.csv`, an AUROC-by-horizon figure,
  and a section in `results.md`. No retraining is needed.
- The all-window results stay the primary, professor-aligned tables. The
  observed-window tables are reported next to them as the shortcut control.
- `ethnicity` is kept as a feature for now; whether to drop it or code `1`
  stays is a professor decision (recorded as an open question).

## Rationale

Rescoring saved predictions answers "how good is the model where the
recording pattern carries no information" cheaply and for every model,
including the RNNs. Changing the cohort or feature set would change the
professor's method, so it is only proposed.
