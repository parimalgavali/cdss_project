# 0006: Per-Hour Modeling and Evaluation

Date: 2026-05-26

## Decision

Update the modeling pipeline to predict at every patient hour, from
`hour_before = 30` down to `hour_before = 1`.

Use three windowing techniques for the next comparison:

- fixed 3-hour padded windows;
- fixed 6-hour padded windows;
- expanding windows starting from 1 hour.

Train four models for each windowing technique:

- Logistic Regression;
- Random Forest;
- LSTM;
- GRU.

Evaluate metrics overall and per prediction hour. Generate comparison tables and
plots under `outputs/experiments/`.

## Rationale

The professor requested per-hour prediction and per-hour performance reporting.
Padding early fixed windows allows fixed 3-hour and fixed 6-hour methods to
predict from the first available hour, instead of waiting until 3 or 6 hours are
available. Masks identify real time steps versus padded positions.

This setup makes the comparison fair across methods because each windowing
technique now produces one prediction opportunity per patient-hour.

## Consequences

- Per-hour padded window outputs produce `1,711,200` windows each on the current
  processed CSV.
- Classical models flatten windows and append mask features.
- LSTM and GRU consume sequence-shaped windows and use masks for padded time
  steps.
- Metrics include AUROC, AUPRC, F1, precision, sensitivity, specificity,
  accuracy, and confusion-matrix values.
- The validation set chooses an F1-maximizing threshold; the test set reports
  final metrics with that threshold.
- Generated windows, trained models, predictions, plots, and result tables stay
  out of Git.
