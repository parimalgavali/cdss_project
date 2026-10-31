# Milestone: Five-Fold Real-Data Run

## What

Ran the full pipeline (LR, RF, `missingness_only`, LSTM, GRU, explanations,
summary) on all 5 folds of the real CSV. Results and interpretation are in
decision `0015`.

## Why

This is the first complete real-data evaluation. It is also where the
recording-pattern shortcut (decision `0013`) and the ethnicity-code artifact
(decision `0014`) were measured across all folds.

## Run / Verify

```bash
caffeinate -i python scripts/run_cv.py --csv <path/to/csv> --device cpu --explain
python scripts/summarize_results.py --overwrite   # includes the observed-window section
```

`outputs/summary/results.md` holds the tables; figures go to `reports/figures/`.

## Out of Scope

- Changing the cohort, the feature set, or threshold selection (proposed in
  decision `0015`, pending the professor).
- The written report text.
