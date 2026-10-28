# Milestone 0003: Logistic Regression and Random Forest Baselines

Date: 2026-10-28

## Purpose

Validate the pipeline end to end and set the reference performance that the
LSTM/GRU models and the Early Detection Advantage are compared against
(outline 3.3). Definition and rationale: decision `0008`.

## What The Pipeline Does

1. Loads `data/processed/windows_w6.npz` (milestone 0002).
2. Aggregates each window: mean/min/max per numeric feature, plus last-hour
   categorical codes (one-hot encoded inside the model pipeline).
3. Holds out one training fold, fits each model on the rest, and picks the
   decision threshold on the held-out fold (max F1, or `--target-sensitivity`).
4. Refits on all training stays and scores the test fold.
5. Reports AUROC, AUPRC, F1, sensitivity, specificity overall and per horizon,
   plus per-stay warning time (earliest alert) for positive stays.

## How To Run

```bash
source .venv/bin/activate
python scripts/train_baselines.py                      # LR + RF, test fold from the windows file
python scripts/train_baselines.py --models lr          # LR only (fast)
python scripts/train_baselines.py --target-sensitivity 0.8 --overwrite
```

Five-fold CV (each fold needs its own preprocessing because imputation stats
are train-only):

```bash
for k in 0 1 2 3 4; do
  python scripts/preprocess_data.py --test-fold $k --overwrite
  python scripts/build_windows.py --overwrite
  python scripts/train_baselines.py --overwrite
done
```

Results: `outputs/baselines/test_fold_<k>/metrics_by_horizon.csv`,
`run_summary.json`, per-window predictions and per-stay alerts.

## Verification

- `pytest`: aggregation, metric formulas, threshold rules, first-alert
  horizon, EDA pairing, validation-fold choice, and both models learning a
  synthetic signal while scoring only test-fold stays.
- End-to-end run on the 2,000-stay synthetic CSV (preprocess → windows →
  baselines): about 9 s, all output files written.
- Not yet run on the real CSV.

## What This Milestone Does Not Do Yet

- No hyperparameter tuning, SMOTE, or saved model files.
- No mean ± std aggregation across folds or plots (milestone 5/6).
- EDA rule is provisional until confirmed with the professor.
