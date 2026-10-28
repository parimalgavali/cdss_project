# Milestone 0004: LSTM and GRU Models

Date: 2026-10-28

## Purpose

Train the sequential models the research question is about and compare them
with the baselines per horizon and by Early Detection Advantage (outline 3.4,
4.2). Definition and rationale: decision `0009`.

## What The Pipeline Does

1. Loads `data/processed/windows_w6.npz`.
2. Holds out the validation fold (the same one as the baselines). Fits the
   scaler and one-hot categories on the remaining training stays.
3. Trains `SepsisRNN` (attention + 3 x LSTM/GRU(16) + BatchNorm + dropout)
   with weighted BCE, Adam and early stopping on validation loss.
4. Picks the threshold on the validation fold and scores the test fold.
5. Writes metrics per horizon, predictions, stay alerts, training history and
   EDA against the baseline alerts for the same test fold.

## How To Run

```bash
source .venv/bin/activate
python scripts/train_baselines.py          # first, so EDA has a reference
python scripts/train_rnn.py                # LSTM + GRU, device auto
python scripts/train_rnn.py --models gru --max-epochs 20 --overwrite
```

Results: `outputs/rnn/test_fold_<k>/` (`metrics_by_horizon.csv`,
`eda_vs_baselines.csv`, `<model>_history.csv`, `run_summary.json`, ...).
For 5-fold CV, add `python scripts/train_rnn.py --overwrite` to the loop in
`docs/milestones/0003-baselines.md`.

## Verification

- `pytest`: encoder (train-only standardization, one-hot, unseen codes),
  output shapes, GRU has fewer parameters than LSTM, early stopping, and both
  cells learning a synthetic signal while scoring only test-fold stays.
- End-to-end smoke run on the 2,000-stay synthetic CSV with 5 epochs,
  including EDA against the synthetic baselines.
- Not yet run on the real CSV.

## What This Milestone Does Not Do Yet

- No hyperparameter tuning, saved weights, or mean ± std across folds.
- No plots or TimeSHAP (milestones 5–6).
