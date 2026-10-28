# Milestone 0006: Explanations and Results Report

Date: 2026-10-28

## Purpose

Show which hours and features drive LSTM/GRU sepsis scores (outline 3.6,
optional TimeSHAP), and generate a Markdown results summary for the written
report. Rules: decision `0011`.

## How To Run

```bash
source .venv/bin/activate
python scripts/train_rnn.py --save-model --overwrite     # per test fold
python scripts/explain_rnn.py                            # 200 positive test windows
python scripts/explain_rnn.py --horizons 1 6 12 --overwrite
python scripts/summarize_results.py --overwrite          # + outputs/summary/results.md
```

Outputs:

- `models/rnn/test_fold_<k>/<cell>.pt`, `<cell>_encoder.npz`
- `outputs/explain/test_fold_<k>/<cell>_{event,feature}_{shap,importance}.csv`
- `reports/figures/<cell>_{event,feature}_importance_fold<k>.png`
- `outputs/summary/results.md`

## Verification

- `pytest`:
  - Shapley values are exact on an additive model, at both event and feature
    level;
  - efficiency holds on a real `SepsisRNN`;
  - the save/load round trip gives identical scores;
  - one-hot grouping, the chunked baseline, and the report tables.
- Synthetic smoke run (3 epochs): `--save-model` → `explain_rnn.py` (100
  windows, about 8 s) → `results.md`.
- Not yet run on the real CSV.

## Out Of Scope

- The `timeshap` package and its plots; cell-level (hour × feature)
  explanations.
- Writing the report text itself, and significance tests.
