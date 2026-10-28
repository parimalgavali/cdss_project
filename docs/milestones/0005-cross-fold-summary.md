# Milestone 0005: Cross-Fold Summary and Plots

Date: 2026-10-28

## Purpose

Turn the per-fold baseline and LSTM/GRU results into the outline's reported
numbers: mean ± std per horizon, the minimum reliable horizon, EDA, and the
figures. Rules: decision `0010`.

## How To Run

`python scripts/run_cv.py` (milestone `0007`) runs the loop below for you.

```bash
source .venv/bin/activate
for k in 0 1 2 3 4; do
  python scripts/preprocess_data.py --test-fold $k --overwrite
  python scripts/build_windows.py --overwrite
  python scripts/train_baselines.py --overwrite
  python scripts/train_rnn.py --overwrite
done
python scripts/summarize_results.py --overwrite
```

Outputs:

- `outputs/summary/metrics_mean_std.csv`
- `outputs/summary/minimum_reliable_horizon.csv`
- `outputs/summary/eda_mean_std.csv`
- `reports/figures/{auroc,auprc,f1,sensitivity,specificity}_by_horizon.png`
- `reports/figures/eda_vs_baselines.png`

## Verification

- `pytest`: mean/std/`n_folds` across two synthetic folds, the contiguous
  reliable-horizon rule (including 0), the EDA summary, and that the figures
  are written.
- Smoke run on the synthetic one-fold outputs from milestones 3–4.
- Not yet run on the real CSV.

## Out Of Scope

- Statistical significance tests between models.
- TimeSHAP and the written report (milestone 6).
