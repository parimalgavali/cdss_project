# Milestone: Missingness-Only Control Model

## What

- `WindowDataset.observed_mask` and the `missingness_only` baseline
  (decision `0013`).
- `run_cv.py` includes it by default.

## Why

On the real CSV, recorded-hour counts alone separate sepsis from non-sepsis
stays (stay-level AUROC 0.943). The control model shows how much of each
model's per-horizon performance is due to that shortcut.

## Run / Verify

```bash
python scripts/train_baselines.py --models lr rf miss --overwrite
python scripts/run_cv.py               # includes miss by default
pytest tests/test_baselines.py tests/test_windows.py
```

## Out of Scope

- Changing preprocessing or restricting the cohort; that needs the
  professor's decision.
