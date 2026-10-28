# Milestone 0002: 6-Hour Sliding Windows With Horizon Labels

Date: 2026-10-28

## Purpose

Turn the preprocessed hourly table into the samples that the baselines and the
LSTM/GRU models train on: overlapping 6-hour windows, each tagged with its
prediction horizon (25 hours down to 1 hour before the decision point).
Definition and rationale: decision `0007`.

## What The Pipeline Does

1. Loads `data/processed/mimic_sepsis_preprocessed.csv` (milestone 0001
   output, including `is_padding`, `fold`, `split` from decision `0006`).
2. Validates that every stay has exactly 30 rows covering `hour_before` 30..1,
   that features are imputed and numeric, and that `stay_has_sepsis` is
   constant within each stay.
3. Orders each stay chronologically and slides a 6-hour window with a 1-hour
   stride: 25 windows per stay.
4. Stores per window: `X` (6 x 27 features), `y` (`stay_has_sepsis`),
   `horizon`, `stay_id`, `observed_hours`, `fold`, `split`.
5. Prints a per-horizon summary (window count, positives, mean observed
   hours, fully padded windows).
6. Optionally writes `data/processed/windows_w6.npz`.

## How To Run

```bash
source .venv/bin/activate
python scripts/preprocess_data.py --overwrite   # refresh the processed CSV
python scripts/build_windows.py --dry-run       # summary only
python scripts/build_windows.py                 # write windows_w6.npz
```

Load in Python:

```python
from pathlib import Path
from cdss_project.windows import load_windows

windows = load_windows(Path("data/processed/windows_w6.npz"))
train = windows.split == "train"
X_train, y_train, horizon_train = windows.X[train], windows.y[train], windows.horizon[train]
```

## Verification

- `pytest`: window shapes, horizon order 25..1, chronological order inside
  windows, no cross-stay windows, observed-hours counts, filtering, grid and
  label validation, npz round trip.
- End-to-end run on a 2,000-stay synthetic CSV that follows the data contract:
  50,000 windows (25 per stay), all horizons with 2,000 windows.
- Not yet run on the real CSV (not available in the development VM). Expected
  full-dataset size: 1,426,000 windows.

## What This Milestone Does Not Do Yet

- No window filtering by default (see `--min-observed-hours`).
- No feature scaling or categorical one-hot encoding.
- No baseline aggregation (mean/min/max) or model training.
- No validation split inside the training folds.
