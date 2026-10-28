# 0007: Sliding Window and Horizon Definition

Date: 2026-10-28

## Decision

Build overlapping fixed-size windows from the preprocessed hourly table, as
described in section 3.2 of the project outline:

- `hour_before` counts down to the decision point: `30` is the first ICU hour,
  `1` the last. Rows are ordered chronologically (`hour_before` descending)
  inside every window.
- Default window size is 6 hours. Window k covers hours k..k+5 after
  admission; its **horizon** is the `hour_before` of its last row. With 30
  hours per stay this gives 25 windows per stay, horizons 25 down to 1.
- Window label `y = stay_has_sepsis` (the patient develops sepsis by hour 30).
  The dataset has no onset timestamp, so every window of a stay shares the
  stay label, and horizon is defined by window position.
- Windows never cross stays. Every stay must have exactly 30 rows covering
  `hour_before` 30..1; anything else raises an error.
- Feature columns must already be imputed and numeric-encoded (default: all
  27 `FEATURE_COLUMNS`). One-hot encoding of categoricals is left to the
  modeling milestones.
- Each window carries `observed_hours` (non-padding rows, from `is_padding`),
  plus `fold` and `split` from preprocessing.
- **No windows are dropped by default**, to stay aligned with the professor's
  setup. `--min-observed-hours` exists for sensitivity analyses only.
- Output: `data/processed/windows_w6.npz` (compressed NumPy, loaded with
  `allow_pickle=False`), git-ignored like all processed data.

## Rationale

The outline fixes the 6-hour window, 1-hour stride, and 25..1 horizon range.
Keeping all windows matches the professor's experimental setup. Recording
`observed_hours` keeps the padding question measurable: the dry-run summary
prints mean observed hours and fully padded windows per horizon, so the effect
of padding on long horizons can be reported or filtered later without
rebuilding the pipeline.

## Consequences

- Full dataset: 57,040 stays x 25 windows = 1,426,000 windows of shape
  `(6, 27)`, about 0.9 GB as float32 in memory (expected from the contract, not
  yet measured on the real CSV).
- Baselines (milestone 3) aggregate each window to mean/min/max per feature.
  LSTM/GRU (milestone 4) consume `X` directly.
- Horizon-wise metrics group test windows by `horizon`. Early Detection
  Advantage still needs a concrete alert definition (for example, the largest
  horizon at which a stay's prediction first crosses the threshold).
- If padding turns out to dominate long horizons on the real data, revisit
  whether to filter or to report those horizons separately.
