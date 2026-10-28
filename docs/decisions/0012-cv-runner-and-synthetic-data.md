# 0012: Cross-Validation Runner and Synthetic Data

Date: 2026-10-28

## Decision

- `scripts/run_cv.py` runs the whole experiment: for each requested test fold
  it calls `preprocess_data.py`, `build_windows.py`, `train_baselines.py`,
  `train_rnn.py` and optionally `explain_rnn.py`, then runs
  `summarize_results.py` once.
- Each step runs as a subprocess of the existing script with
  `sys.executable` and an argument list (no shell). The scripts stay the single
  implementation of each step; the runner only wires paths and options.
- The processed CSV and windows file are per-fold scratch files
  (`<work-dir>/cv_preprocessed.csv`, `cv_windows_w6.npz`) overwritten for each
  fold, because imputation statistics are refit on every fold's training
  stays. Results keep the existing `test_fold_<k>/` layout.
- `--resume` skips a fold whose result directories already contain
  `run_summary.json`; the first failing step stops the run with its exit code.
- `cdss_project.synthetic.make_synthetic_frame` generates a random CSV that
  follows the data contract: 30 rows per stay, leading structural padding
  (`sepsis` and all features NaN), constant per-stay labels, and a drift in
  five vitals for positive stays that grows from 0 at `hour_before=30` to
  `signal` SD at `hour_before=1`.

- Amendment (2026-10-28): `--alert-persistence N` is passed to both
  `train_baselines.py` and `train_rnn.py`, so EDA stays like for like
  (decision `0008` amendment). Use a separate `--output-root` per N.

## Rationale

- The 5-fold shell loop was easy to get wrong (forgetting `--test-fold`,
  mixing folds, rerunning finished folds after an interruption). One command
  with resume makes the real-data run reproducible.
- The clinical CSV can never be in CI. A committed generator plus an
  end-to-end test catches breakage between steps that unit tests miss.
- The horizon-dependent drift makes short horizons easier than long ones, so
  per-horizon plots on synthetic data have the expected shape.

## Consequences

- Synthetic results say nothing about clinical performance; they only test
  that the pipeline runs.
- Adding a pipeline step means adding a command in `run_cv.fold_commands`.
