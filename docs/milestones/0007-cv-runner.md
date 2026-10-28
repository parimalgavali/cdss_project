# Milestone 0007: One-Command Cross-Validation and Synthetic Data

Date: 2026-10-28

## Purpose

Run the full 5-fold experiment on the real CSV with one command, and test the
whole pipeline end to end in CI without clinical data. Rules: decision `0012`.

## How To Run

```bash
source .venv/bin/activate
python scripts/run_cv.py --dry-run                 # print every command
python scripts/run_cv.py --skip-rnn                # baselines only (fast first pass)
python scripts/run_cv.py --device cuda --explain   # full run, saves models + explanations
python scripts/run_cv.py --resume                  # skip folds that already finished
python scripts/run_cv.py --folds 0 --max-epochs 5  # one fold, short training
```

Synthetic data:

```bash
python scripts/make_synthetic_data.py --n-stays 2000
python scripts/run_cv.py --csv data/synthetic/synthetic_sepsis.csv --max-epochs 3
```

## Verification

- `pytest`:
  - the synthetic frame passes `inspect_dataframe`, has leading structural
    padding, constant per-stay labels, and drift that grows towards hour 1;
  - `run_cv.py --dry-run` lists the steps per fold in order;
  - an end-to-end run on 150 synthetic stays (2 of 3 folds, 1 epoch, explain
    4 windows) writes the summary, models and explanations, and `--resume`
    skips both finished folds.
- Not yet run on the real CSV.

## Out Of Scope

- Parallel folds, and caching per-fold windows between runs.
