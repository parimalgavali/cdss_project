# Milestone 0008: Persistent-Alert Option for Early Detection Advantage

Date: 2026-10-28

## Purpose

The EDA alert rule is still open with the professor: does the first window
over the threshold count, or must the alert stay on? This adds the second
reading as an option so either answer can be reported. Rule: decision `0008`
amendment.

## How To Run

```bash
source .venv/bin/activate
python scripts/train_baselines.py --alert-persistence 3 --overwrite
python scripts/train_rnn.py --alert-persistence 3 --overwrite
python scripts/summarize_results.py --overwrite
```

Or in one go: `python scripts/run_cv.py --alert-persistence 3 --output-root outputs/persist3`.

Default `--alert-persistence 1` keeps the earlier first-crossing results.
Write persistence runs to separate `--output-dir`s if both are needed.

## Verification

- `pytest`: persistence 1..4 on stays with a single spike, alternating
  alerts, and non-contiguous horizons; invalid persistence is rejected.
- Synthetic smoke run: baselines with persistence 3, then `train_rnn.py`
  with persistence 1 is refused and with persistence 3 writes EDA.
- Not yet run on the real CSV.

## Out Of Scope

- Alerts that may switch off and on again (refractory periods), and
  per-stay alert counts.
