# Milestone: Observed-Window Evaluation

## What

`summarize_results.py` also scores every model on windows with at least
`--min-observed-hours` (default 6) recorded hours (decision `0014`).

## Why

On the real data the recording pattern nearly determines the label
(decision `0013`); this table shows performance where it cannot help.

## Run / Verify

```bash
python scripts/summarize_results.py --overwrite                   # default N = 6
python scripts/summarize_results.py --min-observed-hours 0 --overwrite   # off
pytest tests/test_reporting.py
```

## Out of Scope

- Retraining on fully observed windows only, or dropping `ethnicity`
  (pending the professor's decision).
