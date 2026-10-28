# 0010: Cross-Fold Summary and Minimum Reliable Horizon

Date: 2026-10-28

## Decision

- Results are combined across the test folds found under
  `outputs/{baselines,rnn}/test_fold_<k>/` (`src/cdss_project/reporting.py`,
  `scripts/summarize_results.py`).
  - Per model × horizon (including `all`) × metric: the fold mean, the sample
    std (ddof=1, NaN when there is one fold) and `n_folds`.
  - EDA is summarized the same way per model × baseline reference.
- **Minimum reliable horizon** (outline 4.5): per model, the largest horizon
  `h` such that the mean AUROC is ≥ 0.80 at *every* horizon 1..h. It is 0 when
  even horizon 1 misses. The threshold is set with `--auroc-threshold`.
  Requiring all shorter horizons to pass stops one lucky long horizon from
  counting.
- Tables go to `outputs/summary/`. Figures go to `reports/figures/`, both
  git-ignored:
  - `<metric>_by_horizon.png` for each metric: the mean line with a ± std
    band, the x axis running from 25 down to 1, and a dashed threshold line on
    AUROC;
  - `eda_vs_baselines.png`: bars with std error bars.

## Rationale

The outline asks for mean ± std over 5 folds and for "the minimum horizon at
which LSTM/GRU achieve clinically meaningful results (AUROC ≥ 0.80)". The
summary reads only the per-fold CSVs, so it does not depend on how the folds
were produced (one by one or in a loop). Models are compared on the same
folds.

## Consequences

- Partial runs are summarized as they are; check `n_folds` before reporting.
- No statistical tests between models yet. Paired tests across folds can be
  added if the professor wants them.
