# 0011: TimeSHAP-Style Explanations and Opt-In RNN Persistence

Date: 2026-10-28

## Decision

- **Persistence (amends `0008`/`0009`):** saving models is opt-in, with
  `scripts/train_rnn.py --save-model`.
  - `rnn.save_rnn` writes the `state_dict` to
    `models/rnn/test_fold_<k>/<cell>.pt` and the encoder arrays to
    `<cell>_encoder.npz`.
  - `rnn.load_rnn` uses `torch.load(weights_only=True)` and
    `np.load(allow_pickle=False)`.
  - `models/` is git-ignored. Baselines are still never saved.
- **Explanations (`src/cdss_project/explain.py`):** Shapley values of the
  predicted probability. A player that is "absent" takes the value of the
  baseline, which is the average encoded training window (TimeSHAP's
  average-event baseline).
  - Event level, `event_shapley`: exact over the 6 hours of a window
    (2^6 = 64 coalitions).
  - Feature level, `feature_shapley`: permutation-sampling Shapley over the
    27 original features (32 permutations by default). The one-hot columns
    of a categorical feature form one player.
  - Both are efficient: per window they sum to f(x) − f(baseline). Tests
    check this.
- `scripts/explain_rnn.py` explains a seeded sample of test-fold windows: by
  default 200 windows from sepsis-positive stays, optionally restricted to
  given `--horizons`.
  - Per-window values and mean |value| importances go to
    `outputs/explain/test_fold_<k>/`.
  - Bar charts go to `reports/figures/`.
- `scripts/summarize_results.py` also writes `outputs/summary/results.md`.
  It contains overall mean ± std, the minimum reliable horizon, EDA and a list
  of figures, as the starting point for the written report.

## Rationale

The outline names TimeSHAP as optional. With windows of only 6 steps, exact
event-level Shapley values are cheap, and feature-level sampling is a direct
computation. This needs no new dependencies. The `timeshap` package would add
shap, plotly, altair, seaborn and `feedzai-altair-theme` for the same
quantities, with TimeSHAP's pruning, which is not needed at 6 steps.

## Consequences

- This is "TimeSHAP-style", not the `timeshap` package. Say so in the report.
- Explanations depend on the baseline choice and use the probability scale.
- The default sample (200 windows, 32 permutations) takes seconds on CPU.
  More windows or permutations give more stable feature rankings.
