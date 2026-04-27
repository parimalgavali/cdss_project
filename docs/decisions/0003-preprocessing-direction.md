# 0003: Preprocessing Direction

Date: 2026-04-27

## Decision

Split preprocessing into two milestones:

1. Data audit and label policy.
2. Preprocessing transform.

Clinical feature columns may be imputed, but missingness should remain visible
to the model through masks and, where useful, time-since-last-observed features.

Fit global imputation, scaling, and encoding statistics on training stays only.

## Rationale

ICU missingness is often informative. Plain interpolation can hide that signal,
and interpolation that uses future values can leak information into early
prediction tasks.

The dataset has very sparse labs and many padded rows, so preprocessing choices
need to be tested before model work begins.

## Consequences

- Start with conservative preprocessing.
- Prefer prior-observation-aware transforms over future-looking interpolation.
- Add missingness masks for clinical features.
- Add train-only fallback imputation for features that are never observed in a
  patient/window.
- Treat class imbalance as a modeling/evaluation concern, not as a reason to
  synthesize time-series rows during preprocessing.
