# 0003: Preprocessing Direction

Date: 2026-04-27

## Decision

Split preprocessing into two milestones:

1. Data audit and label policy.
2. Preprocessing transform.

For clinical feature columns, use the professor's 2025 paper as the primary
preprocessing style:

- aggregate all features into 1-hour windows;
- when multiple values occur within the same patient-hour-feature, keep the
  most recent measurement;
- linearly interpolate missing values within each patient's time series;
- replace features entirely missing for a patient with a global training-set
  mean.

Fit global imputation, scaling, and encoding statistics on training stays only.
Do not statistically impute labels.

## Rationale

The professor's paper, `docs/Sepsis_Paper_2025_by_Prof.pdf`, used this
preprocessing strategy for the same sepsis-onset prediction research direction.
The user discussed preprocessing with the professor and was specifically guided
toward linear interpolation plus mean fallback.

Earlier discussion noted that ICU missingness can be informative and that
future-looking interpolation is less suitable for a fully causal real-time
system. For this project, however, alignment with the professor's published
experimental setup is the primary goal. The caveat should be documented in
reports: this is an offline experimental preprocessing pipeline, not necessarily
a strict real-time deployment pipeline.

## Consequences

- Start with professor-style interpolation and train-set mean fallback for
  clinical feature columns.
- Keep the `sepsis` label policy separate: do not impute `sepsis` NaNs.
- Use training-set global means for fallback values to avoid train/test leakage.
- Consider missingness masks or forward-fill variants only as later sensitivity
  analyses, not as the primary pipeline.
- Treat class imbalance as a modeling/evaluation concern, not as a reason to
  synthesize time-series rows during preprocessing.
