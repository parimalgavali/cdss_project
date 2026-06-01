# 0005: Windowing Direction

Date: 2026-05-12

## Decision

Implement both fixed-length sliding windows and expanding windows for the next
modeling milestone.

Use fixed-length windows as the primary baseline:

- fixed 6-hour windows for the main comparison;
- fixed 3-hour windows as a shorter-context comparison.

Also support expanding windows starting from a configurable minimum length. The
first implemented comparison uses a 3-hour minimum, then grows the window by one
hour at a time.

Window order follows the dataset time direction:

```text
hour_before 30 -> 29 -> ... -> 1
```

Use `stay_has_sepsis` as the v1 window label. Do not fill or overwrite the
original `sepsis` column during windowing.

## Rationale

Fixed windows give every sample the same input shape, which is useful for
Logistic Regression, Random Forest, LSTM, and GRU comparisons. A 6-hour window
captures short-term clinical trends, while a 3-hour window tests whether a more
immediate signal is sufficient.

Expanding windows test the professor's second requested approach: using all
available patient history up to each prediction time. This is clinically
reasonable but produces variable-length sequences, so the implementation pads
expanding windows and includes a mask for sequence models.

## Consequences

- Generated windows are derived from
  `data/processed/mimic_sepsis_preprocessed.csv`.
- Generated `.npz` window files are written under `data/windows/` and remain
  ignored by Git.
- Every window stores metadata: stay ID, window type, window length, start/end
  `hour_before`, prediction `hour_before`, and label.
- Dry-run summaries count windows without allocating the full window arrays.
- Windowing does not perform oversampling, undersampling, model training, or
  horizon-specific relabeling yet.
