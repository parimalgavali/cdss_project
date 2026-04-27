# 0002: Sepsis Label NaN Policy

Date: 2026-04-27

## Decision

Treat `sepsis == NaN` rows as structural padding/no-observation rows, not as
unknown outcomes and not as negative labels.

Do not statistically impute the `sepsis` label. Derive stay-level outcome from
observed labels only:

```text
stay_has_sepsis = max(observed sepsis values for the stay)
```

## Evidence

Dataset inspection showed:

- rows with `sepsis == NaN` have zero non-missing clinical feature values;
- no stay has both observed `0` and observed `1` labels;
- positive stays contain observed `1` labels plus padded `NaN` rows;
- negative stays contain observed `0` labels plus padded `NaN` rows.

This pattern is more consistent with structural empty hourly rows than with
unknown clinical outcome labels.

## Consequences

- Do not fill `sepsis` NaNs with `0`.
- Do not use `sepsis` NaNs directly as training labels.
- Keep padded rows when needed for hourly sequence continuity.
- Derive window and stay targets from observed labels, then document that policy
  in preprocessing outputs.
- If the professor later provides a different label definition, revise this
  decision before changing code.
