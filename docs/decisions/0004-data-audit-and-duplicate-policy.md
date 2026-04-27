# 0004: Data Audit and Duplicate Policy

Date: 2026-04-27

## Decision

Derive stay-level outcomes from observed `sepsis` labels only, and treat rows
with `sepsis == NaN` plus no clinical feature values as structural empty rows.

Drop duplicate `stay_id + hour_before` rows only when the full duplicate rows
are exact copies. Raise an error when duplicate stay-hour rows contain
conflicting values.

## Evidence

Plan A inspection of the local CSV showed:

- `908,798` structural empty rows;
- `56` rows across `28` duplicate stay-hour groups;
- all duplicate groups belong to stay `34934165`;
- the duplicate stay-hour rows are exact copies;
- stay-level outcomes derived from observed labels are `9,657` positive stays
  and `47,383` negative stays.

## Consequences

- `sepsis` NaNs remain unfilled.
- Exact duplicate stay-hour rows can be safely deduplicated.
- Conflicting duplicate stay-hour rows stop preprocessing instead of being
  silently resolved.
- Plan B preprocessing should run after this audit step.
