# 0005: Security Model

Date: 2026-10-26

## Decision

Keep a threat model of the repository in `SECURITY_MODEL.md` at the repository
root. It describes the system, trust boundaries, assets, attacker profiles,
existing controls, accepted risks, and open questions.

## Rationale

Security scans and future sessions need a durable, repository-backed
description of what matters for this project (chiefly confidentiality of the
MIMIC-derived dataset under the PhysioNet Data Use Agreement and integrity of
research results) instead of re-deriving it from chat history.

## Consequences

- Update `SECURITY_MODEL.md` when adding components that change the attack
  surface (e.g. model checkpoint loading, new dependencies, deployment,
  shared compute).
- Items marked "(assumption)" should be confirmed or corrected by the owner.
