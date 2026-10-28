# 0001: Repository Continuity and Hygiene

Date: 2026-04-27

## Decision

Maintain repository-backed project memory and decision records. Future sessions
must read `docs/project_memory.md` and this decision log before making project
changes.

Keep the local dataset, virtual environment, caches, generated models, and
generated outputs out of Git.

## Rationale

The project requires many research and preprocessing choices. Chat history can
be compacted or lost across sessions, so decisions need to live in the
repository.

The clinical dataset is large and should not be uploaded unless the user
explicitly changes that policy.

## Consequences

- Durable choices belong in `docs/project_memory.md`.
- Important choices get their own file in `docs/decisions/`.
- `AGENTS.md` instructs future coding agents to read these files first.
- `.gitignore` remains part of the project hygiene contract.
- Amendment (2026-10-28): GitHub Actions (`.github/workflows/ci.yml`) runs
  `ruff check .` and `python -m pytest -q` on every PR and push to `main`, with
  CPU-only PyTorch. Tests must stay synthetic: CI never has clinical data.
