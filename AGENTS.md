# Agent Instructions

Before making project decisions or code changes, read these files first:

1. `docs/project_memory.md`
2. `docs/decisions/README.md`
3. Relevant decision records in `docs/decisions/`

Do not rely on chat history as the only source of project intent. Chat can be
compacted, old sessions can be unavailable, and future agents need repository
documents to recover the current project direction.

## Memory Maintenance Rule

Keep the project memory documents current as part of the work, not as an
optional cleanup step.

Before every commit, explicitly check whether the change introduced or changed:

- a project assumption;
- a data or label policy;
- a preprocessing, modeling, or evaluation decision;
- a milestone, next step, or open question;
- a dependency, environment, or repository workflow;
- a result from data inspection or research that future sessions need.

If yes, update `docs/project_memory.md` and add or amend a decision record in
`docs/decisions/` before committing. If no, the commit message or final summary
should still mention that no project-memory update was needed.

After completing each project milestone, create or update a separate Markdown
summary under `docs/milestones/`. The milestone summary should explain what was
done, why it was done, how to run or verify it, and what remains out of scope.

## Superpowers Workflow Requirement

When using the Superpowers plugin, invoke the relevant Superpowers skill for the
task. Before any commit, push, or completion claim, use
`superpowers:verification-before-completion` and include the memory-maintenance
check in that verification.

The local dataset under `data/` is intentionally ignored by Git. Do not commit
clinical data, virtual environments, caches, generated models, or generated
outputs unless the user explicitly changes that policy.
