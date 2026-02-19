# Agent Workflow Rules

## Documents under `docs/`
- `docs/design.md`: Master design document for architecture and specification.
- `docs/tasks.md`: Task tracking document.
  - Do not delete completed tasks.
  - Move completed items to the `Done` section.

## Session Types
- `Design session`
  - Purpose: design discussion, specification updates, task creation.
  - Allowed updates:
    - `docs/tasks.md`: editable.
    - `docs/design.md`: editable only with explicit user approval before editing.
- `Impl session`
  - Purpose: implementation, bug fixing, and test management.
  - Work must follow tasks defined in `docs/tasks.md`.
  - Allowed updates:
    - `docs/tasks.md`: progress reports, completion marks, and related updates.
    - `docs/design.md`: read-only.
  - If design changes are needed, add feedback/request items to `docs/tasks.md`.

## Default Session Rule
- If session type is not explicitly specified by the user, treat it as `Impl session`.

## Implementation Quality Rule
- After code changes, run compile checks and tests.
- Continue fixing until errors are resolved as a rule.

## Commit Command Rule
- When the user says `commit`, run `git commit` with an appropriate commit message.
- Commits are basically performed in `Impl session`.
