# Agent Workflow Rules

## Documents under `docs/`
- `docs/design.md`: Master design document for architecture and specification.
- `docs/tasks.md`: Task tracking document.
  - Do not delete completed tasks.
  - Move completed items to the `Done` section.

## Design/Task Documentation Policy
- `docs/design.md` must contain only the latest target specification/state.
  - Do not keep change history, migration notes, or "how it changed" narratives in `design.md`.
- All update context (background, transition plan, implementation sequencing) must be tracked in `docs/tasks.md`.
- `docs/tasks.md` should be organized by implementation unit (Work Unit), not a flat short checklist.
  - Each Work Unit should include: background, scope, subtasks, and completion criteria.

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

## Runtime Verification Rule
- Do not stop verification at compile checks and tests when a task requires runtime behavior validation.
- For each implementation item, design an appropriate verification artifact before running:
  - snapshot (state dump / metrics JSON), and/or
  - screenshot (render output), as needed by the feature.
- When sandbox execution cannot validate the target behavior (especially GPU/rendering), use escalated execution and run the app directly.
- Prefer repeatable automated loops (auto-load scenario, fixed frame count, artifact export, auto-exit) so Codex can verify results without manual user observation each time.
- Report verification outcomes based on the generated artifacts.

## Test Change Approval Rule
- Unit tests:
  - The agent may add/update/remove unit tests without prior approval.
- Physics integration tests (headless scenario tests):
  - Adding or changing test cases, pass/fail thresholds, or golden/snapshot baselines requires explicit user approval before editing.
  - Bug fixes for test harness infrastructure (runner, serializer, CLI wiring) are allowed, but behavior-affecting scenario changes still require approval.

## Commit Command Rule
- When the user says `commit`, run `git commit` with an appropriate commit message.
- When the user says `commit`, include all currently accumulated uncommitted changes, and split them into as many coherent work-unit commits as practical.
- Commits are basically performed in `Impl session`.
- Commit unit must be "one intent" (one design decision or one implementation outcome).
- Do not mix unrelated changes in the same commit.

## Commit Policy (Best Practice)
- `Design session`:
  - Commit only when a design/spec decision is finalized.
  - Commit targets are `docs/design.md` and `docs/tasks.md` only.
- `Impl session`:
  - Commit per completed task or coherent implementation slice.
  - Commit targets are implementation files + related tests + `docs/tasks.md` status update.
  - Do not include `docs/design.md` edits.
- Before implementation commits, compile checks and tests must pass.

## `done` Command Rule
- When the user says `done`, treat it as: "a task unit has reached a completion checkpoint."
- Before finishing:
  - Mark completed subtasks in `docs/tasks.md`.
  - Report work that was newly requested by the user but not previously listed in tasks.
  - If implementation behavior has diverged from `docs/design.md`, add the change as Design Feedback in `docs/tasks.md`.
- After documentation updates, create git commits split by coherent work unit/intention as much as practical.
