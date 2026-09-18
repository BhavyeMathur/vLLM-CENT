# Project documentation

This directory is the shared source of truth for work that must survive across
Codex tasks. It complements the code; it does not repeat docstrings, nearby
`TODO` comments, tests, or commit history.

## Read this first

1. Read the repository-wide rules in [`AGENTS.md`](../AGENTS.md).
2. Read [`PROJECT_STATE.md`](PROJECT_STATE.md) for the current boundary,
   implemented behavior, blockers, and next milestone.
3. Read [`FUNCTIONAL_SIMULATOR_DESIGN.md`](FUNCTIONAL_SIMULATOR_DESIGN.md)
   before implementing the functional executor or its runtime-data contract.
   Its architecture is accepted; its milestone list distinguishes implemented
   behavior from work that still depends on unresolved ISA contracts.
4. Read [`decisions/README.md`](decisions/README.md) before changing a public
   interface, data contract, module boundary, or target assumption.
5. Read only the research note relevant to the task. For example,
   [`research/aim-simulator.md`](research/aim-simulator.md) explains what the
   AiM simulator can and cannot validate.
6. Inspect `git status`, recent commits, and the source files in scope. These
   documents summarize the repository; they do not replace the current code.

## Keeping tasks synchronized

Before a task finishes, it should update these documents only when it has
changed shared project knowledge:

- Update `PROJECT_STATE.md` when a milestone, blocker, or next step changes.
- Update the decision register only after a decision is accepted. Keep open
  questions in `PROJECT_STATE.md` and beside affected code as `TODO` comments.
- Add a research note only for source-backed findings that another task will
  need. State which facts came from the paper, a simulator, reference code, or
  local design.
- Use `docs/handoffs/` only when work cannot yet be integrated. A handoff must
  state its objective, files inspected or changed, evidence, proposed or
  accepted decisions, unresolved questions, tests, and commit hash. Delete it
  after its durable findings are absorbed here or in the code.

One coordinating task should reconcile conflicting findings, update the
project state, integrate focused commits, and run final verification. A chat
conclusion remains provisional until it is represented by code, a test, or one
of these documents.

## What does not belong here

- Line-by-line explanations already expressed by source comments.
- Complete inventories of local `TODO` comments; search the source for those.
- Speculative designs with no current consumer.
- A permanent diary for each task or agent.
- Generated test output or large copied excerpts from external sources.
