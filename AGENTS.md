# Repository Development Standards

These instructions apply to every file in this repository.

## Collaboration

- The primary agent owns the user's complete objective, maintains the coherent
  project understanding, and handles all user interaction. It may work directly;
  delegation is a tool for preserving context and improving execution, not a
  requirement for every task.
- Delegate concrete, bounded subtasks when they are independent and one of the
  following is true: parallel execution will save meaningful time, exploration
  would produce large tool outputs that do not belong in the primary context, a
  specialist can inspect a separate source or subsystem, or an independent
  review would materially improve correctness.
- Good delegation candidates include inspecting separate reference repositories,
  tracing disjoint subsystems, researching independent source questions, running
  focused test investigations, reviewing a completed change, and implementing
  changes in non-overlapping files.
- Work directly when the task is small, tightly coupled to the current reasoning,
  depends heavily on the user's conversational context, modifies the same files
  as active work, or consists of final integration and verification. Never spawn
  a subagent merely to satisfy a delegation policy.
- Give every subagent a concrete scope, the relevant context and source
  boundaries, an explicit deliverable, and clear permission about whether it may
  edit files. Subagents must distinguish established evidence, inference, local
  design proposals, and unresolved questions.
- Require concise, evidence-rich handoffs. A useful handoff states the conclusion,
  supporting file paths and line numbers or external sources, files changed,
  checks run, failures, assumptions, and remaining questions. Do not copy large
  command outputs into the primary context when a short summary and precise
  pointers are sufficient.
- Avoid overlapping writes. Subagents share the workspace, so assign read-only
  investigations or disjoint files whenever work runs concurrently. The primary
  agent must inspect the working tree before integration and preserve unrelated
  user or agent changes.
- The primary agent must evaluate rather than merely relay subagent conclusions.
  It remains responsible for resolving conflicts, integrating changes, checking
  claims against the cited evidence, running proportionate final verification,
  and accurately reporting what was completed.
- Keep durable project knowledge outside conversation history. Record accepted
  architectural decisions, source-backed findings, unresolved contracts, and
  important invariants in the appropriate code, tests, TODOs, or project
  documentation. Use subagents to reduce disposable exploration in the primary
  context, but keep the decisions and reasoning needed for future user dialogue
  in the primary thread.
- Ask the user a question only when an unresolved choice would materially change
  the design or behavior. Do not ask questions merely to confirm routine,
  reversible implementation decisions that can be derived from the repository.

## Documentation and shared context

- Treat `docs/` as curated shared memory for work that must survive across
  tasks, not as a transcript archive. Follow `docs/README.md` for the directory
  structure and keep source code, tests, and Git history as the authoritative
  record of implementation details.
- At the start of substantial work, the primary agent reads `docs/README.md`,
  `docs/PROJECT_STATE.md`, the relevant accepted decisions and research note,
  `git status`, and the source files in scope. Give subagents only the relevant
  excerpts or file pointers rather than making every agent load every document.
- Use the smallest useful team, normally one to three subagents. Give each a
  distinct role such as source research, isolated implementation, or independent
  review. Do not launch duplicate investigations unless independent confirmation
  is the purpose.
- During coordinated work, one primary agent owns edits to shared coordination
  documents, especially `docs/PROJECT_STATE.md` and the decision register.
  Subagents return concise handoffs to the primary unless explicitly assigned a
  non-overlapping research document. This prevents merge conflicts and several
  agents recording incompatible conclusions as accepted fact.
- Keep `docs/PROJECT_STATE.md` short and current. Rewrite or remove stale entries
  when the boundary changes; do not append a chronological work log. It should
  contain only the current verified capabilities, major blockers, immediate
  milestone, and ordered next steps.
- Update an existing accepted-decision topic before creating another document.
  Record only decisions that affect multiple modules or future tasks, including
  the evidence and consequences needed to apply them. Keep unresolved local
  questions beside the affected code as `TODO` comments; put a question in
  `PROJECT_STATE.md` only when it blocks multiple areas.
- Create or extend a research note only when it contains source-backed findings
  another task is likely to reuse. Separate paper claims, reference behavior,
  simulator behavior, local inference, and unanswered questions. Prefer precise
  links and file locations over copied source passages or command output.
- Do not create a permanent handoff file for every task or agent. Use agent
  messages for work that the primary can integrate immediately. Use
  `docs/handoffs/` only when unfinished work must outlive its task, and delete the
  handoff after its durable findings are incorporated into code, tests, project
  state, a decision, or a research note.
- Before declaring coordinated work complete, the primary agent reconciles the
  subagent reports with the current working tree, updates only the shared
  documents whose facts actually changed, removes obsolete temporary handoffs,
  and verifies that documentation describes the current code rather than a
  proposed future state.

## Project goal

The long-term goal is to run any model supported by vLLM on custom CENT
hardware. The compiler should translate model operations into a typed,
hardware-valid CENT program without coupling the generic CENT backend to a
specific model family.

Use the [CENT paper](https://arxiv.org/abs/2502.07578) as the strongest source
for architectural and ISA behavior. The
[original CENT repository](https://github.com/Yufeng98/CENT) is useful reference
code, but it does not override the paper and does not define compatibility
requirements for this repository.

## Python API design

- Follow PEP 8 throughout the repository, including naming, import grouping,
  whitespace, line length, and module organization.
- Every Python module except `__init__.py` must define `__all__` near its
  imports. List only the names intentionally exposed by that module. Use an
  explicitly empty `__all__` for an internal-only module.
- Test modules must not define `__all__`. They are not public APIs.
- Treat a package, rather than every individual file, as a possible public API
  boundary. Re-export stable package APIs from `__init__.py`; do not make an
  implementation helper public merely to give its file a public symbol.
- Use precise type annotations for every function, method, parameter, return
  value, dataclass field, and module-level collection.
- Trust annotations for type correctness. Do not duplicate them with
  `isinstance` checks or `_require_<type>` helpers.
- When branching among instance types, use `isinstance(value, SomeType)`. This
  makes the intended polymorphism clear and lets type checkers narrow the value.
  Do not save `type(value)` and compare it with classes.
- A parameter such as `type[SomeInstruction]` contains a class, not an instance.
  Use `instruction_type is SomeInstruction` for exact-class dispatch, or use
  `issubclass` when subclasses are intentionally supported.
- Keep runtime validation for constraints annotations cannot express, including
  numeric ranges, alignment, capacity, and relationships among fields.
- Prefer immutable, slotted dataclasses for structured values and instructions.
- Do not use a dictionary when the keys form a known schema that can be modeled
  by a dataclass or another custom type.
- When conversion to a dictionary is genuinely required, use dataclass tools
  such as `dataclasses.asdict` or field introspection instead of manually
  rebuilding the same schema.
- Prefer enums and typed constants over free-form strings.
- Avoid metaclasses when an ordinary class, property, or convention expresses
  the same design. Use a metaclass only when its behavior is genuinely needed.
- Do not use nontrivial magic numbers. Give architectural values, IDs, offsets,
  capacities, and repeated constants descriptive module-level names. Trivial
  values such as zero-based initial indices do not require constants.

## Files and component boundaries

- Create a new file or folder when the code represents a cohesive concept that
  can be understood, tested, and maintained independently.
- Do not create a module merely to hold one tiny private helper when that helper
  belongs naturally beside its caller.
- Merge files whose responsibilities are inseparable or whose separation only
  creates import indirection.
- Keep functions and classes focused. A function should perform one task unless
  it is explicitly orchestrating several single-purpose operations.
- Break up large functions or classes when they combine unrelated decisions,
  state, or algorithms. Do not split code solely to reduce line count.
- Keep helpers private unless external callers have a stable reason to use
  them. Public APIs should be small and deliberate.
- Put reusable, domain-independent checks and calculations in an appropriate
  `utils` module. Keep domain-specific validation with its domain types.
- Treat duplicated behavior as a design problem. Move shared calculations,
  validation, and control flow into one well-named helper or utility unless the
  copies have meaningfully different semantics. Document the reason when
  duplication is intentionally retained.

## Documentation

- Every class, function, and method requires a developer-facing docstring,
  including private helpers and test helpers.
- Function and method docstrings must document every argument.
- Document every return value. Omit the `Returns` section when a function
  returns `None`; never write `Returns: None`.
- Document every intentionally raised exception and the condition that causes
  it.
- Dataclass docstrings must document every field. Explain what the field means,
  including units and relationships to other fields when they matter.
- State a valid range only when it teaches the reader something that is not
  obvious. Do not add words such as "positive" or "nonnegative" merely because
  the implementation validates them. Put detailed validation rules in
  ``Raises`` and in the validation code.
- Enum docstrings must document every member.
- Write docstrings as durable API documentation, not as conversational notes,
  implementation history, or messages to the current reviewer.
- Use the same plain language in docstrings as in comments. Define a value by
  what it represents before describing constraints or implementation details.

## Comments and readability

- Code must be understandable to a developer who is unfamiliar with CENT and
  LLM inference. Use explanatory comments throughout nontrivial control flow.
- Explain the sequence of operations, relevant hardware behavior, formulas,
  partitioning, address calculations, and why a particular implementation is
  correct.
- Place comments next to the code they explain.
- Explain unusual language mechanisms even when the resulting code is short.
  State why the ordinary implementation cannot be used and why the chosen
  mechanism is appropriate. For example, explain that ``object.__setattr__``
  is needed to normalize a field during initialization of a frozen dataclass.
- Do not merely translate a line of Python into English. Comments should supply
  the missing reasoning or domain context.
- Explain every nontrivial hard-coded expected value in tests, including how it
  follows from model dimensions, hardware geometry, or instruction semantics.
- Prefer readable intermediate names over compressed expressions. Split a
  calculation into named steps when those names help explain the algorithm.
- Write comments in plain language and keep sentences short. A comment should
  make the code easier to read, not introduce several new terms at once.
- For a complicated TODO, first say what is already known. Then list the open
  questions separately. Concrete examples are better than compressed jargon.
- Leave a blank comment line between a TODO and the ordinary comment that
  follows it. This makes it clear where the TODO ends.

## CENT and model separation

- Generic hardware types, addresses, instructions, mapping, validation,
  program building, and rendering belong under `vllm_cent/cent/`.
- Keep layout selection separate from instruction emission. Model and target
  planners should create immutable operation plans with explicit partitions,
  placements, channels, and memory bindings. Lowerers should validate and emit
  the supplied plan without silently choosing a different layout.
- Reusable operation lowering belongs under `vllm_cent/lowering/`. Keep
  transformer-family operations in a transformer subpackage rather than
  treating them as universal neural-network operations.
- Model specifications, model-specific tensor planning, and operation ordering
  belong under `vllm_cent/models/<family>/`.
- Never place Llama- or transformer-specific assumptions in the generic CENT
  backend.
- Represent a logical Shared Buffer vector with its vector binding, not a raw
  capacity span. A producer must write every lane in every occupied slot. It
  must set lanes outside the logical vector to zero separately in each physical
  partition. Consumers may rely on this invariant and should preserve it.
- Keep results as raw Shared Buffer spans when the ISA or runtime cannot yet
  prove their lane order or zero padding. Do not label unresolved `RD_MAC`,
  activation, reduction, exponential, RISC-V, or CXL output as a logical vector
  merely because enough slots were reserved.
- Use descriptive Python names while documenting the corresponding paper terms,
  such as `row` for `RO`, `column` for `CO`, and `operation_size` for `OPsize`.
- Distinguish paper-defined facts, behavior found only in reference code, local
  design decisions, and unresolved assumptions.
- Record every unresolved design or source question as a nearby `TODO` comment
  as soon as it affects the implementation. Say where the uncertainty comes
  from, such as the paper, target ABI, or unfinished code. State the unanswered
  question in plain language. Do not leave the only record in chat, a review,
  or commit history. Remove or update the TODO after it is resolved and tested.
  Check the paper and reference code first. If either answers the question,
  document the answer instead of adding a TODO.
- Put the TODO at the first function or type whose behavior depends on the
  missing answer. A TODO in a later caller does not adequately document an
  incomplete lowerer.
- Audit every lowerer for numerical completeness, not only for valid instruction
  construction. Trace where every input is initialized, where every result is
  stored, and whether required packing, padding, reductions, synchronization,
  and cross-stage transfers are present.
- A public operation must not silently emit only part of the operation named by
  its API. If implementation must proceed in stages, name and document the
  partial stage explicitly and place a TODO at that stage describing what still
  separates it from the complete mathematical result.
- Do not treat a structurally valid program or a passing instruction-level test
  as evidence of numerical correctness. Tests must cover the dataflow contract
  once an emulator or other numerical oracle is available.
- Do not invent unsupported instructions, operands, or numeric encodings. If a
  required value is absent from the paper, make it explicit configuration or
  report the unsupported operation clearly.

## Testing standards

- Write tests before implementation when designing a new interface or behavior.
- Directly unit-test every nontrivial public and private function, method, and
  class. Private compiler helpers are not exempt.
- Trivial constants, passive marker types, and import-only modules do not need
  dedicated tests.
- Use small fake model specifications and hardware geometries so expected
  instructions and addresses can be calculated by hand.
- Assert concrete values: instruction types, operands, addresses, ordering,
  counts, layouts, and rendered output. Avoid tests that only assert that code
  ran without raising.
- When the full typed result is known, compare it with a complete expected
  dataclass or list of dataclasses. This is clearer and stronger than extracting
  selected fields into tuples. Compare one field only when that field is the
  specific behavior under test or the complete result would hide the intent.
- Cover representative valid inputs, boundary values, invalid ranges,
  alignment failures, capacity failures, and important supported model shapes.
- Keep tests readable and minimal, but never weaken assertions merely to make a
  changed implementation pass.
- Do not add placeholder, tautological, empty, or permanently passing tests.

## Scope and completion

- Implement only what the current requirement needs. Do not add speculative
  abstractions, values, compatibility layers, or future features.
- Prefer the simplest design that remains correct, readable, and extensible.
- Preserve unrelated user changes in the working tree.
- A change is complete only when its public boundary is deliberate, its
  assumptions are explicit, its documentation is complete, and all relevant
  tests pass.
