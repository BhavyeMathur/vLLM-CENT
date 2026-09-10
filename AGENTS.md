# Repository Development Standards

These instructions apply to every file in this repository.

## Collaboration

- Coding agents may spawn multiple subagents whenever independent work can be
  completed in parallel more efficiently or an independent review would improve
  correctness.
- Give each subagent a concrete, bounded responsibility. Avoid delegation when
  coordination would cost more than completing the task directly.
- The primary agent remains responsible for integrating subagent results,
  resolving conflicts, running final verification, and accurately reporting
  what was completed.

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
- Model specifications, tensor planning, and lowering belong under
  `vllm_cent/models/<family>/`.
- Never place Llama- or transformer-specific assumptions in the generic CENT
  backend.
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
