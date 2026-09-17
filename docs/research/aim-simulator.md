# AiM simulator compatibility

The [AiM simulator](https://github.com/arkhadem/aim_simulator) is the timing
target for CENT instructions executed inside GDDR6. It parses an instruction
trace, expands each instruction into DRAM commands, and schedules those
commands with Ramulator 2.0. It does not store tensor values or calculate the
numerical results of those instructions.

Consequently, a CENT program has two different compatibility requirements:

1. Its DRAM-side instructions must serialize to AiM's trace grammar and obey
   the simulator's fixed organization.
2. Its PNM and CXL instructions need a separate functional executor. AiM does
   not implement instructions such as `EXP`, `RED`, `ACC`, `RISCV`,
   `SEND_CXL`, or `RECV_CXL`.

`vllm_cent.cent.render_aim_trace` enforces the first requirement. It fails
instead of silently discarding an operand that the trace grammar cannot
represent.

## Target contract

The adapter follows the behavior in AiM's trace frontend, request definition,
and DRAM memory system:

- The target has 32 channels, 16 banks per channel, and 16 BF16/FP16 values per
  256-bit burst.
- Channel masks use AiM's physical order. Channel 0 is bit 31, and channel 31
  is bit 0.
- Every DRAM record starts with `AiM`, and a complete trace ends with
  `AiM EOC`.
- `MAC_ABK` reads its second operand from the Global Buffer when CFR0 is 0 and
  from the adjacent bank when CFR0 is 1. The adapter emits the CFR0 write
  immediately before each `MAC_ABK`.
- `AF` selects its activation function through CFR2. The adapter emits the
  configured function ID before each activation.
- `RD_AF` reads the activation output. It is not interchangeable with
  `RD_MAC`, which reads the MAC result.
- `COPY_BKGB` and `COPY_GBBK` include an explicit bank in the trace even though
  the CENT paper's instruction table omits that operand.
- `WR_SBK` and `RD_SBK` have no operation-size field in the trace grammar. A
  multi-burst typed instruction is therefore expanded into repeated one-burst
  records for timing.

The trace format omits the paper-level column and accumulator-register fields
from several instructions. The adapter accepts only column 0 and accumulator
register 0 in those cases. Other values raise
`AimSimulatorCompatibilityError` because serializing them would lose meaning.

## Compiler consequences

The lowering pipeline now records the operand source for every `MAC_ABK`:

- linear layers and attention use the Global Buffer;
- the RMSNorm sum-of-squares reduction uses the adjacent bank.

CENT's functional reference uses bank 0 and bank 1 as elementwise inputs and
bank 2 as the result within each four-bank group. RMSNorm and SiLU therefore
copy bank 2 through the Global Buffer into bank 1 before multiplication. In the
feed-forward SiLU sequence, the W3 operand is staged in bank 0 so it does not
overwrite the copied activation in bank 1.

These rules establish syntactic and DRAM-timing compatibility. They do not by
themselves establish numerical correctness. A complete Llama program still
contains PNM/CXL operations and unresolved data-placement contracts that must
be implemented and validated in a functional runtime before end-to-end model
execution is possible.

## Source provenance

The fixed trace grammar and hidden control-register behavior come from the AiM
simulator rather than the CENT paper. The bank roles used by elementwise
lowering are also consistent with the functional simulator in the original
[CENT repository](https://github.com/Yufeng98/CENT). Where the paper-level ISA
and simulator trace differ, the typed CENT instruction remains paper-oriented
and `render_aim_trace` performs the target-specific conversion.
