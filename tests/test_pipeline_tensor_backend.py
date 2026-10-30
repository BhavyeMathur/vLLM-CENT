"""Check concrete bank staging, storage reuse, and tensor results."""

import unittest
from dataclasses import replace

from vllm_cent.cent import (
    CentHardwareSpec,
    CentMemoryAddress,
    CentSharedBufferAddress,
    CentChannelSet,
    ElementwiseMultiply,
    ReadSingleBank,
    WriteSingleBank,
)
from vllm_cent.ir import (
    AddOp,
    CopyOp,
    DType,
    Function,
    MulOp,
    ReshapeOp,
    TensorType,
    Value,
)
from vllm_cent.pipeline import CentTargetProfile, ValuePlacement, compile_graph
from vllm_cent.pipeline.target import TargetCommandKind, _emit_multiply_chunk
from vllm_cent.runtime import CentNamedScalars
from vllm_cent.simulator import execute_functionally


def target(*, banks: int = 8) -> CentTargetProfile:
    """Build two-value slots and a four-value scratch row.

    Args:
        banks: Banks per channel; eight exercises both four-bank groups.

    Returns:
        Reference target with room for the test tensors and one scratch row.
    """
    return CentTargetProfile(
        name="tensor-reference",
        hardware=CentHardwareSpec(
            num_channels=2,
            num_banks=banks,
            dram_rows=1,
            dram_columns=4,
            burst_length=2,
            accumulator_slots_per_bank=2,
            sigmoid_activation_function_id=0,
            shared_buffer_slots=32,
        ),
        max_accumulate_slots=1,
    )


def _fp32(name: str, *shape: int) -> Value:
    """Create a named fp32 function argument.

    Args:
        name: Argument name, which also names the compiled program input.
        *shape: Tensor dimensions, outermost first; none gives a scalar.

    Returns:
        New argument value with the requested shape and fp32 elements.
    """
    return Value(type=TensorType(shape=shape, dtype=DType.FP32), name=name)


def multiply_graph(*, square: bool = False, preserve: bool = False) -> Function:
    """Create a five-value multiply crossing the scratch-row boundary.

    Args:
        square: Read the same input twice instead of two distinct inputs.
        preserve: Return the left input as well, requiring a preservation copy.

    Returns:
        Function whose padded six lanes occupy three slots and two scratch
        chunks. The product is named ``z``.
    """
    x = _fp32("x", 5)
    inputs = (x,) if square else (x, _fp32("y", 5))
    multiply = MulOp(operands=(x, inputs[-1]), name_hint="z")
    return Function(
        name="main",
        operands=inputs,
        operations=(multiply,),
        results=(multiply.results[0], x) if preserve else multiply.results,
    )


class TensorBackendTests(unittest.TestCase):
    """Exercise supported tensor operations through the public execution path."""

    def test_multiply_initializes_all_groups_chunks_rows_and_zeros_padding(
        self,
    ) -> None:
        """Initialize all EW_MUL readers, then gather only the first bank group."""
        artifact = compile_graph(multiply_graph(), target())
        values = (
            CentNamedScalars(name="x", values=(1.0, -2.0, 3.0, 4.0, 5.0)),
            CentNamedScalars(name="y", values=(2.0, 3.0, -4.0, 5.0, 6.0)),
        )
        self.assertEqual(
            artifact.execute(values),
            (CentNamedScalars(name="z", values=(2.0, -6.0, -12.0, 20.0, 30.0)),),
        )
        self.assertEqual(
            execute_functionally(artifact.bind(values)).outputs,
            (CentNamedScalars(name="z", values=(2.0, -6.0, -12.0, 20.0, 30.0, 0.0)),),
        )
        instructions = artifact.executable.program.instructions
        # Three slots split as two plus one. Every chunk writes both operands
        # to banks 0/1 and 4/5; EW_MUL reads both groups even though only bank 2
        # is gathered. The other channel is never selected or initialized.
        self.assertEqual(len(instructions), 12)
        for start, count in ((0, 2), (6, 1)):
            chunk = instructions[start : start + 6]
            self.assertEqual(
                [i.address.bank for i in chunk if isinstance(i, WriteSingleBank)],
                [0, 1, 4, 5],
            )
            self.assertEqual(
                [i.operation_size for i in chunk],
                [count] * 6,
            )
            self.assertIsInstance(chunk[4], ElementwiseMultiply)
            self.assertEqual(chunk[4].channels.channels, (0,))
            self.assertIsInstance(chunk[5], ReadSingleBank)
            self.assertEqual(chunk[5].address.bank, 2)
        self.assertEqual(
            artifact.target_ir.commands[0].kind, TargetCommandKind.MULTIPLY
        )

    def test_square_preserves_input_and_reuses_scratch_between_runs(self) -> None:
        """Copy a still-live left operand before squaring both signs."""
        artifact = compile_graph(multiply_graph(square=True, preserve=True), target())
        self.assertEqual(
            [c.kind for c in artifact.target_ir.commands],
            [
                TargetCommandKind.COPY,
                TargetCommandKind.MULTIPLY,
            ],
        )
        for scalars in ((1.0, -2.0, 3.0, -4.0, 5.0), (-5.0, 4.0, -3.0, 2.0, -1.0)):
            with self.subTest(scalars=scalars):
                self.assertEqual(
                    artifact.execute((CentNamedScalars(name="x", values=scalars),)),
                    (
                        CentNamedScalars(
                            name="z", values=tuple(x * x for x in scalars)
                        ),
                        CentNamedScalars(name="x", values=scalars),
                    ),
                )

    def test_right_alias_and_live_left_alias_are_rejected(self) -> None:
        """A preservation copy cannot erase the right operand or a requested output."""
        for preserve, storage in ((False, "y"), (True, "x")):
            with (
                self.subTest(preserve=preserve),
                self.assertRaisesRegex(ValueError, "live"),
            ):
                compile_graph(
                    multiply_graph(preserve=preserve),
                    target(),
                    constraints=(
                        ValuePlacement(value="x", storage="x"),
                        ValuePlacement(value="y", storage="y"),
                        ValuePlacement(value="z", storage=storage),
                    ),
                )

    def test_unary_chain_can_reuse_storage_without_instructions(self) -> None:
        """Copy and reshape record new tensors without changing flat values."""
        x = _fp32("x", 6)
        copy = CopyOp(operands=(x,), name_hint="c")
        reshape = ReshapeOp(operands=copy.results, shape=(2, 3), name_hint="r")
        graph = Function(
            name="main",
            operands=(x,),
            operations=(copy, reshape),
            results=reshape.results,
        )
        artifact = compile_graph(graph, target())
        self.assertEqual(artifact.target_ir.commands, ())
        self.assertEqual(artifact.executable.program.instructions, ())
        self.assertEqual(
            artifact.outputs[0].type, TensorType(shape=(2, 3), dtype=DType.FP32)
        )
        self.assertEqual(
            artifact.execute(
                (CentNamedScalars(name="x", values=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)),)
            ),
            (CentNamedScalars(name="r", values=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)),),
        )

    def test_reshape_preserves_a_requested_input_with_a_real_copy(self) -> None:
        """Changing shape cannot permit overwriting a still-requested input."""
        x = _fp32("x", 6)
        reshape = ReshapeOp(operands=(x,), shape=(3, 2), name_hint="r")
        graph = Function(
            name="main",
            operands=(x,),
            operations=(reshape,),
            results=(reshape.results[0], x),
        )
        artifact = compile_graph(graph, target())
        self.assertEqual(
            [c.kind for c in artifact.target_ir.commands], [TargetCommandKind.COPY]
        )
        # Six values occupy three slots, so a two-slot scratch row requires
        # two write/read pairs. Both logical outputs retain the original order.
        self.assertEqual(len(artifact.executable.program.instructions), 4)
        scalars = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        self.assertEqual(
            artifact.execute((CentNamedScalars(name="x", values=scalars),)),
            (
                CentNamedScalars(name="r", values=scalars),
                CentNamedScalars(name="x", values=scalars),
            ),
        )

    def test_copy_reshape_multiply_add_chain_preserves_original_output(self) -> None:
        """Preserve an input, reshape its copy, square it, and add a second matrix."""
        x = _fp32("x", 6)
        y = _fp32("y", 2, 3)
        copy = CopyOp(operands=(x,), name_hint="c")
        reshape = ReshapeOp(operands=copy.results, shape=(2, 3), name_hint="r")
        square = MulOp(operands=reshape.results * 2, name_hint="s")
        add = AddOp(operands=(square.results[0], y), name_hint="z")
        graph = Function(
            name="main",
            operands=(x, y),
            operations=(copy, reshape, square, add),
            results=(add.results[0], x),
        )
        artifact = compile_graph(graph, target(banks=4))
        self.assertEqual(
            artifact.execute(
                (
                    CentNamedScalars(
                        name="x", values=(1.0, -2.0, 3.0, -4.0, 5.0, -6.0)
                    ),
                    CentNamedScalars(
                        name="y", values=(10.0, 20.0, 30.0, 40.0, 50.0, 60.0)
                    ),
                )
            ),
            (
                CentNamedScalars(name="z", values=(11.0, 24.0, 39.0, 56.0, 75.0, 96.0)),
                CentNamedScalars(name="x", values=(1.0, -2.0, 3.0, -4.0, 5.0, -6.0)),
            ),
        )

    def test_multiply_chunk_emits_exact_bank_operands_and_checks_row_bounds(
        self,
    ) -> None:
        """Exercise the helper's source ordering and full-row capacity directly."""
        hardware = target(banks=4).hardware
        source = CentSharedBufferAddress(slot=7)
        destination = CentSharedBufferAddress(slot=3)
        # Two two-value slots exactly fill the scratch row. Both bank writes
        # precede the multiply and result read, including for an in-place square.
        self.assertEqual(
            _emit_multiply_chunk(hardware, source, destination, 2),
            (
                WriteSingleBank(
                    address=CentMemoryAddress(channel=0, bank=0, row=0, column=0),
                    operation_size=2,
                    source=destination,
                ),
                WriteSingleBank(
                    address=CentMemoryAddress(channel=0, bank=1, row=0, column=0),
                    operation_size=2,
                    source=source,
                ),
                ElementwiseMultiply(
                    channels=CentChannelSet(channels=(0,)),
                    operation_size=2,
                    row=0,
                    column=0,
                ),
                ReadSingleBank(
                    address=CentMemoryAddress(channel=0, bank=2, row=0, column=0),
                    operation_size=2,
                    destination=destination,
                ),
            ),
        )
        for count in (0, 3):
            with (
                self.subTest(count=count),
                self.assertRaisesRegex(ValueError, "scratch row"),
            ):
                _emit_multiply_chunk(hardware, source, destination, count)

    def test_capacity_still_accounts_for_preserved_operands(self) -> None:
        """A preserved five-value input and product each require three slots."""
        small = target()
        small = replace(small, hardware=replace(small.hardware, shared_buffer_slots=5))
        with self.assertRaisesRegex(ValueError, "capacity"):
            compile_graph(multiply_graph(square=True, preserve=True), small)
