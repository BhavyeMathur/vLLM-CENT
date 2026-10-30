"""Check rotary position embedding from the IR op to the reference service."""

import unittest
from dataclasses import replace

import torch
from torch import Tensor, nn

from vllm_cent.cent import CentHardwareSpec, CentProgram, CentSharedBufferAddress
from vllm_cent.cent.aim import render_aim_instruction
from vllm_cent.cent.reference import ReferenceRotaryEmbedding
from vllm_cent.cent.render import render_instruction
from vllm_cent.frontend import aloi, export_model, import_exported_program
from vllm_cent.ir import Function
from vllm_cent.pipeline import (
    CentTargetProfile,
    CompiledArtifact,
    ExecutionContract,
    compile_graph,
)
from vllm_cent.runtime import CentNamedScalars
from vllm_cent.simulator import CentExecutionFault
from vllm_cent.simulator.execution import execute_program
from vllm_cent.simulator.state import CentDeviceState

# Shared Buffer layout of the hand-checked service tests, in four-lane slots:
# slot 0 holds one head vector, slot 1 one position, slots 2-3 two table rows.
HEAD = CentSharedBufferAddress(slot=0)
POSITION = CentSharedBufferAddress(slot=1)
TABLE = CentSharedBufferAddress(slot=2)

# Two table rows of two cosines then two sines. Row 1 keeps pair 0
# (cos 1, sin 0) and turns pair 1 by ninety degrees (cos 0, sin 1); row 0 is
# never selected and holds values that would show up if it were.
TWO_ROWS = (9.0, 9.0, 9.0, 9.0, 1.0, 0.0, 0.0, 1.0)


def hardware() -> CentHardwareSpec:
    """Use four lanes per slot so one head vector of four fills one slot.

    Returns:
        A small reference geometry with eight Shared Buffer slots.
    """
    return CentHardwareSpec(
        num_channels=1,
        num_banks=4,
        dram_rows=1,
        dram_columns=8,
        global_buffer_columns=8,
        burst_length=4,
        accumulator_slots_per_bank=2,
        sigmoid_activation_function_id=0,
        shared_buffer_slots=8,
    )


def rotary(**changes: object) -> ReferenceRotaryEmbedding:
    """Build the one-token, one-head service of the hand-checked tests.

    Args:
        **changes: Fields to replace, such as ``is_neox_style=False``.

    Returns:
        A rotation of the head at slot 0 in place, by table row ``position``.
    """
    base = ReferenceRotaryEmbedding(
        source=HEAD,
        positions=POSITION,
        table=TABLE,
        destination=HEAD,
        token_count=1,
        head_count=1,
        head_dim=4,
        rotary_dim=4,
        position_count=2,
        is_neox_style=True,
    )
    return replace(base, **changes)  # type: ignore[arg-type]


def run(instruction: ReferenceRotaryEmbedding, position: float) -> tuple[float, ...]:
    """Rotate the head ``(1, 2, 3, 4)`` by ``TWO_ROWS[position]`` and read it back.

    Args:
        instruction: Service reading the layout described at module level.
        position: Stored position value; only an integer selects a row.

    Returns:
        The four values of slot 0 after execution.
    """
    state = CentDeviceState(hardware=hardware())
    state.write_shared_buffer(HEAD, (1.0, 2.0, 3.0, 4.0))
    state.write_shared_buffer(POSITION, (position, 0.0, 0.0, 0.0))
    state.write_shared_buffer(TABLE, TWO_ROWS)
    execute_program(
        CentProgram(hardware=hardware(), instructions=(instruction,)), state
    )
    return state.read_shared_buffer(HEAD, slot_count=1)


class ReferenceRotaryEmbeddingTests(unittest.TestCase):
    """The service rotates pairs by table angles and rejects invalid inputs."""

    def test_neox_pairs_element_i_with_i_plus_half(self) -> None:
        """Head (1, 2, 3, 4) pairs (1, 3) and (2, 4); row 1 turns only pair 1.

        Pair 1 (a=2, b=4) at ninety degrees becomes (a*0 - b*1, b*0 + a*1)
        = (-4, 2), written back at elements 1 and 3.
        """
        self.assertEqual(run(rotary(), 1.0), (1.0, -4.0, 3.0, 2.0))

    def test_gptj_pairs_adjacent_elements(self) -> None:
        """Head (1, 2, 3, 4) pairs (1, 2) and (3, 4); row 1 turns pair (3, 4).

        Pair 1 (a=3, b=4) becomes (-4, 3); pair 0 is unchanged.
        """
        self.assertEqual(run(rotary(is_neox_style=False), 1.0), (1.0, 2.0, -4.0, 3.0))

    def test_partial_rotary_passes_the_rest_through(self) -> None:
        """With rotary_dim 2 of head_dim 4, only pair (1, 2) turns; (3, 4) is copied.

        Two-value rows (cos, sin) split ``TWO_ROWS`` into four rows; row 3 is
        (0, 1), ninety degrees. The pair (a=1, b=2) becomes
        (a*0 - b*1, b*0 + a*1) = (-2, 1).
        """
        instruction = rotary(rotary_dim=2, position_count=4)
        self.assertEqual(run(instruction, 3.0), (-2.0, 1.0, 3.0, 4.0))

    def test_position_must_be_an_integer_row(self) -> None:
        """1.5 names no row, and row 2 is past the two-row table."""
        for position in (1.5, 2.0, -1.0):
            with self.subTest(position=position), self.assertRaises(CentExecutionFault):
                run(rotary(), position)

    def test_validation_and_physical_export(self) -> None:
        """Counts, pairing, and spans are checked; physical renderers reject it."""
        cases = {
            "odd rotary_dim": rotary(rotary_dim=3),
            "rotary_dim above head_dim": rotary(rotary_dim=6),
            "no tokens": rotary(token_count=0),
            # Eight slots hold 32 values; 9 rows of 4 starting at slot 2 need
            # nine slots of table alone.
            "table past capacity": rotary(position_count=9),
        }
        for label, instruction in cases.items():
            with self.subTest(case=label), self.assertRaises(ValueError):
                CentProgram(hardware=hardware(), instructions=(instruction,))
        for renderer in (render_instruction, render_aim_instruction):
            with self.assertRaisesRegex(ValueError, "no physical"):
                renderer(rotary())


class _Rope(nn.Module):
    """Apply ``aloi.rope`` to a supplied head tensor, positions, and table."""

    def __init__(self, is_neox_style: bool) -> None:
        """Remember the pairing style.

        Args:
            is_neox_style: Pairing passed to ``aloi.rope``.
        """
        super().__init__()
        self.is_neox_style = is_neox_style

    def forward(self, x: Tensor, positions: Tensor, table: Tensor) -> Tensor:
        """Rotate ``x`` by the table rows of ``positions``.

        Args:
            x: ``[batch, tokens, heads, head_dim]``.
            positions: ``[batch, tokens]`` integer positions.
            table: ``[positions, rotary_dim]`` cosines then sines.

        Returns:
            The rotated tensor.
        """
        rotated: Tensor = aloi.rope(x, positions, table, self.is_neox_style)
        return rotated


def _target(
    contract: ExecutionContract = ExecutionContract.REFERENCE,
) -> CentTargetProfile:
    """Return a small two-lane target with room for the test tensors.

    Args:
        contract: Execution contract; RoPE needs the reference one.

    Returns:
        A one-channel target with 512 Shared Buffer slots.
    """
    return CentTargetProfile(
        name="rope-reference",
        execution_contract=contract,
        max_accumulate_slots=2,
        hardware=CentHardwareSpec(
            num_channels=1,
            num_banks=4,
            dram_rows=1,
            dram_columns=4,
            burst_length=2,
            accumulator_slots_per_bank=2,
            sigmoid_activation_function_id=0,
            shared_buffer_slots=512,
        ),
    )


def _import(model: nn.Module, inputs: tuple[Tensor, ...]) -> Function:
    """Export ``model`` on meta tensors shaped like ``inputs`` and import it.

    Args:
        model: Module without parameters.
        inputs: Example inputs; only their shapes and dtypes are used.

    Returns:
        The imported ``forward`` function.
    """
    meta = tuple(torch.empty(t.shape, dtype=t.dtype, device="meta") for t in inputs)
    return import_exported_program(export_model(model, meta), "m").get_function(
        "forward"
    )


def _execute(artifact: CompiledArtifact, tensors: dict[str, Tensor]) -> Tensor:
    """Bind tensors by input name, run the program, and return its one output.

    Args:
        artifact: Compiled program.
        tensors: A tensor for every input name; integers are bound as floats.

    Returns:
        The flattened output values.
    """
    bindings = tuple(
        CentNamedScalars(
            name=value.name,
            values=tuple(float(e) for e in tensors[value.name].flatten().tolist()),
        )
        for value in artifact.inputs
    )
    (output,) = artifact.execute(bindings)
    return torch.tensor(output.values)


class RopePipelineTests(unittest.TestCase):
    """PyTorch ``aloi.rope`` compiles to CENT and matches its reference kernel."""

    def test_matches_torch_for_each_pairing_and_partial_rotary(self) -> None:
        """Three tokens of two heads, positions out of order, over an 8-row table.

        ``head_dim=6`` with a 4-wide table rotates four elements per head and
        passes the last two through.
        """
        torch.manual_seed(0)
        for is_neox_style, head_dim, rotary_dim in (
            (True, 4, 4),
            (False, 4, 4),
            (True, 6, 4),
        ):
            with self.subTest(neox=is_neox_style, head_dim=head_dim, rotary=rotary_dim):
                tensors = {
                    "x": torch.randn(1, 3, 2, head_dim),
                    "positions": torch.tensor([[5, 0, 2]]),
                    "table": torch.randn(8, rotary_dim),
                }
                model = _Rope(is_neox_style)
                artifact = compile_graph(
                    _import(model, tuple(tensors.values())), _target()
                )
                torch.testing.assert_close(
                    _execute(artifact, tensors),
                    model(*tensors.values()).flatten(),
                )

    def test_rotary_embedding_layer_reads_its_table_buffer(self) -> None:
        """The layer's cos_sin_cache buffer becomes a named program input.

        The table is computed from (head_dim, max_position, base) alone, so the
        reference layer below and the one inside the exported module hold the
        same values.
        """
        layer = aloi.RotaryEmbedding(head_dim=4, max_position=8, base=10000.0)
        q = torch.randn(1, 2, 2, 4)
        k = torch.randn(1, 2, 1, 4)
        positions = torch.tensor([[3, 7]])

        class _QueryOnly(nn.Module):
            """Return only the rotated query, so the program has one output."""

            def __init__(self) -> None:
                """Own a copy of the rotary layer."""
                super().__init__()
                self.rotary_emb = aloi.RotaryEmbedding(
                    head_dim=4, max_position=8, base=10000.0
                )

            def forward(self, positions: Tensor, q: Tensor, k: Tensor) -> Tensor:
                """Rotate q and k; return q.

                Args:
                    positions: ``[1, 2]`` positions.
                    q: ``[1, 2, 2, 4]`` queries.
                    k: ``[1, 2, 1, 4]`` keys.

                Returns:
                    The rotated query.
                """
                rotated_q, _ = self.rotary_emb(positions, q, k)
                return rotated_q

        artifact = compile_graph(_import(_QueryOnly(), (positions, q, k)), _target())
        self.assertIn("rotary_emb.cos_sin_cache", [v.name for v in artifact.inputs])
        tensors = {
            "rotary_emb.cos_sin_cache": layer.cos_sin_cache,
            "positions": positions,
            "q": q,
            "k": k,
        }
        expected, _ = layer(positions, q, k)
        torch.testing.assert_close(_execute(artifact, tensors), expected.flatten())

    def test_requires_the_reference_contract(self) -> None:
        """Without opting in to reference services, the program is rejected."""
        tensors = (torch.randn(1, 1, 1, 2), torch.tensor([[0]]), torch.randn(2, 2))
        with self.assertRaisesRegex(ValueError, "REFERENCE"):
            compile_graph(
                _import(_Rope(True), tensors), _target(ExecutionContract.STANDARD)
            )


if __name__ == "__main__":
    unittest.main()
