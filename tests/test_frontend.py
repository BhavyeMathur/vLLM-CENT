"""Tests for the PyTorch frontend: custom ops, export policy, importer.

Run from the repository root (``python -m unittest tests.test_frontend``) so
that the ``examples`` package is importable.
"""

from __future__ import annotations

import unittest
from unittest import mock

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.export import Dim

from examples.llama2_70b import LLAMA2_70B, build_meta_block, example_inputs
from vllm_cent.frontend import custom_ops, export_model, import_exported_program
from vllm_cent.frontend import importer
from vllm_cent.ir import ClonePass, PassManager, print_module


# A test-only op whose fake kernel claims one more output feature than the
# real kernel produces. Routed to the linear converter, it makes PyTorch's and
# ALOI's shape inference disagree. Registered once at import: registering the
# same name twice raises.
@torch.library.custom_op("aloi_test::lying_linear", mutates_args=())
def _lying_linear(x: Tensor, weight: Tensor) -> Tensor:
    """Compute ``x @ weight.T``, like ``aloi::linear``."""

    return F.linear(x, weight)


@_lying_linear.register_fake
def _(x: Tensor, weight: Tensor) -> Tensor:
    """Return a result one column wider than the real kernel's."""

    return x.new_empty((*x.shape[:-1], weight.shape[0] + 1))


def _import(model: nn.Module, *inputs: Tensor) -> str:
    """Export, import and print ``model``.

    Args:
        model: Model to compile.
        inputs: Example inputs.

    Returns:
        The printed IR module.
    """

    program = export_model(model, inputs)
    return print_module(import_exported_program(program, "m"))


def _meta_input(*shape: int) -> Tensor:
    """Return an fp32 meta tensor of the given shape.

    Args:
        shape: Tensor shape.

    Returns:
        The tensor.
    """

    return torch.empty(shape, device="meta")


class _LinearWith(nn.Module):
    """Calls one linear-shaped op directly in the root forward."""

    def __init__(self, op: object, as_buffer: bool = False) -> None:
        """Create a ``[4, 16]`` weight.

        Args:
            op: Callable ``op(x, weight)``.
            as_buffer: Register the weight as a buffer, not a parameter.
        """

        super().__init__()
        self.op = op
        weight = torch.empty(4, 16)
        if as_buffer:
            self.register_buffer("w", weight)
        else:
            self.w = nn.Parameter(weight)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the op.

        Args:
            x: Input of shape ``[*, 16]``.

        Returns:
            Output of shape ``[*, 4]``.
        """

        return self.op(x, self.w)  # type: ignore[operator, no-any-return]


class _RmsNormWithoutEps(nn.Module):
    """Calls ``aloi::rms_norm`` without passing eps."""

    def __init__(self) -> None:
        """Create a ``[16]`` scale."""

        super().__init__()
        self.g = nn.Parameter(torch.empty(16))

    def forward(self, x: Tensor) -> Tensor:
        """Normalize ``x``.

        Args:
            x: Input of shape ``[*, 16]``.

        Returns:
            Output with the shape of ``x``.
        """

        result: Tensor = custom_ops.rms_norm(x, self.g)
        return result


class _Relu(nn.Module):
    """Uses an aten op the importer has no converter for."""

    def forward(self, x: Tensor) -> Tensor:
        """Apply ReLU.

        Args:
            x: Any tensor.

        Returns:
            ``relu(x)``.
        """

        return torch.relu(x)


class CustomOpTest(unittest.TestCase):
    """The custom ops compute what their F.* counterparts compute."""

    def test_real_kernels_match_functional(self) -> None:
        """On CPU, both ops equal F.rms_norm / F.linear bit for bit."""

        torch.manual_seed(0)
        x = torch.randn(2, 3, 16)
        gamma = torch.randn(16)
        weight = torch.randn(4, 16)
        self.assertTrue(
            torch.equal(
                custom_ops.rms_norm(x, gamma, 1e-5),
                F.rms_norm(x, (16,), gamma, 1e-5),
            )
        )
        self.assertTrue(
            torch.equal(custom_ops.linear(x, weight), F.linear(x, weight))
        )

    def test_meta_shapes(self) -> None:
        """Linear maps [*, in] to [*, out]; rms_norm keeps the shape."""

        x = _meta_input(2, 3, 16)
        self.assertEqual(
            custom_ops.linear(x, _meta_input(4, 16)).shape, (2, 3, 4)
        )
        self.assertEqual(
            custom_ops.rms_norm(x, _meta_input(16), 1e-5).shape, (2, 3, 16)
        )


class ExportTest(unittest.TestCase):
    """The example block and the export policy."""

    def test_meta_block_parameters(self) -> None:
        """Parameters have Llama2-70B shapes, fp16, and no storage."""

        block = build_meta_block(LLAMA2_70B)
        # q_proj maps hidden 8192 to 64 heads * 128 head_dim = 8192.
        self.assertEqual(block.q_proj.weight.shape, (8192, 8192))
        self.assertEqual(block.input_layernorm.weight.shape, (8192,))
        for parameter in block.parameters():
            self.assertEqual(parameter.dtype, torch.float16)
            self.assertEqual(parameter.device.type, "meta")

    def test_export_keeps_custom_ops_whole(self) -> None:
        """The graph calls exactly the two aloi ops, in forward order."""

        program = export_model(
            build_meta_block(LLAMA2_70B), example_inputs(LLAMA2_70B)
        )
        targets = [
            node.target
            for node in program.graph.nodes
            if node.op == "call_function"
        ]
        self.assertEqual(
            targets,
            [torch.ops.aloi.rms_norm.default, torch.ops.aloi.linear.default],
        )


class ImporterTest(unittest.TestCase):
    """Exported programs become verified ALOI IR."""

    def test_llama_block_golden(self) -> None:
        """The example block imports to the expected text exactly.

        Arguments are named by FQN, results by module path. The 1x1x8192
        input is one decode token; q_proj keeps the width at 64 * 128 = 8192.
        """

        program = export_model(
            build_meta_block(LLAMA2_70B), example_inputs(LLAMA2_70B)
        )
        module = import_exported_program(program, "toy_llama2_70b")
        expected = (
            "module @toy_llama2_70b\n"
            "\n"
            "func @forward(%input_layernorm.weight: tensor<8192xfp16>, "
            "%q_proj.weight: tensor<8192x8192xfp16>, "
            "%x: tensor<1x1x8192xfp16>) -> (tensor<1x1x8192xfp16>) {\n"
            "  %input_layernorm = aloi.rms_norm(%x, %input_layernorm.weight) "
            "{eps = 1e-05} : tensor<1x1x8192xfp16>\n"
            "  %q_proj = aloi.linear(%input_layernorm, %q_proj.weight) "
            ": tensor<1x1x8192xfp16>\n"
            "  return %q_proj\n"
            "}\n"
        )
        self.assertEqual(print_module(module), expected)

    def test_arguments_are_named_by_fqn(self) -> None:
        """Parameters use state_dict keys; the user input keeps its name."""

        program = export_model(
            build_meta_block(LLAMA2_70B), example_inputs(LLAMA2_70B)
        )
        function = import_exported_program(program, "m").get_function("forward")
        self.assertEqual(
            [argument.name for argument in function.operands],
            ["input_layernorm.weight", "q_proj.weight", "x"],
        )

    def test_clone_pass_round_trip(self) -> None:
        """Imported IR runs through a pass and prints unchanged."""

        program = export_model(
            build_meta_block(LLAMA2_70B), example_inputs(LLAMA2_70B)
        )
        module = import_exported_program(program, "m")
        before = print_module(module)
        cloned = PassManager((ClonePass(),)).run(module)
        self.assertEqual(print_module(cloned), before)
        self.assertEqual(print_module(module), before)

    def test_omitted_eps_imports_as_none(self) -> None:
        """Without eps, the FX call has two args and the IR stores None."""

        text = _import(_RmsNormWithoutEps(), _meta_input(1, 16))
        self.assertEqual(
            text.splitlines()[3],
            "  %0 = aloi.rms_norm(%x, %g) {eps = none} : tensor<1x16xfp32>",
        )

    def test_root_level_op_is_numbered(self) -> None:
        """An op called in the root forward has no module path."""

        with torch.device("meta"):
            model = _LinearWith(custom_ops.linear)
        text = _import(model, _meta_input(1, 16))
        self.assertEqual(
            text.splitlines()[3],
            "  %0 = aloi.linear(%x, %w) : tensor<1x4xfp32>",
        )

    def test_unknown_op_is_rejected(self) -> None:
        """An aten op without a converter names itself in the error."""

        with self.assertRaisesRegex(NotImplementedError, r"aten\.relu\.default"):
            _import(_Relu(), _meta_input(1, 16))

    def test_dynamic_shape_is_rejected(self) -> None:
        """A symbolic dimension is not silently specialized.

        seq_len is 4 because export specializes sizes 0 and 1 to constants.
        """

        program = torch.export.export(
            build_meta_block(LLAMA2_70B),
            example_inputs(LLAMA2_70B, seq_len=4),
            dynamic_shapes={"x": {1: Dim("seq_len")}},
        )
        with self.assertRaisesRegex(NotImplementedError, "dynamic dimension"):
            import_exported_program(program, "m")

    def test_buffer_input_is_rejected(self) -> None:
        """Buffers are not supported yet."""

        with torch.device("meta"):
            model = _LinearWith(custom_ops.linear, as_buffer=True)
        with self.assertRaisesRegex(NotImplementedError, "BUFFER"):
            _import(model, _meta_input(1, 16))

    def test_shape_disagreement_is_rejected(self) -> None:
        """IR and PyTorch shape inference must agree.

        The real result is [1, 4]; the lying fake kernel says [1, 5].
        """

        with torch.device("meta"):
            model = _LinearWith(_lying_linear)
        lying = torch.ops.aloi_test.lying_linear.default
        with mock.patch.dict(
            importer._CONVERTERS, {lying: importer._convert_linear}
        ):
            with self.assertRaisesRegex(
                ValueError, r"infers tensor<1x4xfp32>, but PyTorch says tensor<1x5xfp32>"
            ):
                _import(model, _meta_input(1, 16))


if __name__ == "__main__":
    unittest.main()
