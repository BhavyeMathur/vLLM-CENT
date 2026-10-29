"""Tests for lowering Semantic IR to Device IR."""

import unittest

from vllm_cent.activations import ActivationKind
from vllm_cent.device import (
    DeviceCapability,
    DeviceSpec,
    TargetDescriptor,
    TraceFormat,
)
from vllm_cent.ir import (
    DType,
    ExecutionDomain,
    Function,
    TensorRole,
    TensorType,
    Value,
    print_function,
)
from vllm_cent.ir.base import Operation
from vllm_cent.ir.device import (
    CXLAllReduceOp,
    DeviceBroadcastOp,
    DeviceOp,
    DeviceReshapeOp,
    DeviceTransferOp,
    PIMActivationOp,
    PIMEwmulOp,
    PIMGemvOp,
    PNMActivationOp,
    PNMAddOp,
    PNMDivOp,
    PNMExpOp,
    PNMMaxOp,
    PNMMulOp,
    PNMRMSNormOp,
    PNMSoftmaxOp,
    PNMSubOp,
    PNMSumOp,
)
from vllm_cent.ir.semantic import (
    ActivationOp,
    AddOp,
    AllReduceOp,
    LinearOp,
    MulOp,
    ReshapeOp,
    RMSNormOp,
    SoftmaxOp,
)
from vllm_cent.ir.semantic.transformer import AttentionOp, TopKSigmoidOp
from vllm_cent.legalize import LowerToDevice
from vllm_cent.legalize.lower import _default_domain, _implementations

from tests.test_device_cent import PAPER

BF16 = DType.BF16
PIM = ExecutionDomain.PIM
PNM = ExecutionDomain.PNM

# The device ops every softmax test target supports: the parts of a
# decomposed softmax, plus transfers.
SOFTMAX_PARTS = (
    PNMMaxOp,
    DeviceBroadcastOp,
    PNMSubOp,
    PNMExpOp,
    PNMSumOp,
    PNMDivOp,
    DeviceTransferOp,
)


def _spec(
    *ops: type[DeviceOp],
    pim_activations: dict[ActivationKind, int] | None = None,
    pnm_activations: frozenset[ActivationKind] = frozenset(),
) -> DeviceSpec:
    """Build a paper-scale CENT target that runs exactly ``ops`` in bf16.

    Args:
        ops: Supported device op classes; ``device.transfer`` must be listed
            explicitly when a test needs it.
        pim_activations: Activation functions in the PIM lookup tables.
        pnm_activations: Activation functions PNM computes.

    Returns:
        The target.
    """

    return DeviceSpec(
        name="test",
        target=TargetDescriptor("cent_aim", TraceFormat.AIM_TEXT_TRACE),
        architecture=PAPER,
        capabilities=DeviceCapability(
            ops={op: frozenset({BF16}) for op in ops},
            pim_activations=pim_activations or {},
            pnm_activations=pnm_activations,
            pim_gemv_bias=False,
        ),
    )


def _value(
    name: str,
    shape: tuple[int, ...] = (1, 4),
    role: TensorRole = TensorRole.ACTIVATION,
) -> Value:
    """Create a named bf16 Semantic IR argument (no domain).

    Args:
        name: Argument name, as printed.
        shape: Tensor shape.
        role: Tensor role, which decides the argument's domain.

    Returns:
        A new value.
    """

    return Value(type=TensorType(shape=shape, dtype=BF16, role=role), name=name)


def _function(*ops: Operation, arguments: tuple[Value, ...]) -> Function:
    """Wrap ``ops`` into a function that returns the last op's results.

    Args:
        ops: The function body, in order.
        arguments: The function's arguments.

    Returns:
        A function named ``f``.
    """

    return Function("f", arguments, ops, ops[-1].results)


def _op_classes(function: Function) -> list[type[Operation]]:
    """Return the class of every op in ``function``, in order."""

    return [type(op) for op in function.operations]


class DefaultDomainTest(unittest.TestCase):
    """Arguments and results are placed by role."""

    def test_roles(self) -> None:
        """Weights and the KV cache stay in DRAM; activations live in PNM."""

        self.assertEqual(
            {role: _default_domain(role) for role in TensorRole},
            {
                TensorRole.ACTIVATION: PNM,
                TensorRole.WEIGHT: PIM,
                TensorRole.KV_CACHE: PIM,
            },
        )


class ImplementationsTest(unittest.TestCase):
    """Device ops are found from the semantic op they inherit."""

    def test_implementations(self) -> None:
        """Every candidate, ordered by name (pim.* before pnm.*).

        - Mul and activation run either in the banks or near memory.
        - Add has only a PNM op: the other elementwise ops share AddOp's
          base class, but no device op inherits that base, so they do not
          count.
        - Softmax and RMSNorm are found through their shared type rules.
        - Fused attention and the MoE router have no device op.
        """

        expected: dict[type[Operation], tuple[type[DeviceOp], ...]] = {
            LinearOp: (PIMGemvOp,),
            MulOp: (PIMEwmulOp, PNMMulOp),
            AddOp: (PNMAddOp,),
            ActivationOp: (PIMActivationOp, PNMActivationOp),
            SoftmaxOp: (PNMSoftmaxOp,),
            RMSNormOp: (PNMRMSNormOp,),
            ReshapeOp: (DeviceReshapeOp,),
            AllReduceOp: (CXLAllReduceOp,),
            AttentionOp: (),
            TopKSigmoidOp: (),
        }
        self.assertEqual(
            {semantic: _implementations(semantic) for semantic in expected},
            expected,
        )


class LowerToDeviceTest(unittest.TestCase):
    """Each semantic op becomes transfers and one supported device op."""

    def test_gemv_and_residual_add(self) -> None:
        """The input moves to PIM for the GEMV, and the product back for the add.

        This is CENT's data flow: the vector is loaded next to the banks
        (WR_GB), and the GEMV result is read out near memory (RD_MAC).
        """

        x, w = _value("x"), _value("w", (4, 4), TensorRole.WEIGHT)
        proj = LinearOp(operands=(x, w), name_hint="proj")
        residual = AddOp(operands=(proj.results[0], x), name_hint="residual")
        function = _function(proj, residual, arguments=(x, w))
        spec = _spec(PIMGemvOp, PNMAddOp, DeviceTransferOp)

        self.assertEqual(
            print_function(LowerToDevice(spec).run_on_function(function)),
            "func @f(%x: tensor<1x4xbf16, domain=pnm>, "
            "%w: tensor<4x4xbf16, role=weight, domain=pim>) "
            "-> (tensor<1x4xbf16, domain=pnm>) {\n"
            "  %0 = device.transfer(%x) {to = pim} : tensor<1x4xbf16, domain=pim>\n"
            "  %proj = pim.gemv(%0, %w) : tensor<1x4xbf16, domain=pim>\n"
            "  %1 = device.transfer(%proj) {to = pnm} : tensor<1x4xbf16, domain=pnm>\n"
            "  %residual = pnm.add(%1, %x) : tensor<1x4xbf16, domain=pnm>\n"
            "  return %residual\n"
            "}",
        )
        # The input is not modified: its arguments still have no domain.
        self.assertIsNone(function.operands[0].type.domain)

    def test_fewest_transfers_then_pnm(self) -> None:
        """Which of pim.ewmul and pnm.mul multiplies a GEMV result ``a``.

        - ``a * b``, both GEMV results in PIM: pim.ewmul needs no transfer,
          pnm.mul two.
        - ``a * y``, with y an activation in PNM: one transfer either way,
          so the tie goes to PNM.
        - ``a * x``: x was already moved to PIM for the GEMV, so that copy is
          reused and pim.ewmul needs no new transfer.

        The last op computes the product, except when a PIM product is an
        activation result and is moved back to PNM.
        """

        x, y = _value("x"), _value("y")
        w = _value("w", (4, 4), TensorRole.WEIGHT)
        a = LinearOp(operands=(x, w))
        b = LinearOp(operands=(x, w))
        spec = _spec(PIMGemvOp, PIMEwmulOp, PNMMulOp, DeviceTransferOp)

        cases = {
            "both in PIM": (
                _function(
                    a, b, MulOp(operands=(a.results[0], b.results[0])), arguments=(x, w)
                ),
                PIMEwmulOp,
            ),
            "tie": (
                _function(a, MulOp(operands=(a.results[0], y)), arguments=(x, y, w)),
                PNMMulOp,
            ),
            "x already moved": (
                _function(a, MulOp(operands=(a.results[0], x)), arguments=(x, w)),
                PIMEwmulOp,
            ),
        }
        for name, (function, expected) in cases.items():
            with self.subTest(case=name):
                lowered = LowerToDevice(spec).run_on_function(function)
                products = [
                    op
                    for op in lowered.operations
                    if isinstance(op, (PIMEwmulOp, PNMMulOp))
                ]
                self.assertEqual([type(op) for op in products], [expected])

    def test_a_value_moves_once_per_domain(self) -> None:
        """q, k and v projections of one x share a single transfer to PIM."""

        x, w = _value("x"), _value("w", (4, 4), TensorRole.WEIGHT)
        projections = [LinearOp(operands=(x, w)) for _ in range(3)]
        total = AddOp(operands=(projections[0].results[0], projections[1].results[0]))
        spec = _spec(PIMGemvOp, PNMAddOp, DeviceTransferOp)
        lowered = LowerToDevice(spec).run_on_function(
            _function(*projections, total, arguments=(x, w))
        )

        transfers = [
            op for op in lowered.operations if isinstance(op, DeviceTransferOp)
        ]
        # One x -> PIM, then each of the two added products -> PNM.
        self.assertEqual(
            [(op.x.name, op.to) for op in transfers],
            [("x", PIM), (None, PNM), (None, PNM)],
        )
        gemvs = [op for op in lowered.operations if isinstance(op, PIMGemvOp)]
        self.assertEqual({op.operands[0] for op in gemvs}, {transfers[0].results[0]})

    def test_results_return_to_the_domain_of_their_role(self) -> None:
        """A product of two weights is computed in PIM but returned in PNM.

        It is an activation, so the next layer expects it near memory; the
        appended transfer is named after its producer.
        """

        a = _value("a", role=TensorRole.WEIGHT)
        b = _value("b", role=TensorRole.WEIGHT)
        product = MulOp(operands=(a, b), name_hint="product")
        spec = _spec(PIMEwmulOp, DeviceTransferOp)
        lowered = LowerToDevice(spec).run_on_function(
            _function(product, arguments=(a, b))
        )

        self.assertEqual(_op_classes(lowered), [PIMEwmulOp, DeviceTransferOp])
        transfer = lowered.operations[-1]
        self.assertEqual(
            (transfer.name_hint, lowered.results), ("product", transfer.results)
        )
        self.assertIs(lowered.results[0].type.domain, PNM)


class DecomposeOrLowerTest(unittest.TestCase):
    """A device op of its own wins; otherwise the op is decomposed."""

    def _softmax(self) -> Function:
        """Return a function computing the softmax of an activation x."""

        x = _value("x")
        return _function(SoftmaxOp(operands=(x,), name_hint="sm"), arguments=(x,))

    def test_supported_softmax_is_kept_whole(self) -> None:
        """With pnm.softmax, no decomposition and no transfer (x is in PNM)."""

        spec = _spec(PNMSoftmaxOp, *SOFTMAX_PARTS)
        lowered = LowerToDevice(spec).run_on_function(self._softmax())
        self.assertEqual(_op_classes(lowered), [PNMSoftmaxOp])

    def test_unsupported_softmax_is_decomposed(self) -> None:
        """Without pnm.softmax, each step of exp(x - max) / sum lowers to PNM."""

        lowered = LowerToDevice(_spec(*SOFTMAX_PARTS)).run_on_function(self._softmax())
        self.assertEqual(
            _op_classes(lowered),
            [
                PNMMaxOp,
                DeviceBroadcastOp,
                PNMSubOp,
                PNMExpOp,
                PNMSumOp,
                DeviceBroadcastOp,
                PNMDivOp,
            ],
        )

    def test_unsupported_activation_function(self) -> None:
        """Sigmoid is in the PIM tables but SiLU nowhere, so SiLU fails.

        Both candidates are listed with their own reason.
        """

        x = _value("x")
        silu = ActivationOp(
            operands=(x,), function=ActivationKind.SILU, name_hint="silu"
        )
        spec = _spec(
            PIMActivationOp,
            PNMActivationOp,
            DeviceTransferOp,
            pim_activations={ActivationKind.SIGMOID: 0},
        )
        with self.assertRaisesRegex(
            ValueError,
            r"cannot lower aloi.activation \(silu\): "
            r"pim.activation has no lookup table for silu; "
            r"pnm.activation cannot compute silu",
        ):
            LowerToDevice(spec).run_on_function(_function(silu, arguments=(x,)))


class LowerToDeviceErrorTest(unittest.TestCase):
    """Every failure names the op and why each candidate was rejected."""

    def test_part_of_a_decomposition_is_unsupported(self) -> None:
        """The name_hint shows the failing exp came from softmax ``sm``."""

        parts = tuple(op for op in SOFTMAX_PARTS if op is not PNMExpOp)
        x = _value("x")
        function = _function(SoftmaxOp(operands=(x,), name_hint="sm"), arguments=(x,))
        with self.assertRaisesRegex(
            ValueError, r"cannot lower aloi.exp \(sm.exp\): pnm.exp is not supported"
        ):
            LowerToDevice(_spec(*parts)).run_on_function(function)

    def test_no_device_op(self) -> None:
        """No device op implements the MoE router."""

        logits = _value("logits")
        topk = TopKSigmoidOp(operands=(logits,), top_k=1, name_hint="router")
        with self.assertRaisesRegex(
            ValueError,
            r"cannot lower aloi.topk_sigmoid \(router\): no device op implements it",
        ):
            LowerToDevice(_spec()).run_on_function(_function(topk, arguments=(logits,)))

    def test_operand_transfer_unsupported(self) -> None:
        """The GEMV itself is supported, but x cannot be moved to PIM."""

        x, w = _value("x"), _value("w", (4, 4), TensorRole.WEIGHT)
        proj = LinearOp(operands=(x, w))
        with self.assertRaisesRegex(
            ValueError,
            "pim.gemv needs operand 0 moved to pim, but device.transfer is not supported",
        ):
            LowerToDevice(_spec(PIMGemvOp)).run_on_function(
                _function(proj, arguments=(x, w))
            )

    def test_result_transfer_unsupported(self) -> None:
        """The product of two weights cannot be returned to PNM."""

        a = _value("a", role=TensorRole.WEIGHT)
        b = _value("b", role=TensorRole.WEIGHT)
        product = MulOp(operands=(a, b))
        with self.assertRaisesRegex(
            ValueError, "cannot move a result of f to pnm: device.transfer"
        ):
            LowerToDevice(_spec(PIMEwmulOp)).run_on_function(
                _function(product, arguments=(a, b))
            )

    def test_input_must_be_semantic_ir(self) -> None:
        """A function lowered once cannot be lowered again."""

        x, w = _value("x"), _value("w", (4, 4), TensorRole.WEIGHT)
        spec = _spec(PIMGemvOp, PNMAddOp, DeviceTransferOp)
        proj = LinearOp(operands=(x, w))
        lowered = LowerToDevice(spec).run_on_function(_function(proj, arguments=(x, w)))
        with self.assertRaisesRegex(ValueError, "already have a domain"):
            LowerToDevice(spec).run_on_function(lowered)


if __name__ == "__main__":
    unittest.main()
