"""Tests for Device IR ops: domain rules, transfers, views, and compute ops."""

from __future__ import annotations

import unittest
from dataclasses import replace

from vllm_cent.activations import ActivationKind
from vllm_cent.ir import (
    DType,
    ExecutionDomain,
    Function,
    Operation,
    Partial,
    Replicate,
    Shard,
    TensorRole,
    TensorType,
    Value,
    packed_axis,
    print_function,
)
from vllm_cent.ir.base import Decomposable
from vllm_cent.ir.device import (
    CXLAllReduceOp,
    DeviceOp,
    DeviceBroadcastOp,
    DeviceCopyOp,
    DeviceMergeHeadsOp,
    DeviceReshapeOp,
    DeviceSplitHeadsOp,
    DeviceTransferOp,
    PIMActivationOp,
    PIMCacheWriteOp,
    PIMEwmulOp,
    PIMGemvOp,
    PIMQKOp,
    PIMSVOp,
    PNMActivationOp,
    PNMAddOp,
    PNMDivOp,
    PNMExpOp,
    PNMMaxOp,
    PNMMulOp,
    PNMRMSNormOp,
    PNMRopeOp,
    PNMSoftmaxOp,
    PNMSubOp,
    PNMSumOp,
)
from vllm_cent.ir.device.domain import _PNMOp
from vllm_cent.ir.passes import DecomposeOps
from vllm_cent.ir.semantic import (
    ActivationRules,
    AddOp,
    AllReduceOp,
    BroadcastOp,
    CopyOp,
    DivOp,
    ExpOp,
    LinearOp,
    MaxOp,
    MulOp,
    ReshapeOp,
    RMSNormRules,
    SoftmaxRules,
    SubOp,
    SumOp,
)
from vllm_cent.ir.semantic.transformer import (
    GqaQKOp,
    GqaSVOp,
    KvUpdateOp,
    MergeHeadsOp,
    RopeOp,
    SplitHeadsOp,
)

PIM = ExecutionDomain.PIM
PNM = ExecutionDomain.PNM

# Llama 70B q_proj under tp=8: each rank holds 8 of 64 query heads of 128,
# so its local width is 1024 of 8192.
HIDDEN = 8192
LOCAL_Q = 1024
Q_OUT = packed_axis("q_head", "head_dim")
TOKEN_AXES = ("batch", "seq_len")
REPLICATED = Replicate()


def _value(
    shape: tuple[int, ...],
    domain: ExecutionDomain | None,
    axes: tuple[str, ...] = (),
    placement: Shard | Replicate | Partial = REPLICATED,
    global_shape: tuple[int, ...] | None = None,
    role: TensorRole = TensorRole.ACTIVATION,
    dtype: DType = DType.BF16,
) -> Value:
    """Create a value in ``domain``; ``None`` means a Semantic IR value.

    Args:
        shape: Local shape.
        domain: Where the tensor lives, or None.
        axes: Axis names, or () before annotation.
        placement: Which part each rank holds.
        global_shape: Whole shape, for a shard.
        role: Tensor role.
        dtype: Element type.

    Returns:
        A new value.
    """

    return Value(
        type=TensorType(
            shape=shape,
            dtype=dtype,
            role=role,
            axes=axes,
            global_shape=global_shape,
            placement=placement,
            domain=domain,
        )
    )


def _q_proj_operands() -> tuple[Value, Value]:
    """Return the whole x and this rank's Wq rows, both in PIM."""

    x = _value((1, 1, HIDDEN), PIM, (*TOKEN_AXES, "hidden"))
    w = _value(
        (LOCAL_Q, HIDDEN),
        PIM,
        (Q_OUT, "hidden"),
        Shard("q_head"),
        (HIDDEN, HIDDEN),
        TensorRole.WEIGHT,
    )
    return x, w


def _q_shard(domain: ExecutionDomain) -> Value:
    """Return this rank's q, [1, 1, 1024] of the whole [1, 1, 8192], in ``domain``."""

    return _value(
        (1, 1, LOCAL_Q), domain, (*TOKEN_AXES, Q_OUT), Shard("q_head"), (1, 1, HIDDEN)
    )


class ComputeOpDomainTest(unittest.TestCase):
    """Compute ops reuse semantic type rules and require one domain."""

    def test_gemv_adds_the_domain_to_the_linear_type(self) -> None:
        """Column-parallel q_proj on PIM: the linear's shard, plus domain=pim."""

        op = PIMGemvOp(operands=_q_proj_operands())
        self.assertEqual(op.results[0].type, _q_shard(PIM).type)

    def test_operand_in_another_domain_is_rejected(self) -> None:
        """A PNM operand of a PIM op, or a Semantic IR operand, needs lowering first."""

        x, w = _q_proj_operands()
        for wrong, message in (
            (replace(x.type, domain=PNM), "operand 0 is in pnm, not pim; insert"),
            (replace(x.type, domain=None), "operand 0 has no domain"),
        ):
            with self.subTest(domain=wrong.domain):
                with self.assertRaisesRegex(ValueError, message):
                    PIMGemvOp(operands=(Value(type=wrong), w))

    def test_every_op_runs_in_its_domain(self) -> None:
        """Each compute op accepts its own domain and labels its results with it.

        Operands are [2, 3] fp32 rows; the reductions return [2].
        """

        def row(domain: ExecutionDomain) -> Value:
            """Return a [2, 3] fp32 value in ``domain``."""
            return _value((2, 3), domain, dtype=DType.FP32)

        cases: dict[
            str,
            tuple[
                type[Operation],
                int,
                dict[str, object],
                ExecutionDomain,
                tuple[int, ...],
            ],
        ] = {
            "pim.ewmul": (PIMEwmulOp, 2, {}, PIM, (2, 3)),
            "pim.activation": (
                PIMActivationOp,
                1,
                {"function": ActivationKind.SILU},
                PIM,
                (2, 3),
            ),
            "pnm.add": (PNMAddOp, 2, {}, PNM, (2, 3)),
            "pnm.sub": (PNMSubOp, 2, {}, PNM, (2, 3)),
            "pnm.mul": (PNMMulOp, 2, {}, PNM, (2, 3)),
            "pnm.div": (PNMDivOp, 2, {}, PNM, (2, 3)),
            "pnm.exp": (PNMExpOp, 1, {}, PNM, (2, 3)),
            "pnm.reduce_max": (PNMMaxOp, 1, {}, PNM, (2,)),
            "pnm.reduce_sum": (PNMSumOp, 1, {}, PNM, (2,)),
            "pnm.activation": (
                PNMActivationOp,
                1,
                {"function": ActivationKind.GELU_TANH},
                PNM,
                (2, 3),
            ),
            "pnm.softmax": (PNMSoftmaxOp, 1, {}, PNM, (2, 3)),
        }
        for name, (op_class, count, attributes, domain, shape) in cases.items():
            with self.subTest(op=name):
                self.assertEqual(op_class.NAME, name)
                op = op_class(
                    operands=tuple(row(domain) for _ in range(count)), **attributes
                )
                self.assertEqual(
                    op.results[0].type,
                    TensorType(shape=shape, dtype=DType.FP32, domain=domain),
                )
                other = PNM if domain is PIM else PIM
                with self.assertRaisesRegex(ValueError, "insert device.transfer"):
                    op_class(
                        operands=tuple(row(other) for _ in range(count)), **attributes
                    )

    def test_semantic_rules_still_apply(self) -> None:
        """Domain checks come first, then the semantic rules, e.g. no partial exp."""

        partial = _value((1, 1, HIDDEN), PNM, (*TOKEN_AXES, "hidden"), Partial())
        with self.assertRaisesRegex(ValueError, "partial sum"):
            PNMExpOp(operands=(partial,))

    def test_all_reduce_reads_partial_sums_from_pnm(self) -> None:
        """o_proj's partial sums must be moved to PNM before cxl.all_reduce."""

        partial = _value((1, 1, HIDDEN), PNM, (*TOKEN_AXES, "hidden"), Partial())
        result = CXLAllReduceOp(operands=(partial,)).results[0].type
        self.assertEqual(result, replace(partial.type, placement=Replicate()))
        with self.assertRaisesRegex(ValueError, "insert device.transfer"):
            CXLAllReduceOp(operands=(Value(type=replace(partial.type, domain=PIM)),))


class PNMSoftmaxTest(unittest.TestCase):
    """pnm.softmax shares aloi.softmax's rules but is not decomposed."""

    def test_is_not_decomposable(self) -> None:
        """DecomposeOps leaves pnm.softmax alone: its parts would be Semantic IR."""

        x = _value((2, 4), PNM, dtype=DType.FP32)
        softmax = PNMSoftmaxOp(operands=(x,))
        self.assertNotIsInstance(softmax, Decomposable)
        function = Function(
            name="main", operands=(x,), operations=(softmax,), results=softmax.results
        )
        self.assertIs(DecomposeOps().run_on_function(function), function)

    def test_needs_whole_rows(self) -> None:
        """Like aloi.softmax, a split last dimension is rejected."""

        with self.assertRaisesRegex(ValueError, "whole dimension"):
            PNMSoftmaxOp(operands=(_q_shard(PNM),))


class TransferTest(unittest.TestCase):
    """device.transfer is the only op that changes a tensor's domain."""

    def test_changes_only_the_domain(self) -> None:
        """A PIM partial sum moves to PNM unchanged otherwise, ready for all_reduce."""

        partial = _value((1, 1, HIDDEN), PIM, (*TOKEN_AXES, "hidden"), Partial())
        op = DeviceTransferOp(operands=(partial,), to=PNM)
        self.assertEqual(op.results[0].type, replace(partial.type, domain=PNM))
        weight = _value((4,), PIM, role=TensorRole.WEIGHT)
        moved = DeviceTransferOp(operands=(weight,), to=PNM).results[0].type
        self.assertIs(moved.role, TensorRole.WEIGHT)

    def test_rejections(self) -> None:
        """A transfer needs a domain to leave, and a different one to reach."""

        with self.assertRaisesRegex(ValueError, "already in pnm"):
            DeviceTransferOp(operands=(_value((4,), PNM),), to=PNM)
        with self.assertRaisesRegex(ValueError, "no domain"):
            DeviceTransferOp(operands=(_value((4,), None),), to=PNM)

    def test_prints_its_destination(self) -> None:
        """The destination prints as an attribute; the type shows the new domain."""

        x = Value(type=TensorType(shape=(4,), dtype=DType.FP32, domain=PIM), name="x")
        op = DeviceTransferOp(operands=(x,), to=PNM, name_hint="x_pnm")
        function = Function(
            name="f", operands=(x,), operations=(op,), results=op.results
        )
        self.assertIn(
            "%x_pnm = device.transfer(%x) {to = pnm} : tensor<4xfp32, domain=pnm>",
            print_function(function),
        )


class SameDomainOpTest(unittest.TestCase):
    """Views, copy and broadcast run where their operand is."""

    def test_results_stay_in_the_operand_domain(self) -> None:
        """The semantic shape rules apply, and the domain carries over.

        q [1, 1, 1024] splits into 8 heads of 128 and merges back; the
        reshape and copy use [2, 3] -> [6]; broadcast appends 4 positions.
        """

        for domain in (PIM, PNM):
            with self.subTest(domain=domain):
                q = _q_shard(domain)
                heads = DeviceSplitHeadsOp(operands=(q,), head_dim=128).results[0]
                self.assertEqual(
                    (heads.type.shape, heads.type.domain), ((1, 1, 8, 128), domain)
                )
                merged = DeviceMergeHeadsOp(operands=(heads,)).results[0]
                self.assertEqual(merged.type, q.type)
                flat = DeviceReshapeOp(
                    operands=(_value((2, 3), domain),), shape=(6,)
                ).results[0]
                self.assertEqual(
                    flat.type, TensorType(shape=(6,), dtype=DType.BF16, domain=domain)
                )
                copy = DeviceCopyOp(operands=(q,)).results[0]
                self.assertEqual(copy.type, q.type)
                rows = DeviceBroadcastOp(
                    operands=(_value((2,), domain),), size=4
                ).results[0]
                self.assertEqual(
                    rows.type, TensorType(shape=(2, 4), dtype=DType.BF16, domain=domain)
                )

    def test_operand_without_domain_is_rejected(self) -> None:
        """A Semantic IR value has to be lowered before a device op reads it."""

        with self.assertRaisesRegex(ValueError, "no domain"):
            DeviceCopyOp(operands=(_value((4,), None),))

    def test_names(self) -> None:
        """Domain-independent ops use the device. prefix."""

        self.assertEqual(
            [
                op.NAME
                for op in (
                    DeviceTransferOp,
                    DeviceCopyOp,
                    DeviceReshapeOp,
                    DeviceSplitHeadsOp,
                    DeviceMergeHeadsOp,
                    DeviceBroadcastOp,
                )
            ],
            [
                "device.transfer",
                "device.copy",
                "device.reshape",
                "device.split_heads",
                "device.merge_heads",
                "device.broadcast",
            ],
        )

    def test_instances_have_no_dict(self) -> None:
        """The mixins declare empty slots, so device ops stay slotted."""

        op = DeviceCopyOp(operands=(_value((4,), PIM),))
        self.assertFalse(hasattr(op, "__dict__"))
        self.assertFalse(hasattr(PIMGemvOp(operands=_q_proj_operands()), "__dict__"))


# Llama 70B attention under tp=8, decode: 8 query heads and 1 KV head of 128
# per rank, a 2048-position cache, one new token.
CACHE_AXES = ("batch", "context", "kv_head", "head_dim")


def _heads(domain: ExecutionDomain) -> Value:
    """Return this rank's q heads [1, 1, 8, 128] of the whole [1, 1, 64, 128]."""

    return _value(
        (1, 1, 8, 128),
        domain,
        (*TOKEN_AXES, "q_head", "head_dim"),
        Shard("q_head"),
        (1, 1, 64, 128),
    )


def _cache(domain: ExecutionDomain) -> Value:
    """Return this rank's KV cache head [1, 2048, 1, 128] of [1, 2048, 8, 128]."""

    return _value(
        (1, 2048, 1, 128),
        domain,
        CACHE_AXES,
        Shard("kv_head"),
        (1, 2048, 8, 128),
        TensorRole.KV_CACHE,
    )


def _positions(domain: ExecutionDomain | None) -> Value:
    """Return the new token's position [1, 1], int64, in ``domain``."""

    return _value((1, 1), domain, TOKEN_AXES, dtype=DType.INT64)


class TransformerDeviceOpTest(unittest.TestCase):
    """pnm.rope, pim.cache_write, pim.qk and pim.sv keep the semantic rules."""

    def test_rope_in_pnm(self) -> None:
        """x, positions and the [131072, 128] table must all be in PNM."""

        table = _value(
            (131072, 128), PNM, ("position", "rotary_dim"), role=TensorRole.WEIGHT
        )
        rope = PNMRopeOp(operands=(_heads(PNM), _positions(PNM), table))
        self.assertEqual(rope.results[0].type, _heads(PNM).type)
        self.assertEqual(PNMRopeOp.NAME, "pnm.rope")
        pim_table = Value(type=replace(table.type, domain=PIM))
        with self.assertRaisesRegex(ValueError, "operand 2 is in pim"):
            PNMRopeOp(operands=(_heads(PNM), _positions(PNM), pim_table))

    def test_cache_write_positions_may_be_anywhere(self) -> None:
        """The cache and new row are in PIM; positions only pick the row to write."""

        new = _value(
            (1, 1, 1, 128),
            PIM,
            (*TOKEN_AXES, "kv_head", "head_dim"),
            Shard("kv_head"),
            (1, 1, 8, 128),
        )
        for domain in (PIM, PNM):
            with self.subTest(positions=domain):
                write = PIMCacheWriteOp(operands=(_cache(PIM), new, _positions(domain)))
                self.assertEqual(write.results[0].type, _cache(PIM).type)
        with self.assertRaisesRegex(ValueError, "operand 2 has no domain"):
            PIMCacheWriteOp(operands=(_cache(PIM), new, _positions(None)))
        pnm_new = Value(type=replace(new.type, domain=PNM))
        with self.assertRaisesRegex(ValueError, "operand 1 is in pnm"):
            PIMCacheWriteOp(operands=(_cache(PIM), pnm_new, _positions(PIM)))

    def test_qk_and_sv_in_pim(self) -> None:
        """8 query heads, 1 KV head: scores [1, 1, 8, 2048], output [1, 1, 8, 128]."""

        scores = PIMQKOp(operands=(_heads(PIM), _cache(PIM)), scale=128**-0.5).results[
            0
        ]
        self.assertEqual(
            (scores.type.shape, scores.type.domain, scores.type.placement),
            ((1, 1, 8, 2048), PIM, Shard("q_head")),
        )
        output = PIMSVOp(operands=(scores, _cache(PIM))).results[0]
        self.assertEqual(output.type, _heads(PIM).type)
        self.assertEqual((PIMQKOp.NAME, PIMSVOp.NAME), ("pim.qk", "pim.sv"))
        with self.assertRaisesRegex(ValueError, "operand 1 is in pnm"):
            PIMQKOp(operands=(_heads(PIM), _cache(PNM)), scale=1.0)

    def test_rms_norm_in_pnm(self) -> None:
        """With or without a weight; the eps attribute carries over."""

        x = _value((1, 1, HIDDEN), PNM, (*TOKEN_AXES, "hidden"))
        weight = _value((HIDDEN,), PNM, ("hidden",), role=TensorRole.WEIGHT)
        for operands in ((x, weight), (x,)):
            with self.subTest(weighted=len(operands) == 2):
                norm = PNMRMSNormOp(operands=operands, eps=1e-5)
                self.assertEqual(norm.results[0].type, x.type)
                self.assertEqual(norm.attributes(), {"eps": 1e-5})
        self.assertNotIsInstance(PNMRMSNormOp(operands=(x,)), Decomposable)
        self.assertEqual(PNMRMSNormOp.NAME, "pnm.rms_norm")


class DeviceOpTest(unittest.TestCase):
    """Every device op records the semantic op it implements."""

    def test_semantic_of_every_device_op(self) -> None:
        """Each op's semantic base, as listed after its device mixin.

        The activation, softmax and RMSNorm device ops inherit only the
        shared type rules, so they never inherit a decomposition;
        ``device.transfer`` implements no semantic op.
        """

        expected: dict[type[DeviceOp], type[Operation] | None] = {
            CXLAllReduceOp: AllReduceOp,
            DeviceBroadcastOp: BroadcastOp,
            DeviceCopyOp: CopyOp,
            DeviceMergeHeadsOp: MergeHeadsOp,
            DeviceReshapeOp: ReshapeOp,
            DeviceSplitHeadsOp: SplitHeadsOp,
            DeviceTransferOp: None,
            PIMActivationOp: ActivationRules,
            PIMCacheWriteOp: KvUpdateOp,
            PIMEwmulOp: MulOp,
            PIMGemvOp: LinearOp,
            PIMQKOp: GqaQKOp,
            PIMSVOp: GqaSVOp,
            PNMActivationOp: ActivationRules,
            PNMAddOp: AddOp,
            PNMDivOp: DivOp,
            PNMExpOp: ExpOp,
            PNMMaxOp: MaxOp,
            PNMMulOp: MulOp,
            PNMRMSNormOp: RMSNormRules,
            PNMRopeOp: RopeOp,
            PNMSoftmaxOp: SoftmaxRules,
            PNMSubOp: SubOp,
            PNMSumOp: SumOp,
        }
        self.assertEqual({op: op.SEMANTIC for op in expected}, expected)

    def test_device_subclass_keeps_the_semantic(self) -> None:
        """A subclass that adds only device behavior still implements linear.

        The probe defines no NAME, so it is not registered as an op.
        """

        class _Probe(PIMGemvOp):
            """A device op derived from another device op."""

        self.assertIs(_Probe.SEMANTIC, LinearOp)

    def test_two_semantic_ops_are_rejected(self) -> None:
        """One device op cannot implement both add and mul."""

        with self.assertRaisesRegex(
            TypeError, r"several semantic ops \['AddOp', 'MulOp'\]"
        ):
            # mypy also rejects this class (the two ops' dataclass methods
            # differ); the test defines it on purpose to see it rejected.
            class _Both(_PNMOp, AddOp, MulOp):  # type: ignore[misc]
                """Inherits two semantic ops."""


if __name__ == "__main__":
    unittest.main()
