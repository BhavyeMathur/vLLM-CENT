"""Optionally compare compiled attention with PyTorch in a separate Python environment.

Set ``VLLM_CENT_TORCH_PYTHON`` to an interpreter with PyTorch installed. Keeping
it separate avoids making the compiler's Python version depend on torch wheels.
"""

import json
import math
import os
import subprocess
import unittest
from dataclasses import asdict, dataclass
from random import Random

from tests.test_attention_layer import _target
from vllm_cent.graphs import build_attention_graph
from vllm_cent.pipeline import compile_graph
from vllm_cent.runtime import CentNamedScalars


@dataclass(frozen=True)
class _TorchCase:
    """Carry the same shapes, scalars, and mask rules to the external oracle.

    Attributes:
        query_shape: Batch/head/query/feature dimensions.
        key_shape: Batch/KV-head/key/feature dimensions.
        value_shape: Batch/KV-head/key/value-feature dimensions.
        query: Flattened query values.
        key: Flattened key values.
        value: Flattened value values.
        mask_shape: Shared or full additive mask shape, or absent.
        mask: Flattened additive mask values, or absent.
        causal: Whether to hide keys beyond each query's supplied position.
        offset: Position of the first query relative to the first key.
        scale: Explicit dot-product multiplier, or the default reciprocal root.
    """

    query_shape: tuple[int, int, int, int]
    key_shape: tuple[int, int, int, int]
    value_shape: tuple[int, int, int, int]
    query: tuple[float, ...]
    key: tuple[float, ...]
    value: tuple[float, ...]
    mask_shape: tuple[int, ...] | None
    mask: tuple[float, ...] | None
    causal: bool
    offset: int
    scale: float | None


# This oracle imports no project modules. It translates our explicit causal
# offset into an additive mask, then invokes the real PyTorch attention API.
_TORCH_ORACLE = """
import json
import sys
import torch
from torch.nn.functional import scaled_dot_product_attention

results = []
for case in json.load(sys.stdin):
    q = torch.tensor(case['query'], dtype=torch.float64).reshape(case['query_shape'])
    k = torch.tensor(case['key'], dtype=torch.float64).reshape(case['key_shape'])
    v = torch.tensor(case['value'], dtype=torch.float64).reshape(case['value_shape'])
    mask = None
    if case['mask'] is not None:
        mask = torch.tensor(case['mask'], dtype=torch.float64).reshape(case['mask_shape'])
    if case['causal']:
        allowed = torch.arange(k.shape[-2])[None, :] <= (
            torch.arange(q.shape[-2])[:, None] + case['offset']
        )
        causal_bias = torch.zeros(allowed.shape, dtype=torch.float64)
        causal_bias.masked_fill_(~allowed, -torch.inf)
        mask = causal_bias if mask is None else mask + causal_bias
    result = scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False,
        scale=case['scale'], enable_gqa=q.shape[1] != k.shape[1],
    )
    results.append(result.flatten().tolist())
print(json.dumps(results))
"""


@unittest.skipUnless(
    os.environ.get("VLLM_CENT_TORCH_PYTHON"),
    "set VLLM_CENT_TORCH_PYTHON for the optional PyTorch oracle",
)
class TorchAttentionTests(unittest.TestCase):
    """Compare real compiled dataflow with an external framework's math implementation."""

    def test_matches_pytorch_scaled_dot_product_attention(self) -> None:
        """Compare GQA, independent value widths, masks, offsets, and signed scaling."""
        random = Random(709)
        cases: list[_TorchCase] = []
        actual: list[tuple[float, ...]] = []
        for mask_mode, causal, offset, scale in (
            ("none", False, 0, None),
            ("shared", True, 0, 0.5),
            ("full", True, 2, None),
            ("full", False, 0, -0.5),
            ("shared", False, 0, 0.0),
        ):
            q_shape, k_shape, v_shape = (2, 4, 3, 3), (2, 2, 5, 3), (2, 2, 5, 2)
            mask_shape = (
                None
                if mask_mode == "none"
                else ((3, 5) if mask_mode == "shared" else (2, 4, 3, 5))
            )
            query, key, value = (
                tuple(random.randrange(-9, 10) / 4 for _ in range(math.prod(shape)))
                for shape in (q_shape, k_shape, v_shape)
            )
            mask = (
                None
                if mask_shape is None
                else tuple(
                    -math.inf if i % 5 == 4 else (i % 3) / 4
                    for i in range(math.prod(mask_shape))
                )
            )
            case = _TorchCase(
                q_shape,
                k_shape,
                v_shape,
                query,
                key,
                value,
                mask_shape,
                mask,
                causal,
                offset,
                scale,
            )
            cases.append(case)
            graph = build_attention_graph(
                q_shape,
                k_shape,
                v_shape,
                mask_shape=mask_shape,
                causal=causal,
                query_position_offset=offset,
                scale=scale,
            )
            inputs = (
                CentNamedScalars(name="query", values=query),
                CentNamedScalars(name="key", values=key),
                CentNamedScalars(name="value", values=value),
            )
            if mask is not None:
                inputs += (CentNamedScalars(name="mask", values=mask),)
            actual.append(compile_graph(graph, _target(2)).execute(inputs)[0].values)
        process = subprocess.run(
            [os.environ["VLLM_CENT_TORCH_PYTHON"], "-c", _TORCH_ORACLE],
            input=json.dumps([asdict(case) for case in cases]),
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
            # The parent may expose a Python-3.14 MLIR environment. Its native
            # modules must not leak into a different interpreter's torch import.
            env={
                key: value for key, value in os.environ.items() if key != "PYTHONPATH"
            },
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        expected = json.loads(process.stdout)
        self.assertEqual(len(actual), len(expected))
        for index, (output, oracle) in enumerate(zip(actual, expected, strict=True)):
            with self.subTest(case=index):
                self.assertEqual(len(output), len(oracle))
                for observed, wanted in zip(output, oracle, strict=True):
                    self.assertTrue(
                        math.isclose(observed, wanted, rel_tol=1e-12, abs_tol=1e-12),
                        (observed, wanted),
                    )
