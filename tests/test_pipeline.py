"""Tests for the compiler driver: registry, run specs, loader, aloi-compile."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from vllm_cent.pipeline import (
    STAGES,
    Phase,
    RunSpec,
    compile_stages,
    get_registration,
    load_model,
    load_run_spec,
    parse_run_spec,
    register_model,
)
from vllm_cent.pipeline import registry as registry_module
from vllm_cent.pipeline.cli import main

# registry.py of a tiny model family: one projection whose output dimension
# ("out", width from the config) is split by tp. MODEL_TYPE is replaced per
# test so registrations do not collide.
_TINY_REGISTRY = textwrap.dedent(
    """
    import torch
    from torch import nn

    from vllm_cent.frontend import aloi
    from vllm_cent.ir import TensorAnnotation, TensorRole
    from vllm_cent.pipeline import register_model


    class Projection(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.proj = aloi.Linear(16, width)

        def forward(self, x):
            return self.proj(x)


    def build(width):
        with torch.device("meta"):
            return Projection(width)


    register_model(
        "MODEL_TYPE",
        parse_config=lambda hf: hf["width"],
        build_model=build,
        example_inputs=lambda width, batch, context_len: (
            torch.empty(batch, 16, device="meta"),
        ),
        annotations=lambda width: {
            "x": TensorAnnotation(TensorRole.ACTIVATION, ("batch", "hidden")),
            "proj.weight": TensorAnnotation(TensorRole.WEIGHT, ("out", "hidden")),
        },
        axis_sizes=lambda width: {},
    )
    """
)

_TP2_RUN = "parallel:\n  tp: 2\n  split: [out]\nworkload:\n  batch: 3\n"


class _ModelDirectoryTestCase(unittest.TestCase):
    """A fresh model directory: tiny.json (width 8), registry.py, run.yaml."""

    def setUp(self) -> None:
        """Write the files; undo the import and the registration afterwards."""

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        # A unique directory name makes registry.py a fresh top-level module
        # name; it is not a package, so it imports as "registry".
        self.directory = Path(directory.name)
        self.model_type = f"tiny_{id(self)}"
        (self.directory / "registry.py").write_text(
            _TINY_REGISTRY.replace("MODEL_TYPE", self.model_type)
        )
        self.config = self.directory / "tiny.json"
        self.config.write_text(json.dumps({"model_type": self.model_type, "width": 8}))
        self.run_path = self.directory / "run.yaml"
        self.run_path.write_text(_TP2_RUN)
        self.addCleanup(sys.modules.pop, "registry", None)
        self.addCleanup(registry_module._REGISTRY.pop, self.model_type, None)
        self.addCleanup(self._remove_from_path)

    def _remove_from_path(self) -> None:
        """Undo the loader's sys.path entry for the temporary directory."""

        if str(self.directory) in sys.path:
            sys.path.remove(str(self.directory))


class RegistryTest(unittest.TestCase):
    """register_model and get_registration."""

    def test_register_and_look_up(self) -> None:
        """A registration is found by its model_type."""

        registration = register_model(
            "registry_test",
            parse_config=dict,
            build_model=lambda config: None,  # type: ignore[arg-type, return-value]
            example_inputs=lambda config, batch, context_len: (),
            annotations=lambda config: {},
            axis_sizes=lambda config: {},
        )
        self.addCleanup(registry_module._REGISTRY.pop, "registry_test", None)
        self.assertIs(get_registration("registry_test"), registration)
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_model(
                "registry_test",
                **{
                    name: getattr(registration, name)
                    for name in (
                        "parse_config",
                        "build_model",
                        "example_inputs",
                        "annotations",
                        "axis_sizes",
                    )
                },
            )

    def test_unknown_model_type(self) -> None:
        """The error lists what is registered."""

        with self.assertRaisesRegex(KeyError, "no model registered"):
            get_registration("no_such_model")


class RunSpecTest(unittest.TestCase):
    """YAML run specs."""

    def test_defaults(self) -> None:
        """An empty document is tp=1, decode, batch 1, context 2048, 1 layer."""

        self.assertEqual(parse_run_spec({}), RunSpec())
        self.assertIs(RunSpec().workload.phase, Phase.DECODE)

    def test_full_document_builds_the_plan(self) -> None:
        """Every setting reaches the ParallelPlan."""

        spec = parse_run_spec(
            {
                "parallel": {
                    "tp": 8,
                    "split": ["q_head", "kv_head"],
                    "replicate": ["k_proj.weight"],
                    "split_only": {"hidden": ["lm_head.weight"]},
                },
                "workload": {"phase": "decode", "batch": 4, "context_len": 4096},
                "scope": {"layers": 1},
            }
        )
        plan = spec.parallel.build_plan()
        self.assertEqual((plan.tp, plan.split_axes), (8, ("q_head", "kv_head")))
        self.assertEqual(set(plan.exceptions), {"k_proj.weight", "lm_head.weight"})
        self.assertEqual((spec.workload.batch, spec.workload.context_len), (4, 4096))

    def test_rejections(self) -> None:
        """Typos and invalid values are reported, not ignored."""

        cases: dict[str, dict[str, object]] = {
            "unknown section": {"paralel": {}},
            "unknown key": {"parallel": {"tpp": 8}},
            "zero ranks": {"parallel": {"tp": 0}},
            "split not a list": {"parallel": {"split": "q_head"}},
            "unknown phase": {"workload": {"phase": "train"}},
            "negative batch": {"workload": {"batch": -1}},
        }
        for name, data in cases.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                parse_run_spec(data)


class LoaderTest(_ModelDirectoryTestCase):
    """load_model imports the registry.py next to the config."""

    def test_loads_family_and_config(self) -> None:
        """The module name is the config's stem; parse_config ran."""

        model = load_model(self.config)
        self.assertEqual(model.name, "tiny")
        self.assertEqual(model.config, 8)
        self.assertEqual(model.registration.model_type, self.model_type)

    def test_example_package_keeps_relative_imports(self) -> None:
        """The Llama registry.py uses relative imports inside its package."""

        root = Path(__file__).resolve().parents[1]
        model = load_model(root / "examples" / "llama_70b" / "llama3_1_70b.json")
        self.assertEqual(
            (model.name, model.registration.model_type), ("llama3_1_70b", "llama")
        )

    def test_rejections(self) -> None:
        """Missing files, no model_type, or an unregistered model_type."""

        no_type = self.directory / "no_type.json"
        no_type.write_text("{}")
        unknown = self.directory / "unknown.json"
        unknown.write_text(json.dumps({"model_type": "not_registered"}))
        other = tempfile.TemporaryDirectory()
        self.addCleanup(other.cleanup)
        other_dir = Path(other.name)
        lonely = other_dir / "lonely.json"
        lonely.write_text(json.dumps({"model_type": self.model_type}))
        cases = {
            "missing config": (self.directory / "missing.json", FileNotFoundError),
            "no registry.py": (lonely, FileNotFoundError),
            "no model_type": (no_type, ValueError),
            "unregistered": (unknown, KeyError),
        }
        for name, (path, error) in cases.items():
            with self.subTest(case=name), self.assertRaises(error):
                load_model(path)


class CompileStagesTest(_ModelDirectoryTestCase):
    """compile_stages runs every stage as the run spec says."""

    def test_stages_follow_the_run_spec(self) -> None:
        """8 output rows over tp=2 leave 4 per rank; batch 3 is kept."""

        texts = compile_stages(load_model(self.config), load_run_spec(self.run_path))
        self.assertEqual(tuple(texts), STAGES)
        self.assertIn(
            "%proj = aloi.linear(%x, %proj.weight) : tensor<3x8xfp32>",
            texts["00_imported"],
        )
        self.assertIn(
            "%proj = aloi.linear(%x, %proj.weight) : tensor<3x4xfp32, "
            "axes=[batch, out], global_shape=3x8, placement=shard(out)>",
            texts["02_sharded"],
        )

    def test_unsupported_runs(self) -> None:
        """Prefill and more than one layer are not compiled yet."""

        model = load_model(self.config)
        for data in ({"workload": {"phase": "prefill"}}, {"scope": {"layers": 2}}):
            with self.subTest(run=data), self.assertRaises(NotImplementedError):
                compile_stages(model, parse_run_spec(data))


class CommandLineTest(_ModelDirectoryTestCase):
    """aloi-compile --model <config.json> --run <run.yaml>."""

    def _run(self, *args: str) -> tuple[int, str, str]:
        """Run main() and capture its output.

        Args:
            *args: Command-line arguments.

        Returns:
            ``(exit status, stdout, stderr)``.
        """

        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                status = main(list(args))
            except SystemExit as exit_:
                status = int(exit_.code or 0)
        return status, stdout.getvalue(), stderr.getvalue()

    def test_prints_every_stage(self) -> None:
        """Without --out-dir, each stage follows a header line."""

        status, out, _ = self._run(
            "--model", str(self.config), "--run", str(self.run_path)
        )
        self.assertEqual(status, 0)
        headers = [line for line in out.splitlines() if line.startswith("// =====")]
        self.assertEqual(headers, [f"// ===== {stage} =====" for stage in STAGES])

    def test_writes_stage_files(self) -> None:
        """--out-dir gets one .aloi file per stage, equal to compile_stages."""

        out_dir = self.directory / "ir"
        status, _, _ = self._run(
            "--model",
            str(self.config),
            "--run",
            str(self.run_path),
            "--out-dir",
            str(out_dir),
        )
        expected = compile_stages(load_model(self.config), load_run_spec(self.run_path))
        self.assertEqual(status, 0)
        for stage in STAGES:
            self.assertEqual((out_dir / f"{stage}.aloi").read_text(), expected[stage])

    def test_bad_inputs_are_usage_errors(self) -> None:
        """One-line message and exit status 2 instead of a traceback."""

        bad_run = self.directory / "bad.yaml"
        bad_run.write_text("parallel:\n  tpp: 8\n")
        cases = {
            "missing model": (
                self.directory / "missing.json",
                self.run_path,
                "not found",
            ),
            "bad run spec": (self.config, bad_run, "unknown keys"),
        }
        for name, (model, run, message) in cases.items():
            with self.subTest(case=name):
                status, _, err = self._run("--model", str(model), "--run", str(run))
                self.assertEqual(status, 2)
                self.assertIn(message, err)


if __name__ == "__main__":
    unittest.main()
