"""Command line: compile a model as a run spec describes.

    aloi-compile --model examples/llama_70b/llama3_1_70b.json \\
                 --run examples/llama_70b/runs/llama3_1_70b_tp8_decode.yaml
    python -m vllm_cent.pipeline --model ... --run ...   # same, no script

``--model`` is an HF config.json; the registry.py next to it provides the
code. ``--run`` is the YAML run spec (parallelism, workload, scope).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .loader import load_model
from .run_spec import load_run_spec
from .stages import compile_stages

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line.

    Args:
        argv: Arguments without the program name; ``sys.argv[1:]`` if None.

    Returns:
        The process exit status.
    """

    parser = argparse.ArgumentParser(
        prog="aloi-compile",
        description="Compile a model to ALOI IR, as a run spec describes.",
    )
    parser.add_argument(
        "--model", required=True, type=Path, help="HF config.json of the model"
    )
    parser.add_argument(
        "--run", required=True, type=Path, help="run spec YAML (parallel, workload)"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="write <stage>.aloi files here instead of printing them",
    )
    args = parser.parse_args(argv)

    # Bad input files are usage errors: one line, exit status 2.
    try:
        model = load_model(args.model)
        run = load_run_spec(args.run)
    except (FileNotFoundError, KeyError, ValueError, ImportError) as error:
        parser.error(str(error))

    texts = compile_stages(model, run)
    if args.out_dir is None:
        for stage, text in texts.items():
            print(f"// ===== {stage} =====\n{text}")
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for stage, text in texts.items():
        path = args.out_dir / f"{stage}.aloi"
        path.write_text(text)
        print(path)
    return 0
