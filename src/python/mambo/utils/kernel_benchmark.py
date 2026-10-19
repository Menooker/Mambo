"""Reusable cuTile autotuning and GGUF benchmark command.

Run ``python -m mambo.utils.kernel_benchmark --help`` for the registered kernels.
Each kernel supplies benchmark cases; this module handles the GGUF input,
tuning, remeasurement of the selected config, and reporting.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any, Callable, Sequence

import cupy as cp
import cuda.tile as ct
import numpy as np

from mambo.utils.gguf_manifest import inspect_gguf


Config = dict[str, Any]


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    stream: Any
    kernel: Any
    search_space: Sequence[Config]
    grid_fn: Callable[[Config], tuple[int, ...]]
    args_fn: Callable[[Config], tuple[Any, ...]]
    hints_fn: Callable[[Config], dict[str, Any]] | None = None
    summary_keys: tuple[str, ...] = ()


def mini_benchmark(cases: Sequence[BenchmarkCase]) -> None:
    """Tune each case, then time its selected config again and print the result."""
    for case in cases:
        tuning = ct.tune.exhaustive_search(
            case.search_space, case.stream, case.grid_fn, case.kernel,
            case.args_fn, case.hints_fn, quiet=True,
        )
        best = tuning.best.config
        measured = ct.tune.exhaustive_search(
            [best], case.stream, case.grid_fn, case.kernel,
            case.args_fn, case.hints_fn, quiet=True,
        ).best
        print(f"{case.name}: best_config={best}, "
              f"latency_us={measured.mean_us:.3f} ± {measured.error_margin_us:.3f}, "
              f"candidates={len(tuning.successes)} passed/{len(tuning.failures)} failed")
        if case.summary_keys:
            by_group: dict[tuple[Any, ...], float] = {}
            for result in tuning.successes:
                group = tuple(result.config[key] for key in case.summary_keys)
                by_group[group] = min(by_group.get(group, float("inf")), result.mean_us)
            print(f"  best latency by {case.summary_keys} (us): {by_group}")


def load_gguf_weight(model: Path, weight_name: str, expected_type: str) -> cp.ndarray:
    """Load a named 2D packed GGUF tensor as [N, K/block_elements, block_bytes]."""
    manifest = inspect_gguf(model)
    tensor = next((item for item in manifest["tensors"] if item["name"] == weight_name), None)
    if tensor is None:
        raise ValueError(f"GGUF has no tensor named {weight_name!r}")
    quant_type = tensor["type"]
    if quant_type["name"] != expected_type or len(tensor["shape"]) != 2:
        raise ValueError(f"{weight_name} must be a 2D {expected_type} tensor")
    k, n_rows = tensor["shape"]
    block_values = quant_type["block_elements"]
    block_bytes = quant_type["block_bytes"]
    if (not block_values or not block_bytes or k % block_values
            or tensor["row_bytes"] != k // block_values * block_bytes
            or tensor["n_bytes"] != n_rows * tensor["row_bytes"]):
        raise ValueError(f"{weight_name} has an inconsistent GGUF block layout")
    raw = np.memmap(model, mode="r", dtype=np.uint8,
                    offset=tensor["offset"]["absolute"],
                    shape=(n_rows, k // block_values, block_bytes))
    return cp.asarray(raw)


# Add a kernel module and an entry here for each new benchmark.
_BENCHMARKS = {
    "iq3_s_fp8": ("iq3_s", "mambo.kernels.iq3_s_fp8"),
}


def benchmark_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Autotune a cuTile kernel on a GGUF weight")
    parser.add_argument("kernel", choices=sorted(_BENCHMARKS))
    parser.add_argument("--model", type=Path, required=True, help="GGUF model path")
    parser.add_argument("--weight-name", required=True, help="tensor name in the GGUF model")
    args = parser.parse_args(argv)

    expected_type, kernel_module = _BENCHMARKS[args.kernel]
    weight = load_gguf_weight(args.model, args.weight_name, expected_type)
    make_cases = importlib.import_module(kernel_module).make_benchmark_cases
    mini_benchmark(make_cases(weight))


if __name__ == "__main__":
    benchmark_main()
