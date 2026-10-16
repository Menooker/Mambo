"""Compute gguf-py and ggml CPU references for an operation on GGUF weights.

This is a library module, not a pytest test. Currently ``matmul`` with an
IQ3_S GGUF weight is supported. The input activation may be a NumPy or CuPy
array; the weight is named by its GGUF tensor name.

Example::

    result = compute_reference_gguf(
        "matmul", {"x": x, "weight": "blk.1.attn_qkv.weight"},
        model_path=model_path, reference_executable=ggml_reference,
    )
    expected = result.gguf_py
    backend = result.ggml_cpu
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from io import BytesIO
import json
import math
from pathlib import Path
import subprocess
import sys
import types
from typing import Any, Literal, Mapping

import numpy as np


__test__ = False

_HEADER_LIMIT = 1 << 20
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_ROOT = _REPO_ROOT / "src" / "python"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from mambo.utils.gguf_manifest import inspect_gguf  # noqa: E402


ArrayBackend = Literal["numpy", "cupy", "same"]


@dataclass(frozen=True)
class ReferenceResult:
    """Reference arrays; the last three come from the pinned C++ ggml tool."""

    gguf_py: Any
    ggml_cpu: Any
    ggml_dequant_f64: Any
    dequant_rows: Any
    metadata: dict[str, Any]


def _as_numpy_f32(value: Any) -> tuple[np.ndarray, bool, bool]:
    is_cupy = hasattr(value, "__cuda_array_interface__")
    if is_cupy:
        try:
            import cupy as cp
        except ImportError as exc:
            raise ImportError("CuPy is required for CUDA array inputs") from exc
        array = cp.asnumpy(value)
    else:
        array = np.asarray(value)
    if array.ndim not in (1, 2):
        raise ValueError("matmul input must have shape [K] or [M,K]")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("matmul input must contain numeric values")
    was_vector = array.ndim == 1
    array = np.asarray(array, dtype=np.dtype("<f4"), order="C")
    if not np.isfinite(array).all():
        raise ValueError("matmul input contains NaN or infinity")
    if array.ndim == 1:
        array = array[None, :]
    return np.ascontiguousarray(array), is_cupy, was_vector


def _load_gguf_decoder(llama_source_dir: Path):
    package_root = (llama_source_dir / "gguf-py").resolve()
    if not (package_root / "gguf" / "quants.py").is_file():
        raise FileNotFoundError(f"gguf-py is missing below {package_root}")
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    # The pinned package __init__ imports optional PyYAML. Its constants and
    # quantization modules only need NumPy, so load them without that __init__.
    package = sys.modules.get("gguf")
    if package is None or Path(getattr(package, "__file__", "")).resolve() != package_root / "gguf" / "__init__.py":
        for name in tuple(sys.modules):
            if name == "gguf" or name.startswith("gguf."):
                del sys.modules[name]
        package = types.ModuleType("gguf")
        package.__path__ = [str(package_root / "gguf")]
        package.__package__ = "gguf"
        package.__file__ = str(package_root / "gguf" / "__init__.py")
        sys.modules["gguf"] = package
    constants = import_module("gguf.constants")
    quants = import_module("gguf.quants")
    return constants.GGMLQuantizationType, quants.dequantize


def _find_weight(model_path: Path, name: str) -> dict[str, Any]:
    manifest = inspect_gguf(model_path)
    for tensor in manifest["tensors"]:
        if tensor["name"] == name:
            return tensor
    raise KeyError(f"GGUF has no tensor named {name!r}")


def _float64_matmul(x: np.ndarray, weight: np.ndarray, chunk_rows: int = 256) -> np.ndarray:
    m, k = x.shape
    n, weight_k = weight.shape
    if k != weight_k:
        raise ValueError("input width does not match GGUF weight width")
    result = np.empty((m, n), dtype=np.float32)
    x64 = x.astype(np.float64)
    for start in range(0, n, chunk_rows):
        stop = min(start + chunk_rows, n)
        result[:, start:stop] = (x64 @ weight[start:stop].astype(np.float64).T).astype(np.float32)
    return result


def _read_cpp_packet(packet: bytes, *, m: int, k: int, n: int, rows: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    stream = BytesIO(packet)
    metadata_array = np.load(stream, allow_pickle=False)
    if metadata_array.dtype != np.uint8 or metadata_array.ndim != 1 or not 0 < metadata_array.size <= _HEADER_LIMIT:
        raise ValueError("ggml reference stdout has invalid NumPy metadata")
    header = json.loads(metadata_array.tobytes().decode("utf-8"))
    if header.get("schema_version") != 1 or header.get("op") != "matmul":
        raise ValueError("ggml reference stdout uses an unsupported schema or op")
    expected = (
        ("dequant_rows", [rows, k]),
        ("ggml_dequant_f64", [m, n]),
        ("ggml_cpu", [m, n]),
    )
    descriptors = header.get("arrays")
    if not isinstance(descriptors, list) or len(descriptors) != len(expected):
        raise ValueError("ggml reference stdout has an unexpected array table")
    arrays: dict[str, np.ndarray] = {}
    for descriptor, (name, shape) in zip(descriptors, expected):
        array = np.load(stream, allow_pickle=False)
        nbytes = math.prod(shape) * 4
        if (descriptor.get("name") != name or descriptor.get("dtype") != "float32"
                or descriptor.get("shape") != shape or descriptor.get("nbytes") != nbytes):
            raise ValueError(f"ggml reference stdout has invalid metadata for {name}")
        if array.dtype != np.dtype("<f4") or list(array.shape) != shape or not array.flags.c_contiguous:
            raise ValueError(f"ggml reference stdout has invalid NumPy array for {name}")
        arrays[name] = array
    if stream.read(1):
        raise ValueError("ggml reference stdout has trailing bytes")
    return arrays, header


def _convert_result_backend(array: np.ndarray, backend: ArrayBackend, input_was_cupy: bool):
    if backend == "numpy" or (backend == "same" and not input_was_cupy):
        return array
    if backend not in ("cupy", "same"):
        raise ValueError("output_backend must be 'numpy', 'cupy', or 'same'")
    try:
        import cupy as cp
    except ImportError as exc:
        raise ImportError("CuPy is required for CuPy reference outputs") from exc
    return cp.asarray(array)


def compute_reference_gguf(
    op: str,
    inputs: Mapping[str, Any],
    *,
    model_path: str | Path,
    reference_executable: str | Path,
    llama_source_dir: str | Path = _REPO_ROOT / "thirdparty" / "llama.cpp",
    output_backend: ArrayBackend = "same",
    dequant_rows: int = 4,
    threads: int = 1,
) -> ReferenceResult:
    """Return independent gguf-py and ggml CPU results for ``op``.

    For ``matmul``, ``inputs`` contains ``x`` (an array) and ``weight`` (a GGUF
    tensor name). For now only ``op='matmul'`` with an
    IQ3_S 2D weight is implemented. A 1D input returns 1D outputs; a 2D input
    returns [M,N]. Arrays are NumPy by default, or CuPy when the input is CuPy.
    No intermediate files are created.
    """
    if op != "matmul":
        raise NotImplementedError(f"GGUF reference op {op!r} is not implemented")
    if set(inputs) != {"x", "weight"}:
        raise ValueError("matmul inputs must contain exactly 'x' and 'weight'")
    x = inputs["x"]
    weight = inputs["weight"]
    if not isinstance(weight, str) or not weight:
        raise TypeError("weight must be a nonempty GGUF tensor name")
    if output_backend not in ("numpy", "cupy", "same"):
        raise ValueError("output_backend must be 'numpy', 'cupy', or 'same'")
    if dequant_rows < 1 or threads < 1:
        raise ValueError("dequant_rows and threads must be positive")
    model_path = Path(model_path).expanduser().resolve()
    executable = Path(reference_executable).expanduser().resolve()
    llama_source_dir = Path(llama_source_dir).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not executable.is_file():
        raise FileNotFoundError(executable)
    x_np, input_was_cupy, input_was_vector = _as_numpy_f32(x)

    tensor = _find_weight(model_path, weight)
    qtype = tensor["type"]
    shape = tensor["shape"]
    if len(shape) != 2:
        raise ValueError(f"matmul weight {weight!r} must have two dimensions")
    k, n = shape
    if x_np.shape[1] != k:
        raise ValueError(f"input width {x_np.shape[1]} does not match weight width {k}")
    if qtype["name"] != "iq3_s":
        raise NotImplementedError(f"matmul reference for {qtype['name']} is not implemented")
    rows = min(dequant_rows, n)
    quant_type, dequantize = _load_gguf_decoder(llama_source_dir)
    if quant_type.IQ3_S.value != qtype["id"]:
        raise ValueError("GGUF type ID disagrees with pinned gguf-py")
    raw = np.memmap(model_path, mode="r", dtype=np.uint8,
                    offset=tensor["offset"]["absolute"], shape=(tensor["n_bytes"],))
    weight_rows = raw.reshape(n, tensor["row_bytes"])
    decoded = dequantize(weight_rows, quant_type.IQ3_S)
    if decoded.shape != (n, k) or decoded.dtype != np.float32 or not np.isfinite(decoded).all():
        raise ValueError("gguf-py produced an invalid decoded weight matrix")
    gguf_py = _float64_matmul(x_np, decoded)

    command = [str(executable), "--op", op, "--model", str(model_path),
               "--weight", weight, "--m", str(x_np.shape[0]),
               "--dequant-rows", str(rows), "--threads", str(threads)]
    run = subprocess.run(command, input=x_np.tobytes(order="C"), capture_output=True, check=False)
    if run.returncode != 0:
        raise RuntimeError(f"ggml reference failed: {run.stderr.decode('utf-8', errors='replace').strip()}")
    cpp_arrays, header = _read_cpp_packet(run.stdout, m=x_np.shape[0], k=k, n=n, rows=rows)
    for key, expected_value in (
        ("tensor_name", weight), ("ggml_type", "iq3_s"),
        ("shape_ggml_ne_order", [k, n]),
        ("row_bytes", tensor["row_bytes"]),
        ("tensor_bytes", tensor["n_bytes"]),
        ("data_offset_absolute", tensor["offset"]["absolute"]),
    ):
        if header.get(key) != expected_value:
            raise ValueError(f"ggml reference header disagrees with GGUF metadata: {key}")
    np.testing.assert_array_equal(cpp_arrays["dequant_rows"], decoded[:rows])
    np.testing.assert_allclose(cpp_arrays["ggml_dequant_f64"], gguf_py, rtol=1e-6, atol=2e-5)
    if not np.isfinite(cpp_arrays["ggml_cpu"]).all():
        raise ValueError("ggml CPU output contains NaN or infinity")

    def returned(array: np.ndarray, *, is_output: bool = False):
        if is_output and input_was_vector:
            array = array[0]
        return _convert_result_backend(array, output_backend, input_was_cupy)

    return ReferenceResult(
        gguf_py=returned(gguf_py, is_output=True),
        ggml_cpu=returned(cpp_arrays["ggml_cpu"], is_output=True),
        ggml_dequant_f64=returned(cpp_arrays["ggml_dequant_f64"], is_output=True),
        dequant_rows=returned(cpp_arrays["dequant_rows"]),
        metadata=header,
    )
