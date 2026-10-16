"""Export a cuTile kernel to cubin and describe its CUDA Driver API ABI.

``launch_config`` is a mapping with ordered ``parameters``: each key is a
Python kernel parameter name and each value is a cuTile ParameterConstraint.
Compile-time arguments may instead be placed in ``constants``. The remaining
keys are ``gpu_code`` (for example ``sm_120``), ``symbol``, ``grid`` and
``block``. The latter two describe the intended Driver API launch.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import metadata as importlib_metadata
import json
import platform
from pathlib import Path

from cuda.tile.compilation import (
    ArrayConstraint,
    CallingConvention,
    ConstantConstraint,
    KernelSignature,
    ScalarConstraint,
    export_kernel,
)


def _dtype_name(dtype):
    return str(dtype)


def _json_value(value):
    """Convert cuTile constraint values to plain JSON values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    raise TypeError(f"ABI metadata cannot serialize {type(value).__name__}")


def _constraint_metadata(name, constraint):
    if isinstance(constraint, ConstantConstraint):
        return None
    if isinstance(constraint, ScalarConstraint):
        return {"name": name, "kind": "scalar", "dtype": _dtype_name(constraint.dtype)}
    if isinstance(constraint, ArrayConstraint):
        fields = (
            "stride_lower_bound_incl",
            "alias_groups",
            "may_alias_internally",
            "stride_constant",
            "shape_constant",
            "stride_divisible_by",
            "shape_divisible_by",
            "base_addr_divisible_by",
        )
        result = {
            "name": name,
            "kind": "array",
            "dtype": _dtype_name(constraint.dtype),
            "ndim": int(constraint.ndim),
            "index_dtype": _dtype_name(constraint.index_dtype),
        }
        for field in fields:
            if not hasattr(constraint, field):
                raise TypeError(f"cuTile ArrayConstraint is missing required field {field}")
            result[field] = _json_value(getattr(constraint, field))
        return result
    raise TypeError(f"ABI metadata does not support {type(constraint).__name__}")


def _toolchain_metadata():
    cutile_version = "unknown"
    for distribution in ("cuda-tile", "cuda_tile", "nvidia-cuda-tile"):
        try:
            cutile_version = importlib_metadata.version(distribution)
            break
        except importlib_metadata.PackageNotFoundError:
            pass
    return {"python": platform.python_version(), "cuda_tile": cutile_version}


def _abi_parameters(name, constraint):
    if isinstance(constraint, ConstantConstraint):
        return []
    if isinstance(constraint, ScalarConstraint):
        return [{"name": name, "type": _dtype_name(constraint.dtype)}]
    if isinstance(constraint, ArrayConstraint):
        index_type = _dtype_name(constraint.index_dtype)
        return (
            [{"name": name, "type": f"{_dtype_name(constraint.dtype)}*"}]
            + [{"name": f"{name}_shape_{i}", "type": index_type} for i in range(constraint.ndim)]
            + [{"name": f"{name}_stride_{i}", "type": index_type} for i in range(constraint.ndim)]
        )
    raise TypeError(f"ABI metadata does not support {type(constraint).__name__}")


def cutile_to_cubin(kernel, launch_config: Mapping, cubin_path):
    """Compile ``kernel`` for one specialization and return its binary signature.

    ``parameters`` supplies nonconstant constraints in Python function order;
    ``constants`` supplies named compile-time values. Array shapes and strides
    are measured in elements. The caller must honor every supplied constraint.
    """
    path = Path(cubin_path)
    parameters = dict(launch_config.get("parameters", {}))
    constants = dict(launch_config.get("constants", {}))
    overlap = parameters.keys() & constants.keys()
    if overlap:
        raise ValueError(f"parameters and constants overlap: {sorted(overlap)}")

    # cuTile's kernel object keeps the Python signature here. This lets callers
    # use named constraints without depending on dictionary insertion order.
    python_parameters = kernel._annotated_function.pysig.parameters
    unknown = (parameters.keys() | constants.keys()) - python_parameters.keys()
    missing = python_parameters.keys() - (parameters.keys() | constants.keys())
    if unknown or missing:
        raise ValueError(f"unknown parameters: {sorted(unknown)}; missing parameters: {sorted(missing)}")

    constraints = []
    abi = []
    constraint_metadata = []
    compile_constants = {}
    for name in python_parameters:
        if name in constants:
            constraint = ConstantConstraint(constants[name])
            compile_constants[name] = constants[name]
        else:
            constraint = parameters[name]
            if isinstance(constraint, ConstantConstraint):
                compile_constants[name] = constraint.value
        constraints.append(constraint)
        abi.extend(_abi_parameters(name, constraint))
        metadata = _constraint_metadata(name, constraint)
        if metadata is not None:
            constraint_metadata.append(metadata)

    symbol = launch_config.get("symbol", kernel._annotated_function.pyfunc.__name__)
    signature = KernelSignature(constraints, CallingConvention.cutile_python_v2(), symbol=symbol)
    gpu_code = launch_config["gpu_code"]
    path.parent.mkdir(parents=True, exist_ok=True)
    export_kernel(kernel, [signature], path, gpu_code=gpu_code, output_format="cubin")

    return {
        "name": symbol,
        "cubin_path": str(path.resolve()),
        "gpu_code": gpu_code,
        "calling_convention": "cutile_python_v2",
        "parameters": abi,
        "constants": compile_constants,
        "constraints": constraint_metadata,
        "grid": tuple(launch_config["grid"]),
        "block": tuple(launch_config["block"]),
        "toolchain": _toolchain_metadata(),
    }


def write_abi_json(info: Mapping, abi_path):
    """Write versioned runtime ABI metadata; compile-time constants are omitted."""
    document = {
        "schema_version": 1,
        "name": info["name"],
        "symbol": info["name"],
        "cubin_path": info["cubin_path"],
        "gpu_code": info["gpu_code"],
        "calling_convention": info["calling_convention"],
        "parameters": info["parameters"],
        "constraints": info["constraints"],
        "grid": list(info["grid"]),
        "block": list(info["block"]),
        "launch": {"grid": list(info["grid"]), "block": list(info["block"])},
        "toolchain": info["toolchain"],
    }
    path = Path(abi_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document


export_cutile_to_cubin = cutile_to_cubin
