"""Read a named two-dimensional GGUF weight in its packed block layout."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.paths import ensure_test_paths

ensure_test_paths()

from mambo.utils.gguf_manifest import inspect_gguf  # noqa: E402


def read_model_weight(
    model_path: str | Path, weight_name: str, *, expected_type: str | None = None
) -> tuple[np.memmap, int, int]:
    """Return packed bytes [N, K/block_elements, block_bytes], N, and K."""
    model_path = Path(model_path)
    manifest = inspect_gguf(model_path)
    tensor = next((item for item in manifest["tensors"] if item["name"] == weight_name), None)
    if tensor is None:
        raise KeyError(f"GGUF has no tensor named {weight_name!r}")
    if expected_type is not None and tensor["type"]["name"] != expected_type:
        raise AssertionError(
            f"{weight_name} must use {expected_type}, got {tensor['type']['name']}"
        )
    if len(tensor["shape"]) != 2:
        raise ValueError(f"{weight_name} must be a two-dimensional weight")
    k, n = tensor["shape"]
    block_elements = tensor["type"]["block_elements"]
    block_bytes = tensor["type"]["block_bytes"]
    if (
        not block_elements or not block_bytes or k % block_elements
        or tensor["row_bytes"] != (k // block_elements) * block_bytes
        or tensor["n_bytes"] != n * tensor["row_bytes"]
    ):
        raise ValueError(f"{weight_name} has an inconsistent GGUF block layout")
    payload = np.memmap(
        model_path,
        mode="r",
        dtype=np.uint8,
        offset=tensor["offset"]["absolute"],
        shape=(n, k // block_elements, block_bytes),
    )
    return payload, n, k
