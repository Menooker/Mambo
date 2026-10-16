"""Read GGUF headers and tensor inventories without loading model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
from pathlib import Path
from typing import Any, BinaryIO, Iterable

SCHEMA_VERSION = 1
_INLINE_ARRAY_LIMIT = 256
_INLINE_STRING_LIMIT = 1 << 20
_HASH_CHUNK_BYTES = 1 << 20

# GGUF metadata value types, by gguf_type ID.
_META_TYPES: dict[int, tuple[str, str, int | None]] = {
    0: ("u8", "B", 1), 1: ("i8", "b", 1), 2: ("u16", "H", 2),
    3: ("i16", "h", 2), 4: ("u32", "I", 4), 5: ("i32", "i", 4),
    6: ("f32", "f", 4), 7: ("bool", "?", 1), 8: ("string", "", None),
    9: ("array", "", None), 10: ("u64", "Q", 8), 11: ("i64", "q", 8),
    12: ("f64", "d", 8),
}

# ggml_type ID -> stable name, elements per block, bytes per block.
_TENSOR_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("f32", 1, 4), 1: ("f16", 1, 2), 2: ("q4_0", 32, 18),
    3: ("q4_1", 32, 20), 6: ("q5_0", 32, 22), 7: ("q5_1", 32, 24),
    8: ("q8_0", 32, 34), 10: ("q2_k", 256, 84), 11: ("q3_k", 256, 110),
    12: ("q4_k", 256, 144), 13: ("q5_k", 256, 176), 14: ("q6_k", 256, 210),
    15: ("q8_k", 256, 292), 16: ("iq2_xxs", 256, 66), 17: ("iq2_xs", 256, 74),
    18: ("iq3_xxs", 256, 98), 19: ("iq1_s", 256, 50), 20: ("iq4_nl", 32, 18),
    21: ("iq3_s", 256, 110), 22: ("iq2_s", 256, 82), 23: ("iq4_xs", 256, 136),
    24: ("i8", 1, 1), 25: ("i16", 1, 2), 26: ("i32", 1, 4), 27: ("i64", 1, 8),
    28: ("f64", 1, 8), 29: ("iq1_m", 256, 56), 30: ("bf16", 1, 2),
    34: ("tq1_0", 256, 54), 35: ("tq2_0", 256, 66), 39: ("mxfp4", 32, 17),
    40: ("nvfp4", 64, 36), 41: ("q1_0", 128, 18), 42: ("q2_0", 64, 18),
}
_LAYER_RE = re.compile(r"^blk\.(\d+)\.")


class GGUFError(ValueError):
    """Raised when a GGUF header or inventory is malformed or unsupported."""


def _read_exact(fp: BinaryIO, size: int, what: str) -> bytes:
    if size < 0:
        raise GGUFError(f"negative length for {what}: {size}")
    data = fp.read(size)
    if len(data) != size:
        raise GGUFError(f"truncated GGUF while reading {what}: wanted {size}, got {len(data)}")
    return data


def _unpack(fp: BinaryIO, fmt: str, what: str) -> tuple[Any, ...]:
    return struct.unpack("<" + fmt, _read_exact(fp, struct.calcsize("<" + fmt), what))


def _read_string_bytes(fp: BinaryIO, what: str) -> tuple[int, int, bytes | None, str]:
    (n_bytes,) = _unpack(fp, "Q", f"{what} length")
    payload_offset = fp.tell()
    digest = hashlib.sha256()
    if n_bytes <= _INLINE_STRING_LIMIT:
        payload = _read_exact(fp, n_bytes, what)
        digest.update(payload)
        return n_bytes, payload_offset, payload, digest.hexdigest()
    remaining = n_bytes
    while remaining:
        chunk = _read_exact(fp, min(remaining, _HASH_CHUNK_BYTES), what)
        digest.update(chunk)
        remaining -= len(chunk)
    return n_bytes, payload_offset, None, digest.hexdigest()


def _decode_string(data: bytes, what: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GGUFError(f"invalid UTF-8 in {what}") from exc


def _hash_range(fp: BinaryIO, size: int, what: str) -> str:
    digest = hashlib.sha256()
    remaining = size
    while remaining:
        chunk = _read_exact(fp, min(remaining, _HASH_CHUNK_BYTES), what)
        digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def _read_metadata_value(fp: BinaryIO, type_id: int, key: str) -> tuple[str, Any]:
    if type_id not in _META_TYPES:
        raise GGUFError(f"metadata key {key!r} uses unknown GGUF value type {type_id}")
    type_name, fmt, fixed_size = _META_TYPES[type_id]
    if type_id == 8:
        n_bytes, payload_offset, payload, digest = _read_string_bytes(fp, f"metadata {key}")
        if payload is None:
            return type_name, {"kind": "string_summary", "length_bytes": n_bytes,
                               "payload_offset": payload_offset, "sha256": digest}
        return type_name, _decode_string(payload, f"metadata {key}")
    if type_id != 9:
        assert fixed_size is not None
        value = struct.unpack("<" + fmt, _read_exact(fp, fixed_size, f"metadata {key}"))[0]
        if type_id == 7:
            value = bool(value)
        elif type_id in (6, 12) and not math.isfinite(value):
            value = str(value)
        return type_name, value

    (element_type_id,) = _unpack(fp, "I", f"metadata {key} array type")
    (count,) = _unpack(fp, "Q", f"metadata {key} array length")
    if element_type_id == 9:
        raise GGUFError(f"metadata key {key!r} contains a nested array")
    if element_type_id not in _META_TYPES:
        raise GGUFError(f"metadata key {key!r} array has unknown element type {element_type_id}")
    element_name, element_fmt, element_size = _META_TYPES[element_type_id]
    data_offset = fp.tell()
    if element_type_id == 8:
        digest = hashlib.sha256()
        values: list[str] | None = [] if count <= _INLINE_ARRAY_LIMIT else None
        for index in range(count):
            length_bytes = _read_exact(fp, 8, f"metadata {key}[{index}] string length")
            (n_bytes,) = struct.unpack("<Q", length_bytes)
            digest.update(length_bytes)
            if values is not None and n_bytes <= _INLINE_STRING_LIMIT:
                raw = _read_exact(fp, n_bytes, f"metadata {key}[{index}]")
                digest.update(raw)
                values.append(_decode_string(raw, f"metadata {key}[{index}]"))
            else:
                values = None
                remaining = n_bytes
                while remaining:
                    chunk = _read_exact(fp, min(remaining, _HASH_CHUNK_BYTES), f"metadata {key}[{index}]")
                    digest.update(chunk)
                    remaining -= len(chunk)
        payload_bytes = fp.tell() - data_offset
        if values is not None:
            result: Any = {"element_type": element_name, "count": count, "values": values}
        else:
            result = {"element_type": element_name, "count": count, "payload_offset": data_offset,
                      "payload_nbytes": payload_bytes, "sha256": digest.hexdigest()}
        return type_name, result

    assert element_size is not None
    total_bytes = count * element_size
    if count <= _INLINE_ARRAY_LIMIT:
        raw = _read_exact(fp, total_bytes, f"metadata {key} array")
        values = list(struct.unpack("<" + element_fmt * count, raw)) if count else []
        if element_type_id == 7:
            values = [bool(value) for value in values]
        if element_type_id in (6, 12):
            values = [value if math.isfinite(value) else str(value) for value in values]
        return type_name, {"element_type": element_name, "count": count, "values": values}
    digest = _hash_range(fp, total_bytes, f"metadata {key} array")
    return type_name, {"element_type": element_name, "count": count, "payload_offset": data_offset,
                       "payload_nbytes": total_bytes, "sha256": digest}


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _canonical_hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _meta_value(metadata: dict[str, Any], key: str, default: Any = None) -> Any:
    return metadata.get(key, {}).get("value", default)


def _meta_int(metadata: dict[str, Any], key: str) -> int | None:
    value = _meta_value(metadata, key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _array_values(value: Any) -> list[Any] | None:
    if isinstance(value, dict) and isinstance(value.get("values"), list):
        return value["values"]
    return None


def _layer_kind_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    label = value.lower().replace("-", "_").replace(" ", "_")
    if any(token in label for token in ("recurrent", "gated_delta", "linear_attention", "gdn")):
        return "gated_delta_net"
    if any(token in label for token in ("full_attention", "full_attn", "dense_attention")):
        return "full_attention"
    if label in ("attention", "attn"):
        return "full_attention"
    if "sliding_window" in label or label == "swa":
        return "sliding_window_attention"
    return None


def _metadata_layer_set(metadata: dict[str, Any], key: str, layer_count: int) -> set[int] | None:
    entry = metadata.get(key)
    if entry is None:
        return None
    value = entry.get("value")
    values = _array_values(value)
    if values is None:
        return None
    element_type = value.get("element_type") if isinstance(value, dict) else None
    is_mask = element_type == "bool" or (
        ("pattern" in key.lower() or "mask" in key.lower())
        and len(values) == layer_count
        and all(item in (0, 1, False, True) for item in values)
    )
    if is_mask:
        return {index for index, enabled in enumerate(values) if bool(enabled)}
    try:
        return {int(item) for item in values}
    except (TypeError, ValueError):
        return None


def _derive_model_summary(metadata: dict[str, Any], tensors: list[dict[str, Any]]) -> dict[str, Any]:
    architecture = _meta_value(metadata, "general.architecture")
    if not isinstance(architecture, str):
        raise GGUFError("missing or invalid general.architecture metadata")
    prefix = architecture + "."
    layer_names: dict[int, list[str]] = {}
    for tensor in tensors:
        match = _LAYER_RE.match(tensor["name"])
        if match:
            layer_names.setdefault(int(match.group(1)), []).append(tensor["name"])

    explicit_types: dict[int, str] = {}
    type_keys: list[str] = []
    explicit_type_arrays: list[tuple[str, list[Any]]] = []
    for key, entry in metadata.items():
        leaf = key.rsplit(".", 1)[-1].lower()
        if key.startswith(prefix) and ("layer_types" in leaf or "block_types" in leaf):
            values = _array_values(entry.get("value"))
            if values is not None:
                type_keys.append(key)
                explicit_type_arrays.append((key, values))
                explicit_types.update({i: _layer_kind_label(value) or "unclassified" for i, value in enumerate(values)})

    block_count = _meta_int(metadata, prefix + "block_count")
    sliding_pattern_entry = metadata.get(prefix + "attention.sliding_window_pattern")
    sliding_pattern = _array_values(sliding_pattern_entry.get("value")) if sliding_pattern_entry else None
    sliding_pattern_valid = (
        sliding_pattern is not None
        and (block_count is None or len(sliding_pattern) == block_count)
        and all(isinstance(value, (bool, int)) and value in (0, 1, False, True) for value in sliding_pattern)
    )

    layers: list[dict[str, Any]] = []
    for index in sorted(layer_names):
        names = layer_names[index]
        lower_names = [name.lower() for name in names]
        has_full_q = any(name.startswith(f"blk.{index}.attn_q.") for name in names)
        has_full_k = any(name.startswith(f"blk.{index}.attn_k.") for name in names)
        has_fused_qkv = any(name.startswith(f"blk.{index}.attn_qkv.") or f"blk.{index}.ssm_" in name for name in names)
        if has_fused_qkv:
            inventory_kind, inventory_source = "gated_delta_net", "tensor_inventory:attn_qkv_or_ssm_tensor"
        elif has_full_q and has_full_k:
            inventory_kind, inventory_source = "full_attention", "tensor_inventory:separate_q_and_k_tensors"
        else:
            inventory_kind, inventory_source = "unclassified", "insufficient_tensor_evidence"

        if architecture.lower().startswith("dflash") and sliding_pattern_valid and index < len(sliding_pattern):
            metadata_kind = "sliding_window_attention" if bool(sliding_pattern[index]) else "full_attention"
            kind, source = metadata_kind, "metadata:" + prefix + "attention.sliding_window_pattern"
        elif index in explicit_types and explicit_types[index] != "unclassified":
            kind, source = explicit_types[index], "metadata:" + ",".join(type_keys)
        else:
            kind, source = inventory_kind, inventory_source

        if (architecture.lower().startswith("qwen35") and inventory_kind != "unclassified"):
            # Tensor names distinguish Qwen35's GDN and full-attention layers.
            kind, source = inventory_kind, inventory_source
        is_mtp = any(".mtp." in name or ".nextn." in name for name in lower_names)
        layers.append({
            "index": index,
            "role": "mtp" if is_mtp else "trunk_or_draft",
            "kind": kind,
            "classification_source": source,
            "evidence_tensors": sorted(
                name for name in names
                if any(token in name.lower() for token in ("attn_q", "attn_k", "ssm_", ".mtp.", ".nextn."))
            ),
        })

    declared_matches_observed = block_count is None or block_count == len(layers)
    contiguous_ids = not layers or [layer["index"] for layer in layers] == list(range(len(layers)))
    all_classified = all(layer["kind"] != "unclassified" for layer in layers)
    consistency_checks: dict[str, bool] = {
        "declared_block_count_matches_observed_layers": declared_matches_observed,
        "observed_layer_ids_contiguous_from_zero": contiguous_ids,
        "all_observed_layers_classified": all_classified,
    }
    layer_by_id = {layer["index"]: layer for layer in layers}
    for key, values in explicit_type_arrays:
        recognized = {
            index: _layer_kind_label(value)
            for index, value in enumerate(values)
            if _layer_kind_label(value) is not None
        }
        if recognized:
            consistency_checks[f"metadata_layer_types_match_tensor_inventory:{key}"] = all(
                index in layer_by_id
                and (
                    layer_by_id[index]["kind"] == expected
                    or (expected == "full_attention" and layer_by_id[index]["kind"] == "sliding_window_attention")
                )
                for index, expected in recognized.items()
            )

    if architecture.lower().startswith("qwen35"):
        nextn_predict_layers = _meta_int(metadata, prefix + "nextn_predict_layers")
        consistency_checks["nextn_predict_layers_matches_detected_mtp_count"] = (
            nextn_predict_layers is not None and nextn_predict_layers == sum(layer["role"] == "mtp" for layer in layers)
        )
        interval = _meta_int(metadata, prefix + "full_attention_interval")
        if interval is not None:
            trunk_layers = [layer for layer in layers if layer["role"] != "mtp"]
            interval_valid = interval > 0 and all(
                layer["kind"] == ("full_attention" if (layer["index"] + 1) % interval == 0 else "gated_delta_net")
                for layer in trunk_layers
            )
            consistency_checks["full_attention_interval_matches_trunk_tensor_inventory"] = interval_valid

        trunk_ids = {layer["index"] for layer in layers if layer["role"] != "mtp"}
        actual_gdn = {layer["index"] for layer in layers if layer["kind"] == "gated_delta_net"}
        actual_full_trunk = {
            layer["index"] for layer in layers
            if layer["role"] != "mtp" and layer["kind"] == "full_attention"
        }
        for key in metadata:
            lowered = key.lower()
            if not key.startswith(prefix) or "layer" not in lowered:
                continue
            if "recurrent" in lowered:
                declared = _metadata_layer_set(metadata, key, block_count or len(layers))
                if declared is not None:
                    consistency_checks[f"metadata_layer_set_matches_tensor_inventory:{key}"] = declared == actual_gdn
            elif "full_attention" in lowered or "full_attn" in lowered:
                declared = _metadata_layer_set(metadata, key, block_count or len(layers))
                if declared is not None:
                    expected = actual_full_trunk if declared <= trunk_ids else {
                        layer["index"] for layer in layers if layer["kind"] == "full_attention"
                    }
                    consistency_checks[f"metadata_layer_set_matches_tensor_inventory:{key}"] = declared == expected

    if architecture.lower().startswith("dflash") and sliding_pattern is not None:
        pattern_matches = (
            sliding_pattern_valid
            and len(sliding_pattern) == len(layers)
            and all(
                layer["kind"] == ("sliding_window_attention" if bool(sliding_pattern[layer["index"]]) else "full_attention")
                for layer in layers
            )
        )
        consistency_checks["sliding_window_pattern_matches_observed_layers"] = pattern_matches

    summary: dict[str, Any] = {
        "architecture": architecture,
        "declared_block_count": block_count,
        "observed_layer_count": len(layers),
        "observed_layer_ids": [layer["index"] for layer in layers],
        "layers": layers,
        "layer_counts": {
            kind: sum(layer["kind"] == kind and layer["role"] != "mtp" for layer in layers)
            for kind in ("gated_delta_net", "full_attention", "sliding_window_attention", "unclassified")
        },
        "mtp_layer_count": sum(layer["role"] == "mtp" for layer in layers),
        "metadata_layer_type_keys": type_keys,
        "consistency_checks": consistency_checks,
    }
    for suffix in (
        "embedding_length", "feed_forward_length", "block_count", "context_length",
        "attention.head_count", "attention.head_count_kv", "attention.key_length",
        "attention.value_length", "attention.sliding_window", "ssm.conv_kernel",
        "ssm.state_size", "ssm.group_count", "ssm.time_step_rank", "mtp.block_count",
        "mtp.n_predict", "nextn_predict_layers", "full_attention_interval",
        "block_size", "conv_kernel_size", "conv_group_size", "selector_rank",
        "selector_top_k", "target_layers", "attention.sliding_window_pattern",
    ):
        value = _meta_value(metadata, prefix + suffix)
        if value is not None:
            summary.setdefault("metadata", {})[suffix] = value
    feature_keys = [
        key for key in metadata
        if key.lower().startswith("dflash.") and "layer" in key.lower()
        and ("target" in key.lower() or "feature" in key.lower())
    ]
    if feature_keys:
        summary["target_feature_metadata"] = {key: metadata[key]["value"] for key in sorted(feature_keys)}
    return summary


def _tensor_record(
    name: str, dims: list[int], type_id: int, relative_offset: int,
    data_offset: int, alignment: int, file_size: int,
) -> dict[str, Any]:
    if type_id not in _TENSOR_TYPES:
        raise GGUFError(f"tensor {name!r} uses unsupported ggml type id {type_id}")
    if not dims or any(dim <= 0 for dim in dims):
        raise GGUFError(f"tensor {name!r} has invalid dimensions: {dims}")
    type_name, block_elements, block_bytes = _TENSOR_TYPES[type_id]
    row_elements = dims[0]
    if row_elements % block_elements:
        raise GGUFError(f"tensor {name!r} row width {row_elements} is not divisible by {type_name} block {block_elements}")
    row_bytes = (row_elements // block_elements) * block_bytes
    n_elements = math.prod(dims)
    n_bytes = row_bytes * math.prod(dims[1:])
    absolute_offset = data_offset + relative_offset
    if relative_offset < 0 or relative_offset % alignment:
        raise GGUFError(f"tensor {name!r} relative offset {relative_offset} is not {alignment}-byte aligned")
    if absolute_offset < data_offset or absolute_offset + n_bytes > file_size:
        raise GGUFError(f"tensor {name!r} payload exceeds file size {file_size}")
    return {
        "name": name,
        "type": {"id": type_id, "name": type_name, "block_elements": block_elements, "block_bytes": block_bytes},
        # GGUF stores dimensions in ggml ne order: dimension 0 is contiguous.
        "shape": dims,
        "shape_order": "ggml_ne_order",
        "n_elements": n_elements,
        "row_bytes": row_bytes,
        "n_bytes": n_bytes,
        "offset": {"relative": relative_offset, "absolute": absolute_offset},
    }


def inspect_gguf(path: str | os.PathLike[str], *, full_sha256: bool = False) -> dict[str, Any]:
    """Inspect GGUF metadata and tensor descriptors without reading weights."""
    model_path = Path(path).expanduser().resolve()
    stat = model_path.stat()
    file_size = stat.st_size
    metadata: dict[str, dict[str, Any]] = {}
    headers: list[tuple[str, list[int], int, int]] = []
    with model_path.open("rb") as fp:
        if _read_exact(fp, 4, "magic") != b"GGUF":
            raise GGUFError(f"{model_path} is not a GGUF file")
        (version,) = _unpack(fp, "I", "version")
        if version not in (2, 3):
            raise GGUFError(f"unsupported GGUF version {version}; expected 2 or 3")
        n_tensors, n_metadata = _unpack(fp, "QQ", "header counts")
        if n_tensors > 10_000_000 or n_metadata > 10_000_000:
            raise GGUFError(f"unreasonable GGUF header counts: tensors={n_tensors}, metadata={n_metadata}")
        for _ in range(n_metadata):
            key_n, _, key_bytes, _ = _read_string_bytes(fp, "metadata key")
            if key_bytes is None:
                raise GGUFError(f"metadata key length {key_n} exceeds supported limit")
            key = _decode_string(key_bytes, "metadata key")
            if key in metadata:
                raise GGUFError(f"duplicate GGUF metadata key {key!r}")
            (type_id,) = _unpack(fp, "I", f"metadata {key} type")
            value_type, value = _read_metadata_value(fp, type_id, key)
            metadata[key] = {"type": value_type, "value": value}

        names: set[str] = set()
        for _ in range(n_tensors):
            name_n, _, name_bytes, _ = _read_string_bytes(fp, "tensor name")
            if name_bytes is None:
                raise GGUFError(f"tensor name length {name_n} exceeds supported limit")
            name = _decode_string(name_bytes, "tensor name")
            if name in names:
                raise GGUFError(f"duplicate tensor name {name!r}")
            names.add(name)
            (n_dims,) = _unpack(fp, "I", f"tensor {name} dimension count")
            if n_dims < 1 or n_dims > 4:
                raise GGUFError(f"tensor {name!r} has unsupported dimension count {n_dims}")
            dims = list(_unpack(fp, "Q" * n_dims, f"tensor {name} dimensions"))
            type_id, relative_offset = _unpack(fp, "IQ", f"tensor {name} type and offset")
            headers.append((name, dims, type_id, relative_offset))

        alignment = _meta_int(metadata, "general.alignment")
        if alignment is None:
            if "general.alignment" in metadata:
                raise GGUFError("GGUF general.alignment must be an integer scalar")
            alignment = 32
        if alignment <= 0 or alignment & (alignment - 1):
            raise GGUFError(f"GGUF general.alignment must be a positive power of two, got {alignment}")
        data_offset = _align_up(fp.tell(), alignment)
        if data_offset > file_size:
            raise GGUFError(f"tensor data section starts outside file at {data_offset} > {file_size}")
        tensors = [
            _tensor_record(name, dims, type_id, relative, data_offset, alignment, file_size)
            for name, dims, type_id, relative in headers
        ]
        previous_end, previous_name = data_offset, None
        for tensor in sorted(tensors, key=lambda item: item["offset"]["absolute"]):
            start = tensor["offset"]["absolute"]
            if start < previous_end:
                raise GGUFError(f"tensor payload overlap: {previous_name!r} overlaps {tensor['name']!r}")
            previous_end = start + tensor["n_bytes"]
            previous_name = tensor["name"]
        data_size = max(0, previous_end - data_offset)

    metadata_hash = _canonical_hash(metadata)
    inventory_hash = _canonical_hash(tensors)
    identity: dict[str, Any] = {
        "path": str(model_path), "size_bytes": file_size, "mtime_ns": stat.st_mtime_ns,
        "metadata_sha256": metadata_hash, "tensor_inventory_sha256": inventory_hash,
    }
    if full_sha256:
        digest = hashlib.sha256()
        with model_path.open("rb") as fp:
            for chunk in iter(lambda: fp.read(_HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
        identity["full_sha256"] = digest.hexdigest()

    counts: dict[str, int] = {}
    type_bytes: dict[str, int] = {}
    for tensor in tensors:
        name = tensor["type"]["name"]
        counts[name] = counts.get(name, 0) + 1
        type_bytes[name] = type_bytes.get(name, 0) + tensor["n_bytes"]
    model_summary = _derive_model_summary(metadata, tensors)
    validation_checks = {
        "tensor_inventory_count_matches_header": len(tensors) == n_tensors,
        "metadata_entry_count_matches_header": len(metadata) == n_metadata,
        **model_summary["consistency_checks"],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "identity": identity,
        "gguf": {"version": version, "tensor_count": n_tensors, "metadata_count": n_metadata,
                 "alignment": alignment, "data_offset": data_offset, "data_size_bytes": data_size},
        "metadata": metadata,
        "tensor_summary": {"count": len(tensors), "type_counts": dict(sorted(counts.items())),
                           "type_bytes": dict(sorted(type_bytes.items())), "inventory_sha256": inventory_hash},
        "model": model_summary,
        "reference": {
            "repository": "llama.cpp",
            "fork_url": "https://github.com/Menooker/llama.cpp-adaptive-kv-streaming",
            "branch": "feature/kv-stream-phase-arena",
            "revision": "21b359d8c13e71b4d60ed1bed00eff5e59c640a2",
            "commit": "21b359d8c13e71b4d60ed1bed00eff5e59c640a2",
            "upstream_url": "https://github.com/ggerganov/llama.cpp.git",
        },
        "validation": {"checks": validation_checks, "passed": all(validation_checks.values())},
        "tensors": tensors,
    }


def read_tensor_payload(
    path: str | os.PathLike[str], tensor: dict[str, Any] | str, *, max_bytes: int | None = None,
) -> bytes:
    """Read one tensor payload described by an ``inspect_gguf`` manifest."""
    if isinstance(tensor, str):
        tensor_name = tensor
        manifest = inspect_gguf(path)
        tensor = next((item for item in manifest["tensors"] if item["name"] == tensor_name), None)
        if tensor is None:
            raise KeyError(f"tensor {tensor_name!r} not found in {path}")
    n_bytes = tensor["n_bytes"]
    if max_bytes is not None and n_bytes > max_bytes:
        raise ValueError(f"tensor payload is {n_bytes} bytes, above max_bytes={max_bytes}")
    with Path(path).open("rb") as fp:
        fp.seek(tensor["offset"]["absolute"])
        return _read_exact(fp, n_bytes, f"tensor {tensor['name']} payload")


def link_model_manifests(*manifests: dict[str, Any]) -> dict[str, Any]:
    """Report inventory matches and shape-checked Qwen35 -> DFlash runtime aliases.

    Same-name and same-layout tensors are only reported as matches; the header
    inventory cannot prove their payload bytes are identical. The DFlash
    runtime aliases are separate: llama.cpp allows its token embedding and LM
    head to be supplied by the target model when the draft file omits them.
    """
    models = [(m["model"]["architecture"], {t["name"]: t for t in m["tensors"]}) for m in manifests]
    matches: list[dict[str, Any]] = []
    runtime_aliases: list[dict[str, Any]] = []
    for i, (left_arch, left) in enumerate(models):
        for right_arch, right in models[i + 1:]:
            for name in sorted(left.keys() & right.keys()):
                a, b = left[name], right[name]
                if a["shape"] == b["shape"] and a["type"]["id"] == b["type"]["id"] and a["n_bytes"] == b["n_bytes"]:
                    matches.append({"tensor": name, "shape": a["shape"], "type": a["type"]["name"],
                                    "models": [left_arch, right_arch],
                                    "offsets": [a["offset"]["absolute"], b["offset"]["absolute"]],
                                    "payload_identity": "not_checked"})

            target = left if left_arch == "qwen35" else right if right_arch == "qwen35" else None
            draft = right if right_arch == "dflash" else left if left_arch == "dflash" else None
            if target is None or draft is None:
                continue
            target_manifest = manifests[i] if left_arch == "qwen35" else manifests[i + 1]
            draft_manifest = manifests[i + 1] if right_arch == "dflash" else manifests[i]
            target_meta = target_manifest["metadata"]
            draft_meta = draft_manifest["metadata"]
            hidden = _meta_int(draft_meta, "dflash.embedding_length")

            def array_count(entry: dict[str, Any]) -> int | None:
                value = entry.get("value")
                if isinstance(value, dict) and isinstance(value.get("count"), int):
                    return value["count"]
                values = _array_values(value)
                return len(values) if values is not None else None

            vocab = _meta_int(draft_meta, "dflash.vocab_size")
            if vocab is None:
                token_entry = draft_meta.get("tokenizer.ggml.tokens")
                if token_entry:
                    vocab = array_count(token_entry)
            for name in ("token_embd.weight", "output.weight"):
                source_tensor = target.get(name)
                if source_tensor is None or name in draft:
                    continue
                expected_shape = [hidden, vocab] if hidden is not None and vocab is not None else None
                if expected_shape is not None and source_tensor["shape"] != expected_shape:
                    continue
                runtime_aliases.append({
                    "consumer_architecture": "dflash",
                    "consumer_tensor": name,
                    "source_architecture": "qwen35",
                    "source_tensor": name,
                    "shape": source_tensor["shape"],
                    "type": source_tensor["type"]["name"],
                    "binding": "reuse_target_tensor_when_dflash_local_tensor_is_absent",
                    "shape_check": "matched dflash hidden size and vocabulary metadata",
                })
    target_feature_layers: list[int] | None = None
    dflash_manifest = next((m for m in manifests if m["model"]["architecture"] == "dflash"), None)
    qwen_manifest = next((m for m in manifests if m["model"]["architecture"] == "qwen35"), None)
    feature_metadata = dflash_manifest["model"].get("target_feature_metadata", {}) if dflash_manifest else {}
    for value in feature_metadata.values():
        values = _array_values(value)
        if values is not None and all(isinstance(item, int) for item in values):
            target_feature_layers = values
            break
    feature_check: dict[str, Any] | None = None
    if target_feature_layers is not None and qwen_manifest is not None:
        trunk_ids = {layer["index"] for layer in qwen_manifest["model"]["layers"] if layer["role"] != "mtp"}
        feature_check = {
            "name": "dflash_target_feature_layers_exist_in_qwen_trunk",
            "passed": all(index in trunk_ids for index in target_feature_layers),
            "target_layer_ids": target_feature_layers,
        }
    link_checks = [] if feature_check is None else [feature_check]
    return {
        "schema_version": SCHEMA_VERSION,
        "same_name_layout_matches_unverified_payload": matches,
        "runtime_aliases": runtime_aliases,
        "target_feature_layers": target_feature_layers,
        "validation": {"checks": link_checks, "passed": all(item["passed"] for item in link_checks)},
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="one or more GGUF model paths")
    parser.add_argument("--output-dir", type=Path, required=True, help="directory for JSON manifests")
    parser.add_argument("--full-sha256", action="store_true", help="hash all file bytes, including weights")
    args = parser.parse_args(argv)
    manifests = [inspect_gguf(path, full_sha256=args.full_sha256) for path in args.paths]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    architecture_names = [re.sub(r"[^A-Za-z0-9_.-]+", "_", m["model"]["architecture"]) for m in manifests]
    for index, (arch, manifest) in enumerate(zip(architecture_names, manifests)):
        suffix = f"-{index + 1}" if architecture_names.count(arch) > 1 else ""
        _write_json(args.output_dir / f"{arch}{suffix}.json", manifest)
    if len(manifests) > 1:
        _write_json(args.output_dir / "model-links.json", link_model_manifests(*manifests))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
