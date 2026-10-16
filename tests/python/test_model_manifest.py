from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/python"))

from mambo.utils.gguf_manifest import GGUFError, inspect_gguf, link_model_manifests, read_tensor_payload


def _gguf_string(value: str) -> bytes:
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _align(value: int, n: int = 32) -> int:
    return (value + n - 1) // n * n


def _make_gguf(path: Path, *, architecture: str, tensors: list[tuple[str, list[int], int]],
               block_count: int, invalid_offset: bool = False, alignment: int = 32,
               extra_metadata: list[tuple[str, int, object]] | None = None) -> None:
    # Metadata tuples are (key, GGUF type ID, value).
    metadata = [
        ("general.architecture", 8, architecture),
        ("general.alignment", 4, alignment),
        (f"{architecture}.embedding_length", 4, 2),
        (f"{architecture}.block_count", 4, block_count),
        ("tokenizer.ggml.tokens", 9, (8, [f"tok-{i}" for i in range(300)])),
    ]
    metadata.extend(extra_metadata or [])
    meta_bytes = bytearray()
    for key, type_id, value in metadata:
        meta_bytes += _gguf_string(key)
        meta_bytes += struct.pack("<I", type_id)
        if type_id == 8:
            meta_bytes += _gguf_string(value)
        elif type_id == 4:
            meta_bytes += struct.pack("<I", value)
        elif type_id == 9:
            element_type_id, elements = value
            meta_bytes += struct.pack("<IQ", element_type_id, len(elements))
            if element_type_id == 8:
                for element in elements:
                    meta_bytes += _gguf_string(element)
            elif element_type_id == 7:
                meta_bytes += bytes(1 if element else 0 for element in elements)
            elif element_type_id == 4:
                for element in elements:
                    meta_bytes += struct.pack("<I", element)
            else:
                raise ValueError(f"unsupported test metadata array element type {element_type_id}")
        elif type_id == 7:
            meta_bytes += struct.pack("<?", value)

    # GGML storage bytes for the fixture's types: F32 and IQ3_S.
    payload_sizes = {0: 4, 21: 110}
    block_sizes = {0: 1, 21: 256}
    offsets: list[int] = []
    cursor = 0
    for _, dims, type_id in tensors:
        cursor = _align(cursor)
        offsets.append(cursor)
        row_elems = dims[0]
        row_bytes = row_elems // block_sizes[type_id] * payload_sizes[type_id]
        cursor += row_bytes
        for dim in dims[1:]:
            cursor += row_bytes * (dim - 1)

    header = bytearray(b"GGUF")
    header += struct.pack("<IQQ", 3, len(tensors), len(metadata))
    header += meta_bytes
    for index, (name, dims, type_id) in enumerate(tensors):
        header += _gguf_string(name)
        header += struct.pack("<I", len(dims))
        header += struct.pack("<" + "Q" * len(dims), *dims)
        relative = (1 << 20) if invalid_offset else offsets[index]
        header += struct.pack("<IQ", type_id, relative)

    data_start = _align(len(header), alignment)
    output = header + bytes(data_start - len(header))
    max_end = 0
    for index, (_, dims, type_id) in enumerate(tensors):
        if invalid_offset:
            continue
        row_bytes = dims[0] // block_sizes[type_id] * payload_sizes[type_id]
        n_bytes = row_bytes
        for dim in dims[1:]:
            n_bytes *= dim
        relative = offsets[index]
        if len(output) < data_start + relative:
            output += bytes(data_start + relative - len(output))
        if len(output) < data_start + relative + n_bytes:
            output += bytes(data_start + relative + n_bytes - len(output))
        max_end = max(max_end, relative + n_bytes)
    path.write_bytes(output)


class GGUFManifestTests(unittest.TestCase):
    def test_inventory_shape_offsets_metadata_and_layer_classification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.gguf"
            _make_gguf(
                path,
                architecture="qwen35",
                block_count=2,
                extra_metadata=[
                    ("qwen35.full_attention_interval", 4, 2),
                    ("qwen35.nextn_predict_layers", 4, 0),
                    ("qwen35.attention.recurrent_layers", 9, (7, [True, False])),
                    ("qwen35.attention.full_attention_layers", 9, (4, [1])),
                    ("qwen35.layer_types", 9, (8, ["recurrent", "full attention"])),
                ],
                tensors=[
                    ("blk.0.attn_qkv.weight", [256, 2], 21),
                    ("blk.0.ssm_a", [1], 0),
                    ("blk.1.attn_q.weight", [2, 2], 0),
                    ("blk.1.attn_k.weight", [2, 2], 0),
                ],
            )
            manifest = inspect_gguf(path)
            tensor = manifest["tensors"][0]
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(tensor["shape"], [256, 2])
            self.assertEqual(tensor["shape_order"], "ggml_ne_order")
            self.assertEqual(tensor["n_elements"], 512)
            self.assertEqual(tensor["row_bytes"], 110)
            self.assertEqual(tensor["n_bytes"], 220)
            self.assertEqual(tensor["offset"]["absolute"], manifest["gguf"]["data_offset"])
            self.assertEqual(tensor["offset"]["relative"], 0)
            self.assertEqual(manifest["model"]["layer_counts"]["gated_delta_net"], 1)
            self.assertEqual(manifest["model"]["layer_counts"]["full_attention"], 1)
            self.assertTrue(manifest["validation"]["checks"]["nextn_predict_layers_matches_detected_mtp_count"])
            self.assertTrue(manifest["validation"]["checks"]["full_attention_interval_matches_trunk_tensor_inventory"])
            self.assertTrue(manifest["validation"]["checks"][
                "metadata_layer_set_matches_tensor_inventory:qwen35.attention.recurrent_layers"])
            self.assertTrue(manifest["validation"]["checks"][
                "metadata_layer_set_matches_tensor_inventory:qwen35.attention.full_attention_layers"])
            self.assertTrue(manifest["validation"]["checks"][
                "metadata_layer_types_match_tensor_inventory:qwen35.layer_types"])
            token_meta = manifest["metadata"]["tokenizer.ggml.tokens"]["value"]
            self.assertEqual(token_meta["count"], 300)
            self.assertNotIn("values", token_meta)
            self.assertEqual(len(token_meta["sha256"]), 64)
            self.assertEqual(len(read_tensor_payload(path, tensor)), 220)

    def test_rejects_payload_outside_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out-of-bounds.gguf"
            _make_gguf(path, architecture="dflash", block_count=1,
                       tensors=[("blk.0.attn_q.weight", [2, 2], 0)], invalid_offset=True)
            with self.assertRaisesRegex(GGUFError, "exceeds file size"):
                inspect_gguf(path)

    def test_rejects_non_power_of_two_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad-alignment.gguf"
            _make_gguf(path, architecture="qwen35", block_count=0, tensors=[], alignment=24)
            with self.assertRaisesRegex(GGUFError, "positive power of two"):
                inspect_gguf(path)

    def test_dflash_uses_per_layer_sliding_window_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pattern.gguf"
            _make_gguf(
                path,
                architecture="dflash",
                block_count=2,
                extra_metadata=[
                    ("dflash.attention.sliding_window", 4, 128),
                    ("dflash.attention.sliding_window_pattern", 9, (7, [True, False])),
                    ("dflash.block_size", 4, 8),
                ],
                tensors=[
                    ("blk.0.attn_q.weight", [2, 2], 0),
                    ("blk.0.attn_k.weight", [2, 2], 0),
                    ("blk.1.attn_q.weight", [2, 2], 0),
                    ("blk.1.attn_k.weight", [2, 2], 0),
                ],
            )
            manifest = inspect_gguf(path)
            self.assertEqual([layer["kind"] for layer in manifest["model"]["layers"]],
                             ["sliding_window_attention", "full_attention"])
            self.assertTrue(manifest["validation"]["checks"]["sliding_window_pattern_matches_observed_layers"])
            self.assertEqual(manifest["model"]["metadata"]["block_size"], 8)

    def test_links_only_same_name_and_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target_path = Path(tmp) / "target.gguf"
            draft_path = Path(tmp) / "draft.gguf"
            shared = ("fixture_shared.weight", [2, 2], 0)
            embedding = ("token_embd.weight", [2, 300], 0)
            output = ("output.weight", [2, 300], 0)
            _make_gguf(target_path, architecture="qwen35", block_count=0,
                       tensors=[shared, embedding, output])
            _make_gguf(draft_path, architecture="dflash", block_count=0, tensors=[shared])
            links = link_model_manifests(inspect_gguf(target_path), inspect_gguf(draft_path))
            self.assertEqual([item["tensor"] for item in links["same_name_layout_matches_unverified_payload"]],
                             ["fixture_shared.weight"])
            self.assertEqual([item["consumer_tensor"] for item in links["runtime_aliases"]],
                             ["token_embd.weight", "output.weight"])


if __name__ == "__main__":
    unittest.main()
