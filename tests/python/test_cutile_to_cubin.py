"""Test cuTile JIT, AOT export, and C++ loading of the exported probe."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Make direct execution work from CMake and from a source checkout without an
# installed mambo package.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_SOURCE = _PROJECT_ROOT / "src" / "python"
if str(_PYTHON_SOURCE) not in sys.path:
    sys.path.insert(0, str(_PYTHON_SOURCE))

import cupy as cp
import cuda.tile as ct
from cuda.tile.compilation import ArrayConstraint

from mambo.utils.cutile_to_cubin import cutile_to_cubin, write_abi_json


PROBE_LENGTH = 1003
PROBE_TILE_SIZE = 256
PROBE_SYMBOL = "vector_add"


@ct.kernel
def vector_add(a, b, out, tile_size: ct.Constant[int]):
    # cuTile scales these tile coordinates by the tile shape.
    index = ct.bid(0)
    a_tile = ct.load(a, index=(index,), shape=(tile_size,), padding_mode=ct.PaddingMode.ZERO)
    b_tile = ct.load(b, index=(index,), shape=(tile_size,), padding_mode=ct.PaddingMode.ZERO)
    ct.store(out, index=(index,), tile=a_tile + b_tile)


def _array_constraint():
    return ArrayConstraint(
        ct.float32,
        1,
        index_dtype=ct.int32,
        stride_lower_bound_incl=1,
        alias_groups=(),
        may_alias_internally=False,
        stride_constant=(1,),
        base_addr_divisible_by=1,
    )


def _export_probe(cubin_path: Path, abi_path: Path, gpu_code: str) -> dict:
    """Compile the test kernel and write the ABI consumed by C++ loadtest."""
    grid = ((PROBE_LENGTH + PROBE_TILE_SIZE - 1) // PROBE_TILE_SIZE, 1, 1)
    info = cutile_to_cubin(
        vector_add,
        {
            "parameters": {name: _array_constraint() for name in ("a", "b", "out")},
            "constants": {"tile_size": PROBE_TILE_SIZE},
            "gpu_code": gpu_code,
            "symbol": PROBE_SYMBOL,
            "grid": grid,
            "block": (1, 1, 1),
        },
        cubin_path,
    )
    document = write_abi_json(info, abi_path)
    document["probe"] = {
        "operation": "vector_add",
        "inputs": ["a", "b"],
        "output": "out",
        "length": PROBE_LENGTH,
    }
    abi_path = Path(abi_path)
    abi_path.parent.mkdir(parents=True, exist_ok=True)
    abi_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document


class CutileToCubinTest(unittest.TestCase):
    def test_vector_add_jit_and_export(self):
        n = PROBE_LENGTH
        tile_size = PROBE_TILE_SIZE
        a = cp.arange(n, dtype=cp.float32)
        b = cp.arange(n, dtype=cp.float32) * 0.5
        out = cp.empty_like(a)
        grid = ((n + tile_size - 1) // tile_size, 1, 1)
        ct.launch(cp.cuda.get_current_stream(), grid, vector_add, (a, b, out, tile_size))
        cp.testing.assert_array_equal(out, a + b)

        compute_capability = cp.cuda.Device().compute_capability
        with tempfile.TemporaryDirectory() as tmp:
            cubin_path = Path(tmp) / "vector_add.cubin"
            abi_path = Path(tmp) / "vector_add.abi.json"
            document = _export_probe(cubin_path, abi_path, f"sm_{compute_capability}")

            self.assertTrue(cubin_path.is_file())
            self.assertGreater(cubin_path.stat().st_size, 0)
            self.assertEqual(document["name"], PROBE_SYMBOL)
            self.assertEqual(len(document["parameters"]), 9)
            self.assertEqual(document["parameters"][0], {"name": "a", "type": "float32*"})
            self.assertEqual(document["parameters"][1], {"name": "a_shape_0", "type": "int32"})
            self.assertEqual(document["constraints"][0]["name"], "a")
            self.assertEqual(document["constraints"][0]["shape_divisible_by"], [1])
            self.assertEqual(document["constraints"][0]["stride_lower_bound_incl"], [None])
            self.assertEqual(document["schema_version"], 1)
            self.assertEqual(document["symbol"], PROBE_SYMBOL)
            self.assertEqual(document["launch"]["grid"], list(grid))
            self.assertIn("cuda_tile", document["toolchain"])
            self.assertEqual(
                document["probe"],
                {"operation": "vector_add", "inputs": ["a", "b"], "output": "out", "length": n},
            )
            self.assertNotIn("constants", document)
            self.assertTrue(abi_path.is_file())
            self.assertEqual(json.loads(abi_path.read_text(encoding="utf-8")), document)
            loadtest = os.environ.get("MAMBO_LOADTEST_EXECUTABLE")
            if loadtest:
                self.assertTrue(Path(loadtest).is_file(), loadtest)
                result = subprocess.run(
                    [loadtest, "--abi-json", str(abi_path), "--n", str(n)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"passed for {n} elements", result.stdout)
