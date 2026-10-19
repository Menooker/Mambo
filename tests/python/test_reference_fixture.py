"""Check IQ3_S matmul references from gguf-py and the ggml CPU subprocess."""

from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
import pytest


from utils.compute_reference_gguf import compute_reference_gguf
from utils.paths import model_reference_config


WEIGHT = "blk.1.attn_qkv.weight"


@pytest.fixture(scope="module")
def reference_config() -> tuple[Path, Path, Path]:
    return model_reference_config()


def test_iq3_s_projection_reference_m1_and_m8(reference_config: tuple[Path, Path, Path]) -> None:
    model, executable, llama_source = reference_config
    x_all = np.random.Generator(np.random.PCG64(20261013)).standard_normal((8, 5120), dtype=np.float32)
    for m in (1, 8):
        result = compute_reference_gguf(
            "matmul", {"x": x_all[:m], "weight": WEIGHT}, model_path=model,
            reference_executable=executable, llama_source_dir=llama_source,
        )
        assert isinstance(result.gguf_py, np.ndarray)
        assert isinstance(result.ggml_cpu, np.ndarray)
        assert result.gguf_py.shape == (m, 10240)
        assert result.ggml_cpu.shape == (m, 10240)
        assert result.dequant_rows.shape == (4, 5120)
        assert result.metadata["ggml_cpu_input_dot_type"] == "q8_K"
        np.testing.assert_allclose(result.ggml_cpu, result.gguf_py, rtol=0.02, atol=0.05)


def test_rejects_truncated_stdin(reference_config: tuple[Path, Path, Path]) -> None:
    model, executable, _ = reference_config
    run = subprocess.run(
        [str(executable), "--op", "matmul", "--model", str(model),
         "--weight", WEIGHT, "--m", "1"],
        input=b"\0", capture_output=True, check=False,
    )
    assert run.returncode != 0
    assert b"input size mismatch" in run.stderr
    assert not run.stdout
