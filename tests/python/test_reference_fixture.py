"""Check IQ3_S matmul references from gguf-py and the ggml CPU subprocess."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.compute_reference_gguf import compute_reference_gguf  # noqa: E402


WEIGHT = "blk.1.attn_qkv.weight"


def _model_from_test_config() -> Path:
    config_path = REPO_ROOT / "tests/config.json"
    if not config_path.is_file():
        pytest.skip("copy tests/config.json.template to tests/config.json and set qwen_gguf_path")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    raw_path = config.get("qwen_gguf_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{config_path}: qwen_gguf_path must be a nonempty string")
    model = Path(raw_path).expanduser()
    if not model.is_absolute():
        model = config_path.resolve().parent / model
    if not model.is_file():
        raise FileNotFoundError(f"{config_path}: qwen_gguf_path does not exist: {model}")
    return model


@pytest.fixture(scope="module")
def reference_config() -> tuple[Path, Path, Path]:
    executable = Path(os.environ.get("MAMBO_REFERENCE_EXECUTABLE", ""))
    if not executable.is_file():
        pytest.skip("build the ggml reference tool and set MAMBO_REFERENCE_EXECUTABLE")
    model = _model_from_test_config()
    llama_source = Path(os.environ.get("MAMBO_LLAMA_SOURCE_DIR", str(REPO_ROOT / "thirdparty/llama.cpp")))
    if not (llama_source / "gguf-py/gguf/quants.py").is_file():
        pytest.skip("initialize the pinned llama.cpp submodule")
    return model, executable, llama_source


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
