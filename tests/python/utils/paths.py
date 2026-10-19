"""Repository paths and shared configuration for Python integration tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
TEST_PYTHON = Path(__file__).resolve().parents[1]
LLAMA_SOURCE = REPO_ROOT / "thirdparty" / "llama.cpp"


def ensure_test_paths() -> None:
    for path in (REPO_ROOT / "src" / "python", TEST_PYTHON):
        source = str(path)
        if source not in sys.path:
            sys.path.insert(0, source)


def model_from_test_config() -> Path:
    config_path = REPO_ROOT / "tests" / "config.json"
    if not config_path.is_file():
        pytest.skip("copy tests/config.json.template to tests/config.json and set qwen_gguf_path")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    value = config.get("qwen_gguf_path")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{config_path}: qwen_gguf_path must be a nonempty string")
    model_path = Path(value).expanduser()
    if not model_path.is_absolute():
        model_path = config_path.resolve().parent / model_path
    if not model_path.is_file():
        raise FileNotFoundError(f"{config_path}: qwen_gguf_path does not exist: {model_path}")
    return model_path


def model_reference_config() -> tuple[Path, Path, Path]:
    model_path = model_from_test_config()
    executable = Path(os.environ.get("MAMBO_REFERENCE_EXECUTABLE", "")).expanduser()
    if not executable.is_file():
        pytest.skip("build the ggml reference tool and set MAMBO_REFERENCE_EXECUTABLE")
    if not (LLAMA_SOURCE / "gguf-py" / "gguf" / "quants.py").is_file():
        pytest.skip("initialize the pinned llama.cpp submodule")
    return model_path, executable, LLAMA_SOURCE
