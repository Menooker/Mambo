"""Check that the C++ ABI loader rejects an unsupported schema version."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


def test_rejects_unsupported_schema() -> None:
    loadtest = os.environ.get("MAMBO_LOADTEST_EXECUTABLE")
    if not loadtest:
        pytest.skip("MAMBO_LOADTEST_EXECUTABLE is not set; build the C++ loadtest first")
    executable = Path(loadtest)
    assert executable.is_file(), f"loadtest executable is missing: {executable}"
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "invalid_schema.abi.json"
    result = subprocess.run(
        [str(executable), "--validate-abi-json", str(fixture)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0, "loadtest accepted unsupported ABI schema version"
    assert "schema_version 1 is required" in result.stderr, result.stderr
