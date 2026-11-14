# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from crucible.common import process_settlement as settlement


def test_process_settlement_import_is_assay_and_provider_free(tmp_path: Path) -> None:
    source_parent = Path(__file__).resolve().parents[2]
    script = """
import sys

sys.path.insert(0, sys.argv[1])
import crucible.common.process_settlement  # noqa: F401

forbidden = sorted(
    name
    for name in sys.modules
    if name == "assay"
    or name.startswith("assay.")
    or name == "anthropic"
    or name.startswith("anthropic.")
    or name == "openai"
    or name.startswith("openai.")
    or name.startswith("crucible.backends.")
    or name.startswith("crucible.workflows.")
)
if forbidden:
    raise SystemExit("eager process-settlement imports: " + ", ".join(forbidden))
"""
    result = subprocess.run(
        (sys.executable, "-I", "-B", "-c", script, str(source_parent)),
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
