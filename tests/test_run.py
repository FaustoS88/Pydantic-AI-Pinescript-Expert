"""Tests for run.py — the command-line entry point."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

RUN_PY = Path(__file__).resolve().parent.parent / "run.py"


class TestHelp:
    def test_help_prints_usage_and_exits_zero(self, tmp_path: Path) -> None:
        # a lone copy: a --help that reached the query would fail on import, not call the model
        shutil.copy(RUN_PY, tmp_path)
        for args in (["--help"], ["-h"], ["query", "--help"]):
            result = subprocess.run(
                [sys.executable, str(tmp_path / "run.py"), *args], capture_output=True, text=True
            )
            assert result.returncode == 0, result.stderr
            assert result.stdout.startswith("Usage:")
            assert "Unknown command" not in result.stdout
