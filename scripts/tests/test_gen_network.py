"""gen_network.py's command line: --help prints and exits without building or writing."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "gen_network.py"


def test_help_prints_the_usage_and_writes_nothing(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--output" in result.stdout and "Stone-Wales" in result.stdout
    assert list(tmp_path.iterdir()) == []
