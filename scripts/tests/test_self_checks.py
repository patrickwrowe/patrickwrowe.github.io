"""Every figure module's in-module `--self-check` suite, run as its command line runs it."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

FIGURES = Path(__file__).resolve().parents[1] / "figures"


@pytest.mark.parametrize("module", ["render_cluster", "combustion_frames", "graphitisation_figure"])
def test_the_modules_self_check_passes(module):
    result = subprocess.run(
        [sys.executable, str(FIGURES / f"{module}.py"), "--self-check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("self-check ok")
