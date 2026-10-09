"""Run render_cluster.py's command line from a test, the way the documented commands do."""

from __future__ import annotations

import sys

import render_cluster


def run_cli(monkeypatch, capsys, *argv: str) -> str:
    """Run `render_cluster.main()` with this command line.

    Args:
        monkeypatch: pytest's monkeypatch fixture, to set `sys.argv`.
        capsys: pytest's capsys fixture, to collect what the script prints.
        *argv: The arguments after the script's name, `--output` included.

    Returns:
        What the script printed to standard output.
    """
    monkeypatch.setattr(sys, "argv", ["render_cluster.py", *argv])
    render_cluster.main()
    return capsys.readouterr().out
