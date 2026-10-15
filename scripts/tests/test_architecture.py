"""Placement rules the 2026-10-15 architecture review asked to keep true, read off the source.

Each rule names one home for one piece of knowledge, so a second copy fails here rather than
drifting quietly: the colour rule of a Blender panel, the bond rule's tolerance, and where the
tests find the papers they check against.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
SOURCES = sorted(SCRIPTS.rglob("*.py"))


def test_the_colour_rule_of_a_panel_is_stated_once_in_panel_checks_is_coloured():
    rule = re.compile(r"len\(set\(species\)\) > 1")
    hits = [
        f"{path.relative_to(SCRIPTS)}:{number}"
        for path in SOURCES
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        if rule.search(line)
    ]
    assert len(hits) == 1 and hits[0].startswith("figures/panel_checks.py:"), hits


def test_the_bond_tolerance_is_imported_by_boxprep_alone():
    importers = sorted(
        str(path.relative_to(SCRIPTS))
        for path in SOURCES
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.ImportFrom)
        and any(alias.name == "BOND_TOLERANCE" for alias in node.names)
    )
    assert importers == ["figures/boxprep.py"]


def test_no_test_reaches_outside_the_repository_through_its_parent():
    parent = re.compile(r"REPO\.parent\b")
    offenders = [
        str(path.relative_to(SCRIPTS))
        for path in sorted((SCRIPTS / "tests").glob("*.py"))
        if parent.search(path.read_text())
    ]
    assert offenders == []
