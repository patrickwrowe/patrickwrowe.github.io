"""Tests for the two cluster-series figure scripts, on the real census.csv."""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))
import cluster_outcomes  # noqa: E402
import cluster_sp3  # noqa: E402

SPHERES = Path(__file__).resolve().parents[1] / "figures" / "data" / "carbon-clusters" / "spheres"
HEX_COLOUR = re.compile(r"#[0-9a-fA-F]{3,8}\b")
ALLOWED_PAINT = {"var(--ink)", "var(--graphite)", "var(--plate)", "none"}


def paints(svg: str) -> set[str]:
    """Every fill and stroke value in an SVG document.

    Args:
        svg: The SVG document.

    Returns:
        The distinct values of all fill= and stroke= attributes.
    """
    return set(re.findall(r'(?:fill|stroke)="([^"]*)"', svg))


def test_outcome_diagram_has_one_glyph_per_run_and_no_hue():
    svg = cluster_outcomes.build(SPHERES)
    classes = re.findall(r'data-run="C\d+-\d+K" data-class="([^"]+)"', svg)
    assert len(classes) == 48
    assert set(classes) <= set(cluster_outcomes.CLASSES)
    assert len(re.findall(r"data-legend=", svg)) == len(cluster_outcomes.CLASSES) == 6
    assert classes.count("cage") == 6 and classes.count("graphitic onion") == 8
    assert not HEX_COLOUR.search(svg)
    assert paints(svg) <= ALLOWED_PAINT


def test_sp3_plot_has_one_line_per_size_and_no_hue():
    svg = cluster_sp3.build(SPHERES)
    sizes = re.findall(r'<polyline data-size="(\d+)"', svg)
    assert sizes == [str(n_atoms) for n_atoms in cluster_outcomes.SIZES]
    assert len(re.findall(r"<circle ", svg)) == 48
    assert not HEX_COLOUR.search(svg)
    assert paints(svg) <= ALLOWED_PAINT
