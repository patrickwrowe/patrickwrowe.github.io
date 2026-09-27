"""Tests for the two cluster-series figure scripts, on the real census.csv."""

from __future__ import annotations

import re
from pathlib import Path

import cluster_outcomes
import cluster_sp3
import pytest

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


def test_a_cage_glyph_is_a_hexagon_centred_on_its_run():
    # C60 at 2000 K is a cage in census.csv: its glyph is a six-cornered polygon whose
    # corners average to the run's (size, temperature) position.
    svg = cluster_outcomes.build(SPHERES)
    match = re.search(r'<g data-run="C60-2000K" data-class="cage"><polygon points="([^"]+)"', svg)
    assert match, "C60-2000K is not drawn as a cage polygon"
    corners = [tuple(map(float, corner.split(","))) for corner in match.group(1).split()]
    assert len(corners) == 6
    centre_x = sum(corner[0] for corner in corners) / 6
    centre_y = sum(corner[1] for corner in corners) / 6
    assert centre_x == pytest.approx(cluster_outcomes.size_to_x(60), abs=0.01)
    assert centre_y == pytest.approx(cluster_outcomes.temperature_to_y(2000), abs=0.01)


def test_an_sp3_point_sits_at_the_height_its_fraction_maps_to():
    # C1000 at 1000 K has the highest sp3 fraction, 0.112 in census.csv.
    svg = cluster_sp3.build(SPHERES)
    match = re.search(r'<circle data-run="C1000-1000K" cx="([\d.]+)" cy="([\d.]+)"', svg)
    assert match, "no marker for C1000-1000K"
    assert float(match.group(1)) == pytest.approx(cluster_sp3.temperature_to_x(1000), abs=0.01)
    assert float(match.group(2)) == pytest.approx(cluster_sp3.percent_to_y(11.2), abs=0.01)
