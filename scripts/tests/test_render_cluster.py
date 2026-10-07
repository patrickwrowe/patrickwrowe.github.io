"""Tests for render_cluster.py: the CHO combustion series and the species palette."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import boxprep
import numpy as np
import pytest
import render_cluster
from constants import SPECIES_COLOUR

REPO = Path(__file__).resolve().parents[2]
CHO_DATA = REPO / "scripts" / "figures" / "data" / "cho-gap"
CHO_FIGURES = REPO / "src" / "content" / "work" / "figures" / "cho-gap"
CARBON_DATA = REPO / "scripts" / "figures" / "data" / "carbon-clusters"
TOKENS = (REPO / "src" / "styles" / "tokens.css").read_text()
HEX_COLOUR = re.compile(r"#[0-9a-fA-F]{3,8}\b")
SERIES_FRAMES = [CHO_DATA / f"combustion-{time_ps}ps.xyz" for time_ps in (0, 10, 20, 50)]


def paints(svg: str) -> set[str]:
    """Every fill and stroke value in an SVG document."""
    return set(re.findall(r'(?:fill|stroke)="([^"]*)"', svg))


@pytest.mark.parametrize(
    ("committed", "extra_args"),
    [
        ("combustion-series.svg", []),
        ("combustion-series-narrow.svg", ["--narrow"]),
    ],
)
def test_the_committed_combustion_series_is_byte_for_byte_what_the_documented_command_writes(
    committed, extra_args, tmp_path, monkeypatch, capsys
):
    # The command in render_cluster.py's module docstring, run through the real CLI.
    written = tmp_path / committed
    argv = [
        "render_cluster.py",
        *map(str, SERIES_FRAMES),
        "--labels",
        "0 ps,10 ps,20 ps,50 ps",
        "--columns",
        "2",
        "--radius-scale",
        "2.2",
        *extra_args,
        "--output",
        str(written),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    render_cluster.main()
    capsys.readouterr()
    assert (CHO_FIGURES / committed).read_bytes() == written.read_bytes(), f"{committed} is stale"


@pytest.mark.parametrize("committed", ["combustion-series.svg", "combustion-series-narrow.svg"])
def test_the_combustion_series_paints_each_species_from_a_token_that_exists(committed):
    svg = (CHO_FIGURES / committed).read_text()
    species_tokens = set(re.findall(r"var\(--species-([a-z]+)\)", svg))
    assert species_tokens == {"c", "h", "o"}
    for symbol in species_tokens:
        assert f"--species-{symbol}:" in TOKENS
    assert not HEX_COLOUR.search(svg)
    assert "--lustre" not in svg  # the accent is state and anomaly, never a species
    assert paints(svg) <= {
        "var(--ink)",  # the group default
        "var(--graphite)",  # the labels
        "none",
        *(f"var(--species-{symbol})" for symbol in species_tokens),
    }


@pytest.mark.parametrize("name", ["C60-Ih.xyz", "C240-0.xyz"])
def test_an_all_carbon_cluster_draws_in_ink_with_no_species_colour(name):
    svg = render_cluster.render(boxprep.read_xyz(CARBON_DATA / name), (15, 25, 0))
    assert "--species" not in svg
    assert paints(svg) == {"var(--ink)", "none"}


def test_a_series_paints_carbon_the_same_in_every_panel_once_any_panel_has_a_second_species():
    # One C-H bond and one C-C bond, in angstrom.
    methane = boxprep.Structure(["C", "H"], np.array([[0.0, 0.0, 0.0], [1.09, 0.0, 0.0]]))
    graphite_bit = boxprep.Structure(["C", "C"], np.array([[0.0, 0.0, 0.0], [1.42, 0.0, 0.0]]))
    alone = render_cluster.render_series([("C2", graphite_bit)], (0, 0, 0))
    beside = render_cluster.render_series([("CH", methane), ("C2", graphite_bit)], (0, 0, 0))
    assert "--species" not in alone
    # The figure has hydrogen, so all three carbons are species-c, including the panel
    # that has no hydrogen of its own.
    assert beside.count('fill="var(--species-c)"') == 3
    assert beside.count("--species-h") == 2  # the hydrogen's disc and its half of the bond


def test_a_species_with_no_palette_entry_is_refused_by_name_not_painted_black(monkeypatch):
    monkeypatch.delitem(SPECIES_COLOUR, "O")
    oxygen_and_carbon = boxprep.Structure(["C", "O"], np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]))
    with pytest.raises(ValueError, match=r"\['O'\]"):
        render_cluster.render(oxygen_and_carbon, (0, 0, 0))
