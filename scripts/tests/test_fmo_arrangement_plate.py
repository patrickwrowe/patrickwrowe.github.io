"""Plate M1 of the excited-states article, the natural FMO arrangement beside a generated one.

Two panels of one render_cluster.py series, hydrogens hidden, in one view and at one scale,
both centred on the natural structure's mean (`--shared-centre`) so that the two fixed
chromophores, BChl 1 and 3, sit at the same place in both. The plate is regenerated through
the real command line and compared byte for byte with the committed SVG; the structures it
draws are checked from the staged files. Nothing here reads docs/, which is gitignored.
"""

from __future__ import annotations

import re
from pathlib import Path

import boxprep
import numpy as np
import pytest
from cluster_svg import circle_centres, panel_centres, run_cli

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "scripts/figures/data/excited-states"
FIGURES = REPO / "src/content/work/figures/excited-states"
HEX_COLOUR = re.compile(r"#[0-9a-fA-F]{3}\b|#[0-9a-fA-F]{6}\b")

ARRANGEMENT_ROTATION = "39.92,21.6,162.07"
ARRANGEMENT_ARGV = [
    str(DATA / "fmo-natural.xyz"),
    str(DATA / "fmo-generated-00300.xyz"),
    "--labels",
    "natural,generated",
    "--hide-hydrogens",
    "--radius-scale",
    "2.0",
    "--shared-centre",
    f"--rotate={ARRANGEMENT_ROTATION}",
]


def test_the_committed_arrangement_plate_is_byte_for_byte_what_the_documented_command_writes(
    tmp_path, monkeypatch, capsys
):
    written = tmp_path / "fmo-natural-vs-generated.svg"
    run_cli(monkeypatch, capsys, *ARRANGEMENT_ARGV, "--output", str(written))
    svg = written.read_text()
    assert (FIGURES / "fmo-natural-vs-generated.svg").read_text() == svg, "plate M1 is stale"


@pytest.mark.parametrize("name", ["fmo-natural.xyz", "fmo-generated-00300.xyz"])
def test_both_arrangements_are_608_atoms_with_eight_magnesium_one_per_chromophore(name):
    structure = boxprep.read_xyz(DATA / name)
    assert len(structure) == 608
    assert structure.species.count("Mg") == 8
    assert structure.species.count("H") == 272


def test_the_arrangement_plate_draws_two_panels_in_one_view_of_336_heavy_atoms_each():
    svg = (FIGURES / "fmo-natural-vs-generated.svg").read_text()
    assert svg.count("<svg") == 1  # one viewBox: one scale and one view for both
    assert svg.count("<circle") == 2 * 336  # hydrogens hidden: 608 - 272
    assert "--species-h" not in svg
    assert svg.count('fill="var(--species-mg)"') == 2 * 8
    assert re.findall(r">(natural|generated)</text>", svg) == ["natural", "generated"]
    assert not HEX_COLOUR.search(svg)
    assert "--lustre" not in svg
    # One data-chart-em, the wrapper's --chart-em: the viewBox width over the label size.
    width = float(re.search(r'viewBox="[-\d.]+ [-\d.]+ ([\d.]+) ', svg).group(1))
    size = float(re.search(r'font-size="([\d.]+)"', svg).group(1))
    assert float(re.search(r'data-chart-em="([\d.]+)"', svg).group(1)) == pytest.approx(
        width / size, abs=0.05
    )


def test_the_generated_structure_moves_six_chromophores_and_keeps_source_and_drain():
    natural = boxprep.read_xyz(DATA / "fmo-natural.xyz")
    generated = boxprep.read_xyz(DATA / "fmo-generated-00300.xyz")
    magnesium = np.array(natural.species) == "Mg"
    shift_angstrom = np.linalg.norm(
        natural.positions_angstrom[magnesium] - generated.positions_angstrom[magnesium], axis=1
    )
    assert (shift_angstrom < 1e-6).sum() == 2  # BChl 1 and 3, the fixed source and drain
    assert (shift_angstrom > 1.0).sum() == 6


def test_the_two_fixed_chromophores_are_drawn_at_the_same_place_in_both_panels_and_no_other():
    natural = boxprep.read_xyz(DATA / "fmo-natural.xyz")
    generated = boxprep.read_xyz(DATA / "fmo-generated-00300.xyz")
    magnesium = np.array(natural.species) == "Mg"
    unmoved = (
        np.linalg.norm(
            natural.positions_angstrom[magnesium] - generated.positions_angstrom[magnesium], axis=1
        )
        < 1e-6
    )
    svg = (FIGURES / "fmo-natural-vs-generated.svg").read_text()
    left_centre, right_centre = panel_centres(svg)
    # The eight magnesium of the natural panel come first in the drawing, then the generated.
    drawn = circle_centres(svg, "Mg")
    natural_drawn = drawn[:8] - [left_centre, 0.0]
    generated_drawn = drawn[8:] - [right_centre, 0.0]
    nearest_angstrom = np.linalg.norm(
        natural_drawn[:, None, :] - generated_drawn[None, :, :], axis=-1
    ).min(axis=1)
    registered = nearest_angstrom < 0.01
    assert registered.sum() == unmoved.sum() == 2
    assert nearest_angstrom[~registered].min() > 0.5  # the six that move are well apart
