"""Plate E1 of the excited-states article, the FMO bacteriochlorophyll pair 1-2.

The pair is drawn by render_cluster.py with `--together`: both chromophores in the frame they
were superposed in, hydrogens kept because the closest contact is an oxygen and a hydrogen. The
plate is regenerated through the real command line, compared byte for byte with the committed
SVG, and checked against the geometry it draws: the numbers a caption could copy are recomputed
from the staged input files, not from the figure. Nothing here reads docs/, which is
gitignored.
"""

from __future__ import annotations

import re
from pathlib import Path

import boxprep
import numpy as np
import pytest
import render_cluster
from cluster_svg import run_cli

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "scripts/figures/data/excited-states"
FIGURES = REPO / "src/content/work/figures/excited-states"
HEX_COLOUR = re.compile(r"#[0-9a-fA-F]{3}\b|#[0-9a-fA-F]{6}\b")

# The commands of render_cluster.py's module docstring.
PAIR_ROTATION = "-137.41,-4.66,-122.33"
PAIR_ARGV = [
    str(DATA / "chromophore-1.xyz"),
    str(DATA / "chromophore-2.xyz"),
    "--together",
    "--radius-scale",
    "2.2",
    f"--rotate={PAIR_ROTATION}",
]


def circle_centres(svg: str) -> np.ndarray:
    """Every drawn disc and ring: (cx, -cy) in angstrom, in the figure's own frame (y up)."""
    return np.array(
        [
            [float(x), -float(y)]
            for x, y in re.findall(r'<circle cx="([-\d.]+)" cy="([-\d.]+)"', svg)
        ]
    )


def test_the_committed_pair_plate_is_byte_for_byte_what_the_documented_command_writes(
    tmp_path, monkeypatch, capsys
):
    written = tmp_path / "bchl-pair-1-2.svg"
    run_cli(monkeypatch, capsys, *PAIR_ARGV, "--output", str(written))
    svg = written.read_text()
    assert (FIGURES / "bchl-pair-1-2.svg").read_text() == svg, "bchl-pair-1-2.svg is stale"


def test_the_pair_plate_draws_both_chromophores_whole_in_the_palette_and_no_hue_outside_it():
    svg = (FIGURES / "bchl-pair-1-2.svg").read_text()
    # 152 atoms, hydrogens included: the closest contact is an oxygen and a hydrogen.
    assert len(circle_centres(svg)) == 152
    assert svg.count('fill="var(--species-mg)"') == 2
    assert {*re.findall(r"var\(--species-([a-z]+)\)", svg)} == {"c", "h", "mg", "n", "o"}
    assert not HEX_COLOUR.search(svg)
    assert "--lustre" not in svg


def test_the_closest_contact_of_the_pair_is_2_26_angstrom_an_oxygen_and_a_hydrogen():
    first = boxprep.read_xyz(DATA / "chromophore-1.xyz")
    second = boxprep.read_xyz(DATA / "chromophore-2.xyz")
    separations_angstrom = np.linalg.norm(
        first.positions_angstrom[:, None, :] - second.positions_angstrom[None, :, :], axis=-1
    )
    closest = np.unravel_index(separations_angstrom.argmin(), separations_angstrom.shape)
    assert separations_angstrom.min() == pytest.approx(2.2613, abs=5e-5)
    assert (first.species[closest[0]], second.species[closest[1]]) == ("O", "H")
    # Without the hydrogens the pair keeps its distance: 3.21 A between heavy atoms.
    heavy_first = np.array(first.species) != "H"
    heavy_second = np.array(second.species) != "H"
    heavy_angstrom = separations_angstrom[np.ix_(heavy_first, heavy_second)].min()
    assert heavy_angstrom == pytest.approx(3.2059, abs=5e-4)


def test_the_view_has_the_closest_contact_side_on_so_the_2_26_angstrom_is_drawn_as_a_gap():
    svg = (FIGURES / "bchl-pair-1-2.svg").read_text()
    first = boxprep.read_xyz(DATA / "chromophore-1.xyz")
    second = boxprep.read_xyz(DATA / "chromophore-2.xyz")
    positions_angstrom = np.vstack([first.positions_angstrom, second.positions_angstrom])
    separations_angstrom = np.linalg.norm(
        first.positions_angstrom[:, None, :] - second.positions_angstrom[None, :, :], axis=-1
    )
    oxygen, hydrogen = np.unravel_index(separations_angstrom.argmin(), separations_angstrom.shape)
    rotation = render_cluster.rotation_matrix(tuple(float(v) for v in PAIR_ROTATION.split(",")))
    rotated = (positions_angstrom - positions_angstrom.mean(axis=0)) @ rotation.T
    contact = rotated[len(first) + hydrogen] - rotated[oxygen]
    # In the picture's plane (x, y) and not along the view (z): the whole distance is drawn.
    assert abs(contact[2]) < 0.02
    assert np.hypot(contact[0], contact[1]) == pytest.approx(separations_angstrom.min(), abs=0.01)
    # And the two atoms are on the page, 2.26 A apart, as discs or rings at those positions.
    drawn = circle_centres(svg)
    for atom in (rotated[oxygen], rotated[len(first) + hydrogen]):
        assert np.linalg.norm(drawn - atom[:2], axis=1).min() < 1e-3
    # Both macrocycles are seen nearly face-on: the view is within 40 degrees of each one's
    # normal, and the two normals are 50 degrees apart.
    view = np.linalg.inv(rotation)[:, 2]
    for chromophore in (first, second):
        core = chromophore.positions_angstrom[np.isin(chromophore.species, ["N", "Mg"])]
        normal = np.linalg.svd(core - core.mean(axis=0))[2][2]
        assert abs(view @ normal) > np.cos(np.radians(40))
