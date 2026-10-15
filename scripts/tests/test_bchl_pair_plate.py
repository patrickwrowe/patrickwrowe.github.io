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
from cluster_svg import circle_centres, run_cli

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
    "--mark-closest",
    "--radius-scale",
    "2.2",
    f"--rotate={PAIR_ROTATION}",
]


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
    rotation = render_cluster.rotation_matrix(
        tuple(float(degrees) for degrees in PAIR_ROTATION.split(","))
    )
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


def test_a_dotted_ink_line_joins_the_oxygen_and_the_hydrogen_edge_to_edge():
    svg = (FIGURES / "bchl-pair-1-2.svg").read_text()
    (line,) = re.findall(r"<line data-contact-angstrom=[^>]*/>", svg)
    assert 'data-contact-angstrom="2.261"' in line
    assert 'stroke="var(--ink)"' in line and "stroke-dasharray" in line
    start, end = (
        np.array([float(value) for value in pair])
        for pair in re.findall(r'x\d="([-\d.]+)" y\d="([-\d.]+)"', line)
    )
    first = boxprep.read_xyz(DATA / "chromophore-1.xyz")
    second = boxprep.read_xyz(DATA / "chromophore-2.xyz")
    positions_angstrom = np.vstack([first.positions_angstrom, second.positions_angstrom])
    rotation = render_cluster.rotation_matrix(
        tuple(float(degrees) for degrees in PAIR_ROTATION.split(","))
    )
    rotated = (positions_angstrom - positions_angstrom.mean(axis=0)) @ rotation.T
    separations_angstrom = np.linalg.norm(
        first.positions_angstrom[:, None, :] - second.positions_angstrom[None, :, :], axis=-1
    )
    oxygen, hydrogen = np.unravel_index(separations_angstrom.argmin(), separations_angstrom.shape)
    oxygen_xy = rotated[oxygen, :2] * (1, -1)  # SVG y runs down
    hydrogen_xy = rotated[len(first) + hydrogen, :2] * (1, -1)
    # The line runs from the oxygen ring's edge to the hydrogen disc's edge, on the line
    # between their centres, and covers neither atom.
    radius = {
        symbol: render_cluster.ATOM_RADIUS * render_cluster.DRAW_RADIUS_FACTOR[symbol] * 2.2
        for symbol in ("O", "H")
    }
    assert np.linalg.norm(start - oxygen_xy) == pytest.approx(radius["O"], abs=2e-3)
    assert np.linalg.norm(end - hydrogen_xy) == pytest.approx(radius["H"], abs=2e-3)
    assert np.linalg.norm(end - start) == pytest.approx(
        np.linalg.norm(hydrogen_xy - oxygen_xy) - radius["O"] - radius["H"], abs=2e-3
    )


@pytest.mark.parametrize(
    ("names", "extra"),
    [
        (["chromophore-1.xyz", "chromophore-2.xyz"], []),
        (["chromophore-1.xyz", "chromophore-2.xyz", "chromophore-1.xyz"], ["--together"]),
    ],
)
def test_mark_closest_is_refused_without_exactly_two_inputs_drawn_together(
    names, extra, tmp_path, monkeypatch, capsys
):
    argv = [*(str(DATA / name) for name in names), "--mark-closest", *extra]
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, capsys, *argv, "--output", str(tmp_path / "pair.svg"))
    assert "--mark-closest needs --together" in capsys.readouterr().err


def test_a_marked_contact_in_a_slab_is_refused_since_the_slab_can_cut_its_atoms_away():
    first = boxprep.read_xyz(DATA / "chromophore-1.xyz")
    second = boxprep.read_xyz(DATA / "chromophore-2.xyz")
    together = boxprep.Structure(
        first.species + second.species,
        np.vstack([first.positions_angstrom, second.positions_angstrom]),
    )
    contact = render_cluster.closest_pair(first, second)
    assert render_cluster.render(together, (0, 0, 0), contact=contact).count("data-contact") == 1
    with pytest.raises(ValueError, match="slab"):
        render_cluster.render(together, (0, 0, 0), slab_angstrom=10.0, contact=contact)


def test_closest_pair_names_the_closest_atoms_one_from_each_structure():
    first = boxprep.Structure(["C", "O"], np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]))
    second = boxprep.Structure(["H", "C"], np.array([[20.0, 0.0, 0.0], [7.0, 0.0, 0.0]]))
    # O (index 1) and the second's C (index 1, so 2 + 1 once drawn together), 2 A apart.
    assert render_cluster.closest_pair(first, second) == (1, 3)


@pytest.mark.parametrize(
    ("second_hydrogen_angstrom", "drawn"),
    [
        ((0.0, 0.0, 2.5), False),  # along the default view direction: no projected gap at all
        ((0.1, 0.0, 2.5), False),  # projected 0.1 A, inside the two hydrogens' drawn radii
        ((2.5, 0.0, 0.0), True),  # side-on: the whole 2.5 A is drawn as a gap
    ],
)
def test_a_marked_contact_inside_its_atoms_drawn_radii_is_refused_and_a_side_on_one_is_drawn(
    second_hydrogen_angstrom, drawn, tmp_path, monkeypatch, capsys
):
    # Two 2-atom files, an O-H pair each, whose closest contact is between the two hydrogens;
    # the second oxygen sits 1 A beyond its hydrogen, away from the first molecule.
    x, y, z = second_hydrogen_angstrom
    oxygen_x, oxygen_z = (x + 1.0, z) if drawn else (x, z + 1.0)
    first, second = tmp_path / "first.xyz", tmp_path / "second.xyz"
    first.write_text("2\n\nO -1.0 0.0 0.0\nH 0.0 0.0 0.0\n")
    second.write_text(f"2\n\nH {x} {y} {z}\nO {oxygen_x} {y} {oxygen_z}\n")
    output = tmp_path / "pair.svg"
    argv = [str(first), str(second), "--together", "--mark-closest", "--output", str(output)]
    if not drawn:
        with pytest.raises(ValueError, match="rotate it side-on"):
            run_cli(monkeypatch, capsys, *argv)
        assert not output.exists()
        return
    run_cli(monkeypatch, capsys, *argv)
    (line,) = re.findall(r"<line data-contact-angstrom=[^>]*/>", output.read_text())
    assert 'data-contact-angstrom="2.500"' in line
    start_x, start_y, end_x, end_y = (
        float(value) for value in re.findall(r'[xy]\d="([-\d.]+)"', line)
    )
    assert np.isfinite([start_x, start_y, end_x, end_y]).all()
    # Edge to edge, from the first hydrogen towards the second, not reversed over them.
    hydrogen_radius = render_cluster.ATOM_RADIUS * render_cluster.DRAW_RADIUS_FACTOR["H"]
    assert end_x - start_x == pytest.approx(2.5 - 2 * hydrogen_radius, abs=2e-3)
