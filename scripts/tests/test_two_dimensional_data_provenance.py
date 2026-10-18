"""The data of the two-dimensional-and-confined figures against their provenance.json."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import boxprep
import numpy as np
import panel_checks
import pytest
import two_dimensional_frames

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "scripts/figures/data/two-dimensional-and-confined"
PROVENANCE = json.loads((DATA / "provenance.json").read_text())
REQUIRED = {"what", "source_path", "md5"}
CHECKSUMMED = sorted(name for name, entry in PROVENANCE["files"].items() if "source_md5" in entry)


def md5_of(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def test_every_data_file_has_an_entry_and_every_entry_a_file():
    # The manifest is the renders' input, not a staged file, and has no entry.
    on_disk = {path.name for path in DATA.iterdir()} - {"provenance.json", "manifest.json"}
    assert on_disk == set(PROVENANCE["files"])


@pytest.mark.parametrize("name", sorted(PROVENANCE["files"]))
def test_an_entry_says_what_the_file_is_and_where_it_came_from_and_its_md5_is_the_files(name):
    entry = PROVENANCE["files"][name]
    assert REQUIRED <= set(entry), f"{name}: missing {REQUIRED - set(entry)}"
    assert md5_of(DATA / name) == entry["md5"]


@pytest.mark.parametrize("name", CHECKSUMMED)
def test_a_source_on_this_machine_is_the_one_the_entry_checksums(name):
    entry = PROVENANCE["files"][name]
    source = REPO / entry["source_path"]
    if not source.exists():
        pytest.skip("the source is not on this machine")
    assert md5_of(source) == entry["source_md5"]
    if entry.get("verbatim_copy"):
        assert entry["md5"] == entry["source_md5"], f"{name} is meant to be a verbatim copy"


@pytest.mark.parametrize("name", sorted(PROVENANCE["rejected"]))
def test_a_rejected_file_on_this_machine_is_the_one_the_entry_checksums(name):
    entry = PROVENANCE["rejected"][name]
    assert {"source_path", "md5", "why"} <= set(entry)
    source = REPO / entry["source_path"]
    if not source.exists():
        pytest.skip("the rejected file is not on this machine")
    assert md5_of(source) == entry["md5"]
    assert name not in PROVENANCE["files"]


# One 2 x 1 cell of the Li-vermiculite run: the 2:1 layer, its four Li and 40 D2O, the
# deuterium written H (88 H = 80 of the waters + 8 hydroxyls; 88 O = 40 + 48 of the layer).
VERMICULITE_CELL = {"Al": 4, "H": 88, "Li": 4, "Mg": 12, "O": 88, "Si": 12}
VERMICULITE_A_ANGSTROM = 10.62


def test_the_vermiculite_frame_is_three_whole_images_of_one_cell_of_forty_waters():
    frame = boxprep.read_xyz(DATA / two_dimensional_frames.VERMICULITE_OUT)
    n_cell = sum(VERMICULITE_CELL.values())
    assert Counter(frame.species) == {species: 3 * n for species, n in VERMICULITE_CELL.items()}
    images = frame.positions_angstrom.reshape(3, n_cell, 3)
    step = np.array([VERMICULITE_A_ANGSTROM, 0.0, 0.0])
    assert np.allclose(np.diff(images, axis=0), step, atol=2e-6)
    species = np.array(frame.species[:n_cell])
    partners = {atom: set() for atom in range(n_cell)}
    for first, second in boxprep.find_bonds(images[0], species.tolist()):
        partners[first].add(second)
        partners[second].add(first)
    # Every hydrogen sits on exactly one oxygen, so the in-plane wrap cut no molecule...
    assert all(
        len(partners[atom]) == 1 and species[next(iter(partners[atom]))] == "O"
        for atom in np.flatnonzero(species == "H")
    )
    # ... and the waters are the 40 oxygens bonded to two hydrogens and nothing else.
    waters = [
        atom
        for atom in np.flatnonzero(species == "O")
        if sorted(species[list(partners[atom])]) == ["H", "H"]
    ]
    assert len(waters) == 40
    # The film is above the layer: every Li and water above the upper silicon sheet's mean.
    upper_si_z = np.sort(images[0][species == "Si", 2])[6:].mean()
    assert (images[0][species == "Li", 2] > upper_si_z).all()
    assert (images[0][waters, 2] > upper_si_z).all()


def test_tile_in_plane_rebuilds_a_water_cut_by_the_cell_edge_and_repeats_it():
    # O just inside x = 0, one H wrapped round to the far side of a 10 A cell.
    positions = np.array([[0.1, 5.0, 5.0], [9.14, 5.0, 5.0], [0.34, 5.93, 5.0]])
    species, tiled = two_dimensional_frames.tile_in_plane(
        ["O", "H", "H"], positions, 10.0, 8.0, (2, 1)
    )
    assert species == ["O", "H", "H"] * 2
    oxygen, far_h, near_h = tiled[:3]
    assert np.linalg.norm(far_h - oxygen) == pytest.approx(0.96)
    assert np.linalg.norm(near_h - oxygen) == pytest.approx(np.hypot(0.24, 0.93))
    assert np.allclose(tiled[3:] - tiled[:3], [10.0, 0.0, 0.0])
    assert ((tiled[:3, 0] >= -1.0) & (tiled[:3, 0] < 10.0)).all()


def test_widest_gap_shift_joins_a_ring_split_by_the_cell_edge_and_leaves_a_slab_alone():
    # Rings 1.2 A apart along a 15.04 A tube cell, the end ring split between z = 0 and c.
    z_angstrom = np.concatenate([[0.02, 0.04], np.arange(1.29, 14.5, 1.2), [15.0, 15.02]])
    moved = z_angstrom + two_dimensional_frames.widest_gap_shift(z_angstrom, 15.04)
    assert np.allclose(np.sort(moved)[-4:], [15.0, 15.02, 15.06, 15.08])
    assert moved.min() == pytest.approx(1.29)
    # A bilayer's widest gap is its vacuum, across the boundary: nothing moves.
    slab_angstrom = np.array([12.8, 13.0, 16.2, 16.9])
    assert not two_dimensional_frames.widest_gap_shift(slab_angstrom, 24.0).any()


HBN_FRAMES = {
    "hbn-nanotube-6-6-frame-347.xyz": 72,
    "hbn-bilayer-frame-218.xyz": 98,
}


@pytest.mark.parametrize("name", sorted(HBN_FRAMES))
def test_an_hbn_frame_is_boron_nitride_bonded_only_b_to_n(name):
    frame = boxprep.read_xyz(DATA / name)
    assert Counter(frame.species) == {"B": HBN_FRAMES[name], "N": HBN_FRAMES[name]}
    bonds = boxprep.find_bonds(frame.positions_angstrom, frame.species)
    assert {frozenset((frame.species[a], frame.species[b])) for a, b in bonds} == {
        frozenset(("B", "N"))
    }


def test_the_nanotube_is_a_6_6_tube_and_the_bilayer_two_sheets_at_their_spacing():
    tube = boxprep.read_xyz(DATA / "hbn-nanotube-6-6-frame-347.xyz").positions_angstrom
    radius_angstrom = np.linalg.norm(tube[:, :2] - tube[:, :2].mean(axis=0), axis=1)
    # (6,6): circumference 6 x sqrt(3) x 2.51 A, so a radius of 4.15 A.
    assert radius_angstrom.mean() == pytest.approx(4.15, abs=0.1)
    assert radius_angstrom.std() < 0.2
    bilayer = boxprep.read_xyz(DATA / "hbn-bilayer-frame-218.xyz").positions_angstrom
    z_angstrom = np.sort(bilayer[:, 2])
    lower, upper = z_angstrom[:98], z_angstrom[98:]
    assert upper.min() - lower.max() > 2.0
    assert upper.mean() - lower.mean() == pytest.approx(3.39, abs=0.01)


PANELS = {panel["id"]: panel for panel in panel_checks.load_panels(DATA / "manifest.json")}
RENDERS = PROVENANCE["renders"]
RENDER_IDS = sorted(key for key in RENDERS if key not in {"note", "method"})


def test_every_panel_of_the_manifest_has_a_render_entry_and_every_entry_a_panel():
    assert (
        RENDER_IDS
        == sorted(PANELS)
        == ["hbn-bilayer", "hbn-nanotube", "hero", "vermiculite-li-film"]
    )


@pytest.mark.parametrize("panel_id", RENDER_IDS)
def test_a_render_entry_records_the_manifests_choices_and_chose_one_candidate(panel_id):
    entry, panel = RENDERS[panel_id], PANELS[panel_id]
    assert entry["manifest"] == "scripts/figures/data/two-dimensional-and-confined/manifest.json"
    assert entry["source"] == Path(panel["source"]).name
    assert entry["direction"] == panel["direction"]
    assert entry["resolution"] == panel["resolution"]
    assert entry["radii"]["sphere_radius_angstrom"] == panel["sphere_radius_angstrom"]
    assert entry["radii"]["bond_radius_angstrom"] == panel["bond_radius_angstrom"]
    assert entry["radii"].get("sphere_scale") == panel.get("sphere_scale")
    assert entry.get("depth_cue") == panel.get("depth_cue")
    chosen = [
        candidate for candidate in entry["candidates"] if candidate["verdict"].startswith("Chosen.")
    ]
    assert [candidate["direction"] for candidate in chosen] == [panel["direction"]]
    assert entry["why"] and entry["limits"] and entry["alt"]
    assert entry["wall_time_s"] > 0 and entry["summary_line"].startswith(f"{panel_id}: ")
    atoms = len(boxprep.read_xyz(REPO / panel["source"]).species)
    assert (
        f": {atoms} atoms, wall {entry['wall_time_s']} s, blender exit 0, " in entry["summary_line"]
    )
    assert str(Path.home()) not in entry["summary_line"]


# The Li/Mo = 0.3 cell: one 1T' MoS2 sheet of 56 Mo and 112 S, 16 Li and 100 waters.
MOS2_CELL = {"H": 200, "Li": 16, "Mo": 56, "O": 100, "S": 112}
MOS2_C_ANGSTROM = 11.5


def test_the_hero_frame_is_a_water_bilayer_and_lithium_between_two_sheets_one_c_apart():
    frame = boxprep.read_xyz(DATA / two_dimensional_frames.MOS2_OUT)
    species = np.array(frame.species)
    n_cell = sum(MOS2_CELL.values())
    cell, image = frame.positions_angstrom[:n_cell], frame.positions_angstrom[n_cell:]
    assert Counter(species[:n_cell].tolist()) == MOS2_CELL
    assert Counter(species[n_cell:].tolist()) == {"Mo": 56, "S": 112}
    assert MOS2_CELL["Li"] / MOS2_CELL["Mo"] == pytest.approx(0.286, abs=1e-3)
    # The upper sheet is the lower one moved one c up: the sheets are 11.5 A apart.
    lower_sheet = cell[np.isin(species[:n_cell], ["Mo", "S"])]
    assert np.allclose(image - lower_sheet, [0.0, 0.0, MOS2_C_ANGSTROM], atol=2e-6)
    lower_mo_z = cell[species[:n_cell] == "Mo", 2].mean()
    # Every hydrogen on one oxygen, and every oxygen a water: no molecule cut by the wrap.
    partners = {atom: [] for atom in range(n_cell)}
    for first, second in boxprep.find_bonds(cell, species[:n_cell].tolist()):
        partners[first].append(second)
        partners[second].append(first)
    for atom in np.flatnonzero(species[:n_cell] == "O"):
        assert sorted(species[partners[atom]]) == ["H", "H"]
    # Water and lithium sit between the sheets' inner sulfur planes (Mo +- 1.6 A), and the
    # oxygens split into two layers of 50 about the midplane: a bilayer.
    water_z = cell[species[:n_cell] == "O", 2] - lower_mo_z
    lithium_z = cell[species[:n_cell] == "Li", 2] - lower_mo_z
    for z in (water_z, lithium_z):
        assert (z > 1.6 + 2.0).all() and (z < MOS2_C_ANGSTROM - 1.6 - 2.0).all()
    lower_layer = water_z[water_z < MOS2_C_ANGSTROM / 2]
    upper_layer = water_z[water_z >= MOS2_C_ANGSTROM / 2]
    assert len(lower_layer) == len(upper_layer) == 50
    assert upper_layer.min() - lower_layer.max() > 0.5
    assert lower_layer.mean() == pytest.approx(4.58, abs=0.01)
    assert upper_layer.mean() == pytest.approx(6.85, abs=0.01)
