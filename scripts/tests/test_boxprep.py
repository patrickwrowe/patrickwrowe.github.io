"""Tests for scripts/figures/boxprep.py against real archive data."""

from __future__ import annotations

from pathlib import Path

import boxprep
import numpy as np
import pytest
from constants import CARBON_BOND_CUTOFF_ANGSTROM

FIXTURES = Path(__file__).parent / "fixtures"
DATA = Path(__file__).resolve().parents[1] / "figures" / "data"


def test_box_edge_matches_the_replicated_seed_cell():
    # 216-atom seed replicated 3 x 3 x 3 at 1.0 g cm^-3; the 2.0 value is the HC-C4 box
    # verified from the LAMMPS logs in the sodium-ion memory.
    assert boxprep.box_edge_from_density(5832, 1.0) == pytest.approx(48.81, abs=0.02)
    assert boxprep.box_edge_from_density(5832, 2.0) == pytest.approx(38.74, abs=0.02)


def test_wrap_folds_the_small_negative_excursions_the_frames_carry():
    positions_angstrom = np.array([[-0.4, 10.0, 48.9], [1.0, 1.0, 1.0]])
    wrapped = boxprep.wrap(positions_angstrom, 48.81)
    assert wrapped[0] == pytest.approx([48.41, 10.0, 0.09], abs=1e-9)
    assert (wrapped >= 0).all() and (wrapped < 48.81).all()


def test_tile_doubles_along_x_only():
    positions_angstrom = np.array([[1.0, 2.0, 3.0]])
    tiled = boxprep.tile(positions_angstrom, 10.0, (2, 1, 1))
    assert tiled.shape == (2, 3)
    assert tiled.tolist() == [[1.0, 2.0, 3.0], [11.0, 2.0, 3.0]]


def test_slab_mask_is_centred_and_inclusive():
    depth = np.array([0.0, 4.9, 5.0, 5.1, 10.0, 15.0, 20.0])
    mask = boxprep.slab_mask(depth, centre_angstrom=10.0, thickness_angstrom=10.0)
    assert mask.tolist() == [False, False, True, True, True, True, False]


def test_make_compact_rejoins_a_cluster_split_by_the_boundary():
    edge_angstrom = 64.8215
    positions_angstrom = np.array(
        [[0.5, 10.0, 10.0], [edge_angstrom - 0.52, 10.0, 10.0], [1.5, 10.0, 10.0]]
    )
    compact_angstrom = boxprep.make_compact(positions_angstrom, edge_angstrom)
    assert np.linalg.norm(compact_angstrom[0] - compact_angstrom[1]) == pytest.approx(
        1.02, abs=1e-6
    )
    assert compact_angstrom.mean(axis=0) == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)


def test_make_compact_preserves_the_pairwise_distances_of_a_real_frame():
    # A synthetic split proves the re-imaging logic; a real frame proves it doesn't distort
    # a structure that was never split. Shifting the real C40 frame by a fixed offset and
    # re-wrapping forces some atoms across the periodic boundary; make_compact should
    # recover exactly the original geometry, atom for atom.
    positions_angstrom, edge_angstrom, _ = boxprep.read_lammps_last_frame(
        FIXTURES / "c40-500K-two-frames.lammpstrj"
    )
    shifted = boxprep.wrap(positions_angstrom + np.array([30.0, 30.0, 30.0]), edge_angstrom)
    compact_angstrom = boxprep.make_compact(shifted, edge_angstrom)
    original_distances = np.linalg.norm(
        positions_angstrom[:, None, :] - positions_angstrom[None, :, :], axis=-1
    )
    compact_distances = np.linalg.norm(
        compact_angstrom[:, None, :] - compact_angstrom[None, :, :], axis=-1
    )
    assert compact_distances == pytest.approx(original_distances, abs=1e-6)


def test_read_lammps_last_frame_reads_the_second_frame_of_the_real_dump():
    positions_angstrom, edge_angstrom, timestep = boxprep.read_lammps_last_frame(
        FIXTURES / "c40-500K-two-frames.lammpstrj"
    )
    assert timestep == 75  # the fixture's frames are timesteps 50 and 75
    # The cell scales with cluster size: 48.489 A for C40, 64.82 A for C1000.
    assert edge_angstrom == pytest.approx(48.489, abs=1e-3)
    assert positions_angstrom.shape == (40, 3)
    assert (positions_angstrom >= -0.01).all() and (
        positions_angstrom <= edge_angstrom + 0.01
    ).all()
    lines = (FIXTURES / "c40-500K-two-frames.lammpstrj").read_text().splitlines()
    lower = np.array([float(line.split()[0]) for line in lines[-44:-41]])  # BOX BOUNDS lo
    last_frame = np.array([line.split() for line in lines[-40:]], dtype=float)
    first_id = int(last_frame[0, 0])
    expected = lower + last_frame[0, 2:5] * edge_angstrom  # xs * (hi - lo) + lo
    assert positions_angstrom[first_id - 1] == pytest.approx(expected, abs=1e-3)


def test_read_lammps_last_frame_rejects_a_truncated_frame(tmp_path):
    text = (FIXTURES / "c40-500K-two-frames.lammpstrj").read_text()
    lines = text.splitlines()

    # Whole atom rows missing: caught by the atom-count check.
    (tmp_path / "cut-rows.lammpstrj").write_text("\n".join(lines[:-3]) + "\n")
    with pytest.raises(ValueError, match="truncated"):
        boxprep.read_lammps_last_frame(tmp_path / "cut-rows.lammpstrj")

    # Cut inside the header: the BOX BOUNDS and ATOMS lines are gone entirely, caught by
    # the header parse's IndexError guard.
    (tmp_path / "cut-header.lammpstrj").write_text("\n".join(lines[:-45]) + "\n")
    with pytest.raises(ValueError, match="truncated"):
        boxprep.read_lammps_last_frame(tmp_path / "cut-header.lammpstrj")

    # A last block with no timestep value: the header is too short to parse.
    (tmp_path / "no-timestep.lammpstrj").write_text(text + "ITEM: TIMESTEP\n")
    with pytest.raises(ValueError, match="truncated"):
        boxprep.read_lammps_last_frame(tmp_path / "no-timestep.lammpstrj")

    # Cut inside the last number, with no trailing newline: a write that stopped mid-token,
    # caught by the newline check rather than silently parsing a truncated float.
    (tmp_path / "cut-number.lammpstrj").write_text(text[:-5])
    with pytest.raises(ValueError, match="truncated"):
        boxprep.read_lammps_last_frame(tmp_path / "cut-number.lammpstrj")


def test_read_xyz_bonds_a_boron_nitrogen_pair_at_the_hbn_bond_length(tmp_path):
    # r(B-N) = 1.45 A in hexagonal boron nitride; the cutoff is 1.2 x (0.84 + 0.71) = 1.86 A.
    source = tmp_path / "bn.xyz"
    source.write_text("2\nB-N pair\nB 0.0 0.0 0.0\nN 1.45 0.0 0.0\n")
    structure = boxprep.read_xyz(source)
    assert structure.species == ["B", "N"]
    assert boxprep.find_bonds(structure.positions_angstrom, structure.species) == [(0, 1)]


def test_an_hbn_hexagon_has_six_bonds_and_none_between_second_neighbours(tmp_path):
    # Alternating B and N on a ring of side 1.45 A: second neighbours are 1.45 * sqrt(3) =
    # 2.51 A apart, beyond the B-B cutoff of 2.02 A and the N-N cutoff of 1.70 A.
    angles = np.deg2rad(60.0 * np.arange(6))
    ring_angstrom = 1.45 * np.column_stack([np.cos(angles), np.sin(angles), np.zeros(6)])
    species = ["B", "N"] * 3
    source = tmp_path / "ring.xyz"
    source.write_text(
        "6\nhBN ring\n"
        + "".join(
            f"{symbol} {x:.4f} {y:.4f} {z:.4f}\n"
            for symbol, (x, y, z) in zip(species, ring_angstrom, strict=True)
        )
    )
    structure = boxprep.read_xyz(source)
    bonds = boxprep.find_bonds(structure.positions_angstrom, structure.species)
    assert bonds == [(0, 1), (0, 5), (1, 2), (2, 3), (3, 4), (4, 5)]


def _bonds_at(species: list[str], distance_angstrom: float) -> list[tuple[int, int]]:
    """`find_bonds` on two atoms `distance_angstrom` apart along x."""
    positions_angstrom = np.array([[0.0, 0.0, 0.0], [distance_angstrom, 0.0, 0.0]])
    return boxprep.find_bonds(positions_angstrom, species)


@pytest.mark.parametrize(
    ("ion", "distance_angstrom"),
    # Each distance is inside the radius cutoff 1.2 x (r_ion + r_O): 2.33, 2.78, 3.23 A.
    [("Li", 2.00), ("Na", 2.40), ("K", 2.80)],
)
@pytest.mark.parametrize("ion_first", [True, False], ids=["ion first", "ion second"])
def test_find_bonds_never_bonds_a_bare_ion_whichever_atom_comes_first(
    ion, distance_angstrom, ion_first
):
    species = [ion, "O"] if ion_first else ["O", ion]
    assert _bonds_at(species, distance_angstrom) == []


@pytest.mark.parametrize(
    ("pair", "distance_angstrom"),
    # Each distance is inside the radius cutoff 1.2 x (r_a + r_b): 3.70, 3.38, 2.90, 3.54,
    # 3.30, 3.14 A.
    [
        (("Mo", "Mo"), 3.16),
        (("Mg", "Mg"), 3.10),
        (("Al", "Al"), 2.80),
        (("Mo", "Mg"), 3.20),
        (("Mo", "Al"), 3.10),
        (("Mg", "Al"), 3.00),
    ],
    ids=lambda value: "-".join(value) if isinstance(value, tuple) else f"{value} A",
)
def test_find_bonds_never_bonds_two_layer_metals(pair, distance_angstrom):
    assert _bonds_at(list(pair), distance_angstrom) == []
    assert _bonds_at(list(reversed(pair)), distance_angstrom) == []


@pytest.mark.parametrize(
    ("species", "distance_angstrom"),
    [(["Mo", "S"], 2.41), (["Mg", "O"], 2.10), (["Al", "O"], 1.90), (["Si", "O"], 1.62)],
    ids=["Mo-S", "Mg-O", "Al-O", "Si-O"],
)
def test_find_bonds_still_bonds_a_layer_metal_to_its_anion(species, distance_angstrom):
    assert _bonds_at(species, distance_angstrom) == [(0, 1)]


def test_read_xyz_refuses_a_symbol_without_a_covalent_radius_and_names_it(tmp_path):
    source = tmp_path / "xx.xyz"
    source.write_text("2\nunknown\nC 0.0 0.0 0.0\nXx 1.5 0.0 0.0\n")
    with pytest.raises(ValueError, match=r"no covalent radius for \['Xx'\]"):
        boxprep.read_xyz(source)


def test_write_pdb_emits_fixed_columns_and_conect(tmp_path):
    positions_angstrom = np.array([[0.0, 0.0, 0.0], [1.42, 0.0, 0.0], [-12.345, 6.0, 0.0]])
    out = tmp_path / "three.pdb"
    boxprep.write_pdb(out, ["C"] * 3, positions_angstrom, [(0, 1)])
    lines = out.read_text().splitlines()
    assert (
        lines[0] == "HETATM    1  C   CBX A   1       0.000   0.000   0.000  1.00  0.00           C"
    )
    assert lines[2][30:54] == " -12.345   6.000   0.000"
    assert "CONECT    1    2" in lines and "CONECT    2    1" in lines
    assert lines[-1] == "END"


def test_write_pdb_rejects_malformed_input(tmp_path):
    out = tmp_path / "bad.pdb"
    with pytest.raises(ValueError, match="99,999"):
        boxprep.write_pdb(out, ["C"] * 100_000, np.zeros((100_000, 3)), [])
    with pytest.raises(ValueError, match="field width"):
        boxprep.write_pdb(out, ["C"], np.array([[10000.0, 0.0, 0.0]]), [])
    with pytest.raises(ValueError, match="bond index"):
        boxprep.write_pdb(out, ["C"], np.array([[0.0, 0.0, 0.0]]), [(0, 1)])
    with pytest.raises(ValueError, match="2 species for 1 positions"):
        boxprep.write_pdb(out, ["C", "C"], np.array([[0.0, 0.0, 0.0]]), [])


def test_write_pdb_refuses_a_symbol_without_a_covalent_radius_and_names_it(tmp_path):
    # An unknown symbol must stop the write, not be written as some other element.
    out = tmp_path / "xx.pdb"
    positions_angstrom = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match=r"no covalent radius for \['Xx'\]"):
        boxprep.write_pdb(out, ["O", "Xx"], positions_angstrom, [(0, 1)])
    assert not out.exists()


@pytest.mark.parametrize(
    ("species", "positions_angstrom", "names", "elements"),
    [
        # Gas-phase water: r(O-H) = 0.9572 A, H-O-H = 104.52 degrees.
        (
            ["O", "H", "H"],
            [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2399, 0.9266, 0.0]],
            [" O  ", " H  ", " H  "],
            [" O", " H", " H"],
        ),
        # A MoS2 fragment: r(Mo-S) = 2.41 A, S-Mo-S = 120 degrees; a two-letter element
        # starts its name in column 13 and fills columns 77-78.
        (
            ["Mo", "S", "S"],
            [[0.0, 0.0, 0.0], [2.41, 0.0, 0.0], [-1.205, 2.087, 0.0]],
            ["MO  ", " S  ", " S  "],
            ["MO", " S", " S"],
        ),
    ],
    ids=["water", "MoS2 fragment"],
)
def test_write_pdb_round_trips_the_species_and_bonds_of_a_molecule(
    tmp_path, species, positions_angstrom, names, elements
):
    positions_angstrom = np.array(positions_angstrom)
    bonds = boxprep.find_bonds(positions_angstrom, species)
    assert bonds == [(0, 1), (0, 2)]
    out = tmp_path / "molecule.pdb"
    boxprep.write_pdb(out, species, positions_angstrom, bonds)

    lines = out.read_text().splitlines()
    atoms = [line for line in lines if line.startswith("HETATM")]
    assert [line[12:16] for line in atoms] == names
    assert [line[76:78] for line in atoms] == elements
    assert [line[30:54] for line in atoms] == [
        "".join(f"{coordinate:8.3f}" for coordinate in position) for position in positions_angstrom
    ]
    # One CONECT line per atom, each bond written from both ends: two bonds in all.
    conect = [line for line in lines if line.startswith("CONECT")]
    recovered = {
        frozenset((int(line[6:11]), int(partner)))
        for line in conect
        for partner in line[11:].split()
    }
    assert len(conect) == 3
    assert recovered == {frozenset((1, 2)), frozenset((1, 3))}


def test_real_dense_slab_has_sensible_coordination():
    # 3.5 g cm^-3 is mostly sp3. The archive's own statistics for this density
    # (scripts/figures/data/graphitisation/density-3.5gcc-coordination.txt) are 5.11%
    # three-, 94.78% four- and 0.11% five-coordinated, mean 3.9465. An untiled slab drags
    # the mean down to about 3.49: atoms near the cell's x/y faces are missing real
    # periodic neighbours that find_bonds, with no wrapping of its own, can't see. Tiling
    # 3 x 3 in x and y gives the centre copy's atoms their true neighbours, exactly the
    # render path a Blender driver would use; scoring only that centre copy, and only
    # atoms at least one bond cutoff (1.824 A) inside both slab faces, then reproduces the
    # archive's chemistry. A wrong edge shows up here as overlapping atoms at the seams.
    structure = boxprep.read_xyz(DATA / "graphitisation" / "final-frame-3.5gcc.xyz")
    edge_angstrom = boxprep.box_edge_from_density(len(structure), 3.5)
    wrapped = boxprep.wrap(structure.positions_angstrom, edge_angstrom)
    tiled = boxprep.tile(wrapped, edge_angstrom, (3, 3, 1))
    centre_angstrom = edge_angstrom / 2.0
    slab = tiled[boxprep.slab_mask(tiled[:, 2], centre_angstrom, 10.0)]
    bonds = boxprep.find_bonds(slab)
    counts = np.bincount(np.array(bonds).ravel(), minlength=len(slab))
    in_centre_tile = (slab[:, 0] >= edge_angstrom) & (slab[:, 0] < 2 * edge_angstrom)
    in_centre_tile &= (slab[:, 1] >= edge_angstrom) & (slab[:, 1] < 2 * edge_angstrom)
    away_from_slab_faces = np.abs(slab[:, 2] - centre_angstrom) <= 5.0 - CARBON_BOND_CUTOFF_ANGSTROM
    bulk_counts = counts[in_centre_tile & away_from_slab_faces]
    assert bulk_counts.mean() == pytest.approx(3.95, abs=0.03)
    assert np.mean(bulk_counts == 4) == pytest.approx(0.948, abs=0.01)
    assert bulk_counts.max() <= 5


def fragment_of(atom: int, n_atoms: int, bonds: list[tuple[int, int]]) -> np.ndarray:
    """Indices of the atoms bonded, directly or through others, to `atom`.

    Args:
        atom: Zero-based index of the atom whose fragment is wanted.
        n_atoms: Number of atoms in the frame.
        bonds: Zero-based (first, second) atom index pairs.

    Returns:
        Sorted zero-based indices of every atom in that fragment, `atom` included.
    """
    reached = np.zeros(n_atoms, dtype=bool)
    reached[atom] = True
    bond_array = np.array(bonds)
    while True:
        grown = reached.copy()
        grown[bond_array[reached[bond_array[:, 0]], 1]] = True
        grown[bond_array[reached[bond_array[:, 1]], 0]] = True
        if np.array_equal(grown, reached):
            return np.flatnonzero(reached)
        reached = grown


def straddle_first_bond(
    compact_angstrom: np.ndarray, edge_angstrom: float
) -> tuple[np.ndarray, tuple[int, int]]:
    """Translate a whole cluster so its first bond's midpoint sits on the x = 0 face, then wrap.

    Args:
        compact_angstrom: Whole-cluster positions, shape (n_atoms, 3), in angstrom.
        edge_angstrom: Cubic cell edge, in angstrom.

    Returns:
        (wrapped positions in angstrom, the (first, second) pair that now straddles the face).
    """
    first, second = boxprep.find_bonds(compact_angstrom)[0]
    midpoint_x_angstrom = 0.5 * (compact_angstrom[first, 0] + compact_angstrom[second, 0])
    shifted = boxprep.wrap(
        compact_angstrom - np.array([midpoint_x_angstrom, 0.0, 0.0]), edge_angstrom
    )
    return shifted, (first, second)


def test_periodic_bonds_and_make_whole_survive_a_bond_cut_by_the_boundary():
    # Adversarial real data: translate the C40 fixture frame so the midpoint of a known bond
    # sits exactly on the x = 0 face, then wrap. The bond's two atoms end up about one cell
    # edge apart in Cartesian space, the case the non-periodic search cuts.
    positions_angstrom, edge_angstrom, _ = boxprep.read_lammps_last_frame(
        FIXTURES / "c40-500K-two-frames.lammpstrj"
    )
    compact_angstrom = boxprep.make_compact(positions_angstrom, edge_angstrom)
    original_bonds = boxprep.find_bonds(compact_angstrom)
    straddling, (first, second) = straddle_first_bond(compact_angstrom, edge_angstrom)
    assert abs(straddling[first, 0] - straddling[second, 0]) > edge_angstrom / 2
    assert (first, second) not in boxprep.find_bonds(straddling)

    periodic_bonds = boxprep.find_bonds_periodic(straddling, edge_angstrom)
    assert periodic_bonds == original_bonds

    # This frame is three fragments (33, 4 and 3 atoms at the 1.824 A cutoff). make_whole
    # restores every bond, and all distances within a fragment; where the fragments sit
    # relative to one another is left as the wrap put them, by design.
    whole_angstrom = boxprep.make_whole(straddling, edge_angstrom, periodic_bonds)
    bond_array = np.array(original_bonds)
    assert len(fragment_of(first, len(compact_angstrom), original_bonds)) == 33
    original_lengths = np.linalg.norm(
        compact_angstrom[bond_array[:, 0]] - compact_angstrom[bond_array[:, 1]], axis=1
    )
    whole_lengths = np.linalg.norm(
        whole_angstrom[bond_array[:, 0]] - whole_angstrom[bond_array[:, 1]], axis=1
    )
    assert whole_lengths == pytest.approx(original_lengths, abs=1e-6)


def test_make_whole_restores_the_distance_matrix_of_a_real_fragment_cut_by_the_boundary():
    # The same straddle on the fixture's 33-atom fragment alone: one fragment, so the whole
    # pairwise distance matrix must come back, not just the bonds.
    positions_angstrom, edge_angstrom, _ = boxprep.read_lammps_last_frame(
        FIXTURES / "c40-500K-two-frames.lammpstrj"
    )
    compact_angstrom = boxprep.make_compact(positions_angstrom, edge_angstrom)
    fragment = compact_angstrom[
        fragment_of(0, len(compact_angstrom), boxprep.find_bonds(compact_angstrom))
    ]
    assert len(fragment) == 33
    straddling, (first, second) = straddle_first_bond(fragment, edge_angstrom)
    assert abs(straddling[first, 0] - straddling[second, 0]) > edge_angstrom / 2
    periodic_bonds = boxprep.find_bonds_periodic(straddling, edge_angstrom)
    assert periodic_bonds == boxprep.find_bonds(fragment)
    whole_angstrom = boxprep.make_whole(straddling, edge_angstrom, periodic_bonds)
    original_distances = np.linalg.norm(fragment[:, None, :] - fragment[None, :, :], axis=-1)
    whole_distances = np.linalg.norm(
        whole_angstrom[:, None, :] - whole_angstrom[None, :, :], axis=-1
    )
    assert whole_distances == pytest.approx(original_distances, abs=1e-6)


def test_make_whole_keeps_each_fragment_root_and_joins_its_bonds():
    # Two dimers in a 10 A cell, the second split across the x face. Each fragment's
    # lowest-indexed atom stays put; its partner moves to the adjacent image.
    edge_angstrom = 10.0
    positions_angstrom = np.array(
        [[5.0, 5.0, 5.0], [6.4, 5.0, 5.0], [0.3, 2.0, 2.0], [9.1, 2.0, 2.0]]
    )
    bonds = boxprep.find_bonds_periodic(positions_angstrom, edge_angstrom)
    assert bonds == [(0, 1), (2, 3)]
    whole_angstrom = boxprep.make_whole(positions_angstrom, edge_angstrom, bonds)
    assert whole_angstrom[[0, 2]] == pytest.approx(positions_angstrom[[0, 2]])
    assert whole_angstrom[3] == pytest.approx([-0.9, 2.0, 2.0])
    assert whole_angstrom[1] == pytest.approx(positions_angstrom[1])


def test_periodic_bonds_reject_a_cell_too_small_for_the_minimum_image():
    with pytest.raises(ValueError, match="twice"):
        boxprep.find_bonds_periodic(np.zeros((2, 3)), 3.0)
    with pytest.raises(ValueError, match="bond index"):
        boxprep.make_whole(np.zeros((2, 3)), 10.0, [(0, 2)])


def test_make_whole_rejects_a_chain_bonded_to_its_own_image():
    # Seven atoms 1.42 A apart along x in a 10 A cell: the last is 1.48 A from the first's
    # periodic image, so the chain closes on itself through the boundary and one bond
    # would have to stretch across the cell. make_whole must refuse, not return it.
    edge_angstrom = 10.0
    chain = np.column_stack([np.arange(7) * 1.42, np.full(7, 5.0), np.full(7, 5.0)])
    bonds = boxprep.find_bonds_periodic(chain, edge_angstrom)
    assert (0, 6) in bonds and len(bonds) == 7
    with pytest.raises(ValueError, match="own periodic image"):
        boxprep.make_whole(chain, edge_angstrom, bonds)


def test_find_bonds_periodic_cutoff_argument_moves_the_cutoff():
    positions = np.array([[0.0, 0.0, 0.0], [1.9, 0.0, 0.0], [0.0, 2.3, 0.0]])
    assert set(boxprep.find_bonds_periodic(positions, 10.0)) == set()
    assert set(boxprep.find_bonds_periodic(positions, 10.0, cutoff_angstrom=2.0)) == {(0, 1)}
    assert set(boxprep.find_bonds_periodic(positions, 10.0, cutoff_angstrom=2.5)) == {
        (0, 1),
        (0, 2),
    }


def test_read_pdb_keeps_model_one_the_first_conformer_and_no_water():
    # toy-protein.pdb holds 13 atom records; read_pdb drops the second conformer of the
    # alpha carbon (serial 3), the water (12) and the whole of model 2 (13).
    protein = boxprep.read_pdb(FIXTURES / "toy-protein.pdb")
    assert len(protein) == 10
    assert protein.species == ["N", "C", "C", "N", "C", "N", "C", "Zn", "Mg", "C"]
    assert protein.chain_ids.tolist() == list("AAAAABBAAB")
    assert protein.residue_numbers.tolist() == [1, 1, 1, 2, 2, 7, 7, 101, 102, 301]
    assert protein.is_hetero.tolist() == [False] * 7 + [True] * 3
    # The conformer kept is A's (x = 1.458), not B's (1.600).
    assert protein.positions_angstrom[1].tolist() == [1.458, 0.0, 0.0]
    assert 1.600 not in protein.positions_angstrom[:, 0]
    assert protein.positions_angstrom.shape == (10, 3)
    # One altloc B atom (serial 3) and one water (12) were dropped; model 2 is not counted.
    assert (protein.n_dropped_altloc, protein.n_dropped_water) == (1, 1)


def test_read_pdb_reads_a_two_letter_element_and_falls_back_to_the_atom_name():
    # Zn has columns 77-78; Mg (HETATM) and the second chain-B alpha carbon (ATOM) do not.
    protein = boxprep.read_pdb(FIXTURES / "toy-protein.pdb")
    assert protein.species[7:9] == ["Zn", "Mg"]
    assert protein.species[6] == "C"


def test_read_pdb_returns_the_kept_records_verbatim_and_the_secondary_structure():
    protein = boxprep.read_pdb(FIXTURES / "toy-protein.pdb")
    serials = [int(record[6:11]) for record in protein.records]
    assert serials == [1, 2, 4, 5, 6, 7, 8, 9, 10, 11]
    assert protein.records[1] == (FIXTURES / "toy-protein.pdb").read_text().splitlines()[5]
    assert [record[:5] for record in protein.secondary_structure_records] == ["HELIX", "SHEET"]


def test_read_pdb_refuses_a_file_with_no_atom_to_draw(tmp_path):
    only_water = tmp_path / "water.pdb"
    only_water.write_text(
        "HETATM    1  O   HOH A   1       0.000   0.000   0.000  1.00 20.00           O\nEND\n"
    )
    with pytest.raises(ValueError, match="no atoms to draw"):
        boxprep.read_pdb(only_water)


def test_read_pdb_keeps_the_insertion_code_apart_from_the_residue_number():
    # Atom 8 of the fixture sits in residue 7A; residue_numbers alone would call it 7.
    protein = boxprep.read_pdb(FIXTURES / "toy-protein.pdb")
    assert protein.insertion_codes.tolist() == [""] * 6 + ["A"] + [""] * 3
    assert protein.residue_numbers[6] == 7
    assert protein.records[6][26] == "A"


def test_drawn_by_species_is_the_stick_chains_and_every_hetero_atom():
    protein = boxprep.read_pdb(FIXTURES / "toy-protein.pdb")
    assert protein.drawn_by_species(["B"]).tolist() == [False] * 5 + [True] * 5
    # No stick chain: only the zinc, the magnesium and the ligand carbon are drawn by species.
    assert protein.drawn_by_species([]).tolist() == [False] * 7 + [True] * 3


@pytest.mark.parametrize(
    ("start", "stop", "text"),
    [(30, 38, "     abc"), (46, 54, "1.2.3.4."), (22, 26, "  x7")],
    ids=["x coordinate", "z coordinate", "residue number"],
)
def test_read_pdb_names_the_file_and_line_of_a_record_it_cannot_parse(tmp_path, start, stop, text):
    lines = (FIXTURES / "toy-protein.pdb").read_text().splitlines()
    index = next(n for n, line in enumerate(lines) if line.startswith("ATOM      5"))
    lines[index] = lines[index][:start] + text[: stop - start] + lines[index][stop:]
    broken = tmp_path / "broken.pdb"
    broken.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match=rf"broken\.pdb: line {index + 1}: .*not numbers"):
        boxprep.read_pdb(broken)
