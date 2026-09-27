"""Tests for scripts/figures/boxprep.py against real archive data."""

from __future__ import annotations

from pathlib import Path

import boxprep
import numpy as np
import pytest

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
    positions_angstrom, edge_angstrom = boxprep.read_lammps_last_frame(
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
    positions_angstrom, edge_angstrom = boxprep.read_lammps_last_frame(
        FIXTURES / "c40-500K-two-frames.lammpstrj"
    )
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

    # Cut inside the last number, with no trailing newline: a write that stopped mid-token,
    # caught by the newline check rather than silently parsing a truncated float.
    (tmp_path / "cut-number.lammpstrj").write_text(text[:-5])
    with pytest.raises(ValueError, match="truncated"):
        boxprep.read_lammps_last_frame(tmp_path / "cut-number.lammpstrj")


def test_write_pdb_emits_fixed_columns_and_conect(tmp_path):
    positions_angstrom = np.array([[0.0, 0.0, 0.0], [1.42, 0.0, 0.0], [-12.345, 6.0, 0.0]])
    out = tmp_path / "three.pdb"
    boxprep.write_pdb(out, positions_angstrom, [(0, 1)])
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
        boxprep.write_pdb(out, np.zeros((100_000, 3)), [])
    with pytest.raises(ValueError, match="field width"):
        boxprep.write_pdb(out, np.array([[10000.0, 0.0, 0.0]]), [])
    with pytest.raises(ValueError, match="bond index"):
        boxprep.write_pdb(out, np.array([[0.0, 0.0, 0.0]]), [(0, 1)])


def test_real_dense_slab_has_sensible_coordination():
    # 3.5 g cm^-3 is mostly sp3. The archive's own statistics for this density
    # (scripts/figures/data/graphitisation/density-3.5gcc-coordination.txt) are 5.11%
    # three-, 94.78% four- and 0.11% five-coordinated, mean 3.9465. An untiled slab drags
    # the mean down to about 3.49: atoms near the cell's x/y faces are missing real
    # periodic neighbours that find_bonds, with no wrapping of its own, can't see. Tiling
    # 3 x 3 in x and y gives the centre copy's atoms their true neighbours, exactly the
    # render path a Blender driver would use; scoring only that centre copy, and only
    # atoms at least 1.82 A (the bond cutoff) inside both slab faces, then reproduces the
    # archive's chemistry. A wrong edge shows up here as overlapping atoms at the seams.
    structure = boxprep.read_xyz(DATA / "graphitisation" / "final-frame-3.5gcc.xyz")
    edge_angstrom = boxprep.box_edge_from_density(len(structure), 3.5)
    wrapped = boxprep.wrap(structure.positions_angstrom, edge_angstrom)
    tiled = boxprep.tile(wrapped, edge_angstrom, (3, 3, 1))
    centre_angstrom = edge_angstrom / 2.0
    slab = tiled[boxprep.slab_mask(tiled[:, 2], centre_angstrom, 10.0)]
    bonds = boxprep.find_bonds(slab)
    counts = np.bincount(np.array(bonds).ravel(), minlength=len(slab))
    bond_cutoff_angstrom = 1.82
    in_centre_tile = (slab[:, 0] >= edge_angstrom) & (slab[:, 0] < 2 * edge_angstrom)
    in_centre_tile &= (slab[:, 1] >= edge_angstrom) & (slab[:, 1] < 2 * edge_angstrom)
    away_from_slab_faces = np.abs(slab[:, 2] - centre_angstrom) <= 5.0 - bond_cutoff_angstrom
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
    positions_angstrom, edge_angstrom = boxprep.read_lammps_last_frame(
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
    positions_angstrom, edge_angstrom = boxprep.read_lammps_last_frame(
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
