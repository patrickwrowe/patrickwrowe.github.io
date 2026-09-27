"""Tests for scripts/figures/boxprep.py against real archive data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))
import boxprep  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
DATA = Path(__file__).resolve().parents[1] / "figures" / "data"


def test_box_edge_matches_the_replicated_seed_cell():
    # 216-atom seed replicated 3 x 3 x 3 at 1.0 g cm^-3; the 2.0 value is the HC-C4 box
    # verified from the LAMMPS logs in the sodium-ion memory.
    assert boxprep.box_edge_from_density(5832, 1.0) == pytest.approx(48.81, abs=0.02)
    assert boxprep.box_edge_from_density(5832, 2.0) == pytest.approx(38.74, abs=0.02)


def test_wrap_folds_the_small_negative_excursions_the_frames_carry():
    positions = np.array([[-0.4, 10.0, 48.9], [1.0, 1.0, 1.0]])
    wrapped = boxprep.wrap(positions, 48.81)
    assert wrapped[0] == pytest.approx([48.41, 10.0, 0.09], abs=1e-9)
    assert (wrapped >= 0).all() and (wrapped < 48.81).all()


def test_tile_doubles_along_x_only():
    positions = np.array([[1.0, 2.0, 3.0]])
    tiled = boxprep.tile(positions, 10.0, (2, 1, 1))
    assert tiled.shape == (2, 3)
    assert tiled.tolist() == [[1.0, 2.0, 3.0], [11.0, 2.0, 3.0]]


def test_slab_mask_is_centred_and_inclusive():
    depth = np.array([0.0, 4.9, 5.0, 5.1, 10.0, 15.0, 20.0])
    mask = boxprep.slab_mask(depth, centre_angstrom=10.0, thickness_angstrom=10.0)
    assert mask.tolist() == [False, False, True, True, True, True, False]


def test_make_compact_rejoins_a_cluster_split_by_the_boundary():
    edge = 64.8215
    positions = np.array([[0.5, 10.0, 10.0], [edge - 0.52, 10.0, 10.0], [1.5, 10.0, 10.0]])
    compact = boxprep.make_compact(positions, edge)
    assert np.linalg.norm(compact[0] - compact[1]) == pytest.approx(1.02, abs=1e-6)
    assert compact.mean(axis=0) == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)


def test_make_compact_preserves_the_pairwise_distances_of_a_real_frame():
    # A synthetic split proves the re-imaging logic; a real frame proves it doesn't distort
    # a structure that was never split. Shifting the real C40 frame by a fixed offset and
    # re-wrapping forces some atoms across the periodic boundary; make_compact should
    # recover exactly the original geometry, atom for atom.
    positions, edge = boxprep.read_lammps_last_frame(FIXTURES / "c40-500K-two-frames.lammpstrj")
    shifted = boxprep.wrap(positions + np.array([30.0, 30.0, 30.0]), edge)
    compact = boxprep.make_compact(shifted, edge)
    original_distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    compact_distances = np.linalg.norm(compact[:, None, :] - compact[None, :, :], axis=-1)
    assert compact_distances == pytest.approx(original_distances, abs=1e-6)


def test_read_lammps_last_frame_reads_the_second_frame_of_the_real_dump():
    positions, edge = boxprep.read_lammps_last_frame(FIXTURES / "c40-500K-two-frames.lammpstrj")
    # The cell scales with cluster size: 48.489 A for C40, 64.82 A for C1000.
    assert edge == pytest.approx(48.489, abs=1e-3)
    assert positions.shape == (40, 3)
    assert (positions >= -0.01).all() and (positions <= edge + 0.01).all()
    lines = (FIXTURES / "c40-500K-two-frames.lammpstrj").read_text().splitlines()
    lower = np.array([float(line.split()[0]) for line in lines[-44:-41]])  # BOX BOUNDS lo
    last_frame = np.array([line.split() for line in lines[-40:]], dtype=float)
    first_id = int(last_frame[0, 0])
    expected = lower + last_frame[0, 2:5] * edge  # xs * (hi - lo) + lo
    assert positions[first_id - 1] == pytest.approx(expected, abs=1e-3)


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
    positions = np.array([[0.0, 0.0, 0.0], [1.42, 0.0, 0.0], [-12.345, 6.0, 0.0]])
    out = tmp_path / "three.pdb"
    boxprep.write_pdb(out, positions, [(0, 1)])
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
    edge = boxprep.box_edge_from_density(len(structure), 3.5)
    wrapped = boxprep.wrap(structure.positions_angstrom, edge)
    tiled = boxprep.tile(wrapped, edge, (3, 3, 1))
    centre_angstrom = edge / 2.0
    slab = tiled[boxprep.slab_mask(tiled[:, 2], centre_angstrom, 10.0)]
    bonds = boxprep.find_bonds(slab)
    counts = np.bincount(np.array(bonds).ravel(), minlength=len(slab))
    bond_cutoff_angstrom = 1.82
    in_centre_tile = (slab[:, 0] >= edge) & (slab[:, 0] < 2 * edge)
    in_centre_tile &= (slab[:, 1] >= edge) & (slab[:, 1] < 2 * edge)
    away_from_slab_faces = np.abs(slab[:, 2] - centre_angstrom) <= 5.0 - bond_cutoff_angstrom
    bulk_counts = counts[in_centre_tile & away_from_slab_faces]
    assert bulk_counts.mean() == pytest.approx(3.95, abs=0.03)
    assert np.mean(bulk_counts == 4) == pytest.approx(0.948, abs=0.01)
    assert bulk_counts.max() <= 5
