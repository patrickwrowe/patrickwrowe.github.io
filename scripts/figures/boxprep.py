# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Geometry for rendering periodic carbon boxes and free carbon clusters.

The archive's final frames are plain XYZ with no cell line and the cluster dumps are
LAMMPS `atom`-style with scaled coordinates. This module recovers the cell, folds and
tiles the box, cuts the slab a camera can see into, re-images a cluster the periodic
dump has split, and writes a PDB with explicit CONECT records, because Molecular Nodes
reads bonds from CONECT and infers none for a residue it does not know
(docs/dossiers/carbon/molrender-api.md section 10).

Pure numpy. The dense bond search is inherited from render_cluster.py and is fine for
the largest thing rendered here, a two-thousand-atom slab; a whole 5,832-atom box is
never bonded, it is sliced first.

Units: every length is in angstrom, every density in g cm^-3.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from render_cluster import (
    BOND_TOLERANCE,
    COVALENT_RADIUS_ANGSTROM,
    Structure,
    find_bonds,
    read_xyz,
)

__all__ = [
    "Structure",
    "box_edge_from_density",
    "find_bonds",
    "find_bonds_periodic",
    "make_compact",
    "make_whole",
    "read_lammps_last_frame",
    "read_xyz",
    "slab_mask",
    "tile",
    "wrap",
    "write_pdb",
]

CARBON_MASS_G_PER_MOL = 12.011
AVOGADRO_PER_MOL = 6.02214076e23
CM_TO_ANGSTROM = 1.0e8
# The flat carbon-carbon cutoff find_bonds applies: 1.2 x (0.76 + 0.76) = 1.824 A.
CARBON_BOND_CUTOFF_ANGSTROM = BOND_TOLERANCE * 2 * COVALENT_RADIUS_ANGSTROM["C"]


def box_edge_from_density(n_atoms: int, density_g_cm3: float) -> float:
    """Edge of the cubic cell holding `n_atoms` carbon atoms at `density_g_cm3`, in angstrom.

    The archive frames carry no cell line, so the cell is recovered from the density the
    run was set up at: 5,832 atoms at 1.0 g cm^-3 give 48.81 A, the 216-atom seed cell
    replicated 3 x 3 x 3.

    Args:
        n_atoms: Number of carbon atoms the cell holds.
        density_g_cm3: Target density, in g cm^-3.

    Returns:
        The cubic cell's edge length, in angstrom.
    """
    mass_g = n_atoms * CARBON_MASS_G_PER_MOL / AVOGADRO_PER_MOL
    volume_cm3 = mass_g / density_g_cm3
    return volume_cm3 ** (1.0 / 3.0) * CM_TO_ANGSTROM


def wrap(positions_angstrom: np.ndarray, edge_angstrom: float) -> np.ndarray:
    """Positions folded into [0, edge) on every axis.

    Args:
        positions_angstrom: Atom positions, shape (n_atoms, 3), in angstrom.
        edge_angstrom: Cubic cell edge length, in angstrom.

    Returns:
        Positions folded into [0, edge_angstrom) on every axis, same shape as the input.
    """
    return np.mod(positions_angstrom, edge_angstrom)


def tile(
    positions_angstrom: np.ndarray, edge_angstrom: float, repeats: tuple[int, int, int]
) -> np.ndarray:
    """Periodic images of a wrapped cell, `repeats` copies along x, y and z, origin kept.

    Args:
        positions_angstrom: Wrapped atom positions, shape (n_atoms, 3), in angstrom.
        edge_angstrom: Cubic cell edge length, in angstrom.
        repeats: Number of copies along x, y and z.

    Returns:
        Tiled positions, shape (n_atoms * repeats[0] * repeats[1] * repeats[2], 3), in
        angstrom: images ordered x slowest through z fastest, the input atom order kept
        within each image.
    """
    shifts_angstrom = np.indices(repeats).reshape(3, -1).T * edge_angstrom
    return (positions_angstrom[None, :, :] + shifts_angstrom[:, None, :]).reshape(-1, 3)


def slab_mask(
    depth_angstrom: np.ndarray, centre_angstrom: float, thickness_angstrom: float
) -> np.ndarray:
    """Atoms whose depth lies within `thickness_angstrom` centred on `centre_angstrom`.

    Args:
        depth_angstrom: Depth coordinate (typically z) of each atom, in angstrom.
        centre_angstrom: Centre of the slab along the depth axis, in angstrom.
        thickness_angstrom: Full thickness of the slab, in angstrom.

    Returns:
        Boolean mask, True for atoms within the slab, same shape as `depth_angstrom`.
    """
    return np.abs(depth_angstrom - centre_angstrom) <= thickness_angstrom / 2.0


def make_compact(positions_angstrom: np.ndarray, edge_angstrom: float) -> np.ndarray:
    """Re-image a cluster that a periodic dump has split across the cell boundary.

    Every atom is moved to its minimum image relative to the first atom, then the cluster
    is centred on the origin. Exact for a cluster smaller than half the cell (the largest
    here spans about 25 A in a 64.8 A cell). A dissociated run's fragments land at their
    nearest images, which is the honest picture of what the dump holds.

    Args:
        positions_angstrom: Atom positions, shape (n_atoms, 3), in angstrom.
        edge_angstrom: Cubic cell edge length, in angstrom.

    Returns:
        Positions re-imaged to their minimum image relative to the first atom and
        centred on the origin, same shape as the input.
    """
    delta = positions_angstrom - positions_angstrom[0]
    delta -= edge_angstrom * np.round(delta / edge_angstrom)
    return delta - delta.mean(axis=0)


def find_bonds_periodic(
    positions_angstrom: np.ndarray, edge_angstrom: float
) -> list[tuple[int, int]]:
    """All carbon-carbon pairs closer than the site's cutoff, by minimum-image distance.

    The periodic counterpart of `find_bonds`: the same flat 1.824 A cutoff (strictly
    less than), measured to the nearest periodic image in a cubic cell, so a bond that
    crosses the cell boundary is found wherever the dump or a re-imaging put its two
    atoms. Dense O(n^2) like `find_bonds`; fine to a few thousand atoms.

    Args:
        positions_angstrom: Atom positions, shape (n_atoms, 3), in angstrom. Need not be
            wrapped into the cell.
        edge_angstrom: Cubic cell edge length, in angstrom. Must exceed twice the cutoff.

    Returns:
        Zero-based (first, second) atom index pairs with first < second, in row-major
        order, as `find_bonds` returns them.

    Raises:
        ValueError: If the cell edge is not more than twice the bond cutoff, where the
            minimum image would miss bonds to a second image.
    """
    if edge_angstrom <= 2 * CARBON_BOND_CUTOFF_ANGSTROM:
        raise ValueError(
            f"cell edge {edge_angstrom} A must exceed twice the "
            f"{CARBON_BOND_CUTOFF_ANGSTROM:.3f} A bond cutoff"
        )
    delta_angstrom = positions_angstrom[:, None, :] - positions_angstrom[None, :, :]
    delta_angstrom -= edge_angstrom * np.round(delta_angstrom / edge_angstrom)
    distance_angstrom = np.linalg.norm(delta_angstrom, axis=-1)
    upper = np.triu(np.ones_like(distance_angstrom, dtype=bool), k=1)
    rows, cols = np.where(upper & (distance_angstrom < CARBON_BOND_CUTOFF_ANGSTROM))
    return list(zip(rows.tolist(), cols.tolist(), strict=True))


def make_whole(
    positions_angstrom: np.ndarray, edge_angstrom: float, bonds: list[tuple[int, int]]
) -> np.ndarray:
    """Unwrap every bonded fragment so no bond crosses the periodic boundary.

    Breadth-first over the bond graph, one fragment at a time: the fragment's
    lowest-indexed atom stays where it is, and each newly reached atom is moved to its
    minimum image relative to the atom it was reached from. Every bond then has its
    true length in the returned Cartesian positions. Fragments are unwrapped
    independently, so their placement relative to one another is whatever the input
    gave (for a dissociated run, arbitrary). Each bond must be shorter than half the
    cell, which `find_bonds_periodic` guarantees.

    Args:
        positions_angstrom: Atom positions, shape (n_atoms, 3), in angstrom.
        edge_angstrom: Cubic cell edge length, in angstrom.
        bonds: Zero-based (first, second) atom index pairs, as from `find_bonds_periodic`.

    Returns:
        Positions of the same shape, in angstrom, each fragment contiguous in space; the
        lowest-indexed atom of every fragment is unmoved.

    Raises:
        ValueError: If a bond index falls outside range(len(positions_angstrom)).
    """
    n_atoms = len(positions_angstrom)
    bond_array = np.array(bonds, dtype=int).reshape(-1, 2)
    if bond_array.size and (bond_array.min() < 0 or bond_array.max() >= n_atoms):
        raise ValueError("a bond index falls outside range(len(positions_angstrom))")
    source = np.concatenate([bond_array[:, 0], bond_array[:, 1]])
    target = np.concatenate([bond_array[:, 1], bond_array[:, 0]])
    whole_angstrom = np.array(positions_angstrom, dtype=float, copy=True)
    placed = np.zeros(n_atoms, dtype=bool)
    for root in range(n_atoms):
        if placed[root]:
            continue
        placed[root] = True
        frontier = np.zeros(n_atoms, dtype=bool)
        frontier[root] = True
        while True:
            step = frontier[source] & ~placed[target]
            if not step.any():
                break
            # One parent per newly reached atom: the first bond in array order.
            reached, first = np.unique(target[step], return_index=True)
            parent = source[step][first]
            delta_angstrom = positions_angstrom[reached] - whole_angstrom[parent]
            delta_angstrom -= edge_angstrom * np.round(delta_angstrom / edge_angstrom)
            whole_angstrom[reached] = whole_angstrom[parent] + delta_angstrom
            placed[reached] = True
            frontier[:] = False
            frontier[reached] = True
    return whole_angstrom


def read_lammps_last_frame(path: Path) -> tuple[np.ndarray, float]:
    """Last frame of a LAMMPS `atom`-style dump with scaled `xs ys zs` columns.

    Returns positions in angstrom ordered by atom id, and the cubic cell edge in angstrom.
    The whole file is read (they are about 22 MB) and the last `ITEM: TIMESTEP` block is
    parsed, so a dump that was cut off mid-frame raises rather than returning a partial
    cluster.

    Args:
        path: Path to the LAMMPS dump file.

    Returns:
        A tuple of (positions, edge): positions is an (n_atoms, 3) array in angstrom,
        ordered by atom id; edge is the cubic cell's edge length, in angstrom.

    Raises:
        ValueError: If the file has no frames, does not end with a newline (a dump cut
            off mid-write), has a truncated or malformed header, is missing the
            `id xs ys zs` columns, has a truncated atom block, or has a non-cubic cell.
    """
    text = path.read_text()
    if not text.endswith("\n"):
        raise ValueError(f"{path}: last frame truncated: file does not end with a newline")
    start = text.rfind("ITEM: TIMESTEP")
    if start < 0:
        raise ValueError(f"{path}: no frames found")
    lines = text[start:].splitlines()
    try:
        n_atoms = int(lines[3])
        bounds = np.array([line.split()[:2] for line in lines[5:8]], dtype=float)
        columns = lines[8].split()[2:]
    except (IndexError, ValueError) as error:
        raise ValueError(f"{path}: last frame truncated: incomplete header") from error
    try:
        id_col, x_col, y_col, z_col = (columns.index(name) for name in ("id", "xs", "ys", "zs"))
    except ValueError as error:
        raise ValueError(f"{path}: expected columns id xs ys zs, found {columns}") from error
    rows = [line.split() for line in lines[9 : 9 + n_atoms]]
    if len(rows) != n_atoms or any(len(row) != len(columns) for row in rows):
        raise ValueError(f"{path}: last frame truncated: {len(rows)} of {n_atoms} atoms")
    table = np.array(rows, dtype=float)
    table = table[np.argsort(table[:, id_col])]
    edges = bounds[:, 1] - bounds[:, 0]
    if not np.allclose(edges, edges[0], atol=1e-3):
        raise ValueError(f"{path}: cell is not cubic: {edges}")
    scaled = table[:, [x_col, y_col, z_col]]
    return bounds[:, 0] + scaled * edges, float(edges[0])


def write_pdb(path: Path, positions_angstrom: np.ndarray, bonds: list[tuple[int, int]]) -> None:
    """Carbon-only PDB: one HETATM per atom, serials from 1, and a CONECT line per atom.

    Fixed columns per the PDB format: serial 7-11, name 13-16, residue CBX, chain A,
    coordinates 31-54, element 77-78. Bonds are written in both directions so any
    reader that trusts CONECT sees each once from either end.

    Args:
        path: Output PDB file path.
        positions_angstrom: Atom positions, shape (n_atoms, 3), in angstrom.
        bonds: Zero-based (first, second) atom index pairs, written as CONECT records.

    Raises:
        ValueError: If there are more than 99,999 atoms (the serial field is 5 wide), if
            any coordinate falls outside the PDB's fixed (-999.999, 9999.999) field
            width, or if a bond index falls outside range(len(positions_angstrom)).
    """
    n_atoms = len(positions_angstrom)
    if n_atoms > 99_999:
        raise ValueError(f"{path}: {n_atoms} atoms exceeds the PDB's 99,999-atom serial limit")
    if np.any(positions_angstrom <= -999.999) or np.any(positions_angstrom >= 9999.999):
        raise ValueError(
            f"{path}: a coordinate falls outside the PDB's (-999.999, 9999.999) field width"
        )
    if any(not (0 <= first < n_atoms and 0 <= second < n_atoms) for first, second in bonds):
        raise ValueError(f"{path}: a bond index falls outside range(len(positions_angstrom))")
    lines = [
        f"HETATM{index + 1:5d}  C   CBX A   1    "
        f"{x_angstrom:8.3f}{y_angstrom:8.3f}{z_angstrom:8.3f}  1.00  0.00           C"
        for index, (x_angstrom, y_angstrom, z_angstrom) in enumerate(positions_angstrom)
    ]
    partners: dict[int, list[int]] = {}
    for first, second in bonds:
        partners.setdefault(first, []).append(second)
        partners.setdefault(second, []).append(first)
    for atom in sorted(partners):
        neighbours = partners[atom]
        for chunk_start in range(0, len(neighbours), 4):
            chunk = neighbours[chunk_start : chunk_start + 4]
            lines.append(
                f"CONECT{atom + 1:5d}" + "".join(f"{neighbour + 1:5d}" for neighbour in chunk)
            )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
