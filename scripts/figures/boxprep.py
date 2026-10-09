# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Geometry for rendering periodic carbon boxes and free carbon clusters.

The archive's final frames are plain XYZ with no cell line and the cluster dumps are
LAMMPS `atom`-style with scaled coordinates. This module reads structures, recovers
the cell, folds and tiles the box, cuts the slab a camera can see into, re-images a
cluster the periodic dump has split, and writes a PDB with explicit CONECT records,
because Molecular Nodes reads bonds from CONECT and infers none for a residue it does
not know (docs/dossiers/carbon/molrender-api.md section 10).

`Structure`, `read_xyz` and `find_bonds` are the shared geometry primitives: this
module uses them to prepare boxes for the Blender driver (`render_box_grid.py`), and
`render_cluster.py` imports them back for its own SVG drawing, so both renderers read
and bond structures the same way.

Pure numpy. The dense bond search (`find_bonds`) is fine for the largest thing
rendered here, a two-thousand-atom slab; a whole 5,832-atom box is never bonded, it is
sliced first.

The bond rule (covalent radii, tolerance, the 1.824 A carbon cutoff) and the physical
constants come from constants.py.

Units: every length is in angstrom, every density in g cm^-3.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
from constants import (
    AVOGADRO_PER_MOL,
    BARE_IONS,
    BOND_TOLERANCE,
    CARBON_BOND_CUTOFF_ANGSTROM,
    CARBON_MASS_G_PER_MOL,
    CM_TO_ANGSTROM,
    COVALENT_RADIUS_ANGSTROM,
    LAYER_METALS,
)

__all__ = [
    "ProteinStructure",
    "Structure",
    "box_edge_from_density",
    "find_bonds",
    "find_bonds_periodic",
    "make_compact",
    "make_whole",
    "read_lammps_last_frame",
    "read_pdb",
    "read_xyz",
    "slab_mask",
    "tile",
    "wrap",
    "write_pdb",
]


class Structure(NamedTuple):
    """One atomic configuration. Positions are (n, 3) in Angstrom."""

    species: list[str]
    positions_angstrom: np.ndarray

    def __len__(self) -> int:
        return len(self.species)


class ProteinStructure(NamedTuple):
    """The atoms of a PDB file that a protein render draws, in the file's own order.

    Every per-atom field has one entry per atom, in the same order. `records` and
    `secondary_structure_records` are the file's own lines, kept so the renderer can write
    exactly these atoms back out: positions in angstrom. `residue_numbers` alone does not
    tell residue 111 from 111A; `insertion_codes` ("" when blank) does. The two
    `n_dropped_*` counts are of the atom records `read_pdb` left out of model 1.
    """

    species: list[str]
    positions_angstrom: np.ndarray
    chain_ids: np.ndarray
    residue_numbers: np.ndarray
    insertion_codes: np.ndarray
    is_hetero: np.ndarray
    records: list[str]
    secondary_structure_records: list[str]
    n_dropped_altloc: int
    n_dropped_water: int

    def __len__(self) -> int:
        return len(self.species)

    def drawn_by_species(self, stick_chains: list[str]) -> np.ndarray:
        """Per atom, whether a protein render draws it by species (sticks and spheres).

        These are the atoms of `stick_chains` and every HETATM (water is already gone); the
        atoms of the other chains are the Cartoon's. `render_box_grid.scene_for` builds its
        species layers from this mask and `render_panel` decides the PNG's colour mode from
        it.

        Args:
            stick_chains: Chain ids drawn as sticks and spheres.

        Returns:
            Boolean array, shape (n_atoms,).
        """
        return np.isin(self.chain_ids, stick_chains) | self.is_hetero


def read_pdb(path: Path) -> ProteinStructure:
    """Read the atoms of a PDB file that a render draws: model 1, one conformer, no water.

    Atoms are the ATOM and HETATM records of the first model. An atom whose alternate
    location (column 17) is anything but blank or "A" is dropped, as is every `HOH`
    residue; both counts are returned. The element is columns 77-78 ("MG" reads "Mg"); a
    record without one takes it from columns 13-14 of the atom name, where the PDB format
    right-justifies it (an ATOM record keeps only the first letter, because protein atoms
    are H, C, N, O and S). Chain id is column 22, residue number columns 23-26, insertion
    code column 27, hetero flag is the record name. ANISOU, CONECT, TER and every other
    record are ignored, except HELIX and SHEET, which Molecular Nodes reads for the
    cartoon's secondary structure and which `ProteinStructure.secondary_structure_records`
    returns.

    `records` are the kept ATOM and HETATM lines verbatim, so writing them (render_box_grid.prepare
    does) gives a file whose atoms are exactly those this function returns, in the same
    order, with every column, the insertion code in column 27 included: the species and
    chain masks the renderer measures here are then indexed by the atoms Molecular Nodes
    loads from that file.

    Insertion codes: Molecular Nodes numbers residues by changes of residue number and
    chain only (`ures_id`), so an insertion-coded residue (6TRO chain E has 111A and 112A)
    may be merged with the residue before it in the cartoon's residue ids. The masks here
    are by chain and are unaffected; whether the merge shows in a render is for the design
    review to look at on the picture.

    Args:
        path: Path to the PDB file.

    Returns:
        The structure. Positions are in angstrom.

    Raises:
        ValueError: If the file has no atom left to draw, or the coordinates (columns
            31-54) or residue number (23-26) of an atom record are not numbers; the message
            names the file and the line.
    """
    lines = path.read_text().splitlines()
    kept: list[tuple[int, str]] = []
    n_dropped_altloc = n_dropped_water = 0
    for line_number, line in enumerate(lines, start=1):
        if line.startswith("ENDMDL"):  # model 1 only
            break
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if line[17:20] == "HOH":
            n_dropped_water += 1
        elif line[16:17] not in " A":  # alternate location: blank, or the first conformer
            n_dropped_altloc += 1
        else:
            kept.append((line_number, line))
    if not kept:
        raise ValueError(f"{path}: no atoms to draw (model 1, altloc blank or A, no HOH)")

    species: list[str] = []
    positions_angstrom: list[list[float]] = []
    residue_numbers: list[int] = []
    for line_number, line in kept:
        try:
            positions_angstrom.append([float(line[start : start + 8]) for start in (30, 38, 46)])
            residue_numbers.append(int(line[22:26]))
        except ValueError as error:
            raise ValueError(
                f"{path}: line {line_number}: coordinates (columns 31-54) or residue number "
                f"(columns 23-26) are not numbers: {line[:54]!r}"
            ) from error
        element = line[76:78].strip().title()
        if not element:
            element = line[12:14].strip(" 0123456789").title()
            if line.startswith("ATOM"):
                element = element[:1]
        species.append(element)
    atom_lines = [line for _, line in kept]
    return ProteinStructure(
        species=species,
        positions_angstrom=np.array(positions_angstrom),
        chain_ids=np.array([line[21] for line in atom_lines]),
        residue_numbers=np.array(residue_numbers),
        insertion_codes=np.array([line[26:27].strip() for line in atom_lines]),
        is_hetero=np.array([line.startswith("HETATM") for line in atom_lines]),
        records=atom_lines,
        secondary_structure_records=[line for line in lines if line.startswith(("HELIX", "SHEET"))],
        n_dropped_altloc=n_dropped_altloc,
        n_dropped_water=n_dropped_water,
    )


def read_xyz(path: Path, frame: int = -1) -> Structure:
    """Read one frame of an XYZ or extended-XYZ file.

    The search output carries trailing bookkeeping columns (index, neighbour list)
    after the coordinates, so only fields 0-3 are read. Trajectories are concatenated
    frames; `frame` indexes them and defaults to the last, which is the relaxed or
    equilibrated structure in everything this renders.

    Args:
        path: Path to the XYZ file.
        frame: Zero-based frame index; negative indexes from the end. Defaults to the
            last frame.

    Returns:
        The structure at that frame.

    Raises:
        ValueError: If the file holds no frames, a frame's atom count does not match
            its header, or an atom's species has no entry in COVALENT_RADIUS_ANGSTROM.
    """
    lines = path.read_text().splitlines()

    frames: list[tuple[int, int]] = []  # (first atom line, atom count)
    cursor = 0
    while cursor < len(lines):
        header = lines[cursor].split()
        if not header:  # trailing blank lines
            break
        n_atoms = int(header[0])
        frames.append((cursor + 2, n_atoms))
        cursor += 2 + n_atoms
    if not frames:
        raise ValueError(f"{path}: no frames found")

    start, n_atoms = frames[frame]
    species: list[str] = []
    coords: list[list[float]] = []
    for line in lines[start : start + n_atoms]:
        fields = line.split()
        species.append(fields[0])
        coords.append([float(fields[1]), float(fields[2]), float(fields[3])])

    positions_angstrom = np.asarray(coords, dtype=float)
    if positions_angstrom.shape != (n_atoms, 3):
        raise ValueError(f"{path}: expected {n_atoms} atoms, parsed {positions_angstrom.shape[0]}")

    unknown = set(species) - COVALENT_RADIUS_ANGSTROM.keys()
    if unknown:
        raise ValueError(
            f"{path}: no covalent radius for {sorted(unknown)}. Add it to "
            f"COVALENT_RADIUS_ANGSTROM in constants.py, and to DRAW_RADIUS_FACTOR in "
            f"render_cluster.py if it will be drawn."
        )
    return Structure(species, positions_angstrom)


def find_bonds(
    positions_angstrom: np.ndarray, species: list[str] | None = None
) -> list[tuple[int, int]]:
    """All atom pairs closer than BOND_TOLERANCE times their summed covalent radii.

    Two kinds of pair are never bonded, however close, because the radius rule would draw
    contacts that are not covalent: any pair with a partner in constants.BARE_IONS
    (intercalated or interlayer ions), and any pair of two constants.LAYER_METALS atoms
    (the metal-metal distance in a layer, 3.16 A in MoS2, is inside the cutoff). The
    radius rule decides every other pair.

    O(n^2) on the distance matrix. The largest cluster this renders is 720 atoms, so
    this is well under a second and a neighbour list would be premature.

    Args:
        positions_angstrom: Atom positions, shape (n_atoms, 3), in angstrom.
        species: Element symbol per atom, or None for all-carbon, which keeps the
            cutoff at the flat 1.82 A the carbon-only figures were drawn with.

    Returns:
        Zero-based (first, second) atom index pairs with first < second, in row-major
        order.
    """
    if species is None:
        species = ["C"] * len(positions_angstrom)
    radii = np.array([COVALENT_RADIUS_ANGSTROM[s] for s in species])
    cutoffs = BOND_TOLERANCE * (radii[:, None] + radii[None, :])

    deltas = positions_angstrom[:, None, :] - positions_angstrom[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    upper = np.triu(np.ones_like(distances, dtype=bool), k=1)
    is_ion = np.isin(species, list(BARE_IONS))
    is_layer_metal = np.isin(species, list(LAYER_METALS))
    never_bonded = (
        is_ion[:, None] | is_ion[None, :] | (is_layer_metal[:, None] & is_layer_metal[None, :])
    )
    rows, cols = np.where(upper & (distances < cutoffs) & ~never_bonded)
    return list(zip(rows.tolist(), cols.tolist(), strict=True))


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


def minimum_image(delta_angstrom: np.ndarray, edge_angstrom: float) -> np.ndarray:
    """Fold displacement vectors into the cubic cell's minimum image.

    Args:
        delta_angstrom: Displacements, any shape with the last axis of length 3, in Å.
        edge_angstrom: Cubic cell edge in Å.

    Returns:
        The displacements with every component moved into (-edge/2, edge/2], in Å.
    """
    return delta_angstrom - edge_angstrom * np.round(delta_angstrom / edge_angstrom)


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
    delta = minimum_image(positions_angstrom - positions_angstrom[0], edge_angstrom)
    return delta - delta.mean(axis=0)


def find_bonds_periodic(
    positions_angstrom: np.ndarray,
    edge_angstrom: float,
    cutoff_angstrom: float = CARBON_BOND_CUTOFF_ANGSTROM,
) -> list[tuple[int, int]]:
    """All carbon-carbon pairs closer than the cutoff, by minimum-image distance.

    The periodic counterpart of `find_bonds`: by default the same flat 1.824 A cutoff
    (strictly less than), measured to the nearest periodic image in a cubic cell, so a
    bond that crosses the cell boundary is found wherever the dump or a re-imaging put
    its two atoms. Dense O(n^2) like `find_bonds`; fine to a few thousand atoms.

    Args:
        positions_angstrom: Atom positions, shape (n_atoms, 3), in angstrom. Need not be
            wrapped into the cell.
        edge_angstrom: Cubic cell edge length, in angstrom. Must exceed twice the cutoff.
        cutoff_angstrom: Bond cutoff, in angstrom; the site's carbon-carbon cutoff by
            default.

    Returns:
        Zero-based (first, second) atom index pairs with first < second, in row-major
        order, as `find_bonds` returns them.

    Raises:
        ValueError: If the cell edge is not more than twice the bond cutoff, where the
            minimum image would miss bonds to a second image.
    """
    if edge_angstrom <= 2 * cutoff_angstrom:
        raise ValueError(
            f"cell edge {edge_angstrom} A must exceed twice the {cutoff_angstrom:.3f} A bond cutoff"
        )
    delta_angstrom = minimum_image(
        positions_angstrom[:, None, :] - positions_angstrom[None, :, :], edge_angstrom
    )
    distance_angstrom = np.linalg.norm(delta_angstrom, axis=-1)
    upper = np.triu(np.ones_like(distance_angstrom, dtype=bool), k=1)
    rows, cols = np.where(upper & (distance_angstrom < cutoff_angstrom))
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
        ValueError: If a bond index falls outside range(len(positions_angstrom)), or if a
            fragment is bonded to its own periodic image (it spans the whole cell, so no
            placement gives every bond its true length).
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
            delta_angstrom = minimum_image(
                positions_angstrom[reached] - whole_angstrom[parent], edge_angstrom
            )
            whole_angstrom[reached] = whole_angstrom[parent] + delta_angstrom
            placed[reached] = True
            frontier[:] = False
            frontier[reached] = True
    bond_vectors = whole_angstrom[bond_array[:, 1]] - whole_angstrom[bond_array[:, 0]]
    if np.any(np.abs(bond_vectors) > edge_angstrom / 2):
        raise ValueError("a fragment is bonded to its own periodic image: it spans the cell")
    return whole_angstrom


def read_lammps_last_frame(path: Path) -> tuple[np.ndarray, float, int]:
    """Last frame of a LAMMPS `atom`-style dump with scaled `xs ys zs` columns.

    Returns positions in angstrom ordered by atom id, the cubic cell edge in angstrom and
    the frame's timestep.
    The whole file is read (they are about 22 MB) and the last `ITEM: TIMESTEP` block is
    parsed, so a dump that was cut off mid-frame raises rather than returning a partial
    cluster.

    Args:
        path: Path to the LAMMPS dump file.

    Returns:
        A tuple of (positions, edge, timestep): positions is an (n_atoms, 3) array in
        angstrom, ordered by atom id; edge is the cubic cell's edge length, in angstrom;
        timestep is the last frame's `ITEM: TIMESTEP` value.

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
        timestep = int(lines[1])
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
    return bounds[:, 0] + scaled * edges, float(edges[0]), timestep


def write_pdb(
    path: Path,
    species: list[str],
    positions_angstrom: np.ndarray,
    bonds: list[tuple[int, int]],
) -> None:
    """PDB of any species the bond rule knows: one HETATM per atom and a CONECT line per atom.

    Fixed columns per the PDB format: serial 7-11, name 13-16, residue CBX, chain A,
    coordinates 31-54, element 77-78. The element symbol is upper case, as the wwPDB
    writes it ("MO"), right-justified in 77-78, and is also the atom name: starting in
    column 14 for a one-letter element and column 13 for a two-letter one. Bonds are
    written in both directions so any reader that trusts CONECT sees each once from
    either end. The species are checked here rather than trusted to the caller, so a
    symbol the bond rule has no radius for stops the write instead of rendering as
    something else.

    Args:
        path: Output PDB file path.
        species: Element symbol per atom ("O", "Mo"); each must be a key of
            COVALENT_RADIUS_ANGSTROM.
        positions_angstrom: Atom positions, shape (n_atoms, 3), in angstrom.
        bonds: Zero-based (first, second) atom index pairs, written as CONECT records.

    Raises:
        ValueError: If any species has no entry in COVALENT_RADIUS_ANGSTROM (the message
            names it), the species and positions differ in number, there are more than
            99,999 atoms (the serial field is 5 wide), any coordinate falls outside the
            PDB's fixed (-999.999, 9999.999) field width, or a bond index falls outside
            range(len(positions_angstrom)).
    """
    n_atoms = len(positions_angstrom)
    if len(species) != n_atoms:
        raise ValueError(f"{path}: {len(species)} species for {n_atoms} positions")
    unknown = sorted(set(species) - COVALENT_RADIUS_ANGSTROM.keys())
    if unknown:
        raise ValueError(
            f"{path}: write_pdb has no covalent radius for {unknown}. Add it to "
            f"COVALENT_RADIUS_ANGSTROM in constants.py."
        )
    if n_atoms > 99_999:
        raise ValueError(f"{path}: {n_atoms} atoms exceeds the PDB's 99,999-atom serial limit")
    if np.any(positions_angstrom <= -999.999) or np.any(positions_angstrom >= 9999.999):
        raise ValueError(
            f"{path}: a coordinate falls outside the PDB's (-999.999, 9999.999) field width"
        )
    if any(not (0 <= first < n_atoms and 0 <= second < n_atoms) for first, second in bonds):
        raise ValueError(f"{path}: a bond index falls outside range(len(positions_angstrom))")
    lines = [
        f"HETATM{index + 1:5d} {symbol.upper().rjust(2):<4} CBX A   1    "
        f"{x_angstrom:8.3f}{y_angstrom:8.3f}{z_angstrom:8.3f}  1.00  0.00          "
        f"{symbol.upper():>2}"
        for index, (symbol, (x_angstrom, y_angstrom, z_angstrom)) in enumerate(
            zip(species, positions_angstrom, strict=True)
        )
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
