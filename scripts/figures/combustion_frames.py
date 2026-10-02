# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Pull frames out of the CHO-GAP methane-combustion trajectory as unwrapped XYZ.

The run is 8 CH4 + 16 O2 in a 10 A periodic cube at 3000 K, driven by CHO-GAP
through LAMMPS' `pair_style quip`. LAMMPS dumps scaled coordinates wrapped into
the box, so molecules that straddle a face come out cut in half; rendering that
directly draws bonds across the whole cell. Each connected molecule is therefore
rebuilt contiguously here, before anything is drawn.

The same connectivity gives the species count, which is the actual result: what
the run was for is whether a potential fitted to condensed-phase carbon does
combustion chemistry it was never shown.

Usage:
    uv run scripts/figures/combustion_frames.py <combustion.lammpstrj> \
        --out-dir scripts/figures/data/cho-gap --picoseconds 0,10,20,40
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from pathlib import Path

import numpy as np
from constants import BOND_TOLERANCE, COVALENT_RADIUS_ANGSTROM

# `pair_coeff * * <model>.xml "" 6 1 8` in in.cho_opt maps LAMMPS types 1, 2, 3
# onto Z = 6, 1, 8. The Masses block of combustion_chamber.data agrees.
TYPE_TO_SPECIES = {1: "C", 2: "H", 3: "O"}

# The bond rule (Cordero covalent radii, tolerance) is constants.py's, the one
# render_cluster.py draws with, so the bonds counted here are the bonds drawn there.

# LAMMPS `timestep 0.0005` (metal units, ps) with `dump ... 20`.
TIMESTEP_PICOSECONDS = 0.0005
DUMP_EVERY_STEPS = 20

# Hill-ish ordering, so H2O reads as H2O rather than OH2.
_FORMULA_ORDER = ["C", "H", "O"]


def read_frames(path: Path, wanted_steps: set[int]) -> dict[int, tuple[list[str], np.ndarray, float]]:
    """Stream the dump, returning {timestep: (species, cartesian positions, box length)}.

    The file is tens of megabytes and only a handful of frames are wanted, so it
    is walked line by line rather than loaded.
    """
    found: dict[int, tuple[list[str], np.ndarray, float]] = {}
    with path.open() as handle:
        while found.keys() != wanted_steps:
            line = handle.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue

            step = int(handle.readline())
            handle.readline()  # ITEM: NUMBER OF ATOMS
            n_atoms = int(handle.readline())
            handle.readline()  # ITEM: BOX BOUNDS
            bounds = [tuple(float(v) for v in handle.readline().split()) for _ in range(3)]
            handle.readline()  # ITEM: ATOMS id type xs ys zs

            if step not in wanted_steps:
                for _ in range(n_atoms):
                    handle.readline()
                continue

            lengths = [hi - lo for lo, hi in bounds]
            if not np.allclose(lengths, lengths[0]):
                raise ValueError(f"{path}: non-cubic box {lengths}, unwrapping assumes a cube")
            box = lengths[0]

            # Atoms are dumped in whatever order the neighbour list holds them,
            # so they are sorted back into id order for reproducible output.
            rows = []
            for _ in range(n_atoms):
                atom_id, atom_type, x, y, z = handle.readline().split()
                rows.append((int(atom_id), int(atom_type), float(x), float(y), float(z)))
            rows.sort()

            species = [TYPE_TO_SPECIES[row[1]] for row in rows]
            scaled = np.array([[row[2], row[3], row[4]] for row in rows])
            found[step] = (species, scaled * box, box)

    missing = wanted_steps - found.keys()
    if missing:
        raise ValueError(f"{path}: timesteps {sorted(missing)} not in the trajectory")
    return found


def minimum_image_bonds(
    positions_angstrom: np.ndarray, species: list[str], box: float
) -> list[tuple[int, int]]:
    """Bonded pairs under the minimum-image convention."""
    deltas = positions_angstrom[:, None, :] - positions_angstrom[None, :, :]
    deltas -= box * np.round(deltas / box)
    distances = np.linalg.norm(deltas, axis=-1)

    radii = np.array([COVALENT_RADIUS_ANGSTROM[s] for s in species])
    cutoffs = BOND_TOLERANCE * (radii[:, None] + radii[None, :])
    upper = np.triu(np.ones_like(distances, dtype=bool), k=1)
    rows, cols = np.where(upper & (distances < cutoffs))
    return list(zip(rows.tolist(), cols.tolist()))


def molecules(n_atoms: int, bonds: list[tuple[int, int]]) -> list[list[int]]:
    """Connected components of the bond graph, each sorted by atom index."""
    adjacency: dict[int, list[int]] = {i: [] for i in range(n_atoms)}
    for i, j in bonds:
        adjacency[i].append(j)
        adjacency[j].append(i)

    seen: set[int] = set()
    components: list[list[int]] = []
    for start in range(n_atoms):
        if start in seen:
            continue
        component = []
        queue = deque([start])
        seen.add(start)
        while queue:
            atom = queue.popleft()
            component.append(atom)
            for neighbour in adjacency[atom]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
        components.append(sorted(component))
    return components


def unwrap(
    positions_angstrom: np.ndarray, bonds: list[tuple[int, int]], groups: list[list[int]], box: float
) -> np.ndarray:
    """Rebuild each molecule contiguously, then re-centre it inside the box.

    Walking the bond graph and placing every atom at the minimum image of its
    parent makes a molecule whole. Re-wrapping its centroid afterwards keeps the
    frame looking like a box of gas rather than a smear, without ever splitting a
    molecule again.
    """
    adjacency: dict[int, list[int]] = {i: [] for i in range(len(positions_angstrom))}
    for i, j in bonds:
        adjacency[i].append(j)
        adjacency[j].append(i)

    unwrapped = positions_angstrom.copy()
    for group in groups:
        placed = {group[0]}
        queue = deque([group[0]])
        while queue:
            atom = queue.popleft()
            for neighbour in adjacency[atom]:
                if neighbour in placed:
                    continue
                offset = unwrapped[neighbour] - unwrapped[atom]
                unwrapped[neighbour] -= box * np.round(offset / box)
                placed.add(neighbour)
                queue.append(neighbour)

        centroid = unwrapped[group].mean(axis=0)
        unwrapped[group] += box * np.floor(centroid / box) * -1.0
    return unwrapped


def formula(species: list[str], group: list[int]) -> str:
    counts = Counter(species[i] for i in group)
    parts = []
    for element in _FORMULA_ORDER:
        if counts[element]:
            parts.append(element + (str(counts[element]) if counts[element] > 1 else ""))
    return "".join(parts)


def write_xyz(path: Path, species: list[str], positions_angstrom: np.ndarray, comment: str) -> None:
    lines = [str(len(species)), comment]
    for element, (x, y, z) in zip(species, positions_angstrom):
        lines.append(f"{element} {x:12.6f} {y:12.6f} {z:12.6f}")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--picoseconds",
        default="0,10,20,40",
        help="Comma-separated times to extract, in ps.",
    )
    args = parser.parse_args()

    times_ps = [float(v) for v in args.picoseconds.split(",")]
    step_of = {
        t: int(round(t / TIMESTEP_PICOSECONDS / DUMP_EVERY_STEPS)) * DUMP_EVERY_STEPS
        for t in times_ps
    }
    frames = read_frames(args.trajectory, set(step_of.values()))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for time_ps in times_ps:
        species, positions_angstrom, box = frames[step_of[time_ps]]
        bonds = minimum_image_bonds(positions_angstrom, species, box)
        groups = molecules(len(species), bonds)
        rebuilt = unwrap(positions_angstrom, bonds, groups, box)

        tally = Counter(formula(species, group) for group in groups)
        summary = ", ".join(f"{n}x{f}" for f, n in tally.most_common())

        out = args.out_dir / f"combustion-{time_ps:g}ps.xyz"
        write_xyz(
            out,
            species,
            rebuilt,
            f"t = {time_ps:g} ps, {len(groups)} molecules: {summary}",
        )
        print(f"{time_ps:5g} ps  {len(groups):3d} molecules  {summary}")
        print(f"           -> {out}")


def _self_check() -> None:
    """Unwrapping and species counting, on a molecule deliberately split by the box."""
    box = 10.0
    # A water molecule straddling the x face: O at 9.9, both H wrapped round to
    # the low side. Wrapped separations are ~9 A; minimum image makes them ~1 A.
    species = ["O", "H", "H"]
    wrapped = np.array([[9.9, 5.0, 5.0], [0.86, 5.0, 5.0], [9.66, 5.93, 5.0]])
    bonds = minimum_image_bonds(wrapped, species, box)
    assert len(bonds) == 2, bonds

    groups = molecules(3, bonds)
    assert groups == [[0, 1, 2]], groups
    assert formula(species, groups[0]) == "H2O"

    rebuilt = unwrap(wrapped, bonds, groups, box)
    o_h_1 = np.linalg.norm(rebuilt[1] - rebuilt[0])
    o_h_2 = np.linalg.norm(rebuilt[2] - rebuilt[0])
    assert 0.9 < o_h_1 < 1.1, o_h_1  # the bond is ~0.96 A once unwrapped
    assert 0.9 < o_h_2 < 1.1, o_h_2
    # ... and the molecule ends up inside the box rather than hanging off it.
    assert (rebuilt.min(axis=0) > -box).all() and (rebuilt.max(axis=0) < 2 * box).all()

    # Two atoms far apart are two molecules, not one.
    far = np.array([[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]])
    assert len(molecules(2, minimum_image_bonds(far, ["C", "O"], box))) == 2

    print("self-check ok")


if __name__ == "__main__":
    import sys

    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()
