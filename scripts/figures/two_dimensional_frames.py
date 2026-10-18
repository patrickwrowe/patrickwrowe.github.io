# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Stage the input frames of the two-dimensional-and-confined renders from their sources.

Three sources, none of which lives in the repository; name any of them and those frames are
staged:

- `--vermiculite`: the Li-vermiculite AIMD trajectory `Li_AIMD_Total.pdb` (CP2K PDB, on the
  read-only backup drive under
  `Vermiculite_Water_Structure/CP2K/AIMD/17_Keyed_Surface_Analysis/Li/`). Its last frame is
  taken. CP2K writes unwrapped positions, so the frame is wrapped into the cell in plane
  only: a and b of its CRYST1 cell lie along x and y (alpha = gamma = 90), so x mod a and
  y mod b are lattice translations, and every hydrogen moves with the oxygen nearest to it
  (by the in-plane minimum image), so no water or hydroxyl is cut. The vacuum direction is
  left alone. The wrapped cell is then tiled VERMICULITE_REPEATS times in plane,
  periodic images of the one simulated cell, as Patrick's own 3 x 3 snapshot
  (`snapframe.POSCAR` beside the trajectory) was. The deuterium of the D2O is labelled H in
  the source and stays H here.
- `--mos2`: the Li/Mo = 0.3 1T' MoS2 AIMD trajectory `MoS2_Water-pos-1.pdb` (CP2K PDB, under
  `Molybdenum_Disulfide_Water_Flow/AIMD_2019_New_Study/4_Multi_Li_CP2K_AIMD/
  2_400_K_Production_Run/4_5_nm_Li_Sep/` on the same drive). Its last frame is taken and
  written as the water and lithium between two sheets (`stage_mos2`): the cell holds one
  sheet, periodic along z, so the second sheet is its image one c away.
- `--hbn-training`: `hBN_GAP_Training.xyz` from the deposit doi 10.17863/CAM.66112 (CC BY 4.0).
  The frames HBN_FRAMES names are copied line for line (count, extended-XYZ comment, atom
  lines with their forces). They lie inside their cells, but the cell boundary along z (the
  c vector is along z in every frame) can split a ring of atoms across it: the tube's end ring
  sat half at z = 0 and half at z = c, which drew one atom on a single bond above the tube.
  So the cell is cut along z at the widest gap between atoms (`widest_gap_shift`) and the
  atoms below the cut move up by c, a lattice translation; only those lines change, in
  their z field. A slab, whose widest gap is its vacuum, is left as it is.

Each trajectory is streamed once (about 1 GB on a slow drive): its md5,
frame count and last frame come from the same read. What each output came from is printed;
`scripts/figures/data/two-dimensional-and-confined/provenance.json` records it.

Usage:
    .venv/bin/python scripts/figures/two_dimensional_frames.py \
        [--vermiculite <.../17_Keyed_Surface_Analysis/Li/Li_AIMD_Total.pdb>] \
        [--mos2 <.../4_5_nm_Li_Sep/MoS2_Water-pos-1.pdb>] \
        [--hbn-training <dir>/hBN_GAP/hBN_GAP_Training.xyz]
"""

from __future__ import annotations

import argparse
import hashlib
from collections import deque
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "data/two-dimensional-and-confined"
VERMICULITE_OUT = "vermiculite-li-last-frame-3x1.xyz"
# Three cells along a, one along b: the film's whole depth stays visible from the side.
VERMICULITE_REPEATS = (3, 1)
MOS2_OUT = "mos2-li-0.3-last-frame-two-sheets.xyz"
# Zero-based frame index in hBN_GAP_Training.xyz -> output name, chosen by inspection
# (provenance.json says why).
HBN_FRAMES = {347: "hbn-nanotube-6-6-frame-347.xyz", 218: "hbn-bilayer-frame-218.xyz"}


def last_pdb_frame(path: Path) -> tuple[list[str], int, str]:
    """Stream a CP2K PDB trajectory: its last frame's lines, its frame count and its md5.

    A frame is a REMARK line, a CRYST1 line, the ATOM records and END.

    Raises:
        ValueError: If the file holds no frame.
    """
    digest = hashlib.md5()
    frame: deque[str] = deque()
    n_frames = 0
    with path.open("rb") as handle:
        for raw in handle:
            digest.update(raw)
            line = raw.decode()
            if line.startswith("REMARK"):
                n_frames += 1
                frame.clear()
            frame.append(line.rstrip("\n"))
    if not n_frames:
        raise ValueError(f"{path}: no REMARK line, so no frame")
    return list(frame), n_frames, digest.hexdigest()


def wrap_with_hydrogens(
    species: list[str], positions_angstrom: np.ndarray, edges_angstrom: tuple[float, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Wrap a frame into its orthorhombic cell along its periodic axes, each H with its O.

    The periodic axes are the first len(edges_angstrom) of x, y, z, each along its cell edge.
    Each hydrogen is placed beside the oxygen nearest to it in space, at that oxygen's minimum
    image; then every other atom is wrapped to [0, edge) on each periodic axis and its
    hydrogens move with it, so no water or hydroxyl is cut.

    Args:
        species: Element symbol per atom.
        positions_angstrom: Shape (n_atoms, 3), in angstrom.
        edges_angstrom: The cell edge along each periodic axis, in angstrom.

    Returns:
        The wrapped positions (angstrom), and for each atom the index of the atom it moved
        with: its oxygen for a hydrogen, itself otherwise.
    """
    periodic = len(edges_angstrom)
    edges = np.asarray(edges_angstrom, dtype=float)
    is_h = np.array(species) == "H"
    oxygens = np.flatnonzero(np.array(species) == "O")
    positions = positions_angstrom.copy()
    delta = positions[is_h, None, :] - positions[None, oxygens, :]
    delta[..., :periodic] -= edges * np.round(delta[..., :periodic] / edges)
    closest = np.argmin(np.linalg.norm(delta, axis=-1), axis=1)
    anchor = np.arange(len(species))
    anchor[is_h] = oxygens[closest]
    positions[is_h] = positions[anchor[is_h]] + delta[np.arange(len(closest)), closest]
    positions[:, :periodic] -= edges * np.floor(positions[anchor, :periodic] / edges)
    return positions, anchor


def tile_in_plane(
    species: list[str],
    positions_angstrom: np.ndarray,
    a_angstrom: float,
    b_angstrom: float,
    repeats: tuple[int, int],
) -> tuple[list[str], np.ndarray]:
    """Wrap a slab into its rectangular in-plane cell, H with its O, and tile it in plane.

    Args:
        species: Element symbol per atom.
        positions_angstrom: Shape (n_atoms, 3), in angstrom.
        a_angstrom, b_angstrom: The in-plane cell edges along x and y, in angstrom.
        repeats: Images along x and y.

    Returns:
        The species and positions (angstrom) of the tiled slab, image by image.
    """
    positions, _ = wrap_with_hydrogens(species, positions_angstrom, (a_angstrom, b_angstrom))
    images = [(i, j) for i in range(repeats[0]) for j in range(repeats[1])]
    tiled = np.concatenate(
        [positions + np.array([i * a_angstrom, j * b_angstrom, 0.0]) for i, j in images]
    )
    return species * len(images), tiled


def read_cp2k_frame(path: Path) -> tuple[str, list[float], list[str], np.ndarray, int, str]:
    """The last frame of a CP2K PDB trajectory, with the trajectory's frame count and md5.

    Returns:
        The frame's step ("Step 29501"), its CRYST1 a, b, c (angstrom) and alpha, beta, gamma
        (degrees), each atom's element symbol (columns 77-78; CP2K's "MO" read as "Mo"), the
        positions in angstrom, the trajectory's frame count and its md5.
    """
    frame, n_frames, md5 = last_pdb_frame(path)
    step = frame[0].removeprefix("REMARK").split(",")[0].strip()
    cell = [float(value) for value in frame[1].split()[1:7]]
    atoms = [line for line in frame if line.startswith("ATOM")]
    species = [line[76:78].strip().capitalize() for line in atoms]
    positions = np.array([[float(line[i : i + 8]) for i in (30, 38, 46)] for line in atoms])
    return step, cell, species, positions, n_frames, md5


def write_xyz(out: Path, species: list[str], positions_angstrom: np.ndarray, comment: str) -> None:
    rows = [
        f"{element:2s} {x:12.6f} {y:12.6f} {z:12.6f}"
        for element, (x, y, z) in zip(species, positions_angstrom, strict=True)
    ]
    out.write_text("\n".join([str(len(rows)), comment, *rows]) + "\n")
    print(f"{out.name}: {comment}")


def stage_vermiculite(path: Path) -> None:
    step, cell, species, positions, n_frames, md5 = read_cp2k_frame(path)
    a, b, c, alpha, beta, gamma = cell
    if alpha != 90.0 or gamma != 90.0:
        raise ValueError(f"{path}: cell {cell}: a and b are not along x and y")
    tiled_species, tiled = tile_in_plane(species, positions, a, b, VERMICULITE_REPEATS)
    nx, ny = VERMICULITE_REPEATS
    comment = (
        f"{path.name} frame {n_frames} of {n_frames} ({step}), {len(species)} atoms, cell "
        f"{a} {b} {c} {alpha} {beta} {gamma}, wrapped in plane and tiled {nx} x {ny}"
    )
    write_xyz(OUT_DIR / VERMICULITE_OUT, tiled_species, tiled, comment)
    print(f"source md5 {md5}")


def stage_mos2(path: Path) -> None:
    """The last frame as two MoS2 sheets with the cell's water and lithium between them.

    The cell is orthorhombic and periodic along z as well: it holds one sheet, and the next
    sheet is that sheet's image one c along z. The frame is wrapped on all three axes, each H
    with its O; along z the cell is cut at the widest gap between heavy atoms (between the
    water and the sheet's outer sulfur), so the sheet lies wholly on one side of the water;
    then a copy of the sheet (its Mo and S) is added one c away on the other side.
    """
    step, cell, species, positions, n_frames, md5 = read_cp2k_frame(path)
    a, b, c = cell[:3]
    if cell[3:] != [90.0, 90.0, 90.0]:
        raise ValueError(f"{path}: cell {cell} is not orthorhombic")
    positions, anchor = wrap_with_hydrogens(species, positions, (a, b, c))
    symbols = np.array(species)
    heavy = symbols != "H"
    shift = np.zeros(len(species))
    shift[heavy] = widest_gap_shift(positions[heavy, 2], c)
    positions[:, 2] += shift[anchor]
    sheet = np.isin(symbols, ["Mo", "S"])
    side = 1.0 if positions[sheet, 2].mean() < positions[symbols == "O", 2].mean() else -1.0
    image = positions[sheet] + np.array([0.0, 0.0, side * c])
    comment = (
        f"{path.name} frame {n_frames} of {n_frames} ({step}), {len(species)} atoms, cell "
        f"{' '.join(str(value) for value in cell)}, wrapped, plus the sheet's image one c "
        f"{'above' if side > 0 else 'below'} ({int(sheet.sum())} atoms)"
    )
    write_xyz(
        OUT_DIR / MOS2_OUT,
        [*species, *symbols[sheet].tolist()],
        np.concatenate([positions, image]),
        comment,
    )
    print(f"source md5 {md5}")


def widest_gap_shift(z_angstrom: np.ndarray, c_angstrom: float) -> np.ndarray:
    """Per atom, 0 or c: c for the atoms at or below the widest gap along a periodic axis.

    Cutting the cell there keeps every layer or ring of atoms on one side of the boundary.
    When the widest gap already spans the boundary (a slab's vacuum) nothing moves.

    Args:
        z_angstrom: Coordinates along the axis, inside [0, c), in angstrom.
        c_angstrom: The cell's period along that axis, in angstrom.

    Returns:
        The shift to add to each coordinate, in angstrom.
    """
    ordered = np.sort(z_angstrom)
    widest = np.argmax(np.diff(np.append(ordered, ordered[0] + c_angstrom)))
    if widest == len(ordered) - 1:
        return np.zeros_like(z_angstrom)
    return np.where(z_angstrom <= ordered[widest], c_angstrom, 0.0)


def stage_hbn(path: Path) -> None:
    lines = path.read_text().splitlines()
    cursor, index = 0, 0
    while index <= max(HBN_FRAMES):
        n_atoms = int(lines[cursor])
        if index in HBN_FRAMES:
            frame = lines[cursor : cursor + 2 + n_atoms]
            lattice = [
                float(value) for value in frame[1].split('Lattice="')[1].split('"')[0].split()
            ]
            if lattice[6:8] != [0.0, 0.0]:
                raise ValueError(f"{path}: frame {index}'s c vector is not along z: {lattice}")
            atoms = [line.split() for line in frame[2:]]
            shift = widest_gap_shift(np.array([float(atom[3]) for atom in atoms]), lattice[8])
            for moved in np.flatnonzero(shift):
                values = [float(value) for value in atoms[moved][1:]]
                values[2] += shift[moved]
                # The source's own layout: species in 2 columns, then " %16.8f" per field.
                frame[2 + moved] = f"{atoms[moved][0]:<2}" + "".join(f" {v:16.8f}" for v in values)
            out = OUT_DIR / HBN_FRAMES[index]
            out.write_text("\n".join(frame) + "\n")
            moved_count = np.count_nonzero(shift)
            print(
                f"{out.name}: frame {index} (line {cursor + 1}), {n_atoms} atoms, "
                f"{moved_count} moved up by c = {lattice[8]} A"
            )
        cursor += 2 + n_atoms
        index += 1
    print(f"{path.name} md5 {hashlib.md5(path.read_bytes()).hexdigest()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--vermiculite", type=Path)
    parser.add_argument("--mos2", type=Path)
    parser.add_argument("--hbn-training", type=Path)
    args = parser.parse_args()
    if not (args.vermiculite or args.mos2 or args.hbn_training):
        parser.error("name at least one source")
    if args.vermiculite:
        stage_vermiculite(args.vermiculite)
    if args.mos2:
        stage_mos2(args.mos2)
    if args.hbn_training:
        stage_hbn(args.hbn_training)


if __name__ == "__main__":
    main()
