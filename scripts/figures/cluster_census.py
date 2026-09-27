# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Structural census of free carbon clusters: coordination, fragments, shells and rings.

Built for the 48 GAP sphere runs (8 sizes x 6 temperatures) whose last frames live in
`scripts/figures/data/carbon-clusters/spheres/`. Every quantity is computed on one bond
graph, the one the site's renderings draw: `boxprep.find_bonds`, carbon-carbon below
1.2 x (0.76 + 0.76) = 1.824 angstrom.

Rings are Franzblau shortest-path rings (Franzblau, Phys. Rev. B 44, 4925 (1991)): a
cycle counts when, for every pair of its atoms, the shortest path through the whole bond
graph is no shorter than the way round the ring. This is the criterion the archive's own
Fortran counter (`Rings2.f90`) applies, and at that program's 1.85 angstrom cutoff this
module reproduces its archived counts for C1000 at 500, 1000 and 3000 K exactly
(`scripts/tests/test_cluster_census.py`).

Units: lengths in angstrom, number densities in atoms per cubic angstrom, times in ps.

Usage:
    uv run scripts/figures/cluster_census.py
        writes census.csv and radial_profiles.csv beside the frames.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import boxprep  # noqa: E402
from render_cluster import BOND_TOLERANCE, COVALENT_RADIUS_ANGSTROM  # noqa: E402

SITE_BOND_CUTOFF_ANGSTROM = BOND_TOLERANCE * 2 * COVALENT_RADIUS_ANGSTROM["C"]
SPHERES_DIR = Path(__file__).resolve().parent / "data" / "carbon-clusters" / "spheres"
SIZES = (40, 60, 80, 120, 160, 373, 686, 1000)
TEMPERATURES_KELVIN = (500, 1000, 2000, 3000, 4000, 5000)
MAX_RING_SIZE = 10

# Classification thresholds (rule in `classify`). Round numbers chosen on physical
# grounds; docs/dossiers/carbon/clusters/classification.md says which were revised after
# a first pass and how many runs sit near each one.
DISSOCIATED_BELOW_LARGEST_FRAGMENT_FRACTION = 0.5
DISSOCIATED_ABOVE_LOW_COORDINATION_FRACTION = 0.8
MOLTEN_BELOW_LARGEST_FRAGMENT_FRACTION = 0.9
MOLTEN_ABOVE_LOW_COORDINATION_FRACTION = 0.3
DIAMOND_LIKE_ABOVE_SP3_FRACTION = 0.4
ONION_ABOVE_SP2_FRACTION = 0.8
ONION_BELOW_RADIAL_BOND_COSINE = 0.3


def bonds_at_cutoff(
    positions_angstrom: np.ndarray, cutoff_angstrom: float = SITE_BOND_CUTOFF_ANGSTROM
) -> np.ndarray:
    """Carbon-carbon bonds shorter than `cutoff_angstrom`, as index pairs.

    `boxprep.find_bonds` has one fixed carbon cutoff (1.824 A). Scaling every coordinate
    by that cutoff over the one requested moves the threshold to `cutoff_angstrom`
    without a second bond search, so the census and the renderings share one bond
    definition and the archive's 1.85 A can still be reproduced.

    Args:
        positions_angstrom: Atom positions, shape (n_atoms, 3), in angstrom. The cluster
            must already be whole (`boxprep.make_compact`): no periodic images are used.
        cutoff_angstrom: Bond cutoff, in angstrom.

    Returns:
        Integer array, shape (n_bonds, 2), each row (i, j) with i < j.
    """
    scaled = positions_angstrom * (SITE_BOND_CUTOFF_ANGSTROM / cutoff_angstrom)
    return np.array(boxprep.find_bonds(scaled), dtype=int).reshape(-1, 2)


def adjacency_matrix(n_atoms: int, bonds: np.ndarray) -> np.ndarray:
    """Symmetric boolean adjacency matrix, shape (n_atoms, n_atoms), from index pairs."""
    adjacency = np.zeros((n_atoms, n_atoms), dtype=bool)
    adjacency[bonds[:, 0], bonds[:, 1]] = True
    adjacency[bonds[:, 1], bonds[:, 0]] = True
    return adjacency


def coordination_fractions(adjacency: np.ndarray) -> np.ndarray:
    """Fraction of atoms with 0, 1, 2, 3, 4 and 5-or-more bonded neighbours.

    Index 4 is the sp3 fraction and index 3 the sp2 fraction in the usual
    coordination-counting sense for carbon.

    Returns:
        Array of length 6 summing to 1; the last entry pools five and more neighbours.
    """
    degree = adjacency.sum(axis=1)
    counts = np.bincount(np.minimum(degree, 5), minlength=6)
    return counts / len(adjacency)


def fragment_sizes(adjacency: np.ndarray) -> np.ndarray:
    """Sizes of the connected fragments of the bond graph, largest first.

    Label propagation: every atom repeatedly takes the smallest label among itself and
    its neighbours until nothing changes, so each fragment ends up labelled by its
    lowest atom index. Vectorised; iterations scale with the fragment diameter.

    Returns:
        Integer array of fragment sizes in atoms, sorted descending, summing to n_atoms.
    """
    n_atoms = len(adjacency)
    labels = np.arange(n_atoms)
    while True:
        neighbour_min = np.where(adjacency, labels[None, :], n_atoms).min(axis=1)
        updated = np.minimum(labels, neighbour_min)
        if np.array_equal(updated, labels):
            break
        labels = updated
    return np.sort(np.bincount(labels)[np.unique(labels)])[::-1]


def radial_density_profile(
    positions_angstrom: np.ndarray, shell_width_angstrom: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Number density in concentric shells about the centre of mass.

    All atoms are carbon, so the centre of mass is the mean position. A graphitic onion
    shows as separate peaks roughly one interlayer spacing (3.4 A) apart; a compact
    disordered or diamond-like sphere shows a plateau falling off at the surface; a
    hollow cage shows a single peak at its radius and nothing inside. For a dissociated
    frame the centre of mass lies between fragments and the profile means little.

    Args:
        positions_angstrom: Atom positions of a whole cluster, shape (n_atoms, 3), in
            angstrom.
        shell_width_angstrom: Shell thickness, in angstrom.

    Returns:
        (shell_inner_radius_angstrom, number_density_per_angstrom3), both of length
        n_shells, covering every atom.
    """
    radius_angstrom = np.linalg.norm(positions_angstrom - positions_angstrom.mean(axis=0), axis=1)
    n_shells = int(radius_angstrom.max() // shell_width_angstrom) + 1
    edges_angstrom = np.arange(n_shells + 1) * shell_width_angstrom
    counts, _ = np.histogram(radius_angstrom, bins=edges_angstrom)
    shell_volume_angstrom3 = 4.0 / 3.0 * np.pi * np.diff(edges_angstrom**3)
    return edges_angstrom[:-1], counts / shell_volume_angstrom3


def mean_radial_bond_cosine(positions_angstrom: np.ndarray, bonds: np.ndarray) -> float:
    """Mean |cos| of the angle between each bond and the radius through its midpoint.

    A scalar for shell-like order that needs no histogram binning: bonds lying in
    concentric shells, as in a fullerene cage or a graphitic onion, are tangential and
    give values near 0; an isotropic network, diamond-like or disordered, gives 0.5.
    Measured from the centre of mass, so meaningless for a dissociated frame.

    Returns:
        Dimensionless mean |cos theta| over all bonds, in [0, 1]; NaN if there are none.
    """
    if len(bonds) == 0:
        return float("nan")
    centred = positions_angstrom - positions_angstrom.mean(axis=0)
    bond_vector = centred[bonds[:, 1]] - centred[bonds[:, 0]]
    midpoint = 0.5 * (centred[bonds[:, 1]] + centred[bonds[:, 0]])
    cosine = np.einsum("ij,ij->i", bond_vector, midpoint) / (
        np.linalg.norm(bond_vector, axis=1) * np.linalg.norm(midpoint, axis=1)
    )
    return float(np.nanmean(np.abs(cosine)))


def shortest_path_lengths(adjacency: np.ndarray, max_length: int) -> np.ndarray:
    """Bond-count shortest paths between every pair of atoms, up to `max_length`.

    Breadth-first search for all sources at once, one boolean matrix product per layer.

    Returns:
        Integer matrix, shape (n_atoms, n_atoms): 0 on the diagonal, the path length where
        it is at most `max_length`, and -1 beyond that or between fragments.
    """
    n_atoms = len(adjacency)
    lengths = np.full((n_atoms, n_atoms), -1, dtype=np.int16)
    np.fill_diagonal(lengths, 0)
    reached = np.eye(n_atoms, dtype=bool)
    frontier = reached.copy()
    step = adjacency.astype(np.float32)
    for length in range(1, max_length + 1):
        frontier = ((frontier.astype(np.float32) @ step) > 0) & ~reached
        if not frontier.any():
            break
        lengths[frontier] = length
        reached |= frontier
    return lengths


def shortest_path_rings(
    adjacency: np.ndarray, max_ring_size: int = MAX_RING_SIZE
) -> list[list[int]]:
    """Every Franzblau shortest-path ring of 3 to `max_ring_size` atoms, each listed once.

    Each ring is found from its lowest-indexed atom, the origin, by a depth-first walk
    over atoms of higher index. In a shortest-path ring the graph distance from the
    origin to the atom k steps round is min(k, L - k), so the walk may only climb one
    bond-distance per step up to the apex, hold once at the apex (odd rings), and then
    descend one per step back to the origin. That prunes the walk to a few paths per
    origin. Each closed candidate is then checked for every pair of its atoms against the
    whole graph, and kept in one direction only.

    Args:
        adjacency: Symmetric boolean adjacency matrix.
        max_ring_size: Largest ring size counted.

    Returns:
        Rings as lists of atom indices in ring order, starting at the lowest index.
    """
    max_apex = max_ring_size // 2
    lengths = shortest_path_lengths(adjacency, max_apex)
    neighbours = [np.flatnonzero(row).tolist() for row in adjacency]
    rings: list[list[int]] = []
    for origin in range(len(adjacency)):
        distance = lengths[origin].tolist()
        # Stack entries: (path, apex); apex is None while still climbing away from origin.
        stack: list[tuple[list[int], int | None]] = [
            ([origin, first], None) for first in neighbours[origin] if first > origin
        ]
        while stack:
            path, apex = stack.pop()
            here = distance[path[-1]]
            for step in neighbours[path[-1]]:
                if step == origin:
                    if here == 1 and len(path) >= 3 and path[1] < path[-1]:
                        rings.append(path)
                    continue
                if step < origin or step in path:
                    continue
                there = distance[step]
                if apex is None:
                    if there == here + 1 and there <= max_apex:
                        stack.append((path + [step], None))
                    elif there == here and 2 * here + 1 <= max_ring_size:
                        stack.append((path + [step], here))  # odd ring: hold at the apex
                    elif there == here - 1:
                        stack.append((path + [step], here))  # even ring: turn at the apex
                elif there == here - 1:
                    stack.append((path + [step], apex))
    return [ring for ring in rings if _is_shortest_path_ring(ring, lengths)]


def _is_shortest_path_ring(ring: list[int], lengths: np.ndarray) -> bool:
    """True if no pair of ring atoms is closer through the graph than round the ring."""
    size = len(ring)
    offset = np.abs(np.arange(size)[:, None] - np.arange(size)[None, :])
    round_ring = np.minimum(offset, size - offset)
    return bool(np.array_equal(lengths[np.ix_(ring, ring)], round_ring))


def ring_counts(adjacency: np.ndarray, max_ring_size: int = MAX_RING_SIZE) -> np.ndarray:
    """Number of shortest-path rings of each size; index = ring size, 0 to max_ring_size."""
    sizes = [len(ring) for ring in shortest_path_rings(adjacency, max_ring_size)]
    return np.bincount(np.array(sizes, dtype=int), minlength=max_ring_size + 1)


def census(positions_angstrom: np.ndarray) -> dict[str, float | int | str]:
    """Every per-frame quantity at the site's bond cutoff, as one flat record.

    Args:
        positions_angstrom: A whole cluster, shape (n_atoms, 3), in angstrom.

    Returns:
        Flat dict: coordination fractions, sp2 and sp3 fractions, fragment count and
        sizes, radius of gyration, mean radial bond cosine and ring counts by size.
    """
    bonds = bonds_at_cutoff(positions_angstrom)
    adjacency = adjacency_matrix(len(positions_angstrom), bonds)
    fractions = coordination_fractions(adjacency)
    fragments = fragment_sizes(adjacency)
    rings = ring_counts(adjacency)
    centred = positions_angstrom - positions_angstrom.mean(axis=0)
    record: dict[str, float | int | str] = {
        f"fraction_coordination_{k}": round(float(fractions[k]), 4) for k in range(5)
    }
    record["fraction_coordination_5plus"] = round(float(fractions[5]), 4)
    record["low_coordination_fraction"] = round(float(fractions[:3].sum()), 4)
    record["sp2_fraction"] = record["fraction_coordination_3"]
    record["sp3_fraction"] = record["fraction_coordination_4"]
    record["n_bonds"] = len(bonds)
    record["n_fragments"] = len(fragments)
    record["largest_fragment_fraction"] = round(float(fragments[0] / len(positions_angstrom)), 4)
    record["fragment_sizes"] = " ".join(str(size) for size in fragments)
    record["radius_of_gyration_angstrom"] = round(
        float(np.sqrt((centred**2).sum(axis=1).mean())), 3
    )
    record["mean_radial_bond_cosine"] = round(mean_radial_bond_cosine(positions_angstrom, bonds), 4)
    record.update({f"rings_{size}": int(rings[size]) for size in range(3, MAX_RING_SIZE + 1)})
    return record


def classify(record: dict[str, float | int | str]) -> str:
    """One of: dissociated, molten, diamond-like, graphitic onion, disordered.

    Rules, applied in order, on the site-cutoff census of a single frame. "Low
    coordination" means two or fewer bonded neighbours.
        1. dissociated: the largest fragment holds under half the atoms, or at least 80%
           of atoms are low-coordination: the cluster has unravelled into chains.
        2. molten: the largest fragment holds under 90% of atoms and at least 30% are
           low-coordination: a body still largely connected but shedding chains.
        3. diamond-like: at least 40% of atoms four-coordinated.
        4. graphitic onion: at least 80% three-coordinated and bonds lying in shells
           (mean radial bond cosine below 0.3, against 0.5 for an isotropic network).
           A single closed cage qualifies; the radial profile says how many shells.
        5. disordered: anything else.
    One frame cannot tell a liquid from a frozen network by its dynamics, so rule 2 uses
    the only frame-level sign of melting in vacuum: a connected body losing chains.
    Small clusters at 500 K are edge-rich (up to 55% low-coordination) yet whole, which
    is why rule 2 also asks for lost atoms.
    """
    low_coordination = float(record["low_coordination_fraction"])
    largest = float(record["largest_fragment_fraction"])
    if (
        largest < DISSOCIATED_BELOW_LARGEST_FRAGMENT_FRACTION
        or low_coordination >= DISSOCIATED_ABOVE_LOW_COORDINATION_FRACTION
    ):
        return "dissociated"
    if (
        largest < MOLTEN_BELOW_LARGEST_FRAGMENT_FRACTION
        and low_coordination >= MOLTEN_ABOVE_LOW_COORDINATION_FRACTION
    ):
        return "molten"
    if float(record["sp3_fraction"]) >= DIAMOND_LIKE_ABOVE_SP3_FRACTION:
        return "diamond-like"
    if (
        float(record["sp2_fraction"]) >= ONION_ABOVE_SP2_FRACTION
        and float(record["mean_radial_bond_cosine"]) < ONION_BELOW_RADIAL_BOND_COSINE
    ):
        return "graphitic onion"
    return "disordered"


def main() -> None:
    """Census of all 48 frames: census.csv and radial_profiles.csv beside the frames."""
    provenance = json.loads((SPHERES_DIR / "provenance.json").read_text())
    rows, profile_rows = [], []
    for temperature_kelvin in TEMPERATURES_KELVIN:
        for n_atoms in SIZES:
            name = f"C{n_atoms}-{temperature_kelvin}K"
            positions_angstrom = boxprep.read_xyz(SPHERES_DIR / f"{name}.xyz").positions_angstrom
            source = provenance[f"{name}.xyz"]
            record = {
                "run": name,
                "n_atoms": n_atoms,
                "temperature_kelvin": temperature_kelvin,
                "nvt_time_at_last_frame_ps": source["nvt_time_at_last_frame_ps"],
                "run_completed": source["run_completed_25000_steps"],
                "minimiser": source["minimiser"].split()[0],
                **census(positions_angstrom),
            }
            record["class"] = classify(record)
            rows.append(record)
            for radius_angstrom, density in zip(
                *radial_density_profile(positions_angstrom), strict=True
            ):
                profile_rows.append(
                    {
                        "run": name,
                        "shell_inner_radius_angstrom": f"{radius_angstrom:.1f}",
                        "number_density_per_angstrom3": f"{density:.5f}",
                    }
                )
            print(f"{name}: {record['class']}", flush=True)
    for filename, table in (("census.csv", rows), ("radial_profiles.csv", profile_rows)):
        with (SPHERES_DIR / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)


if __name__ == "__main__":
    main()
