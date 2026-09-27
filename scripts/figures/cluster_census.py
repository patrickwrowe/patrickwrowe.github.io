# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Structural census of free carbon clusters: coordination, fragments, shells and rings.

Built for the 48 GAP sphere runs (8 sizes x 6 temperatures) whose last frames live in
`scripts/figures/data/carbon-clusters/spheres/` (written by `extract_spheres.py`). Every
quantity is computed on one bond graph: carbon-carbon pairs closer than the site's
renderer cutoff, 1.2 x (0.76 + 0.76) = 1.824 angstrom, measured to the nearest periodic
image in the run's cubic cell (`boxprep.find_bonds_periodic`), so a chain that crosses
the cell boundary is not cut.

Rings are Franzblau shortest-path rings (Franzblau, Phys. Rev. B 44, 4925 (1991)): a
cycle counts when, for every pair of its atoms, the shortest path through the whole bond
graph is no shorter than the way round the ring. This is the criterion the archive's own
Fortran counter (`Rings2.f90`) applies, and at that program's 1.85 angstrom cutoff this
module reproduces its archived counts for C1000 at 500, 1000 and 3000 K exactly
(`scripts/tests/test_cluster_census.py`).

Shells are counted from the radial number density about the largest fragment's centre of
mass; see `shells`. The classification rule is in `classify`.

Units: lengths in angstrom, number densities in atoms per cubic angstrom, areal
densities in atoms per square angstrom, times in ps.

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

SITE_BOND_CUTOFF_ANGSTROM = boxprep.CARBON_BOND_CUTOFF_ANGSTROM
SPHERES_DIR = Path(__file__).resolve().parent / "data" / "carbon-clusters" / "spheres"
SIZES = (40, 60, 80, 120, 160, 373, 686, 1000)
TEMPERATURES_KELVIN = (500, 1000, 2000, 3000, 4000, 5000)
MAX_RING_SIZE = 10

# Shell counting (see `shells`). Graphene holds 0.382 atoms per square angstrom; the
# smallest closed carbon cage, C20, has 20 atoms on a sphere of radius about 2.0 A.
SHELL_SMOOTHING_ANGSTROM = 0.4
SHELL_MIN_DIP_FRACTION = 0.2
SHELL_MIN_ATOMS = 20
SHELL_MIN_RADIUS_ANGSTROM = 2.0
SHELL_MIN_COVERAGE_PER_ANGSTROM2 = 0.19
# An enclosed fragment is a shell if its own bonds are tangential about its own centre.
ENCLOSED_SHELL_BELOW_RADIAL_BOND_COSINE = 0.3
# Core atoms: more than half a graphite interlayer spacing (3.4 A) inside the innermost
# shell, i.e. nearer the next shell in than to this one.
CORE_DEPTH_ANGSTROM = 1.7
# Interior sp3: atoms deeper than one graphite interlayer spacing inside the
# 98th-percentile radius, so the edge-rich surface does not dilute the core.
INTERIOR_DEPTH_ANGSTROM = 3.4

# Classification thresholds (rule in `classify`). Round numbers chosen on physical
# grounds; docs/dossiers/carbon/clusters/classification.md says which were revised after
# a first pass and how many runs sit near each one.
DISSOCIATED_BELOW_LARGEST_FRAGMENT_FRACTION = 0.5
DISSOCIATED_ABOVE_LOW_COORDINATION_FRACTION = 0.8
MOLTEN_BELOW_LARGEST_FRAGMENT_FRACTION = 0.9
MOLTEN_ABOVE_LOW_COORDINATION_FRACTION = 0.3
DIAMOND_LIKE_ABOVE_SP3_FRACTION = 0.4
SHELLED_ABOVE_SP2_FRACTION = 0.8
SHELLED_BELOW_RADIAL_BOND_COSINE = 0.3
CAGE_MAX_LOW_COORDINATION_FRACTION = 0.2


def load_frame(name: str) -> tuple[np.ndarray, float]:
    """One extracted frame and its run's cell edge.

    Args:
        name: Run name, `C<n>-<T>K`.

    Returns:
        (positions, shape (n_atoms, 3), in angstrom; cubic cell edge, in angstrom, from
        provenance.json).

    Raises:
        FileNotFoundError: If the frame or provenance.json is missing.
        KeyError: If provenance.json has no cell edge for the run.
    """
    provenance = json.loads((SPHERES_DIR / "provenance.json").read_text())
    positions_angstrom = boxprep.read_xyz(SPHERES_DIR / f"{name}.xyz").positions_angstrom
    return positions_angstrom, float(provenance[f"{name}.xyz"]["cell_edge_angstrom"])


def bonds_at_cutoff(
    positions_angstrom: np.ndarray,
    edge_angstrom: float,
    cutoff_angstrom: float = SITE_BOND_CUTOFF_ANGSTROM,
) -> np.ndarray:
    """Carbon-carbon bonds shorter than `cutoff_angstrom` to the nearest periodic image.

    `boxprep.find_bonds_periodic` has one fixed carbon cutoff (1.824 A). Scaling the
    coordinates and the cell edge together by that cutoff over the one requested moves
    the threshold to `cutoff_angstrom` without a second bond search, so the census and
    the renderings share one bond definition and the archive's 1.85 A can still be
    reproduced.

    Args:
        positions_angstrom: Atom positions, shape (n_atoms, 3), in angstrom.
        edge_angstrom: Cubic cell edge, in angstrom. For a free cluster not taken from a
            periodic dump, any edge wider than the cluster plus twice the cutoff.
        cutoff_angstrom: Bond cutoff, in angstrom.

    Returns:
        Integer array, shape (n_bonds, 2), each row (first, second) with first < second.

    Raises:
        ValueError: From `find_bonds_periodic`, if the scaled edge is not more than twice
            the cutoff.
    """
    scale = SITE_BOND_CUTOFF_ANGSTROM / cutoff_angstrom
    bonds = boxprep.find_bonds_periodic(positions_angstrom * scale, edge_angstrom * scale)
    return np.array(bonds, dtype=int).reshape(-1, 2)


def adjacency_matrix(n_atoms: int, bonds: np.ndarray) -> np.ndarray:
    """Symmetric boolean adjacency matrix from bond index pairs.

    Args:
        n_atoms: Number of atoms.
        bonds: Integer array, shape (n_bonds, 2), of zero-based atom index pairs.

    Returns:
        Boolean matrix, shape (n_atoms, n_atoms), True where two atoms are bonded.
    """
    adjacency = np.zeros((n_atoms, n_atoms), dtype=bool)
    adjacency[bonds[:, 0], bonds[:, 1]] = True
    adjacency[bonds[:, 1], bonds[:, 0]] = True
    return adjacency


def coordination_fractions(adjacency: np.ndarray) -> np.ndarray:
    """Fraction of atoms with 0, 1, 2, 3, 4 and 5-or-more bonded neighbours.

    Index 4 is the sp3 fraction and index 3 the sp2 fraction in the usual
    coordination-counting sense for carbon.

    Args:
        adjacency: Symmetric boolean adjacency matrix, shape (n_atoms, n_atoms).

    Returns:
        Array of length 6 summing to 1; the last entry pools five and more neighbours.
    """
    degree = adjacency.sum(axis=1)
    counts = np.bincount(np.minimum(degree, 5), minlength=6)
    return counts / len(adjacency)


def fragment_labels(adjacency: np.ndarray) -> np.ndarray:
    """Connected-fragment label of every atom: the lowest atom index in its fragment.

    Label propagation: every atom repeatedly takes the smallest label among itself and
    its neighbours until nothing changes. Vectorised; iterations scale with the
    fragment diameter.

    Args:
        adjacency: Symmetric boolean adjacency matrix, shape (n_atoms, n_atoms).

    Returns:
        Integer array, shape (n_atoms,).
    """
    n_atoms = len(adjacency)
    labels = np.arange(n_atoms)
    while True:
        neighbour_min = np.where(adjacency, labels[None, :], n_atoms).min(axis=1)
        updated = np.minimum(labels, neighbour_min)
        if np.array_equal(updated, labels):
            return labels
        labels = updated


def fragment_sizes(adjacency: np.ndarray) -> np.ndarray:
    """Sizes of the connected fragments of the bond graph, largest first.

    Args:
        adjacency: Symmetric boolean adjacency matrix, shape (n_atoms, n_atoms).

    Returns:
        Integer array of fragment sizes in atoms, sorted descending, summing to n_atoms.
    """
    labels = fragment_labels(adjacency)
    return np.sort(np.bincount(labels)[np.unique(labels)])[::-1]


def largest_fragment_centre(positions_angstrom: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
    """Centre of mass of the largest bonded fragment (all atoms are carbon).

    The reference point for every radial quantity, so a detached chain far from the body
    does not drag the centre off it. For a dissociated frame it is the centre of one
    fragment among many.

    Args:
        positions_angstrom: Whole-fragment positions, shape (n_atoms, 3), in angstrom.
        adjacency: Symmetric boolean adjacency matrix, shape (n_atoms, n_atoms).

    Returns:
        Position, shape (3,), in angstrom. Ties go to the fragment with the lowest index.
    """
    labels = fragment_labels(adjacency)
    fragment_ids, counts = np.unique(labels, return_counts=True)
    return positions_angstrom[labels == fragment_ids[np.argmax(counts)]].mean(axis=0)


def radial_density_profile(
    positions_angstrom: np.ndarray, centre_angstrom: np.ndarray, shell_width_angstrom: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Number density in concentric 1 A (by default) shells about a centre.

    A graphitic onion shows as separate peaks roughly one interlayer spacing apart; a
    compact sphere shows a plateau falling off at the surface; a hollow cage shows one
    peak at its radius and nothing inside. For a dissociated frame it means little.

    Args:
        positions_angstrom: Atom positions, shape (n_atoms, 3), in angstrom.
        centre_angstrom: Reference point, shape (3,), in angstrom.
        shell_width_angstrom: Shell thickness, in angstrom.

    Returns:
        (shell_inner_radius_angstrom, number_density_per_angstrom3), both of length
        n_shells, covering every atom.
    """
    radius_angstrom = np.linalg.norm(positions_angstrom - centre_angstrom, axis=1)
    n_shells = int(radius_angstrom.max() // shell_width_angstrom) + 1
    edges_angstrom = np.arange(n_shells + 1) * shell_width_angstrom
    counts, _ = np.histogram(radius_angstrom, bins=edges_angstrom)
    shell_volume_angstrom3 = 4.0 / 3.0 * np.pi * np.diff(edges_angstrom**3)
    return edges_angstrom[:-1], counts / shell_volume_angstrom3


def shells(
    positions_angstrom: np.ndarray,
    centre_angstrom: np.ndarray,
    smoothing_angstrom: float = SHELL_SMOOTHING_ANGSTROM,
    min_dip_fraction: float = SHELL_MIN_DIP_FRACTION,
    min_atoms: int = SHELL_MIN_ATOMS,
    min_radius_angstrom: float = SHELL_MIN_RADIUS_ANGSTROM,
    min_coverage_per_angstrom2: float = SHELL_MIN_COVERAGE_PER_ANGSTROM2,
) -> list[tuple[float, int]]:
    """Concentric carbon shells, counted as peaks of the smoothed radial number density.

    Method:
        1. Radial number density rho(r): a Gaussian of width `smoothing_angstrom` on each
           atom's distance from the centre, summed on a 0.05 A grid and divided by
           4 pi r^2. Dividing by the sphere area gives every graphitic shell a peak of
           similar height whatever its radius.
        2. Peaks are local maxima of rho. Two neighbouring peaks are merged while the
           lowest rho between them is less than `min_dip_fraction` below the lower of
           the two (the shallowest dip first), so ripples within one shell do not count.
        3. Each surviving peak owns the atoms between the minima on either side. It is a
           shell if it holds at least `min_atoms` atoms, their mean radius is at least
           `min_radius_angstrom`, and its coverage, atoms over 4 pi (mean radius)^2, is at
           least `min_coverage_per_angstrom2`. The defaults: C20, the smallest closed
           carbon cage (20 atoms, radius about 2.0 A), and half the areal density of
           graphene (0.382 A^-2). Chains trailing off the surface and central lumps fail.

    A compact, disordered sphere can also show density ripples that pass; shell-like
    bonding is judged separately by `mean_radial_bond_cosine`. `all_shells` applies this
    to the largest fragment's atoms and adds enclosed fragments that are shells of their
    own, which a profile about the outer centre misses when they sit off-centre.

    Args:
        positions_angstrom: Whole-fragment positions, shape (n_atoms, 3), in angstrom.
        centre_angstrom: Reference point, shape (3,), in angstrom.
        smoothing_angstrom: Gaussian width, in angstrom.
        min_dip_fraction: Relative depth of the minimum that separates two peaks.
        min_atoms: Fewest atoms a shell may hold.
        min_radius_angstrom: Smallest mean radius of a shell, in angstrom.
        min_coverage_per_angstrom2: Smallest areal density of a shell, per square angstrom.

    Returns:
        One (mean radius in angstrom, atom count) per shell, innermost first.
    """
    radius_angstrom = np.linalg.norm(positions_angstrom - centre_angstrom, axis=1)
    grid_angstrom = np.arange(0.025, radius_angstrom.max() + 3 * smoothing_angstrom, 0.05)
    gaussian = np.exp(
        -0.5 * ((grid_angstrom[:, None] - radius_angstrom[None, :]) / smoothing_angstrom) ** 2
    )
    atoms_per_angstrom = gaussian.sum(axis=1) / (smoothing_angstrom * np.sqrt(2 * np.pi))
    density = atoms_per_angstrom / (4 * np.pi * grid_angstrom**2)
    rising = density[1:-1] > density[:-2]
    not_falling_next = density[1:-1] >= density[2:]
    peaks = (np.flatnonzero(rising & not_falling_next) + 1).tolist()
    while len(peaks) > 1:
        depths = [
            1 - density[left : right + 1].min() / min(density[left], density[right])
            for left, right in zip(peaks[:-1], peaks[1:], strict=True)
        ]
        shallowest = int(np.argmin(depths))
        if depths[shallowest] >= min_dip_fraction:
            break
        left, right = peaks[shallowest], peaks[shallowest + 1]
        peaks[shallowest : shallowest + 2] = [left if density[left] >= density[right] else right]
    boundaries_angstrom = [0.0]
    for left, right in zip(peaks[:-1], peaks[1:], strict=True):
        boundaries_angstrom.append(grid_angstrom[left + int(np.argmin(density[left : right + 1]))])
    boundaries_angstrom.append(np.inf)
    found: list[tuple[float, int]] = []
    for inner, outer in zip(boundaries_angstrom[:-1], boundaries_angstrom[1:], strict=True):
        members = radius_angstrom[(radius_angstrom >= inner) & (radius_angstrom < outer)]
        if len(members) < min_atoms:
            continue
        mean_radius_angstrom = float(members.mean())
        coverage = len(members) / (4 * np.pi * mean_radius_angstrom**2)
        if mean_radius_angstrom >= min_radius_angstrom and coverage >= min_coverage_per_angstrom2:
            found.append((mean_radius_angstrom, len(members)))
    return found


def enclosed_shells(
    positions_angstrom: np.ndarray,
    bonds: np.ndarray,
    labels: np.ndarray,
    min_atoms: int = SHELL_MIN_ATOMS,
    max_cosine: float = ENCLOSED_SHELL_BELOW_RADIAL_BOND_COSINE,
) -> list[tuple[float, int]]:
    """Bonded fragments nested inside the largest one that are shells in their own right.

    A fragment other than the largest counts if (a) it is enclosed: every one of its atoms
    lies closer to the largest fragment's centre of mass than that fragment's median
    atom; (b) it holds at least `min_atoms` atoms; and (c) its own bonds lie tangentially
    about its own centre of mass: mean radial bond cosine below `max_cosine`. Such a
    fragment is a small cage inside the outer one; if it sits off-centre its atoms smear
    across the outer shell's radii in a profile about the outer centre, so the profile
    alone cannot see it.

    Args:
        positions_angstrom: Whole-fragment positions, shape (n_atoms, 3), in angstrom.
        bonds: Integer array, shape (n_bonds, 2), of zero-based atom index pairs.
        labels: Fragment label per atom, from `fragment_labels`.
        min_atoms: Fewest atoms an enclosed shell may hold.
        max_cosine: Mean radial bond cosine, about the fragment's own centre, below which
            its bonds count as tangential.

    Returns:
        One (mean radius about the fragment's own centre in angstrom, atom count) per
        enclosed shell, smallest first.
    """
    fragment_ids, counts = np.unique(labels, return_counts=True)
    largest = labels == fragment_ids[np.argmax(counts)]
    outer_centre_angstrom = positions_angstrom[largest].mean(axis=0)
    outer_median_angstrom = np.median(
        np.linalg.norm(positions_angstrom[largest] - outer_centre_angstrom, axis=1)
    )
    found: list[tuple[float, int]] = []
    for fragment_id, count in zip(fragment_ids, counts, strict=True):
        members = labels == fragment_id
        if count < min_atoms or np.array_equal(members, largest):
            continue
        distance_from_outer = np.linalg.norm(
            positions_angstrom[members] - outer_centre_angstrom, axis=1
        )
        if distance_from_outer.max() >= outer_median_angstrom:
            continue
        own_centre_angstrom = positions_angstrom[members].mean(axis=0)
        own_bonds = bonds[members[bonds[:, 0]] & members[bonds[:, 1]]]
        cosine = mean_radial_bond_cosine(positions_angstrom, own_bonds, own_centre_angstrom)
        if cosine < max_cosine:
            own_radius_angstrom = np.linalg.norm(
                positions_angstrom[members] - own_centre_angstrom, axis=1
            ).mean()
            found.append((float(own_radius_angstrom), int(count)))
    return sorted(found)


def all_shells(
    positions_angstrom: np.ndarray, bonds: np.ndarray, labels: np.ndarray
) -> tuple[list[tuple[float, int]], list[tuple[float, int]]]:
    """Every shell of a cluster: density-profile shells of its body plus enclosed shells.

    The profile (`shells`) is taken over the largest fragment's atoms only, about their
    centre of mass, so no atom can count towards both a profile shell and an enclosed
    shell (`enclosed_shells`). Detached chains and small cores are in neither.

    Args:
        positions_angstrom: Whole-fragment positions, shape (n_atoms, 3), in angstrom.
        bonds: Integer array, shape (n_bonds, 2), of zero-based atom index pairs.
        labels: Fragment label per atom, from `fragment_labels`.

    Returns:
        (profile shells, enclosed shells), each a list of (mean radius in angstrom, atom
        count). Profile radii are about the largest fragment's centre; enclosed radii
        about each enclosed fragment's own centre.
    """
    fragment_ids, counts = np.unique(labels, return_counts=True)
    largest = labels == fragment_ids[np.argmax(counts)]
    body_angstrom = positions_angstrom[largest]
    profile = shells(body_angstrom, body_angstrom.mean(axis=0))
    return profile, enclosed_shells(positions_angstrom, bonds, labels)


def core_atom_count(
    positions_angstrom: np.ndarray,
    centre_angstrom: np.ndarray,
    shell_radius_angstrom: float,
    depth_angstrom: float = CORE_DEPTH_ANGSTROM,
) -> int:
    """Atoms more than `depth_angstrom` inside a shell's mean radius, bonded or not.

    For a cage, the number of atoms inside it; the default depth is half a graphite
    interlayer spacing.

    Args:
        positions_angstrom: Whole-fragment positions, shape (n_atoms, 3), in angstrom.
        centre_angstrom: The shell's centre, shape (3,), in angstrom.
        shell_radius_angstrom: The shell's mean radius, in angstrom.
        depth_angstrom: How far inside the shell an atom must be to count, in angstrom.

    Returns:
        The atom count.
    """
    radius_angstrom = np.linalg.norm(positions_angstrom - centre_angstrom, axis=1)
    return int(np.sum(radius_angstrom < shell_radius_angstrom - depth_angstrom))


def is_dissociated(largest_fragment_fraction: float, low_coordination_fraction: float) -> bool:
    """Rule 1 of `classify`: the cluster has come apart or unravelled into chains.

    Args:
        largest_fragment_fraction: Share of atoms in the largest fragment.
        low_coordination_fraction: Share of atoms with two or fewer neighbours.

    Returns:
        True if the largest fragment holds under half the atoms, or at least 80% of atoms
        are low-coordination.
    """
    return (
        largest_fragment_fraction < DISSOCIATED_BELOW_LARGEST_FRAGMENT_FRACTION
        or low_coordination_fraction >= DISSOCIATED_ABOVE_LOW_COORDINATION_FRACTION
    )


def mean_radial_bond_cosine(
    positions_angstrom: np.ndarray, bonds: np.ndarray, centre_angstrom: np.ndarray
) -> float:
    """Mean |cos| of the angle between each bond and the radius through its midpoint.

    A scalar for shell-like order that needs no binning: bonds lying in concentric
    shells, as in a fullerene cage or a graphitic onion, are tangential and give values
    near 0; an isotropic network, diamond-like or disordered, gives 0.5. Positions must
    be whole (no bond across the cell boundary), as `extract_spheres.py` writes them.

    Args:
        positions_angstrom: Whole-fragment positions, shape (n_atoms, 3), in angstrom.
        bonds: Integer array, shape (n_bonds, 2), of zero-based atom index pairs.
        centre_angstrom: Reference point, shape (3,), in angstrom.

    Returns:
        Dimensionless mean |cos theta| over all bonds, in [0, 1]; NaN if there are none.
    """
    if len(bonds) == 0:
        return float("nan")
    centred = positions_angstrom - centre_angstrom
    bond_vector = centred[bonds[:, 1]] - centred[bonds[:, 0]]
    midpoint = 0.5 * (centred[bonds[:, 1]] + centred[bonds[:, 0]])
    cosine = np.einsum("ij,ij->i", bond_vector, midpoint) / (
        np.linalg.norm(bond_vector, axis=1) * np.linalg.norm(midpoint, axis=1)
    )
    return float(np.nanmean(np.abs(cosine)))


def interior_sp3_fraction(
    positions_angstrom: np.ndarray, adjacency: np.ndarray, centre_angstrom: np.ndarray
) -> float:
    """sp3 fraction of the atoms deeper than one interlayer spacing inside the surface.

    The surface is the 98th-percentile distance from the centre; atoms more than
    `INTERIOR_DEPTH_ANGSTROM` (3.4 A) inside it count as interior, so edge atoms and
    chain tails do not dilute a four-coordinated core.

    Args:
        positions_angstrom: Whole-fragment positions, shape (n_atoms, 3), in angstrom.
        adjacency: Symmetric boolean adjacency matrix, shape (n_atoms, n_atoms).
        centre_angstrom: Reference point, shape (3,), in angstrom.

    Returns:
        Fraction of interior atoms with exactly four neighbours; NaN if none is interior.
    """
    radius_angstrom = np.linalg.norm(positions_angstrom - centre_angstrom, axis=1)
    interior = radius_angstrom < np.percentile(radius_angstrom, 98) - INTERIOR_DEPTH_ANGSTROM
    if not interior.any():
        return float("nan")
    return float(np.mean(adjacency[interior].sum(axis=1) == 4))


def shortest_path_lengths(adjacency: np.ndarray, max_length: int) -> np.ndarray:
    """Bond-count shortest paths between every pair of atoms, up to `max_length`.

    Breadth-first search for all sources at once, one boolean matrix product per layer.

    Args:
        adjacency: Symmetric boolean adjacency matrix, shape (n_atoms, n_atoms).
        max_length: Longest path, in bonds, to resolve.

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
    """Whether no pair of ring atoms is closer through the graph than round the ring.

    Args:
        ring: Atom indices in ring order.
        lengths: Shortest-path matrix from `shortest_path_lengths`, resolved to at least
            half the ring size.

    Returns:
        True if every pair's graph distance equals its distance round the ring.
    """
    size = len(ring)
    offset = np.abs(np.arange(size)[:, None] - np.arange(size)[None, :])
    round_ring = np.minimum(offset, size - offset)
    return bool(np.array_equal(lengths[np.ix_(ring, ring)], round_ring))


def ring_counts(adjacency: np.ndarray, max_ring_size: int = MAX_RING_SIZE) -> np.ndarray:
    """Number of shortest-path rings of each size.

    Args:
        adjacency: Symmetric boolean adjacency matrix, shape (n_atoms, n_atoms).
        max_ring_size: Largest ring size counted.

    Returns:
        Integer array of length max_ring_size + 1, indexed by ring size (entries 0 to 2
        are always 0).
    """
    sizes = [len(ring) for ring in shortest_path_rings(adjacency, max_ring_size)]
    return np.bincount(np.array(sizes, dtype=int), minlength=max_ring_size + 1)


def census(positions_angstrom: np.ndarray, edge_angstrom: float) -> dict[str, float | int | str]:
    """Every per-frame quantity at the site's bond cutoff, as one flat record.

    Args:
        positions_angstrom: Whole-fragment positions, shape (n_atoms, 3), in angstrom.
        edge_angstrom: The run's cubic cell edge, in angstrom, for periodic bonding.

    Returns:
        Flat dict: coordination fractions, low-coordination (closure) fraction, sp2, sp3
        and interior sp3 fractions, bond and fragment counts, fragment sizes, radius of
        gyration, mean radial bond cosine, shell counts and radii, core atom count and
        ring counts by size. Some values are NaN, written to CSV as the literal `nan`
        (`numpy.genfromtxt` and `float()` read it back):
        `radius_of_gyration_angstrom` for a dissociated frame with more than one fragment,
        where it would measure only where make_compact happened to place the fragments;
        `core_atoms` when there is no profile shell; `interior_sp3_fraction` when no atom
        is interior; `mean_radial_bond_cosine` when there are no bonds. For an intact frame
        with a detached chain (C120-3000K), the radius of gyration still includes it.
    """
    bonds = bonds_at_cutoff(positions_angstrom, edge_angstrom)
    adjacency = adjacency_matrix(len(positions_angstrom), bonds)
    fractions = coordination_fractions(adjacency)
    fragments = fragment_sizes(adjacency)
    labels = fragment_labels(adjacency)
    rings = ring_counts(adjacency)
    centre_angstrom = largest_fragment_centre(positions_angstrom, adjacency)
    profile_shells, nested_shells = all_shells(positions_angstrom, bonds, labels)
    spread = positions_angstrom - positions_angstrom.mean(axis=0)
    record: dict[str, float | int | str] = {
        f"fraction_coordination_{neighbours}": round(float(fractions[neighbours]), 4)
        for neighbours in range(5)
    }
    record["fraction_coordination_5plus"] = round(float(fractions[5]), 4)
    record["low_coordination_fraction"] = round(float(fractions[:3].sum()), 4)
    record["sp2_fraction"] = record["fraction_coordination_3"]
    record["sp3_fraction"] = record["fraction_coordination_4"]
    record["interior_sp3_fraction"] = round(
        interior_sp3_fraction(positions_angstrom, adjacency, centre_angstrom), 4
    )
    record["n_bonds"] = len(bonds)
    record["n_fragments"] = len(fragments)
    record["largest_fragment_fraction"] = round(float(fragments[0] / len(positions_angstrom)), 4)
    record["fragment_sizes"] = " ".join(str(size) for size in fragments)
    dissociated = is_dissociated(
        float(record["largest_fragment_fraction"]), float(record["low_coordination_fraction"])
    )
    record["radius_of_gyration_angstrom"] = (
        float("nan")
        if dissociated and len(fragments) > 1
        else round(float(np.sqrt((spread**2).sum(axis=1).mean())), 3)
    )
    record["mean_radial_bond_cosine"] = round(
        mean_radial_bond_cosine(positions_angstrom, bonds, centre_angstrom), 4
    )
    record["n_shells"] = len(profile_shells) + len(nested_shells)
    record["n_enclosed_shells"] = len(nested_shells)
    record["shell_radii_angstrom"] = " ".join(
        f"{radius:.2f}" for radius, _ in sorted(profile_shells + nested_shells)
    )
    record["core_atoms"] = (
        core_atom_count(positions_angstrom, centre_angstrom, profile_shells[0][0])
        if profile_shells
        else float("nan")
    )
    record.update({f"rings_{size}": int(rings[size]) for size in range(3, MAX_RING_SIZE + 1)})
    return record


def classify(record: dict[str, float | int | str]) -> str:
    """One of: dissociated, molten, diamond-like, graphitic onion, cage, disordered.

    Rules, applied in order, on the site-cutoff census of a single frame. "Low
    coordination" means two or fewer bonded neighbours; it is also the closure measure,
    since a closed sp2 shell has no edge atoms.
        1. dissociated: the largest fragment holds under half the atoms, or at least 80%
           of atoms are low-coordination: the cluster has unravelled into chains.
        2. molten: the largest fragment holds under 90% of atoms and at least 30% are
           low-coordination: a body still largely connected but shedding chains.
        3. diamond-like: at least 40% of atoms four-coordinated.
        4. graphitic onion: at least 80% three-coordinated, bonds lying in shells (mean
           radial bond cosine below 0.3, against 0.5 for an isotropic network), and two
           or more shells (`all_shells`: profile shells plus enclosed shells).
        5. cage: as rule 4 but exactly one shell, and closed: at most 20% of atoms
           low-coordination. Hollowness is not tested: interior atoms that form no shell
           (fewer than `SHELL_MIN_ATOMS`, or not tangentially bonded) are allowed.
        6. disordered: anything else.
    One frame cannot tell a liquid from a frozen network by its dynamics, so rule 2 uses
    the only frame-level sign of melting in vacuum: a connected body losing chains.
    Small clusters at 500 K are edge-rich (up to 55% low-coordination) yet whole, which
    is why rule 2 also asks for lost atoms.

    Args:
        record: A `census` record (values may be strings, as read back from CSV).

    Returns:
        The class name.
    """
    low_coordination = float(record["low_coordination_fraction"])
    largest = float(record["largest_fragment_fraction"])
    if is_dissociated(largest, low_coordination):
        return "dissociated"
    if (
        largest < MOLTEN_BELOW_LARGEST_FRAGMENT_FRACTION
        and low_coordination >= MOLTEN_ABOVE_LOW_COORDINATION_FRACTION
    ):
        return "molten"
    if float(record["sp3_fraction"]) >= DIAMOND_LIKE_ABOVE_SP3_FRACTION:
        return "diamond-like"
    shelled = (
        float(record["sp2_fraction"]) >= SHELLED_ABOVE_SP2_FRACTION
        and float(record["mean_radial_bond_cosine"]) < SHELLED_BELOW_RADIAL_BOND_COSINE
    )
    n_shells = int(record["n_shells"])
    if shelled and n_shells >= 2:
        return "graphitic onion"
    if shelled and n_shells == 1 and low_coordination <= CAGE_MAX_LOW_COORDINATION_FRACTION:
        return "cage"
    return "disordered"


def main() -> None:
    """Census of all 48 frames: census.csv and radial_profiles.csv beside the frames."""
    provenance = json.loads((SPHERES_DIR / "provenance.json").read_text())
    rows, profile_rows = [], []
    for temperature_kelvin in TEMPERATURES_KELVIN:
        for n_atoms in SIZES:
            name = f"C{n_atoms}-{temperature_kelvin}K"
            positions_angstrom, edge_angstrom = load_frame(name)
            source = provenance[f"{name}.xyz"]
            record = {
                "run": name,
                "n_atoms": n_atoms,
                "temperature_kelvin": temperature_kelvin,
                "nvt_time_at_last_frame_ps": source["nvt_time_at_last_frame_ps"],
                "run_completed": source["run_completed_25000_steps"],
                "minimiser": source["minimiser"].split()[0],
                **census(positions_angstrom, edge_angstrom),
            }
            record["class"] = classify(record)
            rows.append(record)
            adjacency = adjacency_matrix(
                n_atoms, bonds_at_cutoff(positions_angstrom, edge_angstrom)
            )
            centre_angstrom = largest_fragment_centre(positions_angstrom, adjacency)
            for radius_angstrom, density in zip(
                *radial_density_profile(positions_angstrom, centre_angstrom), strict=True
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
