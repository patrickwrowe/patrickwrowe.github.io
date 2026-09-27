"""Tests for scripts/figures/cluster_census.py on known graphs and real archive frames."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))
import boxprep  # noqa: E402
import cluster_census as census  # noqa: E402

CLUSTERS = Path(__file__).resolve().parents[1] / "figures" / "data" / "carbon-clusters"
SPHERES = CLUSTERS / "spheres"

# Shortest-path ring counts for sizes 3 to 20 printed by the archive's Rings2.f90 (bond
# cutoff 1.85 A, rings up to 20) for the last frame of each C1000 trajectory, from
# Carbon_Cluster_RSS/4_Analysis/3_GAP_Spheres_Opt2/<T>K-rings.out.
ARCHIVED_RINGS2_COUNTS = {
    500: [6, 17, 96, 118, 87, 65, 48, 49, 45, 52, 40, 39, 15, 19, 2, 2, 0, 2],
    1000: [2, 14, 139, 141, 104, 59, 45, 42, 45, 37, 41, 18, 13, 9, 8, 12, 6, 0],
    3000: [3, 1, 111, 246, 85, 16, 1, 4, 0, 5, 5, 0, 1, 2, 3, 1, 7, 0],
}
RINGS2_CUTOFF_ANGSTROM = 1.85


def graph(n_atoms: int, edges: list[tuple[int, int]]) -> np.ndarray:
    return census.adjacency_matrix(n_atoms, np.array(edges))


def test_a_hexagon_of_real_bond_length_is_one_six_ring():
    angles = np.arange(6) * np.pi / 3
    positions_angstrom = 1.42 * np.column_stack([np.cos(angles), np.sin(angles), np.zeros(6)])
    adjacency = census.adjacency_matrix(6, census.bonds_at_cutoff(positions_angstrom))
    assert adjacency.sum() == 12
    counts = census.ring_counts(adjacency)
    assert counts[6] == 1 and counts.sum() == 1


def test_a_cube_has_six_square_faces_and_four_petrie_hexagons():
    # Brief's expectation was six 4-rings. By Franzblau's criterion the cube also has four
    # skew hexagons (each skirts one pair of opposite corners): every pair on them is as
    # close round the ring as through the cube. Rings2.f90's check agrees.
    corners = list(itertools.product((0, 1), repeat=3))
    edges = [
        (i, j)
        for i, j in itertools.combinations(range(8), 2)
        if sum(a != b for a, b in zip(corners[i], corners[j], strict=True)) == 1
    ]
    counts = census.ring_counts(graph(8, edges))
    assert counts[4] == 6
    assert counts[6] == 4
    assert counts.sum() == 10


def test_two_separated_triangles_are_two_fragments_of_three():
    adjacency = graph(6, [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)])
    assert census.fragment_sizes(adjacency).tolist() == [3, 3]
    assert census.ring_counts(adjacency)[3] == 2


def test_a_single_atom_is_its_own_fragment():
    adjacency = graph(4, [(0, 1), (1, 2)])
    assert census.fragment_sizes(adjacency).tolist() == [3, 1]
    assert census.coordination_fractions(adjacency).tolist() == [0.25, 0.5, 0.25, 0, 0, 0]


def test_icosahedral_c60_has_twelve_pentagons_twenty_hexagons_and_tangential_bonds():
    positions_angstrom = boxprep.read_xyz(CLUSTERS / "C60-Ih.xyz").positions_angstrom
    bonds = census.bonds_at_cutoff(positions_angstrom)
    adjacency = census.adjacency_matrix(60, bonds)
    counts = census.ring_counts(adjacency)
    assert (counts[5], counts[6], counts.sum()) == (12, 20, 32)
    assert census.coordination_fractions(adjacency)[3] == 1.0
    # Every atom of a perfect cage is equidistant from its centre, so each bond is exactly
    # perpendicular to the radius through its midpoint.
    assert census.mean_radial_bond_cosine(positions_angstrom, bonds) < 0.01


@pytest.mark.parametrize("temperature_kelvin", sorted(ARCHIVED_RINGS2_COUNTS))
def test_census_reproduces_the_archived_rings2_counts(temperature_kelvin):
    positions_angstrom = boxprep.read_xyz(
        SPHERES / f"C1000-{temperature_kelvin}K.xyz"
    ).positions_angstrom
    bonds = census.bonds_at_cutoff(positions_angstrom, RINGS2_CUTOFF_ANGSTROM)
    counts = census.ring_counts(census.adjacency_matrix(1000, bonds), max_ring_size=20)
    assert counts[3:].tolist() == ARCHIVED_RINGS2_COUNTS[temperature_kelvin]


def test_ring_census_does_not_depend_on_atom_order():
    positions_angstrom = boxprep.read_xyz(SPHERES / "C373-3000K.xyz").positions_angstrom
    order = np.random.default_rng(7).permutation(len(positions_angstrom))
    counts = [
        census.ring_counts(census.adjacency_matrix(373, census.bonds_at_cutoff(frame)))
        for frame in (positions_angstrom, positions_angstrom[order])
    ]
    assert counts[0].tolist() == counts[1].tolist()
    assert counts[0][5:7].sum() > 100


def test_radial_profile_accounts_for_every_atom():
    positions_angstrom = boxprep.read_xyz(SPHERES / "C686-3000K.xyz").positions_angstrom
    inner_radius_angstrom, density_per_angstrom3 = census.radial_density_profile(positions_angstrom)
    shell_volume_angstrom3 = (
        4 / 3 * np.pi * ((inner_radius_angstrom + 1) ** 3 - inner_radius_angstrom**3)
    )
    assert (density_per_angstrom3 * shell_volume_angstrom3).sum() == pytest.approx(686)


def test_c1000_at_500k_is_mostly_three_coordinated_with_a_minority_sp3():
    # The decks call this frame "diamond-like, high sp3 content". Measured at the site's
    # 1.824 A cutoff it is 81.3% three-coordinated and 7.9% four-coordinated.
    record = census.census(boxprep.read_xyz(SPHERES / "C1000-500K.xyz").positions_angstrom)
    measured = float(record["sp3_fraction"])
    assert measured == pytest.approx(0.079, abs=0.02), f"sp3 fraction measured {measured}"
    assert record["n_fragments"] == 1


def test_a_hot_small_cluster_is_classed_dissociated():
    record = census.census(boxprep.read_xyz(SPHERES / "C60-5000K.xyz").positions_angstrom)
    assert record["n_fragments"] > 1
    assert census.classify(record) == "dissociated"
