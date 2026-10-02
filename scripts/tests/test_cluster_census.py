"""Tests for scripts/figures/cluster_census.py on known graphs and real archive frames."""

from __future__ import annotations

import itertools
from pathlib import Path

import boxprep
import cluster_census as census
import numpy as np
import pytest

CLUSTERS = Path(__file__).resolve().parents[1] / "figures" / "data" / "carbon-clusters"
RUNS = [
    f"C{n_atoms}-{temperature_kelvin}K"
    for temperature_kelvin in census.TEMPERATURES_KELVIN
    for n_atoms in census.SIZES
]
# A free cluster not taken from a periodic dump gets a cell far wider than itself.
FREE_CLUSTER_EDGE_ANGSTROM = 100.0
GRAPHENE_AREAL_DENSITY_PER_ANGSTROM2 = 0.382
# Margins for measured real-frame values: fractions and cosines to 0.02 absolute, shell
# radii to 0.2 A. Wide enough for a re-extraction at five decimals, far narrower than the
# gaps between classes.
FRACTION_MARGIN = 0.02
RADIUS_MARGIN_ANGSTROM = 0.2

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
    """Adjacency matrix of a graph given by its edges.

    Args:
        n_atoms: Number of vertices.
        edges: Zero-based (first, second) vertex pairs.

    Returns:
        Symmetric boolean adjacency matrix, shape (n_atoms, n_atoms).
    """
    return census.adjacency_matrix(n_atoms, np.array(edges))


def sphere_points(radius_angstrom: float, n_points: int) -> np.ndarray:
    """Evenly spread points on a sphere about the origin (Fibonacci lattice).

    Args:
        radius_angstrom: Sphere radius, in angstrom.
        n_points: Number of points.

    Returns:
        Positions, shape (n_points, 3), in angstrom.
    """
    index = np.arange(n_points) + 0.5
    polar = np.arccos(1 - 2 * index / n_points)
    azimuth = np.pi * (1 + 5**0.5) * index
    return radius_angstrom * np.column_stack(
        [np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth), np.cos(polar)]
    )


def measured(record: dict[str, float | int | str], key: str) -> float:
    """One census value as a float, for assertions that print what was measured.

    Args:
        record: A `census.census` record.
        key: Field name.

    Returns:
        The value as a float.
    """
    return float(record[key])


def test_a_hexagon_of_real_bond_length_is_one_six_ring():
    angles = np.arange(6) * np.pi / 3
    positions_angstrom = 1.42 * np.column_stack([np.cos(angles), np.sin(angles), np.zeros(6)])
    bonds = census.bonds_at_cutoff(positions_angstrom, FREE_CLUSTER_EDGE_ANGSTROM)
    adjacency = census.adjacency_matrix(6, bonds)
    assert adjacency.sum() == 12
    counts = census.ring_counts(adjacency)
    assert counts[6] == 1 and counts.sum() == 1


def test_a_cube_has_six_square_faces_and_four_petrie_hexagons():
    # Brief's expectation was six 4-rings. By Franzblau's criterion the cube also has four
    # skew hexagons (each skirts one pair of opposite corners): every pair on them is as
    # close round the ring as through the cube. Rings2.f90's check agrees.
    corners = list(itertools.product((0, 1), repeat=3))
    edges = [
        (first, second)
        for first, second in itertools.combinations(range(8), 2)
        if sum(
            coordinate_first != coordinate_second
            for coordinate_first, coordinate_second in zip(
                corners[first], corners[second], strict=True
            )
        )
        == 1
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


def test_icosahedral_c60_is_a_cage_of_twelve_pentagons_and_twenty_hexagons():
    positions_angstrom = boxprep.read_xyz(CLUSTERS / "C60-Ih.xyz").positions_angstrom
    record = census.census(positions_angstrom, FREE_CLUSTER_EDGE_ANGSTROM)
    assert (record["rings_5"], record["rings_6"]) == (12, 20)
    assert record["sp2_fraction"] == 1.0
    # Every atom of this cage lies 3.41-3.47 A from its centre, so each bond is almost
    # exactly perpendicular to the radius through its midpoint, and the one shell sits at
    # the atoms' mean radius.
    assert measured(record, "mean_radial_bond_cosine") < 0.01
    assert record["n_shells"] == 1
    mean_radius_angstrom = np.linalg.norm(
        positions_angstrom - positions_angstrom.mean(axis=0), axis=1
    ).mean()
    assert float(record["shell_radii_angstrom"]) == pytest.approx(mean_radius_angstrom, abs=0.01)
    assert census.classify(record) == "cage"


@pytest.mark.parametrize("radii_angstrom", [(3.5, 7.0), (3.4, 6.8, 10.2), (4.0, 7.6)])
def test_shells_finds_each_concentric_sheet_and_ignores_a_trailing_chain(radii_angstrom):
    # Sheets at graphene's areal density, one interlayer spacing apart, plus a 10-atom
    # chain 4 A beyond the outermost: one shell per sheet, none for the chain.
    sheets = [
        sphere_points(radius, round(4 * np.pi * radius**2 * GRAPHENE_AREAL_DENSITY_PER_ANGSTROM2))
        for radius in radii_angstrom
    ]
    chain = np.column_stack([np.linspace(0, 12, 10), np.zeros(10), np.zeros(10)])
    chain[:, 2] += radii_angstrom[-1] + 4.0
    positions_angstrom = np.vstack([*sheets, chain])
    found = census.shells(positions_angstrom, np.zeros(3))
    assert [radius for radius, _ in found] == pytest.approx(list(radii_angstrom), abs=0.1)
    assert [count for _, count in found] == [len(sheet) for sheet in sheets]


def test_a_central_lump_is_not_a_shell():
    # 15 atoms within 1.2 A of the centre inside a 3.5 A sheet: fewer atoms and a smaller
    # radius than C20, the smallest closed cage, so only the sheet counts.
    lump = sphere_points(1.2, 15)
    sheet = sphere_points(3.5, 59)
    found = census.shells(np.vstack([lump, sheet]), np.zeros(3))
    assert len(found) == 1
    assert found[0][0] == pytest.approx(3.5, abs=0.1)


@pytest.mark.parametrize("run", RUNS)
def test_every_extracted_frame_is_whole(run):
    # extract_spheres.py unwraps each fragment along its periodic bonds, so the plain
    # Cartesian bond search on the frame on disk must find exactly the periodic bonds.
    positions_angstrom, edge_angstrom = census.load_frame(run)
    assert boxprep.find_bonds(positions_angstrom) == boxprep.find_bonds_periodic(
        positions_angstrom, edge_angstrom
    )


@pytest.mark.parametrize("temperature_kelvin", sorted(ARCHIVED_RINGS2_COUNTS))
def test_census_reproduces_the_archived_rings2_counts(temperature_kelvin):
    positions_angstrom, edge_angstrom = census.load_frame(f"C1000-{temperature_kelvin}K")
    bonds = census.bonds_at_cutoff(positions_angstrom, edge_angstrom, RINGS2_CUTOFF_ANGSTROM)
    counts = census.ring_counts(census.adjacency_matrix(1000, bonds), max_ring_size=20)
    assert counts[3:].tolist() == ARCHIVED_RINGS2_COUNTS[temperature_kelvin]


def test_ring_census_does_not_depend_on_atom_order():
    positions_angstrom, edge_angstrom = census.load_frame("C373-3000K")
    order = np.random.default_rng(7).permutation(len(positions_angstrom))
    counts = [
        census.ring_counts(
            census.adjacency_matrix(373, census.bonds_at_cutoff(frame, edge_angstrom))
        )
        for frame in (positions_angstrom, positions_angstrom[order])
    ]
    assert counts[0].tolist() == counts[1].tolist()
    assert counts[0][5:7].sum() > 100


def test_radial_profile_accounts_for_every_atom():
    positions_angstrom, _ = census.load_frame("C686-3000K")
    inner_radius_angstrom, density_per_angstrom3 = census.radial_density_profile(
        positions_angstrom, positions_angstrom.mean(axis=0)
    )
    shell_volume_angstrom3 = (
        4 / 3 * np.pi * ((inner_radius_angstrom + 1) ** 3 - inner_radius_angstrom**3)
    )
    assert (density_per_angstrom3 * shell_volume_angstrom3).sum() == pytest.approx(686)


def test_c60_at_2000k_is_a_single_closed_cage():
    record = census.census(*census.load_frame("C60-2000K"))
    assert census.classify(record) == "cage"
    assert record["n_shells"] == 1 and record["n_fragments"] == 1
    radius = float(record["shell_radii_angstrom"])
    assert radius == pytest.approx(3.64, abs=RADIUS_MARGIN_ANGSTROM), f"radius {radius}"
    for key, value in {
        "low_coordination_fraction": 0.117,
        "sp2_fraction": 0.883,
        "mean_radial_bond_cosine": 0.294,
    }.items():
        assert measured(record, key) == pytest.approx(value, abs=FRACTION_MARGIN), (
            f"{key} measured {record[key]}"
        )


def test_c1000_at_3000k_is_a_three_shell_graphitic_onion():
    record = census.census(*census.load_frame("C1000-3000K"))
    assert census.classify(record) == "graphitic onion"
    radii = [float(radius) for radius in str(record["shell_radii_angstrom"]).split()]
    assert radii == pytest.approx([3.98, 8.11, 11.74], abs=RADIUS_MARGIN_ANGSTROM), radii
    for key, value in {"sp2_fraction": 0.940, "mean_radial_bond_cosine": 0.228}.items():
        assert measured(record, key) == pytest.approx(value, abs=FRACTION_MARGIN), (
            f"{key} measured {record[key]}"
        )


def test_c1000_at_500k_is_disordered_with_a_minority_sp3():
    # The decks call this frame "diamond-like, high sp3 content". Measured at the site's
    # 1.824 A cutoff it is 81.3% three- and 7.9% four-coordinated; 13.9% of the atoms more
    # than 3.4 A inside the 98th-percentile radius are four-coordinated. Its bonds are
    # isotropic (radial cosine 0.40), so its density ripples do not make it an onion.
    record = census.census(*census.load_frame("C1000-500K"))
    assert census.classify(record) == "disordered"
    assert record["n_fragments"] == 1
    for key, value in {
        "sp3_fraction": 0.079,
        "sp2_fraction": 0.813,
        "interior_sp3_fraction": 0.139,
        "mean_radial_bond_cosine": 0.404,
    }.items():
        assert measured(record, key) == pytest.approx(value, abs=FRACTION_MARGIN), (
            f"{key} measured {record[key]}"
        )


def test_c686_at_4000k_is_molten():
    record = census.census(*census.load_frame("C686-4000K"))
    assert census.classify(record) == "molten"
    for key, value in {
        "largest_fragment_fraction": 0.638,
        "low_coordination_fraction": 0.555,
    }.items():
        assert measured(record, key) == pytest.approx(value, abs=FRACTION_MARGIN), (
            f"{key} measured {record[key]}"
        )


def test_c160_at_2000k_is_an_onion_by_its_off_centre_inner_cage():
    # A 20-atom fragment sits about 1.3 A off the outer cage's centre, so in the profile
    # about that centre it smears into the outer peak; about its own centre its bonds are
    # tangential (radial cosine 0.22), so it counts as a second, enclosed shell.
    positions_angstrom, edge_angstrom = census.load_frame("C160-2000K")
    record = census.census(positions_angstrom, edge_angstrom)
    assert census.classify(record) == "graphitic onion"
    assert (record["n_shells"], record["n_enclosed_shells"]) == (2, 1)
    radii = [float(radius) for radius in str(record["shell_radii_angstrom"]).split()]
    assert radii == pytest.approx([2.16, 5.51], abs=RADIUS_MARGIN_ANGSTROM), radii
    for key, value in {"sp2_fraction": 0.844, "mean_radial_bond_cosine": 0.284}.items():
        assert measured(record, key) == pytest.approx(value, abs=FRACTION_MARGIN), (
            f"{key} measured {record[key]}"
        )
    bonds = census.bonds_at_cutoff(positions_angstrom, edge_angstrom)
    labels = census.fragment_labels(census.adjacency_matrix(160, bonds))
    assert census.enclosed_shells(positions_angstrom, bonds, labels) == [
        (pytest.approx(2.16, abs=RADIUS_MARGIN_ANGSTROM), 20)
    ]


def test_c60_at_5000k_is_dissociated():
    record = census.census(*census.load_frame("C60-5000K"))
    assert census.classify(record) == "dissociated"
    # Where make_compact put the fragments is arbitrary, so no radius of gyration.
    assert np.isnan(measured(record, "radius_of_gyration_angstrom"))
    assert record["n_fragments"] == 7, f"fragments measured {record['n_fragments']}"
    largest = measured(record, "largest_fragment_fraction")
    assert largest == pytest.approx(0.367, abs=FRACTION_MARGIN), f"largest measured {largest}"


def test_census_csv_numeric_columns_parse_with_their_nan_values():
    # The pattern a figure script should use: name the columns. Whole-file type detection
    # (dtype=None without usecols) fails on the space-separated list columns
    # fragment_sizes and shell_radii_angstrom, not on the literal nan values.
    table = np.genfromtxt(
        census.SPHERES_DIR / "census.csv",
        delimiter=",",
        names=True,
        dtype=float,
        usecols=("n_atoms", "radius_of_gyration_angstrom", "sp3_fraction", "core_atoms"),
    )
    assert len(table) == 48
    assert np.isnan(table["radius_of_gyration_angstrom"]).sum() == 19
    assert not np.isnan(table["sp3_fraction"]).any()


def test_the_census_regenerates_the_committed_csvs_byte_for_byte(tmp_path):
    census.main(["--output-dir", str(tmp_path)])
    for name in ("census.csv", "radial_profiles.csv"):
        assert (tmp_path / name).read_bytes() == (census.SPHERES_DIR / name).read_bytes(), name


def test_an_unknown_option_exits_before_writing_anything():
    committed = {
        name: (census.SPHERES_DIR / name).stat().st_mtime_ns
        for name in ("census.csv", "radial_profiles.csv")
    }
    with pytest.raises(SystemExit) as exit_info:
        census.main(["--definitely-not-an-option"])
    assert exit_info.value.code == 2
    for name, mtime_ns in committed.items():
        assert (census.SPHERES_DIR / name).stat().st_mtime_ns == mtime_ns, name
