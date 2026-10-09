"""Tests for render_cluster.py: the CHO combustion series and the species palette."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import boxprep
import numpy as np
import pytest
import render_cluster
from cluster_svg import run_cli
from constants import COVALENT_RADIUS_ANGSTROM, SPECIES_COLOUR

REPO = Path(__file__).resolve().parents[2]
CHO_DATA = REPO / "scripts" / "figures" / "data" / "cho-gap"
CHO_FIGURES = REPO / "src" / "content" / "work" / "figures" / "cho-gap"
CARBON_DATA = REPO / "scripts" / "figures" / "data" / "carbon-clusters"
TOKENS = (REPO / "src" / "styles" / "tokens.css").read_text()
HEX_COLOUR = re.compile(r"#[0-9a-fA-F]{3,8}\b")
SERIES_FRAMES = [CHO_DATA / f"combustion-{time_ps}ps.xyz" for time_ps in (0, 10, 20, 50)]


def paints(svg: str) -> set[str]:
    """Every fill and stroke value in an SVG document."""
    return set(re.findall(r'(?:fill|stroke)="([^"]*)"', svg))


@pytest.mark.parametrize(
    ("committed", "extra_args"),
    [
        ("combustion-series.svg", []),
        ("combustion-series-narrow.svg", ["--narrow"]),
    ],
)
def test_the_committed_combustion_series_is_byte_for_byte_what_the_documented_command_writes(
    committed, extra_args, tmp_path, monkeypatch, capsys
):
    # The command in render_cluster.py's module docstring, run through the real CLI.
    written = tmp_path / committed
    argv = [
        "render_cluster.py",
        *map(str, SERIES_FRAMES),
        "--labels",
        "0 ps,10 ps,20 ps,50 ps",
        "--columns",
        "2",
        "--radius-scale",
        "2.2",
        *extra_args,
        "--output",
        str(written),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    render_cluster.main()
    capsys.readouterr()
    assert (CHO_FIGURES / committed).read_bytes() == written.read_bytes(), f"{committed} is stale"


@pytest.mark.parametrize("committed", ["combustion-series.svg", "combustion-series-narrow.svg"])
def test_the_combustion_series_paints_each_species_from_a_token_that_exists(committed):
    svg = (CHO_FIGURES / committed).read_text()
    species_tokens = set(re.findall(r"var\(--species-([a-z]+)\)", svg))
    assert species_tokens == {"c", "h", "o"}
    for symbol in species_tokens:
        assert f"--species-{symbol}:" in TOKENS
    assert not HEX_COLOUR.search(svg)
    assert "--lustre" not in svg  # the accent is state and anomaly, never a species
    assert paints(svg) <= {
        "var(--ink)",  # the group default
        "var(--graphite)",  # the labels
        "none",
        *(f"var(--species-{symbol})" for symbol in species_tokens),
    }


@pytest.mark.parametrize("name", ["C60-Ih.xyz", "C240-0.xyz"])
def test_an_all_carbon_cluster_draws_in_ink_with_no_species_colour(name):
    svg = render_cluster.render(boxprep.read_xyz(CARBON_DATA / name), (15, 25, 0))
    assert "--species" not in svg
    assert paints(svg) == {"var(--ink)", "none"}


def test_a_series_paints_carbon_the_same_in_every_panel_once_any_panel_has_a_second_species():
    # One C-H bond and one C-C bond, in angstrom.
    methane = boxprep.Structure(["C", "H"], np.array([[0.0, 0.0, 0.0], [1.09, 0.0, 0.0]]))
    graphite_bit = boxprep.Structure(["C", "C"], np.array([[0.0, 0.0, 0.0], [1.42, 0.0, 0.0]]))
    alone = render_cluster.render_series([("C2", graphite_bit)], (0, 0, 0))
    beside = render_cluster.render_series([("CH", methane), ("C2", graphite_bit)], (0, 0, 0))
    assert "--species" not in alone
    # The figure has hydrogen, so all three carbons are species-c, including the panel
    # that has no hydrogen of its own.
    assert beside.count('fill="var(--species-c)"') == 3
    assert beside.count("--species-h") == 2  # the hydrogen's disc and its half of the bond


def test_a_species_with_no_palette_entry_is_refused_by_name_not_painted_black(monkeypatch):
    monkeypatch.delitem(SPECIES_COLOUR, "O")
    oxygen_and_carbon = boxprep.Structure(["C", "O"], np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]))
    with pytest.raises(ValueError, match=r"\['O'\]"):
        render_cluster.render(oxygen_and_carbon, (0, 0, 0))


# The carbon figures, as the script drew them before it learnt nitrogen, magnesium and the rest
# of the palette (the SHA-256 of the SVG text, taken from the unchanged script): adding species
# must not move a single byte of an all-carbon figure.
CARBON_CLUSTER_NAMES = ["C60-Ih.xyz", "C240-0.xyz", "C540-0.xyz", "C720-0.xyz"]
CARBON_SHA256 = {
    "C60 single": "5a4b0b1e57b0c9e8aa26df7c6e850aa0ea20d20742c93702041df00046c5fc3f",
    "four-cluster series": "3b6f39fd55b54de03a58b6fb2e4d412e9e974ab6b0fa4f99ff668213c83ed8a2",
    "C720 slab": "3844160fe664b3f89b9df4266fc5fa34f2659d9053265210ba6c8e828603999c",
}
ROTATION_DEGREES = (15, 25, 0)


def sha256_of(svg: str) -> str:
    return hashlib.sha256(svg.encode()).hexdigest()


def test_the_all_carbon_svgs_are_byte_for_byte_what_the_script_drew_before_the_palette_grew():
    single = render_cluster.render(boxprep.read_xyz(CARBON_DATA / "C60-Ih.xyz"), ROTATION_DEGREES)
    series = render_cluster.render_series(
        [
            (name.split("-")[0], boxprep.read_xyz(CARBON_DATA / name))
            for name in CARBON_CLUSTER_NAMES
        ],
        ROTATION_DEGREES,
    )
    slab = render_cluster.render(
        boxprep.read_xyz(CARBON_DATA / "C720-0.xyz"), ROTATION_DEGREES, 1.0, 4.0
    )
    assert {
        "C60 single": sha256_of(single),
        "four-cluster series": sha256_of(series),
        "C720 slab": sha256_of(slab),
    } == CARBON_SHA256


@pytest.mark.parametrize("symbol", sorted(SPECIES_COLOUR))
def test_every_species_of_the_palette_has_a_draw_colour_and_a_radius(symbol):
    assert render_cluster.DRAW_COLOUR[symbol] in {"var(--ink)", "var(--graphite)"}
    assert render_cluster.DRAW_RADIUS_FACTOR[symbol] > 0
    # What was tuned by eye stays, and the rest is the covalent radius over carbon's.
    if symbol not in {"H", "O"}:
        assert render_cluster.DRAW_RADIUS_FACTOR[symbol] == pytest.approx(
            COVALENT_RADIUS_ANGSTROM[symbol] / COVALENT_RADIUS_ANGSTROM["C"]
        )


# A square-planar Mg(N)4 with a methyl carbon on one nitrogen: Mg-N 2.1 A (inside the radius
# rule's 2.54 A cutoff), N-C 1.47 A, in angstrom.
MG_N4_SPECIES = ["Mg", "N", "N", "N", "N", "C"]
MG_N4_POSITIONS = np.array(
    [[0.0, 0, 0], [2.1, 0, 0], [0, 2.1, 0], [-2.1, 0, 0], [0, -2.1, 0], [3.5, 0.5, 0]]
)


def test_a_structure_with_nitrogen_and_magnesium_draws_in_the_palette_and_not_as_a_keyerror():
    svg = render_cluster.render(boxprep.Structure(MG_N4_SPECIES, MG_N4_POSITIONS), (0, 0, 0))
    assert paints(svg) == {
        "var(--ink)",  # the group default
        "none",
        "var(--species-mg)",
        "var(--species-n)",
        "var(--species-c)",
    }
    # Four Mg-N bonds split in two halves, and the N-C bond.
    assert svg.count("<line") == 4 * 2 + 2
    assert svg.count('stroke="var(--species-mg)"') == 4
    radii = {
        symbol: max(
            float(radius)
            for radius in re.findall(
                rf'<circle[^>]*r="([\d.]+)"[^>]*fill="var\(--species-{symbol}\)"', svg
            )
        )
        for symbol in ("mg", "n")
    }
    assert radii["mg"] > radii["n"]  # magnesium draws the biggest disc


@pytest.mark.parametrize("symbol", ["N", "Mg"])
def test_a_figure_of_one_new_species_draws_in_ink_like_any_single_species_figure(symbol):
    pair = boxprep.Structure([symbol, symbol], np.array([[0.0, 0, 0], [1.4, 0, 0]]))
    assert paints(render_cluster.render(pair, (0, 0, 0))) == {"var(--ink)", "none"}


def write_xyz(path: Path, species: list[str], positions: np.ndarray) -> Path:
    """An XYZ file of the species at these positions in angstrom, in the archive's tab format."""
    rows = [
        f"{symbol} \t{x:.6f}\t{y:.6f}\t{z:.6f}\t"
        for symbol, (x, y, z) in zip(species, positions, strict=True)
    ]
    path.write_text("\n".join([f"{len(species)}\t", "", *rows]) + "\n")
    return path


def test_hide_hydrogens_leaves_no_hydrogen_in_the_drawing_and_says_how_many_atoms_remain(
    tmp_path, monkeypatch, capsys
):
    # Methanol-like: C-O 1.43 A, three C-H and one O-H.
    species = ["C", "O", "H", "H", "H", "H"]
    positions = np.array(
        [
            [0.0, 0, 0],
            [1.43, 0, 0],
            [-0.36, 1.03, 0],
            [-0.36, -0.51, 0.89],
            [-0.36, -0.51, -0.89],
            [1.74, 0.9, 0],
        ]
    )
    source = write_xyz(tmp_path / "methanol.xyz", species, positions)
    shown, hidden = tmp_path / "shown.svg", tmp_path / "hidden.svg"
    run_cli(monkeypatch, capsys, str(source), "--output", str(shown))
    printed = run_cli(monkeypatch, capsys, str(source), "--hide-hydrogens", "--output", str(hidden))
    assert "--species-h" in shown.read_text()
    assert "--species-h" not in hidden.read_text()
    assert hidden.read_text().count("<circle") == 2
    assert "2 atoms, C1 O1" in printed


def test_hiding_every_atom_is_refused_by_name(tmp_path, monkeypatch, capsys):
    source = write_xyz(tmp_path / "hydrogen.xyz", ["H", "H"], np.array([[0.0, 0, 0], [0.74, 0, 0]]))
    monkeypatch.setattr(
        sys,
        "argv",
        ["render_cluster.py", str(source), "--hide-hydrogens", "--output", str(tmp_path / "x.svg")],
    )
    with pytest.raises(SystemExit):
        render_cluster.main()
    assert "hydrogen.xyz: nothing is left to draw" in capsys.readouterr().err


def test_together_draws_two_files_as_one_cluster_each_atom_where_its_file_puts_it(
    tmp_path, monkeypatch, capsys
):
    # Two C-C pairs 5 A apart along x, in one frame.
    first = write_xyz(tmp_path / "first.xyz", ["C", "C"], np.array([[0.0, 0, 0], [1.4, 0, 0]]))
    second = write_xyz(tmp_path / "second.xyz", ["C", "C"], np.array([[6.4, 0, 0], [7.8, 0, 0]]))
    together, apart = tmp_path / "together.svg", tmp_path / "apart.svg"
    run_cli(monkeypatch, capsys, str(first), str(second), "--together", "--output", str(together))
    run_cli(monkeypatch, capsys, str(first), str(second), "--output", str(apart))

    def circles(svg: str) -> np.ndarray:
        return np.array(
            [
                [float(x), float(y)]
                for x, y in re.findall(r'<circle cx="([-\d.]+)" cy="([-\d.]+)"', svg)
            ]
        )

    drawn = circles(together.read_text())
    assert drawn.shape == (4, 2)
    # Centred on the cluster's mean, 3.9 A from the first atom, and the 5 A gap is kept.
    assert sorted(drawn[:, 0]) == pytest.approx([-3.9, -2.5, 2.5, 3.9], abs=1e-3)
    # As panels of a series the same two files are laid side by side with a label each.
    assert apart.read_text().count("<text") == 2
    assert together.read_text().count("<text") == 0


def test_two_structures_in_a_series_share_one_scale_and_one_view(tmp_path, monkeypatch, capsys):
    small = write_xyz(tmp_path / "small.xyz", ["Mg", "N"], np.array([[0.0, 0, 0], [2.1, 0, 0]]))
    large = write_xyz(
        tmp_path / "large.xyz", ["Mg", "N", "N"], np.array([[0.0, 0, 0], [2.1, 0, 0], [-6.0, 0, 0]])
    )
    written = tmp_path / "pair.svg"
    run_cli(
        monkeypatch,
        capsys,
        str(small),
        str(large),
        "--labels",
        "small,large",
        "--output",
        str(written),
    )
    svg = written.read_text()
    assert svg.count("<svg") == 1  # one document, one viewBox, hence one scale
    assert len(set(re.findall(r'r="([\d.]+)" stroke="none" fill="var\(--species-mg\)"', svg))) == 1
    assert "var(--species-mg)" in svg
