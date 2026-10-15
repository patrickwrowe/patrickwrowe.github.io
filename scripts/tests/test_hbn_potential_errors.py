"""Tests for scripts/figures/hbn_potential_errors.py.

The committed hbn-potential-errors.json is the root of trust: the chart is checked against it
and against its own lower-bound flags, and the paper's pages are checked against it where the
paper is on this machine. Nothing here reads docs/, which is gitignored.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import unicodedata
from pathlib import Path

import hbn_potential_errors
import numpy as np
import pytest
from chart_svg import assert_text_fits, numeric_labels, panel_body
from figure_style import LABEL_SIZE, MARKER_RADIUS, MARKER_STROKE, NARROW_CHART_EM

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "scripts/figures/data/two-dimensional-and-confined"
FIGURES = REPO / "src/content/work/figures/two-dimensional-and-confined"
PAPER = REPO.parent / "papers/hbn-rippling/paper.pdf"
HEX_COLOUR = re.compile(r"#[0-9a-fA-F]{3}\b|#[0-9a-fA-F]{6}\b")
LAYOUTS = [(False, "hbn-potential-errors.svg"), (True, "hbn-potential-errors-narrow.svg")]
POTENTIALS = json.loads((DATA / "hbn-potential-errors.json").read_text())["potentials"]
KEY = {
    "force": "force_rmse_ev_per_angstrom",
    "energy": "formation_energy_rmse_mev_per_atom",
}


def read_dots(svg: str, quantity: str) -> dict[str, dict]:
    """Each row of a panel as drawn: its marker's value read off the axis, its labels.

    The axis is calibrated from the printed tick labels and where they sit: linear for the
    energies, in log10 for the forces.

    Args:
        svg: The figure.
        quantity: "force" or "energy".

    Returns:
        Potential name -> {"value", "kind", "bound", "label", "marker"}.
    """
    panel = panel_body(svg, panel=quantity)
    positions, values = numeric_labels(panel, "middle")
    slope, offset = np.polyfit(np.log10(values) if quantity == "force" else values, positions, 1)
    rows = {}
    for name, kind, bound, body in re.findall(
        r'<g data-potential="([^"]+)" data-kind="(\w+)"( data-bound="lower")?>(.*?)</g>',
        panel,
        re.DOTALL,
    ):
        marker_x = float(re.search(r'<circle cx="([\d.]+)"', body).group(1))
        position = (marker_x - offset) / slope
        texts = re.findall(r'text-anchor="(\w+)"[^>]*>([^<]*)</text>', body)
        rows[name] = {
            "value": 10**position if quantity == "force" else position,
            "kind": kind,
            "bound": bool(bound),
            "label": next(content for anchor, content in texts if anchor == "start"),
            "named": any(anchor == "end" and content == name for anchor, content in texts),
            "marker": re.search(r"<circle [^>]*/>", body).group(0),
        }
    return rows


@pytest.mark.parametrize(("narrow", "committed"), LAYOUTS)
def test_the_committed_svg_is_byte_for_byte_what_the_script_writes(narrow, committed):
    written = hbn_potential_errors.build(DATA, narrow=narrow).encode("utf-8")
    assert (FIGURES / committed).read_bytes() == written, f"{committed} is stale"


@pytest.mark.parametrize("narrow", [False, True])
@pytest.mark.parametrize("quantity", ["force", "energy"])
def test_every_dot_reads_back_as_its_value_on_the_axis_a_reader_sees(narrow, quantity):
    rows = read_dots(hbn_potential_errors.build(DATA, narrow=narrow), quantity)
    assert list(rows) == [potential["name"] for potential in POTENTIALS]
    for potential in POTENTIALS:
        expected = potential[KEY[quantity]]
        tolerance = {"rel": 2e-3} if quantity == "force" else {"abs": 0.2}
        assert rows[potential["name"]]["value"] == pytest.approx(expected, **tolerance)


def test_the_numbers_printed_beside_the_dots_are_the_data_with_the_lower_bounds_marked():
    svg = hbn_potential_errors.build(DATA)
    forces = read_dots(svg, "force")
    energies = read_dots(svg, "energy")
    assert {name: row["label"] for name, row in forces.items()} == {
        "hBN-GAP": "0.09",
        "Tersoff": "1.09",
        "ExTeP": "1.19",
        "ReaxFF": "6.17",
        "Tersoff+ILP": "1.11",
        "ExTeP+ILP": "1.22",
    }
    assert {name: row["label"] for name, row in energies.items()} == {
        "hBN-GAP": "3",
        "Tersoff": "&gt;54",
        "ExTeP": "&gt;67",
        "ReaxFF": "45",
        "Tersoff+ILP": "4",
        "ExTeP+ILP": "26",
    }
    assert [name for name, row in energies.items() if row["bound"]] == ["Tersoff", "ExTeP"]
    assert not any(row["bound"] for row in forces.values())


def test_the_chart_follows_the_flags_of_the_data_on_which_energies_are_lower_bounds():
    flagged = [
        potential["name"]
        for potential in POTENTIALS
        if potential["formation_energy_is_lower_bound"]
    ]
    assert flagged == ["Tersoff", "ExTeP"]  # paper p.8: "more than 54 and 67 meV/atom"
    energies = read_dots(hbn_potential_errors.build(DATA), "energy")
    for potential in POTENTIALS:
        row = energies[potential["name"]]
        assert row["bound"] == potential["formation_energy_is_lower_bound"]
        printed = f"{potential['formation_energy_rmse_mev_per_atom']:g}"
        assert row["label"] == ("&gt;" if row["bound"] else "") + printed


def test_hbn_gap_is_the_filled_marker_and_the_classical_potentials_are_open_in_ink_and_plate():
    svg = hbn_potential_errors.build(DATA)
    assert not HEX_COLOUR.search(svg)
    assert set(re.findall(r'(?:fill|stroke)="(var\(--[a-z]+\))"', svg)) == {
        "var(--ink)",
        "var(--graphite)",
        "var(--plate)",
    }
    for quantity in ("force", "energy"):
        for name, row in read_dots(svg, quantity).items():
            if name == "hBN-GAP":
                assert row["kind"] == "gap"
                assert f'r="{MARKER_RADIUS}" fill="var(--ink)"' in row["marker"]
            else:
                assert row["kind"] == "classical"
                assert 'fill="var(--plate)" stroke="var(--ink)"' in row["marker"]
                assert f'stroke-width="{MARKER_STROKE}"' in row["marker"]


def test_the_wide_chart_names_the_potentials_once_and_the_narrow_names_them_in_each_panel():
    for narrow, panels_with_names in ((False, ["force"]), (True, ["force", "energy"])):
        svg = hbn_potential_errors.build(DATA, narrow=narrow)
        named = [
            quantity
            for quantity in ("force", "energy")
            if all(row["named"] for row in read_dots(svg, quantity).values())
        ]
        assert named == panels_with_names


def test_every_axis_title_carries_its_unit():
    svg = hbn_potential_errors.build(DATA)
    assert "force RMSE (eV/Å, log scale)" in svg
    assert "formation-energy RMSE (meV per atom)" in svg


def test_the_order_of_magnitude_holds_for_every_classical_force_but_not_for_every_energy():
    # What a caption may copy, from the data and from the drawing alone. The paper says the
    # classical potentials are "one order of magnitude less accurate" for forces (p.7); for
    # formation energies the ILP variants close in (Tersoff+ILP 4 against 3 meV per atom).
    svg = hbn_potential_errors.build(DATA)
    ratios = {}
    for quantity in ("force", "energy"):
        rows = read_dots(svg, quantity)
        gap = rows["hBN-GAP"]["value"]
        ratios[quantity] = {
            name: row["value"] / gap for name, row in rows.items() if name != "hBN-GAP"
        }
    assert ratios["force"] == pytest.approx(
        {
            "Tersoff": 1.09 / 0.09,
            "ExTeP": 1.19 / 0.09,
            "ReaxFF": 6.17 / 0.09,
            "Tersoff+ILP": 1.11 / 0.09,
            "ExTeP+ILP": 1.22 / 0.09,
        },
        rel=0.02,
    )
    assert min(ratios["force"].values()) > 10
    assert ratios["energy"] == pytest.approx(
        {
            "Tersoff": 18.0,
            "ExTeP": 67 / 3,
            "ReaxFF": 15.0,
            "Tersoff+ILP": 4 / 3,
            "ExTeP+ILP": 26 / 3,
        },
        rel=0.05,
    )
    assert min(ratios["energy"].values()) < 10
    # Three of the five exceed ten times hBN-GAP's energy error; two of those are lower bounds.
    over_ten = [name for name, ratio in ratios["energy"].items() if ratio > 10]
    assert over_ten == ["Tersoff", "ExTeP", "ReaxFF"]


@pytest.mark.skipif(
    not (PAPER.exists() and shutil.which("pdftotext")), reason="no paper or no pdftotext here"
)
def test_the_data_are_what_pages_7_and_8_of_the_paper_say_including_the_lower_bounds():
    def page(number: int) -> str:
        text = subprocess.run(
            ["pdftotext", "-f", str(number), "-l", str(number), str(PAPER), "-"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        # pdftotext may write the angstrom sign (U+212B) or a letter and a ring.
        return " ".join(unicodedata.normalize("NFKC", text).split())

    forces, energies = page(7), page(8)
    assert "the hBN-GAP achieves a RMSE of 0.09 eV/Å" in forces
    assert "RMSEs of 1.09, 1.19, and 6.17 eV/Å" in forces
    assert "1.11 and 1.22 eV/Å for Tersoff+ILP and ExTeP+ILP" in forces
    assert "validation set of 1450" in forces
    assert "RMSE of more than 54 and 67 meV/atom, respectively" in energies
    assert "RMSE of 4 and 26 meV/atom" in energies
    assert "error of 45 meV/atom" in energies
    assert "RMSE of 3 meV/atom" in energies


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        (lambda potential: potential.pop("kind"), r"lacks \['kind'\]"),
        (lambda potential: potential.update(force_rmse_ev_per_angstrom=-1.0), "is -1.0"),
        (lambda potential: potential.update(force_rmse_ev_per_angstrom=float("nan")), "is nan"),
        (lambda potential: potential.update(force_rmse_ev_per_angstrom="6.17"), "is '6.17'"),
        (lambda potential: potential.update(force_rmse_ev_per_angstrom=True), "is True"),
        (
            lambda potential: potential.update(kind="GAP"),
            r"kind is 'GAP', not one of \['classical', 'gap'\]",
        ),
        (lambda potential: potential.update(force_rmse_ev_per_angstrom=60.0), "leaves its axis"),
        (lambda potential: potential.update(formation_energy_rmse_mev_per_atom=0.0), "is 0.0"),
    ],
)
def test_a_value_that_would_draw_off_its_panel_is_refused_naming_the_file_and_the_potential(
    tmp_path, corruption, message
):
    data = json.loads((DATA / "hbn-potential-errors.json").read_text())
    corruption(data["potentials"][2])
    (tmp_path / "hbn-potential-errors.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match=r"hbn-potential-errors\.json: ExTeP.*" + message):
        hbn_potential_errors.build(tmp_path)


@pytest.mark.parametrize(("narrow", "committed"), LAYOUTS)
def test_chart_em_is_the_viewbox_width_in_labels_and_the_narrow_one_fits_the_phone(
    narrow, committed
):
    svg = hbn_potential_errors.build(DATA, narrow=narrow)
    width, _height = map(float, re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg).groups())
    assert float(re.search(r'data-chart-em="([\d.]+)"', svg).group(1)) == pytest.approx(
        width / LABEL_SIZE, abs=0.05
    )
    if narrow:
        assert width == NARROW_CHART_EM * LABEL_SIZE
    else:
        assert width / LABEL_SIZE * 11 <= 700, "wider than a 700 px plate"
    assert_text_fits(svg)
