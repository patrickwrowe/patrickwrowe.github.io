"""Tests for scripts/figures/vermiculite_contact_angles.py."""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pytest
import vermiculite_contact_angles
from chart_svg import assert_text_fits, panel_body, read_curves
from figure_style import LABEL_SIZE, NARROW_CHART_EM
from xlsx_source_data import read_sheet

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "scripts/figures/data/two-dimensional-and-confined"
FIGURES = REPO / "src/content/work/figures/two-dimensional-and-confined"
# The papers live outside the repository, wherever PAPERS_DIR says; unset, their tests skip.
PAPERS_DIR = os.environ.get("PAPERS_DIR")
WORKBOOK = (
    Path(PAPERS_DIR, "vermiculite-membranes/originals/nature_commun_source_data.xlsx")
    if PAPERS_DIR
    else None
)
HEX_COLOUR = re.compile(r"#[0-9a-fA-F]{3}\b|#[0-9a-fA-F]{6}\b")
LAYOUTS = [
    (False, "vermiculite-contact-angles.svg"),
    (True, "vermiculite-contact-angles-narrow.svg"),
]
PANELS = [(angle, cation) for angle in ("theta", "phi") for cation in ("K", "Li")]
FILE_OF = {"theta": "oh_vector", "phi": "hoh_plane"}
SHEET_OF = {"theta": "Fig. 4c", "phi": "Fig. 4d"}
# Columns of the sheets: cos, layer 1, cos, layer 2 for KV, then the same for LiV.
SHEET_COLUMNS = {"K": [0, 1, 2, 3], "Li": [4, 5, 6, 7]}


def load(angle: str, cation: str, fold: bool = True) -> np.ndarray:
    """A data file as it is committed; phi's mirror half is dropped when `fold`."""
    columns = np.loadtxt(DATA / f"{cation.lower()}_{FILE_OF[angle]}_angle_layer1_layer2_data.dat")
    return columns[columns[:, 0] >= -1e-3] if fold and angle == "phi" else columns


@pytest.mark.parametrize(("narrow", "committed"), LAYOUTS)
def test_the_committed_svg_is_byte_for_byte_what_the_script_writes(narrow, committed):
    written = vermiculite_contact_angles.build(DATA, narrow=narrow).encode("utf-8")
    assert (FIGURES / committed).read_bytes() == written, f"{committed} is stale"


@pytest.mark.parametrize("narrow", [False, True])
@pytest.mark.parametrize(("angle", "cation"), PANELS)
def test_every_curve_is_its_column_of_the_data_to_the_precision_of_the_drawing(
    narrow, angle, cation
):
    svg = vermiculite_contact_angles.build(DATA, narrow=narrow)
    curves = read_curves(panel_body(svg, panel=cation, variable=angle))
    columns = load(angle, cation)
    assert set(curves) == {"layer1", "layer2"}
    for layer, column in (("layer1", 1), ("layer2", 3)):
        cosine, probability_au = curves[layer]
        assert len(cosine) == len(columns)
        assert cosine == pytest.approx(columns[:, 0], abs=5e-4)
        assert probability_au == pytest.approx(columns[:, column], abs=5e-4)


def test_each_cation_is_drawn_in_its_own_colour_the_first_layer_solid_and_the_second_dashed():
    svg = vermiculite_contact_angles.build(DATA)
    assert not HEX_COLOUR.search(svg)
    for angle, cation in PANELS:
        panel = panel_body(svg, panel=cation, variable=angle)
        series = dict(re.findall(r'<g data-series="(\w+)">(.*?)</g>', panel, re.DOTALL))
        for layer, dash in (("layer1", None), ("layer2", vermiculite_contact_angles.DASHED_SERIES)):
            assert f'stroke="var(--species-{cation.lower()})"' in series[layer], (angle, cation)
            assert (f'stroke-dasharray="{dash}"' in series[layer]) == (dash is not None)
        # The panel's swatch beside its name is the same colour.
        swatch = re.search(r'<line [^>]*stroke="([^"]+)"[^>]*/>', panel).group(1)
        assert swatch == f"var(--species-{cation.lower()})"
        assert f">{cation}-vermiculite<" in panel


def test_the_legend_samples_are_the_two_line_styles_in_the_neutral_stroke():
    svg = vermiculite_contact_angles.build(DATA)
    entries = {
        layer: (
            re.search(r'stroke="([^"]+)"', body).group(1),
            "stroke-dasharray" in body,
            re.search(r">([^<]+)</text>", body).group(1),
        )
        for layer, body in re.findall(r'<g data-legend="(\w+)">(.*?)</g>', svg, re.DOTALL)
    }
    assert entries == {
        "layer1": ("var(--graphite)", False, "first contact layer"),
        "layer2": ("var(--graphite)", True, "second contact layer"),
    }


@pytest.mark.parametrize(
    ("angle", "cation", "layer", "peak_cosine", "peak_au"),
    [
        # What a caption may say: where each distribution peaks and how high.
        ("theta", "K", 1, 0.687, 0.2350),
        ("theta", "K", 3, -0.018, 0.1309),
        ("theta", "Li", 1, 0.027, 0.1476),
        ("theta", "Li", 3, -0.003, 0.1519),
        ("phi", "K", 1, 0.0, 0.1227),
        ("phi", "Li", 1, 0.98, 0.0609),
    ],
)
def test_the_highest_point_of_a_distribution_is_where_the_caption_would_put_it(
    angle, cation, layer, peak_cosine, peak_au
):
    columns = load(angle, cation)
    assert columns[columns[:, layer].argmax(), 0] == pytest.approx(peak_cosine, abs=5e-4)
    assert columns[:, layer].max() == pytest.approx(peak_au, abs=5e-5)

    svg = vermiculite_contact_angles.build(DATA)
    cosine, probability_au = read_curves(panel_body(svg, panel=cation, variable=angle))[
        f"layer{(layer + 1) // 2}"
    ]
    assert cosine[probability_au.argmax()] == pytest.approx(peak_cosine, abs=1e-3)
    assert probability_au.max() == pytest.approx(peak_au, abs=5e-4)


@pytest.mark.parametrize(
    ("angle", "k_difference_au", "li_difference_au"),
    [("theta", 0.2148, 0.0539), ("phi", 0.0922, 0.0152)],
)
def test_the_k_layers_differ_four_to_six_times_more_than_the_li_layers(
    angle, k_difference_au, li_difference_au
):
    # The largest gap between a cation's two layers, which a caption may call "four times".
    gaps = {}
    for cation in ("K", "Li"):
        columns = load(angle, cation)
        gaps[cation] = np.abs(columns[:, 1] - columns[:, 3]).max()
    assert gaps["K"] == pytest.approx(k_difference_au, abs=5e-5)
    assert gaps["Li"] == pytest.approx(li_difference_au, abs=5e-5)
    assert 3.9 < gaps["K"] / gaps["Li"] < 6.1


def test_k_theta_first_layer_is_not_a_unit_normalised_density_and_dips_below_zero():
    # The docstring's two claims about the axis: arbitrary units, and an overshoot drawn.
    columns = load("theta", "K")
    assert np.trapezoid(columns[:, 1], columns[:, 0]) == pytest.approx(0.114, abs=5e-4)
    assert columns[:, 1].min() == pytest.approx(-0.0726, abs=5e-5)
    _cosine, probability_au = read_curves(
        panel_body(vermiculite_contact_angles.build(DATA), panel="K", variable="theta")
    )["layer1"]
    assert probability_au.min() == pytest.approx(-0.0726, abs=5e-4)


@pytest.mark.parametrize("cation", ["k", "li"])
def test_the_phi_files_are_mirror_symmetric_so_the_half_from_zero_loses_nothing(cation):
    columns = load("phi", cation, fold=False)
    assert columns.shape == (100, 4)
    for layer in (1, 3):
        assert columns[50:, layer] == pytest.approx(columns[:50, layer][::-1], abs=1e-12)
    assert (columns[:, 0] == columns[:, 2]).all()
    assert len(load("phi", cation)) == 50


@pytest.mark.skipif(
    not (WORKBOOK and WORKBOOK.exists()), reason="the Source Data workbook is not on this machine"
)
@pytest.mark.parametrize(("angle", "cation"), PANELS)
def test_the_data_are_the_source_data_sheets_of_fig_4c_and_4d_to_the_sheets_rounding(angle, cation):
    header, body = read_sheet(WORKBOOK, SHEET_OF[angle])
    assert header[0] == f"Cos ({'θ' if angle == 'theta' else 'φ'})"
    columns = SHEET_COLUMNS[cation]
    sheet = body[:, columns]
    committed = load(angle, cation)
    assert committed.shape == sheet.shape
    # Five decimal places in the sheet, so the files differ from it by at most 5e-6.
    assert np.abs(committed - sheet).max() <= 5e-6 + 1e-12


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        (lambda columns: columns[:, :3], "4 finite columns"),
        (lambda columns: np.where(columns == columns[7, 1], np.nan, columns), "4 finite columns"),
        (lambda columns: columns + [[0, 0, 0.001, 0]], "cosine columns differ"),
        (lambda columns: columns * [[1, 4, 1, 1]], "layer1 leaves the axes"),
    ],
)
def test_a_data_file_that_would_draw_outside_its_panel_is_refused_naming_the_file(
    tmp_path, corruption, message
):
    for angle, cation in PANELS:
        name = f"{cation.lower()}_{FILE_OF[angle]}_angle_layer1_layer2_data.dat"
        np.savetxt(tmp_path / name, load(angle, cation, fold=False))
    path = tmp_path / "k_oh_vector_angle_layer1_layer2_data.dat"
    np.savetxt(path, corruption(np.loadtxt(path)))
    with pytest.raises(ValueError, match=re.escape(path.name) + ".*" + message):
        vermiculite_contact_angles.build(tmp_path)


@pytest.mark.parametrize(("narrow", "committed"), LAYOUTS)
def test_chart_em_is_the_viewbox_width_in_labels_and_the_narrow_one_fits_the_phone(
    narrow, committed
):
    svg = vermiculite_contact_angles.build(DATA, narrow=narrow)
    width, _height = map(float, re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg).groups())
    assert float(re.search(r'data-chart-em="([\d.]+)"', svg).group(1)) == pytest.approx(
        width / LABEL_SIZE, abs=0.05
    )
    if narrow:
        assert width == NARROW_CHART_EM * LABEL_SIZE
    else:
        assert width / LABEL_SIZE * 11 <= 700, "wider than a 700 px plate"
    assert_text_fits(svg)
