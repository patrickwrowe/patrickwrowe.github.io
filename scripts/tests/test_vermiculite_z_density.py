"""Tests for scripts/figures/vermiculite_z_density.py."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import vermiculite_z_density
from figure_style import LABEL_SIZE, NARROW_CHART_EM
from xlsx_source_data import read_sheet

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "scripts/figures/data/two-dimensional-and-confined"
FIGURES = REPO / "src/content/work/figures/two-dimensional-and-confined"
WORKBOOK = REPO.parent / "papers/vermiculite-membranes/originals/nature_commun_source_data.xlsx"
HEX_COLOUR = re.compile(r"#[0-9a-fA-F]{3}\b|#[0-9a-fA-F]{6}\b")
SHEETS = {"K": "Fig. 4a", "Li": "Fig.4b"}
# Column pair (density, height) of each series in the data files.
COLUMNS = {"ion": 0, "oxygen": 2, "hydrogen": 4}
LAYOUTS = [(False, "vermiculite-z-density.svg"), (True, "vermiculite-z-density-narrow.svg")]


def read_numbers(svg_fragment: str, anchor: str) -> tuple[np.ndarray, np.ndarray]:
    """Positions and values of the numeric tick labels with one text anchor.

    Args:
        svg_fragment: SVG text holding the `<text>` elements.
        anchor: "middle" for the bottom axis's labels, "end" for the left axis's.

    Returns:
        (the labels' x for a bottom axis or baseline y for a left axis, in viewBox units;
        the numbers they print). A left label's baseline sits 1.0 below its tick.
    """
    labels = re.findall(
        rf'<text x="([\d.]+)" y="([\d.]+)" text-anchor="{anchor}" [^>]*>([^<]*)</text>',
        svg_fragment,
    )
    numeric = [
        (float(x), float(y), float(content.replace("−", "-")))
        for x, y, content in labels
        if re.fullmatch(r"−?[\d.]+", content)
    ]
    positions = [x if anchor == "middle" else y - 1.0 for x, y, _ in numeric]
    return np.array(positions), np.array([value for _, _, value in numeric])


def drawn_curves(svg: str, cation: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read each curve of a panel back into data units from the drawing alone.

    The axes are calibrated from the printed tick labels and their positions, so this checks
    what a reader would read, whatever function placed the points.

    Args:
        svg: The figure.
        cation: The panel, "K" or "Li".

    Returns:
        Series key -> (density in a.u., height in angstrom) per drawn point.
    """
    panel = re.search(rf'<g data-panel="{cation}">(.*?)\n  </g>', svg, re.DOTALL).group(1)
    x_units, densities = read_numbers(panel, "middle")
    y_units, heights = read_numbers(panel, "end")
    density_slope, density_offset = np.polyfit(densities, x_units, 1)
    height_slope, height_offset = np.polyfit(heights, y_units, 1)
    curves = {}
    for series, body in re.findall(r'<g data-series="(\w+)">(.*?)</g>', panel, re.DOTALL):
        points = np.array(
            [
                [float(number) for number in pair.split(",")]
                for pair in re.search(r'points="([^"]+)"', body).group(1).split()
            ]
        )
        curves[series] = (
            (points[:, 0] - density_offset) / density_slope,
            (points[:, 1] - height_offset) / height_slope,
        )
    return curves


@pytest.mark.parametrize(("narrow", "committed"), LAYOUTS)
def test_the_committed_svg_is_byte_for_byte_what_the_script_writes(narrow, committed):
    written = vermiculite_z_density.build(DATA, narrow=narrow).encode("utf-8")
    assert (FIGURES / committed).read_bytes() == written, f"{committed} is stale"


@pytest.mark.parametrize("narrow", [False, True])
def test_every_curve_is_its_column_of_the_data_to_the_precision_of_the_drawing(narrow):
    svg = vermiculite_z_density.build(DATA, narrow=narrow)
    for cation, data_file in vermiculite_z_density.DATA_FILES.items():
        columns = np.loadtxt(DATA / data_file)
        curves = drawn_curves(svg, cation)
        assert set(curves) == set(COLUMNS)
        for series, first in COLUMNS.items():
            density_au, height_angstrom = curves[series]
            assert density_au == pytest.approx(columns[:, first], abs=1e-5), (cation, series)
            assert height_angstrom == pytest.approx(columns[:, first + 1], abs=5e-3), (
                cation,
                series,
            )


def test_each_panel_draws_the_ion_in_its_own_colour_and_the_water_in_o_and_h_colours():
    svg = vermiculite_z_density.build(DATA)
    assert not HEX_COLOUR.search(svg)
    for cation, ion_token in (("K", "var(--species-k)"), ("Li", "var(--species-li)")):
        panel = re.search(rf'<g data-panel="{cation}">(.*?)\n  </g>', svg, re.DOTALL).group(1)
        strokes = {
            series: re.search(r'stroke="([^"]+)"', body).group(1)
            for series, body in re.findall(r'<g data-series="(\w+)">(.*?)</g>', panel, re.DOTALL)
        }
        assert strokes == {
            "ion": ion_token,
            "oxygen": "var(--species-o)",
            "hydrogen": "var(--species-h)",
        }


def test_the_legend_of_each_panel_names_the_colour_of_each_series():
    svg = vermiculite_z_density.build(DATA)
    for cation in vermiculite_z_density.DATA_FILES:
        panel = re.search(rf'<g data-panel="{cation}">(.*?)\n  </g>', svg, re.DOTALL).group(1)
        entries = {
            species: (
                re.search(r'stroke="([^"]+)"', body).group(1),
                re.search(r">([^<]+)</text>", body).group(1),
            )
            for species, body in re.findall(r'<g data-legend="(\w+)">(.*?)</g>', panel, re.DOTALL)
        }
        assert entries == {
            cation: (f"var(--species-{cation.lower()})", f"{cation} ion"),
            "O": ("var(--species-o)", "water O"),
            "H": ("var(--species-h)", "water H"),
        }


@pytest.mark.parametrize(
    ("cation", "series", "peak_height_angstrom", "peak_density_au"),
    [
        # What a caption may say: where the highest point of each curve is, and how high.
        ("K", "ion", 1.316, 0.03133),
        ("K", "oxygen", 2.506, 0.02754),
        ("Li", "ion", -0.193, 0.0295),
        ("Li", "oxygen", 2.837, 0.02073),
    ],
)
def test_the_highest_point_of_a_curve_is_where_the_caption_would_put_it(
    cation, series, peak_height_angstrom, peak_density_au
):
    columns = np.loadtxt(DATA / vermiculite_z_density.DATA_FILES[cation])
    density_au = columns[:, COLUMNS[series]]
    height_angstrom = columns[:, COLUMNS[series] + 1]
    assert height_angstrom[density_au.argmax()] == pytest.approx(peak_height_angstrom, abs=5e-4)
    assert density_au.max() == pytest.approx(peak_density_au, abs=5e-5)

    drawn_density_au, drawn_height_angstrom = drawn_curves(
        vermiculite_z_density.build(DATA), cation
    )[series]
    assert drawn_height_angstrom[drawn_density_au.argmax()] == pytest.approx(
        peak_height_angstrom, abs=5e-3
    )


def test_li_has_two_ion_sites_where_k_has_one():
    # Local maxima above a fifth of the curve's highest point, read from the drawing and
    # grouped into sites: peaks within half an angstrom of each other are one site.
    svg = vermiculite_z_density.build(DATA)
    sites = {}
    for cation in vermiculite_z_density.DATA_FILES:
        density_au, height_angstrom = drawn_curves(svg, cation)["ion"]
        interior = np.arange(1, len(density_au) - 1)
        is_peak = (
            (density_au[interior] > density_au[interior - 1])
            & (density_au[interior] >= density_au[interior + 1])
            & (density_au[interior] > 0.2 * density_au.max())
        )
        peak_heights = height_angstrom[interior[is_peak]]
        sites[cation] = np.split(peak_heights, np.where(np.diff(peak_heights) > 0.5)[0] + 1)
    assert len(sites["K"]) == 1
    assert sites["K"][0].mean() == pytest.approx(1.32, abs=0.05)
    assert [site.min() for site in sites["Li"]] == pytest.approx([-0.193, 1.763], abs=5e-3)


def test_the_densities_are_histograms_summing_to_one_with_a_bin_width_that_differs_by_species():
    # Heights of an ion, an O and an H curve are therefore not comparable (the docstring says
    # so); positions are.
    expected_bin_width_angstrom = {
        "K": {"ion": 0.0127, "oxygen": 0.1335, "hydrogen": 0.1531},
        "Li": {"ion": 0.0321, "oxygen": 0.1407, "hydrogen": 0.1562},
    }
    for cation, data_file in vermiculite_z_density.DATA_FILES.items():
        columns = np.loadtxt(DATA / data_file)
        for series, first in COLUMNS.items():
            assert columns[:, first].sum() == pytest.approx(1.0, abs=0.01), (cation, series)
            bin_width = np.diff(columns[:, first + 1]).mean()
            assert bin_width == pytest.approx(expected_bin_width_angstrom[cation][series], abs=5e-5)
        assert (np.diff(columns[:, [1, 3, 5]], axis=0) > 0).all(), "heights must rise"


@pytest.mark.skipif(not WORKBOOK.exists(), reason="the Source Data workbook is not on this machine")
@pytest.mark.parametrize("cation", ["K", "Li"])
def test_the_data_are_the_source_data_sheets_of_fig_4a_and_4b(cation):
    # The first of the cross-checks that failed for the archive's data_K.dat and data_Li.dat.
    header, body = read_sheet(WORKBOOK, SHEETS[cation])
    assert header == [
        "Probability Density",
        "Ion",
        "Probability Density",
        "Oxygen",
        "Probability Density",
        "Hydrogen",
    ]
    committed = np.loadtxt(DATA / vermiculite_z_density.DATA_FILES[cation])
    assert committed.shape == body.shape == (100, 6)
    assert (committed == body).all()


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        (lambda columns: columns[:, :5], "6 finite columns"),
        (
            lambda columns: np.where(np.arange(600).reshape(100, 6) == 123, np.nan, columns),
            "6 finite",
        ),
        (lambda columns: columns + [[0, 0, 0, 0, 0, 30.0]], "hydrogen leaves the axes"),
        (lambda columns: columns * [[10, 1, 1, 1, 1, 1]], "ion leaves the axes"),
    ],
)
def test_a_data_file_that_would_draw_outside_its_panel_is_refused_naming_the_file(
    tmp_path, corruption, message
):
    for data_file in vermiculite_z_density.DATA_FILES.values():
        np.savetxt(tmp_path / data_file, np.loadtxt(DATA / data_file))
    path = tmp_path / vermiculite_z_density.DATA_FILES["K"]
    np.savetxt(path, corruption(np.loadtxt(path)))
    with pytest.raises(ValueError, match=re.escape(path.name) + ".*" + message):
        vermiculite_z_density.build(tmp_path)


@pytest.mark.parametrize(("narrow", "committed"), LAYOUTS)
def test_chart_em_is_the_viewbox_width_in_labels_and_the_narrow_one_fits_the_phone(
    narrow, committed
):
    svg = vermiculite_z_density.build(DATA, narrow=narrow)
    width, height = map(float, re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg).groups())
    assert float(re.search(r'data-chart-em="([\d.]+)"', svg).group(1)) == pytest.approx(
        width / LABEL_SIZE, abs=0.05
    )
    if narrow:
        assert width == NARROW_CHART_EM * LABEL_SIZE
    else:
        assert width / LABEL_SIZE * 11 <= 700, "wider than a 700 px plate"
    texts = re.findall(
        r'<text x="([\d.]+)" y="([\d.]+)" text-anchor="(\w+)" font-size="([\d.]+)"([^>]*)>([^<]*)<',
        svg,
    )
    assert texts
    for x, y, anchor, size, attributes, content in texts:
        assert float(size) >= LABEL_SIZE, content
        extent = 0.6 * len(content) * float(size)  # the mono face advances 0.6 em
        reach = {"start": 0.0, "middle": 0.5, "end": 1.0}[anchor] * extent
        if "rotate" in attributes:  # runs up the page from its anchor
            assert float(y) - reach >= 0 and float(y) - reach + extent <= height, content
        else:
            assert float(x) - reach >= 0 and float(x) - reach + extent <= width, content
