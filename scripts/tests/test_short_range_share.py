"""Tests for scripts/figures/short_range_share.py.

The committed table1-short-range-share.json is the root of trust: the chart is checked against
it, and the numbers a caption could copy (the 28 per cent at 2.26 angstrom, the 7 angstrom
threshold, the largest share of each system) are recomputed from the drawing, from the data
and, for the 2.26, from the staged chromophore geometries. Nothing here reads docs/, which is
gitignored.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import boxprep
import numpy as np
import pytest
import short_range_share
from chart_svg import TEXT, assert_text_fits, numeric_labels
from figure_style import LABEL_SIZE, MARKER_RADIUS, NARROW_CHART_EM

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "scripts/figures/data/excited-states"
FIGURES = REPO / "src/content/work/figures/excited-states"
TABLE = json.loads((DATA / "table1-short-range-share.json").read_text())
ROWS = TABLE["rows"]
HEX_COLOUR = re.compile(r"#[0-9a-fA-F]{3}\b|#[0-9a-fA-F]{6}\b")
LAYOUTS = [(False, "short-range-share.svg"), (True, "short-range-share-narrow.svg")]


def read_points(svg: str) -> list[dict]:
    """Each marker as drawn: its system, pair and states, and its place read off the axes.

    The axes are calibrated from the printed tick labels and where they sit, so the data come
    back as a reader would read them.

    Args:
        svg: The figure.

    Returns:
        One dict per marker in drawing order: system, kind, pair, states, distance_angstrom,
        share_percent, label (the value written above it, or None).
    """
    x_positions, x_values = numeric_labels(svg, "middle")
    y_positions, y_values = numeric_labels(svg, "end")
    x_slope, x_offset = np.polyfit(x_values, x_positions, 1)
    y_slope, y_offset = np.polyfit(y_values, y_positions, 1)
    points = []
    for system, kind, series in re.findall(
        r'<g data-series="(\w+)" data-marker="(\w+)">(.*?)\n    </g>', svg, re.DOTALL
    ):
        for pair, states, body in re.findall(
            r'<g data-pair="([^"]+)" data-states="([^"]+)" [^>]*>(.*?)</g>', series, re.DOTALL
        ):
            if kind == "square":
                x, y, side = (
                    float(v)
                    for v in re.search(
                        r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)"', body
                    ).groups()
                )
                centre_x, centre_y = x + side / 2, y + side / 2
            else:
                centre_x, centre_y = (
                    float(v)
                    for v in re.search(r'<circle cx="([\d.]+)" cy="([\d.]+)"', body).groups()
                )
            label = TEXT.search(body)
            points.append(
                {
                    "system": system,
                    "kind": kind,
                    "pair": pair,
                    "states": states,
                    "distance_angstrom": (centre_x - x_offset) / x_slope,
                    "share_percent": (centre_y - y_offset) / y_slope,
                    "label": label.group(6) if label else None,
                }
            )
    return points


def guide_distance_angstrom(svg: str) -> float:
    """Where the dotted close-contact guide stands on the distance axis, read off the drawing."""
    positions, values = numeric_labels(svg, "middle")
    slope, offset = np.polyfit(values, positions, 1)
    guide_x = float(
        re.search(r'<g data-guide="close-contact"[^>]*><line x1="([\d.]+)"', svg).group(1)
    )
    return (guide_x - offset) / slope


@pytest.mark.parametrize(("narrow", "committed"), LAYOUTS)
def test_the_committed_svg_is_byte_for_byte_what_the_script_writes(narrow, committed):
    written = short_range_share.build(DATA, narrow=narrow).encode("utf-8")
    assert (FIGURES / committed).read_bytes() == written, f"{committed} is stale"


@pytest.mark.parametrize("narrow", [False, True])
def test_every_marker_reads_back_as_its_row_of_table_1(narrow):
    points = read_points(short_range_share.build(DATA, narrow=narrow))
    assert len(points) == len(ROWS) == 15
    by_system = {
        system: [row for row in ROWS if row["system"] == system] for system in ("FMO", "LH2", "PCP")
    }
    assert [point["system"] for point in points] == [
        row["system"] for system in by_system for row in by_system[system]
    ]
    for point, row in zip(
        points, [row for rows in by_system.values() for row in rows], strict=True
    ):
        assert (point["pair"], point["states"]) == (row["pair"], row["coupled_states"])
        assert point["distance_angstrom"] == pytest.approx(row["min_distance_angstrom"], abs=0.01)
        assert point["share_percent"] == pytest.approx(row["j_short_share_percent"], abs=0.05)


def test_each_system_has_its_own_marker_and_the_legend_shows_all_three():
    svg = short_range_share.build(DATA)
    kinds = {point["system"]: point["kind"] for point in read_points(svg)}
    assert kinds == {"FMO": "filled", "LH2": "open", "PCP": "square"}
    assert re.findall(r'<g data-legend="(\w+)">', svg) == ["FMO", "LH2", "PCP"]
    assert not HEX_COLOUR.search(svg)
    assert set(re.findall(r'(?:fill|stroke)="(var\(--[a-z]+\))"', svg)) == {
        "var(--ink)",
        "var(--graphite)",
        "var(--plate)",
    }
    assert f'r="{MARKER_RADIUS}" fill="var(--ink)"' in svg  # the filled circle of FMO
    assert 'fill="var(--plate)" stroke="var(--ink)"' in svg  # the open circle of LH2


def test_the_guide_stands_at_the_papers_7_angstrom_and_every_marker_is_inside_it():
    for narrow in (False, True):
        svg = short_range_share.build(DATA, narrow=narrow)
        assert guide_distance_angstrom(svg) == pytest.approx(7.0, abs=0.01)
        assert TABLE["close_contact_threshold_angstrom"] == 7.0  # the data's, from manuscript p.9
        assert "D = 7 Å" in svg
        assert max(point["distance_angstrom"] for point in read_points(svg)) < 7.0


def test_the_28_per_cent_at_2_26_angstrom_is_the_fmo_pair_1_2_and_the_staged_geometry_agrees():
    svg = short_range_share.build(DATA)
    point = next(
        point for point in read_points(svg) if (point["system"], point["pair"]) == ("FMO", "1-2")
    )
    assert point["share_percent"] == pytest.approx(28.0, abs=0.1)
    assert point["label"] == "28%"
    # The distance on the chart, recomputed from the geometries the pair plate draws.
    first = boxprep.read_xyz(DATA / "chromophore-1.xyz").positions_angstrom
    second = boxprep.read_xyz(DATA / "chromophore-2.xyz").positions_angstrom
    minimum_angstrom = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=-1).min()
    assert minimum_angstrom == pytest.approx(2.2613, abs=5e-5)
    assert point["distance_angstrom"] == pytest.approx(round(minimum_angstrom, 2), abs=0.01)
    # And from the couplings: 4.29 of 15.2 meV is 28.2 per cent.
    row = next(row for row in ROWS if (row["system"], row["pair"]) == ("FMO", "1-2"))
    assert 100 * row["j_short_mev"] / row["j_total_mev"] == pytest.approx(28.0, abs=0.5)


def test_the_largest_share_of_each_system_is_written_above_its_marker_and_nowhere_else():
    points = read_points(short_range_share.build(DATA))
    labelled = {point["system"]: point["label"] for point in points if point["label"]}
    assert labelled == {"FMO": "28%", "LH2": "66%", "PCP": "22%"}  # the paper's conclusions
    for system, label in labelled.items():
        largest = max(point["share_percent"] for point in points if point["system"] == system)
        assert float(label.rstrip("%")) == pytest.approx(largest, abs=0.1)


def test_every_axis_title_carries_its_unit():
    svg = short_range_share.build(DATA)
    assert "minimum distance D (Å)" in svg
    assert "short-range share of J_TOT (%)" in svg


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        (lambda row: row.pop("pair"), r"lacks \['pair'\]"),
        (lambda row: row.update(system="XYZ"), "no marker: 'XYZ'"),
        (lambda row: row.update(j_short_share_percent=float("nan")), "is nan"),
        (lambda row: row.update(j_short_share_percent="28"), "is '28'"),
        (lambda row: row.update(j_short_share_percent=80.0), "leaves its axis"),
        (lambda row: row.update(j_short_share_percent=0), "leaves its axis"),
        (lambda row: row.update(min_distance_angstrom=9.0), "leaves its axis"),
    ],
)
def test_a_row_that_would_draw_off_the_plot_is_refused_naming_the_file_and_the_row(
    tmp_path, corruption, message
):
    table = json.loads((DATA / "table1-short-range-share.json").read_text())
    corruption(table["rows"][3])
    (tmp_path / "table1-short-range-share.json").write_text(json.dumps(table))
    with pytest.raises(ValueError, match=r"table1-short-range-share\.json: \w+ \S+ .*" + message):
        short_range_share.build(tmp_path)


def test_a_threshold_off_the_distance_axis_is_refused(tmp_path):
    table = json.loads((DATA / "table1-short-range-share.json").read_text())
    table["close_contact_threshold_angstrom"] = 12.0
    (tmp_path / "table1-short-range-share.json").write_text(json.dumps(table))
    with pytest.raises(ValueError, match="threshold 12.0 leaves its axis"):
        short_range_share.build(tmp_path)


@pytest.mark.parametrize(("narrow", "committed"), LAYOUTS)
def test_chart_em_is_the_viewbox_width_in_labels_and_the_narrow_one_fits_the_phone(
    narrow, committed
):
    svg = short_range_share.build(DATA, narrow=narrow)
    width, _height = map(float, re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg).groups())
    assert float(re.search(r'data-chart-em="([\d.]+)"', svg).group(1)) == pytest.approx(
        width / LABEL_SIZE, abs=0.05
    )
    if narrow:
        assert width == NARROW_CHART_EM * LABEL_SIZE
    else:
        assert width / LABEL_SIZE * 11 <= 700, "wider than a 700 px plate"
    assert_text_fits(svg)
