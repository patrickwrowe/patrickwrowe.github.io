"""Tests for scripts/figures/short_range_share.py.

The committed table1-short-range-share.json is the root of trust: the chart is checked against
it, and the numbers a caption could copy (the 28 per cent at 2.26 angstrom, the 7 angstrom
threshold, the largest share of each system, the ranges of distance and share, the rows of
each system) are recomputed from the drawing, from the data and, for the 2.26, from the staged
chromophore geometries. The markers are also checked for hiding one another: rows of Table 1
lie nearer than a marker is wide, and the drawing must keep each one findable. Nothing here
reads docs/, which is gitignored.
"""

from __future__ import annotations

import itertools
import json
import re
from functools import partial
from pathlib import Path

import boxprep
import numpy as np
import pytest
import short_range_share
from chart_svg import TEXT, assert_text_fits, numeric_labels
from figure_style import LABEL_SIZE, MARKER_RADIUS, NARROW_CHART_EM, svg_marker, svg_square_marker

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
                left, top, side = (
                    float(value)
                    for value in re.search(
                        r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)"', body
                    ).groups()
                )
                centre_x, centre_y = left + side / 2, top + side / 2
            else:
                centre_x, centre_y = (
                    float(value)
                    for value in re.search(r'<circle cx="([\d.]+)" cy="([\d.]+)"', body).groups()
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
    # The open circle of LH2 is unfilled, so a marker beneath it shows through its ring.
    assert 'fill="none" stroke="var(--ink)"' in svg


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


def test_the_ranges_and_row_counts_a_caption_quotes_are_recomputed_from_the_drawing_and_the_data():
    for narrow in (False, True):
        points = read_points(short_range_share.build(DATA, narrow=narrow))
        distances = [point["distance_angstrom"] for point in points]
        shares = [point["share_percent"] for point in points]
        # "all 15 pairs at 0.98 to 2.98 angstrom, shares from 7.1 to 66 per cent"
        assert (min(distances), max(distances)) == pytest.approx((0.98, 2.98), abs=0.01)
        assert (min(shares), max(shares)) == pytest.approx((7.1, 66.0), abs=0.05)
        by_system = {
            system: [point for point in points if point["system"] == system]
            for system in ("FMO", "LH2", "PCP")
        }
        assert {system: len(rows) for system, rows in by_system.items()} == {
            "FMO": 5,
            "LH2": 5,
            "PCP": 5,
        }
        span = {
            system: (
                min(point["share_percent"] for point in rows),
                max(point["share_percent"] for point in rows),
            )
            for system, rows in by_system.items()
        }
        assert span["FMO"] == pytest.approx((19.0, 28.0), abs=0.05)
        assert span["LH2"] == pytest.approx((10.0, 66.0), abs=0.05)
        assert span["PCP"] == pytest.approx((7.1, 22.0), abs=0.05)
        # LH2 without its one 66 per cent: "between 10 and 27".
        assert max(
            point["share_percent"] for point in by_system["LH2"] if point["share_percent"] < 60
        ) == pytest.approx(27.0, abs=0.05)
        # FMO's other three pairs: 2.18 to 2.61 angstrom, 19 to 25 per cent.
        fmo_others = [point for point in by_system["FMO"] if point["pair"] != "1-2"]
        assert len(fmo_others) == 4  # pairs 5-6 (two rows), 4-7 and 3-7
        assert (
            min(point["distance_angstrom"] for point in fmo_others),
            max(point["distance_angstrom"] for point in fmo_others),
        ) == pytest.approx((2.18, 2.61), abs=0.01)
        assert (
            min(point["share_percent"] for point in fmo_others),
            max(point["share_percent"] for point in fmo_others),
        ) == pytest.approx((19.0, 25.0), abs=0.05)
    # The 66 per cent is the S1-S2 cross coupling of LH2 pair A, 4.98 meV in total (the data).
    cross = next(row for row in ROWS if row["j_short_share_percent"] == 66)
    assert (cross["system"], cross["pair"], cross["coupled_states"]) == ("LH2", "A", "1,2")
    assert cross["j_total_mev"] == 4.98
    assert 100 * cross["j_short_mev"] / cross["j_total_mev"] == pytest.approx(66.0, abs=0.5)


def write_table(tmp_path: Path, rows: list[dict]) -> None:
    """The committed table with these rows, in a directory the chart can be built from."""
    table = json.loads((DATA / "table1-short-range-share.json").read_text())
    table["rows"] = rows
    (tmp_path / "table1-short-range-share.json").write_text(json.dumps(table))


def test_a_tie_for_the_largest_share_of_a_system_labels_every_row_that_ties(tmp_path):
    rows = [dict(row) for row in ROWS]
    tied = next(row for row in rows if (row["system"], row["pair"]) == ("FMO", "5-6"))
    tied["j_short_share_percent"] = 28  # equal to pair 1-2, the largest of FMO
    write_table(tmp_path, rows)
    points = read_points(short_range_share.build(tmp_path))
    assert sorted(
        (point["pair"], point["states"])
        for point in points
        if point["system"] == "FMO" and point["label"] == "28%"
    ) == [("1-2", "1,2"), ("5-6", "1,2")]


def test_a_system_with_no_row_is_refused_naming_the_file_and_the_system(tmp_path):
    write_table(tmp_path, [row for row in ROWS if row["system"] != "LH2"])
    with pytest.raises(
        ValueError, match=r"table1-short-range-share\.json: no row of the system LH2"
    ):
        short_range_share.build(tmp_path)


# Markers are `2 * MARKER_RADIUS` across. Rows of Table 1 that lie nearer than that overlap in
# the drawing: these are the known, reviewed pairs, named "system pair states". A new row, or
# a layout change, that adds one fails `test_the_markers_that_overlap_are_the_known_pairs`.
OVERLAPPING_PAIRS = {
    False: {
        frozenset({"FMO 5-6 1,2", "FMO 4-7 1,2"}),
        frozenset({"FMO 5-6 2,1", "FMO 4-7 1,2"}),
        frozenset({"LH2 B 1,1", "LH2 B 2,2"}),
        frozenset({"PCP 1-3 1,1", "PCP 7-8 1,1"}),
    },
    True: {
        frozenset({"FMO 5-6 1,2", "FMO 4-7 1,2"}),
        frozenset({"FMO 5-6 2,1", "FMO 4-7 1,2"}),
        frozenset({"LH2 B 1,1", "LH2 B 2,2"}),
        frozenset({"PCP 1-3 1,1", "PCP 7-8 1,1"}),
        frozenset({"PCP 1-3 1,1", "PCP 2-4 1,1"}),
    },
}
# The least share of a marker's ink that later markers may leave showing. A circle is read by
# its outline, so most of it must show: the plain open circle of LH2 pair B's 13 per cent row
# showed 0.50 beneath its neighbour's plate-filled one, the drawn one shows 0.75. A square of
# PCP overlaps its neighbour more (pairs 1-3 and 7-8 are 0.12 angstrom apart, 1.2 units in the
# phone layout), and its plate outline costs it ink: it keeps 0.56 wide and 0.28 narrow, a bar
# a third of a square wide beside the square that overlaps it, still a second marker.
MIN_VISIBLE_INK = {"circle": 0.7, "rect": 0.25}
SAMPLE_STEP = 0.05  # viewBox units


def read_shapes(svg: str) -> list[dict]:
    """Every marker's geometry and paint, in drawing order, as the SVG states them.

    Args:
        svg: The figure.

    Returns:
        One dict per marker: name ("system pair states"), kind ("circle" or "rect"), centre
        (x, y), extent (the circle's radius or the square's half side), fill, stroke and
        outline (the stroke's width, 0 without one), all in viewBox units.
    """
    shapes = []
    for system, series in re.findall(
        r'<g data-series="(\w+)"[^>]*>(.*?)\n    </g>', svg, re.DOTALL
    ):
        for pair, states, body in re.findall(
            r'<g data-pair="([^"]+)" data-states="([^"]+)" [^>]*>(.*?)</g>', series, re.DOTALL
        ):
            element = re.search(r"<(circle|rect) ([^>]*)/>", body)
            attributes = dict(re.findall(r'([\w-]+)="([^"]*)"', element.group(2)))
            if element.group(1) == "circle":
                centre = (float(attributes["cx"]), float(attributes["cy"]))
                extent = float(attributes["r"])
            else:
                extent = float(attributes["width"]) / 2
                centre = (float(attributes["x"]) + extent, float(attributes["y"]) + extent)
            shapes.append(
                {
                    "name": f"{system} {pair} {states}",
                    "kind": element.group(1),
                    "centre": centre,
                    "extent": extent,
                    "fill": attributes["fill"],
                    "stroke": attributes["stroke"],
                    "outline": 0.0
                    if attributes["stroke"] == "none"
                    else float(attributes["stroke-width"]),
                }
            )
    return shapes


def paint(shape: dict, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Where a marker leaves ink and where it paints anything, at these points.

    A stroke is centred on the shape's edge and paints over the fill; a fill of "none" paints
    nothing, so a marker beneath an unfilled ring shows through it.

    Args:
        shape: One entry of `read_shapes`.
        points: Shape (n, 2), in viewBox units.

    Returns:
        Two boolean arrays of length n: the points inked (in `var(--ink)`), and the points
        painted in any colour, `var(--plate)` included.
    """
    offset = points - shape["centre"]
    if shape["kind"] == "circle":
        distance = np.hypot(offset[:, 0], offset[:, 1])
    else:
        distance = np.abs(offset).max(axis=1)
    half_outline = shape["outline"] / 2
    in_stroke = (np.abs(distance - shape["extent"]) <= half_outline) & (shape["outline"] > 0)
    in_fill = distance <= shape["extent"]
    fill_shows = in_fill & ~in_stroke & (distance < shape["extent"] - half_outline)
    inked = (in_stroke & (shape["stroke"] == "var(--ink)")) | (
        fill_shows & (shape["fill"] == "var(--ink)")
    )
    painted = in_stroke | (in_fill & (shape["fill"] != "none"))
    return inked, painted


def visible_ink(svg: str) -> dict[str, float]:
    """The share of each marker's ink that no later marker paints over, by marker name.

    Sampled on a grid of `SAMPLE_STEP`. Markers are drawn in series order, so one is covered
    only by those after it.

    Args:
        svg: The figure.

    Returns:
        Marker name to the visible fraction of its ink, between 0 and 1.
    """
    shapes = read_shapes(svg)
    reach = max(shape["extent"] + shape["outline"] for shape in shapes)
    offsets = np.arange(-reach, reach + SAMPLE_STEP, SAMPLE_STEP)
    grid = np.array(list(itertools.product(offsets, offsets)))
    visible = {}
    for index, shape in enumerate(shapes):
        points = grid + shape["centre"]
        inked, _ = paint(shape, points)
        covered = np.zeros(len(points), dtype=bool)
        for later in shapes[index + 1 :]:
            covered |= paint(later, points)[1]
        visible[shape["name"]] = float((inked & ~covered).sum() / inked.sum())
    return visible


@pytest.mark.parametrize("narrow", [False, True])
def test_the_markers_that_overlap_are_the_known_pairs(narrow):
    shapes = read_shapes(short_range_share.build(DATA, narrow=narrow))
    assert len(shapes) == 15
    near = {
        frozenset({first["name"], second["name"]})
        for first, second in itertools.combinations(shapes, 2)
        if np.hypot(*np.subtract(first["centre"], second["centre"])) < 2 * MARKER_RADIUS
    }
    assert near == OVERLAPPING_PAIRS[narrow]


@pytest.mark.parametrize("narrow", [False, True])
def test_every_row_stays_findable_with_most_of_its_outline_showing(narrow):
    svg = short_range_share.build(DATA, narrow=narrow)
    fractions = visible_ink(svg)
    kinds = {shape["name"]: shape["kind"] for shape in read_shapes(svg)}
    assert len(fractions) == 15
    hidden = {
        name: round(fraction, 2)
        for name, fraction in fractions.items()
        if fraction < MIN_VISIBLE_INK[kinds[name]]
    }
    assert not hidden, f"markers hidden by a later one: {hidden}"


def test_the_visibility_check_sees_the_plain_markers_hide_lh2_pair_b(monkeypatch):
    # The markers as first drawn (the open circle filled with the plate, nothing outlined):
    # the 14 per cent circle covered all but a crescent of the 13 per cent one.
    plain = {
        "FMO": ("filled", partial(svg_marker, filled=True)),
        "LH2": ("open", partial(svg_marker, filled=False)),
        "PCP": ("square", svg_square_marker),
    }
    monkeypatch.setattr(short_range_share, "MARKERS", plain)
    fractions = visible_ink(short_range_share.build(DATA))
    assert fractions["LH2 B 1,1"] == pytest.approx(0.50, abs=0.03)
    assert fractions["LH2 B 1,1"] < MIN_VISIBLE_INK["circle"]


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
