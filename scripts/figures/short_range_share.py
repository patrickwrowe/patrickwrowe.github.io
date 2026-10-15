# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Short-range share of the excitonic coupling against the distance of closest approach.

Scatter of J_short / J_TOT (per cent) against the minimum intermolecular distance D (angstrom)
for the fifteen rows of the JCTC paper's own Table 1, three series: FMO (filled circle), LH2
(open circle) and PCP (filled square), the filled-versus-open device of the site's other
charts with one shape more. The largest share of each series is written beside its marker.
Every row of Table 1 lies inside D = 7 angstrom, the paper's limit between pairs in close
contact (D < 7) and distant ones (p.9): the table lists the couplings with the largest
short-range contributions, not a sample of all pairs, so the chart shows how large the
short-range term gets, not how it falls with distance. The distance axis stops at 3.5 angstrom,
just past the farthest row (2.98); drawn out to a guide at 7 it left about 63 per cent of the
plot empty and crowded the fifteen markers into the rest, and the caption states the limit.

Data: scripts/figures/data/excited-states/table1-short-range-share.json, typed from Fornari,
Rowe, Padula and Troisi, J. Chem. Theory Comput. 13, 3754 (2017), Table 1 (manuscript p.11);
provenance.json beside it records the source. The share is the one the table prints, from the
paper's unrounded couplings. J_TOT and J_short are in the file but not drawn.

House style of the site's figures (figure_style.py): SVG by hand over a unitless viewBox,
every stroke and fill in `var(--ink)`, `var(--graphite)` or `var(--plate)`, no hue. Each
series is a `<g data-series>` and each row a `<g data-pair>` carrying its row's numbers, so the
figure can be checked against the data without reading pixels.

`--narrow` draws the same markers in a viewBox 31.5 em of LABEL_SIZE wide (the site's
phone-width budget below 700 px, Plate.astro's `.chart--narrow`). No label is smaller than
LABEL_SIZE.

Usage:
    FIGURES=src/content/work/figures/excited-states
    uv run scripts/figures/short_range_share.py \
        --data-dir scripts/figures/data/excited-states \
        --output $FIGURES/short-range-share.svg

    uv run scripts/figures/short_range_share.py \
        --data-dir scripts/figures/data/excited-states \
        --output $FIGURES/short-range-share-narrow.svg --narrow
"""

from __future__ import annotations

import json
import math
from functools import partial
from pathlib import Path

from figure_style import (
    AXIS_BAND,
    AXIS_TITLE_SIZE,
    LABEL_SIZE,
    LEGEND_ROW_HEIGHT,
    NARROW_CHART_EM,
    X_AXIS_TITLE_DROP,
    Y_TITLE_OFFSET,
    svg_axis,
    svg_document,
    svg_marker,
    svg_square_marker,
    svg_text,
    write_figure,
)

DATA_FILE = "table1-short-range-share.json"
ROW_FIELDS = (
    "system",
    "pair",
    "min_distance_angstrom",
    "coupled_states",
    "j_total_mev",
    "j_short_mev",
    "j_short_share_percent",
)

# The three series, in legend order, and the marker each is drawn with. Rows of Table 1 lie
# nearer than a marker is wide (LH2 pair B's two rows are 1.0 viewBox unit apart against a
# marker 3.4 across), so each marker is drawn `separate`: the open circle unfilled, the filled
# ones outlined in the plate's colour, and every row stays findable.
MARKERS = {
    "FMO": ("filled", partial(svg_marker, filled=True, separate=True)),
    "LH2": ("open", partial(svg_marker, filled=False, separate=True)),
    "PCP": ("square", partial(svg_square_marker, separate=True)),
}

# The axes, (minimum, maximum) and labelled values. D runs to 3.5 angstrom, past the farthest
# row, 2.98; the share runs to 70 per cent, above the largest, 66.
DISTANCE_AXIS_ANGSTROM = (0.0, 3.5)
DISTANCE_TICKS_ANGSTROM = tuple(range(4))
SHARE_AXIS_PERCENT = (0.0, 70.0)
SHARE_TICKS_PERCENT = (0, 20, 40, 60)

# Plot geometry, in viewBox user units. One unit of height is one per cent of share.
MARGIN_LEFT = 13.0
MARGIN_TOP = 6.0
MARGIN_RIGHT = 6.0
PLOT_WIDTH = 128.0
PLOT_HEIGHT = 70.0
# From a marker's centre up to the baseline of the value written above it.
VALUE_LABEL_RISE = 3.0
# The legend sits in the empty top right of the plot: a marker, a gap, a name, one row per
# series, the marker this far in from the plot's right edge.
LEGEND_OFFSET_FROM_RIGHT = 14.0
LEGEND_NAME_GAP = 3.5

NARROW_WIDTH = NARROW_CHART_EM * LABEL_SIZE
NARROW_MARGIN_RIGHT = 4.0
NARROW_PLOT_WIDTH = NARROW_WIDTH - MARGIN_LEFT - NARROW_MARGIN_RIGHT


def build(data_dir: Path, narrow: bool = False) -> str:
    """The short-range-share chart as an SVG document.

    Args:
        data_dir: Directory holding `table1-short-range-share.json`.
        narrow: Draw the 31.5 em phone-width layout instead of the wide one.

    Returns:
        The SVG document as a string.

    Raises:
        ValueError: If a row lacks a field or names a system the chart has no marker for, if a
            system has no row, or if a distance or share is not a finite number (a JSON `true`
            included) or leaves its
            axis, which would draw a marker off the plot.
    """
    path = data_dir / DATA_FILE
    table = json.loads(path.read_text())
    rows = table["rows"]
    distance_low, distance_high = DISTANCE_AXIS_ANGSTROM
    share_low, share_high = SHARE_AXIS_PERCENT
    for row in rows:
        name = f"{row.get('system')} {row.get('pair')}"
        if missing := set(ROW_FIELDS) - set(row):
            raise ValueError(f"{path}: {name} lacks {sorted(missing)}")
        if row["system"] not in MARKERS:
            raise ValueError(f"{path}: {name} is of a system with no marker: {row['system']!r}")
        for key, low, high in (
            ("min_distance_angstrom", distance_low, distance_high),
            ("j_short_share_percent", share_low, share_high),
        ):
            value = row[key]
            is_number = isinstance(value, int | float) and not isinstance(value, bool)
            if not (is_number and math.isfinite(value)):
                raise ValueError(f"{path}: {name} {key} is {value!r}")
            if not low < value <= high:
                raise ValueError(f"{path}: {name} {key} leaves its axis")

    plot_width = NARROW_PLOT_WIDTH if narrow else PLOT_WIDTH
    plot_left, plot_top = MARGIN_LEFT, MARGIN_TOP
    plot_bottom = plot_top + PLOT_HEIGHT
    distance_span, share_span = distance_high - distance_low, share_high - share_low

    def x_of(distance_angstrom: float) -> float:
        return plot_left + (distance_angstrom - distance_low) / distance_span * plot_width

    def y_of(share_percent: float) -> float:
        return plot_bottom - (share_percent - share_low) / share_span * PLOT_HEIGHT

    parts = svg_axis(
        "bottom",
        plot_left,
        plot_bottom,
        plot_width,
        [(x_of(tick) - plot_left, f"{tick:g}") for tick in DISTANCE_TICKS_ANGSTROM],
    )
    parts += svg_axis(
        "left",
        plot_left,
        plot_bottom,
        PLOT_HEIGHT,
        [(plot_bottom - y_of(tick), f"{tick:g}") for tick in SHARE_TICKS_PERCENT],
    )
    parts.append(
        svg_text(
            plot_left + plot_width / 2,
            plot_bottom + X_AXIS_TITLE_DROP,
            "minimum distance D (Å)",
            AXIS_TITLE_SIZE,
        )
    )
    parts.append(
        svg_text(
            plot_left - Y_TITLE_OFFSET,
            plot_top + PLOT_HEIGHT / 2,
            "short-range share (%)",
            AXIS_TITLE_SIZE,
            rotate=-90,
        )
    )

    for system, (kind, marker) in MARKERS.items():
        system_rows = [row for row in rows if row["system"] == system]
        if not system_rows:
            raise ValueError(f"{path}: no row of the system {system}")
        # Every row that ties for the largest share is labelled, not only the first.
        largest_percent = max(row["j_short_share_percent"] for row in system_rows)
        points = []
        for row in system_rows:
            point_x = x_of(row["min_distance_angstrom"])
            point_y = y_of(row["j_short_share_percent"])
            drawing = marker(point_x, point_y)
            if row["j_short_share_percent"] == largest_percent:
                drawing += svg_text(
                    point_x,
                    point_y - VALUE_LABEL_RISE,
                    f"{row['j_short_share_percent']:g}%",
                    LABEL_SIZE,
                )
            points.append(
                f'<g data-pair="{row["pair"]}" data-states="{row["coupled_states"]}" '
                f'data-distance-angstrom="{row["min_distance_angstrom"]:g}" '
                f'data-share-percent="{row["j_short_share_percent"]:g}">{drawing}</g>'
            )
        point_body = "\n      ".join(points)
        parts.append(
            f'<g data-series="{system}" data-marker="{kind}">\n      {point_body}\n    </g>'
        )

    for row_index, (system, (_kind, marker)) in enumerate(MARKERS.items()):
        row_y = plot_top + 4.0 + row_index * LEGEND_ROW_HEIGHT
        marker_x = plot_left + plot_width - LEGEND_OFFSET_FROM_RIGHT
        parts.append(
            f'<g data-legend="{system}">'
            + marker(marker_x, row_y)
            + svg_text(marker_x + LEGEND_NAME_GAP, row_y + 1.0, system, LABEL_SIZE, "start")
            + "</g>"
        )

    width = NARROW_WIDTH if narrow else MARGIN_LEFT + plot_width + MARGIN_RIGHT
    return svg_document(width, MARGIN_TOP + PLOT_HEIGHT + AXIS_BAND, parts)


if __name__ == "__main__":
    write_figure(build, __doc__.splitlines()[0])
