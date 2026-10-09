"""Shared drawing style for the site's hand-written SVG data figures.

The figure scripts (graphitisation_figure.py, cluster_outcomes.py,
cluster_benchmark_c60.py, vermiculite_z_density.py, vermiculite_contact_angles.py,
hbn_potential_errors.py) emit SVG by hand so that every stroke and fill resolves to a
design token (`var(--ink)`, `var(--graphite)`, `var(--plate)`) and no hue enters a figure
(spec 01 section 6.2), except that a chart of more than one species may draw a species in
its own `--species-*` token (`species_colour`). The viewBox is unitless, so every size here
is in drawing units, not pixels; the figure scales with its container.

Import it as a sibling module (`from figure_style import ...`); the scripts run from
`scripts/figures/`, which is then on the import path.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable
from pathlib import Path

from constants import SPECIES_COLOUR

__all__ = [
    "AXIS_TITLE_SIZE",
    "AXIS_WIDTH",
    "LABEL_SIZE",
    "MARKER_RADIUS",
    "MARKER_STROKE",
    "MINUS_SIGN",
    "NARROW_CHART_EM",
    "SERIES_WIDTH",
    "TICK_LENGTH",
    "X_AXIS_TITLE_DROP",
    "format_tick",
    "species_colour",
    "svg_axis",
    "svg_document",
    "svg_line",
    "svg_marker",
    "svg_series",
    "svg_text",
    "write_figure",
]

AXIS_WIDTH = 0.45
TICK_LENGTH = 1.6
LABEL_SIZE = 3.0
AXIS_TITLE_SIZE = 3.2
# From a bottom axis line to the baseline of its title: the ticks, one line of tick labels
# (svg_axis puts them a label size and 0.4 below the ticks) and a gap of 3.2.
X_AXIS_TITLE_DROP = TICK_LENGTH + 2 * LABEL_SIZE + 3.2
# A data line.
SERIES_WIDTH = 0.9
# A data marker: the circle's outer radius, and the stroke of an open one.
MARKER_RADIUS = 1.7
MARKER_STROKE = 0.7
# The mono face draws U+2212 as a real minus; Python's float formatting uses hyphen-minus,
# which reads shorter and sits high next to a digit.
MINUS_SIGN = "−"
# The mono face has no Greek (IBM Plex Mono ships none), so a Greek letter in a label would
# fall back to whatever monospace font the reader's system has. The serif loads the Greek
# subset (astro.config.mjs), so a run of Greek is set in it: the same glyph as the prose and
# the caption beside the chart.
GREEK_RUN = re.compile("[\u0370-\u03ff]+")
# The phone-width budget of a chart's narrow variant, in units of its label size: 31.5
# labels of 11 px (--t-label) is 346.5 px, inside the 350 px plate of a 390 px screen.
# Plate.astro shows a `.chart--narrow` below 700 px.
NARROW_CHART_EM = 31.5


def svg_text(
    x_position: float,
    y_position: float,
    content: str,
    size: float,
    anchor: str = "middle",
    rotate: float | None = None,
) -> str:
    """A text element in the site's mono face, filled with `var(--graphite)`.

    A run of Greek letters is wrapped in a `<tspan>` in `var(--serif)` (`GREEK_RUN`).

    Args:
        x_position: Anchor x, in viewBox units.
        y_position: Baseline y, in viewBox units.
        content: The text; not escaped, so it must not contain markup characters.
        size: Font size, in viewBox units.
        anchor: SVG text-anchor: "start", "middle" or "end".
        rotate: Rotation in degrees about the anchor point, or None for none.

    Returns:
        The SVG `<text>` element as a string.
    """
    transform = (
        f' transform="rotate({rotate} {x_position:.2f} {y_position:.2f})"'
        if rotate is not None
        else ""
    )
    return (
        f'<text x="{x_position:.2f}" y="{y_position:.2f}" text-anchor="{anchor}" '
        f'font-size="{size}" fill="var(--graphite)" stroke="none" font-family="var(--mono)"'
        f"{transform}>"
        + GREEK_RUN.sub(lambda run: f'<tspan font-family="var(--serif)">{run[0]}</tspan>', content)
        + "</text>"
    )


def format_tick(value: float, decimals: int) -> str:
    """A tick label with a true minus sign.

    Args:
        value: The tick's value, in the axis's unit.
        decimals: Digits after the decimal point.

    Returns:
        The value to `decimals` places, e.g. "−0.5" or "0.02".
    """
    return f"{value:.{decimals}f}".replace("-", MINUS_SIGN)


def species_colour(symbol: str) -> str:
    """The CSS colour of a species in a chart: its `--species-*` token.

    The tokens mirror `constants.SPECIES_COLOUR` (scripts/tests/test_palette.py fails if they
    drift), so a chart's series match the species of the Blender panels beside it.

    Args:
        symbol: Element symbol, e.g. "K" or "Li".

    Returns:
        `var(--species-k)` and the like.

    Raises:
        ValueError: If the species has no colour in `constants.SPECIES_COLOUR`.
    """
    if symbol not in SPECIES_COLOUR:
        raise ValueError(f"no species colour for {symbol!r}")
    return f"var(--species-{symbol.lower()})"


def svg_line(
    x_start: float,
    y_start: float,
    x_end: float,
    y_end: float,
    width: float = AXIS_WIDTH,
    dash: str | None = None,
    colour: str = "var(--graphite)",
) -> str:
    """A straight stroke, in `var(--graphite)` unless told otherwise.

    Args:
        x_start: Start x, in viewBox units.
        y_start: Start y, in viewBox units.
        x_end: End x, in viewBox units.
        y_end: End y, in viewBox units.
        width: Stroke width, in viewBox units.
        dash: SVG stroke-dasharray, or None for a solid line.
        colour: Any CSS colour or token.

    Returns:
        The SVG `<line>` element as a string.
    """
    dashing = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x_start:.2f}" y1="{y_start:.2f}" x2="{x_end:.2f}" y2="{y_end:.2f}" '
        f'stroke="{colour}" stroke-width="{width}"{dashing}/>'
    )


def svg_marker(x_position: float, y_position: float, filled: bool) -> str:
    """A data marker: filled in `var(--ink)`, or open with a `var(--plate)` centre.

    The open marker is filled with `var(--plate)`, not "none", so it knocks out the line
    behind it rather than reading as a bead threaded on a wire. Both are `MARKER_RADIUS`
    across; the open one's stroke (`MARKER_STROKE`) lies inside that radius.

    Args:
        x_position: Marker centre x, in viewBox units.
        y_position: Marker centre y, in viewBox units.
        filled: True for a filled marker, False for an open one.

    Returns:
        The SVG `<circle>` element as a string.
    """
    if filled:
        return (
            f'<circle cx="{x_position:.2f}" cy="{y_position:.2f}" r="{MARKER_RADIUS}" '
            f'fill="var(--ink)" stroke="none"/>'
        )
    return (
        f'<circle cx="{x_position:.2f}" cy="{y_position:.2f}" '
        f'r="{MARKER_RADIUS - MARKER_STROKE / 2:.2f}" fill="var(--plate)" '
        f'stroke="var(--ink)" stroke-width="{MARKER_STROKE}"/>'
    )


def svg_series(points: list[tuple[float, float]], colour: str, dash: str | None = None) -> str:
    """A data line through points, `SERIES_WIDTH` wide with round joins.

    Args:
        points: (x, y) pairs, in viewBox units.
        colour: Any CSS colour or token.
        dash: SVG stroke-dasharray, or None for a solid line.

    Returns:
        The SVG `<polyline>` element as a string.
    """
    path = " ".join(f"{x_position:.2f},{y_position:.2f}" for x_position, y_position in points)
    dashing = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<polyline points="{path}" fill="none" stroke="{colour}" stroke-width="{SERIES_WIDTH}" '
        f'stroke-linejoin="round" stroke-linecap="round"{dashing}/>'
    )


def svg_axis(
    side: str,
    origin_x: float,
    origin_y: float,
    length: float,
    ticks: list[tuple[float, str]],
) -> list[str]:
    """An axis line with its ticks and tick labels, without a title.

    A "bottom" axis runs right from the origin with ticks and centred labels below it; a
    "left" axis runs up from the origin with ticks and right-aligned labels to its left.
    Labels set at `LABEL_SIZE`, the page's label size.

    Args:
        side: "bottom" or "left".
        origin_x: x of the axis's start, in viewBox units.
        origin_y: y of the axis's start, in viewBox units (the bottom end for a left axis).
        length: Length of the axis, in viewBox units.
        ticks: (distance from the origin along the axis in viewBox units, label) pairs.

    Returns:
        The SVG elements: the line, then a tick and a label per tick.

    Raises:
        ValueError: If `side` is neither "bottom" nor "left".
    """
    if side not in ("bottom", "left"):
        raise ValueError(f"side must be 'bottom' or 'left', not {side!r}")
    if side == "bottom":
        parts = [svg_line(origin_x, origin_y, origin_x + length, origin_y)]
        for distance, label in ticks:
            tick_x = origin_x + distance
            parts.append(svg_line(tick_x, origin_y, tick_x, origin_y + TICK_LENGTH))
            label_y = origin_y + TICK_LENGTH + LABEL_SIZE + 0.4
            parts.append(svg_text(tick_x, label_y, label, LABEL_SIZE))
        return parts
    parts = [svg_line(origin_x, origin_y, origin_x, origin_y - length)]
    for distance, label in ticks:
        tick_y = origin_y - distance
        parts.append(svg_line(origin_x - TICK_LENGTH, tick_y, origin_x, tick_y))
        parts.append(svg_text(origin_x - TICK_LENGTH - 1.0, tick_y + 1.0, label, LABEL_SIZE, "end"))
    return parts


def svg_document(width: float, height: float, body: list[str]) -> str:
    """The SVG root of a chart, carrying the `data-chart-em` the page wrapper copies.

    Args:
        width: viewBox width, in viewBox units.
        height: viewBox height, in viewBox units.
        body: The chart's elements.

    Returns:
        The SVG document as a string, ending in a newline. `data-chart-em` is the width in
        labels (`LABEL_SIZE`), the wrapper's `--chart-em`.
    """
    elements = "\n  ".join(body)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.2f} {height:.2f}" '
        f'role="img" data-chart-em="{width / LABEL_SIZE:.1f}">\n'
        f"  <g>\n  {elements}\n  </g>\n</svg>\n"
    )


def write_figure(
    build: Callable[..., str],
    description: str,
    argv: list[str] | None = None,
    data_dir: bool = True,
) -> None:
    """Parse a figure script's command line, build the figure and write it.

    The command line is `--output PATH [--data-dir DIR] [--narrow]`; `--data-dir` is there
    only for a figure that reads files.

    Args:
        build: The script's `build`, called as `build(data_dir, narrow)` or, without a data
            directory, `build(narrow)`; it returns the SVG document.
        description: One line for `--help`, usually the module docstring's first line.
        argv: Command-line arguments; None reads `sys.argv`.
        data_dir: Whether the figure reads a data directory.
    """
    parser = argparse.ArgumentParser(description=description)
    if data_dir:
        parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--narrow", action="store_true", help="write the phone-width variant instead"
    )
    args = parser.parse_args(argv)
    svg = build(args.data_dir, args.narrow) if data_dir else build(args.narrow)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg)
    print(f"-> {args.output}")
