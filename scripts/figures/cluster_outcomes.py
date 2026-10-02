# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Outcome diagram of the 48 GAP sphere runs: cluster size against temperature, one glyph per class.

Reads the class of every run from `census.csv` (written by cluster_census.py) and draws
one glyph per run at (atoms, temperature), size on a log axis. The classes and the rule
that assigns them are in docs/dossiers/carbon/clusters/classification.md.

House style of the site's figures (figure_style.py): emitted as SVG by hand so that every
stroke and fill resolves to `var(--ink)`, `var(--graphite)` or `var(--plate)`, with no hue;
classes are told apart by shape alone. The viewBox is unitless and font sizes are in its
units. Each data glyph is a `<g>` carrying `data-run` and `data-class`, so the figure can
be checked against the census without reading pixels.

`--narrow` draws the same 48 glyphs and the same legend in a layout that fits a 31.5 em
viewBox (the site's phone-width budget below 700 px, Plate.astro's `.chart--narrow`):
temperature across, cluster size down the log axis, legend below the plot instead of
beside it.

Usage:
    uv run scripts/figures/cluster_outcomes.py \
        --data-dir scripts/figures/data/carbon-clusters/spheres \
        --output src/content/work/figures/carbon/cluster-outcomes.svg

    uv run scripts/figures/cluster_outcomes.py \
        --data-dir scripts/figures/data/carbon-clusters/spheres \
        --output src/content/work/figures/carbon/cluster-outcomes-narrow.svg --narrow
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

from cluster_census import read_census_columns
from constants import SIZES, TEMPERATURES_KELVIN, OutcomeClass
from figure_style import (
    AXIS_TITLE_SIZE,
    AXIS_WIDTH,
    LABEL_SIZE,
    NARROW_CHART_EM,
    TICK_LENGTH,
    svg_text,
)

# The legend lists constants.OutcomeClass in its order, most ordered to least. Diamond-like
# is part of the scheme but no run is classed so; the legend says "(none)" rather than
# hiding it.

PLOT_WIDTH = 100.0
PLOT_HEIGHT = 62.0
MARGIN_LEFT = 17.0
MARGIN_TOP = 6.0
MARGIN_BOTTOM = 17.0
LEGEND_GAP = 10.0
LEGEND_WIDTH = 40.0
GLYPH_STROKE = 0.55
GLYPH_RADIUS = 2.0
DOT_RADIUS = 0.8
# Two glyph radii of padding keep the end columns and rows off the axes.
AXIS_PADDING = 2 * GLYPH_RADIUS
BASELINE = MARGIN_TOP + PLOT_HEIGHT

# --narrow: temperature across (6, linear), cluster size down (8, log), legend below the
# plot. The viewBox width is fixed by the site's phone-width budget, 31.5 em of
# LABEL_SIZE (Plate.astro's `.chart--narrow`); everything else here is chosen to fit it
# without crowding the tick labels (checked by hand against the rendered coordinates,
# scripts/tests/test_cluster_figures.py pins the result).
NARROW_WIDTH = NARROW_CHART_EM * LABEL_SIZE
NARROW_PLOT_WIDTH = 76.0
NARROW_PLOT_HEIGHT = 100.0
NARROW_MARGIN_LEFT = MARGIN_LEFT
NARROW_MARGIN_TOP = MARGIN_TOP
NARROW_BASELINE = NARROW_MARGIN_TOP + NARROW_PLOT_HEIGHT
NARROW_LEGEND_GAP = LEGEND_GAP
# Gap between the last legend row's baseline and the bottom of the viewBox, matching the
# wide chart's gap between its axis title baseline and BASELINE + MARGIN_BOTTOM.
NARROW_BOTTOM_PAD = MARGIN_BOTTOM - TICK_LENGTH - 2 * LABEL_SIZE - 3.2


def size_to_x(n_atoms: float) -> float:
    """Horizontal position of a cluster size on the log axis.

    Args:
        n_atoms: Cluster size, in atoms.

    Returns:
        x, in viewBox units.
    """
    log_smallest, log_largest = math.log10(SIZES[0]), math.log10(SIZES[-1])
    span = PLOT_WIDTH - 2 * AXIS_PADDING
    fraction = (math.log10(n_atoms) - log_smallest) / (log_largest - log_smallest)
    return MARGIN_LEFT + AXIS_PADDING + fraction * span


def temperature_to_y(kelvin: float) -> float:
    """Vertical position of a temperature on the linear axis, hotter higher.

    Args:
        kelvin: Temperature, in K.

    Returns:
        y, in viewBox units (SVG y grows downwards).
    """
    coldest, hottest = TEMPERATURES_KELVIN[0], TEMPERATURES_KELVIN[-1]
    span = PLOT_HEIGHT - 2 * AXIS_PADDING
    return BASELINE - AXIS_PADDING - (kelvin - coldest) / (hottest - coldest) * span


def temperature_to_x_narrow(kelvin: float) -> float:
    """Horizontal position of a temperature on the linear axis, --narrow layout.

    Args:
        kelvin: Temperature, in K.

    Returns:
        x, in viewBox units.
    """
    coldest, hottest = TEMPERATURES_KELVIN[0], TEMPERATURES_KELVIN[-1]
    span = NARROW_PLOT_WIDTH - 2 * AXIS_PADDING
    fraction = (kelvin - coldest) / (hottest - coldest)
    return NARROW_MARGIN_LEFT + AXIS_PADDING + fraction * span


def size_to_y_narrow(n_atoms: float) -> float:
    """Vertical position of a cluster size on the log axis, --narrow layout, larger lower.

    Args:
        n_atoms: Cluster size, in atoms.

    Returns:
        y, in viewBox units.
    """
    log_smallest, log_largest = math.log10(SIZES[0]), math.log10(SIZES[-1])
    span = NARROW_PLOT_HEIGHT - 2 * AXIS_PADDING
    fraction = (math.log10(n_atoms) - log_smallest) / (log_largest - log_smallest)
    return NARROW_MARGIN_TOP + AXIS_PADDING + fraction * span


def glyph(class_name: str, x_centre: float, y_centre: float) -> str:
    """The shape that stands for one class, centred on (x_centre, y_centre).

    Open shapes are filled with `var(--plate)` so that one drawn over an axis or its
    neighbour knocks it out, as in graphitisation_figure.py.

    Args:
        class_name: An OutcomeClass, or its name.
        x_centre: Centre x, in viewBox units.
        y_centre: Centre y, in viewBox units.

    Returns:
        SVG markup for the glyph.

    Raises:
        ValueError: If `class_name` is not an OutcomeClass.
    """
    outline = f'stroke="var(--ink)" stroke-width="{GLYPH_STROKE}"'
    radius = GLYPH_RADIUS
    centre = f'cx="{x_centre:.2f}" cy="{y_centre:.2f}"'
    if class_name == OutcomeClass.DIAMOND_LIKE:
        return f'<circle {centre} r="{radius:.2f}" fill="var(--ink)" stroke="none"/>'
    if class_name == OutcomeClass.GRAPHITIC_ONION:
        return f'<circle {centre} r="{radius:.2f}" fill="var(--plate)" {outline}/>'
    if class_name == OutcomeClass.CAGE:
        corners = " ".join(
            f"{x_centre + radius * math.cos(math.radians(30 + 60 * corner)):.2f},"
            f"{y_centre + radius * math.sin(math.radians(30 + 60 * corner)):.2f}"
            for corner in range(6)
        )
        return f'<polygon points="{corners}" fill="var(--plate)" {outline}/>'
    if class_name == OutcomeClass.DISORDERED:
        side = radius * 1.6
        return (
            f'<rect x="{x_centre - side / 2:.2f}" y="{y_centre - side / 2:.2f}" '
            f'width="{side:.2f}" height="{side:.2f}" fill="var(--plate)" {outline}/>'
        )
    if class_name == OutcomeClass.MOLTEN:
        arm = radius * 0.8
        return (
            f'<path d="M {x_centre - arm:.2f} {y_centre - arm:.2f} '
            f"L {x_centre + arm:.2f} {y_centre + arm:.2f} "
            f"M {x_centre - arm:.2f} {y_centre + arm:.2f} "
            f'L {x_centre + arm:.2f} {y_centre - arm:.2f}" fill="none" '
            f'stroke="var(--ink)" stroke-width="{GLYPH_STROKE * 1.6:.2f}" stroke-linecap="round"/>'
        )
    if class_name == OutcomeClass.DISSOCIATED:
        return f'<circle {centre} r="{DOT_RADIUS:.2f}" fill="var(--ink)" stroke="none"/>'
    raise ValueError(f"unknown class {class_name!r}; expected one of {list(OutcomeClass)}")


def _read_outcomes(data_dir: Path) -> dict[tuple[int, int], str]:
    """Every run's class, keyed by (atoms, kelvin), checked against the full 8x6 grid.

    Args:
        data_dir: Directory holding census.csv.

    Returns:
        One entry per (n_atoms, temperature_kelvin) in SIZES x TEMPERATURES_KELVIN.

    Raises:
        ValueError: If census.csv does not hold exactly one row per size and temperature.
    """
    table = read_census_columns(data_dir, ("n_atoms", "temperature_kelvin", "class"))
    outcome = {
        (int(row["n_atoms"]), int(row["temperature_kelvin"])): str(row["class"]) for row in table
    }
    expected = {(n_atoms, kelvin) for n_atoms in SIZES for kelvin in TEMPERATURES_KELVIN}
    if len(table) != len(expected) or set(outcome) != expected:
        raise ValueError(f"{data_dir / 'census.csv'}: expected one row per size and temperature")
    return outcome


def build(data_dir: Path) -> str:
    """The outcome diagram as an SVG document.

    Args:
        data_dir: Directory holding census.csv.

    Returns:
        The SVG document as a string.

    Raises:
        ValueError: If census.csv does not hold exactly one run per size and temperature,
            or a class is unknown.
    """
    outcome = _read_outcomes(data_dir)
    left, top = MARGIN_LEFT, MARGIN_TOP
    parts = [
        f'<path d="M {left:.2f} {top:.2f} V {BASELINE:.2f} H {left + PLOT_WIDTH:.2f}" '
        f'fill="none" stroke="var(--graphite)" stroke-width="{AXIS_WIDTH}"/>'
    ]
    for kelvin in TEMPERATURES_KELVIN:
        tick_y = temperature_to_y(kelvin)
        parts.append(
            f'<line x1="{left - TICK_LENGTH:.2f}" y1="{tick_y:.2f}" x2="{left:.2f}" '
            f'y2="{tick_y:.2f}" stroke="var(--graphite)" stroke-width="{AXIS_WIDTH}"/>'
        )
        parts.append(
            svg_text(left - TICK_LENGTH - 1.2, tick_y + 1.0, str(kelvin), LABEL_SIZE, "end")
        )
    for n_atoms in SIZES:
        tick_x = size_to_x(n_atoms)
        parts.append(
            f'<line x1="{tick_x:.2f}" y1="{BASELINE:.2f}" x2="{tick_x:.2f}" '
            f'y2="{BASELINE + TICK_LENGTH:.2f}" stroke="var(--graphite)" '
            f'stroke-width="{AXIS_WIDTH}"/>'
        )
        parts.append(
            svg_text(tick_x, BASELINE + TICK_LENGTH + LABEL_SIZE + 0.4, str(n_atoms), LABEL_SIZE)
        )
    parts.append(
        svg_text(
            left + PLOT_WIDTH / 2,
            BASELINE + TICK_LENGTH + 2 * LABEL_SIZE + 3.2,
            "cluster size / atoms (log scale)",
            AXIS_TITLE_SIZE,
        )
    )
    parts.append(
        svg_text(left - 12.5, top + PLOT_HEIGHT / 2, "temperature / K", AXIS_TITLE_SIZE, rotate=-90)
    )
    for (n_atoms, kelvin), class_name in sorted(outcome.items()):
        parts.append(
            f'<g data-run="C{n_atoms}-{kelvin}K" data-class="{class_name}">'
            f"{glyph(class_name, size_to_x(n_atoms), temperature_to_y(kelvin))}</g>"
        )

    counts = Counter(outcome.values())  # a class with no runs reads 0
    legend_x = left + PLOT_WIDTH + LEGEND_GAP
    row_height = LABEL_SIZE * 2.4
    legend_top = top + (PLOT_HEIGHT - row_height * (len(OutcomeClass) - 1)) / 2
    for index, class_name in enumerate(OutcomeClass):
        row_y = legend_top + index * row_height
        label = f"{class_name} ({counts[class_name] or 'none'})"
        parts.append(f'<g data-legend="{class_name}">{glyph(class_name, legend_x, row_y)}</g>')
        parts.append(
            svg_text(legend_x + GLYPH_RADIUS + 2.4, row_y + 1.0, label, LABEL_SIZE, "start")
        )

    width = left + PLOT_WIDTH + LEGEND_GAP + LEGEND_WIDTH
    height = BASELINE + MARGIN_BOTTOM
    body = "\n    ".join(parts)
    # data-chart-em: the viewBox width in labels, the page wrapper's --chart-em.
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.2f} {height:.2f}" '
        f'role="img" data-chart-em="{width / LABEL_SIZE:.1f}">\n'
        f"  <g>\n    {body}\n  </g>\n</svg>\n"
    )


def build_narrow(data_dir: Path) -> str:
    """The phone-width outcome diagram, for `.chart--narrow` below 700 px.

    Same 48 glyphs, same classes, same legend content and the same house style as
    `build`, laid out to fit a viewBox 31.5 x LABEL_SIZE units wide: temperature across
    the linear axis, cluster size down the log axis (larger lower), and the legend below
    the plot instead of beside it.

    Args:
        data_dir: Directory holding census.csv.

    Returns:
        The SVG document as a string.

    Raises:
        ValueError: If census.csv does not hold exactly one run per size and temperature,
            or a class is unknown.
    """
    outcome = _read_outcomes(data_dir)
    left, top, baseline = NARROW_MARGIN_LEFT, NARROW_MARGIN_TOP, NARROW_BASELINE
    parts = [
        f'<path d="M {left:.2f} {top:.2f} V {baseline:.2f} H {left + NARROW_PLOT_WIDTH:.2f}" '
        f'fill="none" stroke="var(--graphite)" stroke-width="{AXIS_WIDTH}"/>'
    ]
    for n_atoms in SIZES:
        tick_y = size_to_y_narrow(n_atoms)
        parts.append(
            f'<line x1="{left - TICK_LENGTH:.2f}" y1="{tick_y:.2f}" x2="{left:.2f}" '
            f'y2="{tick_y:.2f}" stroke="var(--graphite)" stroke-width="{AXIS_WIDTH}"/>'
        )
        parts.append(
            svg_text(left - TICK_LENGTH - 1.2, tick_y + 1.0, str(n_atoms), LABEL_SIZE, "end")
        )
    for kelvin in TEMPERATURES_KELVIN:
        tick_x = temperature_to_x_narrow(kelvin)
        parts.append(
            f'<line x1="{tick_x:.2f}" y1="{baseline:.2f}" x2="{tick_x:.2f}" '
            f'y2="{baseline + TICK_LENGTH:.2f}" stroke="var(--graphite)" '
            f'stroke-width="{AXIS_WIDTH}"/>'
        )
        parts.append(
            svg_text(tick_x, baseline + TICK_LENGTH + LABEL_SIZE + 0.4, str(kelvin), LABEL_SIZE)
        )
    title_y = baseline + TICK_LENGTH + 2 * LABEL_SIZE + 3.2
    parts.append(
        svg_text(left + NARROW_PLOT_WIDTH / 2, title_y, "temperature / K", AXIS_TITLE_SIZE)
    )
    parts.append(
        svg_text(
            left - 12.5,
            top + NARROW_PLOT_HEIGHT / 2,
            "cluster size / atoms (log scale)",
            AXIS_TITLE_SIZE,
            rotate=-90,
        )
    )
    for (n_atoms, kelvin), class_name in sorted(outcome.items()):
        parts.append(
            f'<g data-run="C{n_atoms}-{kelvin}K" data-class="{class_name}">'
            f"{glyph(class_name, temperature_to_x_narrow(kelvin), size_to_y_narrow(n_atoms))}</g>"
        )

    counts = Counter(outcome.values())  # a class with no runs reads 0
    row_height = LABEL_SIZE * 2.4
    legend_top = title_y + NARROW_LEGEND_GAP
    for index, class_name in enumerate(OutcomeClass):
        row_y = legend_top + index * row_height
        label = f"{class_name} ({counts[class_name] or 'none'})"
        parts.append(f'<g data-legend="{class_name}">{glyph(class_name, left, row_y)}</g>')
        parts.append(svg_text(left + GLYPH_RADIUS + 2.4, row_y + 1.0, label, LABEL_SIZE, "start"))

    height = legend_top + (len(OutcomeClass) - 1) * row_height + NARROW_BOTTOM_PAD
    body = "\n    ".join(parts)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {NARROW_WIDTH:.2f} {height:.2f}" '
        f'role="img" data-chart-em="{NARROW_WIDTH / LABEL_SIZE:.1f}">\n'
        f"  <g>\n    {body}\n  </g>\n</svg>\n"
    )


def main(argv: list[str] | None = None) -> None:
    """Write the outcome diagram.

    Args:
        argv: Command-line arguments; None reads `sys.argv`.

    Raises:
        ValueError: From `build` or `build_narrow`, if census.csv is incomplete or holds
            an unknown class.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--narrow", action="store_true", help="write the phone-width variant instead"
    )
    args = parser.parse_args(argv)
    svg = build_narrow(args.data_dir) if args.narrow else build(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg)
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
