# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Outcome diagram of the 48 GAP sphere runs: cluster size against temperature, one glyph per class.

Reads the class of every run from `census.csv` (written by cluster_census.py) and draws
one glyph per run at (atoms, temperature), size on a log axis. The classes and the rule
that assigns them are in docs/dossiers/carbon/clusters/classification.md.

House style of graphitisation_figure.py: emitted as SVG by hand so that every stroke and
fill resolves to `var(--ink)`, `var(--graphite)` or `var(--plate)`, with no hue; classes
are told apart by shape alone. The viewBox is unitless and font sizes are in its units.
Each data glyph is a `<g>` carrying `data-run` and `data-class`, so the figure can be
checked against the census without reading pixels.

Usage:
    uv run scripts/figures/cluster_outcomes.py \
        --data-dir scripts/figures/data/carbon-clusters/spheres \
        --output src/content/work/figures/carbon/cluster-outcomes.svg
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

SIZES = (40, 60, 80, 120, 160, 373, 686, 1000)
TEMPERATURES_KELVIN = (500, 1000, 2000, 3000, 4000, 5000)
# Legend order: from most ordered to least. Diamond-like is part of the scheme but no run
# is classed so; the legend says "(none)" rather than hiding it.
CLASSES = (
    "diamond-like",
    "graphitic onion",
    "cage",
    "disordered",
    "molten",
    "dissociated",
)

PLOT_WIDTH = 100.0
PLOT_HEIGHT = 62.0
MARGIN_LEFT = 17.0
MARGIN_TOP = 6.0
MARGIN_BOTTOM = 17.0
LEGEND_GAP = 10.0
LEGEND_WIDTH = 40.0
# Half a glyph of padding keeps the end columns and rows off the axes.
AXIS_PADDING = 4.0

AXIS_WIDTH = 0.45
GLYPH_STROKE = 0.55
GLYPH_RADIUS = 2.0
DOT_RADIUS = 0.8
TICK_LENGTH = 1.6
LABEL_SIZE = 3.0
AXIS_TITLE_SIZE = 3.2


def read_census_columns(data_dir: Path, columns: tuple[str, ...]) -> np.ndarray:
    """Named columns of census.csv as a structured array, NaN kept as NaN.

    Uses `numpy.genfromtxt` with named `usecols`, the pattern
    `scripts/tests/test_cluster_census.py` pins: whole-file type detection fails on the
    space-separated list columns.

    Args:
        data_dir: Directory holding census.csv.
        columns: Column names to read.

    Returns:
        Structured array with one field per requested column, one row per run.

    Raises:
        FileNotFoundError: If census.csv is missing.
        ValueError: If a requested column is not in the file.
    """
    return np.genfromtxt(
        data_dir / "census.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
        usecols=columns,
    )


def svg_text(
    x: float,
    y: float,
    content: str,
    size: float,
    anchor: str = "middle",
    rotate: float | None = None,
) -> str:
    """A text element in the figure's mono face, filled with `var(--graphite)`.

    Args:
        x: Anchor x, in viewBox units.
        y: Baseline y, in viewBox units.
        content: The text.
        size: Font size, in viewBox units.
        anchor: SVG text-anchor.
        rotate: Rotation in degrees about the anchor, or None.

    Returns:
        The SVG element as a string.
    """
    transform = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate is not None else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" font-size="{size}" '
        f'fill="var(--graphite)" stroke="none" font-family="var(--mono)"{transform}>'
        f"{content}</text>"
    )


def glyph(class_name: str, x: float, y: float) -> str:
    """The shape that stands for one class, centred on (x, y).

    Open shapes are filled with `var(--plate)` so that one drawn over an axis or its
    neighbour knocks it out, as in graphitisation_figure.py.

    Args:
        class_name: One of CLASSES.
        x: Centre x, in viewBox units.
        y: Centre y, in viewBox units.

    Returns:
        SVG markup for the glyph.

    Raises:
        ValueError: If `class_name` is not one of CLASSES.
    """
    outline = f'stroke="var(--ink)" stroke-width="{GLYPH_STROKE}"'
    radius = GLYPH_RADIUS
    if class_name == "diamond-like":
        return (
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="var(--ink)" stroke="none"/>'
        )
    if class_name == "graphitic onion":
        return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="var(--plate)" {outline}/>'
    if class_name == "cage":
        corners = " ".join(
            f"{x + radius * math.cos(math.radians(30 + 60 * corner)):.2f},"
            f"{y + radius * math.sin(math.radians(30 + 60 * corner)):.2f}"
            for corner in range(6)
        )
        return f'<polygon points="{corners}" fill="var(--plate)" {outline}/>'
    if class_name == "disordered":
        side = radius * 1.6
        return (
            f'<rect x="{x - side / 2:.2f}" y="{y - side / 2:.2f}" width="{side:.2f}" '
            f'height="{side:.2f}" fill="var(--plate)" {outline}/>'
        )
    if class_name == "molten":
        arm = radius * 0.8
        return (
            f'<path d="M {x - arm:.2f} {y - arm:.2f} L {x + arm:.2f} {y + arm:.2f} '
            f'M {x - arm:.2f} {y + arm:.2f} L {x + arm:.2f} {y - arm:.2f}" fill="none" '
            f'stroke="var(--ink)" stroke-width="{GLYPH_STROKE * 1.6:.2f}" stroke-linecap="round"/>'
        )
    if class_name == "dissociated":
        return (
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{DOT_RADIUS:.2f}" '
            f'fill="var(--ink)" stroke="none"/>'
        )
    raise ValueError(f"unknown class {class_name!r}; expected one of {CLASSES}")


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
    table = read_census_columns(data_dir, ("n_atoms", "temperature_kelvin", "class"))
    outcome = {
        (int(row["n_atoms"]), int(row["temperature_kelvin"])): str(row["class"]) for row in table
    }
    expected = {(n_atoms, kelvin) for n_atoms in SIZES for kelvin in TEMPERATURES_KELVIN}
    if len(table) != len(expected) or set(outcome) != expected:
        raise ValueError(f"{data_dir / 'census.csv'}: expected one row per size and temperature")

    log_min, log_max = math.log10(SIZES[0]), math.log10(SIZES[-1])
    left, top = MARGIN_LEFT, MARGIN_TOP
    baseline = top + PLOT_HEIGHT

    def x_of(n_atoms: float) -> float:
        span = PLOT_WIDTH - 2 * AXIS_PADDING
        return left + AXIS_PADDING + (math.log10(n_atoms) - log_min) / (log_max - log_min) * span

    def y_of(kelvin: float) -> float:
        span = PLOT_HEIGHT - 2 * AXIS_PADDING
        low, high = TEMPERATURES_KELVIN[0], TEMPERATURES_KELVIN[-1]
        return baseline - AXIS_PADDING - (kelvin - low) / (high - low) * span

    parts = [
        f'<path d="M {left:.2f} {top:.2f} V {baseline:.2f} H {left + PLOT_WIDTH:.2f}" '
        f'fill="none" stroke="var(--graphite)" stroke-width="{AXIS_WIDTH}"/>'
    ]
    for kelvin in TEMPERATURES_KELVIN:
        y = y_of(kelvin)
        parts.append(
            f'<line x1="{left - TICK_LENGTH:.2f}" y1="{y:.2f}" x2="{left:.2f}" y2="{y:.2f}" '
            f'stroke="var(--graphite)" stroke-width="{AXIS_WIDTH}"/>'
        )
        parts.append(svg_text(left - TICK_LENGTH - 1.2, y + 1.0, str(kelvin), LABEL_SIZE, "end"))
    for n_atoms in SIZES:
        x = x_of(n_atoms)
        parts.append(
            f'<line x1="{x:.2f}" y1="{baseline:.2f}" x2="{x:.2f}" '
            f'y2="{baseline + TICK_LENGTH:.2f}" stroke="var(--graphite)" '
            f'stroke-width="{AXIS_WIDTH}"/>'
        )
        parts.append(
            svg_text(x, baseline + TICK_LENGTH + LABEL_SIZE + 0.4, str(n_atoms), LABEL_SIZE)
        )
    parts.append(
        svg_text(
            left + PLOT_WIDTH / 2,
            baseline + TICK_LENGTH + 2 * LABEL_SIZE + 3.2,
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
            f"{glyph(class_name, x_of(n_atoms), y_of(kelvin))}</g>"
        )

    counts = {class_name: 0 for class_name in CLASSES}
    for class_name in outcome.values():
        counts[class_name] += 1
    legend_x = left + PLOT_WIDTH + LEGEND_GAP
    row_height = LABEL_SIZE * 2.4
    legend_top = top + (PLOT_HEIGHT - row_height * (len(CLASSES) - 1)) / 2
    for index, class_name in enumerate(CLASSES):
        y = legend_top + index * row_height
        label = f"{class_name} ({counts[class_name] or 'none'})"
        parts.append(f'<g data-legend="{class_name}">{glyph(class_name, legend_x, y)}</g>')
        parts.append(svg_text(legend_x + GLYPH_RADIUS + 2.4, y + 1.0, label, LABEL_SIZE, "start"))

    width = left + PLOT_WIDTH + LEGEND_GAP + LEGEND_WIDTH
    height = baseline + MARGIN_BOTTOM
    body = "\n    ".join(parts)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.2f} {height:.2f}" '
        f'role="img">\n  <g>\n    {body}\n  </g>\n</svg>\n'
    )


def main(argv: list[str] | None = None) -> None:
    """Write the outcome diagram.

    Args:
        argv: Command-line arguments; None reads `sys.argv`.

    Raises:
        ValueError: From `build`, if census.csv is incomplete or holds an unknown class.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    svg = build(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg)
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
