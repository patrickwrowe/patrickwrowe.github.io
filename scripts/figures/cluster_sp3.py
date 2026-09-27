# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""sp3 fraction against temperature for the 48 GAP sphere runs, one line per cluster size.

Reads `sp3_fraction` (atoms with exactly four neighbours within 1.824 A) from `census.csv`,
written by cluster_census.py. The three largest clusters, the only ones that ever pass
2.5%, are drawn in `var(--ink)` and labelled one by one at their 500 K end, where they are
furthest apart; the five smaller are drawn in `var(--graphite)` and share one label, since
their lines cross one another near zero. Labels are pushed apart vertically where they
would overlap, keeping their order. No hue, unitless viewBox, house style of
graphitisation_figure.py.

Usage:
    uv run scripts/figures/cluster_sp3.py \
        --data-dir scripts/figures/data/carbon-clusters/spheres \
        --output src/content/work/figures/carbon/cluster-sp3.svg
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cluster_outcomes import (  # noqa: E402
    AXIS_TITLE_SIZE,
    AXIS_WIDTH,
    LABEL_SIZE,
    SIZES,
    TEMPERATURES_KELVIN,
    TICK_LENGTH,
    read_census_columns,
    svg_text,
)

PLOT_WIDTH = 100.0
PLOT_HEIGHT = 56.0
MARGIN_LEFT = 17.0
MARGIN_RIGHT = 6.0
MARGIN_TOP = 6.0
MARGIN_BOTTOM = 17.0
# Room between the y axis and the 500 K points for the right-aligned line labels.
LABEL_COLUMN = 20.0
LINE_WIDTH = 0.7
MARKER_RADIUS = 0.9
Y_MAX_PERCENT = 12.0
Y_TICKS_PERCENT = (0, 4, 8, 12)
# Sizes drawn in ink: the only ones whose sp3 fraction ever exceeds 2.5%.
INK_SIZES = (373, 686, 1000)


def build(data_dir: Path) -> str:
    """The sp3-against-temperature plot as an SVG document.

    Args:
        data_dir: Directory holding census.csv.

    Returns:
        The SVG document as a string.

    Raises:
        ValueError: If census.csv lacks a size-temperature pair, or a value exceeds the
            plotted range (12%).
    """
    table = read_census_columns(data_dir, ("n_atoms", "temperature_kelvin", "sp3_fraction"))
    percent = {
        (int(row["n_atoms"]), int(row["temperature_kelvin"])): 100.0 * float(row["sp3_fraction"])
        for row in table
    }
    missing = [
        (n_atoms, kelvin)
        for n_atoms in SIZES
        for kelvin in TEMPERATURES_KELVIN
        if (n_atoms, kelvin) not in percent
    ]
    if missing:
        raise ValueError(f"{data_dir / 'census.csv'}: no row for {missing}")
    if max(percent.values()) > Y_MAX_PERCENT:
        raise ValueError(f"an sp3 fraction exceeds the plotted {Y_MAX_PERCENT}%")

    left, top = MARGIN_LEFT, MARGIN_TOP
    baseline = top + PLOT_HEIGHT
    low, high = TEMPERATURES_KELVIN[0], TEMPERATURES_KELVIN[-1]

    def x_of(kelvin: float) -> float:
        span = PLOT_WIDTH - LABEL_COLUMN - 3.0
        return left + LABEL_COLUMN + (kelvin - low) / (high - low) * span

    def y_of(value_percent: float) -> float:
        return baseline - value_percent / Y_MAX_PERCENT * PLOT_HEIGHT

    parts = [
        f'<path d="M {left:.2f} {top:.2f} V {baseline:.2f} H {left + PLOT_WIDTH:.2f}" '
        f'fill="none" stroke="var(--graphite)" stroke-width="{AXIS_WIDTH}"/>'
    ]
    for value_percent in Y_TICKS_PERCENT:
        y = y_of(value_percent)
        parts.append(
            f'<line x1="{left - TICK_LENGTH:.2f}" y1="{y:.2f}" x2="{left:.2f}" y2="{y:.2f}" '
            f'stroke="var(--graphite)" stroke-width="{AXIS_WIDTH}"/>'
        )
        parts.append(
            svg_text(left - TICK_LENGTH - 1.2, y + 1.0, str(value_percent), LABEL_SIZE, "end")
        )
    for kelvin in TEMPERATURES_KELVIN:
        x = x_of(kelvin)
        parts.append(
            f'<line x1="{x:.2f}" y1="{baseline:.2f}" x2="{x:.2f}" '
            f'y2="{baseline + TICK_LENGTH:.2f}" stroke="var(--graphite)" '
            f'stroke-width="{AXIS_WIDTH}"/>'
        )
        parts.append(
            svg_text(x, baseline + TICK_LENGTH + LABEL_SIZE + 0.4, str(kelvin), LABEL_SIZE)
        )
    parts.append(
        svg_text(
            left + LABEL_COLUMN + (PLOT_WIDTH - LABEL_COLUMN) / 2,
            baseline + TICK_LENGTH + 2 * LABEL_SIZE + 3.2,
            "temperature / K",
            AXIS_TITLE_SIZE,
        )
    )
    parts.append(
        svg_text(
            left - 10.5, top + PLOT_HEIGHT / 2, "sp³ / % of atoms", AXIS_TITLE_SIZE, rotate=-90
        )
    )

    # Small clusters first, so the ink lines of the large ones sit on top.
    for n_atoms in SIZES:
        tone = "var(--ink)" if n_atoms in INK_SIZES else "var(--graphite)"
        points = [(x_of(kelvin), y_of(percent[n_atoms, kelvin])) for kelvin in TEMPERATURES_KELVIN]
        path = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        parts.append(
            f'<polyline data-size="{n_atoms}" points="{path}" fill="none" stroke="{tone}" '
            f'stroke-width="{LINE_WIDTH}" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        parts.extend(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{MARKER_RADIUS}" fill="{tone}" stroke="none"/>'
            for x, y in points
        )

    # Ink lines are labelled one by one at their 500 K end; the grey lines of the small
    # clusters, which never pass 2.5% and cross one another near zero, share one label.
    small = [n_atoms for n_atoms in SIZES if n_atoms not in INK_SIZES]
    labels = sorted(
        [(y_of(percent[n_atoms, low]) + 1.0, f"C{n_atoms}") for n_atoms in INK_SIZES]
        + [
            (
                y_of(max(percent[n_atoms, low] for n_atoms in small)) + 1.0,
                f"C{small[0]}–C{small[-1]}",
            )
        ]
    )
    minimum_gap = LABEL_SIZE * 1.25
    placed: list[tuple[float, str]] = []
    for y, text in reversed(labels):  # bottom (largest y) first, then upwards
        if placed and y > placed[-1][0] - minimum_gap:
            y = placed[-1][0] - minimum_gap
        placed.append((y, text))
    label_x = x_of(low) - MARKER_RADIUS - 1.6
    parts.extend(svg_text(label_x, y, text, LABEL_SIZE, "end") for y, text in placed)

    width = left + PLOT_WIDTH + MARGIN_RIGHT
    height = baseline + MARGIN_BOTTOM
    body = "\n    ".join(parts)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.2f} {height:.2f}" '
        f'role="img">\n  <g>\n    {body}\n  </g>\n</svg>\n'
    )


def main(argv: list[str] | None = None) -> None:
    """Write the sp3 plot and print the values it draws.

    Args:
        argv: Command-line arguments; None reads `sys.argv`.

    Raises:
        ValueError: From `build`, if census.csv is incomplete or out of range.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    svg = build(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg)
    table = read_census_columns(args.data_dir, ("n_atoms", "temperature_kelvin", "sp3_fraction"))
    peak = max(table, key=lambda row: row["sp3_fraction"])
    print(
        f"  highest sp3: C{int(peak['n_atoms'])}-{int(peak['temperature_kelvin'])}K "
        f"{100 * peak['sp3_fraction']:.1f}%; any NaN: "
        f"{any(math.isnan(float(row['sp3_fraction'])) for row in table)}"
    )
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
