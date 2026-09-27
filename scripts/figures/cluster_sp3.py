# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""sp3 fraction against temperature for the 48 GAP sphere runs, one line per cluster size.

Reads `sp3_fraction` (atoms with exactly four neighbours within 1.824 A) from `census.csv`,
written by cluster_census.py. Sizes whose sp3 fraction ever exceeds 2.5% are drawn in
`var(--ink)` and labelled one by one at their 500 K end, where the lines are furthest
apart; the rest (in these data, the five smallest) are drawn in `var(--graphite)` under
one shared label naming their range, since their lines cross one another near zero. Both
groups are derived from the data. Labels are pushed apart vertically where they would
overlap, keeping their order. No hue, unitless viewBox, house style of figure_style.py.

Usage:
    uv run scripts/figures/cluster_sp3.py \
        --data-dir scripts/figures/data/carbon-clusters/spheres \
        --output src/content/work/figures/carbon/cluster-sp3.svg
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from cluster_census import SIZES, TEMPERATURES_KELVIN, read_census_columns
from figure_style import AXIS_TITLE_SIZE, AXIS_WIDTH, LABEL_SIZE, TICK_LENGTH, svg_text

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
# A size is drawn in ink and labelled on its own if its sp3 fraction ever exceeds this.
INK_ABOVE_PERCENT = 2.5
BASELINE = MARGIN_TOP + PLOT_HEIGHT


def temperature_to_x(kelvin: float) -> float:
    """Horizontal position of a temperature on the linear axis.

    Args:
        kelvin: Temperature, in K.

    Returns:
        x, in viewBox units.
    """
    coldest, hottest = TEMPERATURES_KELVIN[0], TEMPERATURES_KELVIN[-1]
    span = PLOT_WIDTH - LABEL_COLUMN - 3.0
    return MARGIN_LEFT + LABEL_COLUMN + (kelvin - coldest) / (hottest - coldest) * span


def percent_to_y(value_percent: float) -> float:
    """Vertical position of an sp3 percentage, 0 on the axis and Y_MAX_PERCENT at the top.

    Args:
        value_percent: sp3 fraction, in per cent of atoms.

    Returns:
        y, in viewBox units (SVG y grows downwards).
    """
    return BASELINE - value_percent / Y_MAX_PERCENT * PLOT_HEIGHT


def read_percentages(data_dir: Path) -> dict[tuple[int, int], float]:
    """sp3 percentage per (size, temperature), checked complete, finite and in range.

    Args:
        data_dir: Directory holding census.csv.

    Returns:
        {(atoms, kelvin): sp3 fraction in per cent}.

    Raises:
        ValueError: If a size-temperature pair is missing, a value is NaN, or a value
            exceeds the plotted range (Y_MAX_PERCENT).
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
    not_a_number = [key for key, value in percent.items() if math.isnan(value)]
    if not_a_number:
        raise ValueError(f"{data_dir / 'census.csv'}: sp3_fraction is NaN for {not_a_number}")
    if max(percent.values()) > Y_MAX_PERCENT:
        raise ValueError(f"an sp3 fraction exceeds the plotted {Y_MAX_PERCENT}%")
    return percent


def split_sizes(percent: dict[tuple[int, int], float]) -> tuple[list[int], list[int]]:
    """Sizes drawn in ink (ever above INK_ABOVE_PERCENT) and the rest, from the data.

    Args:
        percent: From `read_percentages`.

    Returns:
        (ink sizes, grey sizes), each in increasing size.

    Raises:
        ValueError: If the grey sizes are not the smallest sizes in one unbroken run, since
            their shared label names a range ("C40–C160").
    """
    ink = [
        n_atoms
        for n_atoms in SIZES
        if max(percent[n_atoms, kelvin] for kelvin in TEMPERATURES_KELVIN) > INK_ABOVE_PERCENT
    ]
    grey = [n_atoms for n_atoms in SIZES if n_atoms not in ink]
    if grey != list(SIZES[: len(grey)]):
        raise ValueError(f"grey sizes {grey} are not the smallest sizes in one unbroken run")
    return ink, grey


def build(data_dir: Path) -> str:
    """The sp3-against-temperature plot as an SVG document.

    Args:
        data_dir: Directory holding census.csv.

    Returns:
        The SVG document as a string.

    Raises:
        ValueError: From `read_percentages` or `split_sizes`.
    """
    percent = read_percentages(data_dir)
    ink_sizes, grey_sizes = split_sizes(percent)
    coldest = TEMPERATURES_KELVIN[0]
    left, top = MARGIN_LEFT, MARGIN_TOP

    parts = [
        f'<path d="M {left:.2f} {top:.2f} V {BASELINE:.2f} H {left + PLOT_WIDTH:.2f}" '
        f'fill="none" stroke="var(--graphite)" stroke-width="{AXIS_WIDTH}"/>'
    ]
    for value_percent in Y_TICKS_PERCENT:
        tick_y = percent_to_y(value_percent)
        parts.append(
            f'<line x1="{left - TICK_LENGTH:.2f}" y1="{tick_y:.2f}" x2="{left:.2f}" '
            f'y2="{tick_y:.2f}" stroke="var(--graphite)" stroke-width="{AXIS_WIDTH}"/>'
        )
        parts.append(
            svg_text(left - TICK_LENGTH - 1.2, tick_y + 1.0, str(value_percent), LABEL_SIZE, "end")
        )
    for kelvin in TEMPERATURES_KELVIN:
        tick_x = temperature_to_x(kelvin)
        parts.append(
            f'<line x1="{tick_x:.2f}" y1="{BASELINE:.2f}" x2="{tick_x:.2f}" '
            f'y2="{BASELINE + TICK_LENGTH:.2f}" stroke="var(--graphite)" '
            f'stroke-width="{AXIS_WIDTH}"/>'
        )
        parts.append(
            svg_text(tick_x, BASELINE + TICK_LENGTH + LABEL_SIZE + 0.4, str(kelvin), LABEL_SIZE)
        )
    parts.append(
        svg_text(
            left + LABEL_COLUMN + (PLOT_WIDTH - LABEL_COLUMN) / 2,
            BASELINE + TICK_LENGTH + 2 * LABEL_SIZE + 3.2,
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
        tone = "var(--ink)" if n_atoms in ink_sizes else "var(--graphite)"
        points = [
            (kelvin, temperature_to_x(kelvin), percent_to_y(percent[n_atoms, kelvin]))
            for kelvin in TEMPERATURES_KELVIN
        ]
        path = " ".join(f"{point_x:.2f},{point_y:.2f}" for _, point_x, point_y in points)
        parts.append(
            f'<polyline data-size="{n_atoms}" points="{path}" fill="none" stroke="{tone}" '
            f'stroke-width="{LINE_WIDTH}" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        parts.extend(
            f'<circle data-run="C{n_atoms}-{kelvin}K" cx="{point_x:.2f}" cy="{point_y:.2f}" '
            f'r="{MARKER_RADIUS}" fill="{tone}" stroke="none"/>'
            for kelvin, point_x, point_y in points
        )

    labels = sorted(
        [(percent_to_y(percent[n_atoms, coldest]) + 1.0, f"C{n_atoms}") for n_atoms in ink_sizes]
        + [
            (
                percent_to_y(max(percent[n_atoms, coldest] for n_atoms in grey_sizes)) + 1.0,
                f"C{grey_sizes[0]}–C{grey_sizes[-1]}",
            )
        ]
    )
    minimum_gap = LABEL_SIZE * 1.25
    placed: list[tuple[float, str]] = []
    for label_y, text in reversed(labels):  # bottom (largest y) first, then upwards
        if placed and label_y > placed[-1][0] - minimum_gap:
            label_y = placed[-1][0] - minimum_gap
        placed.append((label_y, text))
    label_x = temperature_to_x(coldest) - MARKER_RADIUS - 1.6
    parts.extend(svg_text(label_x, label_y, text, LABEL_SIZE, "end") for label_y, text in placed)

    width = left + PLOT_WIDTH + MARGIN_RIGHT
    height = BASELINE + MARGIN_BOTTOM
    body = "\n    ".join(parts)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.2f} {height:.2f}" '
        f'role="img">\n  <g>\n    {body}\n  </g>\n</svg>\n'
    )


def main(argv: list[str] | None = None) -> None:
    """Write the sp3 plot and print the highest value it draws.

    The data are checked (complete, no NaN, in range) inside `build`, before anything is
    written.

    Args:
        argv: Command-line arguments; None reads `sys.argv`.

    Raises:
        ValueError: From `build`, if census.csv is incomplete, holds NaN or is out of range.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    svg = build(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg)
    percent = read_percentages(args.data_dir)
    (peak_atoms, peak_kelvin), peak_percent = max(percent.items(), key=lambda item: item[1])
    ink_sizes, grey_sizes = split_sizes(percent)
    print(f"  highest sp3: C{peak_atoms}-{peak_kelvin}K {peak_percent:.1f}%")
    print(f"  ink: {ink_sizes}; grey: {grey_sizes}")
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
