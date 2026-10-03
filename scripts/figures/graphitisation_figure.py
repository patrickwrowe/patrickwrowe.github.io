# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Two-panel SVG of the GAP-driven graphitisation anneals.

Left: how the annealed network's coordination depends on density, at a fixed
3500 K anneal. Right: how little it depends on the anneal temperature, at two
fixed densities either side of the transition. Together they say that density is
the control variable and temperature is not, which is the result.

Emitted as SVG rather than through matplotlib so that every stroke resolves to
`var(--ink)` and `var(--graphite)` and the figure picks up the page's tokens
(spec 01 section 6.2). Series are told apart by filled versus open markers, the
same device render_cluster.py uses for carbon versus oxygen, because the page has
one ink colour to spend.

Input files are `<coordination number> <percentage of atoms>`, averaged over the
three replicas of each run, as written by the project's own analysis notebook.

Usage:
    uv run scripts/figures/graphitisation_figure.py \
        --data-dir scripts/figures/data/graphitisation \
        --output archive/work/figures/carbon-gap-20/graphitisation.svg
"""

from __future__ import annotations

import argparse
from pathlib import Path

from figure_style import AXIS_TITLE_SIZE, AXIS_WIDTH, LABEL_SIZE, TICK_LENGTH, svg_text

DENSITIES_GCC = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
TEMPERATURES_K = [2000, 2500, 3000, 3500, 4000, 4500]

# Panel geometry in user units. The viewBox is unitless here (unlike
# render_cluster.py, where it is Angstrom), so these are just drawing units.
PANEL_WIDTH = 88.0
PANEL_HEIGHT = 56.0
PANEL_GAP = 26.0
MARGIN_LEFT = 15.0
MARGIN_TOP = 9.0
MARGIN_BOTTOM = 22.0

LINE_WIDTH = 0.9
MARKER_RADIUS = 1.5
PANEL_TITLE_SIZE = 3.4


def read_coordination(path: Path) -> dict[int, float]:
    """Return {coordination number: percentage of atoms}."""
    fractions: dict[int, float] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        coordination, percentage = line.split()
        fractions[int(float(coordination))] = float(percentage)
    return fractions


def _series(
    points: list[tuple[float, float]], filled: bool
) -> str:
    """A polyline with markers. Open markers read as a second series without colour."""
    path = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    parts = [
        f'<polyline points="{path}" fill="none" stroke="var(--ink)" '
        f'stroke-width="{LINE_WIDTH}" stroke-linejoin="round" stroke-linecap="round"'
        + (' stroke-dasharray="2.5 1.8"' if not filled else "")
        + "/>"
    ]
    for x, y in points:
        if filled:
            parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{MARKER_RADIUS}" '
                f'fill="var(--ink)" stroke="none"/>'
            )
        else:
            # Filled with --plate, not "none": the marker has to knock out the
            # line behind it or the open circle reads as a bead on a wire.
            parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{MARKER_RADIUS - LINE_WIDTH / 2:.2f}" '
                f'fill="var(--plate)" stroke="var(--ink)" stroke-width="{LINE_WIDTH}"/>'
            )
    return "\n    ".join(parts)


def _panel(
    origin_x: float,
    x_values: list[float],
    x_ticks: list[tuple[float, str]],
    series: list[tuple[list[float], bool, str]],
    x_title: str,
    panel_title: str,
    show_y_title: bool,
    y_title: str = "% of atoms",
    legend: bool = False,
) -> str:
    """One axes box with its series. y is always 0-100 per cent."""
    x_min, x_max = min(x_values), max(x_values)

    def sx(value: float) -> float:
        return origin_x + (value - x_min) / (x_max - x_min) * PANEL_WIDTH

    def sy(percentage: float) -> float:
        return MARGIN_TOP + PANEL_HEIGHT - percentage / 100.0 * PANEL_HEIGHT

    parts: list[str] = []
    baseline = MARGIN_TOP + PANEL_HEIGHT

    # Axes: left and bottom only. A full box would add two lines that carry
    # nothing, and the site's figures are quiet.
    parts.append(
        f'<path d="M {origin_x:.2f} {MARGIN_TOP:.2f} V {baseline:.2f} '
        f'H {origin_x + PANEL_WIDTH:.2f}" fill="none" stroke="var(--graphite)" '
        f'stroke-width="{AXIS_WIDTH}"/>'
    )

    for percentage in (0, 25, 50, 75, 100):
        y = sy(percentage)
        parts.append(
            f'<line x1="{origin_x - TICK_LENGTH:.2f}" y1="{y:.2f}" x2="{origin_x:.2f}" '
            f'y2="{y:.2f}" stroke="var(--graphite)" stroke-width="{AXIS_WIDTH}"/>'
        )
        parts.append(svg_text(origin_x - TICK_LENGTH - 1.2, y + 1.0, str(percentage), LABEL_SIZE, "end"))

    for value, label in x_ticks:
        x = sx(value)
        parts.append(
            f'<line x1="{x:.2f}" y1="{baseline:.2f}" x2="{x:.2f}" '
            f'y2="{baseline + TICK_LENGTH:.2f}" stroke="var(--graphite)" '
            f'stroke-width="{AXIS_WIDTH}"/>'
        )
        parts.append(svg_text(x, baseline + TICK_LENGTH + LABEL_SIZE + 0.4, label, LABEL_SIZE))

    end_labels: list[tuple[float, float, str]] = []
    for index, (percentages, filled, label) in enumerate(series):
        points = [(sx(x), sy(p)) for x, p in zip(x_values, percentages)]
        parts.append(_series(points, filled))
        if legend:
            # A keyed legend in the empty top-right. Series that both end near
            # zero cannot be labelled at the line end: the labels land on the
            # axis and on each other.
            sample_x = origin_x + PANEL_WIDTH * 0.62
            row_y = MARGIN_TOP + 4.0 + index * LABEL_SIZE * 1.7
            parts.append(_series([(sample_x, row_y), (sample_x + 7.0, row_y)], filled))
            parts.append(svg_text(sample_x + 9.5, row_y + 1.0, label, LABEL_SIZE, "start"))
        else:
            # Otherwise label at the right-hand end, which needs no legend at all.
            end_x, end_y = points[-1]
            end_labels.append((end_x + 2.6, end_y + 1.0, label))

    # Where two series converge, their end labels would print on top of one
    # another. Separate them vertically, keeping their original order.
    minimum_gap = LABEL_SIZE * 1.4
    end_labels.sort(key=lambda item: item[1])
    for i in range(1, len(end_labels)):
        x, y, label = end_labels[i]
        floor = end_labels[i - 1][1] + minimum_gap
        if y < floor:
            end_labels[i] = (x, floor, label)
    for x, y, label in end_labels:
        parts.append(svg_text(x, y, label, LABEL_SIZE, "start"))

    parts.append(
        svg_text(origin_x + PANEL_WIDTH / 2, baseline + TICK_LENGTH + LABEL_SIZE * 2 + 3.2, x_title, AXIS_TITLE_SIZE)
    )
    parts.append(svg_text(origin_x, MARGIN_TOP - 3.4, panel_title, PANEL_TITLE_SIZE, "start"))
    if show_y_title:
        parts.append(
            svg_text(origin_x - 10.5, MARGIN_TOP + PANEL_HEIGHT / 2, y_title, AXIS_TITLE_SIZE, "middle", rotate=-90)
        )
    return "\n    ".join(parts)


def build(data_dir: Path) -> str:
    density = {d: read_coordination(data_dir / f"density-{d}gcc-coordination.txt") for d in DENSITIES_GCC}
    temp_low = {t: read_coordination(data_dir / f"temp-1.5gcc-{t}K-coordination.txt") for t in TEMPERATURES_K}
    temp_high = {t: read_coordination(data_dir / f"temp-3.0gcc-{t}K-coordination.txt") for t in TEMPERATURES_K}

    left = _panel(
        origin_x=MARGIN_LEFT,
        x_values=DENSITIES_GCC,
        x_ticks=[(d, f"{d:g}") for d in DENSITIES_GCC],
        series=[
            ([density[d][3] for d in DENSITIES_GCC], True, "sp²"),
            ([density[d][4] for d in DENSITIES_GCC], False, "sp³"),
        ],
        x_title="density / g cm⁻³",
        panel_title="(a) anneal at 3500 K",
        show_y_title=True,
    )

    right_origin = MARGIN_LEFT + PANEL_WIDTH + PANEL_GAP
    right = _panel(
        origin_x=right_origin,
        x_values=[float(t) for t in TEMPERATURES_K],
        x_ticks=[(float(t), f"{t / 1000:g}k") for t in TEMPERATURES_K],
        series=[
            ([temp_low[t][3] for t in TEMPERATURES_K], True, "sp², 1.5"),
            ([temp_high[t][4] for t in TEMPERATURES_K], False, "sp³, 3.0"),
        ],
        x_title="anneal temperature / K",
        panel_title="(b) fixed density",
        show_y_title=False,
    )

    width = right_origin + PANEL_WIDTH + 22.0
    height = MARGIN_TOP + PANEL_HEIGHT + MARGIN_BOTTOM

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.2f} {height:.2f}" role="img">\n'
        f"  <g>\n    {left}\n    {right}\n  </g>\n"
        f"</svg>\n"
    )


RING_SIZES = [3, 4, 5, 6, 7, 8, 9, 10]


def read_rings(path: Path) -> dict[int, float]:
    """Return {ring size: percentage of all rings counted}.

    The analysis writes absolute counts, averaged over the three replicas, and the
    totals differ by a factor of four across the density series. Percentages are
    what makes the shapes comparable.
    """
    counts = {int(float(a)): float(b) for a, b in (l.split() for l in path.read_text().splitlines() if l.strip())}
    total = sum(counts.values())
    return {size: 100.0 * counts.get(size, 0.0) / total for size in RING_SIZES}


def build_rings(data_dir: Path) -> str:
    """One panel: ring-size distribution either side of the density transition."""
    low = read_rings(data_dir / "density-1.0gcc-rings.txt")
    high = read_rings(data_dir / "density-3.5gcc-rings.txt")

    panel = _panel(
        origin_x=MARGIN_LEFT,
        x_values=[float(s) for s in RING_SIZES],
        x_ticks=[(float(s), str(s)) for s in RING_SIZES],
        series=[
            ([low[s] for s in RING_SIZES], True, "1.0 g cm⁻³"),
            ([high[s] for s in RING_SIZES], False, "3.5 g cm⁻³"),
        ],
        x_title="ring size",
        panel_title="rings after annealing at 3500 K",
        show_y_title=True,
        y_title="% of rings",
        legend=True,
    )
    width = MARGIN_LEFT + PANEL_WIDTH + 8.0
    height = MARGIN_TOP + PANEL_HEIGHT + MARGIN_BOTTOM
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.2f} {height:.2f}" role="img">\n'
        f"  <g>\n    {panel}\n  </g>\n"
        f"</svg>\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rings-output",
        type=Path,
        default=None,
        help="Also write the ring-size distribution figure here.",
    )
    args = parser.parse_args()

    svg = build(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg)

    if args.rings_output:
        args.rings_output.parent.mkdir(parents=True, exist_ok=True)
        args.rings_output.write_text(build_rings(args.data_dir))
        for label, name in (("1.0", "density-1.0gcc-rings.txt"), ("3.5", "density-3.5gcc-rings.txt")):
            rings = read_rings(args.data_dir / name)
            share = " ".join(f"{s}:{rings[s]:.0f}%" for s in RING_SIZES if rings[s] >= 1)
            print(f"  rings at {label} g/cc   {share}")
        print(f"-> {args.rings_output}")

    # Print the numbers the caption quotes, so a claim on the page can be checked
    # against the data without opening the notebook.
    for d in DENSITIES_GCC:
        fractions = read_coordination(args.data_dir / f"density-{d}gcc-coordination.txt")
        print(f"  {d:>3} g/cc   sp2 {fractions[3]:5.1f}%   sp3 {fractions[4]:5.1f}%")
    print(f"-> {args.output}")


def _self_check() -> None:
    """Parsing and scaling, on the shape the analysis notebook actually writes."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "c.txt"
        path.write_text(
            "1.000000000000000000e+00 4.572473466666666236e-02\n"
            "2.000000000000000000e+00 2.080475606666666533e+00\n"
            "3.000000000000000000e+00 9.721079253333334691e+01\n"
            "4.000000000000000000e+00 6.630086703333333276e-01\n"
            "5.000000000000000000e+00 0.000000000000000000e+00\n"
        )
        fractions = read_coordination(path)
        assert set(fractions) == {1, 2, 3, 4, 5}, fractions
        assert abs(fractions[3] - 97.21) < 0.01, fractions[3]
        # Percentages must sum to 100, or the file is not what we think it is.
        assert abs(sum(fractions.values()) - 100.0) < 0.05, sum(fractions.values())

    # An open marker knocks out the line behind it; a filled one does not.
    assert 'fill="var(--plate)"' in _series([(0.0, 0.0), (1.0, 1.0)], filled=False)
    assert 'fill="var(--ink)"' in _series([(0.0, 0.0), (1.0, 1.0)], filled=True)

    # Ring counts are absolute and the totals differ several-fold between
    # densities, so they must be normalised before the two can be compared.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "r.txt"
        path.write_text("5.0 100.0\n6.0 300.0\n")
        rings = read_rings(path)
        assert abs(rings[5] - 25.0) < 1e-9, rings
        assert abs(rings[6] - 75.0) < 1e-9, rings
        assert rings[3] == 0.0, rings  # sizes absent from the file read as zero
        assert abs(sum(rings.values()) - 100.0) < 1e-9, rings

    print("self-check ok")


if __name__ == "__main__":
    import sys

    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()
