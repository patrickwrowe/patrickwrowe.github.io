# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Dot-and-line benchmark of C₆₀ (Iₕ) cohesive energy against DFT, one row per method.

DFT sits first, then the seven potentials Karasulu et al. benchmarked against it, in the
paper's own order. A dashed vertical line marks DFT's value; every other row's horizontal
stem runs from that line to the row's own value, so the drawing itself is the deviation
from DFT the article's prose discusses. DFT and its GAP-20 surrogate share the filled
marker — both are ab initio-grade values GAP-20 was fitted to reproduce — while the six
classical, empirically-fitted potentials get the open marker. This is the same
filled-versus-open device graphitisation_figure.py uses to tell two series apart without
spending a second colour, just drawn across three groups instead of two.

House style of the site's figures (figure_style.py): emitted as SVG by hand so that every
stroke and fill resolves to `var(--ink)`, `var(--graphite)` or `var(--plate)`, with no hue,
over a unitless viewBox.

Data: docs/dossiers/carbon/dossier.md lines 472-473, Table II of Karasulu et al., "A
transferable machine-learning potential for carbon", Carbon 191, 255-266 (2022) — cohesive
energy of the icosahedral C₆₀ isomer, eV per atom, copied verbatim.

`--narrow` draws the same rows, marks and labels in a viewBox 31.5 em of LABEL_SIZE wide
(the site's phone-width budget below 700 px, Plate.astro's `.chart--narrow`, as
cluster_outcomes.py --narrow): the method-label column shrinks to the longest label and the
energy axis takes the rest. No label is smaller than LABEL_SIZE.

Usage:
    uv run scripts/figures/cluster_benchmark_c60.py \
        --output src/content/work/figures/carbon/cluster-benchmark-c60.svg

    uv run scripts/figures/cluster_benchmark_c60.py \
        --output src/content/work/figures/carbon/cluster-benchmark-c60-narrow.svg --narrow
"""

from __future__ import annotations

import argparse
from pathlib import Path

from figure_style import (
    AXIS_TITLE_SIZE,
    AXIS_WIDTH,
    LABEL_SIZE,
    NARROW_CHART_EM,
    TICK_LENGTH,
    svg_text,
)

# (method label, cohesive energy in eV per atom, marker kind). Order is DFT first, then
# the seven potentials as Table II lists them.
METHODS: tuple[tuple[str, float, str], ...] = (
    ("DFT", -7.47, "reference"),
    ("GAP-20", -7.57, "gap"),
    ("ReaxFF", -7.17, "classical"),
    ("LCBOP-I", -6.93, "classical"),
    ("REBO-II", -6.84, "classical"),
    ("AIREBO", -6.81, "classical"),
    ("Tersoff", -6.73, "classical"),
    ("C-EDIP", -6.56, "classical"),
)

DFT_REFERENCE_EV_PER_ATOM = -7.47

# Plot geometry, in viewBox user units (unitless, like graphitisation_figure.py).
LABEL_END_X = 34.0
LABEL_GAP = 3.0
PLOT_LEFT = LABEL_END_X + LABEL_GAP
PLOT_WIDTH = 95.0
MARGIN_TOP = 6.0
MARGIN_RIGHT = 5.0
MARGIN_BOTTOM = 20.0
ROW_HEIGHT = 8.5
PLOT_HEIGHT = ROW_HEIGHT * len(METHODS)

DOMAIN_PADDING_EV_PER_ATOM = 0.15
DOMAIN_MIN_EV_PER_ATOM = min(value for _, value, _ in METHODS) - DOMAIN_PADDING_EV_PER_ATOM
DOMAIN_MAX_EV_PER_ATOM = max(value for _, value, _ in METHODS) + DOMAIN_PADDING_EV_PER_ATOM

X_TICKS_EV_PER_ATOM = (-7.6, -7.4, -7.2, -7.0, -6.8, -6.6)

# --narrow: the label column fits the longest method label, seven characters of the mono
# face (0.6 em advance) at LABEL_SIZE, 12.6 units, ending at NARROW_LABEL_END_X; the axis
# takes the rest of the 31.5 em viewBox. Its tick labels, four characters each (7.2 units),
# then sit 10.9 units apart.
NARROW_WIDTH = NARROW_CHART_EM * LABEL_SIZE
NARROW_LABEL_END_X = 15.0
NARROW_PLOT_LEFT = NARROW_LABEL_END_X + LABEL_GAP
NARROW_PLOT_WIDTH = NARROW_WIDTH - NARROW_PLOT_LEFT - MARGIN_RIGHT

STEM_WIDTH = 0.6
MARKER_STROKE = 0.7
MARKER_RADIUS = 1.7

MINUS_SIGN = "−"


def _format_ev_per_atom(cohesive_energy_ev_per_atom: float) -> str:
    """Format a cohesive energy to one decimal place with a true minus sign.

    The mono face draws U+2212 as a real minus, matching the prose; Python's own
    float formatting uses hyphen-minus, which reads shorter and sits high next to
    a digit.

    Args:
        cohesive_energy_ev_per_atom: Cohesive energy of C60, in eV per atom.

    Returns:
        The value formatted to one decimal place, e.g. "−7.6" or "1.0".
    """
    return f"{cohesive_energy_ev_per_atom:.1f}".replace("-", MINUS_SIGN)


def x_for_value(
    cohesive_energy_ev_per_atom: float,
    plot_left: float = PLOT_LEFT,
    plot_width: float = PLOT_WIDTH,
) -> float:
    """Map a C₆₀ cohesive energy to an x position in viewBox units.

    Args:
        cohesive_energy_ev_per_atom: Cohesive energy of C₆₀, in eV per atom.
        plot_left: x of the axis's left end, in viewBox units; the wide layout's default.
        plot_width: Length of the axis, in viewBox units; the wide layout's default.

    Returns:
        The x coordinate in viewBox user units.
    """
    span = DOMAIN_MAX_EV_PER_ATOM - DOMAIN_MIN_EV_PER_ATOM
    fraction = (cohesive_energy_ev_per_atom - DOMAIN_MIN_EV_PER_ATOM) / span
    return plot_left + fraction * plot_width


def _marker(pos_x: float, pos_y: float, filled: bool) -> str:
    """A row's data marker: filled for DFT and GAP-20, open for the classical potentials.

    The open marker is filled with `var(--plate)`, not "none", so it knocks out the stem
    line behind it rather than reading as a bead threaded on a wire — the same reasoning
    graphitisation_figure.py's `_series` gives for its own open markers.

    Args:
        pos_x: Marker centre x, in viewBox units.
        pos_y: Marker centre y, in viewBox units.
        filled: True for a filled marker (DFT, GAP-20), False for an open one.

    Returns:
        The SVG element as a string.
    """
    if filled:
        return (
            f'<circle cx="{pos_x:.2f}" cy="{pos_y:.2f}" r="{MARKER_RADIUS}" '
            f'fill="var(--ink)" stroke="none"/>'
        )
    radius = MARKER_RADIUS - MARKER_STROKE / 2
    return (
        f'<circle cx="{pos_x:.2f}" cy="{pos_y:.2f}" r="{radius:.2f}" fill="var(--plate)" '
        f'stroke="var(--ink)" stroke-width="{MARKER_STROKE}"/>'
    )


def build(narrow: bool = False) -> str:
    """The C₆₀ cohesive-energy benchmark figure as an SVG document.

    Args:
        narrow: Draw the 31.5 em phone-width layout instead of the wide one.

    Returns:
        The SVG document as a string.
    """
    plot_left = NARROW_PLOT_LEFT if narrow else PLOT_LEFT
    plot_width = NARROW_PLOT_WIDTH if narrow else PLOT_WIDTH
    baseline_y = MARGIN_TOP + PLOT_HEIGHT
    reference_x = x_for_value(DFT_REFERENCE_EV_PER_ATOM, plot_left, plot_width)

    parts: list[str] = [
        # Vertical reference line for DFT, drawn first so the rows sit on top of it. Its x
        # is printed to full float precision, not the usual .2f, so a test can check it
        # against x_for_value() without a rounding error of its own to allow for.
        f'<line x1="{reference_x:.6f}" y1="{MARGIN_TOP:.2f}" x2="{reference_x:.6f}" '
        f'y2="{baseline_y:.2f}" stroke="var(--graphite)" stroke-width="{AXIS_WIDTH}" '
        f'stroke-dasharray="2 1.6" data-reference="DFT"/>',
        # Bottom axis.
        f'<line x1="{plot_left:.2f}" y1="{baseline_y:.2f}" x2="{plot_left + plot_width:.2f}" '
        f'y2="{baseline_y:.2f}" stroke="var(--graphite)" stroke-width="{AXIS_WIDTH}"/>',
    ]

    for tick_value in X_TICKS_EV_PER_ATOM:
        tick_x = x_for_value(tick_value, plot_left, plot_width)
        parts.append(
            f'<line x1="{tick_x:.2f}" y1="{baseline_y:.2f}" x2="{tick_x:.2f}" '
            f'y2="{baseline_y + TICK_LENGTH:.2f}" stroke="var(--graphite)" '
            f'stroke-width="{AXIS_WIDTH}"/>'
        )
        tick_label_y = baseline_y + TICK_LENGTH + LABEL_SIZE + 0.4
        parts.append(svg_text(tick_x, tick_label_y, _format_ev_per_atom(tick_value), LABEL_SIZE))
    parts.append(
        svg_text(
            plot_left + plot_width / 2,
            baseline_y + TICK_LENGTH + LABEL_SIZE * 2 + 3.2,
            # Plain "C60": the mono face's loaded subset has no subscript digits, so
            # "C₆₀" falls back glyph by glyph and reads "C 6 0" (design review C7).
            # Mono data labels stay plain, like the grid labels ("C40", not "C₄₀").
            "Cohesive energy of C60 (eV per atom)",
            AXIS_TITLE_SIZE,
        )
    )

    for row_index, (method_label, cohesive_energy_ev_per_atom, marker_kind) in enumerate(METHODS):
        row_y = MARGIN_TOP + (row_index + 0.5) * ROW_HEIGHT
        value_x = x_for_value(cohesive_energy_ev_per_atom, plot_left, plot_width)
        filled = marker_kind in ("reference", "gap")

        row_parts = []
        if marker_kind != "reference":
            row_parts.append(
                f'<line x1="{reference_x:.2f}" y1="{row_y:.2f}" x2="{value_x:.2f}" '
                f'y2="{row_y:.2f}" stroke="var(--graphite)" stroke-width="{STEM_WIDTH}"/>'
            )
        row_parts.append(_marker(value_x, row_y, filled))
        label_x = plot_left - LABEL_GAP
        row_parts.append(svg_text(label_x, row_y + 1.0, method_label, LABEL_SIZE, "end"))

        row_body = "\n    ".join(row_parts)
        row_open = f'<g data-method="{method_label}" data-kind="{marker_kind}">'
        parts.append(f"{row_open}\n    {row_body}\n  </g>")

    width = plot_left + plot_width + MARGIN_RIGHT
    height = MARGIN_TOP + PLOT_HEIGHT + MARGIN_BOTTOM
    body = "\n  ".join(parts)
    # data-chart-em: the viewBox width in labels, the page wrapper's --chart-em.
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.2f} {height:.2f}" '
        f'role="img" data-chart-em="{width / LABEL_SIZE:.1f}">\n  <g>\n  {body}\n  </g>\n</svg>\n'
    )


def main(argv: list[str] | None = None) -> None:
    """Write the C₆₀ cohesive-energy benchmark figure.

    Args:
        argv: Command-line arguments; None reads `sys.argv`.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--narrow", action="store_true", help="write the phone-width variant instead"
    )
    args = parser.parse_args(argv)
    svg = build(args.narrow)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg)
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
