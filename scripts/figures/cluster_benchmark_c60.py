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

House style of graphitisation_figure.py: emitted as SVG by hand so that every stroke and
fill resolves to `var(--ink)`, `var(--graphite)` or `var(--plate)`, with no hue, over a
unitless viewBox.

Data: docs/dossiers/carbon/dossier.md lines 472-473, Table II of Karasulu et al., "A
transferable machine-learning potential for carbon", Carbon 191, 255-266 (2022) — cohesive
energy of the icosahedral C₆₀ isomer, eV per atom, copied verbatim.

Usage:
    uv run scripts/figures/cluster_benchmark_c60.py \
        --output src/content/work/figures/carbon/cluster-benchmark-c60.svg
"""

from __future__ import annotations

import argparse
from pathlib import Path

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

AXIS_WIDTH = 0.45
STEM_WIDTH = 0.6
MARKER_STROKE = 0.7
MARKER_RADIUS = 1.7
TICK_LENGTH = 1.6
LABEL_SIZE = 3.0
AXIS_TITLE_SIZE = 3.2


def x_for_value(cohesive_energy_ev_per_atom: float) -> float:
    """Map a C₆₀ cohesive energy to an x position in viewBox units.

    Args:
        cohesive_energy_ev_per_atom: Cohesive energy of C₆₀, in eV per atom.

    Returns:
        The x coordinate in viewBox user units.
    """
    span = DOMAIN_MAX_EV_PER_ATOM - DOMAIN_MIN_EV_PER_ATOM
    fraction = (cohesive_energy_ev_per_atom - DOMAIN_MIN_EV_PER_ATOM) / span
    return PLOT_LEFT + fraction * PLOT_WIDTH


def _text(
    pos_x: float,
    pos_y: float,
    content: str,
    size: float,
    anchor: str = "middle",
) -> str:
    """A text element in the figure's mono face, filled with `var(--graphite)`.

    Args:
        pos_x: Anchor x, in viewBox units.
        pos_y: Baseline y, in viewBox units.
        content: The text.
        size: Font size, in viewBox units (unitless, matching graphitisation_figure.py).
        anchor: SVG text-anchor.

    Returns:
        The SVG element as a string.
    """
    return (
        f'<text x="{pos_x:.2f}" y="{pos_y:.2f}" text-anchor="{anchor}" font-size="{size}" '
        f'fill="var(--graphite)" stroke="none" font-family="var(--mono)">{content}</text>'
    )


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


def build() -> str:
    """The C₆₀ cohesive-energy benchmark figure as an SVG document.

    Returns:
        The SVG document as a string.
    """
    baseline_y = MARGIN_TOP + PLOT_HEIGHT
    reference_x = x_for_value(DFT_REFERENCE_EV_PER_ATOM)

    parts: list[str] = [
        # Vertical reference line for DFT, drawn first so the rows sit on top of it. Its x
        # is printed to full float precision, not the usual .2f, so a test can check it
        # against x_for_value() without a rounding error of its own to allow for.
        f'<line x1="{reference_x:.6f}" y1="{MARGIN_TOP:.2f}" x2="{reference_x:.6f}" '
        f'y2="{baseline_y:.2f}" stroke="var(--graphite)" stroke-width="{AXIS_WIDTH}" '
        f'stroke-dasharray="2 1.6" data-reference="DFT"/>',
        # Bottom axis.
        f'<line x1="{PLOT_LEFT:.2f}" y1="{baseline_y:.2f}" x2="{PLOT_LEFT + PLOT_WIDTH:.2f}" '
        f'y2="{baseline_y:.2f}" stroke="var(--graphite)" stroke-width="{AXIS_WIDTH}"/>',
    ]

    for tick_value in X_TICKS_EV_PER_ATOM:
        tick_x = x_for_value(tick_value)
        parts.append(
            f'<line x1="{tick_x:.2f}" y1="{baseline_y:.2f}" x2="{tick_x:.2f}" '
            f'y2="{baseline_y + TICK_LENGTH:.2f}" stroke="var(--graphite)" '
            f'stroke-width="{AXIS_WIDTH}"/>'
        )
        tick_label_y = baseline_y + TICK_LENGTH + LABEL_SIZE + 0.4
        parts.append(_text(tick_x, tick_label_y, f"{tick_value:.1f}", LABEL_SIZE))
    parts.append(
        _text(
            PLOT_LEFT + PLOT_WIDTH / 2,
            baseline_y + TICK_LENGTH + LABEL_SIZE * 2 + 3.2,
            "Cohesive energy of C₆₀ (eV per atom)",
            AXIS_TITLE_SIZE,
        )
    )

    for row_index, (method_label, cohesive_energy_ev_per_atom, marker_kind) in enumerate(METHODS):
        row_y = MARGIN_TOP + (row_index + 0.5) * ROW_HEIGHT
        value_x = x_for_value(cohesive_energy_ev_per_atom)
        filled = marker_kind in ("reference", "gap")

        row_parts = []
        if marker_kind != "reference":
            row_parts.append(
                f'<line x1="{reference_x:.2f}" y1="{row_y:.2f}" x2="{value_x:.2f}" '
                f'y2="{row_y:.2f}" stroke="var(--graphite)" stroke-width="{STEM_WIDTH}"/>'
            )
        row_parts.append(_marker(value_x, row_y, filled))
        row_parts.append(_text(PLOT_LEFT - LABEL_GAP, row_y + 1.0, method_label, LABEL_SIZE, "end"))

        row_body = "\n    ".join(row_parts)
        row_open = f'<g data-method="{method_label}" data-kind="{marker_kind}">'
        parts.append(f"{row_open}\n    {row_body}\n  </g>")

    width = PLOT_LEFT + PLOT_WIDTH + MARGIN_RIGHT
    height = MARGIN_TOP + PLOT_HEIGHT + MARGIN_BOTTOM
    body = "\n  ".join(parts)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.2f} {height:.2f}" '
        f'role="img">\n  <g>\n  {body}\n  </g>\n</svg>\n'
    )


def main(argv: list[str] | None = None) -> None:
    """Write the C₆₀ cohesive-energy benchmark figure.

    Args:
        argv: Command-line arguments; None reads `sys.argv`.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    svg = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg)
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
