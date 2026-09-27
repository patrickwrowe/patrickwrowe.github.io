"""Shared drawing style for the site's hand-written SVG data figures.

The figure scripts (graphitisation_figure.py, cluster_outcomes.py, cluster_sp3.py,
cluster_benchmark_c60.py) emit
SVG by hand so that every stroke and fill resolves to a design token (`var(--ink)`,
`var(--graphite)`, `var(--plate)`) and no hue enters a figure (spec 01 section 6.2). The
viewBox is unitless, so every size here is in drawing units, not pixels; the figure
scales with its container.

Import it as a sibling module (`from figure_style import ...`); the scripts run from
`scripts/figures/`, which is then on the import path.
"""

from __future__ import annotations

__all__ = ["AXIS_TITLE_SIZE", "AXIS_WIDTH", "LABEL_SIZE", "TICK_LENGTH", "svg_text"]

AXIS_WIDTH = 0.45
TICK_LENGTH = 1.6
LABEL_SIZE = 3.0
AXIS_TITLE_SIZE = 3.2


def svg_text(
    x_position: float,
    y_position: float,
    content: str,
    size: float,
    anchor: str = "middle",
    rotate: float | None = None,
) -> str:
    """A text element in the site's mono face, filled with `var(--graphite)`.

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
        f"{transform}>{content}</text>"
    )
