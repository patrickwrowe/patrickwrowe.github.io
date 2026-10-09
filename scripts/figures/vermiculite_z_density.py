# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Probability density of the cation, water oxygen and water hydrogen above K- and Li-vermiculite.

Redraws Fig. 4a and 4b of Huang et al., "Cation-controlled wetting properties of vermiculite
membranes", Nat. Commun. 11, 1097 (2020), from the paper's own Source Data workbook: one
panel per cation, height above the surface on the vertical axis (zero is the mean z of the
surface oxygen atoms, the paper's caption) and probability density on the horizontal, as the
paper draws it. Every curve is one column pair of the workbook, drawn as it is stored.

Read two things off it. The ion is a narrow spike near the surface (K at one height, Li at
two). The first contact layer of water is a sharp oxygen peak above K and a lower, broader
one above Li.

The densities are histograms of 100 bins that each sum to 1, and the bins differ between
species: the ion's span about a hundredth of the water's width (K: 0.013 Å against 0.134 Å
for O). The paper plots them together in arbitrary units (the axis says "a.u."), so the
heights of an ion, an O and an H curve are not comparable in magnitude; positions are. Some
O and H bins at the edge of the film are slightly negative (smoothing overshoot, as the
paper's own curves show); they are drawn, not clipped, and the density axis starts below 0
to hold them.

Not the archive's `data_K.dat` and `data_Li.dat` (Vermiculite_Ion_Data): their columns are
labelled the other way round (the first of each pair is the density, the second the height),
and checked against these sheets on 2026-10-07 they are a different analysis. Their z ranges
(K ion 5.2-11.2 Å) put the cation 5 Å above the surface where the paper has it at 1.3 Å, and
their ion curves have two peaks 4.7 Å apart where the sheet's K curve has one. They are not
Fig. 4a and 4b and are not used.

House style of the site's figures (figure_style.py): emitted as SVG by hand over a unitless
viewBox, strokes in `var(--graphite)`, and each series in the `--species-*` token of its
species (K, Li, O, H; constants.SPECIES_COLOUR). A legend inside each panel names the
colours. Each panel is a `<g data-panel>` and each curve a `<g data-series>`, so the figure
can be checked against the data without reading pixels.

`--narrow` draws the same panels, curves and labels in a viewBox 31.5 em of LABEL_SIZE wide
(the site's phone-width budget below 700 px, Plate.astro's `.chart--narrow`), the two panels
stacked instead of side by side. No label is smaller than LABEL_SIZE.

Usage:
    FIGURES=src/content/work/figures/two-dimensional-and-confined
    uv run scripts/figures/vermiculite_z_density.py \
        --data-dir scripts/figures/data/two-dimensional-and-confined \
        --output $FIGURES/vermiculite-z-density.svg

    uv run scripts/figures/vermiculite_z_density.py \
        --data-dir scripts/figures/data/two-dimensional-and-confined \
        --output $FIGURES/vermiculite-z-density-narrow.svg --narrow
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from figure_style import (
    AXIS_BAND,
    AXIS_TITLE_SIZE,
    LABEL_SIZE,
    LEGEND_ROW_HEIGHT,
    NARROW_CHART_EM,
    SERIES_WIDTH,
    X_AXIS_TITLE_DROP,
    Y_TITLE_OFFSET,
    format_tick,
    species_colour,
    svg_axis,
    svg_document,
    svg_line,
    svg_series,
    svg_text,
    write_figure,
)

# Sheets "Fig. 4a" (K) and "Fig.4b" (Li) of the Source Data workbook, six columns each:
# density and height in angstrom for the ion, the water oxygen and the water hydrogen.
DATA_FILES = {"K": "fig4a-k-z-density.dat", "Li": "fig4b-li-z-density.dat"}
COLUMN_COUNT = 6

# (series key, species symbol or None for the panel's cation, legend label, density column),
# drawn first to last so the ion's spike lies on top of the water curves. The height
# is the column after the density.
SERIES = (
    ("hydrogen", "H", "water H", 4),
    ("oxygen", "O", "water O", 2),
    ("ion", None, "{cation} ion", 0),
)

HEIGHT_MIN_ANGSTROM = -1.0
HEIGHT_MAX_ANGSTROM = 17.0
HEIGHT_TICKS_ANGSTROM = (0, 5, 10, 15)
# Below zero by the deepest smoothing overshoot in the data (-0.0024 for K water H).
DENSITY_MIN_AU = -0.003
DENSITY_MAX_AU = 0.035
DENSITY_TICKS_AU = (0.0, 0.01, 0.02, 0.03)

# Plot geometry, in viewBox user units. The wide layout puts the panels side by side.
PLOT_WIDTH = 78.0
PLOT_HEIGHT = 100.0
MARGIN_LEFT = 13.0
MARGIN_TOP = 9.0
MARGIN_RIGHT = 4.0
PANEL_GAP = 10.0

# --narrow: the panels stack, each as wide as the 31.5 em viewBox leaves after the margins.
NARROW_WIDTH = NARROW_CHART_EM * LABEL_SIZE
NARROW_PLOT_WIDTH = NARROW_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
NARROW_PLOT_HEIGHT = 60.0

# The legend sits in the empty top-right of each panel, one row per series.
LEGEND_SWATCH_LENGTH = 5.0
LEGEND_WIDTH = 22.0


def build(data_dir: Path, narrow: bool = False) -> str:
    """The K- and Li-vermiculite z-density figure as an SVG document.

    Args:
        data_dir: Directory holding the two `DATA_FILES`.
        narrow: Draw the 31.5 em phone-width layout instead of the wide one.

    Returns:
        The SVG document as a string.

    Raises:
        ValueError: If a data file does not have six columns, has a value that is not
            finite, or has a density or height outside the axis, which would draw a curve
            that leaves its panel.
    """
    plot_width = NARROW_PLOT_WIDTH if narrow else PLOT_WIDTH
    plot_height = NARROW_PLOT_HEIGHT if narrow else PLOT_HEIGHT
    panel_height = MARGIN_TOP + plot_height + AXIS_BAND

    parts: list[str] = []
    for panel_index, cation in enumerate(DATA_FILES):
        path = data_dir / DATA_FILES[cation]
        columns = np.loadtxt(path, ndmin=2)
        if columns.shape[1] != COLUMN_COUNT or not np.isfinite(columns).all():
            raise ValueError(f"{path}: expected {COLUMN_COUNT} finite columns")

        if narrow:
            plot_left = MARGIN_LEFT
            plot_top = panel_index * panel_height + MARGIN_TOP
        else:
            plot_left = MARGIN_LEFT + panel_index * (PLOT_WIDTH + PANEL_GAP)
            plot_top = MARGIN_TOP
        plot_bottom = plot_top + plot_height
        plot_right = plot_left + plot_width

        panel = [
            svg_text(plot_left, plot_top - 3.0, f"{cation}-vermiculite", AXIS_TITLE_SIZE, "start")
        ]
        density_span_au = DENSITY_MAX_AU - DENSITY_MIN_AU
        height_span_angstrom = HEIGHT_MAX_ANGSTROM - HEIGHT_MIN_ANGSTROM
        panel += svg_axis(
            "bottom",
            plot_left,
            plot_bottom,
            plot_width,
            [
                (
                    (density_au - DENSITY_MIN_AU) / density_span_au * plot_width,
                    format_tick(density_au, 2),
                )
                for density_au in DENSITY_TICKS_AU
            ],
        )
        panel += svg_axis(
            "left",
            plot_left,
            plot_bottom,
            plot_height,
            [
                (
                    (height_angstrom - HEIGHT_MIN_ANGSTROM) / height_span_angstrom * plot_height,
                    format_tick(height_angstrom, 0),
                )
                for height_angstrom in HEIGHT_TICKS_ANGSTROM
            ],
        )
        panel.append(
            svg_text(
                plot_left + plot_width / 2,
                plot_bottom + X_AXIS_TITLE_DROP,
                "probability density (a.u.)",
                AXIS_TITLE_SIZE,
            )
        )
        if narrow or panel_index == 0:
            panel.append(
                svg_text(
                    plot_left - Y_TITLE_OFFSET,
                    plot_top + plot_height / 2,
                    "height above surface (Å)",
                    AXIS_TITLE_SIZE,
                    rotate=-90,
                )
            )

        for series_key, symbol, _label, density_column in SERIES:
            density_au = columns[:, density_column]
            height_angstrom = columns[:, density_column + 1]
            if not (
                DENSITY_MIN_AU <= density_au.min()
                and density_au.max() <= DENSITY_MAX_AU
                and HEIGHT_MIN_ANGSTROM <= height_angstrom.min()
                and height_angstrom.max() <= HEIGHT_MAX_ANGSTROM
            ):
                raise ValueError(f"{path}: {series_key} leaves the axes")
            x_positions = plot_left + (density_au - DENSITY_MIN_AU) / density_span_au * plot_width
            y_positions = (
                plot_bottom
                - (height_angstrom - HEIGHT_MIN_ANGSTROM) / height_span_angstrom * plot_height
            )
            colour = species_colour(symbol or cation)
            curve = svg_series(
                list(zip(x_positions.tolist(), y_positions.tolist(), strict=True)), colour
            )
            panel.append(f'<g data-series="{series_key}">{curve}</g>')

        for row_index, (_key, symbol, label, _column) in enumerate(reversed(SERIES)):
            row_y = plot_top + 4.0 + row_index * LEGEND_ROW_HEIGHT
            swatch_left = plot_right - LEGEND_WIDTH
            panel.append(
                f'<g data-legend="{symbol or cation}">'
                + svg_line(
                    swatch_left,
                    row_y,
                    swatch_left + LEGEND_SWATCH_LENGTH,
                    row_y,
                    SERIES_WIDTH,
                    colour=species_colour(symbol or cation),
                )
                + svg_text(
                    swatch_left + LEGEND_SWATCH_LENGTH + 2.0,
                    row_y + 1.0,
                    label.format(cation=cation),
                    LABEL_SIZE,
                    "start",
                )
                + "</g>"
            )

        panel_body = "\n    ".join(panel)
        parts.append(f'<g data-panel="{cation}">\n    {panel_body}\n  </g>')

    if narrow:
        width, height = NARROW_WIDTH, len(DATA_FILES) * panel_height
    else:
        width = MARGIN_LEFT + len(DATA_FILES) * PLOT_WIDTH + PANEL_GAP + MARGIN_RIGHT
        height = panel_height
    return svg_document(width, height, parts)


if __name__ == "__main__":
    write_figure(build, __doc__.splitlines()[0])
