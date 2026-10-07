# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Orientation of water in the first and second contact layers above K- and Li-vermiculite.

Redraws Fig. 4c and 4d of Huang et al., "Cation-controlled wetting properties of vermiculite
membranes", Nat. Commun. 11, 1097 (2020), as a matrix of four panels: one column per angle,
one row per cation, the first contact layer solid and the second dashed in the cation's
colour. Within a panel that is the comparison the paper draws from it (the K layers differ,
the Li layers hardly do); down a column the two cations share their axes.

The angles: theta is that between an O-H bond and the surface normal, phi that between the
plane of the water molecule and the basal plane (the paper's caption). Both are plotted as
the files give them, as the cosine, which is dimensionless; the distributions' normalisation
is not stated (their integrals are not 1: K's first-layer theta integrates to 0.114), so
their axes are in arbitrary units, as the paper's Fig. 4a and 4b are. Some values are
slightly negative where the smoothing of the distribution overshoots (down to -0.073 at the
upper edge of K's first-layer theta); they are drawn, not clipped.

Data: papers/vermiculite-membranes/originals/{k,li}_{oh_vector,hoh_plane}_angle_layer1_layer2_
data.dat, copied unchanged; each is cos, probability for layer 1, cos, probability for layer
2 (the two cos columns are the same). They equal the Source Data sheets Fig. 4c (theta, 100
rows) and Fig. 4d (phi) to the sheet's rounding, which scripts/tests/test_vermiculite_contact_
angles.py checks. The phi files run from cos -1 to 0.98 in 100 bins and are mirror-symmetric
(bin j and bin 99 - j hold the same value, so the bin starting at cos -0.02 and the one
starting at 0 agree), as a plane's angle to a plane is; the paper's axis and this one show the
half from 0 to 1. The cos column is each bin's lower edge, drawn as the files give it.

House style of the site's figures (figure_style.py): emitted as SVG by hand over a unitless
viewBox, strokes in `var(--graphite)` and each cation's curves in its `--species-*` token
(constants.SPECIES_COLOUR); a swatch beside each panel's cation name says which. Each panel
is a `<g data-panel data-variable>` and each curve a `<g data-series>`, so the figure can be
checked against the data without reading pixels. The mono face's loaded subset (latin) has no
Greek, so theta and phi fall back to the next monospace font glyph by glyph.

`--narrow` draws the same panels, curves and labels in a viewBox 31.5 em of LABEL_SIZE wide
(the site's phone-width budget below 700 px, Plate.astro's `.chart--narrow`): the two columns
stacked and the legend on two lines. No label is smaller than LABEL_SIZE.

Usage:
    FIGURES=src/content/work/figures/two-dimensional-and-confined
    uv run scripts/figures/vermiculite_contact_angles.py \
        --data-dir scripts/figures/data/two-dimensional-and-confined \
        --output $FIGURES/vermiculite-contact-angles.svg

    uv run scripts/figures/vermiculite_contact_angles.py \
        --data-dir scripts/figures/data/two-dimensional-and-confined \
        --output $FIGURES/vermiculite-contact-angles-narrow.svg --narrow
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from figure_style import (
    AXIS_TITLE_SIZE,
    DASHED_SERIES,
    LABEL_SIZE,
    NARROW_CHART_EM,
    SERIES_WIDTH,
    TICK_LENGTH,
    format_tick,
    species_colour,
    svg_axis,
    svg_document,
    svg_line,
    svg_series,
    svg_text,
)

CATIONS = ("K", "Li")
COLUMN_COUNT = 4

# The two angles. `file` is the middle of the data file's name; `fold` keeps the half of the
# bins from cos 0 up (the distribution is mirror-symmetric about 0); the cosine axis and the
# probability axis (arbitrary units) are given as (minimum, maximum) and their ticks with the
# decimals their labels print.
ANGLES = {
    "theta": {
        "file": "oh_vector",
        "heading": "θ: O–H bond against the surface normal",
        "cos_label": "cos θ (dimensionless)",
        "probability_label": "P(cos θ) (a.u.)",
        "fold": False,
        "cos_axis": (-0.8, 0.8),
        "cos_ticks": (-0.5, 0.0, 0.5),
        "probability_axis": (-0.1, 0.25),
        "probability_ticks": (0.0, 0.1, 0.2),
        "probability_decimals": 1,
    },
    "phi": {
        "file": "hoh_plane",
        "heading": "φ: H–O–H plane against the surface",
        "cos_label": "cos φ (dimensionless)",
        "probability_label": "P(cos φ) (a.u.)",
        "fold": True,
        "cos_axis": (0.0, 1.0),
        "cos_ticks": (0.0, 0.5, 1.0),
        "probability_axis": (0.0, 0.13),
        "probability_ticks": (0.0, 0.05, 0.10),
        "probability_decimals": 2,
    },
}
# A bin starting at -0.000004 is the bin starting at 0 (the files' grid is 0.02 apart): the
# fold keeps it and the axis check lets it stand at the axis's end.
COS_TOLERANCE = 1e-3

# (series key, column of its probability, stroke-dasharray, legend label).
LAYERS = (
    ("layer1", 1, None, "first contact layer"),
    ("layer2", 3, DASHED_SERIES, "second contact layer"),
)

# Geometry in viewBox user units. The wide layout puts the angles side by side, the two
# cations stacked in each.
PLOT_WIDTH = 75.0
PLOT_HEIGHT = 34.0
MARGIN_LEFT = 15.5
MARGIN_RIGHT = 4.0
# Above a group's first panel: the angle's heading. Above each plot: its cation. Below each
# plot: tick labels and air. After the last plot of a group: the x axis title.
HEADING_BAND = 7.0
PANEL_HEADER = 7.0
PANEL_BELOW = 7.0
X_TITLE_BAND = 5.0
Y_TITLE_OFFSET = 11.5

# --narrow: the angles stack, each as wide as the 31.5 em viewBox leaves after the margins.
NARROW_WIDTH = NARROW_CHART_EM * LABEL_SIZE
NARROW_PLOT_WIDTH = NARROW_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
NARROW_PLOT_HEIGHT = 26.0

# The legend of line styles, at the top: a row in the wide layout, a column in the narrow.
LEGEND_ROW_HEIGHT = 5.0
LEGEND_SAMPLE_LENGTH = 8.0
LEGEND_PITCH = 62.0
SWATCH_LENGTH = 5.0


def build(data_dir: Path, narrow: bool = False) -> str:
    """The contact-angle figure as an SVG document.

    Args:
        data_dir: Directory holding the four `{k,li}_{oh_vector,hoh_plane}_angle_layer1_
            layer2_data.dat` files.
        narrow: Draw the 31.5 em phone-width layout instead of the wide one.

    Returns:
        The SVG document as a string.

    Raises:
        ValueError: If a data file does not have four finite columns, its two cosine columns
            differ, or a distribution leaves the axes, which would draw a curve outside its
            panel.
    """
    plot_width = NARROW_PLOT_WIDTH if narrow else PLOT_WIDTH
    plot_height = NARROW_PLOT_HEIGHT if narrow else PLOT_HEIGHT
    panel_pitch = PANEL_HEADER + plot_height + PANEL_BELOW
    group_height = HEADING_BAND + len(CATIONS) * panel_pitch + X_TITLE_BAND
    legend_height = 3.0 + LEGEND_ROW_HEIGHT * (len(LAYERS) if narrow else 1)

    parts: list[str] = []
    for layer_index, (layer_key, _column, dash, label) in enumerate(LAYERS):
        legend_x = MARGIN_LEFT + (0 if narrow else layer_index * LEGEND_PITCH)
        legend_y = 3.5 + (layer_index * LEGEND_ROW_HEIGHT if narrow else 0)
        parts.append(
            f'<g data-legend="{layer_key}">'
            + svg_line(
                legend_x,
                legend_y,
                legend_x + LEGEND_SAMPLE_LENGTH,
                legend_y,
                SERIES_WIDTH,
                dash,
            )
            + svg_text(
                legend_x + LEGEND_SAMPLE_LENGTH + 2.0, legend_y + 1.0, label, LABEL_SIZE, "start"
            )
            + "</g>"
        )

    for group_index, (angle, spec) in enumerate(ANGLES.items()):
        if narrow:
            group_left = MARGIN_LEFT
            group_top = legend_height + group_index * group_height
        else:
            group_left = MARGIN_LEFT + group_index * (PLOT_WIDTH + MARGIN_LEFT)
            group_top = legend_height
        cos_min, cos_max = spec["cos_axis"]
        probability_min, probability_max = spec["probability_axis"]
        # viewBox units per unit of cosine, and per arbitrary unit of probability.
        x_scale = plot_width / (cos_max - cos_min)
        y_scale = plot_height / (probability_max - probability_min)
        parts.append(
            svg_text(group_left, group_top + 4.5, spec["heading"], AXIS_TITLE_SIZE, "start")
        )

        for panel_index, cation in enumerate(CATIONS):
            path = data_dir / f"{cation.lower()}_{spec['file']}_angle_layer1_layer2_data.dat"
            columns = np.loadtxt(path, ndmin=2)
            if columns.shape[1] != COLUMN_COUNT or not np.isfinite(columns).all():
                raise ValueError(f"{path}: expected {COLUMN_COUNT} finite columns")
            if not np.array_equal(columns[:, 0], columns[:, 2]):
                raise ValueError(f"{path}: the two layers' cosine columns differ")
            if spec["fold"]:
                columns = columns[columns[:, 0] >= -COS_TOLERANCE]

            plot_top = group_top + HEADING_BAND + panel_index * panel_pitch + PANEL_HEADER
            plot_bottom = plot_top + plot_height
            colour = species_colour(cation)
            panel = [
                svg_line(
                    group_left,
                    plot_top - 3.5,
                    group_left + SWATCH_LENGTH,
                    plot_top - 3.5,
                    SERIES_WIDTH,
                    colour=colour,
                ),
                svg_text(
                    group_left + SWATCH_LENGTH + 2.0,
                    plot_top - 2.5,
                    f"{cation}-vermiculite",
                    AXIS_TITLE_SIZE,
                    "start",
                ),
            ]
            if probability_min < 0:
                zero_y = plot_bottom + probability_min * y_scale
                panel.append(svg_line(group_left, zero_y, group_left + plot_width, zero_y))
            panel += svg_axis(
                "bottom",
                group_left,
                plot_bottom,
                plot_width,
                [((cos - cos_min) * x_scale, format_tick(cos, 1)) for cos in spec["cos_ticks"]],
            )
            panel += svg_axis(
                "left",
                group_left,
                plot_bottom,
                plot_height,
                [
                    (
                        (value - probability_min) * y_scale,
                        format_tick(value, spec["probability_decimals"]),
                    )
                    for value in spec["probability_ticks"]
                ],
            )
            panel.append(
                svg_text(
                    group_left - Y_TITLE_OFFSET,
                    plot_top + plot_height / 2,
                    spec["probability_label"],
                    AXIS_TITLE_SIZE,
                    rotate=-90,
                )
            )
            if panel_index == len(CATIONS) - 1:
                panel.append(
                    svg_text(
                        group_left + plot_width / 2,
                        plot_bottom + TICK_LENGTH + 2 * LABEL_SIZE + 3.2,
                        spec["cos_label"],
                        AXIS_TITLE_SIZE,
                    )
                )

            for layer_key, column, dash, _label in LAYERS:
                probability = columns[:, column]
                if not (
                    cos_min - COS_TOLERANCE <= columns[:, 0].min()
                    and columns[:, 0].max() <= cos_max + COS_TOLERANCE
                    and probability_min <= probability.min()
                    and probability.max() <= probability_max
                ):
                    raise ValueError(f"{path}: {layer_key} leaves the axes")
                x_positions = group_left + (columns[:, 0] - cos_min) * x_scale
                y_positions = plot_bottom - (probability - probability_min) * y_scale
                curve = svg_series(
                    list(zip(x_positions.tolist(), y_positions.tolist(), strict=True)), colour, dash
                )
                panel.append(f'<g data-series="{layer_key}">{curve}</g>')

            panel_body = "\n    ".join(panel)
            parts.append(
                f'<g data-panel="{cation}" data-variable="{angle}">\n    {panel_body}\n  </g>'
            )

    if narrow:
        width, height = NARROW_WIDTH, legend_height + len(ANGLES) * group_height
    else:
        width = len(ANGLES) * (MARGIN_LEFT + PLOT_WIDTH) + MARGIN_RIGHT
        height = legend_height + group_height
    return svg_document(width, height, parts)


def main(argv: list[str] | None = None) -> None:
    """Write the contact-angle figure.

    Args:
        argv: Command-line arguments; None reads `sys.argv`.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--narrow", action="store_true", help="write the phone-width variant instead"
    )
    args = parser.parse_args(argv)
    svg = build(args.data_dir, args.narrow)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg)
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
