# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Force and formation-energy errors of hBN-GAP and five classical potentials, one row each.

Dot chart of the RMSE of each potential against DFT in two panels that share their rows:
forces in eV/angstrom on a log axis (they run from 0.09 to 6.17), formation energies in meV
per atom on a linear one. hBN-GAP, the machine-learned potential the paper fits, is the
filled marker; the five classical potentials (Tersoff, ExTeP and ReaxFF, and Tersoff and
ExTeP again with the interlayer potential, ILP) are open ones: the filled-versus-open device
cluster_benchmark_c60.py draws GAP-20 and the classical potentials with. A dotted guide runs
from the axis's left end to each dot and stops there, so the value beside the dot is clear.
A dot, not a bar: a bar's length means nothing on a log axis.

Two of the energies are lower bounds. The paper says Tersoff and ExTeP, which cannot tell
one number of sheets from another for want of a cut-off beyond 2.0 angstrom, yield an RMSE
"of more than 54 and 67 meV/atom" (p.8); the chart labels them ">54" and ">67" and marks their
rows `data-bound="lower"`, from the `formation_energy_is_lower_bound` flags of the data.

Data: scripts/figures/data/two-dimensional-and-confined/hbn-potential-errors.json. The values
are the paper's own, Thiemann et al., J. Phys. Chem. C 2020: the force RMSEs on p.7 (on a
validation set of 1450 structures), the formation-energy RMSEs on p.8. Every value the chart
needs is printed there; none is added here. The paper is the source recorded in provenance.json.

House style of the site's figures (figure_style.py): emitted as SVG by hand over a unitless
viewBox, every stroke and fill in `var(--ink)`, `var(--graphite)` or `var(--plate)`, no hue.
Each panel is a `<g data-panel>` and each row a `<g data-potential>`, so the figure can be
checked against the data without reading pixels.

`--narrow` draws the same rows, markers and labels in a viewBox 31.5 em of LABEL_SIZE wide
(the site's phone-width budget below 700 px, Plate.astro's `.chart--narrow`): the two panels
stacked, each carrying its own potential labels. No label is smaller than LABEL_SIZE.

Usage:
    FIGURES=src/content/work/figures/two-dimensional-and-confined
    uv run scripts/figures/hbn_potential_errors.py \
        --data-dir scripts/figures/data/two-dimensional-and-confined \
        --output $FIGURES/hbn-potential-errors.svg

    uv run scripts/figures/hbn_potential_errors.py \
        --data-dir scripts/figures/data/two-dimensional-and-confined \
        --output $FIGURES/hbn-potential-errors-narrow.svg --narrow
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from figure_style import (
    AXIS_TITLE_SIZE,
    AXIS_WIDTH,
    LABEL_SIZE,
    MARKER_RADIUS,
    NARROW_CHART_EM,
    X_AXIS_TITLE_DROP,
    svg_axis,
    svg_document,
    svg_line,
    svg_marker,
    svg_text,
    write_figure,
)

DATA_FILE = "hbn-potential-errors.json"
POTENTIAL_FIELDS = (
    "name",
    "kind",
    "force_rmse_ev_per_angstrom",
    "formation_energy_rmse_mev_per_atom",
    "formation_energy_is_lower_bound",
)

# The two panels, left to right or top to bottom. `key` is the quantity's field in the data;
# `axis` is its (minimum, maximum) and `ticks` its labelled values, in the quantity's unit;
# `value_format` prints the number beside a dot.
QUANTITIES = {
    "force": {
        "key": "force_rmse_ev_per_angstrom",
        "title": "force RMSE (eV/Å, log scale)",
        "log": True,
        "axis": (0.05, 30.0),
        "ticks": (0.1, 1.0, 10.0),
        "value_format": ".2f",
    },
    "energy": {
        "key": "formation_energy_rmse_mev_per_atom",
        "title": "formation-energy RMSE (meV per atom)",
        "log": False,
        "axis": (0.0, 90.0),
        "ticks": (0.0, 20.0, 40.0, 60.0, 80.0),
        "value_format": "g",
    },
}

# Plot geometry, in viewBox user units. The wide layout puts the panels side by side, the
# potential labels once at the left; the narrow stacks them, each with its labels.
LABEL_END_X = 22.0
LABEL_GAP = 3.0
PLOT_LEFT = LABEL_END_X + LABEL_GAP
PLOT_WIDTH = 56.0
PANEL_GAP = 14.0
MARGIN_TOP = 5.0
MARGIN_RIGHT = 8.0
ROW_HEIGHT = 8.0
# Below a plot: tick labels, then the axis title, then a little air.
AXIS_BAND = 13.0
VALUE_LABEL_GAP = 3.0
GUIDE_DASH = "0.5 1.5"

NARROW_WIDTH = NARROW_CHART_EM * LABEL_SIZE
NARROW_MARGIN_RIGHT = 5.0
NARROW_PLOT_WIDTH = NARROW_WIDTH - PLOT_LEFT - NARROW_MARGIN_RIGHT
NARROW_ROW_HEIGHT = 7.5


def build(data_dir: Path, narrow: bool = False) -> str:
    """The hBN potential-error figure as an SVG document.

    Args:
        data_dir: Directory holding `hbn-potential-errors.json`.
        narrow: Draw the 31.5 em phone-width layout instead of the wide one.

    Returns:
        The SVG document as a string.

    Raises:
        ValueError: If the data lack a field, hold a value that is not a positive finite
            number, or hold one outside its axis, which would draw a dot off its panel.
    """
    path = data_dir / DATA_FILE
    potentials = json.loads(path.read_text())["potentials"]
    for potential in potentials:
        if missing := set(POTENTIAL_FIELDS) - set(potential):
            raise ValueError(f"{path}: {potential.get('name')} lacks {sorted(missing)}")
        for spec in QUANTITIES.values():
            value = potential[spec["key"]]
            low, high = spec["axis"]
            if not (isinstance(value, int | float) and math.isfinite(value) and 0 < value):
                raise ValueError(f"{path}: {potential['name']} {spec['key']} is {value!r}")
            if not low <= value <= high:
                raise ValueError(f"{path}: {potential['name']} {spec['key']} leaves its axis")

    plot_width = NARROW_PLOT_WIDTH if narrow else PLOT_WIDTH
    row_height = NARROW_ROW_HEIGHT if narrow else ROW_HEIGHT
    plot_height = row_height * len(potentials)
    panel_height = MARGIN_TOP + plot_height + AXIS_BAND

    parts: list[str] = []
    for panel_index, (quantity, spec) in enumerate(QUANTITIES.items()):
        plot_left = PLOT_LEFT if narrow or panel_index == 0 else PLOT_LEFT + plot_width + PANEL_GAP
        plot_top = MARGIN_TOP + (panel_index * panel_height if narrow else 0)
        plot_bottom = plot_top + plot_height
        axis_low, axis_high = spec["axis"]
        transform = math.log10 if spec["log"] else float
        axis_span = transform(axis_high) - transform(axis_low)

        panel = svg_axis(
            "bottom",
            plot_left,
            plot_bottom,
            plot_width,
            [
                ((transform(tick) - transform(axis_low)) / axis_span * plot_width, f"{tick:g}")
                for tick in spec["ticks"]
            ],
        )
        panel.append(
            svg_text(
                plot_left + plot_width / 2,
                plot_bottom + X_AXIS_TITLE_DROP,
                spec["title"],
                AXIS_TITLE_SIZE,
            )
        )

        for row_index, potential in enumerate(potentials):
            row_y = plot_top + (row_index + 0.5) * row_height
            value = potential[spec["key"]]
            dot_x = plot_left + (transform(value) - transform(axis_low)) / axis_span * plot_width
            is_lower_bound = quantity == "energy" and potential["formation_energy_is_lower_bound"]
            value_label = ("&gt;" if is_lower_bound else "") + format(value, spec["value_format"])
            row = [
                svg_line(plot_left, row_y, dot_x - MARKER_RADIUS, row_y, AXIS_WIDTH, GUIDE_DASH),
                svg_marker(dot_x, row_y, potential["kind"] == "gap"),
                svg_text(dot_x + VALUE_LABEL_GAP, row_y + 1.0, value_label, LABEL_SIZE, "start"),
            ]
            if narrow or panel_index == 0:
                row.append(
                    svg_text(
                        plot_left - LABEL_GAP, row_y + 1.0, potential["name"], LABEL_SIZE, "end"
                    )
                )
            bound = ' data-bound="lower"' if is_lower_bound else ""
            row_body = "\n      ".join(row)
            panel.append(
                f'<g data-potential="{potential["name"]}" data-kind="{potential["kind"]}"{bound}>'
                f"\n      {row_body}\n    </g>"
            )

        panel_body = "\n    ".join(panel)
        parts.append(f'<g data-panel="{quantity}">\n    {panel_body}\n  </g>')

    if narrow:
        width, height = NARROW_WIDTH, len(QUANTITIES) * panel_height
    else:
        width = PLOT_LEFT + len(QUANTITIES) * PLOT_WIDTH + PANEL_GAP + MARGIN_RIGHT
        height = panel_height
    return svg_document(width, height, parts)


if __name__ == "__main__":
    write_figure(build, __doc__.splitlines()[0])
