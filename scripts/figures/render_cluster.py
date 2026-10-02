# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Render a carbon cluster from an XYZ file as a monochrome ball-and-stick SVG.

The structures are Patrick's own AIRSS structure-search output for Carbon 191,
255-266 (2022). They ship as bare geometries with no renderings, so site figures
are generated here rather than lifted from the paper — which also keeps them in
the site's own palette (spec 01 section 6.2, and the `scripts/figures/`
convention in CLAUDE.md).

Depth is encoded as opacity rather than as a grey ramp, and every stroke and fill
resolves to a design token (`--ink` or `--graphite`) rather than a hard-coded
palette. Far atoms fade towards whatever `--plate` happens to be.

Elements are told apart by size, by filled versus open, and by one coarse step
down the ink ramp. There is no hue anywhere: the palette's only accent is
reserved for state and anomaly.

Usage:
    uv run scripts/figures/render_cluster.py <input.xyz> <output.svg> [--rotate X,Y,Z]

The CHO-GAP combustion series (Fig. 14 of the carbon article), and with `--narrow` its
phone-width variant, whose labels are sized so the viewBox is figure_style.NARROW_CHART_EM
labels wide:
    D=scripts/figures/data/cho-gap
    uv run scripts/figures/render_cluster.py $D/combustion-0ps.xyz $D/combustion-10ps.xyz \
        $D/combustion-20ps.xyz $D/combustion-50ps.xyz --labels "0 ps,10 ps,20 ps,50 ps" \
        --columns 2 --radius-scale 2.2 \
        --output src/content/work/figures/cho-gap/combustion-series.svg
    and again with --narrow and
        --output src/content/work/figures/cho-gap/combustion-series-narrow.svg
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path

import boxprep
import numpy as np
from figure_style import NARROW_CHART_EM

# Deliberately far below a physical carbon radius. These cages are hollow and
# nested front-to-back, so anything approaching space-filling collapses into a
# dark disc; the structure only reads as a cage in near-wireframe.
ATOM_RADIUS = 0.16
BOND_WIDTH = 0.11

# Species is carried by three cues at once: size, fill, and a step down the
# ink-to-graphite ramp. No hue is involved. The palette's one accent, --lustre, is
# reserved for state and anomaly (CLAUDE.md), and element identity is neither.
#
# Size alone was not enough: carbon and hydrogen came out as two black discs a
# little apart in radius, which is unreadable in a crowded frame. So hydrogen also
# steps to --graphite and shrinks towards its true covalent ratio, and oxygen
# stays at ink but draws open. Dark disc, dark ring, small pale disc.
#
# The ramp is three coarse, well-separated steps rather than a gradient, because
# depth is drawn as opacity and a fine lightness ramp would be read as distance.
GROUP_COLOUR = "var(--ink)"
DRAW_COLOUR = {"C": "var(--ink)", "H": "var(--graphite)", "O": "var(--ink)"}
DRAW_RADIUS_FACTOR = {"C": 1.0, "H": 0.48, "O": 1.25}
HOLLOW_SPECIES = frozenset({"O"})


def _colour_attr(species: str, attribute: str) -> str:
    """`fill=`/`stroke=` for one species, omitted when it matches the group default.

    Keeping the attribute off carbon means a carbon-only figure emits exactly the
    markup it did before any of this existed, so the published cage figures are
    untouched by adding elements.
    """
    colour = DRAW_COLOUR[species]
    return "" if colour == GROUP_COLOUR else f' {attribute}="{colour}"'

# Opacity at the back and front of the cluster. Never reaches 1.0 at the front:
# a solid black silhouette loses the ball-and-stick reading at small sizes.
OPACITY_FAR = 0.07
OPACITY_NEAR = 0.95


def rotation_matrix(degrees_xyz: tuple[float, float, float]) -> np.ndarray:
    """Extrinsic X-then-Y-then-Z rotation."""
    rx, ry, rz = (math.radians(d) for d in degrees_xyz)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    mat_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    mat_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    mat_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return mat_z @ mat_y @ mat_x


def depth_opacity(depth: float, near: float, far: float) -> float:
    if math.isclose(near, far):
        return OPACITY_NEAR
    t = (depth - far) / (near - far)
    return OPACITY_FAR + t * (OPACITY_NEAR - OPACITY_FAR)


def _draw(
    rotated: np.ndarray,
    species: list[str],
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    radius_scale: float = 1.0,
) -> str:
    """Depth-sorted SVG elements for one already-rotated, already-centred cluster.

    Bonds are found on the rotated coordinates; rotation is rigid, so distances
    are unchanged and this is equivalent to finding them beforehand.
    """
    xs, ys, zs = rotated[:, 0] + offset_x, rotated[:, 1] - offset_y, rotated[:, 2]
    near, far = zs.max(), zs.min()

    # Draw everything back to front in one pass. A bond takes the depth of its
    # nearer atom so it never floats in front of the atom it joins.
    drawables: list[tuple[float, str]] = []

    for i, j in boxprep.find_bonds(rotated, species):
        opacity = depth_opacity(max(zs[i], zs[j]), near, far)
        width = BOND_WIDTH * radius_scale
        if DRAW_COLOUR[species[i]] == DRAW_COLOUR[species[j]]:
            segments = [(xs[i], ys[i], xs[j], ys[j], species[i])]
        else:
            # Split at the midpoint, each half taking its own atom's colour. A
            # heteronuclear bond drawn entirely in ink runs into a pale hydrogen
            # and swallows it; this is the usual ball-and-stick answer.
            mid_x, mid_y = (xs[i] + xs[j]) / 2, (ys[i] + ys[j]) / 2
            segments = [
                (xs[i], ys[i], mid_x, mid_y, species[i]),
                (mid_x, mid_y, xs[j], ys[j], species[j]),
            ]
        for x1, y1, x2, y2, element in segments:
            drawables.append(
                (
                    max(zs[i], zs[j]),
                    f'<line x1="{x1:.3f}" y1="{-y1:.3f}" '
                    f'x2="{x2:.3f}" y2="{-y2:.3f}" '
                    f'fill="none"{_colour_attr(element, "stroke")} '
                    f'stroke-width="{width:.3f}" '
                    f'opacity="{opacity:.3f}"/>',
                )
            )

    for i in range(len(rotated)):
        opacity = depth_opacity(zs[i], near, far)
        radius = ATOM_RADIUS * DRAW_RADIUS_FACTOR[species[i]] * radius_scale
        if species[i] in HOLLOW_SPECIES:
            # An open ring. The ring is inset by half its stroke width to keep the
            # drawn extent equal to `radius`, which is what the viewBox margin
            # assumes.
            ring_width = BOND_WIDTH * radius_scale
            shape = (
                f'<circle cx="{xs[i]:.3f}" cy="{-ys[i]:.3f}" '
                f'r="{radius - ring_width / 2:.3f}" fill="none"'
                f'{_colour_attr(species[i], "stroke")} '
                f'stroke-width="{ring_width:.3f}" opacity="{opacity:.3f}"/>'
            )
        else:
            # stroke="none" is load-bearing: circles otherwise inherit the group
            # stroke at the SVG default width of 1 user unit, which is
            # 1 Angstrom here, and every atom renders as a ring.
            shape = (
                f'<circle cx="{xs[i]:.3f}" cy="{-ys[i]:.3f}" '
                f'r="{radius:.3f}" stroke="none"'
                f'{_colour_attr(species[i], "fill")} '
                f'opacity="{opacity:.3f}"/>'
            )
        drawables.append((zs[i] + 1e-6, shape))  # ties favour the atom, not the bond

    drawables.sort(key=lambda item: item[0])
    return "\n    ".join(element for _, element in drawables)


def render(
    structure: boxprep.Structure,
    rotate_degrees: tuple[float, float, float],
    radius_scale: float = 1.0,
    slab_angstrom: float | None = None,
) -> str:
    centred = structure.positions_angstrom - structure.positions_angstrom.mean(axis=0)
    rotated = centred @ rotation_matrix(rotate_degrees).T
    species = structure.species
    if slab_angstrom is not None:
        # A periodic cell of a few thousand atoms projects to a solid black square: the
        # front face hides everything and there is no structure to read. Cutting a slab
        # perpendicular to the view gives the cross-section an electron micrograph of the
        # same material would show. It is also what keeps the bond search affordable: an
        # order of magnitude fewer atoms keeps the O(n^2) distance matrix small.
        keep = boxprep.slab_mask(rotated[:, 2], float(np.median(rotated[:, 2])), slab_angstrom)
        rotated = rotated[keep]
        species = [s for s, k in zip(species, keep) if k]

    margin = ATOM_RADIUS * radius_scale * 2.0
    min_x, max_x = rotated[:, 0].min() - margin, rotated[:, 0].max() + margin
    min_y, max_y = rotated[:, 1].min() - margin, rotated[:, 1].max() + margin

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{min_x:.3f} {-max_y:.3f} {max_x - min_x:.3f} {max_y - min_y:.3f}" '
        f'role="img">\n'
        f'  <g fill="var(--ink)" stroke="var(--ink)" stroke-linecap="round">\n'
        f"    {_draw(rotated, species, radius_scale=radius_scale)}\n"
        f"  </g>\n"
        f"</svg>\n"
    )


SERIES_GAP_ANGSTROM = 3.0

# Label size as a fraction of the figure's total width, not a fixed number of
# Angstrom. A 30 A box of gas and a 76 A row of cages are drawn at the same
# width on the page, so a fixed Angstrom size renders one of them three times
# larger than the other. The constant is set to reproduce the 1.6 A that the
# cage series was drawn with.
LABEL_SIZE_FRACTION = 0.021
LABEL_BAND_RATIO = 1.625  # band height per unit of label size

# Subscript digits, so a label reads C60 the way the prose around it does. Unicode
# subscripts would be simpler but render unevenly in a monospaced face, and half the
# glyphs are missing from most of them.
#
# Only digits directly after a letter are subscripted. Without the lookbehind a
# time-series label like "10 ps" is read as a formula and drops its number to the
# baseline, which is wrong and not obviously wrong until it is on the page.
_SUBSCRIPT_RE = re.compile(r"(?<=[A-Za-z])(\d+)")


# Subscript size and drop, as fractions of the label size they sit inside.
SUBSCRIPT_SIZE_RATIO = 0.65625
SUBSCRIPT_DROP_RATIO = 0.2625


def _formula_markup(label: str, label_size: float = 1.6) -> str:
    """Wrap digit runs in a subscript tspan: C60 -> C<sub>60</sub>."""
    size = label_size * SUBSCRIPT_SIZE_RATIO
    drop = label_size * SUBSCRIPT_DROP_RATIO
    return _SUBSCRIPT_RE.sub(
        lambda m: f'<tspan font-size="{size:.3f}" dy="{drop:.3f}">{m.group(1)}</tspan>'
        f'<tspan dy="{-drop:.3f}"></tspan>',
        label,
    )


def render_series(
    clusters: list[tuple[str, boxprep.Structure]],
    rotate_degrees: tuple[float, float, float],
    radius_scale: float = 1.0,
    columns: int | None = None,
    slab_angstrom: float | None = None,
    label_em: float | None = None,
) -> str:
    """Lay several clusters out on a grid at one shared scale.

    `label_em`, when given, sizes the labels so the viewBox is that many labels wide (the
    page's `--chart-em`); otherwise a label is LABEL_SIZE_FRACTION of the drawing's width.

    The viewBox is in Angstrom throughout, so the clusters are drawn true to
    relative size — which is the point of the figure. Scaling each to fit its
    own cell would throw away the only quantity being compared.

    `columns` defaults to one row. Wrapping matters on a phone: four panels
    across a 390 px column are 90 px each, which is below the size at which a
    molecule reads as anything.

    Column widths and row heights are taken from the widest and tallest member
    of each, so a single row reduces exactly to per-cluster spacing and the
    one-row figures are unaffected by the grid code.
    """
    rot = rotation_matrix(rotate_degrees)
    margin = ATOM_RADIUS * radius_scale
    n_columns = columns or len(clusters)

    rotated_all: list[np.ndarray] = []
    species_all: list[list[str]] = []
    half_widths: list[float] = []
    half_heights: list[float] = []
    for _label, structure in clusters:
        centred = structure.positions_angstrom - structure.positions_angstrom.mean(axis=0)
        rotated = centred @ rot.T
        species = structure.species
        if slab_angstrom is not None:
            # See render(): a slab perpendicular to the view, centred on its own median
            # depth, keeps the cross-section legible and the bond search affordable.
            keep = boxprep.slab_mask(rotated[:, 2], float(np.median(rotated[:, 2])), slab_angstrom)
            rotated = rotated[keep]
            species = [s for s, k in zip(species, keep) if k]
        rotated_all.append(rotated)
        species_all.append(species)
        half_widths.append(float(np.abs(rotated[:, 0]).max()) + margin)
        half_heights.append(float(np.abs(rotated[:, 1]).max()) + margin)

    n_rows = -(-len(clusters) // n_columns)
    column_half_width = [
        max(half_widths[i] for i in range(len(clusters)) if i % n_columns == c)
        for c in range(min(n_columns, len(clusters)))
    ]
    row_half_height = [
        max(half_heights[i] for i in range(len(clusters)) if i // n_columns == r)
        for r in range(n_rows)
    ]

    column_centre_x: list[float] = []
    cursor = 0.0
    for half_width in column_half_width:
        column_centre_x.append(cursor + half_width)
        cursor += 2 * half_width + SERIES_GAP_ANGSTROM
    total_width = cursor - SERIES_GAP_ANGSTROM

    view_width = total_width + 2 * ATOM_RADIUS
    label_size = view_width / label_em if label_em else total_width * LABEL_SIZE_FRACTION
    label_band = label_size * LABEL_BAND_RATIO

    # The first row is centred on y = 0, which is where a single-row figure has
    # always drawn, so wrapping stays a pure addition.
    row_centre_y: list[float] = []
    cursor = -row_half_height[0]
    for half_height in row_half_height:
        row_centre_y.append(cursor + half_height)
        cursor += 2 * half_height + label_band
    total_height = cursor + row_half_height[0]

    placed: list[str] = []
    labels: list[tuple[float, float, str]] = []
    for index, (label, structure) in enumerate(clusters):
        row, column = divmod(index, n_columns)
        offset_x = column_centre_x[column]
        offset_y = row_centre_y[row]
        placed.append(
            _draw(
                rotated_all[index],
                species_all[index],
                offset_x=offset_x,
                offset_y=offset_y,
                radius_scale=radius_scale,
            )
        )
        labels.append((offset_x, offset_y + row_half_height[row] + label_band * 0.75, label))

    label_markup = "\n    ".join(
        f'<text x="{x:.3f}" y="{y:.3f}" text-anchor="middle" '
        f'font-size="{label_size:.3f}" fill="var(--graphite)" stroke="none" '
        f'font-family="var(--mono)">{_formula_markup(text, label_size)}</text>'
        for x, y, text in labels
    )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{-ATOM_RADIUS:.3f} {-row_half_height[0]:.3f} '
        f'{view_width:.3f} {total_height:.3f}" role="img" '
        # The viewBox width in labels, the page wrapper's --chart-em.
        f'data-chart-em="{view_width / label_size:.1f}">\n'
        f'  <g fill="var(--ink)" stroke="var(--ink)" stroke-linecap="round">\n'
        f'    {"".join(placed)}\n'
        f"    {label_markup}\n"
        f"  </g>\n"
        f"</svg>\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rotate",
        default="0,0,0",
        help="Extrinsic X,Y,Z rotation in degrees, e.g. --rotate 15,25,0",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Comma-separated labels, one per input. Series mode only.",
    )
    parser.add_argument(
        "--radius-scale",
        type=float,
        default=1.0,
        help="Scale atom and bond thickness. The default is tuned for the hollow "
        "720-atom cages; a hundred-atom molecule wants roughly 2.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=-1,
        help="Frame index for multi-frame XYZ trajectories. Default is the last.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=None,
        help="Wrap a series onto this many columns. Default is a single row.",
    )
    parser.add_argument(
        "--narrow",
        action="store_true",
        help="Series mode: size the labels so the viewBox is figure_style.NARROW_CHART_EM "
        "labels wide, the phone-width variant of a chart.",
    )
    parser.add_argument(
        "--slab",
        type=float,
        default=None,
        help="Draw only a slab this many Angstrom thick, centred on the view axis. "
        "Required for periodic cells: the whole cell projects to a black square.",
    )
    args = parser.parse_args()

    rotate = tuple(float(v) for v in args.rotate.split(","))
    if len(rotate) != 3:
        parser.error("--rotate needs exactly three comma-separated degrees")

    def describe(structure: boxprep.Structure) -> str:
        counts = Counter(structure.species)
        return " ".join(f"{s}{counts[s]}" for s in sorted(counts))

    if len(args.inputs) == 1 and not args.labels:
        structure = boxprep.read_xyz(args.inputs[0], args.frame)
        svg = render(structure, rotate, args.radius_scale, args.slab)  # type: ignore[arg-type]
        print(f"{args.inputs[0].name}: {len(structure)} atoms, {describe(structure)}")
    else:
        labels = args.labels.split(",") if args.labels else [p.stem for p in args.inputs]
        if len(labels) != len(args.inputs):
            parser.error(f"{len(labels)} labels for {len(args.inputs)} inputs")
        clusters = [
            (label, boxprep.read_xyz(path, args.frame))
            for label, path in zip(labels, args.inputs, strict=True)
        ]
        svg = render_series(  # type: ignore[arg-type]
            clusters,
            rotate,
            args.radius_scale,
            args.columns,
            args.slab,
            NARROW_CHART_EM if args.narrow else None,
        )
        for label, structure in clusters:
            print(f"  {label}: {len(structure)} atoms, {describe(structure)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg)
    print(f"-> {args.output}")


def _self_check() -> None:
    """Bond detection and depth ordering, on a geometry with known answers."""
    # Four carbons in a 1.4 A square: four bonds around the edge, and the
    # 1.98 A diagonals correctly excluded.
    square = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [1.4, 1.4, 0.0], [0.0, 1.4, 0.0]])
    assert len(boxprep.find_bonds(square)) == 4, boxprep.find_bonds(square)

    # Cutoff boundary: 1.7 A bonded, 1.9 A not. This is the carbon-only cutoff
    # the cage figures were drawn with, and the covalent-radius rule must not
    # have moved it.
    assert len(boxprep.find_bonds(np.array([[0.0, 0, 0], [1.7, 0, 0]]))) == 1
    assert len(boxprep.find_bonds(np.array([[0.0, 0, 0], [1.9, 0, 0]]))) == 0

    # Per-species cutoffs. A 1.1 A C-H bond is real; the same separation between
    # two carbons would be far too short, but the flat 1.8 A rule accepted it and
    # a flat rule tight enough for C-H would have broken every C-C bond.
    pair = np.array([[0.0, 0, 0], [1.1, 0, 0]])
    assert len(boxprep.find_bonds(pair, ["C", "H"])) == 1
    # ... and 1.6 A is a C-O bond but not an O-H one.
    pair = np.array([[0.0, 0, 0], [1.6, 0, 0]])
    assert len(boxprep.find_bonds(pair, ["C", "O"])) == 1
    assert len(boxprep.find_bonds(pair, ["O", "H"])) == 0

    # Two hydrogens at a typical non-bonded contact must not be joined: the
    # summed radii are small enough that the flat carbon cutoff would have.
    assert len(boxprep.find_bonds(np.array([[0.0, 0, 0], [1.5, 0, 0]]), ["H", "H"])) == 0

    # Every element must be distinguishable from every other. A regression here is
    # silent in the SVG and only visible once the figure is on the page.
    water = np.array([[0.0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    markup = _draw(water, ["O", "H", "H"])
    assert markup.count("<circle") == 3, markup
    # Oxygen is an open ring at ink; hydrogen is a filled disc at graphite.
    assert markup.count('stroke="none" opacity') == 0, markup  # no ink-filled disc here
    assert markup.count('stroke="none" fill="var(--graphite)"') == 2, markup
    assert markup.count('fill="var(--graphite)"') == 2, markup
    # Both O-H bonds are split, so four segments, the hydrogen half of each in graphite.
    assert markup.count("<line") == 4, markup
    assert markup.count('stroke="var(--graphite)"') == 2, markup

    # Carbon on its own must emit no colour attributes at all, which is what keeps
    # the carbon-only figures byte-identical to the ones already published.
    carbon_only = _draw(np.array([[0.0, 0, 0], [1.4, 0, 0]]), ["C", "C"])
    assert "var(--" not in carbon_only, carbon_only
    assert carbon_only.count("<line") == 1, carbon_only  # same-colour bonds stay whole

    # A C-H bond is split in two; a C-C bond is not.
    pair = np.array([[0.0, 0, 0], [1.1, 0, 0]])
    assert _draw(pair, ["C", "H"]).count("<line") == 2
    assert _draw(np.array([[0.0, 0, 0], [1.4, 0, 0]]), ["C", "C"]).count("<line") == 1

    # Wrapping onto a grid. Four identical single atoms on two columns must give
    # two rows: a taller, narrower figure than the same four in one row.
    one = boxprep.Structure(["C"], np.zeros((1, 3)))
    four = [(f"C{i}", one) for i in range(4)]
    row = render_series(four, (0, 0, 0))
    grid = render_series(four, (0, 0, 0), columns=2)

    def viewbox(svg: str) -> list[float]:
        return [float(v) for v in re.search(r'viewBox="([^"]+)"', svg).group(1).split()]

    row_box, grid_box = viewbox(row), viewbox(grid)
    assert grid_box[2] < row_box[2], (grid_box, row_box)  # narrower
    assert grid_box[3] > row_box[3], (grid_box, row_box)  # taller
    # Every cluster is still drawn, and every label with it.
    assert grid.count("<circle") == 4, grid.count("<circle")
    assert grid.count("<text") == 4, grid.count("<text")

    # Depth maps to the stated opacity range, near-end brightest.
    assert math.isclose(depth_opacity(5.0, 5.0, -5.0), OPACITY_NEAR)
    assert math.isclose(depth_opacity(-5.0, 5.0, -5.0), OPACITY_FAR)
    assert OPACITY_FAR < depth_opacity(0.0, 5.0, -5.0) < OPACITY_NEAR

    # A degenerate flat-in-z cluster must not divide by zero.
    assert math.isclose(depth_opacity(1.0, 1.0, 1.0), OPACITY_NEAR)

    # Digit runs become subscripts; letters are left alone. Every dy is undone by a
    # matching one, or later labels on the same baseline would creep downwards.
    assert _formula_markup("C60").startswith("C<tspan")
    assert _formula_markup("BN") == "BN"
    for markup in (_formula_markup("C60"), _formula_markup("C34O20H20")):
        assert sum(float(d) for d in re.findall(r'dy="(-?[\d.]+)"', markup)) == 0.0

    print("self-check ok")


if __name__ == "__main__":
    import sys

    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()
