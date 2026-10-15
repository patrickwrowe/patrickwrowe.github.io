# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Render a cluster from an XYZ file as a ball-and-stick SVG.

The structures are Patrick's own AIRSS structure-search output for Carbon 191,
255-266 (2022). They ship as bare geometries with no renderings, so site figures
are generated here rather than lifted from the paper — which also keeps them in
the site's own palette (spec 01 section 6.2, and the `scripts/figures/`
convention in CLAUDE.md).

Depth is encoded as opacity rather than as a grey ramp, and every stroke and fill
resolves to a design token rather than a hard-coded colour. Far atoms fade towards
whatever `--plate` happens to be.

A figure of one species is monochrome: `--ink`, with `--graphite` for the one coarse step
down the ramp. A figure of more than one species (the CHO combustion series) draws each
species with its `--species-*` token, the palette defined once in constants.py's
SPECIES_COLOUR and mirrored in src/styles/tokens.css. Species are also told apart by size
and by filled versus open, so the figure still reads in greyscale. `--lustre` is never a
species: it is reserved for state and anomaly.

Every species of the palette can be drawn (carbon, hydrogen, oxygen, nitrogen, magnesium and
the rest); the single-species ramp and the sizes are DRAW_COLOUR and DRAW_RADIUS_FACTOR.

Usage:
    uv run scripts/figures/render_cluster.py <input.xyz> --output <output.svg> [--rotate X,Y,Z]
    (a negative first angle needs the equals sign: --rotate=-154.3,-14.7,-109.3)

`--hide-hydrogens` leaves every hydrogen out before the cluster is centred, bonded and drawn.
Two or more inputs are a series: panels on one grid at one scale and one rotation, which is
how two structures are drawn side by side in one view, e.g. the natural FMO arrangement and
a generated one (608 atoms each, hydrogens hidden). Each panel is centred on its own atoms'
mean; `--shared-centre` centres every panel on the mean of the FIRST input instead, so an atom
the inputs share (two geometries of one system, here the two fixed chromophores) is drawn at
the same place in every panel:
    D=scripts/figures/data/excited-states
    uv run scripts/figures/render_cluster.py $D/fmo-natural.xyz $D/fmo-generated-00300.xyz \
        --labels "natural,generated" --hide-hydrogens --radius-scale 2.0 --shared-centre \
        --rotate=39.92,21.6,162.07 \
        --output src/content/work/figures/excited-states/fmo-natural-vs-generated.svg
    and again with `--columns 1 --narrow` (the phone variant, the two panels stacked) and
        --output src/content/work/figures/excited-states/fmo-natural-vs-generated-narrow.svg
`--together` instead draws all the inputs as ONE cluster, each atom where its file puts it, so
two molecules superposed in a common frame keep their relative position: the pair of FMO
bacteriochlorophylls 1 and 2, seen face-on with their closest contact (an O...H, 2.26 A) side-on,
and `--mark-closest` joining those two atoms with a dotted ink line, edge to edge, so the contact
a caption names can be found among 152 atoms:
    uv run scripts/figures/render_cluster.py $D/chromophore-1.xyz $D/chromophore-2.xyz \
        --together --mark-closest --radius-scale 2.2 --rotate=-137.41,-4.66,-122.33 \
        --output src/content/work/figures/excited-states/bchl-pair-1-2.svg

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
from constants import COVALENT_RADIUS_ANGSTROM, SPECIES_COLOUR
from figure_style import NARROW_CHART_EM, species_colour

# Deliberately far below a physical carbon radius. These cages are hollow and
# nested front-to-back, so anything approaching space-filling collapses into a
# dark disc; the structure only reads as a cage in near-wireframe.
ATOM_RADIUS = 0.16
BOND_WIDTH = 0.11

# In a figure of one species, species is carried by three cues at once: size, fill, and a
# step down the ink-to-graphite ramp. No hue is involved. In a figure of more than one,
# the species' own colour (see _species_colour) replaces the ramp, and size and fill stay
# as the cues that survive greyscale. The palette's one accent, --lustre, is reserved for
# state and anomaly (CLAUDE.md), and element identity is neither.
#
# Size alone was not enough: carbon and hydrogen came out as two black discs a
# little apart in radius, which is unreadable in a crowded frame. So hydrogen also
# steps to --graphite and shrinks towards its true covalent ratio, and oxygen
# stays at ink but draws open. Dark disc, dark ring, small pale disc.
#
# The ramp is three coarse, well-separated steps rather than a gradient, because
# depth is drawn as opacity and a fine lightness ramp would be read as distance.
#
# Every species of the palette has an entry. The ramp has only two steps, so every species but
# hydrogen is ink; a species other than carbon, hydrogen and oxygen is told apart by its size
# alone until a figure has a second species, when its `--species-*` colour does it. Carbon,
# hydrogen and oxygen are sized by eye (hydrogen above its covalent ratio of 0.41, oxygen well
# above 0.87 so its open ring is legible); every other species is its covalent radius over
# carbon's (constants.COVALENT_RADIUS_ANGSTROM), which draws magnesium nearly twice carbon.
GROUP_COLOUR = "var(--ink)"
DRAW_COLOUR = dict.fromkeys(SPECIES_COLOUR, GROUP_COLOUR) | {"H": "var(--graphite)"}
DRAW_RADIUS_FACTOR = {
    symbol: COVALENT_RADIUS_ANGSTROM[symbol] / COVALENT_RADIUS_ANGSTROM["C"]
    for symbol in SPECIES_COLOUR
} | {"H": 0.48, "O": 1.25}
HOLLOW_SPECIES = frozenset({"O"})
# The marked closest contact (`--mark-closest`): round dots this fraction of a bond wide, centre
# to centre this many dot widths apart, so it reads as a non-bonded contact and not a bond.
CONTACT_WIDTH_FRACTION = 0.7
CONTACT_DOT_SPACING = 2.2


def _species_colour(species: str, multi_species: bool) -> str:
    """The CSS paint for one species: its palette token in a multi-species figure, else the ramp.

    Args:
        species: Element symbol.
        multi_species: Whether the figure draws more than one species.

    Returns:
        `var(--species-<symbol>)` when `multi_species`, otherwise DRAW_COLOUR's `var(--ink)` or
        `var(--graphite)`.

    Raises:
        ValueError: From figure_style.species_colour, if `multi_species` and the species has no
            entry in SPECIES_COLOUR, because its `--species-*` token would not exist and the
            browser would paint it black without a warning.
    """
    return species_colour(species) if multi_species else DRAW_COLOUR[species]


def _colour_attr(species: str, attribute: str, multi_species: bool) -> str:
    """`fill=`/`stroke=` for one species, omitted when it matches the group default.

    Keeping the attribute off carbon in a single-species figure means a carbon-only figure
    emits exactly the markup it did before any of this existed, so the published cage
    figures are untouched by adding elements. A multi-species figure names every species,
    carbon included, because carbon there is `--species-c`, not the group's `--ink`.
    """
    colour = _species_colour(species, multi_species)
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


def _is_multi_species(species: list[str]) -> bool:
    """Whether a figure draws more than one species, which switches it to the palette.

    A species with no palette entry is refused when it is drawn (`_species_colour`).

    Args:
        species: Element symbol of every atom the figure draws.

    Returns:
        True for two or more distinct species.
    """
    distinct_species = set(species)
    return len(distinct_species) > 1


def _draw(
    rotated: np.ndarray,
    species: list[str],
    multi_species: bool,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    radius_scale: float = 1.0,
) -> str:
    """Depth-sorted SVG elements for one already-rotated, already-centred cluster.

    Bonds are found on the rotated coordinates; rotation is rigid, so distances
    are unchanged and this is equivalent to finding them beforehand.

    Args:
        rotated: Rotated, centred positions, shape (n_atoms, 3), in angstrom.
        species: Element symbol per atom.
        multi_species: Whether the figure this belongs to draws more than one species, so
            every species takes its palette colour; a series decides this once for all its
            panels so one species is one colour throughout.
        offset_x: Shift of the cluster's centre in the figure, in angstrom.
        offset_y: Shift of the cluster's centre in the figure, in angstrom.
        radius_scale: Factor on the atom and bond sizes.
    """
    xs, ys, zs = rotated[:, 0] + offset_x, rotated[:, 1] - offset_y, rotated[:, 2]
    near, far = zs.max(), zs.min()

    # Draw everything back to front in one pass. A bond takes the depth of its
    # nearer atom so it never floats in front of the atom it joins.
    drawables: list[tuple[float, str]] = []

    for i, j in boxprep.find_bonds(rotated, species):
        opacity = depth_opacity(max(zs[i], zs[j]), near, far)
        width = BOND_WIDTH * radius_scale
        if _species_colour(species[i], multi_species) == _species_colour(species[j], multi_species):
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
                    f'fill="none"{_colour_attr(element, "stroke", multi_species)} '
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
                f"{_colour_attr(species[i], 'stroke', multi_species)} "
                f'stroke-width="{ring_width:.3f}" opacity="{opacity:.3f}"/>'
            )
        else:
            # stroke="none" is load-bearing: circles otherwise inherit the group
            # stroke at the SVG default width of 1 user unit, which is
            # 1 Angstrom here, and every atom renders as a ring.
            shape = (
                f'<circle cx="{xs[i]:.3f}" cy="{-ys[i]:.3f}" '
                f'r="{radius:.3f}" stroke="none"'
                f"{_colour_attr(species[i], 'fill', multi_species)} "
                f'opacity="{opacity:.3f}"/>'
            )
        drawables.append((zs[i] + 1e-6, shape))  # ties favour the atom, not the bond

    drawables.sort(key=lambda item: item[0])
    return "\n    ".join(element for _, element in drawables)


def closest_pair(first: boxprep.Structure, second: boxprep.Structure) -> tuple[int, int]:
    """The closest pair of atoms, one from each structure, as `render`'s `contact`.

    Args:
        first: The first structure, in angstrom.
        second: The second structure, in the same frame.

    Returns:
        The pair's indices in the two structures drawn together: the first's atom as it is,
        the second's offset by `len(first)`.
    """
    separations_angstrom = np.linalg.norm(
        first.positions_angstrom[:, None, :] - second.positions_angstrom[None, :, :], axis=-1
    )
    first_atom, second_atom = np.unravel_index(
        separations_angstrom.argmin(), separations_angstrom.shape
    )
    return int(first_atom), len(first) + int(second_atom)


def render(
    structure: boxprep.Structure,
    rotate_degrees: tuple[float, float, float],
    radius_scale: float = 1.0,
    slab_angstrom: float | None = None,
    contact: tuple[int, int] | None = None,
) -> str:
    """One cluster as a ball-and-stick SVG.

    Args:
        structure: The atoms, in angstrom.
        rotate_degrees: Extrinsic X, Y, Z rotation of the view, in degrees.
        radius_scale: Factor on the atom and bond sizes.
        slab_angstrom: Draw only a slab this thick about the median depth, or None for all.
        contact: Indices of two atoms to join with a dotted line edge to edge (see
            `closest_pair`), or None.

    Returns:
        The SVG document.

    Raises:
        ValueError: If `contact` and `slab_angstrom` are both given, since the slab can cut
            the contact's atoms away; or if the two atoms of `contact` are, in this view, no
            further apart than their drawn radii (a contact along the view direction), so the
            line would have no length or run backwards over both atoms.
    """
    if contact is not None and slab_angstrom is not None:
        raise ValueError("a marked contact cannot be drawn in a slab, which may cut its atoms")
    centred = structure.positions_angstrom - structure.positions_angstrom.mean(axis=0)
    rotated = centred @ rotation_matrix(rotate_degrees).T
    species = structure.species
    contact_markup = ""
    if contact is not None:
        # A dotted ink line between the two atoms' drawn edges, on top of everything: the
        # marked contact is a fact the caption states, so it is never hidden behind an atom.
        first, second = contact
        start, end = rotated[first, :2], rotated[second, :2]
        projected_gap_angstrom = np.linalg.norm(end - start)
        drawn_radii_angstrom = (
            ATOM_RADIUS
            * radius_scale
            * (DRAW_RADIUS_FACTOR[species[first]] + DRAW_RADIUS_FACTOR[species[second]])
        )
        if projected_gap_angstrom <= drawn_radii_angstrom:
            raise ValueError(
                f"the marked contact is {projected_gap_angstrom:.2f} A apart in this view, "
                f"inside its atoms' drawn radii ({drawn_radii_angstrom:.2f} A); "
                "rotate it side-on"
            )
        unit = (end - start) / projected_gap_angstrom
        start = start + unit * ATOM_RADIUS * DRAW_RADIUS_FACTOR[species[first]] * radius_scale
        end = end - unit * ATOM_RADIUS * DRAW_RADIUS_FACTOR[species[second]] * radius_scale
        width = CONTACT_WIDTH_FRACTION * BOND_WIDTH * radius_scale
        separation = np.linalg.norm(rotated[first] - rotated[second])
        contact_markup = (
            f'\n    <line data-contact-angstrom="{separation:.3f}" '
            f'x1="{start[0]:.3f}" y1="{-start[1]:.3f}" x2="{end[0]:.3f}" y2="{-end[1]:.3f}" '
            f'fill="none" stroke="var(--ink)" stroke-width="{width:.3f}" '
            f'stroke-dasharray="0 {CONTACT_DOT_SPACING * width:.3f}"/>'
        )
    if slab_angstrom is not None:
        # A periodic cell of a few thousand atoms projects to a solid black square: the
        # front face hides everything and there is no structure to read. Cutting a slab
        # perpendicular to the view gives the cross-section an electron micrograph of the
        # same material would show. It is also what keeps the bond search affordable: an
        # order of magnitude fewer atoms keeps the O(n^2) distance matrix small.
        keep = boxprep.slab_mask(rotated[:, 2], float(np.median(rotated[:, 2])), slab_angstrom)
        rotated = rotated[keep]
        species = [s for s, k in zip(species, keep, strict=True) if k]

    margin = ATOM_RADIUS * radius_scale * 2.0
    min_x, max_x = rotated[:, 0].min() - margin, rotated[:, 0].max() + margin
    min_y, max_y = rotated[:, 1].min() - margin, rotated[:, 1].max() + margin

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{min_x:.3f} {-max_y:.3f} {max_x - min_x:.3f} {max_y - min_y:.3f}" '
        f'role="img">\n'
        f'  <g fill="var(--ink)" stroke="var(--ink)" stroke-linecap="round">\n'
        f"    {_draw(rotated, species, _is_multi_species(species), radius_scale=radius_scale)}"
        f"{contact_markup}\n"
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
        lambda m: (
            f'<tspan font-size="{size:.3f}" dy="{drop:.3f}">{m.group(1)}</tspan>'
            f'<tspan dy="{-drop:.3f}"></tspan>'
        ),
        label,
    )


def render_series(
    clusters: list[tuple[str, boxprep.Structure]],
    rotate_degrees: tuple[float, float, float],
    radius_scale: float = 1.0,
    columns: int | None = None,
    slab_angstrom: float | None = None,
    label_em: float | None = None,
    shared_centre: bool = False,
) -> str:
    """Lay several clusters out on a grid at one shared scale.

    `label_em`, when given, sizes the labels so the viewBox is that many labels wide (the
    page's `--chart-em`); otherwise a label is LABEL_SIZE_FRACTION of the drawing's width.

    Each cluster is centred on its own mean, unless `shared_centre`: then every cluster is
    centred on the first one's mean, so an atom that two clusters share sits at the same
    place in both panels (a panel's drawn extent is then no longer symmetric about its centre,
    which the panel's half width, the largest distance from it, still bounds).

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
    shared_origin = clusters[0][1].positions_angstrom.mean(axis=0)
    for _label, structure in clusters:
        origin = shared_origin if shared_centre else structure.positions_angstrom.mean(axis=0)
        rotated = (structure.positions_angstrom - origin) @ rot.T
        species = structure.species
        if slab_angstrom is not None:
            # See render(): a slab perpendicular to the view, centred on its own median
            # depth, keeps the cross-section legible and the bond search affordable.
            keep = boxprep.slab_mask(rotated[:, 2], float(np.median(rotated[:, 2])), slab_angstrom)
            rotated = rotated[keep]
            species = [s for s, k in zip(species, keep, strict=True) if k]
        rotated_all.append(rotated)
        species_all.append(species)
        half_widths.append(float(np.abs(rotated[:, 0]).max()) + margin)
        half_heights.append(float(np.abs(rotated[:, 1]).max()) + margin)

    # One decision for the whole figure, so a species is one colour in every panel.
    multi_species = _is_multi_species([symbol for panel in species_all for symbol in panel])

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
    for index, (label, _structure) in enumerate(clusters):
        row, column = divmod(index, n_columns)
        offset_x = column_centre_x[column]
        offset_y = row_centre_y[row]
        placed.append(
            _draw(
                rotated_all[index],
                species_all[index],
                multi_species,
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
        f"    {''.join(placed)}\n"
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
        "--hide-hydrogens",
        action="store_true",
        help="Leave every hydrogen out before the cluster is centred, bonded and drawn.",
    )
    parser.add_argument(
        "--shared-centre",
        action="store_true",
        help="Series mode: centre every panel on the mean of the first input, not on its own, so "
        "an atom the inputs share is drawn at the same place in every panel.",
    )
    parser.add_argument(
        "--together",
        action="store_true",
        help="Draw all the inputs as one cluster, each atom where its file puts it, instead of "
        "as panels of a series: two molecules in the frame they were superposed in.",
    )
    parser.add_argument(
        "--mark-closest",
        action="store_true",
        help="With --together and two inputs: join the closest pair of atoms, one from each "
        "input, with a dotted ink line between their drawn edges.",
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

    structures = []
    for path in args.inputs:
        structure = boxprep.read_xyz(path, args.frame)
        if args.hide_hydrogens:
            kept = np.array(structure.species) != "H"
            structure = boxprep.Structure(
                [symbol for symbol, keep in zip(structure.species, kept, strict=True) if keep],
                structure.positions_angstrom[kept],
            )
            if not len(structure):
                parser.error(f"{path}: nothing is left to draw once the hydrogens are hidden")
        structures.append(structure)
    contact = None
    if args.mark_closest:
        if not args.together or len(structures) != 2:
            parser.error("--mark-closest needs --together and exactly two inputs")
        if args.slab is not None:
            parser.error(
                "--mark-closest cannot be drawn with --slab, which can cut the contact away"
            )
        contact = closest_pair(*structures)
    if args.together:
        structures = [
            boxprep.Structure(
                [symbol for structure in structures for symbol in structure.species],
                np.vstack([structure.positions_angstrom for structure in structures]),
            )
        ]

    if len(structures) == 1 and not args.labels:
        svg = render(structures[0], rotate, args.radius_scale, args.slab, contact)  # type: ignore[arg-type]
        names = " + ".join(path.name for path in args.inputs)
        print(f"{names}: {len(structures[0])} atoms, {describe(structures[0])}")
    else:
        labels = args.labels.split(",") if args.labels else [p.stem for p in args.inputs]
        if len(labels) != len(structures):
            parser.error(f"{len(labels)} labels for {len(structures)} drawings")
        clusters = list(zip(labels, structures, strict=True))
        svg = render_series(  # type: ignore[arg-type]
            clusters,
            rotate,
            args.radius_scale,
            args.columns,
            args.slab,
            NARROW_CHART_EM if args.narrow else None,
            args.shared_centre,
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
    water_species = ["O", "H", "H"]
    assert _is_multi_species(water_species)
    markup = _draw(water, water_species, True)
    assert markup.count("<circle") == 3, markup
    # Oxygen is an open ring in its own colour; hydrogen is a filled disc in its own.
    assert len(re.findall(r'<circle[^>]*fill="none" stroke="var\(--species-o\)"', markup)) == 1
    assert markup.count('stroke="none" fill="var(--species-h)"') == 2, markup
    # Both O-H bonds are split, so four segments, one half of each in each species' colour.
    assert markup.count("<line") == 4, markup
    assert markup.count('stroke="var(--species-h)"') == 2, markup
    assert markup.count('stroke="var(--species-o)"') == 3, markup  # two bond halves and the ring
    assert "--ink" not in markup and "--graphite" not in markup, markup

    # Carbon on its own must emit no colour attributes at all, which is what keeps
    # the carbon-only figures byte-identical to the ones already published.
    assert not _is_multi_species(["C", "C"])
    carbon_only = _draw(np.array([[0.0, 0, 0], [1.4, 0, 0]]), ["C", "C"], False)
    assert "var(--" not in carbon_only, carbon_only
    assert carbon_only.count("<line") == 1, carbon_only  # same-colour bonds stay whole

    # A C-H bond is split in two, each half its own species' colour; a C-C bond is not.
    pair = np.array([[0.0, 0, 0], [1.1, 0, 0]])
    carbon_hydrogen = _draw(pair, ["C", "H"], True)
    assert carbon_hydrogen.count("<line") == 2
    assert 'stroke="var(--species-c)"' in carbon_hydrogen
    assert 'stroke="var(--species-h)"' in carbon_hydrogen
    assert _draw(np.array([[0.0, 0, 0], [1.4, 0, 0]]), ["C", "C"], False).count("<line") == 1

    # A species with no palette entry is refused by name rather than painted black.
    try:
        _species_colour("Xe", True)
    except ValueError as error:
        assert "Xe" in str(error), error
    else:
        raise AssertionError("an unpainted species was accepted")

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
