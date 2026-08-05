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

Depth is encoded as opacity against a single colour rather than as a grey ramp,
so every stroke and fill resolves to `var(--ink)` and the figure inherits the
design tokens instead of hard-coding a palette. Far atoms fade towards whatever
`--plate` happens to be.

Usage:
    uv run scripts/figures/render_cluster.py <input.xyz> <output.svg> [--rotate X,Y,Z]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

# Carbon-carbon bonds. 1.8 A sits above the 1.55 A single bond and below the
# 2.4 A second-neighbour distance in every phase these searches produce.
BOND_CUTOFF_ANGSTROM = 1.8

# Deliberately far below a physical carbon radius. These cages are hollow and
# nested front-to-back, so anything approaching space-filling collapses into a
# dark disc; the structure only reads as a cage in near-wireframe.
ATOM_RADIUS = 0.16
BOND_WIDTH = 0.11

# Opacity at the back and front of the cluster. Never reaches 1.0 at the front:
# a solid black silhouette loses the ball-and-stick reading at small sizes.
OPACITY_FAR = 0.07
OPACITY_NEAR = 0.95


def read_xyz(path: Path) -> np.ndarray:
    """Return (n, 3) coordinates in Angstrom.

    The search output carries trailing bookkeeping columns (index, neighbour
    list) after the coordinates, so only fields 1-3 are read.
    """
    lines = path.read_text().splitlines()
    n_atoms = int(lines[0].split()[0])
    coords = []
    for line in lines[2 : 2 + n_atoms]:
        fields = line.split()
        coords.append([float(fields[1]), float(fields[2]), float(fields[3])])
    positions_angstrom = np.asarray(coords, dtype=float)
    if positions_angstrom.shape != (n_atoms, 3):
        raise ValueError(f"{path}: expected {n_atoms} atoms, parsed {positions_angstrom.shape[0]}")
    return positions_angstrom


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


def find_bonds(positions_angstrom: np.ndarray) -> list[tuple[int, int]]:
    """All atom pairs within the bond cutoff.

    O(n^2) on the distance matrix. The largest cluster here is 720 atoms, so
    this is well under a second and a neighbour list would be premature.
    """
    deltas = positions_angstrom[:, None, :] - positions_angstrom[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    upper = np.triu(np.ones_like(distances, dtype=bool), k=1)
    rows, cols = np.where(upper & (distances < BOND_CUTOFF_ANGSTROM))
    return list(zip(rows.tolist(), cols.tolist()))


def depth_opacity(depth: float, near: float, far: float) -> float:
    if math.isclose(near, far):
        return OPACITY_NEAR
    t = (depth - far) / (near - far)
    return OPACITY_FAR + t * (OPACITY_NEAR - OPACITY_FAR)


def _draw(rotated: np.ndarray, offset_x: float = 0.0) -> str:
    """Depth-sorted SVG elements for one already-rotated, already-centred cluster.

    Bonds are found on the rotated coordinates; rotation is rigid, so distances
    are unchanged and this is equivalent to finding them beforehand.
    """
    xs, ys, zs = rotated[:, 0] + offset_x, rotated[:, 1], rotated[:, 2]
    near, far = zs.max(), zs.min()

    # Draw everything back to front in one pass. A bond takes the depth of its
    # nearer atom so it never floats in front of the atom it joins.
    drawables: list[tuple[float, str]] = []

    for i, j in find_bonds(rotated):
        opacity = depth_opacity(max(zs[i], zs[j]), near, far)
        drawables.append(
            (
                max(zs[i], zs[j]),
                f'<line x1="{xs[i]:.3f}" y1="{-ys[i]:.3f}" '
                f'x2="{xs[j]:.3f}" y2="{-ys[j]:.3f}" '
                f'fill="none" stroke-width="{BOND_WIDTH:.3f}" opacity="{opacity:.3f}"/>',
            )
        )

    for i in range(len(rotated)):
        opacity = depth_opacity(zs[i], near, far)
        drawables.append(
            (
                zs[i] + 1e-6,  # ties resolve in favour of the atom, not the bond
                # stroke="none" is load-bearing: circles otherwise inherit the
                # group stroke at the SVG default width of 1 user unit, which is
                # 1 Angstrom here, and every atom renders as a ring.
                f'<circle cx="{xs[i]:.3f}" cy="{-ys[i]:.3f}" '
                f'r="{ATOM_RADIUS:.3f}" stroke="none" opacity="{opacity:.3f}"/>',
            )
        )

    drawables.sort(key=lambda item: item[0])
    return "\n    ".join(element for _, element in drawables)


def render(positions_angstrom: np.ndarray, rotate_degrees: tuple[float, float, float]) -> str:
    centred = positions_angstrom - positions_angstrom.mean(axis=0)
    rotated = centred @ rotation_matrix(rotate_degrees).T

    margin = ATOM_RADIUS * 2.0
    min_x, max_x = rotated[:, 0].min() - margin, rotated[:, 0].max() + margin
    min_y, max_y = rotated[:, 1].min() - margin, rotated[:, 1].max() + margin

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{min_x:.3f} {-max_y:.3f} {max_x - min_x:.3f} {max_y - min_y:.3f}" '
        f'role="img">\n'
        f'  <g fill="var(--ink)" stroke="var(--ink)" stroke-linecap="round">\n'
        f"    {_draw(rotated)}\n"
        f"  </g>\n"
        f"</svg>\n"
    )


SERIES_GAP_ANGSTROM = 3.0


def render_series(
    clusters: list[tuple[str, np.ndarray]], rotate_degrees: tuple[float, float, float]
) -> str:
    """Lay several clusters out in a row at one shared scale.

    The viewBox is in Angstrom throughout, so the clusters are drawn true to
    relative size — which is the point of the figure. Scaling each to fit its
    own cell would throw away the only quantity being compared.
    """
    rot = rotation_matrix(rotate_degrees)
    placed: list[str] = []
    cursor_x = 0.0
    max_half_height = 0.0
    labels: list[tuple[float, float, str]] = []

    for label, positions_angstrom in clusters:
        centred = positions_angstrom - positions_angstrom.mean(axis=0)
        rotated = centred @ rot.T
        half_width = float(np.abs(rotated[:, 0]).max()) + ATOM_RADIUS
        half_height = float(np.abs(rotated[:, 1]).max()) + ATOM_RADIUS
        max_half_height = max(max_half_height, half_height)

        offset_x = cursor_x + half_width
        body = _draw(rotated, offset_x=offset_x)
        placed.append(body)
        labels.append((offset_x, half_height, label))
        cursor_x = offset_x + half_width + SERIES_GAP_ANGSTROM

    total_width = cursor_x - SERIES_GAP_ANGSTROM
    label_band = 2.6
    min_y = -max_half_height
    height = 2 * max_half_height + label_band

    label_y = max_half_height + label_band * 0.75
    label_markup = "\n    ".join(
        f'<text x="{x:.3f}" y="{label_y:.3f}" text-anchor="middle" '
        f'font-size="1.6" fill="var(--graphite)" stroke="none" '
        f'font-family="var(--mono)">{text}</text>'
        for x, _half, text in labels
    )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{-ATOM_RADIUS:.3f} {min_y:.3f} '
        f'{total_width + 2 * ATOM_RADIUS:.3f} {height:.3f}" role="img">\n'
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
    args = parser.parse_args()

    rotate = tuple(float(v) for v in args.rotate.split(","))
    if len(rotate) != 3:
        parser.error("--rotate needs exactly three comma-separated degrees")

    if len(args.inputs) == 1 and not args.labels:
        positions_angstrom = read_xyz(args.inputs[0])
        svg = render(positions_angstrom, rotate)  # type: ignore[arg-type]
        print(f"{args.inputs[0].name}: {len(positions_angstrom)} atoms")
    else:
        labels = args.labels.split(",") if args.labels else [p.stem for p in args.inputs]
        if len(labels) != len(args.inputs):
            parser.error(f"{len(labels)} labels for {len(args.inputs)} inputs")
        clusters = [(label, read_xyz(path)) for label, path in zip(labels, args.inputs)]
        svg = render_series(clusters, rotate)  # type: ignore[arg-type]
        for label, positions in clusters:
            print(f"  {label}: {len(positions)} atoms")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg)
    print(f"-> {args.output}")


def _self_check() -> None:
    """Bond detection and depth ordering, on a geometry with known answers."""
    # Four carbons in a 1.4 A square: four bonds around the edge, and the
    # 1.98 A diagonals correctly excluded.
    square = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [1.4, 1.4, 0.0], [0.0, 1.4, 0.0]])
    assert len(find_bonds(square)) == 4, find_bonds(square)

    # Cutoff boundary: 1.7 A bonded, 1.9 A not.
    assert len(find_bonds(np.array([[0.0, 0, 0], [1.7, 0, 0]]))) == 1
    assert len(find_bonds(np.array([[0.0, 0, 0], [1.9, 0, 0]]))) == 0

    # Depth maps to the stated opacity range, near-end brightest.
    assert math.isclose(depth_opacity(5.0, 5.0, -5.0), OPACITY_NEAR)
    assert math.isclose(depth_opacity(-5.0, 5.0, -5.0), OPACITY_FAR)
    assert OPACITY_FAR < depth_opacity(0.0, 5.0, -5.0) < OPACITY_NEAR

    # A degenerate flat-in-z cluster must not divide by zero.
    assert math.isclose(depth_opacity(1.0, 1.0, 1.0), OPACITY_NEAR)

    print("self-check ok")


if __name__ == "__main__":
    import sys

    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()
