# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pillow"]
# ///
"""Manifest, geometry and pixel checks for `render_box_grid.py`, importable from `.venv`.

`render_box_grid.py` imports `molrender` at module top, which only exists in the separate
Blender environment named in its own docstring, so nothing in that file can be exercised
from the project's `.venv`. The checks that do not themselves need Blender live here
instead, pure numpy and Pillow: manifest loading and validation (`load_panels`), the cage
containment guard that keeps every panel of a plate at one shared scale
(`assert_fits_cage`), and the rendered-PNG whiteness and chroma check
(`check_and_flatten`). `render_box_grid.py` imports all three and keeps only the parts that
need molrender: the PDB-writing half of `prepare`, `scene_for`, `render_panel` and `main`.

Units: lengths in angstrom, pixel values 0-255.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

REQUIRED_KEYS = (
    "id",
    "source",
    "output_dir",
    "direction",
    "resolution",
    "sphere_radius_angstrom",
    "bond_radius_angstrom",
    "grey",
)
# Optional plate- or panel-level keys documented in render_box_grid.py's module
# docstring, beyond REQUIRED_KEYS. Anything else in a panel's merged dict is a typo
# (a plate-level key misspelled, e.g. "slab_angstorm" for "slab_angstrom") that would
# otherwise render the whole panel at its default rather than failing loudly.
OPTIONAL_KEYS = frozenset({"margin", "cage_angstrom", "slab_angstrom", "density_g_cm3", "tile"})
ALLOWED_KEYS = frozenset(REQUIRED_KEYS) | OPTIONAL_KEYS
PANEL_ID = re.compile(r"[\w.-]+")


def _is_number(value: object) -> bool:
    """True for an int or float that is not a bool (JSON true would otherwise pass)."""
    return isinstance(value, int | float) and not isinstance(value, bool)


def _positive_list(value: object, length: int, integers: bool = False) -> bool:
    """True for a list of `length` positive numbers, integers only if `integers`."""
    return (
        isinstance(value, list)
        and len(value) == length
        and all(
            _is_number(item) and item > 0 and (not integers or isinstance(item, int))
            for item in value
        )
    )


# Value rules for load_panels, checked on each panel's merged dict wherever the key is
# present: (test, what a valid value is). Paths are relative to the repository root,
# where every command in CLAUDE.md runs.
VALUE_RULES = {
    "resolution": (
        lambda value: _positive_list(value, 2, integers=True),
        "two positive integers, [width, height] in pixels",
    ),
    "grey": (lambda value: _is_number(value) and 0 <= value <= 1, "a linear grey from 0 to 1"),
    "cage_angstrom": (
        lambda value: value is None or _positive_list(value, 3),
        "null or three positive extents [x, y, z] in angstrom",
    ),
    "density_g_cm3": (
        lambda value: _is_number(value) and value > 0,
        "a positive density in g cm^-3",
    ),
    "tile": (
        lambda value: _positive_list(value, 3, integers=True),
        "three positive integers [nx, ny, nz]",
    ),
    "source": (
        lambda value: isinstance(value, str) and Path(value).is_file(),
        "the path of an existing XYZ file",
    ),
}

WHITE = 255
CORNER_BLOCK_PX = 8
MAX_BORDER_STRAYS = 2  # compositor artefacts, see render_box_grid.py's module docstring
# Guards against a coloured material, which lands far above 100/255. (AgX keeps a neutral
# grey neutral, so this does not catch it; render_box_grid.scene_for's
# view_transform="Standard" enforces that.) The residual is HDRI specular on sphere
# highlights, measured 14-16/255 on 2026-09-27; the output is saved as 8-bit grey anyway.
MAX_CHROMA = 32
INK_THRESHOLD = 245  # pixels darker than this are drawn atoms or bonds
# Border round the ink of a self-framed panel, per side, as a fraction of its square.
CROP_MARGIN = 0.03


def load_panels(manifest_path: Path) -> list[dict]:
    """Panels with the plate-level defaults folded in, later keys winning.

    Args:
        manifest_path: Path to the JSON manifest described in render_box_grid.py's
            module docstring.

    Returns:
        One dict per panel, each holding every plate-level key (lengths in angstrom,
        resolution in pixels) overridden by the panel's own keys.

    Raises:
        ValueError: If the manifest has no "panels" list, a panel lacks a key in
            REQUIRED_KEYS, holds a key outside REQUIRED_KEYS and OPTIONAL_KEYS, has a
            value that breaks its VALUE_RULES entry (a resolution of two positive
            integers, a grey in [0, 1], a cage of null or three positive extents, a
            positive density, a tile of three positive integers, a source that exists),
            an id does not match PANEL_ID (letters, digits, "_", "." and "-"), or two
            panels share an id. Every message names the manifest, the panel and the key.
    """
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest.get("panels"), list):
        raise ValueError(f'{manifest_path}: no "panels" list')
    plate = {key: value for key, value in manifest.items() if key != "panels"}
    panels = [{**plate, **panel} for panel in manifest["panels"]]
    seen_ids: set[str] = set()
    for index, panel in enumerate(panels):
        name = panel.get("id", f"panels[{index}]")
        missing = [key for key in REQUIRED_KEYS if key not in panel]
        if missing:
            raise ValueError(f"{manifest_path}: panel {name} lacks {missing}")
        unknown = sorted(set(panel) - ALLOWED_KEYS)
        if unknown:
            raise ValueError(f"{manifest_path}: panel {name} has unknown key(s) {unknown}")
        for key, (is_valid, expected) in VALUE_RULES.items():
            if key in panel and not is_valid(panel[key]):
                raise ValueError(
                    f"{manifest_path}: panel {name}: {key} is {panel[key]!r}, expected {expected}"
                )
        if not PANEL_ID.fullmatch(str(name)):
            raise ValueError(f"{manifest_path}: panel id {name!r} does not match [\\w.-]+")
        if name in seen_ids:
            raise ValueError(f"{manifest_path}: panel id {name} appears twice")
        seen_ids.add(name)
    return panels


def assert_fits_cage(
    positions_angstrom: np.ndarray,
    centre_angstrom: np.ndarray,
    cage_angstrom: list[float] | tuple[float, float, float],
    sphere_radius_angstrom: float,
    panel_id: str,
) -> None:
    """Raise unless the subject, with its drawn spheres, fits inside the cage.

    A fixed cage gives every panel of a plate the same angstrom per pixel (molrender
    fits the camera to everything loaded, hidden or not); a subject too big for its
    cage would otherwise frame itself and quietly break that shared scale.

    Args:
        positions_angstrom: Atom positions, shape (n_atoms, 3), in angstrom.
        centre_angstrom: The cage's centre, shape (3,), in angstrom.
        cage_angstrom: [x, y, z] full extents of the cage, in angstrom.
        sphere_radius_angstrom: Drawn sphere radius, in angstrom.
        panel_id: Panel id, named in the error.

    Raises:
        ValueError: If, on any axis, twice the farthest atom's distance from
            `centre_angstrom` plus twice `sphere_radius_angstrom` exceeds the cage's
            extent on that axis.
    """
    half_extent_angstrom = np.asarray(cage_angstrom, dtype=float) / 2.0
    reach_angstrom = np.abs(positions_angstrom - centre_angstrom).max(axis=0)
    needed_angstrom = 2.0 * (reach_angstrom + float(sphere_radius_angstrom))
    if (needed_angstrom > 2.0 * half_extent_angstrom).any():
        raise ValueError(
            f"panel {panel_id}: subject with spheres needs a cage of "
            f"{np.round(needed_angstrom, 2).tolist()} A about its centre, "
            f"cage_angstrom is {list(cage_angstrom)}"
        )


def check_and_flatten(
    png: Path, expected_size: tuple[int, int]
) -> list[tuple[int, int, list[int]]]:
    """Assert size, a white world and near-zero chroma, then save as 8-bit grey.

    The world is judged by blocks, not single pixels, because the Molecular Nodes
    compositor can leave a stray pixel on the border (render_box_grid.py's module
    docstring): the median of each CORNER_BLOCK_PX square corner block must be 255 on
    every channel, and at most MAX_BORDER_STRAYS pixels on the outer one-pixel border
    may be non-white, each with no non-white pixel among its eight neighbours. A
    single-pixel tangential graze of a sphere against the border would pass as one of
    those strays; anything wider is a run of non-white pixels and still fails.

    Args:
        png: The rendered PNG, overwritten in place as mode L.
        expected_size: (width, height) in pixels the file on disk must have.

    Returns:
        The tolerated border strays as (row, column, [r, g, b]) in pixels, 0-255.

    Raises:
        RuntimeError: If the size differs, a corner block's median is not white, the
            border has more than MAX_BORDER_STRAYS non-white pixels or one that is not
            isolated, or the largest per-pixel channel spread exceeds MAX_CHROMA.
    """
    image = Image.open(png).convert("RGB")
    if image.size != expected_size:
        raise RuntimeError(f"{png}: rendered {image.size}, expected {expected_size}")
    pixels = np.asarray(image, dtype=np.int16)
    block = CORNER_BLOCK_PX
    corner_blocks = [
        pixels[:block, :block],
        pixels[:block, -block:],
        pixels[-block:, :block],
        pixels[-block:, -block:],
    ]
    medians = [np.median(corner.reshape(-1, 3), axis=0) for corner in corner_blocks]
    if any((median != WHITE).any() for median in medians):
        raise RuntimeError(
            f"{png}: corner block medians are not white: {[median.tolist() for median in medians]}"
        )
    non_white = (pixels < WHITE).any(axis=2)
    on_border = np.ones_like(non_white)
    on_border[1:-1, 1:-1] = False
    rows, columns = np.nonzero(non_white & on_border)
    strays = [
        (int(row), int(column), pixels[row, column].tolist())
        for row, column in zip(rows, columns, strict=True)
    ]
    if len(strays) > MAX_BORDER_STRAYS:
        raise RuntimeError(
            f"{png}: {len(strays)} non-white border pixels, at most {MAX_BORDER_STRAYS}: "
            f"{strays[:8]}"
        )
    padded = np.pad(non_white, 1)
    for row, column, value in strays:
        if padded[row : row + 3, column : column + 3].sum() > 1:
            raise RuntimeError(
                f"{png}: non-white border pixel ({row}, {column}) {value} has a non-white neighbour"
            )
    chroma = int((pixels.max(axis=2) - pixels.min(axis=2)).max())
    if chroma > MAX_CHROMA:
        raise RuntimeError(f"{png}: max chroma {chroma}/255 exceeds {MAX_CHROMA}")
    image.convert("L").save(png, optimize=True)
    return strays


def crop_to_ink(png: Path, margin_fraction: float = CROP_MARGIN) -> int:
    """Crop a flattened grey panel, in place, to a square round its ink.

    For a panel framed to its own subject (`cage_angstrom` null) molrender fits the
    camera to the subject's three-dimensional box, so the drawn atoms fill a median 77%
    of the frame's long side (design critique 2, finding 2b). The square is centred on
    the ink's bounding box, as long as that box's longer side plus `margin_fraction` on
    each side, and shifted inside the image where it would cross an edge (which loses no
    ink). It is never shrunk: a square that does not fit raises instead, so a panel is
    never silently cut to its short side. Only a square render is cropped this way;
    render_box_grid.py leaves a non-square one (the hero) at its full frame. Isolated
    dark pixels do not count: the
    Molecular Nodes compositor leaves one at the centre and sometimes one on the border
    (render_box_grid.py's module docstring), and a border stray would otherwise pin the
    crop to the edge. A panel that shares a cage must never be cropped, or the shared
    angstrom per pixel is lost; render_box_grid.py enforces that.

    Args:
        png: An 8-bit grey PNG written by `check_and_flatten`, overwritten in place.
        margin_fraction: Border added on each side, as a fraction of the square's side.

    Returns:
        The side of the cropped square in pixels.

    Raises:
        RuntimeError: If the panel holds no connected ink darker than INK_THRESHOLD, or
            the ink's square with its margin is larger than the image's shorter side.
    """
    image = Image.open(png)
    pixels = np.asarray(image.convert("L"))
    ink = pixels < INK_THRESHOLD
    padded = np.pad(ink, 1)
    neighbours = sum(
        padded[1 + dr : padded.shape[0] - 1 + dr, 1 + dc : padded.shape[1] - 1 + dc]
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if (dr, dc) != (0, 0)
    )
    connected = ink & (neighbours > 0)
    rows, columns = np.nonzero(connected)
    if rows.size == 0:
        raise RuntimeError(f"{png}: no connected ink darker than {INK_THRESHOLD}/255 to crop to")
    height_px, width_px = pixels.shape
    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(columns.min()), int(columns.max()) + 1
    side_px = int(round(max(bottom - top, right - left) * (1 + 2 * margin_fraction)))
    if side_px > min(height_px, width_px):
        raise RuntimeError(
            f"{png}: the ink's {right - left} x {bottom - top} px box with a "
            f"{margin_fraction:.0%} margin needs a {side_px} px square, larger than the "
            f"{width_px} x {height_px} px image; widen the render or give it a cage"
        )
    row0 = int(round((top + bottom) / 2 - side_px / 2))
    column0 = int(round((left + right) / 2 - side_px / 2))
    row0 = min(max(row0, 0), height_px - side_px)
    column0 = min(max(column0, 0), width_px - side_px)
    image.crop((column0, row0, column0 + side_px, row0 + side_px)).save(png, optimize=True)
    return side_px
