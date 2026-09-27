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

WHITE = 255
CORNER_BLOCK_PX = 8
MAX_BORDER_STRAYS = 2  # compositor artefacts, see render_box_grid.py's module docstring
# Guards against a coloured material, which lands far above 100/255. (AgX keeps a neutral
# grey neutral, so this does not catch it; render_box_grid.scene_for's
# view_transform="Standard" enforces that.) The residual is HDRI specular on sphere
# highlights, measured 14-16/255 on 2026-09-27; the output is saved as 8-bit grey anyway.
MAX_CHROMA = 32


def load_panels(manifest_path: Path) -> list[dict]:
    """Panels with the plate-level defaults folded in, later keys winning.

    Args:
        manifest_path: Path to the JSON manifest described in render_box_grid.py's
            module docstring.

    Returns:
        One dict per panel, each holding every plate-level key (lengths in angstrom,
        resolution in pixels) overridden by the panel's own keys.

    Raises:
        ValueError: If a panel lacks a key in REQUIRED_KEYS, holds a key outside
            REQUIRED_KEYS and OPTIONAL_KEYS, an id does not match PANEL_ID (letters,
            digits, "_", "." and "-"), or two panels share an id.
    """
    manifest = json.loads(manifest_path.read_text())
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
