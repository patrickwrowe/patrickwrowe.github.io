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
(`check_and_flatten`), and the one rule that decides whether a panel is coloured or grey
(`is_coloured`). `render_box_grid.py` imports them and keeps only the parts that need
molrender: the PDB-writing half of `prepare`, `scene_for`, `render_panel` and `main`.

Units: lengths in angstrom, pixel values 0-255.
"""

from __future__ import annotations

import json
import math
import re
from enum import StrEnum
from pathlib import Path

import numpy as np
from constants import VDW_RADIUS_ANGSTROM
from PIL import Image

# A relative `source` or `output_dir` in a manifest is relative to the repository root,
# whatever the current directory (pytest from scripts/tests, the renderer from anywhere).
REPO_ROOT = Path(__file__).resolve().parents[2]


class PanelKind(StrEnum):
    """What a panel draws: a periodic box or free cluster from XYZ, or a PDB protein."""

    BOX = "box"
    PROTEIN = "protein"


REQUIRED_KEYS = (
    "id",
    "source",
    "output_dir",
    "direction",
    "resolution",
    "sphere_radius_angstrom",
    "bond_radius_angstrom",
)
# Keys only a box panel must carry: a protein is always coloured, so it has no grey.
BOX_KEYS = ("grey",)
# Optional plate- or panel-level keys documented in render_box_grid.py's module
# docstring, beyond REQUIRED_KEYS. Anything else in a panel's merged dict is a typo
# (a plate-level key misspelled, e.g. "slab_angstorm" for "slab_angstrom") that would
# otherwise render the whole panel at its default rather than failing loudly.
OPTIONAL_KEYS = frozenset(
    {
        "margin",
        "cage_angstrom",
        "slab_angstrom",
        "density_g_cm3",
        "tile",
        "colour",
        "kind",
        "cartoon_chains",
        "stick_chains",
        "chain_colour",
        "sphere_scale",
        "light",
        "quality",
        "depth_cue",
    }
)
# What a manifest's "light" may override in constants.RIG, and its "quality" in molrender's
# PRODUCTION or DRAFT tier (Patrick, 2026-10-17: the excited-states hero's render settings).
LIGHT_OVERRIDE_KEYS = frozenset({"key_power", "sun_strength", "side_power", "under_power"})
QUALITY_OVERRIDE_KEYS = frozenset(
    {"diffuse_bounces", "glossy_bounces", "ao_bounces_render", "fast_gi_method"}
)
# What a manifest's "depth_cue" holds: how far the farthest atoms fade toward the backdrop's
# white (molrender DepthRamp; Patrick, 2026-10-17: the hBN nanotube and bilayer).
DEPTH_CUE_KEYS = frozenset({"strength"})
ALLOWED_KEYS = frozenset(REQUIRED_KEYS) | frozenset(BOX_KEYS) | OPTIONAL_KEYS
PANEL_ID = re.compile(r"[\w.-]+")
ELEMENT_SYMBOL = re.compile(r"[A-Z][a-z]?")
HEX_COLOUR = re.compile(r"#[0-9a-fA-F]{6}")


def _is_number(value: object) -> bool:
    """True for a finite int or float that is not a bool, and that `float()` can hold.

    JSON `true` would otherwise pass as 1, and Python's json module reads `Infinity` and
    `NaN`, which pass every `>= 1` or `> 0` rule that does not rule them out. An integer
    too large for a float (`10**400`) is refused too: the driver calls `float()` on these
    values, which would raise a bare `OverflowError` with no panel or key in it.
    """
    if not isinstance(value, int | float) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


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


def _is_direction(value: object) -> bool:
    """Three numbers with at least one non-zero; components may be negative."""
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(_is_number(component) for component in value)
        and any(component != 0 for component in value)
    )


def _is_chain_list(value: object) -> bool:
    """A list of distinct PDB chain ids, each one character (column 22), possibly empty."""
    return (
        isinstance(value, list)
        and all(isinstance(chain, str) and len(chain) == 1 and chain != " " for chain in value)
        and len(set(value)) == len(value)
    )


def _is_colour_map(value: object) -> bool:
    """A dict of element symbol ("O", "Mo") to sRGB hex ("#rrggbb"), possibly empty."""
    return isinstance(value, dict) and all(
        ELEMENT_SYMBOL.fullmatch(symbol)
        and isinstance(hex_colour, str)
        and HEX_COLOUR.fullmatch(hex_colour)
        for symbol, hex_colour in value.items()
    )


def _is_scale_map(value: object) -> bool:
    """A dict of element symbol ("Mg") to a positive factor on sphere_radius_angstrom."""
    return isinstance(value, dict) and all(
        isinstance(symbol, str)
        and ELEMENT_SYMBOL.fullmatch(symbol)
        and _is_number(factor)
        and factor > 0
        for symbol, factor in value.items()
    )


def _is_light_override(value: object) -> bool:
    """A dict of LIGHT_OVERRIDE_KEYS to non-negative powers, merged over constants.RIG."""
    return (
        isinstance(value, dict)
        and set(value) <= LIGHT_OVERRIDE_KEYS
        and all(_is_number(power) and power >= 0 for power in value.values())
    )


def _is_quality_override(value: object) -> bool:
    """A dict of QUALITY_OVERRIDE_KEYS: bounce counts of at least 1, a Fast GI method by name."""
    return (
        isinstance(value, dict)
        and set(value) <= QUALITY_OVERRIDE_KEYS
        and all(
            setting in ("REPLACE", "ADD")
            if key == "fast_gi_method"
            else isinstance(setting, int) and not isinstance(setting, bool) and setting >= 1
            for key, setting in value.items()
        )
    )


def _is_depth_cue(value: object) -> bool:
    """{"strength": s}, 0 < s <= 1: the mix toward the backdrop at the far side of the subject."""
    return (
        isinstance(value, dict)
        and set(value) == DEPTH_CUE_KEYS
        and _is_number(value["strength"])
        and 0 < value["strength"] <= 1
    )


def _is_chain_colour_map(value: object) -> bool:
    """A dict of one-character chain id ("A") to sRGB hex ("#rrggbb"), possibly empty."""
    return (
        isinstance(value, dict)
        and _is_chain_list(list(value))
        and all(
            isinstance(hex_colour, str) and HEX_COLOUR.fullmatch(hex_colour)
            for hex_colour in value.values()
        )
    )


# Value rules for load_panels, checked on each panel's merged dict wherever the key is
# present: (test, what a valid value is). Paths are relative to the repository root,
# where every command in CLAUDE.md runs.
VALUE_RULES = {
    "margin": (lambda value: _is_number(value) and value >= 1, "a margin of at least 1"),
    "sphere_radius_angstrom": (
        lambda value: _is_number(value) and value > 0,
        "a positive radius in angstrom",
    ),
    "slab_angstrom": (
        lambda value: value is None or (_is_number(value) and value > 0),
        "null or a positive thickness in angstrom",
    ),
    "direction": (_is_direction, "three numbers, not all zero"),
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
    "colour": (_is_colour_map, "a map of element symbol to sRGB hex"),
    "kind": (lambda value: value in tuple(PanelKind), 'either "box" or "protein"'),
    "cartoon_chains": (
        lambda value: _is_chain_list(value) and len(value) > 0,
        "a non-empty list of distinct one-character chain ids",
    ),
    "stick_chains": (_is_chain_list, "a list of distinct one-character chain ids"),
    "chain_colour": (_is_chain_colour_map, "a map of one-character chain id to sRGB hex"),
    "sphere_scale": (_is_scale_map, "a map of element symbol to a positive factor"),
    "light": (
        _is_light_override,
        f"a map of {sorted(LIGHT_OVERRIDE_KEYS)} to non-negative powers",
    ),
    "quality": (
        _is_quality_override,
        f"a map of {sorted(QUALITY_OVERRIDE_KEYS)}: bounces of at least 1, "
        'fast_gi_method "REPLACE" or "ADD"',
    ),
    "depth_cue": (_is_depth_cue, 'a map {"strength": s} with 0 < s <= 1'),
    "source": (
        lambda value: isinstance(value, str) and Path(value).is_file(),
        "the path of an existing XYZ file (a PDB file for a protein panel)",
    ),
}
# Keys that only a protein panel (kind "protein") may carry, and must carry.
PROTEIN_KEYS = ("cartoon_chains", "stick_chains")
# Keys that only a protein panel may carry, and need not.
PROTEIN_OPTIONAL_KEYS = ("chain_colour",)
# Keys a protein panel cannot use: it is framed to its own complex, with no periodic cell.
PROTEIN_UNUSED_KEYS = ("cage_angstrom", "slab_angstrom", "density_g_cm3", "tile")

WHITE = 255
CORNER_BLOCK_PX = 8
MAX_BORDER_STRAYS = 2  # compositor artefacts, see render_box_grid.py's module docstring
# Guards against a coloured material, which lands far above 100/255. (AgX keeps a neutral
# grey neutral, so this does not catch it; render_box_grid.scene_for's
# view_transform="Standard" enforces that.) The residual is HDRI specular on sphere
# highlights, measured 14-16/255 on 2026-09-27; the output is saved as 8-bit grey anyway.
# A coloured panel (more than one species, or any protein) is coloured on purpose, so there
# the limit judges the world instead: the CORNER_BLOCK_PX border round the frame, where
# nothing is drawn.
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
        resolution in pixels) overridden by the panel's own keys, with a relative
        `source` or `output_dir` resolved against REPO_ROOT, and its `kind` always set
        (PanelKind.BOX when the manifest leaves it out), so no reader applies a default.

    Raises:
        ValueError: If the manifest has no "panels" list, a panel lacks a key in
            REQUIRED_KEYS, holds a key outside REQUIRED_KEYS and OPTIONAL_KEYS, has a
            value that breaks its VALUE_RULES entry (a resolution of two positive
            integers, a grey in [0, 1], a cage of null or three positive extents, a
            positive density, a tile of three positive integers, a source that exists, a
            margin of at least 1, a positive sphere radius, a slab of null or a positive
            thickness, a direction of three numbers that are not all zero, a colour map
            of element symbol to "#rrggbb"; every number must be finite; a kind of "box" or
            "protein"; chain lists of distinct one-character ids, the cartoon list not
            empty), a protein panel that lacks PROTEIN_KEYS, whose source is not a .pdb
            file, that sets a non-null PROTEIN_UNUSED_KEYS entry, or whose two chain lists
            share a chain, whose chain_colour (a map of chain id to "#rrggbb") names a chain
            not in cartoon_chains, a box panel that carries chain lists or chain_colour,
            an id does not match PANEL_ID (letters, digits, "_", "." and "-"), or two
            panels share an id, or a box panel lacks BOX_KEYS. Every message names the
            manifest, the panel and the key.
    """
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest.get("panels"), list):
        raise ValueError(f'{manifest_path}: no "panels" list')
    plate = {key: value for key, value in manifest.items() if key != "panels"}
    panels = [{**plate, **panel} for panel in manifest["panels"]]
    for panel in panels:
        panel.setdefault("kind", PanelKind.BOX)
        for key in ("source", "output_dir"):
            if isinstance(panel.get(key), str) and not Path(panel[key]).is_absolute():
                panel[key] = str(REPO_ROOT / panel[key])
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
        if panel["kind"] == PanelKind.PROTEIN:
            lacking = [key for key in PROTEIN_KEYS if key not in panel]
            if lacking:
                raise ValueError(f"{manifest_path}: panel {name} is a protein and lacks {lacking}")
            if not panel["source"].endswith(".pdb"):
                raise ValueError(f"{manifest_path}: panel {name}: source is not a .pdb file")
            unused = [key for key in PROTEIN_UNUSED_KEYS if panel.get(key) is not None]
            if unused:
                raise ValueError(
                    f"{manifest_path}: panel {name}: a protein is framed to its own complex, "
                    f"so {unused} must be null or absent"
                )
            shared = sorted(set(panel["cartoon_chains"]) & set(panel["stick_chains"]))
            if shared:
                raise ValueError(
                    f"{manifest_path}: panel {name}: chain(s) {shared} are in both "
                    "cartoon_chains and stick_chains"
                )
            uncoloured = sorted(set(panel.get("chain_colour", {})) - set(panel["cartoon_chains"]))
            if uncoloured:
                raise ValueError(
                    f"{manifest_path}: panel {name}: chain_colour names chain(s) {uncoloured}, "
                    "which are not in cartoon_chains"
                )
        elif any(key in panel for key in (*PROTEIN_KEYS, *PROTEIN_OPTIONAL_KEYS)):
            raise ValueError(f"{manifest_path}: panel {name} has chain keys but is not a protein")
        elif lacking := [key for key in BOX_KEYS if key not in panel]:
            raise ValueError(f"{manifest_path}: panel {name} lacks {lacking}")
        if not PANEL_ID.fullmatch(str(name)):
            raise ValueError(f"{manifest_path}: panel id {name!r} does not match [\\w.-]+")
        if name in seen_ids:
            raise ValueError(f"{manifest_path}: panel id {name} appears twice")
        seen_ids.add(name)
    return panels


def is_coloured(spec: dict, species: list[str]) -> bool:
    """Whether a panel is drawn in colour, the one rule both the scene and the saved PNG follow.

    CLAUDE.md: a render of more than one species takes one colour per species, and a
    protein's cartoon one colour per chain, so a protein is coloured whatever it draws by
    species; a box of one species is grey.

    Args:
        spec: One panel from `load_panels`, whose `kind` is always set.
        species: Element symbol of every atom the panel draws.

    Returns:
        True for a protein or a frame of more than one species.
    """
    return spec["kind"] == PanelKind.PROTEIN or len(set(species)) > 1


def assert_fits_cage(
    positions_angstrom: np.ndarray,
    species: list[str],
    centre_angstrom: np.ndarray,
    cage_angstrom: list[float] | tuple[float, float, float],
    sphere_radius_angstrom: float,
    panel_id: str,
) -> None:
    """Raise unless the subject, with its drawn spheres, fits inside the cage.

    A fixed cage gives every panel of a plate the same angstrom per pixel (molrender
    fits the camera to everything loaded, hidden or not); a subject too big for its
    cage would otherwise frame itself and quietly break that shared scale.

    Each atom is allowed the radius the renderer draws it at: Molecular Nodes scales each
    species' own van der Waals radius by `sphere_radius_angstrom` over carbon's
    (constants.VDW_RADIUS_ANGSTROM), so carbon reaches `sphere_radius_angstrom` and
    potassium 1.62 times as far.

    Args:
        positions_angstrom: Atom positions, shape (n_atoms, 3), in angstrom.
        species: Element symbol per atom.
        centre_angstrom: The cage's centre, shape (3,), in angstrom.
        cage_angstrom: [x, y, z] full extents of the cage, in angstrom.
        sphere_radius_angstrom: Drawn radius of a carbon sphere, in angstrom.
        panel_id: Panel id, named in the error.

    Raises:
        ValueError: If, on any axis, twice the largest reach of an atom's drawn sphere
            from `centre_angstrom` (its distance plus its drawn radius) exceeds the cage's
            extent on that axis.
    """
    half_extent_angstrom = np.asarray(cage_angstrom, dtype=float) / 2.0
    drawn_radius_angstrom = (
        float(sphere_radius_angstrom)
        * np.array([VDW_RADIUS_ANGSTROM[symbol] for symbol in species])
        / VDW_RADIUS_ANGSTROM["C"]
    )
    reach_angstrom = (
        np.abs(positions_angstrom - centre_angstrom) + drawn_radius_angstrom[:, None]
    ).max(axis=0)
    needed_angstrom = 2.0 * reach_angstrom
    if (needed_angstrom > 2.0 * half_extent_angstrom).any():
        raise ValueError(
            f"panel {panel_id}: subject with spheres needs a cage of "
            f"{np.round(needed_angstrom, 2).tolist()} A about its centre, "
            f"cage_angstrom is {list(cage_angstrom)}"
        )


def check_and_flatten(
    png: Path,
    expected_size: tuple[int, int],
    label: str | None = None,
    coloured: bool = False,
) -> list[tuple[int, int, list[int]]]:
    """Assert size, a white world and near-zero chroma, then save as 8-bit grey or RGB.

    The world is judged by blocks, not single pixels, because the Molecular Nodes
    compositor can leave a stray pixel on the border (render_box_grid.py's module
    docstring): the median of each CORNER_BLOCK_PX square corner block must be 255 on
    every channel, and at most MAX_BORDER_STRAYS pixels on the outer one-pixel border
    may be non-white, each with no non-white pixel among its eight neighbours. A
    single-pixel tangential graze of a sphere against the border would pass as one of
    those strays; anything wider is a run of non-white pixels and still fails.

    A grey panel (a box of one species) is held to the chroma limit over the whole frame and
    saved as mode L. A coloured panel (a box of more than one species, or any protein) is
    coloured on purpose: the limit then holds only over the CORNER_BLOCK_PX border round the
    frame, and the file keeps its three channels.

    Args:
        png: The rendered PNG, overwritten in place as mode L, or as mode RGB when
            `coloured`.
        expected_size: (width, height) in pixels the file on disk must have.
        label: What every error message names in place of `png`: the panel and its output
            path, since the scratch file checked here is gone by the time anyone reads it.
        coloured: Whether the panel is coloured: a box of more than one species, or a
            protein.

    Returns:
        The tolerated border strays as (row, column, [r, g, b]) in pixels, 0-255.

    Raises:
        RuntimeError: If the size differs, a corner block's median is not white, the
            border has more than MAX_BORDER_STRAYS non-white pixels or one that is not
            isolated, or the largest per-pixel channel spread exceeds MAX_CHROMA (over
            the whole frame, or over the CORNER_BLOCK_PX border when `coloured`).
    """
    name = label or png
    image = Image.open(png).convert("RGB")
    if image.size != expected_size:
        raise RuntimeError(f"{name}: rendered {image.size}, expected {expected_size}")
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
            f"{name}: corner block medians are not white: {[median.tolist() for median in medians]}"
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
            f"{name}: {len(strays)} non-white border pixels, at most {MAX_BORDER_STRAYS}: "
            f"{strays[:8]}"
        )
    padded = np.pad(non_white, 1)
    for row, column, value in strays:
        if padded[row : row + 3, column : column + 3].sum() > 1:
            raise RuntimeError(
                f"{name}: non-white border pixel ({row}, {column}) {value} "
                "has a non-white neighbour"
            )
    spread = pixels.max(axis=2) - pixels.min(axis=2)
    if coloured:
        ring = np.ones_like(non_white)
        ring[block:-block, block:-block] = False
        spread = spread[ring]
    chroma = int(spread.max())
    if chroma > MAX_CHROMA:
        where = f" in the {block} px border" if coloured else ""
        raise RuntimeError(f"{name}: max chroma {chroma}/255{where} exceeds {MAX_CHROMA}")
    (image if coloured else image.convert("L")).save(png, optimize=True)
    return strays


def crop_to_ink(png: Path, margin_fraction: float = CROP_MARGIN) -> int:
    """Crop a flattened panel, grey or RGB, in place, to a square round its ink.

    For a panel framed to its own subject (`cage_angstrom` null) molrender fits the
    camera to the subject's three-dimensional box, so the drawn atoms fill a median 77%
    of the frame's long side (design critique 2, finding 2b). The square is centred on
    the ink's bounding box, as long as that box's longer side plus `margin_fraction` on
    each side, and shifted inside the image where it would cross an edge (which loses no
    ink). A square larger than the image's shorter side is clamped to it, which in a
    square frame gives up margin only; it raises when the ink's box itself exceeds that
    side, so a panel is never cut to its short side. Only a square render is cropped
    this way; render_box_grid.py leaves a non-square one (the hero) at its full frame.
    The ink is judged on luminance, so a coloured panel crops exactly as a grey one of the
    same lightness does, and keeps its mode. Isolated dark pixels do not count: the
    Molecular Nodes compositor leaves one at the centre and sometimes one on the border
    (render_box_grid.py's module docstring), and a border stray would otherwise pin the
    crop to the edge. A panel that shares a cage must never be cropped, or the shared
    angstrom per pixel is lost; render_box_grid.py enforces that.

    Args:
        png: An 8-bit grey or RGB PNG written by `check_and_flatten`, overwritten in place.
        margin_fraction: Border added on each side, as a fraction of the square's side.

    Returns:
        The side of the cropped square in pixels.

    Raises:
        RuntimeError: If the panel holds no connected ink darker than INK_THRESHOLD, or
            the ink's bounding box is larger than the image's shorter side.
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
    ink_side_px = max(bottom - top, right - left)
    if ink_side_px > min(height_px, width_px):
        raise RuntimeError(
            f"{png}: the ink's {right - left} x {bottom - top} px box does not fit a square "
            f"inside the {width_px} x {height_px} px image; widen the render or give it a cage"
        )
    side_px = min(int(round(ink_side_px * (1 + 2 * margin_fraction))), height_px, width_px)
    row0 = int(round((top + bottom) / 2 - side_px / 2))
    column0 = int(round((left + right) / 2 - side_px / 2))
    row0 = min(max(row0, 0), height_px - side_px)
    column0 = min(max(column0, 0), width_px - side_px)
    image.crop((column0, row0, column0 + side_px, row0 + side_px)).save(png, optimize=True)
    return side_px
