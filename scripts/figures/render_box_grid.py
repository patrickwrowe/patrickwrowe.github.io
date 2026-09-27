# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pillow", "molrender"]
# ///
"""Render periodic carbon boxes and free clusters as greyscale PNG panels through molrender.

One PNG per panel. The page lays panels out with `src/components/Grid.astro`, so labels
are real text in the site's mono face rather than pixels, and the grid reflows to three
columns at phone width without a second render (restructure spec section 6.2, 7).

Runs in the molrender environment, which is not the project venv:

    MOLRENDER_PYTHON=/home/patrick/.local/share/mamba/envs/molrender/bin/python
    $MOLRENDER_PYTHON scripts/figures/render_box_grid.py \
        --manifest scripts/figures/data/graphitisation/manifest.json \
        [--only ID ...] [--draft] [--force]

Molecular Nodes copies its startup template into Blender's user-scripts directory on every
render (molecularnodes scene/base.py, Canvas.__init__), and ~/.config is read-only inside a
sandboxed session. So when BLENDER_USER_SCRIPTS is unset the driver points it at a fresh
directory inside the run's scratch directory before the first render; the same command then
works inside and outside the sandbox. Fresh per run because Blender executes any Python under
<dir>/startup/. It holds only the template; extensions and preferences still load from
~/.config.

Manifest (JSON; plate-level keys are defaults every panel may override):
    output_dir             where `<id>.png` goes
    direction              vector from the subject towards the camera (molrender normalises
                           it), same for every panel; its z-up projection is the image's up
    margin                 molrender View.margin (1.02 = a two per cent border)
    cage_angstrom          [x, y, z] extents of a hidden box loaded with the subject, or null.
                           molrender fits the camera to everything loaded, hidden or not, so a
                           fixed cage gives every panel of a plate the same angstrom per pixel
                           (molrender session.py lines 26-30). null fits the subject alone.
                           The cage is centred on the subject's centre and must hold it with
                           its spheres: on every axis, twice the farthest atom's distance
                           from that centre plus 2 x sphere_radius_angstrom must not exceed
                           the cage, or the panel is refused (ValueError) rather than framed
                           at its own scale.
    slab_angstrom          slab thickness cut about the box centre perpendicular to z, or null
    resolution             [width, height] pixels on disk at the production tier
    sphere_radius_angstrom, bond_radius_angstrom
                           Molecular Nodes reads BallAndStick.sphere_radius as a factor on
                           each atom's vdW radius, not a length, so the sphere radius is
                           divided by carbon's 1.70 A here; every atom drawn is carbon.
    grey                   the material's linear RGB value, 0-1, on all three channels (what
                           Solid(rgb=...) receives); not a display value, since the Standard
                           view transform applies the sRGB curve on output
    panels[].id, .source   an XYZ file readable by render_cluster.read_xyz
    panels[].density_g_cm3 present for a periodic box: the cell edge follows from it and the
                           frame is wrapped; absent for a free cluster
    panels[].tile          [nx, ny, nz] periodic images (boxes only), default [1, 1, 1]

Every panel must end up with each key in REQUIRED_KEYS, from the plate or its own entry;
ids are unique and use only letters, digits, "_", "." and "-", because each names a file.

Rules this script enforces (spec section 6.2): production tier with the resolution named;
view transform Standard, never AgX; transparent film and shadow catcher off together; every
corner white (the median of each CORNER_BLOCK_PX square corner block is 255 on every
channel); at most MAX_BORDER_STRAYS non-white pixels on the outer one-pixel border, each
isolated, and reported in the summary line; residual chroma at most MAX_CHROMA; output
saved as 8-bit greyscale.
Each panel renders into the scratch directory and moves to `<id>.png` only once every check
passes, so a failed panel leaves nothing behind for the resume run to skip.
`--draft` uses molrender's DRAFT tier for framing checks only: EEVEE at half resolution, and
the size assertion is relaxed to match. EEVEE draws Point spheres far smaller than Cycles
does, so judge radii at the production tier.

Known defect: Molecular Nodes' compositor overlays an empty annotations image and leaves one
black pixel at the exact image centre and, on a non-square frame, can leave a stray pixel on
the border (the hero's top-right corner came back [234, 255, 232], the only non-white pixel
in its 40 x 40 corner region). molrender exposes no switch for it (2026-09-27), so the
whiteness check judges the world, corner blocks and isolated border strays, rather than
single pixels. Both artefacts vanish once molrender turns compositing off; re-render every
panel with `--force` then.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import boxprep
import numpy as np
from molrender import (
    DRAFT,
    PRODUCTION,
    BallAndStick,
    Layer,
    Scene,
    Solid,
    Structure,
    View,
    render,
)
from PIL import Image

WHITE = 255
CORNER_BLOCK_PX = 8
MAX_BORDER_STRAYS = 2  # compositor artefacts, see the module docstring
# Guards against a coloured material, which lands far above 100/255. (AgX keeps a neutral
# grey neutral, so this does not catch it; scene_for's view_transform="Standard" enforces
# that.) The residual is HDRI specular on sphere highlights, measured 14-16/255 on
# 2026-09-27; the output is saved as 8-bit grey anyway.
MAX_CHROMA = 32
RENDER_TIMEOUT_S = 3600
# Molecular Nodes assets/data.py; its BallAndStick sphere_radius scales this, bond_radius does not
CARBON_VDW_RADIUS_ANGSTROM = 1.70
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
PANEL_ID = re.compile(r"[\w.-]+")


def load_panels(manifest_path: Path) -> list[dict]:
    """Panels with the plate-level defaults folded in, later keys winning.

    Args:
        manifest_path: Path to the JSON manifest described in the module docstring.

    Returns:
        One dict per panel, each holding every plate-level key (lengths in angstrom,
        resolution in pixels) overridden by the panel's own keys.

    Raises:
        ValueError: If a panel lacks a key in REQUIRED_KEYS, an id does not match
            PANEL_ID (letters, digits, "_", "." and "-"), or two panels share an id.
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
        if not PANEL_ID.fullmatch(str(name)):
            raise ValueError(f"{manifest_path}: panel id {name!r} does not match [\\w.-]+")
        if name in seen_ids:
            raise ValueError(f"{manifest_path}: panel id {name} appears twice")
        seen_ids.add(name)
    return panels


def prepare(spec: dict, work_dir: Path) -> tuple[Path, Path | None, int]:
    """Write the subject PDB and, when the panel shares a scale, the cage PDB.

    A periodic box is wrapped into its cell, tiled and centred on the tiled cell's
    centre; a free cluster is centred on its mean position. The slab is cut about that
    centre along z, and the cage's eight corners sit at that centre plus or minus half
    of `cage_angstrom`.

    Args:
        spec: One panel from `load_panels`.
        work_dir: Scratch directory for the PDB files.

    Returns:
        The subject PDB path, the cage PDB path (None when `cage_angstrom` is null), and
        the number of atoms drawn.

    Raises:
        ValueError: If the cage does not hold the subject with its spheres on every axis,
            which would let the subject frame itself and break the plate's shared scale.
    """
    structure = boxprep.read_xyz(Path(spec["source"]))
    positions_angstrom = structure.positions_angstrom
    if "density_g_cm3" in spec:
        edge_angstrom = boxprep.box_edge_from_density(
            len(positions_angstrom), spec["density_g_cm3"]
        )
        repeats = tuple(spec.get("tile", (1, 1, 1)))
        positions_angstrom = boxprep.tile(
            boxprep.wrap(positions_angstrom, edge_angstrom), edge_angstrom, repeats
        )
        centre_angstrom = np.array(repeats, dtype=float) * edge_angstrom / 2.0
    else:
        centre_angstrom = positions_angstrom.mean(axis=0)
    thickness_angstrom = spec.get("slab_angstrom")
    if thickness_angstrom:
        in_slab = boxprep.slab_mask(
            positions_angstrom[:, 2], centre_angstrom[2], thickness_angstrom
        )
        positions_angstrom = positions_angstrom[in_slab]
    subject = work_dir / f"{spec['id']}.pdb"
    boxprep.write_pdb(subject, positions_angstrom, boxprep.find_bonds(positions_angstrom))
    extent_angstrom = spec.get("cage_angstrom")
    if not extent_angstrom:
        return subject, None, len(positions_angstrom)
    half_extent_angstrom = np.asarray(extent_angstrom, dtype=float) / 2.0
    reach_angstrom = np.abs(positions_angstrom - centre_angstrom).max(axis=0)
    needed_angstrom = 2.0 * (reach_angstrom + float(spec["sphere_radius_angstrom"]))
    if (needed_angstrom > 2.0 * half_extent_angstrom).any():
        raise ValueError(
            f"panel {spec['id']}: subject with spheres needs a cage of "
            f"{np.round(needed_angstrom, 2).tolist()} A about its centre, "
            f"cage_angstrom is {list(extent_angstrom)}"
        )
    signs = np.array([(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    cage = work_dir / f"{spec['id']}-cage.pdb"
    boxprep.write_pdb(cage, centre_angstrom + half_extent_angstrom * signs, [])
    return subject, cage, len(positions_angstrom)


def scene_for(spec: dict, subject: Path, cage: Path | None, draft: bool) -> Scene:
    """The molrender scene for one panel under the section 6.2 constraints.

    Args:
        spec: One panel from `load_panels`; radii in angstrom, grey linear RGB in 0-1.
        subject: PDB of the atoms to draw.
        cage: PDB of the hidden cage that fixes the scale, or None to fit the subject.
        draft: Use the DRAFT tier (EEVEE, half resolution) instead of PRODUCTION.

    Returns:
        A scene with view transform Standard, opaque film and no shadow catcher.
    """
    grey = float(spec["grey"])
    layer = Layer(
        style=BallAndStick(
            sphere_radius=float(spec["sphere_radius_angstrom"]) / CARBON_VDW_RADIUS_ANGSTROM,
            bond_radius=float(spec["bond_radius_angstrom"]),
            sphere_geometry="Point",
        ),
        material=Solid(rgb=(grey, grey, grey)),
    )
    entities = [Structure(path=subject, name="subject", layers=(layer,), remove_solvent=False)]
    if cage is not None:
        entities.append(
            Structure(path=cage, name="cage", layers=(layer,), remove_solvent=False, hide=True)
        )
    tier = DRAFT if draft else PRODUCTION
    return Scene(
        entities=tuple(entities),
        view=View(direction=tuple(spec["direction"]), margin=float(spec.get("margin", 1.02))),
        quality=replace(tier, view_transform="Standard", look=None),
        resolution=tuple(spec["resolution"]),
        transparent=False,
        shadow_catcher=False,
    )


def check_and_flatten(
    png: Path, expected_size: tuple[int, int]
) -> list[tuple[int, int, list[int]]]:
    """Assert size, a white world and near-zero chroma, then save as 8-bit grey.

    The world is judged by blocks, not single pixels, because the Molecular Nodes
    compositor can leave a stray pixel on the border (module docstring): the median of
    each CORNER_BLOCK_PX square corner block must be 255 on every channel, and at most
    MAX_BORDER_STRAYS pixels on the outer one-pixel border may be non-white, each with no
    non-white pixel among its eight neighbours. A sphere reaching the border is a run of
    non-white pixels and still fails.

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


def render_panel(spec: dict, work_dir: Path, draft: bool) -> None:
    """Render one panel to `<output_dir>/<id>.png` and print one summary line.

    Args:
        spec: One panel from `load_panels`.
        work_dir: Scratch directory for the PDB files.
        draft: Use the DRAFT tier; the expected size is then half the resolution.

    Raises:
        ValueError: From `prepare` when the cage does not hold the subject.
        RuntimeError: From `check_and_flatten` when the PNG breaks a section 6.2 rule;
            nothing is then written to `<output_dir>/<id>.png`.
        molrender.RenderError: If Blender fails, times out or writes nothing.
    """
    output = Path(spec["output_dir"]) / f"{spec['id']}.png"
    started_s = time.perf_counter()
    subject, cage, n_atoms = prepare(spec, work_dir)
    rendered = work_dir / output.name
    result = render(scene_for(spec, subject, cage, draft), rendered, timeout=RENDER_TIMEOUT_S)
    scale = 0.5 if draft else 1.0
    width_px, height_px = spec["resolution"]
    strays = check_and_flatten(rendered, (int(width_px * scale), int(height_px * scale)))
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(rendered, output)
    elapsed_s = time.perf_counter() - started_s
    print(
        f"{spec['id']}: {n_atoms} atoms, wall {elapsed_s:.0f} s, "
        f"blender exit {result.returncode}, {output}, "
        f"border strays (row, column, rgb): {strays or 'none'}"
    )


def main(argv: list[str] | None = None) -> int:
    """Render every panel of the manifest, or the `--only` subset, skipping existing PNGs.

    Args:
        argv: Command-line arguments; None reads `sys.argv`.

    Returns:
        Process exit status, 0 on success.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--only", nargs="+", default=None, help="Panel ids to render")
    parser.add_argument("--draft", action="store_true", help="DRAFT tier, for framing checks only")
    parser.add_argument("--force", action="store_true", help="Re-render panels whose PNG exists")
    args = parser.parse_args(argv)

    panels = load_panels(args.manifest)
    if args.only:
        unknown = set(args.only) - {panel["id"] for panel in panels}
        if unknown:
            parser.error(f"unknown panel ids: {sorted(unknown)}")
        panels = [panel for panel in panels if panel["id"] in args.only]
    with tempfile.TemporaryDirectory(prefix="render-box-grid-") as scratch:
        if "BLENDER_USER_SCRIPTS" not in os.environ:
            user_scripts = Path(scratch) / "blender-user-scripts"
            user_scripts.mkdir()
            os.environ["BLENDER_USER_SCRIPTS"] = str(user_scripts)
        for spec in panels:
            output = Path(spec["output_dir"]) / f"{spec['id']}.png"
            if output.exists() and not args.force:
                print(f"{spec['id']}: exists, skipped (use --force)")
                continue
            render_panel(spec, Path(scratch), args.draft)
    return 0


if __name__ == "__main__":
    sys.exit(main())
