"""Render periodic carbon boxes and free clusters as PNG panels through molrender.

A frame of one species is greyscale; a frame of more than one is coloured from the species
palette (`constants.SPECIES_COLOUR`), the same colours the SVG renderer reads from the
`--species-*` tokens.

One PNG per panel. The page lays panels out with `src/components/Grid.astro`, so labels
are real text in the site's mono face rather than pixels (restructure spec section 6.2, 7).

This module imports `molrender` at the top, which lives only in the environment named
below, not the project `.venv` — there is no PEP 723 header to run it with `uv run`,
because molrender is not on PyPI and `uv` would fetch an unrelated package of that name.
The command below, with the molrender Python, is the only way to run this script. The
checks that do not need molrender (manifest loading, the cage containment guard, the
rendered-PNG whiteness check) live in `panel_checks.py` instead, so they can still be
imported and tested from `.venv`.

    # MOLRENDER_PYTHON (required): the python of the environment that has molrender
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
                           Null fits the camera to the subject and, when the resolution is
                           square, crops the flattened PNG to a square round its ink plus 3%
                           a side (panel_checks.crop_to_ink, which gives up margin, never
                           ink); an uncaged non-square panel (the hero) and a caged panel are
                           never cropped. Any plate-level key, resolution included, may be
                           overridden per panel.
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
                           divided by constants.CARBON_VDW_RADIUS_ANGSTROM (1.70 A, Molecular
                           Nodes' carbon, so each species is drawn at that factor of its own van
                           der Waals radius). bond_radius is a length.
    grey                   the material's linear RGB value, 0-1, on all three channels (what
                           Solid(rgb=...) receives); not a display value, since the Standard
                           view transform applies the sRGB curve on output. Used for a frame
                           of one species (and the hidden cage); a frame of more than one is
                           coloured instead.
    colour                 optional {"O": "#rrggbb", ...}: sRGB hex per element symbol, which
                           beats constants.SPECIES_COLOUR for that species on every panel of the
                           plate (a panel's own map replaces the plate's whole map). A frame of
                           more than one species draws a Sticks layer over every atom in the
                           darkest colour present (by relative luminance) and one Spheres
                           layer per species in its own colour, each converted to linear RGB
                           with constants.srgb_hex_to_linear; a species with no colour in
                           either place is refused by name before Blender starts.
    panels[].id, .source   an XYZ file readable by boxprep.read_xyz
    panels[].density_g_cm3 present for a periodic box: the cell edge follows from it and the
                           frame is wrapped; absent for a free cluster
    panels[].tile          [nx, ny, nz] periodic images (boxes only), default [1, 1, 1]

Every panel must end up with each key in panel_checks.REQUIRED_KEYS, from the plate or its
own entry, and no key outside panel_checks.ALLOWED_KEYS (a typo such as "slab_angstorm"
would otherwise fall back to a default rather than failing); ids are unique and use only
letters, digits, "_", "." and "-", because each names a file. Values are checked too
(panel_checks.VALUE_RULES): resolution two positive integers, grey in [0, 1], cage null or
three positive extents, density positive, tile three positive integers, colour a map of
element symbol to "#rrggbb", every number finite, and a source that exists, relative to the
repository root where this command runs. `panel_checks.load_panels` enforces all of this,
naming the manifest, panel and key of the first value it refuses.

Rules this script enforces (spec section 6.2): production tier with the resolution named;
view transform Standard, never AgX; transparent film and shadow catcher off together; every
corner white (the median of each panel_checks.CORNER_BLOCK_PX square corner block is 255 on
every channel); at most panel_checks.MAX_BORDER_STRAYS non-white pixels on the outer
one-pixel border, each isolated, and reported in the summary line; residual chroma at most
panel_checks.MAX_CHROMA, over the whole frame for one species and over the
CORNER_BLOCK_PX border only for more than one; output saved as 8-bit greyscale, or as RGB
for more than one species; the colour path in `scene_for` is exercised only through a
stand-in molrender (the first multi-species draft render is its real check).
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
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import boxprep
import numpy as np
import panel_checks
from constants import (
    CARBON_VDW_RADIUS_ANGSTROM,
    RELATIVE_LUMINANCE_WEIGHTS,
    SPECIES_COLOUR,
    srgb_hex_to_linear,
)
from molrender import (
    DRAFT,
    PRODUCTION,
    AtomMask,
    BallAndStick,
    Layer,
    Scene,
    Solid,
    Spheres,
    Sticks,
    Structure,
    StructureId,
    View,
    find_blender,
    render,
)

RENDER_TIMEOUT_S = 3600


@cache
def tool_versions() -> str:
    """Blender, Molecular Nodes and molrender versions, for every render's summary line.

    This script runs under the molrender interpreter, which cannot `import bpy` (the `bpy`
    wheel is cp313-only and molrender's own README says its environment has none), so
    Blender's version comes from asking the executable molrender itself would start
    (`molrender.find_blender`: `BLENDER_PATH`, the macOS bundle, then `PATH`). molrender
    records neither where Molecular Nodes is installed nor its version (the add-on is
    imported only inside Blender, and its version sits in a `blender_manifest.toml` whose
    location molrender never stores), so that one reads "unknown" until molrender exposes it.
    Cached: Blender starts once per run, not once per panel.

    Returns:
        "Blender <version>, Molecular Nodes <version>, molrender <version>"; any version
        that cannot be found reads "unknown".
    """
    try:
        result = subprocess.run(
            [str(find_blender()), "--version"], capture_output=True, text=True, check=False
        )
        blender_version = (
            result.stdout.splitlines()[0].removeprefix("Blender ")
            if result.returncode == 0 and result.stdout
            else "unknown"
        )
    except OSError:  # no Blender located (FileNotFoundError), or it would not start
        blender_version = "unknown"
    try:
        molrender_version = version("molrender")
    except PackageNotFoundError:
        molrender_version = "unknown"
    return f"Blender {blender_version}, Molecular Nodes unknown, molrender {molrender_version}"


def prepare(spec: dict, work_dir: Path) -> tuple[Path, Path | None, list[str]]:
    """Write the subject PDB and, when the panel shares a scale, the cage PDB.

    A periodic box is wrapped into its cell, tiled and centred on the tiled cell's
    centre; a free cluster is centred on its mean position. The slab is cut about that
    centre along z, and the cage's eight corners sit at that centre plus or minus half
    of `cage_angstrom`.

    Args:
        spec: One panel from `panel_checks.load_panels`.
        work_dir: Scratch directory for the PDB files.

    Returns:
        The subject PDB path, the cage PDB path (None when `cage_angstrom` is null), and
        the element symbol of every atom drawn, in the order the subject PDB lists them.

    Raises:
        ValueError: If a species has no entry in constants.COVALENT_RADIUS_ANGSTROM
            (`boxprep.read_xyz` names it), or the cage does not hold the subject
            with its spheres on every axis, which would let the subject frame itself and
            break the plate's shared scale (`panel_checks.assert_fits_cage`).
    """
    structure = boxprep.read_xyz(Path(spec["source"]))
    species = np.asarray(structure.species)
    positions_angstrom = structure.positions_angstrom
    if "density_g_cm3" in spec:
        edge_angstrom = boxprep.box_edge_from_density(
            len(positions_angstrom), spec["density_g_cm3"]
        )
        repeats = tuple(spec.get("tile", (1, 1, 1)))
        positions_angstrom = boxprep.tile(
            boxprep.wrap(positions_angstrom, edge_angstrom), edge_angstrom, repeats
        )
        species = np.tile(species, int(np.prod(repeats)))  # boxprep.tile keeps atom order
        centre_angstrom = np.array(repeats, dtype=float) * edge_angstrom / 2.0
    else:
        centre_angstrom = positions_angstrom.mean(axis=0)
    thickness_angstrom = spec.get("slab_angstrom")
    if thickness_angstrom:
        in_slab = boxprep.slab_mask(
            positions_angstrom[:, 2], centre_angstrom[2], thickness_angstrom
        )
        positions_angstrom = positions_angstrom[in_slab]
        species = species[in_slab]
    subject = work_dir / f"{spec['id']}.pdb"
    boxprep.write_pdb(
        subject,
        species.tolist(),
        positions_angstrom,
        boxprep.find_bonds(positions_angstrom, species.tolist()),
    )
    extent_angstrom = spec.get("cage_angstrom")
    if not extent_angstrom:
        return subject, None, species.tolist()
    panel_checks.assert_fits_cage(
        positions_angstrom,
        centre_angstrom,
        extent_angstrom,
        float(spec["sphere_radius_angstrom"]),
        str(spec["id"]),
    )
    half_extent_angstrom = np.asarray(extent_angstrom, dtype=float) / 2.0
    signs = np.array([(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    cage = work_dir / f"{spec['id']}-cage.pdb"
    boxprep.write_pdb(cage, ["C"] * len(signs), centre_angstrom + half_extent_angstrom * signs, [])
    return subject, cage, species.tolist()


def scene_for(
    spec: dict, subject: Path, cage: Path | None, species: list[str], draft: bool
) -> Scene:
    """The molrender scene for one panel under the section 6.2 constraints.

    A subject of one species is one grey `BallAndStick` layer. A subject of more than one
    is a `Sticks` layer over every atom, in the darkest of the colours present, and one
    `Spheres` layer per species, selected by an `AtomMask` against the subject PDB and
    coloured from the manifest's `colour` map, then `constants.SPECIES_COLOUR`.

    Args:
        spec: One panel from `panel_checks.load_panels`; radii in angstrom, grey linear RGB in 0-1.
        subject: PDB of the atoms to draw.
        cage: PDB of the hidden cage that fixes the scale, or None to fit the subject.
        species: Element symbol of every atom in `subject`, in file order.
        draft: Use the DRAFT tier (EEVEE, half resolution) instead of PRODUCTION.

    Returns:
        A scene with view transform Standard, opaque film and no shadow catcher.

    Raises:
        ValueError: If the subject has more than one species and one has a colour neither in
            the panel's `colour` map nor in `constants.SPECIES_COLOUR`.
    """
    grey = float(spec["grey"])
    sphere_radius = float(spec["sphere_radius_angstrom"]) / CARBON_VDW_RADIUS_ANGSTROM
    grey_layer = Layer(
        style=BallAndStick(
            sphere_radius=sphere_radius,
            bond_radius=float(spec["bond_radius_angstrom"]),
            sphere_geometry="Point",
        ),
        material=Solid(rgb=(grey, grey, grey)),
    )
    symbols = sorted(set(species))
    if len(symbols) == 1:
        subject_layers = (grey_layer,)
    else:
        palette = {**SPECIES_COLOUR, **spec.get("colour", {})}
        unpainted = [symbol for symbol in symbols if symbol not in palette]
        if unpainted:
            raise ValueError(
                f"panel {spec['id']}: no colour for {unpainted}; add it to "
                "constants.SPECIES_COLOUR or to the manifest's colour map"
            )
        rgb = {symbol: srgb_hex_to_linear(palette[symbol]) for symbol in symbols}
        atom_symbols = np.asarray(species)
        subject_id = StructureId(
            sha256=hashlib.sha256(subject.read_bytes()).hexdigest(), n_atoms=len(species)
        )
        darkest = min(
            rgb.values(), key=lambda channels: np.dot(RELATIVE_LUMINANCE_WEIGHTS, channels)
        )
        subject_layers = (
            Layer(
                style=Sticks(radius=float(spec["bond_radius_angstrom"])),
                material=Solid(rgb=darkest),
            ),
            *(
                Layer(
                    style=Spheres(radius=sphere_radius, geometry="Point"),
                    material=Solid(rgb=rgb[symbol]),
                    select=AtomMask(mask=tuple((atom_symbols == symbol).tolist()), of=subject_id),
                )
                for symbol in symbols
            ),
        )
    entities = [
        Structure(path=subject, name="subject", layers=subject_layers, remove_solvent=False)
    ]
    if cage is not None:
        entities.append(
            Structure(path=cage, name="cage", layers=(grey_layer,), remove_solvent=False, hide=True)
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


def render_panel(spec: dict, work_dir: Path, draft: bool) -> None:
    """Render one panel to `<output_dir>/<id>.png` and print one summary line.

    Args:
        spec: One panel from `panel_checks.load_panels`.
        work_dir: Scratch directory for the PDB files.
        draft: Use the DRAFT tier; the expected size is then half the resolution.

    Raises:
        ValueError: From `prepare` when a species has no covalent radius or the cage does
            not hold the subject, or from `scene_for` when a species has no colour.
        RuntimeError: From `panel_checks.check_and_flatten` when the PNG breaks a
            section 6.2 rule, or from `panel_checks.crop_to_ink` when an uncaged square
            panel has no ink; nothing is then written to
            `<output_dir>/<id>.png`.
        molrender.RenderError: If Blender fails, times out or writes nothing.
    """
    output = Path(spec["output_dir"]) / f"{spec['id']}.png"
    started_s = time.perf_counter()
    subject, cage, species = prepare(spec, work_dir)
    rendered = work_dir / output.name
    result = render(
        scene_for(spec, subject, cage, species, draft), rendered, timeout=RENDER_TIMEOUT_S
    )
    scale = 0.5 if draft else 1.0
    width_px, height_px = spec["resolution"]
    strays = panel_checks.check_and_flatten(
        rendered,
        (int(width_px * scale), int(height_px * scale)),
        label=f"{spec['id']} ({output})",
        multi_species=len(set(species)) > 1,
    )
    # A panel with no cage is framed to its own subject, so nothing is lost by cropping
    # its empty ground away to a square; a panel with a cage shares a scale and keeps its
    # frame, and a non-square frame (the hero) is composed as it is and keeps it too.
    if spec.get("cage_angstrom"):
        crop = "none (caged)"
    elif width_px != height_px:
        crop = f"skipped (uncaged, but {width_px} x {height_px} is not square)"
    else:
        crop = f"{panel_checks.crop_to_ink(rendered)} px"
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(rendered, output)
    elapsed_s = time.perf_counter() - started_s
    print(
        f"{spec['id']}: {len(species)} atoms, wall {elapsed_s:.0f} s, "
        f"blender exit {result.returncode}, {output}, "
        f"crop {crop}, "
        f"border strays (row, column, rgb): {strays or 'none'}, "
        f"versions: {tool_versions()}"
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

    panels = panel_checks.load_panels(args.manifest)
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
