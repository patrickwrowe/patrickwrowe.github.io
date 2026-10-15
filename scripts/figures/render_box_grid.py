"""Render periodic carbon boxes and free clusters as PNG panels through molrender.

A frame of one species is greyscale; a frame of more than one is coloured from the species
palette (`constants.SPECIES_COLOUR`), the same colours the SVG renderer reads from the
`--species-*` tokens, and a protein's cartoon takes one colour per chain
(`constants.CHAIN_COLOUR`, the `--chain-*` tokens). Every material is the matte
`constants.MATTE`, lit by the rig in `constants.RIG` and `constants.WORLD_STRENGTH`: molrender's
three-point key at half power, a sun fill just above the camera axis, a side light
perpendicular to the view and the studio HDRI (candidate c3-three-point-plus, chosen
2026-10-15).

One PNG per panel. The page lays panels out with `src/components/Grid.astro`, so labels
are real text in the site's mono face rather than pixels (restructure spec section 6.2, 7).

This module imports `molrender` at the top, which lives only in the environment named
below, not the project `.venv` — there is no PEP 723 header to run it with `uv run`,
because molrender is not on PyPI and `uv` would fetch an unrelated package of that name.
The command below, with the molrender Python, is the only way to run this script. It needs
molrender at commit 081ded2 or later on the branch `fix-masked-layer-collision` of the local
checkout (an opt-in `Light.sun_strength` and `side_power`, and a camera fitted after a
view-layer update); importing this module against a molrender whose `Light` lacks a field of
`constants.RIG` (molrender `main`) raises `ImportError` naming that commit. A summary line
names the commit molrender ran from when it is a git checkout (`tool_versions`). The checks
that do not need molrender (manifest loading, the cage containment guard, the rendered-PNG
whiteness check, whether a panel is coloured) live in `panel_checks.py` instead, so they can
still be imported and tested from `.venv`.

    # MOLRENDER_PYTHON (required): the python of the environment that has molrender
    $MOLRENDER_PYTHON scripts/figures/render_box_grid.py \
        --manifest scripts/figures/data/graphitisation/manifest.json \
        [--only ID ...] [--draft] [--force]

Molecular Nodes copies its startup template into Blender's user-scripts directory on every
render (molecularnodes scene/base.py, Canvas.__init__), and ~/.config is read-only inside a
sandboxed session. So when BLENDER_USER_SCRIPTS is unset the driver points it at a fresh
directory inside the run's scratch directory before the first render, and unsets it again when
the run ends; the same command then works inside and outside the sandbox. Fresh per run
because Blender executes any Python under <dir>/startup/. It holds only the template;
extensions and preferences still load from ~/.config.

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
                           divided by carbon's in constants.VDW_RADIUS_ANGSTROM (1.70 A,
                           Molecular Nodes' own), so each species is drawn at that factor of
                           its own van der Waals radius, the radius the cage guard allows it.
                           bond_radius is a length.
    grey                   box panels only, required: the material's linear RGB value, 0-1, on
                           all three channels (what Solid(rgb=...) receives); not a display
                           value, since the Standard view transform applies the sRGB curve on
                           output. Used for a frame of one species (and the hidden cage); a
                           frame of more than one is coloured instead.
    colour                 optional {"O": "#rrggbb", ...}: sRGB hex per element symbol, which
                           beats constants.SPECIES_COLOUR for that species on every panel of the
                           plate (a panel's own map replaces the plate's whole map). A frame of
                           more than one species draws a Sticks layer over every atom in the
                           palette's carbon (a protein's cartoons take `chain_colour`
                           instead) and one Spheres layer per species in its own colour, each
                           converted to linear RGB with constants.srgb_hex_to_linear; a
                           species with no colour in either place is refused by name before
                           Blender starts.
    panels[].id, .source   an XYZ file readable by boxprep.read_xyz; for a protein panel, a PDB file
    panels[].kind          "box" (the default: periodic boxes and free clusters, as above) or
                           "protein": a PDB file drawn as it is, never moved, tiled, caged or
                           sliced (cage_angstrom, slab_angstrom, density_g_cm3 and tile must
                           be null or absent), framed to the complex with `margin` and
                           `direction`. boxprep.read_pdb reads model 1, one conformer
                           (altloc blank or A) and no water; a protein is one Cartoon layer
                           per chain of `cartoon_chains`, coloured from `chain_colour`, then
                           constants.CHAIN_COLOUR in list order, and every atom of
                           `stick_chains` and every
                           HETATM drawn as a multi-species frame is (the sticks and
                           spheres above), its bonds written as CONECT records by `prepare`
                           because Molecular Nodes bonds no ligand of an older file, so the
                           file's chains must be exactly the two lists together. A hetero
                           atom of a residue of more than one atom must have a covalent
                           radius, or the panel is refused naming it; a lone ion of a metal
                           in constants.METALS that bonds to no other hetero atom is not
                           drawn. sphere_radius_angstrom sets the spheres and
                           bond_radius_angstrom the sticks; a protein has no `grey`
    panels[].cartoon_chains
                           protein only, required: a non-empty list of auth chain ids (PDB
                           column 22) drawn as Cartoon
    panels[].stick_chains  protein only, required: a list of auth chain ids drawn
                           species-coloured as sticks and spheres; [] for none, disjoint from
                           cartoon_chains
    chain_colour           protein only, optional {"A": "#rrggbb", ...}: sRGB hex per chain
                           of cartoon_chains, which beats constants.CHAIN_COLOUR for that
                           chain; a protein of more cartoon chains than CHAIN_COLOUR holds
                           must name the rest here
    panels[].density_g_cm3 present for a periodic box: the cell edge follows from it and the
                           frame is wrapped; absent for a free cluster
    panels[].tile          [nx, ny, nz] periodic images (boxes only), default [1, 1, 1]

Every panel must end up with each key in panel_checks.REQUIRED_KEYS, from the plate or its
own entry, and no key outside panel_checks.ALLOWED_KEYS (a typo such as "slab_angstorm"
would otherwise fall back to a default rather than failing); ids are unique and use only
letters, digits, "_", "." and "-", because each names a file. Values are checked too
(panel_checks.VALUE_RULES): resolution two positive integers, grey in [0, 1], cage null or
three positive extents, density positive, tile three positive integers, colour a map of
element symbol to "#rrggbb", kind "box" or "protein", the chain lists of one-character ids,
every number finite, and a source that exists, relative to the repository root where this
command runs; a box panel also needs a grey, and a protein panel both chain lists (the
cartoon list not empty, the two disjoint), a .pdb source and no cage, slab, density or tile.
`panel_checks.load_panels` enforces all of this, naming the manifest, panel and key of the
first value it refuses, and sets every panel's `kind`.

Rules this script enforces (spec section 6.2): production tier with the resolution named;
view transform Standard, never AgX; transparent film and shadow catcher off together; every
corner white (the median of each panel_checks.CORNER_BLOCK_PX square corner block is 255 on
every channel); at most panel_checks.MAX_BORDER_STRAYS non-white pixels on the outer
one-pixel border, each isolated, and reported in the summary line; residual chroma at most
panel_checks.MAX_CHROMA, over the whole frame for a grey panel (a box of one species) and
over the CORNER_BLOCK_PX border only for a coloured one (a box of more than one species, or
any protein, whose chains are coloured); output saved as 8-bit greyscale, or as RGB when
coloured. The colour path in `scene_for` is
exercised without Blender only through a stand-in molrender, which cannot show what Blender
draws: the species spheres were Points that all took one colour until the first real
multi-species render (6TRO, 2026-10-09) showed it, so a change to that path is checked on the
render's pixels.
The species spheres are therefore Instances, and that has a cost: molrender measured Instance
spheres about 3x slower than Point and advises Point "for a whole solvated system"
(molrender scene/styles.py, BallAndStick). A multi-species frame of thousands of atoms pays
it, and a panel has no switch back to Point (only one-species spheres are Points). Nothing has
yet compared the visible size of an Instance sphere with a Point sphere of the same
`sphere_radius_angstrom`, which matters when a multi-species panel sits beside a one-species
one on a plate; the first large multi-species draft should record its wall time and that
comparison.
Each panel renders into the scratch directory and moves to `<id>.png` only once every check
passes, so a failed panel leaves nothing behind for the resume run to skip.
`--draft` uses molrender's DRAFT tier for framing checks only: EEVEE at half resolution, and
the size assertion is relaxed to match. EEVEE draws Point spheres (the one-species grey layer
and the hidden cage) far smaller than Cycles does; nobody has compared the Instance species
spheres draft against production. Either way, judge radii at the production tier.

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
from collections import Counter
from dataclasses import fields, replace
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import boxprep
import molrender
import numpy as np
import panel_checks
from constants import (
    CHAIN_COLOUR,
    COVALENT_RADIUS_ANGSTROM,
    MATTE,
    METALS,
    RIG,
    SPECIES_COLOUR,
    VDW_RADIUS_ANGSTROM,
    WORLD_STRENGTH,
    srgb_hex_to_linear,
)
from molrender import (
    ALL,
    DRAFT,
    PRODUCTION,
    AtomMask,
    BallAndStick,
    Cartoon,
    Layer,
    Light,
    Scene,
    Solid,
    Spheres,
    Sticks,
    Structure,
    StructureId,
    View,
    World,
    find_blender,
    render,
)
from panel_checks import PanelKind

# RIG needs a Light with the opt-in sun fill and side light, which only molrender at commit
# 081ded2 or later on its branch fix-masked-layer-collision has: refuse any other molrender
# here, before a manifest is read, not with a TypeError at the first Light(...).
if missing_light_fields := sorted(set(RIG) - {field.name for field in fields(Light)}):
    raise ImportError(
        f"molrender's Light has no {missing_light_fields}: render_box_grid needs molrender at "
        "commit 081ded2 or later on the branch fix-masked-layer-collision"
    )

RENDER_TIMEOUT_S = 3600
# Run inside the one Blender start `tool_versions` makes: Blender's version and Molecular
# Nodes', read from the extension's own manifest under Blender's user extensions directory.
VERSIONS_EXPR = (
    "import bpy, pathlib, tomllib\n"
    "manifests = sorted(pathlib.Path(bpy.utils.user_resource('EXTENSIONS'))"
    ".glob('*/molecularnodes/blender_manifest.toml'))\n"
    "print('TOOL_VERSIONS', bpy.app.version_string, "
    "tomllib.loads(manifests[0].read_text())['version'] if manifests else 'unknown')"
)


@cache
def tool_versions() -> str:
    """Blender, Molecular Nodes and molrender versions, for every render's summary line.

    This script runs under the molrender interpreter, which cannot `import bpy` (the `bpy`
    wheel is cp313-only and molrender's own README says its environment has none), so
    Blender's version and Molecular Nodes' come from one background start of the executable
    molrender itself would start (`molrender.find_blender`: `BLENDER_PATH`, the macOS bundle,
    then `PATH`), running VERSIONS_EXPR: `bpy.app.version_string`, and the `version` of the
    `blender_manifest.toml` of the molecularnodes extension under Blender's user extensions
    directory, "unknown" when it has none.
    molrender's version alone cannot tell a release from the unmerged branch the renders need,
    so when the package's file lives in a git checkout the version carries that checkout's
    short SHA, as "0.1.0 (081ded2)", with "-dirty" appended when the checkout has uncommitted
    changes to tracked files, since a SHA alone would then name code that did not render; a
    failed `git` call leaves the plain version.
    Cached: Blender starts once per run, not once per panel.

    Returns:
        "Blender <version>, Molecular Nodes <version>, molrender <version>"; any version
        that cannot be found reads "unknown".
    """
    blender_version = molecular_nodes_version = "unknown"
    try:
        result = subprocess.run(
            [
                str(find_blender()),
                "--background",
                "--factory-startup",
                "--python-expr",
                VERSIONS_EXPR,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        reported = next(
            (
                line.split()[1:]
                for line in result.stdout.splitlines()
                if line.startswith("TOOL_VERSIONS ")
            ),
            [],
        )
        if result.returncode == 0 and len(reported) == 2:
            blender_version, molecular_nodes_version = reported
    except OSError:  # no Blender located (FileNotFoundError), or it would not start
        pass
    try:
        molrender_version = version("molrender")
    except PackageNotFoundError:
        molrender_version = "unknown"
    try:
        checkout_sha = subprocess.run(
            ["git", "-C", str(Path(molrender.__file__).parent), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            [
                "git",
                "-C",
                str(Path(molrender.__file__).parent),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        molrender_version += f" ({checkout_sha}{'-dirty' if dirty else ''})"
    except (OSError, subprocess.CalledProcessError):  # no git, or not a checkout: plain version
        pass
    return (
        f"Blender {blender_version}, Molecular Nodes {molecular_nodes_version}, "
        f"molrender {molrender_version}"
    )


def prepare(spec: dict, work_dir: Path) -> tuple[Path, Path | None, list[str]]:
    """Write the subject PDB and, when the panel shares a scale, the cage PDB.

    A periodic box is wrapped into its cell, tiled and centred on the tiled cell's
    centre; a free cluster is centred on its mean position. The slab is cut about that
    centre along z, and the cage's eight corners sit at that centre plus or minus half
    of `cage_angstrom`.

    A protein panel (`kind` "protein") is not moved, tiled or caged. The subject file is
    the atom records `boxprep.read_pdb` kept (model 1, one conformer, no water), verbatim and
    in the file's order, after the source's HELIX and SHEET records, which Molecular Nodes
    reads for the cartoon: the chain masks `scene_for` measures on this file are then
    indexed by the atoms Molecular Nodes loads from it. It ends with a CONECT record for each
    bond `boxprep.find_bonds` finds among the HETATM atoms whose element has a covalent
    radius, using the file's own serial numbers: Molecular Nodes would otherwise draw a ligand
    as unbonded spheres. The source's own CONECT records are not copied. A lone ion of a metal
    (a one-atom HETATM residue whose element is in constants.METALS) that bonds to no other
    hetero atom is left out of the file: it is an ion of the crystal, such as 3BSD's Mg 3262
    at occupancy 0.33, whose partners were waters, and would draw as a loose sphere. A lone
    ion with no covalent radius (zinc) is drawn as it is, unbonded.

    Args:
        spec: One panel from `panel_checks.load_panels`.
        work_dir: Scratch directory for the PDB files.

    Returns:
        The subject PDB path, the cage PDB path (None when `cage_angstrom` is null), and
        the element symbol of every atom drawn, in the order the subject PDB lists them.

    Raises:
        ValueError: If a species has no entry in constants.COVALENT_RADIUS_ANGSTROM
            (`boxprep.read_xyz` names it), the frame has no atoms left after the slab is
            cut (the panel is named), or the cage does not hold the subject
            with its spheres on every axis, which would let the subject frame itself and
            break the plate's shared scale (`panel_checks.assert_fits_cage`); for a protein,
            if the file's chains are not exactly the chains the panel names in
            `cartoon_chains` and `stick_chains`, so that no chain is left undrawn by a typo,
            if a hetero atom with no covalent radius belongs to a residue of more than one
            atom (a phosphate's P, a drug's Cl), whose bonds could then not be written, or
            if an atom serial is not a number.
    """
    if spec["kind"] == PanelKind.PROTEIN:
        protein = boxprep.read_pdb(Path(spec["source"]))
        chains = set(protein.chain_ids.tolist())
        named = {*spec["cartoon_chains"], *spec["stick_chains"]}
        if chains != named:
            raise ValueError(
                f"panel {spec['id']}: {spec['source']} has chains {sorted(chains)} but the "
                f"panel names {sorted(named)} in cartoon_chains and stick_chains"
            )
        # Molecular Nodes bonds a residue from biotite's templates, which know the twenty amino
        # acids by their modern atom names and nothing of a ligand in an older file: the eight
        # BCL of 3BSD came out as 504 unbonded spheres (2026-10-09). So every bond the radius
        # rule finds among the hetero atoms is written as a CONECT record, which it reads. An
        # atom whose element has no covalent radius is left unbonded, which is right only for
        # a lone ion (zinc): in a ligand of more atoms it would be a loose sphere, so refuse it.
        residue_of = [record[17:27] for record in protein.records]  # name, chain, number, code
        atoms_in_residue = Counter(residue_of)
        has_radius = np.isin(protein.species, list(COVALENT_RADIUS_ANGSTROM))
        if in_ligand := sorted(
            {
                f"{protein.species[atom]} in {' '.join(residue_of[atom].split())}"
                for atom in np.flatnonzero(protein.is_hetero & ~has_radius)
                if atoms_in_residue[residue_of[atom]] > 1
            }
        ):
            raise ValueError(
                f"panel {spec['id']}: {spec['source']}: no covalent radius for {in_ligand}, so "
                "their bonds cannot be written and they would draw as loose spheres; add the "
                "species to constants.COVALENT_RADIUS_ANGSTROM"
            )
        hetero = np.flatnonzero(protein.is_hetero & has_radius)
        hetero_bonds = [
            (int(hetero[first]), int(hetero[second]))
            for first, second in boxprep.find_bonds(
                protein.positions_angstrom[hetero], [protein.species[atom] for atom in hetero]
            )
        ]
        # ponytail: bonds are looked for among hetero atoms only, so a metal held by protein
        # side chains alone (a structural Mg) would be dropped too; bond METALS against every
        # atom if a panel ever needs one.
        drawn = ~(
            protein.is_hetero
            & (np.array([atoms_in_residue[residue] for residue in residue_of]) == 1)
            & np.isin(protein.species, list(METALS))
            & ~np.isin(np.arange(len(protein)), hetero_bonds)
        )
        try:
            serials = [int(record[6:11]) for record in protein.records]
        except ValueError as error:
            raise ValueError(
                f"panel {spec['id']}: {spec['source']} has an atom serial (columns 7-11) that is "
                "not a number, which a CONECT record needs"
            ) from error
        subject = work_dir / f"{spec['id']}.pdb"
        subject.write_text(
            "\n".join(
                [
                    *protein.secondary_structure_records,
                    *(record for record, keep in zip(protein.records, drawn, strict=True) if keep),
                    *boxprep.conect_records(hetero_bonds, serials),
                    "END",
                ]
            )
            + "\n"
        )
        return subject, None, np.asarray(protein.species)[drawn].tolist()
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
    if species.size == 0:
        raise ValueError(
            f"panel {spec['id']}: no atoms to draw from {spec['source']} "
            f"(slab_angstrom {thickness_angstrom})"
        )
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
        species.tolist(),
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
    spec: dict,
    subject: Path,
    cage: Path | None,
    species: list[str],
    *,
    draft: bool,
) -> Scene:
    """The molrender scene for one panel under the section 6.2 constraints.

    Every layer's material is `Solid` with `constants.MATTE`, and the scene's light is
    `Light(**constants.RIG, world=World(strength=constants.WORLD_STRENGTH))`: the three-point
    key at 2 W, a sun fill, a side light and the studio HDRI, with the backdrop still pure
    white. Whether the panel is coloured is `panel_checks.is_coloured`, the rule
    `render_panel` saves the PNG by. A box of one species is one grey `BallAndStick` layer. A
    subject of more than one is a
    `Sticks` layer over every atom, in the palette's carbon,
    and one `Spheres` layer per species, selected by an `AtomMask`
    against the subject PDB and coloured from the manifest's `colour` map, then
    `constants.SPECIES_COLOUR`. The species spheres are Instances, not Points: a Blender
    point cloud carries one material, so Point spheres in several layers all took the last
    layer's colour (and dropped an atom), found on the first real multi-species render,
    2026-10-09, of 6TRO's peptide. The one-material grey layer keeps Point. The sticks were
    once the darkest colour present, which with magnesium drawn was magnesium's own dark green,
    so the sphere at the centre of each bacteriochlorophyll could not be told from its bonds
    (the excited-states hero, 2026-10-09); in carbon, the colour of most bonds, a sphere of any
    other species stands out from the sticks it sits on. Cartoons take `CHAIN_COLOUR` per
    chain, never the carbon.

    A protein panel (`kind` "protein") is always coloured: one `Cartoon` layer per chain of
    `cartoon_chains`, selected by an `AtomMask` over that chain and coloured from the
    manifest's `chain_colour`, then `constants.CHAIN_COLOUR` in list order, and, over the
    atoms of `stick_chains` and every HETATM (water is already gone), the multi-species
    `Sticks` and per-species `Spheres` layers restricted by an `AtomMask`. With no such
    atoms (the hero) it is the Cartoon layers alone. The masks are measured on `subject`,
    the file `prepare` wrote, by `boxprep.read_pdb`.

    One layer per chain needs a molrender whose camera fit evaluates the scene after its
    last layer: one that does not fits the frame to every layer but the last, which cut the
    hero's last chain off at the top edge (2026-10-09), and the border check refused it.

    Args:
        spec: One panel from `panel_checks.load_panels`; radii in angstrom, a box's grey
            linear RGB in 0-1.
        subject: PDB of the atoms to draw.
        cage: PDB of the hidden cage that fixes the scale, or None to fit the subject.
        species: Element symbol of every atom in `subject`, in file order.
        draft: Use the DRAFT tier (EEVEE, half resolution) instead of PRODUCTION.

    Returns:
        A scene with view transform Standard, the `constants.RIG` light, opaque film and no
        shadow catcher.

    Raises:
        ValueError: If a drawn species has a colour neither in the panel's `colour` map nor
            in `constants.SPECIES_COLOUR`, or a cartoon chain has none in `chain_colour` and
            lies beyond the length of `constants.CHAIN_COLOUR`.
    """
    sphere_radius = float(spec["sphere_radius_angstrom"]) / VDW_RADIUS_ANGSTROM["C"]
    protein = boxprep.read_pdb(subject) if spec["kind"] == PanelKind.PROTEIN else None
    grey_layer = None  # a protein has no grey: it is coloured and never caged
    if protein is None:
        grey = float(spec["grey"])
        grey_layer = Layer(
            style=BallAndStick(
                sphere_radius=sphere_radius,
                bond_radius=float(spec["bond_radius_angstrom"]),
                sphere_geometry="Point",
            ),
            material=Solid(rgb=(grey, grey, grey), **MATTE),
        )
    if not panel_checks.is_coloured(spec, species):
        subject_layers = (grey_layer,)
    else:
        atom_symbols = np.asarray(species)
        subject_id = StructureId(
            sha256=hashlib.sha256(subject.read_bytes()).hexdigest(), n_atoms=len(species)
        )
        if protein is None:
            drawn = np.ones(len(species), dtype=bool)
        else:
            drawn = protein.drawn_by_species(spec["stick_chains"])
        symbols = sorted(set(atom_symbols[drawn].tolist()))
        palette = {**SPECIES_COLOUR, **spec.get("colour", {})}
        unpainted = [symbol for symbol in symbols if symbol not in palette]
        if unpainted:
            raise ValueError(
                f"panel {spec['id']}: no colour for {unpainted}; add it to "
                "constants.SPECIES_COLOUR or to the manifest's colour map"
            )
        rgb = {symbol: srgb_hex_to_linear(palette[symbol]) for symbol in symbols}
        carbon = srgb_hex_to_linear(palette["C"])
        layers = []
        if protein is not None:
            chain_colour = {
                **dict(zip(spec["cartoon_chains"], CHAIN_COLOUR, strict=False)),
                **spec.get("chain_colour", {}),
            }
            uncoloured = [chain for chain in spec["cartoon_chains"] if chain not in chain_colour]
            if uncoloured:
                raise ValueError(
                    f"panel {spec['id']}: no colour for chain(s) {uncoloured}; constants."
                    f"CHAIN_COLOUR holds {len(CHAIN_COLOUR)}, so name them in chain_colour"
                )
            layers.extend(
                Layer(
                    style=Cartoon(),
                    material=Solid(rgb=srgb_hex_to_linear(chain_colour[chain]), **MATTE),
                    select=AtomMask(
                        mask=tuple((protein.chain_ids == chain).tolist()), of=subject_id
                    ),
                )
                for chain in spec["cartoon_chains"]
            )
        if symbols:  # a protein of cartoon chains alone draws no atom by species
            layers.append(
                Layer(
                    style=Sticks(radius=float(spec["bond_radius_angstrom"])),
                    material=Solid(rgb=carbon, **MATTE),
                    select=ALL
                    if protein is None
                    else AtomMask(mask=tuple(drawn.tolist()), of=subject_id),
                )
            )
        layers.extend(
            Layer(
                style=Spheres(radius=sphere_radius, geometry="Instance"),
                material=Solid(rgb=rgb[symbol], **MATTE),
                select=AtomMask(
                    mask=tuple((drawn & (atom_symbols == symbol)).tolist()), of=subject_id
                ),
            )
            for symbol in symbols
        )
        subject_layers = tuple(layers)
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
        light=Light(**RIG, world=World(strength=WORLD_STRENGTH)),
        quality=replace(tier, view_transform="Standard", look=None),
        resolution=tuple(spec["resolution"]),
        transparent=False,
        shadow_catcher=False,
    )


def render_panel(spec: dict, work_dir: Path, draft: bool) -> None:
    """Render one panel to `<output_dir>/<id>.png` and print one summary line.

    The PNG is saved grey and judged over the whole frame when the panel is a box of one
    species, and in colour, judged on its border, when it is a box of more or any protein
    (whose chains are coloured). A protein's summary line also gives the atom records
    `boxprep.read_pdb` dropped as altloc B or later and as water.

    Args:
        spec: One panel from `panel_checks.load_panels`.
        work_dir: Scratch directory for the PDB files.
        draft: Use the DRAFT tier; the expected size is then half the resolution.

    Raises:
        ValueError: From `prepare` when a species has no covalent radius, the frame has no
            atoms left, the cage does not hold the subject, or a protein's chains are not
            the chains the panel names, or from `scene_for` when a species has no colour.
        RuntimeError: From `panel_checks.check_and_flatten` when the PNG breaks a
            section 6.2 rule, or from `panel_checks.crop_to_ink` when an uncaged square
            panel has no ink; nothing is then written to
            `<output_dir>/<id>.png`.
        molrender.RenderError: If Blender fails, times out or writes nothing.
    """
    output = Path(spec["output_dir"]) / f"{spec['id']}.png"
    started_s = time.perf_counter()
    subject, cage, species = prepare(spec, work_dir)
    dropped = ""
    if spec["kind"] == PanelKind.PROTEIN:
        protein = boxprep.read_pdb(Path(spec["source"]))
        dropped = f" (dropped {protein.n_dropped_altloc} altloc, {protein.n_dropped_water} water)"
    rendered = work_dir / output.name
    result = render(
        scene_for(spec, subject, cage, species, draft=draft),
        rendered,
        timeout=RENDER_TIMEOUT_S,
    )
    scale = 0.5 if draft else 1.0
    width_px, height_px = spec["resolution"]
    frame_px = (int(width_px * scale), int(height_px * scale))  # what the tier rendered
    strays = panel_checks.check_and_flatten(
        rendered,
        frame_px,
        label=f"{spec['id']} ({output})",
        coloured=panel_checks.is_coloured(spec, species),
    )
    # A panel with no cage is framed to its own subject, so nothing is lost by cropping
    # its empty ground away to a square; a panel with a cage shares a scale and keeps its
    # frame, and a non-square frame (the hero) is composed as it is and keeps it too.
    if spec.get("cage_angstrom"):
        crop = "none (caged)"
    elif width_px != height_px:
        crop = f"skipped (uncaged, but {frame_px[0]} x {frame_px[1]} is not square)"
    else:
        crop = f"{panel_checks.crop_to_ink(rendered)} px"
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(rendered, output)
    elapsed_s = time.perf_counter() - started_s
    print(
        f"{spec['id']}: {len(species)} atoms{dropped}, wall {elapsed_s:.0f} s, "
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
        # Set for this run only: after it the directory is gone, and a dead path would reach
        # the next Blender this process starts.
        set_here = "BLENDER_USER_SCRIPTS" not in os.environ
        if set_here:
            user_scripts = Path(scratch) / "blender-user-scripts"
            user_scripts.mkdir()
            os.environ["BLENDER_USER_SCRIPTS"] = str(user_scripts)
        try:
            for spec in panels:
                output = Path(spec["output_dir"]) / f"{spec['id']}.png"
                if output.exists() and not args.force:
                    print(f"{spec['id']}: exists, skipped (use --force)")
                    continue
                render_panel(spec, Path(scratch), args.draft)
        finally:
            if set_here:
                os.environ.pop("BLENDER_USER_SCRIPTS", None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
