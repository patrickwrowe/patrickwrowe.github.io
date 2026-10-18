"""Tests for render_box_grid: render_panel's crop decision and scene_for's layers, without Blender.

render_box_grid imports molrender, which exists only in the Blender environment, so a
stand-in module takes its place. Its scene classes are frozen dataclasses with the field
names and defaults of molrender's own (molrender/scene/graph.py, styles.py, materials.py,
view.py, quality.py), so `scene_for` builds a real, inspectable scene; `render` writes a PNG of
the requested size instead of calling Blender. Everything after the render (the whiteness
check, the crop decision and the move into place) is the real code.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
import types
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import panel_checks
import pytest
from constants import CHAIN_COLOUR, MATTE, RIG, SPECIES_COLOUR, WORLD_STRENGTH, srgb_hex_to_linear
from PIL import Image

TOY_PROTEIN = Path(__file__).parent / "fixtures" / "toy-protein.pdb"


@dataclass(frozen=True)
class StandInSolid:
    rgb: tuple[float, float, float]
    roughness: float = 0.55
    coat: float = 0.20
    subsurface: float = 0.20
    specular: float = 0.5
    ramp: tuple[tuple[float, float, float], float] | None = None

    def ramped(self, toward, strength):
        return replace(self, ramp=(toward, strength))


@dataclass(frozen=True)
class StandInCartoon:
    quality: int | None = None
    peptide_rounded: bool = True


@dataclass(frozen=True)
class StandInSticks:
    radius: float = 0.2


@dataclass(frozen=True)
class StandInSpheres:
    radius: float = 0.8
    geometry: str = "Instance"


@dataclass(frozen=True)
class StandInBallAndStick:
    sphere_radius: float = 0.3
    bond_radius: float = 0.3
    sphere_geometry: str = "Instance"


@dataclass(frozen=True)
class StandInStructureId:
    sha256: str
    n_atoms: int


@dataclass(frozen=True)
class StandInAtomMask:
    mask: tuple[bool, ...]
    of: StandInStructureId


@dataclass(frozen=True)
class StandInLayer:
    style: object
    material: object
    select: object = None


@dataclass(frozen=True)
class StandInStructure:
    path: Path
    layers: tuple[StandInLayer, ...]
    name: str = ""
    remove_solvent: bool = True
    hide: bool = False


@dataclass(frozen=True)
class StandInView:
    direction: tuple[float, float, float] | None = None
    margin: float = 1.06


@dataclass(frozen=True)
class StandInQuality:
    view_transform: str
    look: str | None = None
    diffuse_bounces: int | None = None
    glossy_bounces: int | None = None
    ao_bounces_render: int | None = None
    fast_gi_method: str | None = None


@dataclass(frozen=True)
class StandInWorld:
    strength: float = 1.0
    background: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class StandInLight:
    key_power: float = 4.0
    world: StandInWorld | None = None
    sun_strength: float = 0.0
    side_power: float = 0.0
    under_power: float = 0.0


@dataclass(frozen=True)
class StandInScene:
    entities: tuple[StandInStructure, ...]
    view: StandInView
    quality: StandInQuality
    resolution: tuple[int, int]
    transparent: bool
    shadow_catcher: bool
    light: StandInLight = StandInLight()


MOLRENDER_STAND_INS = {
    "ALL": None,  # what a stand-in Layer selects when `select` is left out
    "DRAFT": StandInQuality("AgX"),
    "PRODUCTION": StandInQuality("AgX"),
    "AtomMask": StandInAtomMask,
    "BallAndStick": StandInBallAndStick,
    "Cartoon": StandInCartoon,
    "Layer": StandInLayer,
    "Light": StandInLight,
    "Scene": StandInScene,
    "Solid": StandInSolid,
    "Spheres": StandInSpheres,
    "Sticks": StandInSticks,
    "Structure": StandInStructure,
    "StructureId": StandInStructureId,
    "View": StandInView,
    "World": StandInWorld,
    "find_blender": None,
    "render": None,
}


@pytest.fixture
def render_box_grid_module(monkeypatch):
    """render_box_grid imported against a stand-in molrender."""
    if "molrender" not in sys.modules:
        stand_in = types.ModuleType("molrender")
        for name, stand_in_object in MOLRENDER_STAND_INS.items():
            setattr(stand_in, name, stand_in_object)
        monkeypatch.setitem(sys.modules, "molrender", stand_in)
    return importlib.import_module("render_box_grid")


@pytest.fixture
def scene_calls() -> list[dict]:
    """The keyword arguments render_panel forwarded to the stand-in `scene_for`, in order."""
    return []


@pytest.fixture
def render_box_grid(render_box_grid_module, monkeypatch, scene_calls):
    """render_box_grid with `prepare` and `scene_for` stubbed, for render_panel."""
    monkeypatch.setattr(
        render_box_grid_module,
        "prepare",
        lambda spec, work_dir: (work_dir / "subject.pdb", None, ["C"] * 7),
    )

    def record_scene(spec, subject, cage, species, *, draft):
        scene_calls.append({"draft": draft})

    monkeypatch.setattr(render_box_grid_module, "scene_for", record_scene)
    monkeypatch.setattr(render_box_grid_module, "tool_versions", lambda: "VERSIONS")
    return render_box_grid_module


@pytest.fixture
def tool_versions(render_box_grid_module, monkeypatch, tmp_path):
    """The real `tool_versions`, with its once-per-run cache emptied before and after.

    The stand-in molrender sits in `tmp_path`, which is no git checkout, so the version
    carries no commit unless a test makes one.
    """
    monkeypatch.setattr(
        render_box_grid_module.molrender,
        "__file__",
        str(tmp_path / "molrender" / "__init__.py"),
        raising=False,
    )
    render_box_grid_module.tool_versions.cache_clear()
    yield render_box_grid_module.tool_versions
    render_box_grid_module.tool_versions.cache_clear()


def test_prepare_writes_a_tiled_slab_of_carbon_with_one_species_per_atom(
    render_box_grid_module, tmp_path
):
    lines = ["4", "four carbons in a 10 A cell"]
    lines += [f"C {x:.1f} 5.0 {z:.1f}" for x, z in ((1.0, 5.0), (2.4, 5.0), (6.0, 5.0), (6.0, 9.5))]
    source = tmp_path / "box.xyz"
    source.write_text("\n".join(lines) + "\n")
    # 4 carbons at 0.08 g cm^-3 fill a 9.99 A cell; tiled twice along x, a 3 A slab about
    # the cell's mid-height keeps the three atoms at z = 5.0 of each image.
    spec = {
        "id": "box",
        "kind": "box",
        "source": str(source),
        "density_g_cm3": 0.08,
        "tile": [2, 1, 1],
        "slab_angstrom": 3.0,
        "cage_angstrom": None,
        "sphere_radius_angstrom": 0.4,
    }
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    subject, cage, species = render_box_grid_module.prepare(spec, work_dir)
    hetatm = [line for line in subject.read_text().splitlines() if line.startswith("HETATM")]
    assert cage is None
    assert species == ["C"] * 6
    assert len(hetatm) == 6
    assert all(line.endswith(" C") for line in hetatm)


def test_prepare_writes_a_frame_of_two_species_with_its_bond(render_box_grid_module, tmp_path):
    source = tmp_path / "co.xyz"
    source.write_text("2\ncarbon monoxide\nC 0.0 0.0 0.0\nO 1.13 0.0 0.0\n")
    spec = {"id": "co", "kind": "box", "source": str(source), "cage_angstrom": None}
    subject, cage, species = render_box_grid_module.prepare(spec, tmp_path)
    lines = subject.read_text().splitlines()
    assert cage is None
    assert species == ["C", "O"]
    assert [line[76:78] for line in lines if line.startswith("HETATM")] == [" C", " O"]
    assert "CONECT    1    2" in lines and "CONECT    2    1" in lines


def test_prepare_refuses_a_slab_that_leaves_no_atoms_naming_the_panel(
    render_box_grid_module, tmp_path
):
    # Two atoms 6 A apart in z: a 2 A slab about their mean z holds neither.
    source = tmp_path / "pair.xyz"
    source.write_text("2\ntwo carbons\nC 0.0 0.0 0.0\nC 0.0 0.0 6.0\n")
    spec = {
        "id": "pair",
        "kind": "box",
        "source": str(source),
        "slab_angstrom": 2.0,
        "cage_angstrom": None,
    }
    with pytest.raises(ValueError, match=r"panel pair: no atoms to draw from .*pair\.xyz"):
        render_box_grid_module.prepare(spec, tmp_path)
    assert not (tmp_path / "pair.pdb").exists()


def _render_writing(
    resolution_px: tuple[int, int],
    ink_box: tuple[int, int, int, int],
    ink_rgb: tuple[int, int, int] = (90, 90, 90),
):
    """A `render` stand-in that writes a white RGB frame with one block of ink.

    Args:
        resolution_px: (width, height) of the frame, in pixels.
        ink_box: (top, bottom, left, right) of the ink, in pixels.
        ink_rgb: The ink's colour, 0-255; grey unless a test draws a coloured species.

    Returns:
        A function with molrender.render's signature.
    """

    def render(scene, path: Path, timeout: float) -> types.SimpleNamespace:
        width_px, height_px = resolution_px
        pixels = np.full((height_px, width_px, 3), 255, dtype=np.uint8)
        top, bottom, left, right = ink_box
        pixels[top:bottom, left:right] = ink_rgb
        Image.fromarray(pixels, mode="RGB").save(path)
        return types.SimpleNamespace(returncode=0)

    return render


def _render_panel(
    module, monkeypatch, tmp_path: Path, spec: dict, ink_box, ink_rgb=(90, 90, 90), draft=False
) -> Path:
    """Run render_panel on `spec` with the stand-in renderer; return the output PNG."""
    frame_px = tuple(length_px // 2 if draft else length_px for length_px in spec["resolution"])
    monkeypatch.setattr(module, "render", _render_writing(frame_px, ink_box, ink_rgb))
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    module.render_panel(
        {"id": "panel", "kind": "box", "output_dir": str(tmp_path / "out"), **spec},
        work_dir,
        draft=draft,
    )
    return tmp_path / "out" / "panel.png"


def test_an_uncaged_non_square_panel_keeps_its_full_frame(
    render_box_grid, monkeypatch, tmp_path, capsys
):
    # The hero: 1728 x 972, uncaged, ink 1708 x 864 px. Squaring it would cut 44% away.
    spec = {"resolution": [1728, 972], "cage_angstrom": None}
    output = _render_panel(render_box_grid, monkeypatch, tmp_path, spec, (54, 918, 10, 1718))
    assert Image.open(output).size == (1728, 972)
    assert "crop skipped (uncaged, but 1728 x 972 is not square)" in capsys.readouterr().out


def test_a_draft_summary_gives_the_draft_frame_not_the_production_one(
    render_box_grid, monkeypatch, tmp_path, capsys
):
    spec = {"resolution": [1728, 972], "cage_angstrom": None}
    output = _render_panel(
        render_box_grid, monkeypatch, tmp_path, spec, (27, 459, 5, 859), draft=True
    )
    assert Image.open(output).size == (864, 486)
    assert "crop skipped (uncaged, but 864 x 486 is not square)" in capsys.readouterr().out


def test_an_uncaged_square_panel_is_cropped_to_its_ink(
    render_box_grid, monkeypatch, tmp_path, capsys
):
    spec = {"resolution": [800, 800], "cage_angstrom": None}
    output = _render_panel(render_box_grid, monkeypatch, tmp_path, spec, (300, 500, 350, 450))
    side_px = round(200 * 1.06)
    assert Image.open(output).size == (side_px, side_px)
    assert f"crop {side_px} px" in capsys.readouterr().out


def test_a_caged_panel_keeps_its_full_frame(render_box_grid, monkeypatch, tmp_path, capsys):
    spec = {"resolution": [800, 800], "cage_angstrom": [44.2, 44.2, 11.0]}
    output = _render_panel(render_box_grid, monkeypatch, tmp_path, spec, (300, 500, 350, 450))
    assert Image.open(output).size == (800, 800)
    summary = capsys.readouterr().out
    assert "crop none (caged)" in summary
    assert summary.rstrip().endswith("versions: VERSIONS")


@pytest.mark.parametrize(
    ("species", "ink_rgb", "mode", "coloured"),
    [
        (["C"] * 7, (90, 90, 90), "L", False),
        (["C"] * 5 + ["O"] * 2, (200, 60, 40), "RGB", True),
    ],
    ids=["one species stays grey", "two species keep their colour"],
)
def test_the_species_count_decides_whether_a_panel_is_saved_grey_or_in_colour(
    render_box_grid, monkeypatch, tmp_path, scene_calls, species, ink_rgb, mode, coloured
):
    monkeypatch.setattr(
        render_box_grid,
        "prepare",
        lambda spec, work_dir: (work_dir / "subject.pdb", None, species),
    )
    spec = {"resolution": [800, 800], "cage_angstrom": [44.2, 44.2, 11.0]}
    output = _render_panel(
        render_box_grid, monkeypatch, tmp_path, spec, (300, 500, 350, 450), ink_rgb
    )
    saved = Image.open(output)
    assert saved.mode == mode
    assert saved.convert("RGB").getpixel((400, 400)) == ink_rgb
    assert scene_calls == [{"draft": False}]


def test_a_coloured_panel_with_ink_in_its_border_is_refused_and_leaves_nothing_behind(
    render_box_grid, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        render_box_grid,
        "prepare",
        lambda spec, work_dir: (work_dir / "subject.pdb", None, ["C", "O"]),
    )
    spec = {"resolution": [800, 800], "cage_angstrom": [44.2, 44.2, 11.0]}
    with pytest.raises(RuntimeError, match="in the 8 px border"):
        _render_panel(
            render_box_grid, monkeypatch, tmp_path, spec, (400, 500, 3, 100), (200, 60, 40)
        )
    assert not (tmp_path / "out" / "panel.png").exists()


SCENE_SPEC = {
    "id": "water",
    "kind": "box",
    "grey": 0.35,
    "sphere_radius_angstrom": 0.4,
    "bond_radius_angstrom": 0.2,
    "direction": [0.0, 0.0, 1.0],
    "resolution": [100, 100],
}


def _scene(module, tmp_path: Path, species: list[str], cage: bool = False, **spec_overrides):
    """`scene_for` on a subject PDB of `species`; returns the scene and the PDB's path."""
    subject = tmp_path / "subject.pdb"
    subject.write_text(f"HETATM stand-in for a PDB of {len(species)} atoms\n")
    scene = module.scene_for(
        {**SCENE_SPEC, **spec_overrides},
        subject,
        subject if cage else None,
        species,
        draft=False,
    )
    return scene, subject


def test_the_scene_is_lit_by_the_rig_and_the_world_strength_in_constants(
    render_box_grid_module, tmp_path
):
    scene, _ = _scene(render_box_grid_module, tmp_path, ["C"] * 4)
    assert scene.light == StandInLight(**RIG, world=StandInWorld(strength=WORLD_STRENGTH))


def test_one_species_is_one_grey_ball_and_stick_layer_whatever_the_colour_map_says(
    render_box_grid_module, tmp_path
):
    scene, _ = _scene(
        render_box_grid_module, tmp_path, ["C"] * 4, cage=True, colour={"C": "#ff0000"}
    )
    grey_layer = StandInLayer(
        style=StandInBallAndStick(
            sphere_radius=0.4 / 1.70, bond_radius=0.2, sphere_geometry="Point"
        ),
        material=StandInSolid(rgb=(0.35, 0.35, 0.35), **MATTE),
    )
    subject, cage = scene.entities
    assert subject.layers == (grey_layer,) and not subject.hide
    assert cage.layers == (grey_layer,) and cage.hide


def test_more_than_one_species_is_a_stick_layer_and_a_masked_sphere_layer_per_species(
    render_box_grid_module, tmp_path
):
    species = ["O", "H", "H", "O", "H"]
    scene, subject = _scene(render_box_grid_module, tmp_path, species, cage=True)
    layers = scene.entities[0].layers
    structure_id = StandInStructureId(
        sha256=hashlib.sha256(subject.read_bytes()).hexdigest(), n_atoms=5
    )

    sticks, *spheres = layers
    # The palette's carbon whatever is drawn, here water: a neutral grey under the spheres.
    assert sticks == StandInLayer(
        style=StandInSticks(radius=0.2),
        material=StandInSolid(rgb=srgb_hex_to_linear(SPECIES_COLOUR["C"]), **MATTE),
    )
    assert spheres == [
        StandInLayer(
            style=StandInSpheres(radius=0.4 / 1.70, geometry="Instance"),
            material=StandInSolid(rgb=srgb_hex_to_linear(SPECIES_COLOUR[symbol]), **MATTE),
            select=StandInAtomMask(mask=mask, of=structure_id),
        )
        for symbol, mask in (
            ("H", (False, True, True, False, True)),
            ("O", (True, False, False, True, False)),
        )
    ]
    # The hidden cage that fixes the scale is not recoloured.
    assert scene.entities[1].layers[0].style.sphere_geometry == "Point"
    assert scene.entities[1].hide


def test_the_sticks_are_the_palettes_carbon_even_beside_a_darker_species(
    render_box_grid_module, tmp_path
):
    # Whatever magnesium's colour (dark green until 2026-10-17), the sticks stay carbon.
    scene, _ = _scene(render_box_grid_module, tmp_path, ["O", "Mg", "C", "C"])
    sticks, *spheres = scene.entities[0].layers
    assert sticks.material.rgb == srgb_hex_to_linear(SPECIES_COLOUR["C"])
    assert srgb_hex_to_linear(SPECIES_COLOUR["Mg"]) in [layer.material.rgb for layer in spheres]
    assert sticks.material.rgb != srgb_hex_to_linear(SPECIES_COLOUR["Mg"])


def test_the_manifest_colour_map_beats_the_palette_and_can_add_a_species(
    render_box_grid_module, tmp_path
):
    scene, _ = _scene(
        render_box_grid_module,
        tmp_path,
        ["C", "O", "Xx"],
        colour={"O": "#112233", "Xx": "#aabbcc"},
    )
    sticks, *spheres = scene.entities[0].layers
    rgb_by_mask = {layer.select.mask: layer.material.rgb for layer in spheres}
    assert rgb_by_mask[(True, False, False)] == srgb_hex_to_linear(SPECIES_COLOUR["C"])
    assert rgb_by_mask[(False, True, False)] == srgb_hex_to_linear("#112233")
    assert rgb_by_mask[(False, False, True)] == srgb_hex_to_linear("#aabbcc")
    # The sticks follow the carbon the map leaves alone, not the darker oxygen it sets.
    assert sticks.material.rgb == srgb_hex_to_linear(SPECIES_COLOUR["C"])


def test_a_sphere_scale_enlarges_one_species_and_light_and_quality_override_the_tier(
    render_box_grid_module, tmp_path
):
    scene, _ = _scene(
        render_box_grid_module,
        tmp_path,
        ["O", "Mg", "C", "C"],
        sphere_scale={"Mg": 4},
        light={"under_power": 8.0},
        quality={"diffuse_bounces": 5, "fast_gi_method": "ADD"},
    )
    _, *spheres = scene.entities[0].layers
    radius = {layer.select.mask: layer.style.radius for layer in spheres}
    assert radius[(False, True, False, False)] == pytest.approx(
        4 * radius[(True, False, False, False)]
    )
    assert scene.light.under_power == 8.0 and scene.light.side_power == RIG["side_power"]
    assert (scene.quality.diffuse_bounces, scene.quality.fast_gi_method) == (5, "ADD")


def _atom_span_ramp(monkeypatch, present: bool = True) -> None:
    """A stand-in `molrender.blender.view`, with or without the atom-span `depth_span`."""
    view = types.ModuleType("molrender.blender.view")
    if present:
        view.depth_span = lambda points, origin, forward: (0.0, 1.0)
    monkeypatch.setitem(sys.modules, "molrender.blender.view", view)


def test_a_depth_cue_ramps_every_subject_material_toward_the_backdrop_and_not_the_cage(
    render_box_grid_module, tmp_path, monkeypatch
):
    _atom_span_ramp(monkeypatch)
    plain, _ = _scene(render_box_grid_module, tmp_path, ["B", "N", "B", "N"], cage=True)
    assert {layer.material.ramp for layer in plain.entities[0].layers} == {None}
    scene, _ = _scene(
        render_box_grid_module,
        tmp_path,
        ["B", "N", "B", "N"],
        cage=True,
        depth_cue={"strength": 0.5},
    )
    subject, cage = scene.entities
    assert len(subject.layers) == 3  # the sticks and one sphere layer per species
    assert {layer.material.ramp for layer in subject.layers} == {((1.0, 1.0, 1.0), 0.5)}
    assert cage.layers[0].material.ramp is None
    # Nothing else about the layers changes: the ramp is the only difference.
    assert [replace(layer.material, ramp=None) for layer in subject.layers] == [
        layer.material for layer in plain.entities[0].layers
    ]
    one_species, _ = _scene(
        render_box_grid_module, tmp_path, ["C"] * 4, depth_cue={"strength": 0.3}
    )
    assert one_species.entities[0].layers[0].material.ramp == ((1.0, 1.0, 1.0), 0.3)


def test_a_depth_cue_on_a_molrender_without_the_atom_span_ramp_is_refused(
    render_box_grid_module, tmp_path, monkeypatch
):
    # The strengths are tuned to the atom span; an older molrender would ramp over the
    # bounding sphere and draw a paler near side with no error.
    _atom_span_ramp(monkeypatch, present=False)
    with pytest.raises(RuntimeError, match=r"panel .*depth_span.*9ed1235"):
        _scene(render_box_grid_module, tmp_path, ["B", "N"], depth_cue={"strength": 0.3})
    # Without a depth cue nothing is asked of molrender.blender.view.
    _scene(render_box_grid_module, tmp_path, ["B", "N"])


def test_a_light_key_the_installed_molrender_lacks_is_refused_naming_the_commit(
    render_box_grid_module, tmp_path
):
    with pytest.raises(RuntimeError, match=r"\['shadow_power'\].*4b26311"):
        _scene(render_box_grid_module, tmp_path, ["O", "C"], light={"shadow_power": 1.0})


def test_a_species_in_neither_the_colour_map_nor_the_palette_is_refused_naming_it(
    render_box_grid_module, tmp_path
):
    with pytest.raises(ValueError, match=r"panel water: no colour for \['Xx'\]"):
        _scene(render_box_grid_module, tmp_path, ["C", "Xx", "O"], colour={"O": "#112233"})


def _fake_blender(tmp_path: Path, body: str) -> Path:
    """An executable shell script standing in for Blender."""
    script = tmp_path / "blender"
    script.write_text(f"#!/bin/sh\n{body}\n")
    script.chmod(0o755)
    return script


@pytest.mark.parametrize(
    ("blender_body", "molrender_installed", "expected"),
    [
        (
            "printf 'TOOL_VERSIONS 9.9.9 4.5.13\\nBlender 9.9.9 (hash x)\\n\\nBlender quit\\n'",
            True,
            "Blender 9.9.9, Molecular Nodes 4.5.13, molrender 1.2.3",
        ),
        (
            "printf 'TOOL_VERSIONS 9.9.9 unknown\\n'",
            True,
            "Blender 9.9.9, Molecular Nodes unknown, molrender 1.2.3",
        ),
        (
            "printf 'TOOL_VERSIONS 9.9.9 4.5.13\\n'",
            False,
            "Blender 9.9.9, Molecular Nodes 4.5.13, molrender unknown",
        ),
        (
            "echo 'TOOL_VERSIONS 9.9.9 4.5.13'; exit 1",
            True,
            "Blender unknown, Molecular Nodes unknown, molrender 1.2.3",
        ),
        (
            "echo 'Blender 9.9.9 (hash x)'",  # the expression raised before it printed
            True,
            "Blender unknown, Molecular Nodes unknown, molrender 1.2.3",
        ),
        (None, True, "Blender unknown, Molecular Nodes unknown, molrender 1.2.3"),
    ],
    ids=[
        "all found",
        "no molecular nodes extension",
        "molrender not installed",
        "blender exits non-zero",
        "no version line",
        "blender not found",
    ],
)
def test_tool_versions_reads_each_version_or_says_unknown(
    render_box_grid_module,
    tool_versions,
    monkeypatch,
    tmp_path,
    blender_body,
    molrender_installed,
    expected,
):
    def find_blender() -> Path:
        if blender_body is None:
            raise FileNotFoundError("Could not locate the Blender executable")
        return _fake_blender(tmp_path, blender_body)

    def installed_version(name: str) -> str:
        if not molrender_installed:
            raise render_box_grid_module.PackageNotFoundError(name)
        return "1.2.3"

    monkeypatch.setattr(render_box_grid_module, "find_blender", find_blender)
    monkeypatch.setattr(render_box_grid_module, "version", installed_version)

    assert tool_versions() == expected


def test_tool_versions_names_the_checkout_commit_molrender_runs_from(
    render_box_grid_module, tool_versions, monkeypatch, tmp_path
):
    checkout = tmp_path / "checkout"
    package_dir = checkout / "src" / "molrender"
    package_dir.mkdir(parents=True)
    git = [
        "git",
        "-C",
        str(checkout),
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "-c",
        "commit.gpgsign=false",
    ]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    short_sha = subprocess.run(
        [*git, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()

    monkeypatch.setattr(
        render_box_grid_module.molrender, "__file__", str(package_dir / "__init__.py")
    )
    monkeypatch.setattr(
        render_box_grid_module,
        "find_blender",
        lambda: _fake_blender(tmp_path, "echo 'TOOL_VERSIONS 9.9.9 4.5.13'"),
    )
    monkeypatch.setattr(render_box_grid_module, "version", lambda name: "1.2.3")

    assert tool_versions() == (
        f"Blender 9.9.9, Molecular Nodes 4.5.13, molrender 1.2.3 ({short_sha})"
    )

    # A tracked file edited after the commit marks the SHA dirty: that code did not render.
    tracked = checkout / "tracked.txt"
    tracked.write_text("a\n")
    subprocess.run([*git, "add", "tracked.txt"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "y"], check=True)
    dirty_sha = subprocess.run(
        [*git, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    tracked.write_text("b\n")
    tool_versions.cache_clear()
    assert tool_versions() == (
        f"Blender 9.9.9, Molecular Nodes 4.5.13, molrender 1.2.3 ({dirty_sha}-dirty)"
    )


# ---------------------------------------------------------------------------
# protein panels
# ---------------------------------------------------------------------------

# toy-protein.pdb keeps 10 atoms and `prepare` draws 9: chain A holds atoms 0-4 (alanine,
# glycine) and zinc (7); chain B holds atoms 5 and 6 and the ligand carbon (8). Its magnesium
# (serial 10) is a lone metal ion bonded to nothing, which `prepare` leaves out; the zinc has
# no covalent radius, so no bond can be looked for, and is drawn. Zinc has no palette colour,
# so a protein spec that draws it names one.
PROTEIN_SPEC = {
    **SCENE_SPEC,
    "id": "toy",
    "kind": "protein",
    "source": str(TOY_PROTEIN),
    "cartoon_chains": ["A"],
    "stick_chains": ["B"],
    "cage_angstrom": None,
    "colour": {"Zn": "#336699"},
}


def _protein_scene(module, work_dir: Path, **spec_overrides):
    """`prepare` then `scene_for` on the toy protein; returns the scene and the subject PDB."""
    spec = {**PROTEIN_SPEC, **spec_overrides}
    subject, cage, species = module.prepare(spec, work_dir)
    assert cage is None
    scene = module.scene_for(spec, subject, cage, species, draft=False)
    return scene, subject


def test_prepare_writes_a_protein_as_the_atoms_read_pdb_kept_in_the_files_order(
    render_box_grid_module, tmp_path
):
    subject, cage, species = render_box_grid_module.prepare(PROTEIN_SPEC, tmp_path)
    lines = subject.read_text().splitlines()
    assert cage is None
    assert species == ["N", "C", "C", "N", "C", "N", "C", "Zn", "C"]
    assert [line[:5] for line in lines if not line.startswith(("ATOM", "HETATM"))] == [
        "HELIX",
        "SHEET",
        "END",
    ]
    atoms = [line for line in lines if line.startswith(("ATOM", "HETATM"))]
    assert [int(line[6:11]) for line in atoms] == [1, 2, 4, 5, 6, 7, 8, 9, 11]
    # Atom 8 is in residue 7A: the insertion code reaches column 27 of the subject file.
    assert atoms[6][22:27] == "   7A"


@pytest.mark.parametrize(
    ("cartoon_chains", "stick_chains"),
    [(["A"], []), (["A", "B", "Z"], []), (["A"], ["B", "Z"])],
    ids=["a chain left undrawn", "a chain the file lacks", "a stick chain the file lacks"],
)
def test_a_protein_whose_chains_are_not_the_files_is_refused_naming_both(
    render_box_grid_module, tmp_path, cartoon_chains, stick_chains
):
    spec = {**PROTEIN_SPEC, "cartoon_chains": cartoon_chains, "stick_chains": stick_chains}
    with pytest.raises(ValueError, match=r"panel toy: .* has chains \['A', 'B'\] but the panel"):
        render_box_grid_module.prepare(spec, tmp_path)
    assert not (tmp_path / "toy.pdb").exists()


def test_a_protein_is_a_cartoon_layer_per_cartoon_chain_and_species_layers_over_the_rest(
    render_box_grid_module, tmp_path
):
    scene, subject = _protein_scene(render_box_grid_module, tmp_path)
    structure_id = StandInStructureId(
        sha256=hashlib.sha256(subject.read_bytes()).hexdigest(), n_atoms=9
    )
    (structure,) = scene.entities  # no cage
    cartoon, sticks, *spheres = structure.layers
    assert not structure.remove_solvent

    # Chain A is atoms 0-4 and the zinc; chain B is atoms 5, 6 and the ligand carbon.
    assert cartoon == StandInLayer(
        style=StandInCartoon(),
        material=StandInSolid(rgb=srgb_hex_to_linear(CHAIN_COLOUR[0]), **MATTE),
        select=StandInAtomMask(mask=(True,) * 5 + (False,) * 2 + (True, False), of=structure_id),
    )
    # The sticks and spheres cover the stick chain and every HETATM, and nothing else.
    drawn = (False,) * 5 + (True,) * 4
    assert sticks.select == StandInAtomMask(mask=drawn, of=structure_id)
    # The sticks are the palette's carbon, whatever darker species is drawn.
    assert sticks.material == StandInSolid(rgb=srgb_hex_to_linear(SPECIES_COLOUR["C"]), **MATTE)
    masks = {
        symbol: layer.select.mask for symbol, layer in zip(["C", "N", "Zn"], spheres, strict=True)
    }
    assert [layer.material.rgb for layer in spheres] == [
        srgb_hex_to_linear(colour)
        for colour in (SPECIES_COLOUR["C"], SPECIES_COLOUR["N"], "#336699")
    ]
    assert masks["C"] == (False,) * 6 + (True, False, True)
    assert masks["N"] == (False,) * 5 + (True,) + (False,) * 3
    assert masks["Zn"] == (False,) * 7 + (True, False)
    assert all(
        layer.style == StandInSpheres(radius=0.4 / 1.70, geometry="Instance") for layer in spheres
    )
    assert sticks.style == StandInSticks(radius=0.2)


def _chain_a_only(tmp_path: Path) -> Path:
    """The toy protein's atoms 1-6, chain A alone: no ligand, no ion, no second chain."""
    chain_a = tmp_path / "chain-a.pdb"  # read_pdb drops the B conformer, atom 3
    chain_a.write_text(
        "".join(
            f"{line}\n"
            for line in TOY_PROTEIN.read_text().splitlines()
            if line.startswith("ATOM") and line[21] == "A" and int(line[6:11]) < 7  # model 1
        )
    )
    return chain_a


def test_a_protein_of_cartoon_chains_alone_is_its_cartoon_layers_alone(
    render_box_grid_module, tmp_path
):
    scene, _ = _protein_scene(
        render_box_grid_module, tmp_path, source=str(_chain_a_only(tmp_path)), stick_chains=[]
    )
    (structure,) = scene.entities
    (cartoon,) = structure.layers
    assert isinstance(cartoon.style, StandInCartoon)
    assert cartoon.select.mask == (True,) * 5


def test_each_cartoon_chain_takes_the_chain_palette_in_order_unless_the_manifest_names_one(
    render_box_grid_module, tmp_path
):
    scene, subject = _protein_scene(
        render_box_grid_module,
        tmp_path,
        cartoon_chains=["B", "A"],
        stick_chains=[],
        chain_colour={"A": "#102030"},
    )
    structure_id = StandInStructureId(
        sha256=hashlib.sha256(subject.read_bytes()).hexdigest(), n_atoms=9
    )
    chain_b, chain_a = scene.entities[0].layers[:2]
    # Chain B is first in the list, so it takes the palette's first colour.
    assert chain_b == StandInLayer(
        style=StandInCartoon(),
        material=StandInSolid(rgb=srgb_hex_to_linear(CHAIN_COLOUR[0]), **MATTE),
        select=StandInAtomMask(mask=(False,) * 5 + (True,) * 2 + (False, True), of=structure_id),
    )
    assert chain_a.material.rgb == srgb_hex_to_linear("#102030")
    assert chain_a.select.mask == (True,) * 5 + (False,) * 2 + (True, False)


def test_a_cartoon_chain_beyond_the_palette_with_no_manifest_colour_is_refused(
    render_box_grid_module, monkeypatch, tmp_path
):
    monkeypatch.setattr(render_box_grid_module, "CHAIN_COLOUR", ("#3f6e5b",))
    with pytest.raises(ValueError, match=r"panel toy: no colour for chain\(s\) \['A'\]"):
        _protein_scene(render_box_grid_module, tmp_path, cartoon_chains=["B", "A"], stick_chains=[])


def test_a_hetero_atom_with_no_colour_is_refused_naming_it(render_box_grid_module, tmp_path):
    with pytest.raises(ValueError, match=r"panel toy: no colour for \['Zn'\]"):
        _protein_scene(render_box_grid_module, tmp_path, colour={})


def test_a_protein_panel_goes_from_pdb_to_a_saved_rgb_png(
    render_box_grid_module, monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(render_box_grid_module, "tool_versions", lambda: "VERSIONS")
    spec = {**PROTEIN_SPEC, "id": "panel", "resolution": [1728, 972]}
    output = _render_panel(
        render_box_grid_module, monkeypatch, tmp_path, spec, (54, 918, 10, 1718), (74, 74, 74)
    )
    saved = Image.open(output)
    assert saved.mode == "RGB" and saved.size == (1728, 972)
    assert "panel: 9 atoms" in capsys.readouterr().out


def test_a_protein_panels_summary_line_counts_the_atom_records_read_pdb_dropped(
    render_box_grid_module, monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(render_box_grid_module, "tool_versions", lambda: "VERSIONS")
    spec = {**PROTEIN_SPEC, "id": "panel", "resolution": [1728, 972]}
    _render_panel(
        render_box_grid_module, monkeypatch, tmp_path, spec, (54, 918, 10, 1718), (74, 74, 74)
    )
    assert "panel: 9 atoms (dropped 1 altloc, 1 water), wall" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("stick_chains", "ink_rgb", "mode", "coloured"),
    [
        (["B"], (200, 60, 40), "RGB", True),
        ([], (63, 110, 91), "RGB", True),
    ],
    ids=["a peptide drawn by species is coloured", "cartoon chains alone are coloured too"],
)
def test_a_protein_is_always_saved_in_colour(
    render_box_grid_module,
    monkeypatch,
    tmp_path,
    scene_calls,
    stick_chains,
    ink_rgb,
    mode,
    coloured,
):
    # Its chains are coloured, so a protein is coloured whether or not it draws an atom by
    # species; the cartoon-alone file (the hero's case) was saved grey before 2026-10-09.
    source = TOY_PROTEIN if stick_chains else _chain_a_only(tmp_path)
    spec = {
        **PROTEIN_SPEC,
        "id": "panel",
        "source": str(source),
        "cartoon_chains": ["A"],
        "stick_chains": stick_chains,
        "resolution": [1728, 972],
    }
    monkeypatch.setattr(render_box_grid_module, "tool_versions", lambda: "VERSIONS")
    monkeypatch.setattr(
        render_box_grid_module,
        "scene_for",
        lambda spec, subject, cage, species, *, draft: scene_calls.append({"draft": draft}),
    )
    output = _render_panel(
        render_box_grid_module, monkeypatch, tmp_path, spec, (54, 918, 10, 1718), ink_rgb
    )
    assert Image.open(output).mode == mode
    assert scene_calls == [{"draft": False}]


def test_a_cartoon_alone_protein_is_judged_on_its_border_not_its_whole_frame(
    render_box_grid_module, monkeypatch, tmp_path
):
    spec = {
        **PROTEIN_SPEC,
        "id": "panel",
        "source": str(_chain_a_only(tmp_path)),
        "stick_chains": [],
        "resolution": [1728, 972],
    }
    monkeypatch.setattr(render_box_grid_module, "tool_versions", lambda: "VERSIONS")
    with pytest.raises(RuntimeError, match=r"max chroma 160/255 in the 8 px border exceeds 32"):
        _render_panel(
            render_box_grid_module, monkeypatch, tmp_path, spec, (54, 918, 3, 1718), (200, 60, 40)
        )


# A three-atom ligand beside the toy protein, serials 14 to 16 (13 is model 2's atom): C-C
# 1.40 A and C-O 1.43 A, and an ion that has no covalent radius.
LIGAND_RECORDS = [
    "HETATM   14  C2  LIG B 301      13.400   3.000   1.000  1.00 20.00           C",
    "HETATM   15  O2  LIG B 301      14.830   3.000   1.000  1.00 20.00           O",
]


def test_prepare_writes_a_conect_record_for_each_bond_among_the_hetero_atoms_with_the_files_serials(
    render_box_grid_module, tmp_path
):
    # The toy file's HETATM C1 (serial 11) is bonded to the two new atoms in a chain.
    lines = TOY_PROTEIN.read_text().splitlines()
    lines.insert(lines.index("ENDMDL"), "\n".join(LIGAND_RECORDS))
    source = tmp_path / "ligand.pdb"
    source.write_text("\n".join(lines) + "\n")
    subject, _cage, _species = render_box_grid_module.prepare(
        {**PROTEIN_SPEC, "source": str(source)}, tmp_path
    )
    written = subject.read_text().splitlines()
    assert [line for line in written if line.startswith("CONECT")] == [
        "CONECT   11   14",
        "CONECT   14   11   15",
        "CONECT   15   14",
    ]
    assert written[-1] == "END" and written[-2].startswith("CONECT")


def test_a_protein_with_no_bonded_hetero_atoms_gets_no_conect_record(
    render_box_grid_module, tmp_path
):
    # The toy's zinc has no covalent radius, and its magnesium, 6 A from the ligand carbon, is
    # a lone ion that is not drawn at all.
    subject, _cage, _species = render_box_grid_module.prepare(PROTEIN_SPEC, tmp_path)
    assert not any(line.startswith("CONECT") for line in subject.read_text().splitlines())


def test_a_protein_atom_serial_that_is_not_a_number_is_refused_naming_the_panel(
    render_box_grid_module, tmp_path
):
    source = tmp_path / "hybrid36.pdb"
    source.write_text(
        TOY_PROTEIN.read_text().replace("ATOM      1  N   ALA", "ATOM  A0000  N   ALA")
    )
    with pytest.raises(ValueError, match=r"panel toy: .*hybrid36\.pdb has an atom serial"):
        render_box_grid_module.prepare({**PROTEIN_SPEC, "source": str(source)}, tmp_path)


REPO = Path(__file__).resolve().parents[2]
FMO_MANIFEST = REPO / "scripts/figures/data/excited-states/manifest.json"


def test_the_fmo_hero_is_prepared_with_the_568_bonds_of_its_eight_bchl_and_no_other(
    render_box_grid_module, tmp_path
):
    # The real 3BSD through the real `prepare`: the ligand's bonds are the one thing Molecular
    # Nodes would not draw for itself (504 unbonded spheres in the first draft, 2026-10-09).
    spec = next(panel for panel in panel_checks.load_panels(FMO_MANIFEST) if panel["id"] == "hero")
    subject, cage, species = render_box_grid_module.prepare(spec, tmp_path)
    assert cage is None and len(species) == 3302
    # The deposited file's lone Mg 3262 (occupancy 0.33, bonded only to waters) is not drawn.
    assert "HETATM 3262 MG    MG A 374" in Path(spec["source"]).read_text()
    assert "HETATM 3262" not in subject.read_text()
    lines = subject.read_text().splitlines()
    atoms = {
        int(line[6:11]): (line[17:20], line[22:26].strip(), line[12:16].strip(), line[21])
        for line in lines
        if line.startswith(("ATOM", "HETATM"))
    }
    partners = {
        int(line[6:11]): [int(line[start : start + 5]) for start in range(11, len(line), 5)]
        for line in lines
        if line.startswith("CONECT")
    }
    bonds = {frozenset((atom, partner)) for atom, others in partners.items() for partner in others}
    assert len(bonds) == 568
    # Both ends of every bond are ligand atoms of one residue, never protein and never two BCL.
    assert all(
        len({atoms[atom][:2] for atom in bond}) == 1 and atoms[next(iter(bond))][0] == "BCL"
        for bond in bonds
    )
    per_residue = {}
    for bond in bonds:
        residue = atoms[next(iter(bond))][1]
        per_residue[residue] = per_residue.get(residue, 0) + 1
    assert per_residue == {
        **dict.fromkeys(("367", "368", "369", "370", "371", "372", "373"), 74),
        "400": 50,
    }
    # Each magnesium is bound to the four pyrrole nitrogens and to nothing else.
    magnesium = [
        atom for atom, (_name, _residue, atom_name, _chain) in atoms.items() if atom_name == "MG"
    ]
    assert len(magnesium) == 8
    for atom in magnesium:
        assert sorted(atoms[partner][2][0] for partner in partners[atom]) == ["N"] * 4


TWO_D_MANIFEST = REPO / "scripts/figures/data/two-dimensional-and-confined/manifest.json"


def test_the_vermiculite_film_is_prepared_with_bare_lithium_drawn_large_and_whole_waters(
    render_box_grid_module, tmp_path
):
    # The real staged frame through the real `prepare` and `scene_for`: lithium is a bare
    # ion at 2.5 times the sphere radius, and every hydrogen is bonded to one oxygen.
    spec = next(
        panel
        for panel in panel_checks.load_panels(TWO_D_MANIFEST)
        if panel["id"] == "vermiculite-li-film"
    )
    subject, cage, species = render_box_grid_module.prepare(spec, tmp_path)
    assert cage is None
    assert {symbol: species.count(symbol) for symbol in set(species)} == {
        "Al": 12, "H": 264, "Li": 12, "Mg": 36, "O": 264, "Si": 36,
    }  # fmt: skip
    partners = {
        int(line[6:11]): [int(line[start : start + 5]) for start in range(11, len(line), 5)]
        for line in subject.read_text().splitlines()
        if line.startswith("CONECT")
    }
    for serial, symbol in enumerate(species, start=1):
        if symbol == "Li":
            assert serial not in partners
        if symbol == "H":
            assert [species[partner - 1] for partner in partners[serial]] == ["O"]
    scene = render_box_grid_module.scene_for(spec, subject, None, species, draft=False)
    _, *spheres = scene.entities[0].layers
    radius = {
        species[mask.index(True)]: layer.style.radius
        for layer in spheres
        for mask in [list(layer.select.mask)]
    }
    assert radius["Li"] == pytest.approx(2.5 * radius["O"]) and radius["O"] == radius["Si"]
    assert scene.light.under_power == 8.0
    assert (scene.quality.ao_bounces_render, scene.quality.fast_gi_method) == (5, "REPLACE")


PHOSPHATE_RECORDS = [
    "HETATM   20  P   PO4 B 401      20.000   3.000   1.000  1.00 20.00           P",
    "HETATM   21  O1  PO4 B 401      21.540   3.000   1.000  1.00 20.00           O",
    "HETATM   22  O2  PO4 B 401      19.487   4.452   1.000  1.00 20.00           O",
    "HETATM   23  O3  PO4 B 401      19.487   2.274   2.257  1.00 20.00           O",
    "HETATM   24  O4  PO4 B 401      19.487   2.274  -0.257  1.00 20.00           O",
]


def test_a_ligand_atom_with_no_covalent_radius_is_refused_naming_it_and_its_residue(
    render_box_grid_module, tmp_path
):
    # Phosphorus has no covalent radius, so the phosphate's P-O bonds could not be written and
    # P would float free of its four oxygens; the run used to exit 0.
    lines = TOY_PROTEIN.read_text().splitlines()
    lines.insert(lines.index("ENDMDL"), "\n".join(PHOSPHATE_RECORDS))
    source = tmp_path / "phosphate.pdb"
    source.write_text("\n".join(lines) + "\n")
    with pytest.raises(
        ValueError, match=r"panel toy: .* no covalent radius for \['P in PO4 B 401'\]"
    ):
        render_box_grid_module.prepare({**PROTEIN_SPEC, "source": str(source)}, tmp_path)
    assert not (tmp_path / "toy.pdb").exists()


def test_a_lone_ion_with_no_covalent_radius_passes_and_is_drawn(render_box_grid_module, tmp_path):
    # The toy's zinc is a one-atom residue, ZN A 101: there is no bond to write.
    subject, _cage, species = render_box_grid_module.prepare(PROTEIN_SPEC, tmp_path)
    assert "Zn" in species
    assert "HETATM    9 ZN    ZN A 101" in subject.read_text()


def test_a_lone_metal_ion_bonded_to_nothing_is_left_out_and_a_bonded_one_is_kept(
    render_box_grid_module, tmp_path
):
    # The toy's MG A 102 is 6 A from every other hetero atom. Moved 2.1 A from the ligand
    # oxygen of LIGAND_RECORDS it is bonded (the Mg-O cutoff is 2.48 A) and drawn.
    subject, _cage, species = render_box_grid_module.prepare(PROTEIN_SPEC, tmp_path)
    assert "Mg" not in species and "HETATM   10" not in subject.read_text()
    lines = TOY_PROTEIN.read_text().splitlines()
    lines.insert(lines.index("ENDMDL"), "\n".join(LIGAND_RECORDS))
    source = tmp_path / "bound.pdb"
    source.write_text(
        "\n".join(lines).replace(
            "HETATM   10 MG    MG A 102       6.000   5.000   5.000",
            "HETATM   10 MG    MG A 102      16.930   3.000   1.000",
        )
        + "\n"
    )
    subject, _cage, species = render_box_grid_module.prepare(
        {**PROTEIN_SPEC, "id": "bound", "source": str(source)}, tmp_path
    )
    assert "Mg" in species
    assert "CONECT   10   15" in subject.read_text().splitlines()


def test_main_leaves_the_environment_as_it_found_it(render_box_grid, monkeypatch, tmp_path):
    source = tmp_path / "c.xyz"
    source.write_text("1\none carbon\nC 0.0 0.0 0.0\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "output_dir": str(tmp_path / "out"),
                "direction": [0.0, 0.0, 1.0],
                "resolution": [100, 100],
                "sphere_radius_angstrom": 0.4,
                "bond_radius_angstrom": 0.2,
                "grey": 0.25,
                "panels": [{"id": "a", "source": str(source)}, {"id": "b", "source": str(source)}],
            }
        )
    )
    monkeypatch.delenv("BLENDER_USER_SCRIPTS", raising=False)
    before = dict(os.environ)
    seen = []

    def render_panel(spec, work_dir, draft):
        seen.append(os.environ["BLENDER_USER_SCRIPTS"])
        assert Path(seen[-1]).is_dir()  # a fresh directory Blender can write its template to
        if spec["id"] == "b":
            raise RuntimeError("panel b failed")

    monkeypatch.setattr(render_box_grid, "render_panel", render_panel)
    with pytest.raises(RuntimeError, match="panel b failed"):
        render_box_grid.main(["--manifest", str(manifest)])
    assert len(seen) == 2
    assert dict(os.environ) == before
    # A variable the caller set is the caller's: used, and left alone.
    monkeypatch.setenv("BLENDER_USER_SCRIPTS", str(tmp_path))
    seen.clear()
    monkeypatch.setattr(
        render_box_grid, "render_panel", lambda spec, work_dir, draft: seen.append(1)
    )
    assert render_box_grid.main(["--manifest", str(manifest)]) == 0
    assert os.environ["BLENDER_USER_SCRIPTS"] == str(tmp_path)


@dataclass(frozen=True)
class MainBranchLight:
    """molrender main's Light, before the opt-in sun fill and side light."""

    key_power: float = 4.0
    world: StandInWorld | None = None


def test_a_molrender_whose_light_lacks_the_rigs_fields_is_refused_at_import(monkeypatch):
    stand_in = types.ModuleType("molrender")
    for name, stand_in_object in {**MOLRENDER_STAND_INS, "Light": MainBranchLight}.items():
        setattr(stand_in, name, stand_in_object)
    monkeypatch.setitem(sys.modules, "molrender", stand_in)
    monkeypatch.delitem(sys.modules, "render_box_grid", raising=False)
    with pytest.raises(ImportError, match=r"\['side_power', 'sun_strength'\].*081ded2"):
        importlib.import_module("render_box_grid")
