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
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from constants import SPECIES_COLOUR, srgb_hex_to_linear
from PIL import Image

TOY_PROTEIN = Path(__file__).parent / "fixtures" / "toy-protein.pdb"


@dataclass(frozen=True)
class StandInSolid:
    rgb: tuple[float, float, float]


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


@dataclass(frozen=True)
class StandInScene:
    entities: tuple[StandInStructure, ...]
    view: StandInView
    quality: StandInQuality
    resolution: tuple[int, int]
    transparent: bool
    shadow_catcher: bool


MOLRENDER_STAND_INS = {
    "ALL": None,  # what a stand-in Layer selects when `select` is left out
    "DRAFT": StandInQuality("AgX"),
    "PRODUCTION": StandInQuality("AgX"),
    "AtomMask": StandInAtomMask,
    "BallAndStick": StandInBallAndStick,
    "Cartoon": StandInCartoon,
    "Layer": StandInLayer,
    "Scene": StandInScene,
    "Solid": StandInSolid,
    "Spheres": StandInSpheres,
    "Sticks": StandInSticks,
    "Structure": StandInStructure,
    "StructureId": StandInStructureId,
    "View": StandInView,
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

    def record_scene(spec, subject, cage, species, *, multi_species, draft):
        scene_calls.append({"multi_species": multi_species, "draft": draft})

    monkeypatch.setattr(render_box_grid_module, "scene_for", record_scene)
    monkeypatch.setattr(render_box_grid_module, "tool_versions", lambda: "VERSIONS")
    return render_box_grid_module


@pytest.fixture
def tool_versions(render_box_grid_module):
    """The real `tool_versions`, with its once-per-run cache emptied before and after."""
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
    spec = {"id": "co", "source": str(source), "cage_angstrom": None}
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
    spec = {"id": "pair", "source": str(source), "slab_angstrom": 2.0, "cage_angstrom": None}
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
    module, monkeypatch, tmp_path: Path, spec: dict, ink_box, ink_rgb=(90, 90, 90)
) -> Path:
    """Run render_panel on `spec` with the stand-in renderer; return the output PNG."""
    monkeypatch.setattr(
        module, "render", _render_writing(tuple(spec["resolution"]), ink_box, ink_rgb)
    )
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    module.render_panel(
        {"id": "panel", "output_dir": str(tmp_path / "out"), **spec}, work_dir, draft=False
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
    ("species", "ink_rgb", "mode", "multi_species"),
    [
        (["C"] * 7, (90, 90, 90), "L", False),
        (["C"] * 5 + ["O"] * 2, (200, 60, 40), "RGB", True),
    ],
    ids=["one species stays grey", "two species keep their colour"],
)
def test_the_species_count_decides_whether_a_panel_is_saved_grey_or_in_colour(
    render_box_grid, monkeypatch, tmp_path, scene_calls, species, ink_rgb, mode, multi_species
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
    # The one decision reaches the scene as well as the PNG's colour mode.
    assert scene_calls == [{"multi_species": multi_species, "draft": False}]


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
        multi_species=len(set(species)) > 1,
        draft=False,
    )
    return scene, subject


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
        material=StandInSolid(rgb=(0.35, 0.35, 0.35)),
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
    # The darker of the two colours present, in linear light: oxygen's, not hydrogen's.
    assert sticks == StandInLayer(
        style=StandInSticks(radius=0.2),
        material=StandInSolid(rgb=srgb_hex_to_linear(SPECIES_COLOUR["O"])),
    )
    assert spheres == [
        StandInLayer(
            style=StandInSpheres(radius=0.4 / 1.70, geometry="Instance"),
            material=StandInSolid(rgb=srgb_hex_to_linear(SPECIES_COLOUR[symbol])),
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


def test_the_sticks_take_the_darkest_colour_present(render_box_grid_module, tmp_path):
    scene, _ = _scene(render_box_grid_module, tmp_path, ["O", "H", "C", "C"])
    sticks = scene.entities[0].layers[0]
    assert sticks.material.rgb == srgb_hex_to_linear(SPECIES_COLOUR["C"])


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
    # #112233 is now the darkest of the three, so the sticks follow it.
    assert sticks.material.rgb == srgb_hex_to_linear("#112233")


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
            "printf 'Blender 9.9.9\\n\\tbuild date: 2026-01-01\\n'",
            True,
            "Blender 9.9.9, Molecular Nodes unknown, molrender 1.2.3",
        ),
        (
            "printf 'Blender 9.9.9\\n'",
            False,
            "Blender 9.9.9, Molecular Nodes unknown, molrender unknown",
        ),
        (
            "echo 'Blender 9.9.9'; exit 1",
            True,
            "Blender unknown, Molecular Nodes unknown, molrender 1.2.3",
        ),
        (None, True, "Blender unknown, Molecular Nodes unknown, molrender 1.2.3"),
    ],
    ids=["all found", "molrender not installed", "blender exits non-zero", "blender not found"],
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


# ---------------------------------------------------------------------------
# protein panels
# ---------------------------------------------------------------------------

# toy-protein.pdb keeps 10 atoms: chain A holds atoms 0-4 (alanine, glycine), zinc (7) and
# magnesium (8); chain B holds atoms 5 and 6 and the ligand carbon (9). Zinc has no palette
# colour, so a protein spec that draws it names one.
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
    scene = module.scene_for(spec, subject, cage, species, multi_species=True, draft=False)
    return scene, subject


def test_prepare_writes_a_protein_as_the_atoms_read_pdb_kept_in_the_files_order(
    render_box_grid_module, tmp_path
):
    subject, cage, species = render_box_grid_module.prepare(PROTEIN_SPEC, tmp_path)
    lines = subject.read_text().splitlines()
    assert cage is None
    assert species == ["N", "C", "C", "N", "C", "N", "C", "Zn", "Mg", "C"]
    assert [line[:5] for line in lines if not line.startswith(("ATOM", "HETATM"))] == [
        "HELIX",
        "SHEET",
        "END",
    ]
    atoms = [line for line in lines if line.startswith(("ATOM", "HETATM"))]
    assert [int(line[6:11]) for line in atoms] == [1, 2, 4, 5, 6, 7, 8, 9, 10, 11]
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


def test_a_protein_is_one_cartoon_layer_over_its_cartoon_chains_and_species_layers_over_the_rest(
    render_box_grid_module, tmp_path
):
    scene, subject = _protein_scene(render_box_grid_module, tmp_path)
    structure_id = StandInStructureId(
        sha256=hashlib.sha256(subject.read_bytes()).hexdigest(), n_atoms=10
    )
    (structure,) = scene.entities  # no cage
    cartoon, sticks, *spheres = structure.layers
    assert not structure.remove_solvent

    # Chain A is atoms 0-4 and the two ions; chain B is atoms 5, 6 and the ligand carbon.
    assert cartoon == StandInLayer(
        style=StandInCartoon(),
        material=StandInSolid(rgb=srgb_hex_to_linear(SPECIES_COLOUR["C"])),
        select=StandInAtomMask(
            mask=(True,) * 5 + (False,) * 2 + (True,) * 2 + (False,), of=structure_id
        ),
    )
    # The sticks and spheres cover the stick chain and every HETATM, and nothing else.
    drawn = (False,) * 5 + (True,) * 5
    assert sticks.select == StandInAtomMask(mask=drawn, of=structure_id)
    # Magnesium is the darkest of the colours drawn, so the sticks take its.
    assert sticks.material == StandInSolid(rgb=srgb_hex_to_linear(SPECIES_COLOUR["Mg"]))
    masks = {
        symbol: layer.select.mask
        for symbol, layer in zip(["C", "Mg", "N", "Zn"], spheres, strict=True)
    }
    assert [layer.material.rgb for layer in spheres] == [
        srgb_hex_to_linear(colour)
        for colour in (SPECIES_COLOUR["C"], SPECIES_COLOUR["Mg"], SPECIES_COLOUR["N"], "#336699")
    ]
    assert masks["C"] == (False,) * 6 + (True,) + (False,) * 2 + (True,)
    assert masks["N"] == (False,) * 5 + (True,) + (False,) * 4
    assert masks["Zn"] == (False,) * 7 + (True,) + (False,) * 2
    assert masks["Mg"] == (False,) * 8 + (True,) + (False,)
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


def test_a_protein_of_cartoon_chains_alone_is_the_cartoon_layer_alone(
    render_box_grid_module, tmp_path
):
    scene, _ = _protein_scene(
        render_box_grid_module, tmp_path, source=str(_chain_a_only(tmp_path)), stick_chains=[]
    )
    (structure,) = scene.entities
    (cartoon,) = structure.layers
    assert isinstance(cartoon.style, StandInCartoon)
    assert cartoon.select.mask == (True,) * 5


def test_the_cartoon_takes_the_manifests_carbon_colour(render_box_grid_module, tmp_path):
    scene, _ = _protein_scene(
        render_box_grid_module, tmp_path, colour={"Zn": "#336699", "C": "#102030"}
    )
    assert scene.entities[0].layers[0].material.rgb == srgb_hex_to_linear("#102030")


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
    assert "panel: 10 atoms" in capsys.readouterr().out


def test_a_protein_panels_summary_line_counts_the_atom_records_read_pdb_dropped(
    render_box_grid_module, monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(render_box_grid_module, "tool_versions", lambda: "VERSIONS")
    spec = {**PROTEIN_SPEC, "id": "panel", "resolution": [1728, 972]}
    _render_panel(
        render_box_grid_module, monkeypatch, tmp_path, spec, (54, 918, 10, 1718), (74, 74, 74)
    )
    assert "panel: 10 atoms (dropped 1 altloc, 1 water), wall" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("stick_chains", "ink_rgb", "mode", "multi_species"),
    [
        (["B"], (200, 60, 40), "RGB", True),
        ([], (74, 74, 74), "L", False),
    ],
    ids=["a peptide drawn by species is coloured", "cartoon chains alone are grey"],
)
def test_a_protein_is_saved_grey_unless_it_draws_an_atom_by_species(
    render_box_grid_module,
    monkeypatch,
    tmp_path,
    scene_calls,
    stick_chains,
    ink_rgb,
    mode,
    multi_species,
):
    # The cartoon-alone file has the atoms of C, N and O, so the old rule (more than one
    # species among all atoms) would have called it coloured.
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
        lambda spec, subject, cage, species, *, multi_species, draft: scene_calls.append(
            {"multi_species": multi_species, "draft": draft}
        ),
    )
    output = _render_panel(
        render_box_grid_module, monkeypatch, tmp_path, spec, (54, 918, 10, 1718), ink_rgb
    )
    assert Image.open(output).mode == mode
    assert scene_calls == [{"multi_species": multi_species, "draft": False}]


def test_a_coloured_cartoon_alone_protein_fails_the_whole_frame_chroma_limit(
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
    with pytest.raises(RuntimeError, match=r"max chroma 160/255 exceeds 32"):
        _render_panel(
            render_box_grid_module, monkeypatch, tmp_path, spec, (54, 918, 10, 1718), (200, 60, 40)
        )
