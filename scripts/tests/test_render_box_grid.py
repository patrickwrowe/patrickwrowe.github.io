"""Tests for render_box_grid.render_panel's crop decision, with Blender stubbed out.

render_box_grid imports molrender, which exists only in the Blender environment, so a
stand-in module takes its place and `render` writes a PNG of the requested size instead of
calling Blender. Everything after the render (the whiteness check, the crop decision and
the move into place) is the real code.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

MOLRENDER_NAMES = (
    "DRAFT",
    "PRODUCTION",
    "BallAndStick",
    "Layer",
    "Scene",
    "Solid",
    "Structure",
    "View",
    "render",
)


@pytest.fixture
def render_box_grid_module(monkeypatch):
    """render_box_grid imported against a stand-in molrender."""
    if "molrender" not in sys.modules:
        stand_in = types.ModuleType("molrender")
        for name in MOLRENDER_NAMES:
            setattr(stand_in, name, None)
        monkeypatch.setitem(sys.modules, "molrender", stand_in)
    return importlib.import_module("render_box_grid")


@pytest.fixture
def render_box_grid(render_box_grid_module, monkeypatch):
    """render_box_grid with `prepare` and `scene_for` stubbed, for render_panel."""
    monkeypatch.setattr(
        render_box_grid_module,
        "prepare",
        lambda spec, work_dir: (work_dir / "subject.pdb", None, 7),
    )
    monkeypatch.setattr(
        render_box_grid_module, "scene_for", lambda spec, subject, cage, draft: None
    )
    return render_box_grid_module


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
    subject, cage, n_atoms = render_box_grid_module.prepare(spec, work_dir)
    hetatm = [line for line in subject.read_text().splitlines() if line.startswith("HETATM")]
    assert cage is None
    assert n_atoms == len(hetatm) == 6
    assert all(line.endswith(" C") for line in hetatm)


def test_prepare_refuses_a_frame_with_oxygen_naming_it(render_box_grid_module, tmp_path):
    source = tmp_path / "co.xyz"
    source.write_text("2\ncarbon monoxide\nC 0.0 0.0 0.0\nO 1.13 0.0 0.0\n")
    spec = {"id": "co", "source": str(source), "cage_angstrom": None}
    with pytest.raises(ValueError, match=r"carbon only, got \['O'\]"):
        render_box_grid_module.prepare(spec, tmp_path)


def _render_writing(resolution_px: tuple[int, int], ink_box: tuple[int, int, int, int]):
    """A `render` stand-in that writes a white RGB frame with one grey block of ink.

    Args:
        resolution_px: (width, height) of the frame, in pixels.
        ink_box: (top, bottom, left, right) of the ink, in pixels.

    Returns:
        A function with molrender.render's signature.
    """

    def render(scene, path: Path, timeout: float) -> types.SimpleNamespace:
        width_px, height_px = resolution_px
        pixels = np.full((height_px, width_px, 3), 255, dtype=np.uint8)
        top, bottom, left, right = ink_box
        pixels[top:bottom, left:right] = 90
        Image.fromarray(pixels, mode="RGB").save(path)
        return types.SimpleNamespace(returncode=0)

    return render


def _render_panel(module, monkeypatch, tmp_path: Path, spec: dict, ink_box) -> Path:
    """Run render_panel on `spec` with the stand-in renderer; return the output PNG."""
    monkeypatch.setattr(module, "render", _render_writing(tuple(spec["resolution"]), ink_box))
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
    assert "crop none (caged)" in capsys.readouterr().out
