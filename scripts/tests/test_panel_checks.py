"""Tests for scripts/figures/panel_checks.py: manifest, cage and pixel checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import panel_checks
import pytest
from PIL import Image

IMAGE_SIZE = 32
GREY_RGB = (200, 200, 200)


def _write_image(path: Path, pixels: np.ndarray) -> None:
    """Save an (height, width, 3) uint8 array as an RGB PNG."""
    Image.fromarray(pixels, mode="RGB").save(path)


def _white_pixels(size: int = IMAGE_SIZE) -> np.ndarray:
    """A size x size x 3 array of pure white pixels."""
    return np.full((size, size, 3), 255, dtype=np.uint8)


def _manifest(tmp_path: Path, panels: list[dict], **plate_overrides: object) -> Path:
    """Write a minimal manifest.json and return its path.

    Each panel's relative source is created as an empty file under `tmp_path` and named
    by its absolute path, so the source-exists rule passes unless a test removes it.
    """
    for panel in panels:
        if isinstance(panel.get("source"), str):
            source = tmp_path / panel["source"]
            source.touch()
            panel["source"] = str(source)
    plate: dict[str, object] = {
        "output_dir": "out",
        "direction": [0.0, 0.0, 1.0],
        "resolution": [100, 100],
        "sphere_radius_angstrom": 0.4,
        "bond_radius_angstrom": 0.2,
        "grey": 0.35,
    }
    plate.update(plate_overrides)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({**plate, "panels": panels}))
    return manifest_path


# ---------------------------------------------------------------------------
# load_panels: manifest validation
# ---------------------------------------------------------------------------


def test_load_panels_merges_plate_defaults_into_each_panel(tmp_path):
    manifest_path = _manifest(
        tmp_path,
        [
            {"id": "a", "source": "a.xyz"},
            {"id": "b", "source": "b.xyz", "grey": 0.5},
        ],
    )
    panels = panel_checks.load_panels(manifest_path)
    assert [panel["id"] for panel in panels] == ["a", "b"]
    assert panels[0]["grey"] == 0.35
    assert panels[1]["grey"] == 0.5
    # The kind's default is set once, here, so no reader applies it again.
    assert [panel["kind"] for panel in panels] == [panel_checks.PanelKind.BOX] * 2


def test_load_panels_requires_a_grey_of_a_box_and_not_of_a_protein(tmp_path):
    manifest_path = _manifest(tmp_path, [{"id": "b", "source": "b.xyz"}], grey=None)
    manifest = json.loads(manifest_path.read_text())
    del manifest["grey"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=r"panel b lacks \['grey'\]"):
        panel_checks.load_panels(manifest_path)
    manifest["panels"] = [{**PROTEIN_PANEL, "source": str(tmp_path / "p.pdb")}]
    (tmp_path / "p.pdb").touch()
    manifest_path.write_text(json.dumps(manifest))
    (protein,) = panel_checks.load_panels(manifest_path)
    assert protein["kind"] == panel_checks.PanelKind.PROTEIN and "grey" not in protein


@pytest.mark.parametrize(
    ("kind", "species", "coloured"),
    [
        ("box", ["C", "C", "C"], False),
        ("box", ["C", "O"], True),
        ("protein", ["C", "C"], True),  # its cartoon chains are coloured
        ("protein", [], True),
    ],
)
def test_a_protein_or_a_frame_of_more_than_one_species_is_coloured(kind, species, coloured):
    assert panel_checks.is_coloured({"kind": kind}, species) is coloured


def test_load_panels_rejects_a_panel_missing_a_required_key(tmp_path):
    manifest_path = _manifest(tmp_path, [{"id": "a"}])  # no "source"
    with pytest.raises(ValueError, match="lacks"):
        panel_checks.load_panels(manifest_path)


def test_load_panels_rejects_duplicate_ids(tmp_path):
    manifest_path = _manifest(
        tmp_path, [{"id": "a", "source": "a.xyz"}, {"id": "a", "source": "b.xyz"}]
    )
    with pytest.raises(ValueError, match="appears twice"):
        panel_checks.load_panels(manifest_path)


def test_load_panels_rejects_a_bad_id(tmp_path):
    manifest_path = _manifest(tmp_path, [{"id": "a b", "source": "a.xyz"}])
    with pytest.raises(ValueError, match="does not match"):
        panel_checks.load_panels(manifest_path)


def test_load_panels_rejects_an_unknown_key(tmp_path):
    # A typo such as "slab_angstorm" for "slab_angstrom" must not silently fall back to
    # the default and render the whole panel wrong.
    manifest_path = _manifest(tmp_path, [{"id": "a", "source": "a.xyz", "slab_angstorm": 9.0}])
    with pytest.raises(ValueError, match="slab_angstorm"):
        panel_checks.load_panels(manifest_path)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("resolution", None),
        ("resolution", [800, 800, 3]),
        ("resolution", "800x800"),
        ("resolution", [800, 0]),
        ("resolution", [800.5, 800]),
        ("grey", 1.5),
        ("grey", -0.1),
        ("grey", True),
        ("cage_angstrom", [44.2, 44.2]),
        ("cage_angstrom", [44.2, -1.0, 11.0]),
        ("density_g_cm3", -1.0),
        ("density_g_cm3", 0),
        ("tile", [0, -1, 1]),
        ("tile", [2, 1]),
        ("margin", 0.5),
        ("margin", "1.02"),
        ("sphere_radius_angstrom", 0),
        ("slab_angstrom", -1),
        ("direction", [0, 0, 0]),
        ("direction", [0, 1]),
        ("direction", [float("nan"), 0, 1]),
        # Python's json module reads Infinity and NaN, which pass every `>= 1` or `> 0` rule.
        ("margin", float("inf")),
        ("margin", float("nan")),
        ("sphere_radius_angstrom", float("inf")),
        ("slab_angstrom", float("inf")),
        ("density_g_cm3", float("inf")),
        ("cage_angstrom", [44.2, float("inf"), 11.0]),
        ("direction", [float("inf"), 0, 1]),
        ("direction", [0, -float("inf"), 1]),
        # A species colour is a map of element symbol to six hex digits.
        ("colour", "#123456"),
        ("colour", ["#123456"]),
        ("colour", {"O": "red"}),
        ("colour", {"O": "#b5533"}),
        ("colour", {"O": "#123456ff"}),
        ("colour", {"O": "123456"}),
        ("colour", {"O": 0x123456}),
        ("colour", {"o": "#123456"}),
        ("colour", {"Oxygen": "#123456"}),
    ],
)
def test_load_panels_rejects_a_bad_value_naming_manifest_panel_and_key(tmp_path, key, value):
    manifest_path = _manifest(tmp_path, [{"id": "a", "source": "a.xyz", key: value}])
    with pytest.raises(ValueError) as error:
        panel_checks.load_panels(manifest_path)
    assert str(manifest_path) in str(error.value)
    assert "panel a" in str(error.value) and key in str(error.value)


def test_load_panels_accepts_a_negative_direction_a_margin_of_one_and_a_null_slab(tmp_path):
    panel_spec = {"id": "a", "source": "a.xyz", "direction": [0, -1, 0], "margin": 1}
    manifest_path = _manifest(tmp_path, [{**panel_spec, "slab_angstrom": None}])
    panel = panel_checks.load_panels(manifest_path)[0]
    assert panel["direction"] == [0, -1, 0] and panel["slab_angstrom"] is None


def test_an_integer_too_large_for_a_float_is_refused_naming_panel_and_key(tmp_path):
    manifest_path = _manifest(tmp_path, [{"id": "a", "source": "a.xyz", "margin": 10**400}])
    with pytest.raises(ValueError) as error:
        panel_checks.load_panels(manifest_path)
    assert str(manifest_path) in str(error.value)
    assert "panel a" in str(error.value) and "margin" in str(error.value)


def test_load_panels_accepts_a_species_colour_map_and_a_panel_map_replaces_the_plates(tmp_path):
    plate_colour = {"O": "#123456", "H": "#ABCDEF"}
    manifest_path = _manifest(
        tmp_path,
        [{"id": "a", "source": "a.xyz"}, {"id": "b", "source": "b.xyz", "colour": {}}],
        colour=plate_colour,
    )
    panels = panel_checks.load_panels(manifest_path)
    assert panels[0]["colour"] == plate_colour
    assert panels[1]["colour"] == {}


def test_load_panels_accepts_a_sphere_scale_and_light_and_quality_overrides(tmp_path):
    manifest_path = _manifest(
        tmp_path,
        [{"id": "a", "source": "a.xyz"}],
        sphere_scale={"Mg": 4},
        light={"under_power": 8.0},
        quality={"diffuse_bounces": 5, "fast_gi_method": "ADD"},
    )
    panel = panel_checks.load_panels(manifest_path)[0]
    assert panel["sphere_scale"] == {"Mg": 4} and panel["light"] == {"under_power": 8.0}


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("sphere_scale", {"Mg": 0}),
        ("sphere_scale", {"mg": 4}),
        ("light", {"under_power": -1}),
        ("light", {"shadow_power": 1}),
        ("quality", {"diffuse_bounces": 0}),
        ("quality", {"diffuse_bounces": True}),
        ("quality", {"fast_gi_method": "MULTIPLY"}),
        ("quality", {"samples": 64}),
    ],
)
def test_load_panels_refuses_a_bad_override_naming_its_key(tmp_path, key, value):
    manifest_path = _manifest(tmp_path, [{"id": "a", "source": "a.xyz", key: value}])
    with pytest.raises(ValueError, match=key):
        panel_checks.load_panels(manifest_path)


def test_load_panels_accepts_null_cage_and_a_tile(tmp_path):
    manifest_path = _manifest(
        tmp_path,
        [{"id": "a", "source": "a.xyz", "cage_angstrom": None, "tile": [2, 1, 1]}],
        density_g_cm3=1.0,
    )
    assert panel_checks.load_panels(manifest_path)[0]["tile"] == [2, 1, 1]


def test_load_panels_rejects_a_missing_source(tmp_path):
    manifest_path = _manifest(tmp_path, [{"id": "a", "source": "a.xyz"}])
    (tmp_path / "a.xyz").unlink()
    with pytest.raises(ValueError, match=r"panel a: source .* existing XYZ file"):
        panel_checks.load_panels(manifest_path)


def test_load_panels_rejects_a_manifest_without_panels(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text('{"output_dir": "out"}')
    with pytest.raises(ValueError, match='no "panels" list'):
        panel_checks.load_panels(manifest_path)


PROTEIN_PANEL = {
    "id": "p",
    "source": "p.pdb",
    "kind": "protein",
    "cartoon_chains": ["A", "B"],
    "stick_chains": ["C"],
}


def test_load_panels_accepts_a_protein_with_a_null_cage_slab_and_no_stick_chains(tmp_path):
    manifest_path = _manifest(
        tmp_path,
        [{**PROTEIN_PANEL, "stick_chains": [], "cage_angstrom": None, "slab_angstrom": None}],
        slab_angstrom=10.0,  # a plate-level slab the panel overrides
    )
    panel = panel_checks.load_panels(manifest_path)[0]
    assert panel["kind"] == "protein" and panel["stick_chains"] == []


def test_load_panels_accepts_a_chain_colour_for_a_cartoon_chain(tmp_path):
    manifest_path = _manifest(tmp_path, [{**PROTEIN_PANEL, "chain_colour": {"B": "#ABCDEF"}}])
    assert panel_checks.load_panels(manifest_path)[0]["chain_colour"] == {"B": "#ABCDEF"}


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"cage_angstrom": [60.0, 60.0, 60.0]}, "cage_angstrom"),
        ({"slab_angstrom": 10.0}, "slab_angstrom"),
        ({"density_g_cm3": 1.0}, "density_g_cm3"),
        ({"tile": [2, 1, 1]}, "tile"),
        ({"stick_chains": ["B", "C"]}, r"\['B'\] are in both"),
        ({"cartoon_chains": []}, "cartoon_chains is"),
        ({"cartoon_chains": ["A", "A"]}, "cartoon_chains is"),
        ({"cartoon_chains": ["AB"]}, "cartoon_chains is"),
        ({"stick_chains": "C"}, "stick_chains is"),
        ({"kind": "cartoon"}, "kind is"),
        ({"source": "p.xyz"}, "not a .pdb file"),
        ({"chain_colour": {"A": "red"}}, "chain_colour is"),
        ({"chain_colour": {"AB": "#123456"}}, "chain_colour is"),
        ({"chain_colour": ["#123456"]}, "chain_colour is"),
        ({"chain_colour": {"C": "#123456"}}, r"chain_colour names chain\(s\) \['C'\]"),
    ],
)
def test_load_panels_refuses_a_protein_that_breaks_a_rule_naming_manifest_and_panel(
    tmp_path, override, expected
):
    manifest_path = _manifest(tmp_path, [{**PROTEIN_PANEL, **override}])
    with pytest.raises(ValueError, match=expected) as error:
        panel_checks.load_panels(manifest_path)
    assert str(manifest_path) in str(error.value) and "panel p" in str(error.value)


@pytest.mark.parametrize("missing", ["cartoon_chains", "stick_chains"])
def test_load_panels_refuses_a_protein_without_both_chain_lists(tmp_path, missing):
    panel = {key: value for key, value in PROTEIN_PANEL.items() if key != missing}
    with pytest.raises(ValueError, match=rf"panel p is a protein and lacks \['{missing}'\]"):
        panel_checks.load_panels(_manifest(tmp_path, [panel]))


@pytest.mark.parametrize(
    "chain_key", [{"cartoon_chains": ["A"]}, {"chain_colour": {"A": "#123456"}}]
)
def test_load_panels_refuses_chain_keys_on_a_box_panel(tmp_path, chain_key):
    panel = {"id": "b", "source": "b.xyz", **chain_key}
    with pytest.raises(ValueError, match="panel b has chain keys but is not a protein"):
        panel_checks.load_panels(_manifest(tmp_path, [panel]))


# ---------------------------------------------------------------------------
# assert_fits_cage: containment guard
# ---------------------------------------------------------------------------


def test_assert_fits_cage_passes_when_the_subject_fits_with_its_spheres():
    positions_angstrom = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [-5.0, 0.0, 0.0]])
    panel_checks.assert_fits_cage(
        positions_angstrom, ["C"] * 3, np.zeros(3), [12.0, 12.0, 12.0], 0.5, "fits-panel"
    )


def test_assert_fits_cage_raises_naming_the_panel_when_it_does_not_fit():
    positions_angstrom = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [-5.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="too-small-panel"):
        panel_checks.assert_fits_cage(
            positions_angstrom, ["C"] * 3, np.zeros(3), [9.0, 9.0, 9.0], 0.5, "too-small-panel"
        )


def test_assert_fits_cage_allows_each_species_the_radius_molecular_nodes_draws_it_at():
    # An atom 5.5 A from the centre of a 12 A cage, spheres 0.4 A for carbon: carbon reaches
    # 5.9 A and fits; potassium is drawn at 0.4 x 2.75 / 1.70 = 0.65 A and reaches 6.15 A,
    # past the 6 A wall.
    positions_angstrom = np.array([[0.0, 0.0, 0.0], [5.5, 0.0, 0.0]])
    panel_checks.assert_fits_cage(
        positions_angstrom, ["C", "C"], np.zeros(3), [12.0, 12.0, 12.0], 0.4, "carbon"
    )
    with pytest.raises(ValueError, match=r"panel potassium: .* needs a cage of \[12\.29, "):
        panel_checks.assert_fits_cage(
            positions_angstrom, ["C", "K"], np.zeros(3), [12.0, 12.0, 12.0], 0.4, "potassium"
        )


# ---------------------------------------------------------------------------
# check_and_flatten: rendered-PNG whiteness and chroma checks
# ---------------------------------------------------------------------------


def test_a_clean_white_image_passes(tmp_path):
    png = tmp_path / "clean.png"
    _write_image(png, _white_pixels())
    strays = panel_checks.check_and_flatten(png, (IMAGE_SIZE, IMAGE_SIZE))
    assert strays == []
    assert Image.open(png).mode == "L"


def test_one_isolated_corner_stray_passes_and_is_reported(tmp_path):
    pixels = _white_pixels()
    pixels[0, 0] = GREY_RGB
    png = tmp_path / "stray.png"
    _write_image(png, pixels)
    strays = panel_checks.check_and_flatten(png, (IMAGE_SIZE, IMAGE_SIZE))
    assert strays == [(0, 0, list(GREY_RGB))]


def test_a_three_pixel_run_on_the_border_fails(tmp_path):
    pixels = _white_pixels()
    pixels[0, 0:3] = GREY_RGB
    png = tmp_path / "run.png"
    _write_image(png, pixels)
    with pytest.raises(RuntimeError, match="non-white border pixels"):
        panel_checks.check_and_flatten(png, (IMAGE_SIZE, IMAGE_SIZE))


def test_two_adjacent_border_pixels_fail(tmp_path):
    pixels = _white_pixels()
    pixels[0, 0] = GREY_RGB
    pixels[0, 1] = GREY_RGB
    png = tmp_path / "adjacent.png"
    _write_image(png, pixels)
    with pytest.raises(RuntimeError, match="non-white neighbour"):
        panel_checks.check_and_flatten(png, (IMAGE_SIZE, IMAGE_SIZE))


def test_a_border_pixel_with_an_interior_neighbour_fails(tmp_path):
    pixels = _white_pixels()
    pixels[0, 5] = GREY_RGB  # on the border
    pixels[1, 5] = GREY_RGB  # one row in, but still touches the border pixel
    png = tmp_path / "interior-neighbour.png"
    _write_image(png, pixels)
    with pytest.raises(RuntimeError, match="non-white neighbour"):
        panel_checks.check_and_flatten(png, (IMAGE_SIZE, IMAGE_SIZE))


def test_three_isolated_strays_fail(tmp_path):
    pixels = _white_pixels()
    for column in (2, 14, 26):  # far enough apart that none is another's neighbour
        pixels[0, column] = GREY_RGB
    png = tmp_path / "three-strays.png"
    _write_image(png, pixels)
    with pytest.raises(RuntimeError, match="non-white border pixels"):
        panel_checks.check_and_flatten(png, (IMAGE_SIZE, IMAGE_SIZE))


def test_a_subject_clipped_at_an_edge_fails(tmp_path):
    # A subject rendered too large for its cage runs off the frame: a solid run of
    # colour along the middle of one edge (away from the corner blocks), not a stray
    # pixel or two.
    pixels = _white_pixels()
    pixels[12:20, 0] = GREY_RGB
    png = tmp_path / "clipped.png"
    _write_image(png, pixels)
    with pytest.raises(RuntimeError, match="non-white border pixels"):
        panel_checks.check_and_flatten(png, (IMAGE_SIZE, IMAGE_SIZE))


def test_an_agx_grey_world_fails(tmp_path):
    # The AgX view transform maps a linear-white world to a mid grey (measured 207/255,
    # render_box_grid.py's module docstring); the corner-block median must catch this
    # even though no single border pixel looks like a defect.
    pixels = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 207, dtype=np.uint8)
    png = tmp_path / "agx.png"
    _write_image(png, pixels)
    with pytest.raises(RuntimeError, match="corner block medians are not white"):
        panel_checks.check_and_flatten(png, (IMAGE_SIZE, IMAGE_SIZE))


@pytest.mark.parametrize("case", ["size", "corner", "strays", "neighbour", "chroma"])
def test_every_failed_check_names_the_label_given_not_the_scratch_file(tmp_path, case):
    pixels = _white_pixels()
    expected_size = (IMAGE_SIZE, IMAGE_SIZE)
    if case == "size":
        expected_size = (IMAGE_SIZE + 1, IMAGE_SIZE)
    elif case == "corner":
        pixels[:] = 207
    elif case == "strays":
        for column in (2, 14, 26):
            pixels[0, column] = GREY_RGB
    elif case == "neighbour":
        pixels[0, 12] = pixels[1, 12] = GREY_RGB
    else:
        pixels[16, 16] = (255, 0, 0)
    png = tmp_path / "scratch-file.png"
    _write_image(png, pixels)

    with pytest.raises(RuntimeError) as error:
        panel_checks.check_and_flatten(png, expected_size, label="density-1.0")

    assert str(error.value).startswith("density-1.0: ")
    assert "scratch-file" not in str(error.value)


def test_without_a_label_a_failed_check_names_the_png(tmp_path):
    png = tmp_path / "scratch-file.png"
    _write_image(png, np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 207, dtype=np.uint8))
    with pytest.raises(RuntimeError, match="scratch-file.png: corner block medians"):
        panel_checks.check_and_flatten(png, (IMAGE_SIZE, IMAGE_SIZE))


# ---------------------------------------------------------------------------
# check_and_flatten: a panel of more than one species is coloured on purpose
# ---------------------------------------------------------------------------

RED_RGB = (200, 60, 40)
PALE_RGB = (150, 150, 150)


def _coloured_panel(
    tmp_path: Path, extra: dict[tuple[int, int], tuple[int, int, int]] | None = None
):
    """A 32 x 32 white RGB panel with a red centre, a pale-grey one beside it and `extra`.

    Args:
        tmp_path: Directory for the PNG.
        extra: Pixel (row, column) to RGB, set after the two blobs.

    Returns:
        The path of the saved PNG.
    """
    pixels = _white_pixels()
    pixels[12:20, 10:16] = RED_RGB
    pixels[12:20, 16:20] = PALE_RGB
    for (row, column), rgb in (extra or {}).items():
        pixels[row, column] = rgb
    png = tmp_path / "coloured.png"
    _write_image(png, pixels)
    return png


def test_a_coloured_centre_in_a_white_world_passes_and_keeps_its_colour(tmp_path):
    png = _coloured_panel(tmp_path)
    strays = panel_checks.check_and_flatten(png, (IMAGE_SIZE, IMAGE_SIZE), coloured=True)
    assert strays == []
    saved = Image.open(png)
    assert saved.mode == "RGB"
    assert saved.getpixel((12, 15)) == RED_RGB and saved.getpixel((18, 17)) == PALE_RGB


def test_the_same_coloured_panel_fails_the_chroma_limit_when_it_is_one_species(tmp_path):
    png = _coloured_panel(tmp_path)
    with pytest.raises(RuntimeError, match=r"max chroma 160/255 exceeds 32"):
        panel_checks.check_and_flatten(png, (IMAGE_SIZE, IMAGE_SIZE))


def test_a_coloured_corner_block_fails_naming_the_corner_blocks(tmp_path):
    pixels = _white_pixels()
    pixels[: panel_checks.CORNER_BLOCK_PX, : panel_checks.CORNER_BLOCK_PX] = RED_RGB
    png = tmp_path / "coloured-corner.png"
    _write_image(png, pixels)
    with pytest.raises(RuntimeError, match="corner block medians are not white"):
        panel_checks.check_and_flatten(png, (IMAGE_SIZE, IMAGE_SIZE), coloured=True)


def test_one_coloured_pixel_in_a_corner_block_fails_the_border_chroma_limit(tmp_path):
    # One pixel does not move a corner block's median, and (3, 3) is not on the outer
    # border, so only the chroma limit over the 8 px border can catch it.
    png = _coloured_panel(tmp_path, {(3, 3): RED_RGB})
    with pytest.raises(RuntimeError, match=r"max chroma 160/255 in the 8 px border"):
        panel_checks.check_and_flatten(png, (IMAGE_SIZE, IMAGE_SIZE), coloured=True)


def test_the_border_chroma_limit_is_the_same_32_as_a_grey_panels(tmp_path):
    # Pale tints in the border are the HDRI residual a grey panel tolerates (it is
    # measured at 14-16/255); a tint past MAX_CHROMA is a coloured world.
    tint = (255, 255, 255 - panel_checks.MAX_CHROMA)
    png = _coloured_panel(tmp_path, {(4, 16): tint})
    panel_checks.check_and_flatten(png, (IMAGE_SIZE, IMAGE_SIZE), coloured=True)
    png = _coloured_panel(tmp_path, {(4, 16): (255, 255, 255 - panel_checks.MAX_CHROMA - 1)})
    with pytest.raises(RuntimeError, match="in the 8 px border"):
        panel_checks.check_and_flatten(png, (IMAGE_SIZE, IMAGE_SIZE), coloured=True)


def test_a_coloured_panel_still_fails_the_stray_rule_on_its_outer_border(tmp_path):
    png = _coloured_panel(tmp_path, {(0, 12): RED_RGB, (0, 13): RED_RGB})
    with pytest.raises(RuntimeError, match="non-white neighbour"):
        panel_checks.check_and_flatten(png, (IMAGE_SIZE, IMAGE_SIZE), coloured=True)
    png = _coloured_panel(tmp_path, {(0, 2): RED_RGB, (0, 14): RED_RGB, (0, 26): RED_RGB})
    with pytest.raises(RuntimeError, match="non-white border pixels"):
        panel_checks.check_and_flatten(png, (IMAGE_SIZE, IMAGE_SIZE), coloured=True)


SMALLEST_COMMITTED_PANEL = (
    Path(__file__).resolve().parents[2] / "src/content/work/figures/carbon/density-0.5gcc-3500K.png"
)


@pytest.mark.parametrize("source", ["synthetic", "committed render"])
def test_a_grey_panel_is_flattened_to_the_same_bytes_as_before_species_colour(tmp_path, source):
    # What check_and_flatten wrote for a grey panel before it knew about species: the
    # renderer's RGB frame, converted to mode L and saved with `optimize`.
    if source == "synthetic":
        pixels = _white_pixels()
        pixels[10:22, 10:22] = 90
        pixels[12:20, 12:20] = 140
        frame = Image.fromarray(pixels, mode="RGB")
    else:
        frame = Image.open(SMALLEST_COMMITTED_PANEL).convert("RGB")
    png = tmp_path / "grey.png"
    frame.save(png)
    reference = tmp_path / "reference.png"
    frame.convert("L").save(reference, optimize=True)

    panel_checks.check_and_flatten(png, frame.size)

    assert png.read_bytes() == reference.read_bytes()
    assert Image.open(png).mode == "L"


def _grey_panel(tmp_path, blob: tuple[int, int, int, int], strays: list[tuple[int, int]]):
    """A 200 x 200 white panel with a dark blob (top, bottom, left, right) and lone dark pixels."""
    pixels = np.full((200, 200), 255, dtype=np.uint8)
    top, bottom, left, right = blob
    pixels[top:bottom, left:right] = 90
    for row, column in strays:
        pixels[row, column] = 0
    png = tmp_path / "panel.png"
    Image.fromarray(pixels, mode="L").save(png)
    return png


def test_crop_to_ink_squares_the_blob_and_ignores_isolated_pixels(tmp_path):
    png = _grey_panel(tmp_path, (50, 90, 120, 150), strays=[(0, 10), (100, 100), (199, 199)])
    side = panel_checks.crop_to_ink(png)
    assert side == round(40 * 1.06)
    cropped = np.asarray(Image.open(png))
    assert cropped.shape == (side, side)
    rows, columns = np.nonzero(cropped < 245)
    assert rows.min() >= 1 and columns.min() >= 1  # the margin survives on every side
    assert rows.max() <= side - 2 and columns.max() <= side - 2


def test_crop_to_ink_shifts_the_square_inside_the_image_edge(tmp_path):
    png = _grey_panel(tmp_path, (0, 60, 0, 30), strays=[])
    side = panel_checks.crop_to_ink(png)
    assert side == round(60 * 1.06)
    cropped = np.asarray(Image.open(png))
    assert cropped.shape == (side, side)
    assert (cropped[:60, :30] == 90).all(), "the shift lost ink"


def test_crop_to_ink_gives_up_margin_not_ink_when_the_ink_nearly_fills_a_square(tmp_path):
    # 195 px of ink plus 3% a side would need a 207 px square in a 200 px image: the
    # square is clamped to the image, so the crop is the whole frame and no ink is lost.
    # (C60-2000K is 7 px from this; before 2026-10-02 the clamp was a RuntimeError.)
    png = _grey_panel(tmp_path, (2, 197, 40, 60), strays=[])
    assert panel_checks.crop_to_ink(png) == 200
    cropped = np.asarray(Image.open(png))
    assert cropped.shape == (200, 200)
    assert (cropped[2:197, 40:60] == 90).all(), "the clamp lost ink"


def test_crop_to_ink_raises_on_the_hero_rather_than_square_it(tmp_path):
    # The hero's shape: a 1728 x 972 frame whose ink spans 1708 x 864 px. Squared, it
    # would lose 44% of its width; render_box_grid never crops it, and this is the guard.
    pixels = np.full((972, 1728), 255, dtype=np.uint8)
    pixels[54:918, 10:1718] = 90
    png = tmp_path / "hero.png"
    Image.fromarray(pixels, mode="L").save(png)
    with pytest.raises(RuntimeError, match="does not fit a square inside the 1728 x 972 px"):
        panel_checks.crop_to_ink(png)


def test_crop_to_ink_refuses_an_empty_panel(tmp_path):
    png = _grey_panel(tmp_path, (0, 0, 0, 0), strays=[(100, 100)])
    with pytest.raises(RuntimeError, match="no connected ink"):
        panel_checks.crop_to_ink(png)


def test_crop_to_ink_crops_a_coloured_panel_as_it_does_a_grey_one_and_keeps_its_colour(tmp_path):
    grey = _grey_panel(tmp_path, (50, 90, 120, 150), strays=[(0, 10), (100, 100), (199, 199)])
    grey_side = panel_checks.crop_to_ink(grey)

    pixels = np.full((200, 200, 3), 255, dtype=np.uint8)
    pixels[50:90, 120:150] = RED_RGB
    for row, column in [(0, 10), (100, 100), (199, 199)]:
        pixels[row, column] = (0, 0, 0)
    coloured = tmp_path / "coloured-panel.png"
    _write_image(coloured, pixels)

    assert panel_checks.crop_to_ink(coloured) == grey_side == round(40 * 1.06)
    cropped = Image.open(coloured)
    assert cropped.mode == "RGB" and cropped.size == (grey_side, grey_side)
    assert (np.asarray(cropped) == RED_RGB).all(axis=2).any(), "the crop lost the colour"
