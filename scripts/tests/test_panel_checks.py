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
    """Write a minimal manifest.json and return its path."""
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


# ---------------------------------------------------------------------------
# assert_fits_cage: containment guard
# ---------------------------------------------------------------------------


def test_assert_fits_cage_passes_when_the_subject_fits_with_its_spheres():
    positions_angstrom = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [-5.0, 0.0, 0.0]])
    panel_checks.assert_fits_cage(
        positions_angstrom, np.zeros(3), [12.0, 12.0, 12.0], 0.5, "fits-panel"
    )


def test_assert_fits_cage_raises_naming_the_panel_when_it_does_not_fit():
    positions_angstrom = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [-5.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="too-small-panel"):
        panel_checks.assert_fits_cage(
            positions_angstrom, np.zeros(3), [9.0, 9.0, 9.0], 0.5, "too-small-panel"
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


def test_crop_to_ink_raises_rather_than_cut_ink_that_does_not_fit(tmp_path):
    # 195 px of ink plus 3% a side needs a 207 px square in a 200 px image: clamping it
    # would cut the subject, so it raises and leaves the file untouched.
    png = _grey_panel(tmp_path, (2, 197, 40, 60), strays=[])
    before = png.read_bytes()
    with pytest.raises(RuntimeError, match="207 px square, larger than the 200 x 200 px"):
        panel_checks.crop_to_ink(png)
    assert png.read_bytes() == before


def test_crop_to_ink_raises_on_the_hero_rather_than_square_it(tmp_path):
    # The hero's shape: a 1728 x 972 frame whose ink spans 1708 x 864 px. Squared, it
    # would lose 44% of its width; render_box_grid never crops it, and this is the guard.
    pixels = np.full((972, 1728), 255, dtype=np.uint8)
    pixels[54:918, 10:1718] = 90
    png = tmp_path / "hero.png"
    Image.fromarray(pixels, mode="L").save(png)
    with pytest.raises(RuntimeError, match="larger than the 1728 x 972 px image"):
        panel_checks.crop_to_ink(png)


def test_crop_to_ink_refuses_an_empty_panel(tmp_path):
    png = _grey_panel(tmp_path, (0, 0, 0, 0), strays=[(100, 100)])
    with pytest.raises(RuntimeError, match="no connected ink"):
        panel_checks.crop_to_ink(png)
