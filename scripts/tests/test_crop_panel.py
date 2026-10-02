"""Tests for scripts/figures/crop_panel.py."""

from __future__ import annotations

from pathlib import Path

import crop_panel
import numpy as np
import pytest
from PIL import Image

IMAGE_SIZE = 40
MARK_BOX = (10, 10, 20, 20)  # left, top, right, bottom: a 10 x 10 marked region.
BACKGROUND_RGB = (255, 255, 255)
MARK_RGB = (10, 20, 30)


def _make_marked_image() -> Image.Image:
    """A BACKGROUND_RGB square with a MARK_RGB block pasted at MARK_BOX."""
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), BACKGROUND_RGB)
    marked_region = Image.new(
        "RGB", (MARK_BOX[2] - MARK_BOX[0], MARK_BOX[3] - MARK_BOX[1]), MARK_RGB
    )
    image.paste(marked_region, MARK_BOX[:2])
    return image


def test_crop_returns_exactly_the_marked_region(tmp_path: Path):
    input_path = tmp_path / "source.png"
    output_path = tmp_path / "cropped.png"
    _make_marked_image().save(input_path)

    crop_panel.crop_panel(input_path, MARK_BOX, output_path)

    with Image.open(output_path) as cropped:
        assert cropped.size == (10, 10)
        pixels = np.array(cropped).reshape(-1, 3)
        assert (pixels == MARK_RGB).all()


def test_box_outside_the_image_raises_value_error(tmp_path: Path):
    input_path = tmp_path / "source.png"
    output_path = tmp_path / "cropped.png"
    _make_marked_image().save(input_path)

    with pytest.raises(ValueError, match="outside the 40 x 40 image"):
        crop_panel.crop_panel(input_path, (30, 30, IMAGE_SIZE + 10, IMAGE_SIZE + 10), output_path)


def test_an_inverted_box_is_reported_as_inverted_not_outside(tmp_path: Path):
    # The user review's case: left greater than right, every coordinate inside the image.
    input_path = tmp_path / "source.png"
    _make_marked_image().save(input_path)
    with pytest.raises(ValueError, match="inverted or empty"):
        crop_panel.crop_panel(input_path, (30, 5, 10, 35), tmp_path / "cropped.png")
