# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow"]
# ///
"""Crop a single panel out of a reproduced journal figure, pixel-exact.

`Image.crop` does not resample: it copies pixels, so the panel is not softened the way
a resize would soften it. The box is inclusive-exclusive in `(left, top, right, bottom)`
order, matching `Image.crop` itself, so callers can read pixel coordinates straight off
an image viewer without converting them.

Usage (panel B of the graphene thermal-expansion figure, the lattice parameter
normalised to each model's 60 K value):
    .venv/bin/python scripts/figures/crop_panel.py \\
        --input src/content/work/figures/graphene-potential/fig2-thermal-expansion.png \\
        --box 0 570 848 1110 \\
        --output src/content/work/figures/carbon/graphene-thermal-expansion-panel-b.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

Box = tuple[int, int, int, int]


def crop_panel(input_path: Path, box: Box, output_path: Path) -> None:
    """Crop a PNG to `box` and write the result, without resampling.

    Args:
        input_path: Source PNG to read.
        box: Pixel box as `(left, top, right, bottom)`, inclusive-exclusive like
            `Image.crop`.
        output_path: Destination PNG to write; parent directories are created as
            needed.

    Raises:
        ValueError: If `box` is degenerate (left >= right or top >= bottom) or falls
            outside the source image's bounds.
    """
    left, top, right, bottom = box
    with Image.open(input_path) as image:
        width, height = image.size
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError(f"box {box} is outside the {width} x {height} image {input_path}")
        cropped = image.crop(box)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(output_path)


def main(argv: list[str] | None = None) -> None:
    """Crop a PNG panel from the command line.

    Args:
        argv: Command-line arguments; None reads `sys.argv`.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--box",
        type=int,
        nargs=4,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    box: Box = tuple(args.box)  # type: ignore[assignment]
    crop_panel(args.input, box, args.output)
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
