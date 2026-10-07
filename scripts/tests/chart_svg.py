"""Read a figure's drawing back, the way a reader would, for the chart tests.

The figure scripts mark each panel `<g data-panel=...>` and each curve `<g data-series=...>`.
These functions read the SVG text alone: they calibrate a panel's axes from the printed tick
labels and where they sit, then turn the curves' points back into data units, so a test
checks what is on the page whatever function placed it.
"""

from __future__ import annotations

import re

import numpy as np
from figure_style import LABEL_SIZE

# The mono face advances 0.6 em, so a label of n characters at size s is 0.6 n s wide.
MONO_ADVANCE_EM = 0.6
TEXT = re.compile(
    r'<text x="([\d.]+)" y="([\d.]+)" text-anchor="(\w+)" font-size="([\d.]+)"([^>]*)>([^<]*)<'
)
NUMBER = re.compile(r"−?\d+(?:\.\d+)?")


def panel_body(svg: str, **attributes: str) -> str:
    """The inside of the `<g data-...>` panel whose opening tag has exactly these attributes.

    Args:
        svg: The figure.
        **attributes: `data-` attributes without the prefix, in the order they are written,
            e.g. `panel="K", variable="theta"`.

    Returns:
        The panel's SVG text.
    """
    opening = "".join(f' data-{key}="{value}"' for key, value in attributes.items())
    return re.search(rf"<g{opening}>(.*?)\n  </g>", svg, re.DOTALL).group(1)


def read_curves(panel: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each curve of a panel in the units of its axes, read from the drawing alone.

    A bottom axis's numeric labels are centred under their ticks; a left axis's end at the
    tick with their baseline 1.0 below it (figure_style.svg_axis).

    Args:
        panel: A panel's SVG text.

    Returns:
        Series key -> (x values, y values) per drawn point.
    """
    labels = [
        (float(x), float(y), anchor, float(NUMBER.fullmatch(content).group(0).replace("−", "-")))
        for x, y, anchor, _size, _attributes, content in TEXT.findall(panel)
        if NUMBER.fullmatch(content)
    ]
    along = np.array([[x, value] for x, _y, anchor, value in labels if anchor == "middle"])
    up = np.array([[y - 1.0, value] for _x, y, anchor, value in labels if anchor == "end"])
    x_slope, x_offset = np.polyfit(along[:, 1], along[:, 0], 1)
    y_slope, y_offset = np.polyfit(up[:, 1], up[:, 0], 1)
    curves = {}
    for series, body in re.findall(r'<g data-series="(\w+)">(.*?)</g>', panel, re.DOTALL):
        points = np.array(
            [
                [float(number) for number in pair.split(",")]
                for pair in re.search(r'points="([^"]+)"', body).group(1).split()
            ]
        )
        curves[series] = ((points[:, 0] - x_offset) / x_slope, (points[:, 1] - y_offset) / y_slope)
    return curves


def assert_text_fits(svg: str) -> None:
    """Every `<text>` is at least `LABEL_SIZE` and lies inside the viewBox.

    Args:
        svg: The figure.
    """
    width, height = map(float, re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg).groups())
    texts = TEXT.findall(svg)
    assert texts
    for x, y, anchor, size, attributes, content in texts:
        assert float(size) >= LABEL_SIZE, content
        extent = MONO_ADVANCE_EM * len(content) * float(size)
        reach = {"start": 0.0, "middle": 0.5, "end": 1.0}[anchor] * extent
        if "rotate" in attributes:  # runs up the page from its anchor
            assert float(y) - reach >= 0 and float(y) - reach + extent <= height, content
        else:
            assert float(x) - reach >= 0 and float(x) - reach + extent <= width, content
