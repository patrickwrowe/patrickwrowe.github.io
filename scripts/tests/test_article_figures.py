"""The carbon article's inlined charts against the SVG files they inline.

Every `?raw` SVG import in `src/content/work/carbon.mdx` is inlined by a chart wrapper
whose `--chart-em` sets its width at the page's label size (Plate.astro). That number must
be the chart's own: its viewBox width in labels, which each figure script writes on the
`<svg>` root as `data-chart-em`. This test reads both from the files, and recomputes the
attribute from the SVG's viewBox and its smallest text, so a re-run with a new width that
nobody copied into the MDX fails here rather than shrinking labels under 11 px.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from figure_style import NARROW_CHART_EM

REPO = Path(__file__).resolve().parents[2]
ARTICLE = REPO / "src" / "content" / "work" / "carbon.mdx"
MDX = ARTICLE.read_text()
RAW_IMPORTS = dict(re.findall(r'^import (\w+) from "(\./[^"]+\.svg)\?raw";$', MDX, re.M))
WRAPPERS = re.findall(
    r'<div class="(chart(?: chart--narrow)?)" style="--chart-em: ([\d.]+)"[^>]*>\s*'
    r"<Fragment set:html=\{(\w+)\} />",
    MDX,
)


def chart_em_of(svg: str) -> tuple[float, float]:
    """The `data-chart-em` on an SVG's root, and the same number measured from the drawing.

    Args:
        svg: The SVG document.

    Returns:
        (the attribute's value, the viewBox width over the smallest `<text>` font size).
    """
    root = re.search(r"<svg [^>]*>", svg).group(0)
    attribute = float(re.search(r'data-chart-em="([\d.]+)"', root).group(1))
    viewbox_width = float(re.search(r'viewBox="[-\d.]+ [-\d.]+ ([\d.]+) ', root).group(1))
    text_sizes = re.findall(r'<text [^>]*font-size="([\d.]+)"', svg)
    smallest_label = min(float(size) for size in text_sizes)
    return attribute, viewbox_width / smallest_label


def test_every_raw_svg_import_is_inlined_by_one_chart_wrapper():
    assert RAW_IMPORTS, "no ?raw SVG imports found in carbon.mdx"
    assert sorted(name for _, _, name in WRAPPERS) == sorted(RAW_IMPORTS)


@pytest.mark.parametrize(("wrapper_class", "mdx_chart_em", "name"), WRAPPERS)
def test_the_wrapper_chart_em_is_the_charts_own(wrapper_class, mdx_chart_em, name):
    svg = (ARTICLE.parent / RAW_IMPORTS[name]).read_text()
    attribute, measured = chart_em_of(svg)
    assert float(mdx_chart_em) == attribute, f"{name}: MDX {mdx_chart_em}, SVG {attribute}"
    assert attribute == pytest.approx(measured, abs=0.05), f"{name}: drawn at {measured:.2f} em"
    if wrapper_class == "chart chart--narrow":
        assert attribute <= NARROW_CHART_EM, f"{name} is wider than the phone budget"
