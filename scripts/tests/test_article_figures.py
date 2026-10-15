"""The theme articles' inlined charts against the SVG files they inline.

Every chart wrapper in `src/content/work/*.mdx` inlines a `?raw` SVG at a `--chart-em` that
sets its width at the page's label size (Plate.astro). That number must be the chart's own:
its viewBox width in labels, which each figure script writes on the `<svg>` root as
`data-chart-em`. This test reads both from the files, and recomputes the
attribute from the SVG's viewBox and its smallest text, so a re-run with a new width that
nobody copied into the MDX fails here rather than shrinking labels under 11 px.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import boxprep
import numpy as np
import panel_checks
import pytest
from constants import SIZES, TEMPERATURES_KELVIN
from figure_style import NARROW_CHART_EM

REPO = Path(__file__).resolve().parents[2]
ARTICLES = sorted((REPO / "src" / "content" / "work").glob("*.mdx"))
ARTICLE = REPO / "src" / "content" / "work" / "carbon.mdx"
MDX = ARTICLE.read_text()
GRID_CSS = (REPO / "src" / "styles" / "grid.css").read_text()
MANIFEST_PANELS = {
    panel["id"]: panel
    for manifest in (REPO / "scripts" / "figures" / "data").glob("**/manifest.json")
    for panel in panel_checks.load_panels(manifest)
}
WEIGHTED_GRIDS = [
    (
        [float(weight) for weight in weights.split(",")],
        re.findall(r'"([^"]+)\.png"', panels),
    )
    for weights, panels in re.findall(r"weights=\{\[([^\]]+)\]\}\s*panels=\{\[([^\]]+)\]\}", MDX)
]
RAW_IMPORT = re.compile(r'^import (\w+) from "(\./[^"]+\.svg)\?raw";$', re.M)
WRAPPER = re.compile(
    r'<div class="(chart(?: chart--narrow)?)" style="--chart-em: ([\d.]+)"[^>]*>\s*'
    r"<Fragment set:html=\{(\w+)\} />"
)
RAW_IMPORTS = dict(RAW_IMPORT.findall(MDX))
WRAPPERS = [
    (article, *wrapper) for article in ARTICLES for wrapper in WRAPPER.findall(article.read_text())
]


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
    assert sorted(name for article, _, _, name in WRAPPERS if article == ARTICLE) == sorted(
        RAW_IMPORTS
    )


@pytest.mark.parametrize("article", ARTICLES, ids=lambda article: article.stem)
def test_every_chart_em_in_an_article_is_in_a_wrapper_the_pattern_reads(article):
    # A wrapper written in a form WRAPPER misses would be skipped by the test below, silently.
    assert article.read_text().count("--chart-em") == sum(
        1 for found, *_ in WRAPPERS if found == article
    )


@pytest.mark.parametrize(
    ("article", "wrapper_class", "mdx_chart_em", "name"),
    WRAPPERS,
    ids=[f"{article.stem}-{name}" for article, _, _, name in WRAPPERS],
)
def test_the_wrapper_chart_em_is_the_charts_own(article, wrapper_class, mdx_chart_em, name):
    raw_imports = dict(RAW_IMPORT.findall(article.read_text()))
    svg = (article.parent / raw_imports[name]).read_text()
    attribute, measured = chart_em_of(svg)
    assert float(mdx_chart_em) == attribute, f"{name}: MDX {mdx_chart_em}, SVG {attribute}"
    assert attribute == pytest.approx(measured, abs=0.05), f"{name}: drawn at {measured:.2f} em"
    if wrapper_class == "chart chart--narrow":
        assert attribute <= NARROW_CHART_EM, f"{name} is wider than the phone budget"


def framed_width_angstrom(cage_angstrom: list[float], direction: list[float]) -> float:
    """The width of world space an orthographic molrender camera frames round a cage.

    molrender's `place_camera` (molrender/blender/view.py) projects the corners of
    everything loaded onto the camera's right and up axes and sets `ortho_scale` to the
    larger full extent, times the margin; for a square render that is the frame's width.
    The camera looks along -direction with z up. The margin is left out: it is the same
    for every panel of a row, so it cancels from their ratios.

    Args:
        cage_angstrom: [x, y, z] full extents of the cage, in angstrom.
        direction: The manifest's vector from the subject towards the camera.

    Returns:
        The framed width, in angstrom.
    """
    forward = -np.asarray(direction, dtype=float) / np.linalg.norm(direction)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    corners = signs * np.asarray(cage_angstrom) / 2
    return float(2 * max(np.abs(corners @ right).max(), np.abs(corners @ up).max()))


def test_the_density_row_is_weighted_and_caged():
    assert len(WEIGHTED_GRIDS) == 1
    _weights, files = WEIGHTED_GRIDS[0]
    assert all(file.startswith("density-") for file in files)


@pytest.mark.parametrize(("weights", "files"), WEIGHTED_GRIDS)
def test_each_weight_is_the_width_its_cage_is_framed_at(weights, files):
    # Equal angstrom per page pixel needs each column in proportion to the world width its
    # square render spans, which is what molrender frames round the panel's cage. The
    # weights are then in the same units only if every render is square and every panel
    # is framed with the same margin, so both are asserted here, not assumed.
    assert len({MANIFEST_PANELS[file].get("margin") for file in files}) == 1
    for weight, file in zip(weights, files, strict=True):
        panel = MANIFEST_PANELS[file]
        assert panel["resolution"][0] == panel["resolution"][1], f"{file} is not a square render"
        assert panel.get("cage_angstrom"), f"{file} shares a scale but has no cage"
        framed = framed_width_angstrom(panel["cage_angstrom"], panel["direction"])
        assert weight == round(framed, 1), f"{file}: weight {weight}, framed {framed:.3f} A"


@pytest.mark.parametrize(("weights", "files"), WEIGHTED_GRIDS)
def test_each_cage_is_its_cell_edge_plus_a_sphere_diameter(weights, files):
    # Rounded up to 0.1 A, so the containment guard holds for an atom on a cell face.
    for file in files:
        panel = MANIFEST_PANELS[file]
        edge_angstrom = boxprep.box_edge_from_density(5832, panel["density_g_cm3"])
        needed_angstrom = edge_angstrom + 2 * panel["sphere_radius_angstrom"]
        side_angstrom = math.ceil(needed_angstrom * 10) / 10
        assert panel["cage_angstrom"] == [side_angstrom, side_angstrom, 11.0], file


def test_the_weight_floor_gives_the_smallest_weighted_panel_100_px():
    floor_px = float(re.search(r"--floor-per-weight: ([\d.]+)px", GRID_CSS).group(1))
    smallest = min(min(weights) for weights, _ in WEIGHTED_GRIDS)
    assert 100 <= floor_px * smallest < 101


def test_the_archive_grid_is_generated_from_the_axes_the_census_runs_on():
    # The 48-run grid names its panels from two arrays in the MDX; they must be the run
    # matrix, or a panel is missing or a file that does not exist is asked for.
    def exported_numbers(name: str) -> tuple[int, ...]:
        body = re.search(rf"^export const {name} = \[([\d, ]+)\];$", MDX, re.M).group(1)
        return tuple(int(number) for number in body.split(","))

    assert exported_numbers("ARCHIVE_SIZES") == SIZES
    assert exported_numbers("ARCHIVE_TEMPERATURES_KELVIN") == TEMPERATURES_KELVIN
