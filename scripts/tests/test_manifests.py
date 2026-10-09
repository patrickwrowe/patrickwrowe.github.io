"""Every committed render manifest against the files it names, without Blender.

For each panel of each `scripts/figures/data/**/manifest.json`: the source frame exists,
the rendered PNG exists, and the PNG has the size render_box_grid.py writes. A caged panel
and a non-square one keep their full `resolution`; an uncaged square panel is cropped to a
square round its ink (panel_checks.crop_to_ink), which is smaller than the frame unless
the ink plus its margin fills it to the pixel. A full-size uncaged square panel is
therefore a render from before the crop that nobody has re-rendered.

No JSON under `scripts/figures/data/` points into `docs/`, which is gitignored and absent from
the public remote: a provenance record must stand on the committed files and the paper.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import boxprep
import panel_checks
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
MANIFESTS = sorted((REPO / "scripts" / "figures" / "data").glob("**/manifest.json"))
PANELS = [
    pytest.param(panel, id=f"{manifest.parent.name}/{panel['id']}")
    for manifest in MANIFESTS
    for panel in panel_checks.load_panels(manifest)
]


def test_every_manifest_is_found():
    assert {manifest.parent.name for manifest in MANIFESTS} == {
        "antibodies",
        "excited-states",
        "graphitisation",
        "spheres",
    }


@pytest.mark.parametrize("panel", PANELS)
def test_each_panel_has_its_source_and_a_png_of_the_size_it_renders_to(panel):
    assert (REPO / panel["source"]).is_file(), f"no source {panel['source']}"
    output = REPO / panel["output_dir"] / f"{panel['id']}.png"
    assert output.is_file(), f"no render {output}"
    width_px, height_px = panel["resolution"]
    with Image.open(output) as image:
        size_px = image.size
    if panel.get("cage_angstrom") or width_px != height_px:
        assert size_px == (width_px, height_px)
        return
    assert size_px[0] == size_px[1], f"{output} is {size_px}, not a square crop"
    assert size_px[0] < min(width_px, height_px), f"{output} is {size_px}: never cropped"


PROTEIN_PANELS = [param for param in PANELS if param.values[0].get("kind") == "protein"]


@pytest.mark.parametrize("panel", PROTEIN_PANELS)
def test_a_protein_panel_names_exactly_the_chains_its_file_draws(panel):
    chains = set(boxprep.read_pdb(REPO / panel["source"]).chain_ids.tolist())
    assert chains == set(panel["cartoon_chains"]) | set(panel["stick_chains"])


def strings_in(value) -> list[str]:
    """Every string in a parsed JSON document, keys included."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for key, item in value.items() for text in [key, *strings_in(item)]]
    if isinstance(value, list):
        return [text for item in value for text in strings_in(item)]
    return []


def test_no_json_under_the_figure_data_points_into_docs():
    files = sorted((REPO / "scripts" / "figures" / "data").glob("**/*.json"))
    assert len(files) >= 11  # the provenance records, tables and manifests of every theme
    pointing = {
        str(file.relative_to(REPO)): text[:80]
        for file in files
        for text in strings_in(json.loads(file.read_text()))
        if re.search(r"(?<![\w-])docs/", text)
    }
    assert not pointing, f"a path into the gitignored docs/: {pointing}"
