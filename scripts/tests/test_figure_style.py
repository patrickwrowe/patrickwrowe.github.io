"""Tests for the drawing helpers in scripts/figures/figure_style.py."""

from __future__ import annotations

import re

import pytest
from constants import SPECIES_COLOUR
from figure_style import (
    LABEL_SIZE,
    MINUS_SIGN,
    TICK_LENGTH,
    format_tick,
    species_colour,
    svg_axis,
    svg_document,
    svg_line,
    svg_series,
)


@pytest.mark.parametrize(
    ("value", "decimals", "expected"),
    [(-0.5, 1, "−0.5"), (0.02, 2, "0.02"), (-15.0, 0, "−15"), (0.0, 2, "0.00")],
)
def test_a_negative_tick_label_uses_the_true_minus_sign(value, decimals, expected):
    assert format_tick(value, decimals) == expected
    assert "-" not in format_tick(value, decimals)
    assert MINUS_SIGN == "−"


@pytest.mark.parametrize("symbol", sorted(SPECIES_COLOUR))
def test_every_species_in_the_palette_has_its_token(symbol):
    assert species_colour(symbol) == f"var(--species-{symbol.lower()})"


def test_a_species_with_no_colour_is_refused():
    with pytest.raises(ValueError, match="no species colour for 'Xx'"):
        species_colour("Xx")


def test_a_bottom_axis_has_a_tick_and_a_centred_label_below_each_position():
    parts = svg_axis("bottom", 10.0, 50.0, 40.0, [(0.0, "0"), (20.0, "1")])
    assert len(parts) == 1 + 2 * 2
    assert 'x1="10.00" y1="50.00" x2="50.00" y2="50.00"' in parts[0]
    assert f'x1="30.00" y1="50.00" x2="30.00" y2="{50 + TICK_LENGTH:.2f}"' in parts[3]
    assert 'x="30.00"' in parts[4] and 'text-anchor="middle"' in parts[4]
    assert f'font-size="{LABEL_SIZE}"' in parts[4]


def test_a_left_axis_runs_up_from_its_origin_with_labels_ending_at_its_ticks():
    parts = svg_axis("left", 10.0, 50.0, 40.0, [(0.0, "0"), (20.0, "1")])
    assert 'x1="10.00" y1="50.00" x2="10.00" y2="10.00"' in parts[0]
    assert 'y1="30.00"' in parts[3]
    assert 'text-anchor="end"' in parts[4] and 'y="31.00"' in parts[4]


def test_an_axis_on_a_side_that_does_not_exist_is_refused():
    with pytest.raises(ValueError, match="'top'"):
        svg_axis("top", 0.0, 0.0, 1.0, [])


def test_a_dashed_series_carries_its_dash_and_a_solid_one_does_not():
    assert 'stroke-dasharray="2.5 1.8"' in svg_series([(0, 0), (1, 1)], "var(--ink)", "2.5 1.8")
    assert "dasharray" not in svg_series([(0, 0), (1, 1)], "var(--ink)")
    assert "dasharray" not in svg_line(0, 0, 1, 1)


def test_the_document_root_carries_its_width_in_labels():
    svg = svg_document(94.5, 30.0, ["<g/>"])
    assert 'viewBox="0 0 94.50 30.00"' in svg
    assert float(re.search(r'data-chart-em="([\d.]+)"', svg).group(1)) == 94.5 / LABEL_SIZE
    assert svg.endswith("</svg>\n")
