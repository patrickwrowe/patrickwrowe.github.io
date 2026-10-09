"""Tests for the drawing helpers in scripts/figures/figure_style.py."""

from __future__ import annotations

import math
import re

import pytest
from constants import SPECIES_COLOUR
from figure_style import (
    AXIS_TITLE_SIZE,
    LABEL_SIZE,
    MARKER_RADIUS,
    MARKER_STROKE,
    MINUS_SIGN,
    TICK_LENGTH,
    X_AXIS_TITLE_DROP,
    format_tick,
    species_colour,
    svg_axis,
    svg_document,
    svg_line,
    svg_marker,
    svg_series,
    svg_square_marker,
    svg_text,
    write_figure,
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


def test_greek_in_a_label_is_set_in_the_serif_and_the_rest_stays_mono():
    label = svg_text(0.0, 0.0, "P(cos θ) and Δφ", LABEL_SIZE)
    assert 'font-family="var(--mono)"' in label
    assert re.findall(r'<tspan font-family="var\(--serif\)">([^<]+)</tspan>', label) == ["θ", "Δφ"]
    assert re.sub(r"<[^>]+>", "", label) == "P(cos θ) and Δφ"
    assert "tspan" not in svg_text(0.0, 0.0, "height above surface (Å)", LABEL_SIZE)


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


def test_a_filled_marker_is_ink_and_an_open_one_has_a_plate_centre_inside_the_same_radius():
    assert svg_marker(10.0, 20.0, True) == (
        f'<circle cx="10.00" cy="20.00" r="{MARKER_RADIUS}" fill="var(--ink)" stroke="none"/>'
    )
    open_radius = MARKER_RADIUS - MARKER_STROKE / 2
    assert svg_marker(10.0, 20.0, False) == (
        f'<circle cx="10.00" cy="20.00" r="{open_radius:.2f}" fill="var(--plate)" '
        f'stroke="var(--ink)" stroke-width="{MARKER_STROKE}"/>'
    )


def test_the_x_axis_title_clears_the_tick_labels_below_the_axis():
    labels_baseline = TICK_LENGTH + LABEL_SIZE + 0.4  # where svg_axis puts the tick labels
    assert X_AXIS_TITLE_DROP - AXIS_TITLE_SIZE > labels_baseline


def test_write_figure_builds_with_the_data_dir_and_the_narrow_flag_and_writes_the_output(
    tmp_path, capsys
):
    calls = []

    def build(data_dir, narrow):
        calls.append((data_dir, narrow))
        return "<svg/>\n"

    output = tmp_path / "new" / "figure.svg"
    argv = ["--data-dir", str(tmp_path), "--output", str(output), "--narrow"]
    write_figure(build, "one line", argv)
    assert calls == [(tmp_path, True)]
    assert output.read_text() == "<svg/>\n"
    assert capsys.readouterr().out == f"-> {output}\n"


def test_write_figure_for_a_figure_without_data_calls_build_with_the_flag_alone(tmp_path):
    output = tmp_path / "figure.svg"
    write_figure(lambda narrow: f"narrow={narrow}", "one line", ["--output", str(output)], False)
    assert output.read_text() == "narrow=False"
    with pytest.raises(SystemExit):  # --data-dir is not an option of such a figure
        write_figure(lambda narrow: "", "one line", ["--data-dir", "x", "--output", "y"], False)


def test_write_figure_without_an_output_exits_before_building(tmp_path):
    with pytest.raises(SystemExit):
        write_figure(lambda *_: pytest.fail("built"), "one line", ["--data-dir", str(tmp_path)])


def test_a_square_marker_is_centred_filled_ink_and_has_the_area_of_the_circle():
    marker = svg_square_marker(10.0, 20.0)
    x, y, width, height = (float(v) for v in re.findall(r'(?:x|y|width|height)="([\d.]+)"', marker))
    assert (x + width / 2, y + height / 2) == pytest.approx((10.0, 20.0), abs=0.01)
    assert width == height
    assert width * height == pytest.approx(math.pi * MARKER_RADIUS**2, rel=0.01)
    assert 'fill="var(--ink)" stroke="none"' in marker
