"""Tests for scripts/figures/cluster_benchmark_c60.py."""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))
import cluster_benchmark_c60  # noqa: E402

HEX_COLOUR = re.compile(r"#[0-9a-fA-F]{3}\b|#[0-9a-fA-F]{6}\b")


def test_svg_has_eight_data_markers_and_no_hex_colour():
    svg = cluster_benchmark_c60.build()
    assert len(re.findall(r"<circle ", svg)) == 8
    assert "var(--ink)" in svg
    assert not HEX_COLOUR.search(svg)


def test_dft_reference_line_sits_at_the_mapped_dft_value():
    svg = cluster_benchmark_c60.build()
    match = re.search(r'<line ([^>]*data-reference="DFT"[^>]*)/>', svg)
    assert match is not None
    x1 = float(re.search(r'x1="([-\d.]+)"', match.group(1)).group(1))
    x2 = float(re.search(r'x2="([-\d.]+)"', match.group(1)).group(1))
    expected_x = cluster_benchmark_c60.x_for_value(cluster_benchmark_c60.DFT_REFERENCE_EV_PER_ATOM)
    assert abs(x1 - expected_x) < 1e-6
    assert abs(x2 - expected_x) < 1e-6


def test_gap20_is_filled_and_classical_potentials_are_open():
    svg = cluster_benchmark_c60.build()
    gap_row = re.search(r'<g data-method="GAP-20" data-kind="gap">.*?</g>', svg, re.DOTALL).group(0)
    assert 'fill="var(--ink)"' in gap_row

    reaxff_row = re.search(
        r'<g data-method="ReaxFF" data-kind="classical">.*?</g>', svg, re.DOTALL
    ).group(0)
    assert 'fill="var(--plate)"' in reaxff_row
    assert 'stroke="var(--ink)"' in reaxff_row


def test_negative_tick_labels_use_the_true_minus_sign_not_hyphen():
    # A hyphen-minus immediately before a digit inside a <text> element would be the
    # bug this guards against (design review C7): negatives should read "−7.6",
    # not "-7.6". The lookbehind excludes a hyphen inside a name like "GAP-20", which
    # is not a negative number.
    svg = cluster_benchmark_c60.build()
    for text_element in re.findall(r"<text [^>]*>([^<]*)</text>", svg):
        assert not re.search(r"(?<![A-Za-z0-9])-\d", text_element)


def test_c60_axis_label_is_plain_not_subscripted():
    svg = cluster_benchmark_c60.build()
    assert "C60" in svg
    assert "C₆₀" not in svg
