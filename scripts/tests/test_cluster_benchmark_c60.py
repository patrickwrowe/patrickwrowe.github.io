"""Tests for scripts/figures/cluster_benchmark_c60.py."""

from __future__ import annotations

import re
from pathlib import Path

import cluster_benchmark_c60
import pytest

HEX_COLOUR = re.compile(r"#[0-9a-fA-F]{3}\b|#[0-9a-fA-F]{6}\b")
FIGURES = Path(__file__).resolve().parents[2] / "src/content/work/figures/carbon"


@pytest.mark.parametrize(
    ("narrow", "committed"),
    [(False, "cluster-benchmark-c60.svg"), (True, "cluster-benchmark-c60-narrow.svg")],
)
def test_the_committed_svg_is_byte_for_byte_what_the_script_writes(narrow, committed):
    written = cluster_benchmark_c60.build(narrow=narrow).encode("utf-8")
    assert (FIGURES / committed).read_bytes() == written, f"{committed} is stale"


def test_svg_has_eight_data_markers_and_no_hex_colour():
    svg = cluster_benchmark_c60.build()
    assert len(re.findall(r"<circle ", svg)) == 8
    assert "var(--ink)" in svg
    assert not HEX_COLOUR.search(svg)


@pytest.mark.parametrize("narrow", [False, True])
def test_dft_reference_line_sits_on_the_dft_marker(narrow):
    # Read both from the drawing, not from x_for_value: the line is meant to pass through
    # the marker the reader sees, whatever function placed either.
    svg = cluster_benchmark_c60.build(narrow=narrow)
    dft_row = re.search(r'<g data-method="DFT" data-kind="reference">.*?</g>', svg, re.DOTALL)
    marker_x = float(re.search(r'<circle cx="([-\d.]+)"', dft_row.group(0)).group(1))
    line = re.search(r'<line ([^>]*data-reference="DFT"[^>]*)/>', svg).group(1)
    assert abs(float(re.search(r'x1="([-\d.]+)"', line).group(1)) - marker_x) < 1e-6
    assert abs(float(re.search(r'x2="([-\d.]+)"', line).group(1)) - marker_x) < 1e-6


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


def test_the_narrow_layout_fits_the_phone_budget_with_every_label_inside_it():
    # The mono face advances 0.6 em, so a label of n characters at size s is 0.6 n s wide.
    svg = cluster_benchmark_c60.build(narrow=True)
    width, _height = map(float, re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg).groups())
    assert width == cluster_benchmark_c60.NARROW_CHART_EM * cluster_benchmark_c60.LABEL_SIZE
    assert len(re.findall(r"<circle ", svg)) == 8
    texts = re.findall(
        r'<text x="([\d.]+)" y="[\d.]+" text-anchor="(\w+)" font-size="([\d.]+)"[^>]*>([^<]*)<',
        svg,
    )
    assert len(texts) == 8 + len(cluster_benchmark_c60.X_TICKS_EV_PER_ATOM) + 1
    for x, anchor, size, content in texts:
        assert float(size) >= cluster_benchmark_c60.LABEL_SIZE, content
        extent = 0.6 * len(content) * float(size)
        left = {"start": 0.0, "middle": 0.5, "end": 1.0}[anchor] * extent
        assert float(x) - left >= 0 and float(x) - left + extent <= width, content


def test_the_narrow_layout_keeps_every_mark_at_its_value():
    svg = cluster_benchmark_c60.build(narrow=True)
    for method_label, value, _kind in cluster_benchmark_c60.METHODS:
        row = re.search(rf'<g data-method="{method_label}".*?</g>', svg, re.DOTALL).group(0)
        centre_x = float(re.search(r'<circle cx="([\d.]+)"', row).group(1))
        expected = cluster_benchmark_c60.x_for_value(
            value, cluster_benchmark_c60.NARROW_PLOT_LEFT, cluster_benchmark_c60.NARROW_PLOT_WIDTH
        )
        assert abs(centre_x - expected) < 0.01, method_label
