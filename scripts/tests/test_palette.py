"""The species and chain palettes are defined once in constants.py and mirrored in tokens.css."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from constants import CHAIN_COLOUR, COVALENT_RADIUS_ANGSTROM, SPECIES_COLOUR, srgb_hex_to_linear

TOKENS = Path(__file__).resolve().parents[2] / "src" / "styles" / "tokens.css"
SPECIES_TOKEN = re.compile(r"--species-([a-z]+):\s*(#[0-9a-fA-F]{6});")
CHAIN_TOKEN = re.compile(r"--chain-([0-9]+):\s*(#[0-9a-fA-F]{6});")


def test_the_species_tokens_are_the_palette_in_constants():
    tokens = {
        symbol: hex_colour.lower()
        for symbol, hex_colour in SPECIES_TOKEN.findall(TOKENS.read_text())
    }
    palette = {symbol.lower(): hex_colour.lower() for symbol, hex_colour in SPECIES_COLOUR.items()}
    assert tokens == palette


def test_the_chain_tokens_are_the_chain_palette_in_constants_in_order():
    tokens = [
        (int(number), hex_colour.lower())
        for number, hex_colour in CHAIN_TOKEN.findall(TOKENS.read_text())
    ]
    assert tokens == [(number, colour.lower()) for number, colour in enumerate(CHAIN_COLOUR, 1)]


def test_every_species_the_bond_rule_knows_has_a_colour():
    assert set(COVALENT_RADIUS_ANGSTROM) <= set(SPECIES_COLOUR)


def test_every_colour_is_six_hex_digits():
    assert all(
        re.fullmatch(r"#[0-9a-f]{6}", colour)
        for colour in (*SPECIES_COLOUR.values(), *CHAIN_COLOUR)
    )


def test_srgb_white_and_black_map_to_themselves():
    assert srgb_hex_to_linear("#ffffff") == (1.0, 1.0, 1.0)
    assert srgb_hex_to_linear("#000000") == (0.0, 0.0, 0.0)


def test_srgb_mid_grey_is_a_fifth_of_white_in_linear_light():
    assert srgb_hex_to_linear("#808080") == pytest.approx((0.216,) * 3, abs=1e-3)


def test_the_linear_value_of_the_toe_is_linear_in_the_srgb_value():
    # Below the 0.04045 knee the transfer is a straight line, channel / 12.92.
    assert srgb_hex_to_linear("#0a0000")[0] == pytest.approx(10 / 255 / 12.92)
