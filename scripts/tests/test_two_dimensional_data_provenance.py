"""The data of the two-dimensional-and-confined figures against their provenance.json."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "scripts/figures/data/two-dimensional-and-confined"
PROVENANCE = json.loads((DATA / "provenance.json").read_text())
REQUIRED = {"what", "source_path", "md5"}


def md5_of(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def test_every_data_file_has_an_entry_and_every_entry_a_file():
    on_disk = {path.name for path in DATA.iterdir() if path.name != "provenance.json"}
    assert on_disk == set(PROVENANCE["files"])


@pytest.mark.parametrize("name", sorted(PROVENANCE["files"]))
def test_an_entry_says_what_the_file_is_and_where_it_came_from_and_its_md5_is_the_files(name):
    entry = PROVENANCE["files"][name]
    assert REQUIRED <= set(entry), f"{name}: missing {REQUIRED - set(entry)}"
    assert md5_of(DATA / name) == entry["md5"]


@pytest.mark.parametrize("name", sorted(PROVENANCE["files"]))
def test_a_source_on_this_machine_is_the_one_the_entry_checksums(name):
    entry = PROVENANCE["files"][name]
    source = REPO / entry["source_path"]
    if "source_md5" not in entry or not source.exists():
        pytest.skip("no checksummed source on this machine")
    assert md5_of(source) == entry["source_md5"]
    if "source_sheet" not in entry:
        assert entry["md5"] == entry["source_md5"], f"{name} is meant to be a verbatim copy"
