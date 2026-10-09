"""The PDB entries of the antibodies figures against their provenance.json."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import boxprep
import pytest

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "scripts/figures/data/antibodies"
PROVENANCE = json.loads((DATA / "provenance.json").read_text())
REQUIRED = {
    "url",
    "downloaded",
    "md5",
    "size_bytes",
    "pdb_id",
    "title",
    "method",
    "resolution_angstrom",
    "release_year",
    "depositors",
    "citation",
    "doi",
    "chains",
    "licence",
}
# What the files hold once boxprep.read_pdb has taken model 1, one conformer and no water.
ATOMS_BY_CHAIN = {
    "1SY6.pdb": {"L": 1639, "H": 1675, "A": 1337},
    "6TRO.pdb": {"A": 2233, "B": 837, "C": 80, "D": 1465, "E": 1957},
}


@pytest.mark.parametrize("name", sorted(PROVENANCE["files"]))
def test_an_entry_names_its_source_and_its_md5_and_size_are_the_files(name):
    entry = PROVENANCE["files"][name]
    data = (DATA / name).read_bytes()
    assert REQUIRED <= set(entry), f"{name}: missing {REQUIRED - set(entry)}"
    assert entry["url"] == f"https://files.rcsb.org/download/{name}"
    assert entry["md5"] == hashlib.md5(data).hexdigest()
    assert entry["size_bytes"] == len(data) < 20_000_000
    assert "CC0 1.0" in entry["licence"]
    assert entry["pdb_id"] + ".pdb" == name


def test_every_pdb_file_has_an_entry_and_every_entry_a_file():
    on_disk = {path.name for path in DATA.glob("*.pdb")}
    assert on_disk == set(PROVENANCE["files"]) == set(ATOMS_BY_CHAIN)


@pytest.mark.parametrize("name", sorted(ATOMS_BY_CHAIN))
def test_the_chains_an_entry_describes_are_the_chains_the_file_draws_and_hold_these_atoms(name):
    protein = boxprep.read_pdb(DATA / name)
    chains, counts = protein.chain_ids.tolist(), {}
    for chain in chains:
        counts[chain] = counts.get(chain, 0) + 1
    assert set(PROVENANCE["files"][name]["chains"]) == set(counts)
    assert counts == ATOMS_BY_CHAIN[name]
    assert not protein.is_hetero.any(), "no ligand: the only HETATM records are water"


@pytest.mark.parametrize("name", sorted(ATOMS_BY_CHAIN))
def test_the_title_year_resolution_and_doi_are_the_files(name):
    entry = PROVENANCE["files"][name]
    text = (DATA / name).read_text()
    lines = text.splitlines()
    # TITLE continues on lines numbered in columns 9-10; its text starts at column 11. A line
    # wrapped after a hyphen rejoins without a space ("HLA A2*01-" + "GVYDGREHTV").
    title = " ".join(line[10:].strip() for line in lines if line.startswith("TITLE"))
    assert title.replace("- ", "-") == entry["title"]
    release = next(line for line in lines if line.startswith("REVDAT   1"))
    assert 2000 + int(release.split()[2][-2:]) == entry["release_year"]
    resolution = next(line for line in lines if line.startswith("REMARK   2 RESOLUTION"))
    assert float(resolution.split()[3]) == entry["resolution_angstrom"]
    assert f"DOI    {entry['doi'].upper()}" in text
