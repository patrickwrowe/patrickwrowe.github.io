"""The inputs of the excited-states figures against their provenance.json.

The committed files are the root of trust; the archive and the JCTC manuscript are checked
where they are on this machine and skipped where they are not. Nothing here reads docs/,
which is gitignored.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import unicodedata
from collections import Counter
from pathlib import Path

import boxprep
import numpy as np
import panel_checks
import pytest

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "scripts/figures/data/excited-states"
PROVENANCE = json.loads((DATA / "provenance.json").read_text())
ARCHIVE = Path(PROVENANCE["archive_root"])
# The papers live outside the repository, wherever PAPERS_DIR says; unset, their tests skip.
PAPERS_DIR = os.environ.get("PAPERS_DIR")
MANUSCRIPT = Path(PAPERS_DIR, "excitonic-interactions/paper.pdf") if PAPERS_DIR else None
TABLE = json.loads((DATA / "table1-short-range-share.json").read_text())
REQUIRED = {"source_path", "what", "edit", "md5", "size_bytes"}
VERBATIM = sorted(
    name for name, entry in PROVENANCE["files"].items() if entry["edit"].startswith("none")
)
COMPOSITION = {
    "chromophore-1.xyz": {"C": 33, "H": 34, "Mg": 1, "N": 4, "O": 4},
    "chromophore-2.xyz": {"C": 33, "H": 34, "Mg": 1, "N": 4, "O": 4},
    "fmo-natural.xyz": {"C": 264, "H": 272, "Mg": 8, "N": 32, "O": 32},
    "fmo-generated-00300.xyz": {"C": 264, "H": 272, "Mg": 8, "N": 32, "O": 32},
}


def md5_of(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def test_every_data_file_has_an_entry_and_every_entry_a_file():
    # The manifest is the hero render's input, not a staged file, and has no entry.
    on_disk = {path.name for path in DATA.iterdir()} - {"provenance.json", "manifest.json"}
    assert on_disk == set(PROVENANCE["files"])


@pytest.mark.parametrize("name", sorted(PROVENANCE["files"]))
def test_an_entry_says_what_the_file_is_and_its_md5_and_size_are_the_files(name):
    entry = PROVENANCE["files"][name]
    assert REQUIRED <= set(entry), f"{name}: missing {REQUIRED - set(entry)}"
    assert entry["md5"] == md5_of(DATA / name)
    assert entry["size_bytes"] == (DATA / name).stat().st_size < 20_000_000


@pytest.mark.parametrize(
    "name", sorted(set(PROVENANCE["files"]) - {"table1-short-range-share.json"})
)
def test_a_source_in_the_archive_is_the_file_the_entry_checksums(name):
    entry = PROVENANCE["files"][name]
    source = ARCHIVE / entry["source_path"]
    if not source.exists():
        pytest.skip("the archive is not on this machine")
    assert md5_of(source) == entry["source_md5"]
    if name in VERBATIM:
        assert entry["md5"] == entry["source_md5"], f"{name} is meant to be a verbatim copy"


def test_the_pdb_is_the_deposited_file_as_it_was_downloaded():
    # Verbatim since 2026-10-15: the lone Mg ion is left out by prepare's rule, not by hand.
    entry = PROVENANCE["files"]["3BSD.pdb"]
    assert entry["edit"].startswith("none") and entry["md5"] == entry["source_md5"]
    assert entry["size_bytes"] == entry["source_size_bytes"]


def test_the_pdb_holds_one_chain_eight_bchl_178_waters_and_one_lone_magnesium_ion():
    lines = (DATA / "3BSD.pdb").read_text().splitlines()
    atoms = [line for line in lines if line.startswith(("ATOM  ", "HETATM"))]
    bchl = [line for line in atoms if line[17:20] == "BCL"]
    assert Counter(line[21] for line in atoms if line.startswith("ATOM  ")) == {"A": 2798}
    assert len(bchl) == 504
    assert Counter(line[22:26].strip() for line in bchl) == {
        **dict.fromkeys(("367", "368", "369", "370", "371", "372", "373"), 66),
        "400": 42,
    }
    assert sum(line[17:20] == "HOH" for line in atoms) == 178
    hetero_residues = {line[17:20].strip() for line in atoms if line.startswith("HETATM")}
    assert hetero_residues == {"BCL", "HOH", "MG"}
    lone_magnesium = [line for line in atoms if line[17:20].strip() == "MG"]
    assert [line[:26] for line in lone_magnesium] == ["HETATM 3262 MG    MG A 374"]
    assert lone_magnesium[0][54:60] == "  0.33"  # occupancy


def test_read_pdb_reads_the_protein_the_eight_bchl_and_the_lone_ion_and_no_water():
    # The lone ion is read here and left out by render_box_grid.prepare (test_render_box_grid).
    protein = boxprep.read_pdb(DATA / "3BSD.pdb")
    assert (protein.n_dropped_altloc, protein.n_dropped_water) == (0, 178)
    assert set(protein.chain_ids.tolist()) == {"A"}
    assert protein.is_hetero.sum() == 505
    assert Counter(np.array(protein.species)[protein.is_hetero].tolist()) == {
        "C": 418,
        "O": 46,
        "N": 32,
        "Mg": 9,
    }


@pytest.mark.parametrize("name", sorted(COMPOSITION))
def test_read_xyz_reads_the_tab_separated_files_as_they_are(name):
    structure = boxprep.read_xyz(DATA / name)
    assert Counter(structure.species) == COMPOSITION[name]
    assert structure.positions_angstrom.shape == (sum(COMPOSITION[name].values()), 3)
    assert np.isfinite(structure.positions_angstrom).all()


def test_the_two_chromophores_come_within_the_2_26_angstrom_of_table_1_pair_1_2():
    first = boxprep.read_xyz(DATA / "chromophore-1.xyz").positions_angstrom
    second = boxprep.read_xyz(DATA / "chromophore-2.xyz").positions_angstrom
    separations_angstrom = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=-1)
    row = next(row for row in TABLE["rows"] if row["system"] == "FMO" and row["pair"] == "1-2")
    assert separations_angstrom.min() == pytest.approx(row["min_distance_angstrom"], abs=0.005)
    assert separations_angstrom.min() == pytest.approx(2.2613, abs=5e-5)


def test_the_natural_and_generated_arrangements_share_the_source_and_drain_chromophores():
    natural = boxprep.read_xyz(DATA / "fmo-natural.xyz")
    generated = boxprep.read_xyz(DATA / "fmo-generated-00300.xyz")
    assert natural.species == generated.species
    moved = np.linalg.norm(natural.positions_angstrom - generated.positions_angstrom, axis=1) > 1e-6
    magnesium = np.array(natural.species) == "Mg"
    assert magnesium.sum() == 8
    assert (moved & magnesium).sum() == 6  # the dossier: BChl 1 and 3 are fixed, six move


def test_table_1_has_the_fifteen_own_rows_of_three_systems_all_inside_the_close_contact_range():
    rows = TABLE["rows"]
    assert Counter(row["system"] for row in rows) == {"FMO": 5, "LH2": 5, "PCP": 5}
    assert TABLE["close_contact_threshold_angstrom"] == 7.0
    assert all(0 < row["min_distance_angstrom"] < 7.0 for row in rows)
    share_of_total = [100 * row["j_short_mev"] / row["j_total_mev"] for row in rows]
    printed = [row["j_short_share_percent"] for row in rows]
    # The printed share is from unrounded couplings: within one percentage point of the ratio.
    assert share_of_total == pytest.approx(printed, abs=1.0)
    assert max(row["j_short_share_percent"] for row in rows if row["system"] == "FMO") == 28
    assert max(row["j_short_share_percent"] for row in rows if row["system"] == "LH2") == 66
    assert max(row["j_short_share_percent"] for row in rows if row["system"] == "PCP") == 22


@pytest.mark.skipif(
    not (MANUSCRIPT and MANUSCRIPT.exists() and shutil.which("pdftotext")),
    reason="no manuscript or no pdftotext here",
)
def test_every_row_is_on_page_11_of_the_manuscript_and_the_threshold_on_page_9():
    def page(number: int) -> str:
        text = subprocess.run(
            [
                "pdftotext",
                "-layout",
                "-f",
                str(number + 1),
                "-l",
                str(number + 1),
                str(MANUSCRIPT),
                "-",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return " ".join(unicodedata.normalize("NFKC", text).split())

    # "<D> <states> <J_TOT> <J_short> (<share>%)": the Scholes rows have no D and do not match.
    row_pattern = r"(\d\.\d\d) (\d,\d) ([\d.]+) ([\d.]+) ?\(([\d.]+)%\)"
    on_page = [
        (float(distance), states, float(total), float(short), float(share))
        for distance, states, total, short, share in re.findall(row_pattern, page(11))
    ]
    assert on_page == [
        (
            row["min_distance_angstrom"],
            row["coupled_states"],
            row["j_total_mev"],
            row["j_short_mev"],
            row["j_short_share_percent"],
        )
        for row in TABLE["rows"]
    ]
    assert "minimum intermolecular distance D = 7 Å" in page(9)


PANELS = {panel["id"]: panel for panel in panel_checks.load_panels(DATA / "manifest.json")}
RENDERS = PROVENANCE["renders"]
RENDER_IDS = sorted(key for key in RENDERS if key not in {"note", "method"})


def test_every_panel_of_the_manifest_has_a_render_entry_and_every_entry_a_panel():
    assert RENDER_IDS == sorted(PANELS) == ["hero"]


@pytest.mark.parametrize("panel_id", RENDER_IDS)
def test_a_render_entry_records_the_manifests_direction_and_radii_and_chose_one_candidate(panel_id):
    entry, panel = RENDERS[panel_id], PANELS[panel_id]
    assert entry["source"] == Path(panel["source"]).name
    assert entry["direction"] == panel["direction"]
    chosen = [
        candidate for candidate in entry["candidates"] if candidate["verdict"].startswith("Chosen.")
    ]
    assert [candidate["direction"] for candidate in chosen] == [panel["direction"]]
    assert len(entry["candidates"]) >= 3  # directions tried, each with its verdict
    assert entry["why"] and entry["limits"]
    assert entry["radii"]["sphere_radius_angstrom"] == panel["sphere_radius_angstrom"]
    assert entry["radii"]["bond_radius_angstrom"] == panel["bond_radius_angstrom"]
    assert entry["wall_time_s"] > 0 and entry["summary_line"].startswith(f"{panel_id}: ")


@pytest.mark.parametrize("panel_id", RENDER_IDS)
def test_a_render_entry_records_the_manifest_resolution_the_dropped_atoms_and_the_summary_line(
    panel_id,
):
    entry, panel = RENDERS[panel_id], PANELS[panel_id]
    protein = boxprep.read_pdb(Path(panel["source"]))
    assert entry["manifest"] == "scripts/figures/data/excited-states/manifest.json"
    assert entry["resolution"] == panel["resolution"] == [1728, 972]
    assert entry["dropped"] == {
        "altloc": protein.n_dropped_altloc,
        "water": protein.n_dropped_water,
    }
    line = entry["summary_line"]
    # prepare draws every atom read_pdb keeps but the lone Mg ion, MG A 374, which it leaves
    # out by rule (test_render_box_grid pins that on this file).
    drawn = len(protein.species) - sum(record[17:20] == " MG" for record in protein.records)
    assert f": {drawn} atoms (dropped {entry['dropped']['altloc']} altloc, " in line
    assert (
        f"{entry['dropped']['water']} water), wall {entry['wall_time_s']} s, blender exit 0" in line
    )
    assert str(Path.home()) not in line  # the repository-relative path, not this machine's


def test_the_hero_ligand_bonds_recorded_are_the_ones_the_radius_rule_finds_among_the_bcl_atoms():
    # What `render_box_grid.prepare` writes as CONECT: the hetero atoms with a covalent radius.
    protein = boxprep.read_pdb(DATA / "3BSD.pdb")
    hetero = np.flatnonzero(protein.is_hetero)
    bonds = boxprep.find_bonds(
        protein.positions_angstrom[hetero], [protein.species[atom] for atom in hetero]
    )
    entry = RENDERS["hero"]["ligand_bonds"]
    assert entry["needed_conect"] is True
    assert entry["bonds"] == len(bonds) == 568
