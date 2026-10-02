"""Tests for scripts/figures/extract_spheres.py: a failed pass never costs a good record.

Every test runs on a scratch copy of the committed provenance.json, never on the real data
directory, and builds whatever archive it needs from the C40 fixture dump.
"""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

import extract_spheres
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
COMMITTED_PROVENANCE = extract_spheres.OUT_DIR / extract_spheres.PROVENANCE_NAME
N_RUNS = len(extract_spheres.SIZES) * len(extract_spheres.TEMPERATURES_KELVIN)
# The parts of a LAMMPS log that extract_spheres.log_facts reads, from a FIRE run.
MINIMAL_LOG = """min_style fire
minimize 1.0e-3 1.0e-3 10000 100000
Loop time of 1.5 on 4 procs for 49 steps with 40 atoms
timestep 0.002
"""


@pytest.fixture
def scratch(tmp_path: Path) -> Path:
    """A scratch output directory holding a copy of the committed provenance.json."""
    out_dir = tmp_path / "spheres"
    out_dir.mkdir()
    shutil.copy(COMMITTED_PROVENANCE, out_dir / extract_spheres.PROVENANCE_NAME)
    return out_dir


def test_a_pass_with_no_archive_leaves_provenance_byte_identical(scratch, tmp_path):
    before = (scratch / extract_spheres.PROVENANCE_NAME).read_bytes()

    status = extract_spheres.main(
        ["--archive", str(tmp_path / "no-such-archive"), "--out-dir", str(scratch)]
    )

    assert status == 1
    assert (scratch / extract_spheres.PROVENANCE_NAME).read_bytes() == before
    failures = json.loads((scratch / extract_spheres.FAILURES_NAME).read_text())
    assert len(failures) == N_RUNS
    assert all("FileNotFoundError" in error for error in failures.values())
    assert not list(scratch.glob("*.xyz")), "a failed run wrote a frame"
    assert not list(scratch.glob("*.tmp")), "a temporary file was left behind"


def test_a_partial_pass_merges_its_record_and_dead_letters_the_rest(scratch, tmp_path):
    archive = tmp_path / "2_LAMMPS_MD"
    run_dir = archive / extract_spheres.SERIES / "500" / "C40"
    run_dir.mkdir(parents=True)
    shutil.copy(FIXTURES / "c40-500K-two-frames.lammpstrj", run_dir / "liquid_carbon.lammpstrj")
    (run_dir / "log.lammps").write_text(MINIMAL_LOG)
    before = json.loads((scratch / extract_spheres.PROVENANCE_NAME).read_text())

    status = extract_spheres.main(["--archive", str(archive), "--out-dir", str(scratch)])

    assert status == 1
    after = json.loads((scratch / extract_spheres.PROVENANCE_NAME).read_text())
    assert list(after) == list(before), "the merge reordered or dropped records"
    record = after["C40-500K.xyz"]
    assert record["archive_path"] == str(run_dir / "liquid_carbon.lammpstrj")
    assert record["last_timestep"] == 75
    assert record["frame_count"] == 2
    assert record["cell_edge_angstrom"] == pytest.approx(48.489, abs=1e-3)
    assert {name: after[name] for name in after if name != "C40-500K.xyz"} == {
        name: before[name] for name in before if name != "C40-500K.xyz"
    }
    failures = json.loads((scratch / extract_spheres.FAILURES_NAME).read_text())
    assert len(failures) == N_RUNS - 1 and "C40-500K.xyz" not in failures
    assert [path.name for path in scratch.glob("*.xyz")] == ["C40-500K.xyz"]


def test_a_clean_pass_removes_the_old_dead_letter_file(scratch, tmp_path, monkeypatch):
    (scratch / extract_spheres.FAILURES_NAME).write_text('{"C40-500K.xyz": "old"}\n')
    monkeypatch.setattr(extract_spheres, "SIZES", (40,))
    monkeypatch.setattr(extract_spheres, "TEMPERATURES_KELVIN", (500,))
    archive = tmp_path / "2_LAMMPS_MD"
    run_dir = archive / extract_spheres.SERIES / "500" / "C40"
    run_dir.mkdir(parents=True)
    shutil.copy(FIXTURES / "c40-500K-two-frames.lammpstrj", run_dir / "liquid_carbon.lammpstrj")
    (run_dir / "log.lammps").write_text(MINIMAL_LOG)

    assert extract_spheres.main(["--archive", str(archive), "--out-dir", str(scratch)]) == 0
    assert not (scratch / extract_spheres.FAILURES_NAME).exists()


def test_the_archive_is_a_required_argument(capsys):
    with pytest.raises(SystemExit) as exit_info:
        extract_spheres.main([])
    assert exit_info.value.code == 2
    assert "--archive" in capsys.readouterr().err


def test_no_figure_or_analysis_module_imports_the_archive_extractor():
    # The run grid and every other shared constant live in constants.py; the /data
    # adapter is a leaf that only its tests import.
    importers = []
    for module in sorted(Path(extract_spheres.__file__).parent.glob("*.py")):
        for node in ast.walk(ast.parse(module.read_text())):
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported = [node.module]
            else:
                continue
            if "extract_spheres" in imported:
                importers.append(module.name)
    assert importers == []
