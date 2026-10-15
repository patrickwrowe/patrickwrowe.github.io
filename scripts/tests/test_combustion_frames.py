"""combustion_frames.py's unwrapping, round-tripped through the committed CHO-GAP frames."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import boxprep
import combustion_frames
import numpy as np
import pytest

CHO_DATA = Path(__file__).resolve().parents[1] / "figures" / "data" / "cho-gap"
BOX_ANGSTROM = 10.0  # the run's periodic cube, combustion_frames.py's module docstring


@pytest.mark.parametrize("time_ps", [0, 10, 20, 50])
def test_unwrapping_a_rewrapped_committed_frame_gives_the_committed_frame_back(time_ps):
    # Folding every atom back into the cell is what LAMMPS dumps; unwrap must rebuild the
    # molecules the committed file holds, on a real frame where they straddle the faces.
    frame = boxprep.read_xyz(CHO_DATA / f"combustion-{time_ps}ps.xyz")
    dumped_angstrom = frame.positions_angstrom % BOX_ANGSTROM
    bonds = boxprep.find_bonds(dumped_angstrom, frame.species, BOX_ANGSTROM)
    groups = combustion_frames.molecules(len(frame), bonds)
    rebuilt_angstrom = combustion_frames.unwrap(dumped_angstrom, bonds, groups, BOX_ANGSTROM)
    np.testing.assert_allclose(rebuilt_angstrom, frame.positions_angstrom, atol=1e-5)


def test_every_staged_frame_has_a_provenance_entry_whose_md5_and_size_are_the_files():
    files = json.loads((CHO_DATA / "provenance.json").read_text())["files"]
    assert set(files) == {path.name for path in CHO_DATA.glob("*.xyz")}
    for name, entry in files.items():
        raw = (CHO_DATA / name).read_bytes()
        assert (entry["md5"], entry["size_bytes"]) == (hashlib.md5(raw).hexdigest(), len(raw))
