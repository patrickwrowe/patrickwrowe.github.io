# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Last frame of each of the 48 GAP sphere runs, as whole-cluster XYZ plus provenance.

Reads `liquid_carbon.lammpstrj` for 8 sizes x 6 temperatures from the read-only archive
(`Carbon_Cluster_RSS/2_LAMMPS_MD/3_GAP_Spheres_Opt2/<T>/C<n>/`), never `final.restart`,
so all 48 frames share one source. Each frame is re-imaged about atom 1
(`boxprep.make_compact`), unwrapped fragment by fragment along its periodic bonds
(`boxprep.find_bonds_periodic`, `boxprep.make_whole`) so no bond crosses the cell
boundary, centred on its mean, and written as plain XYZ. A dissociated run's fragments
keep the placement make_compact gave them, which is arbitrary.

`provenance.json` records, per frame: the archive path, size, md5, last timestep, frame
count, atom count and cell edge; from the run's `log.lammps`, the minimiser, minimise
line, minimisation length, timestep, NVT start step, NVT time at the last frame and
whether the run completed; whether `final.restart` exists; and, where the run's log is
byte-identical to the earlier default-minimiser test set (`1_RSS_Spheres_Test`), a
`copy_of` entry with that set's trajectory and log paths and md5s.

The archive is on a slow HDD: each trajectory is read once from disk (md5, frame count
and timestep come from those bytes; boxprep re-reads the file from the page cache). About
350 MB in all. A run that fails is logged and recorded with its error; the rest go on.

Usage:
    uv run scripts/figures/extract_spheres.py [--archive DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import boxprep  # noqa: E402

ARCHIVE_ROOT = Path(
    "/data/pr_archive/source_drives/sdb2_Patrick_4Tb_BU/Happy_Electron_Backup/Research/"
    "Carbon_Cluster_RSS/2_LAMMPS_MD"
)
SERIES = "3_GAP_Spheres_Opt2"
TEST_SERIES = "1_RSS_Spheres_Test"
OUT_DIR = Path(__file__).resolve().parent / "data" / "carbon-clusters" / "spheres"
SIZES = (40, 60, 80, 120, 160, 373, 686, 1000)
TEMPERATURES_KELVIN = (500, 1000, 2000, 3000, 4000, 5000)
NVT_STEPS = 25_000
DUMP_INTERVAL_STEPS = 25


def md5_of(path: Path) -> str:
    """Hex md5 of a file's bytes.

    Args:
        path: File to hash.

    Returns:
        The 32-character hex digest.
    """
    return hashlib.md5(path.read_bytes()).hexdigest()


def log_facts(log_path: Path, last_timestep: int) -> dict[str, object]:
    """Protocol facts for one run, parsed from its LAMMPS log.

    The first `Loop time` line is the minimisation; the NVT run starts one step after it
    (the input runs one zero-velocity step before assigning velocities).

    Args:
        log_path: The run's `log.lammps`.
        last_timestep: Timestep of the trajectory's last frame.

    Returns:
        Minimiser name, minimise line, minimisation steps, timestep in ps, NVT start step,
        NVT time at the last frame in ps, and whether the 25,000-step run completed.

    Raises:
        ValueError: If the log lacks a minimise line, a timestep or a Loop time line.
    """
    log = log_path.read_text()
    minimise = re.search(r"^minimize .*$", log, re.M)
    timestep = re.search(r"^timestep\s+(\S+)", log, re.M)
    loop = re.search(r"Loop time of \S+ on \d+ procs for (\d+) steps", log)
    if not (minimise and timestep and loop):
        raise ValueError(f"{log_path}: missing minimize, timestep or Loop time line")
    minimisation_steps = int(loop.group(1))
    timestep_ps = float(timestep.group(1))
    nvt_start_timestep = minimisation_steps + 1
    return {
        "minimiser": "fire" if re.search(r"^min_style fire", log, re.M) else "cg (LAMMPS default)",
        "minimise_line": minimise.group(0).strip(),
        "minimisation_steps": minimisation_steps,
        "timestep_ps": timestep_ps,
        "nvt_start_timestep": nvt_start_timestep,
        "nvt_time_at_last_frame_ps": round((last_timestep - nvt_start_timestep) * timestep_ps, 3),
        "run_completed_25000_steps": last_timestep
        > nvt_start_timestep + NVT_STEPS - DUMP_INTERVAL_STEPS,
    }


def copy_of(archive_root: Path, run_dir: Path) -> dict[str, str] | None:
    """The test-set run this run was copied from, if its log is byte-identical.

    Args:
        archive_root: The `2_LAMMPS_MD` directory.
        run_dir: This run's directory, `<SERIES>/<T>/C<n>`.

    Returns:
        Paths and md5s of the test-set trajectory and log, or None if there is no
        test-set log at the same temperature and size, or it differs.
    """
    test_dir = archive_root / TEST_SERIES / run_dir.relative_to(archive_root / SERIES)
    test_log = test_dir / "log.lammps"
    if not test_log.exists() or md5_of(test_log) != md5_of(run_dir / "log.lammps"):
        return None
    test_trajectory = test_dir / "liquid_carbon.lammpstrj"
    return {
        "trajectory_path": str(test_trajectory),
        "trajectory_md5": md5_of(test_trajectory),
        "log_path": str(test_log),
        "log_md5": md5_of(test_log),
    }


def extract_one(archive_root: Path, n_atoms: int, temperature_kelvin: int) -> dict[str, object]:
    """Write one run's whole last frame and return its provenance record.

    Args:
        archive_root: The `2_LAMMPS_MD` directory.
        n_atoms: Cluster size.
        temperature_kelvin: Thermostat temperature.

    Returns:
        The provenance record for `C<n>-<T>K.xyz`.

    Raises:
        ValueError: If the frame's atom count differs from `n_atoms`, or the dump or log
            is malformed.
    """
    run_dir = archive_root / SERIES / str(temperature_kelvin) / f"C{n_atoms}"
    source = run_dir / "liquid_carbon.lammpstrj"
    raw = source.read_bytes()
    last_block = raw[raw.rfind(b"ITEM: TIMESTEP") :].split(b"\n")
    last_timestep = int(last_block[1])
    positions_angstrom, edge_angstrom = boxprep.read_lammps_last_frame(source)
    if len(positions_angstrom) != n_atoms:
        raise ValueError(f"{source}: {len(positions_angstrom)} atoms, expected {n_atoms}")
    compact_angstrom = boxprep.make_compact(positions_angstrom, edge_angstrom)
    bonds = boxprep.find_bonds_periodic(compact_angstrom, edge_angstrom)
    whole_angstrom = boxprep.make_whole(compact_angstrom, edge_angstrom, bonds)
    whole_angstrom -= whole_angstrom.mean(axis=0)
    lines = [
        str(n_atoms),
        f"C{n_atoms} {temperature_kelvin} K last frame of liquid_carbon.lammpstrj",
    ]
    lines += [
        f"C {x_angstrom:.5f} {y_angstrom:.5f} {z_angstrom:.5f}"
        for x_angstrom, y_angstrom, z_angstrom in whole_angstrom
    ]
    (OUT_DIR / f"C{n_atoms}-{temperature_kelvin}K.xyz").write_text("\n".join(lines) + "\n")
    record: dict[str, object] = {
        "archive_path": str(source),
        "archive_size_bytes": len(raw),
        "archive_md5": hashlib.md5(raw).hexdigest(),
        "last_timestep": last_timestep,
        "frame_count": raw.count(b"ITEM: TIMESTEP"),
        "n_atoms": n_atoms,
        "temperature_kelvin": temperature_kelvin,
        "cell_edge_angstrom": round(edge_angstrom, 6),
        **log_facts(run_dir / "log.lammps", last_timestep),
        "final_restart_present": (run_dir / "final.restart").exists(),
    }
    copied_from = copy_of(archive_root, run_dir)
    if copied_from is not None:
        record["copy_of"] = copied_from
    return record


def main() -> None:
    """Extract all 48 frames and write provenance.json; failures are logged, not fatal."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive", type=Path, default=ARCHIVE_ROOT)
    archive_root = parser.parse_args().archive
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, object]] = {}
    for temperature_kelvin in TEMPERATURES_KELVIN:
        for n_atoms in SIZES:
            name = f"C{n_atoms}-{temperature_kelvin}K.xyz"
            try:
                records[name] = extract_one(archive_root, n_atoms, temperature_kelvin)
                print(f"{name} ok", flush=True)
            except (OSError, ValueError) as error:
                records[name] = {"error": repr(error)}
                print(f"{name} FAILED {error!r}", file=sys.stderr, flush=True)
    (OUT_DIR / "provenance.json").write_text(json.dumps(records, indent=2) + "\n")
    failed = [name for name, record in records.items() if "error" in record]
    print(f"{len(records) - len(failed)} of {len(records)} written; failed: {failed or 'none'}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
