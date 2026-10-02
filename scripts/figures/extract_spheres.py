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

The archive is on a slow HDD: each trajectory is read once from disk (md5, size and frame
count come from those bytes; boxprep re-reads the file from the page cache and returns the
last timestep). About 350 MB in all.

A run that fails is logged and the rest go on. Its error goes to the dead-letter file
`provenance-failures.json` beside the frames, never into `provenance.json`: the records of
the runs that succeeded are merged into the existing `provenance.json` (written to a
temporary file and renamed into place), so a failed run keeps its last good record and the
frame it describes. A pass with no failures removes any old dead-letter file. The exit
status is 1 if any run failed; re-run the script to retry them. Each frame is written only
after its whole record is built, so a run that fails never leaves a new frame beside an
old record.

The archive is a required argument: it lives on whatever drive holds it, not in the repo.

Usage:
    uv run scripts/figures/extract_spheres.py --archive <.../Carbon_Cluster_RSS/2_LAMMPS_MD>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import boxprep

SERIES = "3_GAP_Spheres_Opt2"
TEST_SERIES = "1_RSS_Spheres_Test"
OUT_DIR = Path(__file__).resolve().parent / "data" / "carbon-clusters" / "spheres"
PROVENANCE_NAME = "provenance.json"
FAILURES_NAME = "provenance-failures.json"
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
    if not test_log.exists():
        return None
    test_log_md5 = md5_of(test_log)
    if test_log_md5 != md5_of(run_dir / "log.lammps"):
        return None
    test_trajectory = test_dir / "liquid_carbon.lammpstrj"
    return {
        "trajectory_path": str(test_trajectory),
        "trajectory_md5": md5_of(test_trajectory),
        "log_path": str(test_log),
        "log_md5": test_log_md5,
    }


def extract_one(
    archive_root: Path, out_dir: Path, n_atoms: int, temperature_kelvin: int
) -> dict[str, object]:
    """Write one run's whole last frame and return its provenance record.

    The record is built in full, log and test-set comparison included, before the frame
    is written, so a run that fails leaves the old frame in place.

    Args:
        archive_root: The `2_LAMMPS_MD` directory.
        out_dir: Directory the `C<n>-<T>K.xyz` frame is written to.
        n_atoms: Cluster size.
        temperature_kelvin: Thermostat temperature.

    Returns:
        The provenance record for `C<n>-<T>K.xyz`.

    Raises:
        OSError: If the trajectory or log cannot be read.
        ValueError: If the frame's atom count differs from `n_atoms`, or the dump or log
            is malformed.
    """
    run_dir = archive_root / SERIES / str(temperature_kelvin) / f"C{n_atoms}"
    source = run_dir / "liquid_carbon.lammpstrj"
    raw = source.read_bytes()
    positions_angstrom, edge_angstrom, last_timestep = boxprep.read_lammps_last_frame(source)
    if len(positions_angstrom) != n_atoms:
        raise ValueError(f"{source}: {len(positions_angstrom)} atoms, expected {n_atoms}")
    compact_angstrom = boxprep.make_compact(positions_angstrom, edge_angstrom)
    bonds = boxprep.find_bonds_periodic(compact_angstrom, edge_angstrom)
    whole_angstrom = boxprep.make_whole(compact_angstrom, edge_angstrom, bonds)
    whole_angstrom -= whole_angstrom.mean(axis=0)
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
    lines = [
        str(n_atoms),
        f"C{n_atoms} {temperature_kelvin} K last frame of liquid_carbon.lammpstrj",
    ]
    lines += [
        f"C {x_angstrom:.5f} {y_angstrom:.5f} {z_angstrom:.5f}"
        for x_angstrom, y_angstrom, z_angstrom in whole_angstrom
    ]
    (out_dir / f"C{n_atoms}-{temperature_kelvin}K.xyz").write_text("\n".join(lines) + "\n")
    return record


def write_json_atomically(path: Path, payload: dict) -> None:
    """Write `payload` as indented JSON to a temporary file beside `path`, then rename it.

    The rename is atomic on POSIX, so an interrupted write never leaves `path` truncated.

    Args:
        path: Destination file.
        payload: JSON-serialisable mapping.
    """
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    """Extract all 48 frames, merge their records and dead-letter the failures.

    Args:
        argv: Command-line arguments; None reads `sys.argv`.

    Returns:
        Process exit status: 0 when every run was extracted, 1 when any failed.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--archive", type=Path, required=True, help="the archive's 2_LAMMPS_MD directory"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=OUT_DIR, help=f"where frames go (default {OUT_DIR})"
    )
    args = parser.parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted: dict[str, dict[str, object]] = {}
    failures: dict[str, str] = {}
    for temperature_kelvin in TEMPERATURES_KELVIN:
        for n_atoms in SIZES:
            name = f"C{n_atoms}-{temperature_kelvin}K.xyz"
            try:
                extracted[name] = extract_one(args.archive, out_dir, n_atoms, temperature_kelvin)
                print(f"{name} ok", flush=True)
            except (OSError, ValueError) as error:
                failures[name] = repr(error)
                print(f"{name} FAILED {error!r}", file=sys.stderr, flush=True)
    provenance_path = out_dir / PROVENANCE_NAME
    if extracted:
        previous = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
        write_json_atomically(provenance_path, {**previous, **extracted})
    failures_path = out_dir / FAILURES_NAME
    if failures:
        write_json_atomically(failures_path, failures)
    else:
        failures_path.unlink(missing_ok=True)
    total = len(extracted) + len(failures)
    print(
        f"{len(extracted)} of {total} written; failed: {sorted(failures) or 'none'}"
        + (f" (errors in {failures_path})" if failures else "")
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
