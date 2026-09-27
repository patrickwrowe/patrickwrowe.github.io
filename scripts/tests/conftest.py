"""Shared pytest setup for scripts/tests: puts scripts/figures on sys.path.

The figure and analysis scripts (boxprep.py, cluster_census.py, extract_spheres.py, ...)
import their siblings by plain name — `import boxprep`, `from cluster_census import
SIZES` — which works when a script runs directly (its own directory is `sys.path[0]`)
but not when pytest imports a test module from `scripts/tests/`. Inserting the path here,
once, before collection, replaces the `sys.path.insert` every test file used to carry.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "figures"))
