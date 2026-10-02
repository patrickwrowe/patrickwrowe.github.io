"""Constants of the carbon figure and analysis scripts, one home each, with sources.

Physical constants, the bond rule every bond graph on the site uses, the radius the
Blender renderer scales its spheres by, the outcome classes of the cluster census and the
run grid of the 48 GAP sphere simulations. Stdlib only, so both the project `.venv` and
the molrender Blender environment (render_box_grid.py) can import it.

Units are in the names: lengths in angstrom, masses in g mol^-1, temperatures in kelvin.
"""

from __future__ import annotations

from enum import StrEnum

# Avogadro constant, exact since the 2019 SI redefinition (CODATA 2018).
AVOGADRO_PER_MOL = 6.02214076e23
# Standard atomic weight of carbon, IUPAC's conventional value.
CARBON_MASS_G_PER_MOL = 12.011
# 1 cm = 1e8 angstrom, by definition.
CM_TO_ANGSTROM = 1.0e8

# Covalent radii in angstrom (Cordero et al., Dalton Trans. 2008, 2832). A pair is bonded
# within BOND_TOLERANCE times the sum of its two radii, which puts C-C at 1.824 A: above
# the 1.55 A single bond, below the 2.4 A second neighbour in every phase these searches
# produce. It is the 1.8 A cutoff render_cluster.py drew its published cage figures with
# before it was generalised beyond carbon, and the cutoff of the cluster census
# (cluster_census.py), so the bonds counted are the bonds drawn.
COVALENT_RADIUS_ANGSTROM = {"C": 0.76, "H": 0.31, "O": 0.66}
BOND_TOLERANCE = 1.2
# The flat carbon-carbon cutoff: 1.2 x (0.76 + 0.76) = 1.824 A.
CARBON_BOND_CUTOFF_ANGSTROM = BOND_TOLERANCE * 2 * COVALENT_RADIUS_ANGSTROM["C"]

# Molecular Nodes' van der Waals radius for carbon (molecularnodes assets/data.py). Its
# BallAndStick style reads sphere_radius as a factor on this, not as a length.
CARBON_VDW_RADIUS_ANGSTROM = 1.70


class OutcomeClass(StrEnum):
    """The six outcome classes of the cluster census, in legend order, most ordered first.

    The rule that assigns them is `cluster_census.classify`.
    """

    DIAMOND_LIKE = "diamond-like"
    GRAPHITIC_ONION = "graphitic onion"
    CAGE = "cage"
    DISORDERED = "disordered"
    MOLTEN = "molten"
    DISSOCIATED = "dissociated"


# The run grid of the 48 GAP sphere simulations (Carbon_Cluster_RSS/2_LAMMPS_MD/
# 3_GAP_Spheres_Opt2/<T>/C<n>/): cluster sizes in atoms and thermostat temperatures.
SIZES = (40, 60, 80, 120, 160, 373, 686, 1000)
TEMPERATURES_KELVIN = (500, 1000, 2000, 3000, 4000, 5000)
