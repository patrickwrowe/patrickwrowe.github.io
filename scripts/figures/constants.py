"""Constants of the carbon figure and analysis scripts, one home each, with sources.

Physical constants, the bond rule every bond graph on the site uses, the radius the
Blender renderer scales its spheres by, the species palette both renderers share, the
outcome classes of the cluster census and the run grid of the 48 GAP sphere simulations.
Stdlib only, so both the project `.venv` and the molrender Blender environment
(render_box_grid.py) can import it.

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

# Species colours for renders of more than one species (Patrick, 2026-10-05). CPK hues,
# desaturated to the site's register; carbon is the site's grey and a single-species render
# stays greyscale. The same values are the --species-* tokens in src/styles/tokens.css;
# scripts/tests/test_palette.py fails if the two drift. sRGB hex.
SPECIES_COLOUR: dict[str, str] = {
    "C": "#4a4a4a",
    "H": "#c9c9c9",
    "O": "#b5533c",
    "N": "#4c6a9c",
    "B": "#b07a8a",
    "S": "#b89a3e",
    "Mo": "#4f8a87",
    "Mg": "#5f9a5f",
    "Li": "#8a6fae",
    "K": "#7f66a8",
    "Na": "#8a78b8",
    "Si": "#8a8a6a",
    "Al": "#9a9aa0",
}

# Molecular Nodes' van der Waals radius for carbon (molecularnodes assets/data.py). Its
# BallAndStick style reads sphere_radius as a factor on this, not as a length.
CARBON_VDW_RADIUS_ANGSTROM = 1.70


def srgb_hex_to_linear(hex_colour: str) -> tuple[float, float, float]:
    """sRGB hex ("#rrggbb") to linear RGB in 0-1, the space Blender's Solid material takes.

    Under the Standard view transform a linear value renders back to the sRGB it came from,
    so a token on the page and the same species in a PNG match.

    Args:
        hex_colour: Six hex digits after a "#".

    Returns:
        The red, green and blue channels in linear light, each 0-1 (dimensionless).
    """
    channels = tuple(
        int(hex_colour.lstrip("#")[index : index + 2], 16) / 255 for index in (0, 2, 4)
    )
    return tuple(
        channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    )


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
