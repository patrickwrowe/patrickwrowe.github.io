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

# Covalent radii in angstrom, every one from Cordero et al., "Covalent radii revisited",
# Dalton Trans. 2008, 2832 (Table 1; carbon is the sp3 value). A pair is bonded
# within BOND_TOLERANCE times the sum of its two radii, which puts C-C at 1.824 A: above
# the 1.55 A single bond, below the 2.4 A second neighbour in every phase these searches
# produce. It is the 1.8 A cutoff render_cluster.py drew its published cage figures with
# before it was generalised beyond carbon, and the cutoff of the cluster census
# (cluster_census.py), so the bonds counted are the bonds drawn.
COVALENT_RADIUS_ANGSTROM = {
    "C": 0.76,
    "H": 0.31,
    "O": 0.66,
    "B": 0.84,
    "N": 0.71,
    "S": 1.05,
    "Mo": 1.54,
    "Li": 1.28,
    "K": 2.03,
    "Mg": 1.41,
    "Na": 1.66,
    "Si": 1.11,
    "Al": 1.21,
}
BOND_TOLERANCE = 1.2
# Intercalated or interlayer ions, drawn as spheres and never bonded.
BARE_IONS = frozenset({"Li", "K", "Na"})
# Layer cations: bonded to their anions by the radius rule, never to each other.
LAYER_METALS = frozenset({"Mo", "Mg", "Al"})
# Every metal with a covalent radius above. In a protein panel a lone ion of one of them, a
# one-atom HETATM residue bonded to no other hetero atom (3BSD's Mg 3262, occupancy 0.33, whose
# partners were waters), is not drawn (render_box_grid.prepare).
METALS = BARE_IONS | LAYER_METALS
# The flat carbon-carbon cutoff: 1.2 x (0.76 + 0.76) = 1.824 A.
CARBON_BOND_CUTOFF_ANGSTROM = BOND_TOLERANCE * 2 * COVALENT_RADIUS_ANGSTROM["C"]

# Species colours for renders of more than one species (Patrick, 2026-10-05). Low chroma,
# lightness between --graphite and --ink, told apart by lightness as well as hue; hydrogen
# is the lightest (oxygen sits just above graphite), carbon is the site's grey, no species
# takes --lustre's gold, and a single-species render stays greyscale. The same values are
# the --species-* tokens in src/styles/tokens.css; scripts/tests/test_palette.py fails if
# the two drift. sRGB hex.
SPECIES_COLOUR: dict[str, str] = {
    "C": "#4a4a4a",
    "H": "#a4a4a4",
    "O": "#a55445",
    "N": "#405b88",
    "B": "#562b36",
    "S": "#53562a",
    "Mo": "#143a39",
    "Mg": "#a64bf4",
    "Li": "#72598d",
    "K": "#394c76",
    "Na": "#164547",
    "Si": "#4a5c64",
    "Al": "#5a3547",
}

# Protein cartoon colours, one per chain in a panel's `cartoon_chains` order (Patrick,
# 2026-10-09: "just black" does not suit a protein). In the species palette's key: low
# chroma, lightness at the graphite end of graphite to ink (CIELAB L* 33-43), no gold, and
# each at least CIEDE2000 13.0 from every SPECIES_COLOUR and --graphite, --ink and --lustre, so
# a ligand's carbon sticks and species spheres still read against the ribbon. Verdigris,
# umber, old rose, petrol: the first two, the pair most often in contact (an antibody's light
# and heavy chains), differ by 33. A manifest's `chain_colour` beats this per chain. The same
# values are the --chain-* tokens in src/styles/tokens.css (scripts/tests/test_palette.py).
# sRGB hex.
CHAIN_COLOUR: tuple[str, ...] = ("#3f6e5b", "#664634", "#7c5e63", "#1c707a")

# The matte finish every Blender panel's material takes (Patrick, 2026-10-09): molrender
# Solid fields, so `Solid(rgb=..., **MATTE)`. A rough, coat-free surface with a low specular
# level has no hard white highlight, so the lightness gradient across each sphere and ribbon
# comes from the light rig alone; no subsurface, which would lighten a dark colour. This is
# candidate c3's recipe, roughness/specular 0.9/0.15, as Patrick judged it on 2026-10-15; the
# other two candidates rendered 1.0/0.05 and 0.95/0.1. With every light fixed, 0.9/0.2 against
# 1.0/0.05 moved the carbon slab's mean ink luma by 0.003, so the choice is about sheen, not tone.
MATTE: dict[str, float] = {"roughness": 0.9, "coat": 0.0, "subsurface": 0.0, "specular": 0.15}

# The light rig every Blender panel takes (Patrick chose candidate c3-three-point-plus,
# 2026-10-15), as the keyword arguments of molrender `Light`: the three-point rig at half
# molrender's 4 W key (its fill and rim ratios stay at their defaults), a sun fill
# (W/m2, a SUN lamp's irradiance) just above the camera axis, and an area light perpendicular
# to the view on the image's left (W at 1 m, scaled by distance squared). Measured on
# 2026-10-09 at the production tier: zero clipped pixels on the antibodies hero, 6TRO, the FMO
# hero and the carbon slab, and the 5th percentile of protein ink luma no lower than 0.133.
RIG: dict[str, float] = {"key_power": 2.0, "sun_strength": 1.5, "side_power": 12.0}

# molrender `World.strength`: the studio HDRI's share of the ambient light. The backdrop stays
# pure white whatever it is (the camera sees `World.background`, the subject sees the HDRI).
WORLD_STRENGTH: float = 0.8

# Molecular Nodes' van der Waals radii for the species of SPECIES_COLOUR (molecularnodes 4.5.13,
# assets/data.py, `elements[symbol]["vdw_radii"]`, given there in picometres). Its BallAndStick
# style reads sphere_radius as a factor on each atom's own radius, not as a length, so a
# manifest's sphere_radius_angstrom is divided by carbon's: carbon is drawn at that length and
# every other species at its own radius times the same factor (potassium 1.62 times carbon).
VDW_RADIUS_ANGSTROM = {
    "C": 1.70,
    "H": 1.20,
    "O": 1.52,
    "N": 1.55,
    "B": 1.92,
    "S": 1.80,
    "Mo": 2.39,
    "Mg": 1.73,
    "Li": 1.82,
    "K": 2.75,
    "Na": 2.27,
    "Si": 2.10,
    "Al": 1.84,
}


def srgb_hex_to_linear(hex_colour: str) -> tuple[float, float, float]:
    """sRGB hex ("#rrggbb") to linear RGB in 0-1, the space Blender's Solid material takes.

    The token becomes the material's base colour, which the Standard view transform leaves
    un-tone-mapped: a flat patch of that material lit at unit irradiance would render back to
    the token's sRGB. A lit sphere does not: its pixels are the base colour times the light
    reaching each point (plus the specular), so it shades through many tones, and the token
    is the species' colour, not any one pixel of it.

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
