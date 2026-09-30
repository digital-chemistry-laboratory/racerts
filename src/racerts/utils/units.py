"""Unit conversions used across racerts."""

EV_TO_KCAL_MOL = 23.06054783061903
HARTREE_TO_KCAL_MOL = 627.5094740629
KJ_TO_KCAL = 1 / 4.184

# Energies stored in kcal/mol, in other units: value(kcal/mol) / ENERGY_UNITS[unit].
ENERGY_UNITS = {
    "kcal/mol": 1.0,
    "kJ/mol": KJ_TO_KCAL,
    "eV": EV_TO_KCAL_MOL,
    "hartree": HARTREE_TO_KCAL_MOL,
}
