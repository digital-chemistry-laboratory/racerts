"""
Helpers: logging (log), optional dependencies (optional), unit conversions (units),
checks of argument values (checks), seeds and random draws (seeds).
"""

# racerts.utils of legacy racerts had these functions; they live in racerts.compat.
_LEGACY = (
    "EV_TO_KCAL_MOL",
    "atom_idx_input_validation",
    "count_electrons",
    "get_frozen_atoms",
    "infer_charge_and_multiplicity",
    "suppress_std",
)


def __getattr__(name):
    if name in _LEGACY:
        from racerts.compat import utils

        return getattr(utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
