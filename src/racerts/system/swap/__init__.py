"""
Swaps: replace a group of a molecule by a new fragment (graph surgery), keeping the
geometry of the other atoms.

apply_swap returns the new graph with every conformer of the reference: the kept atoms
at their coordinates and, for a single attachment, the fragment grafted rigidly along
the removed bond. racerts.swap then samples the new atoms (api.py).

The modules, in the order of a swap: model (Swap, SwapResult), selection (what comes
and what leaves), graph (the new graph and its checks), configuration (its stereo),
coordinates, apply (apply_swap), groups (several groups at once).
"""

from .apply import apply_swap
from .coordinates import rotation_between
from .groups import label_hydrogen, substitute_groups
from .model import (
    ATOM_STEREO,
    BOND_STEREO,
    BOND_TYPES,
    MIN_KEPT_SHARE,
    MODES,
    Swap,
    SwapError,
    SwapResult,
)

__all__ = [
    "ATOM_STEREO",
    "BOND_STEREO",
    "BOND_TYPES",
    "MIN_KEPT_SHARE",
    "MODES",
    "Swap",
    "SwapError",
    "SwapResult",
    "apply_swap",
    "label_hydrogen",
    "rotation_between",
    "substitute_groups",
]
