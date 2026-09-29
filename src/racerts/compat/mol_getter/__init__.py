"""racerts.mol_getter of legacy racerts."""

from .mol_getter import (
    BaseMolGetter,
    MolGetterBonds,
    MolGetterConnectivity,
    MolGetterSMILES,
)

mol_getters = {
    "base": BaseMolGetter,
    "bonds": MolGetterBonds,
    "connect": MolGetterConnectivity,
    "smiles": MolGetterSMILES,
}

__all__ = [
    "BaseMolGetter",
    "MolGetterBonds",
    "MolGetterConnectivity",
    "MolGetterSMILES",
    "mol_getters",
]
