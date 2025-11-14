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
