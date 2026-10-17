"""MMFF94 of RDKit as an ASE calculator, for the tests of stages that run dynamics."""

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from rdkit import Chem
from rdkit.Chem import AllChem

from racerts.utils.units import EV_TO_KCAL_MOL


class MMFFCalculator(Calculator):
    """MMFF94 on the fixed graph of mol (explicit hydrogens): energy (eV), forces (eV/A)."""

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, mol: Chem.Mol, **kwargs):
        super().__init__(**kwargs)
        graph = Chem.Mol(mol)
        # all atoms in one place: RDKit leaves out the pairs that are far apart in the
        # conformer it builds the force field from, and here none is
        graph.RemoveAllConformers()
        graph.AddConformer(Chem.Conformer(graph.GetNumAtoms()))
        self._graph = graph
        self._numbers = np.array([atom.GetAtomicNum() for atom in graph.GetAtoms()])
        self._build()

    def _build(self):
        properties = AllChem.MMFFGetMoleculeProperties(self._graph)
        self._field = AllChem.MMFFGetMoleculeForceField(self._graph, properties)
        self._field.Initialize()

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        assert np.array_equal(self.atoms.numbers, self._numbers)
        positions = self.atoms.get_positions().ravel().tolist()
        energy = self._field.CalcEnergy(positions) / EV_TO_KCAL_MOL
        gradient = np.array(self._field.CalcGrad(positions)).reshape(-1, 3)
        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": -gradient / EV_TO_KCAL_MOL,
        }

    def __getstate__(self):  # the force field is not picklable: built again in a worker
        return {**self.__dict__, "_field": None}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._build()


def numerical_forces(atoms, step=1e-4):
    """-dE/dx by central differences (eV/A)."""
    forces = np.zeros((len(atoms), 3))
    start = atoms.get_positions()
    for atom in range(len(atoms)):
        for axis in range(3):
            energies = []
            for sign in (1, -1):
                moved = start.copy()
                moved[atom, axis] += sign * step
                atoms.set_positions(moved)
                energies.append(atoms.get_potential_energy())
            forces[atom, axis] = -(energies[0] - energies[1]) / (2 * step)
    atoms.set_positions(start)
    return forces
