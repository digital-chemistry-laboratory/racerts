"""Conformer searches of other programs as racerts ensembles: CREST and ORCA's GOAT.

    from racerts.interfaces import Crest, Goat

    crest = Crest(executable="/path/to/crest", threads=4).run(mol)
    goat = Goat(executable="/path/to/orca", workers=4).run(mol)

Each run starts from one conformer of the molecule in a new folder and returns the
program's conformers with its energies (hartree, stored in kcal/mol), for a benchmark
against the searches of racerts: refine them at one level with racerts.Refine and prune
them alike before comparing. read_crest and read_goat read the ensembles of runs made
elsewhere. Neither program is installed with racerts.
"""

from .crest import Crest, read_crest
from .goat import Goat, read_goat

__all__ = ["Crest", "Goat", "read_crest", "read_goat"]
