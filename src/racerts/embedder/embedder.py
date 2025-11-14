from abc import abstractmethod
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.AllChem import EmbedMultipleConfs  # type: ignore

from .utils import get_bounds_matrix, print_bounds_matrix_errors


class BaseEmbedder:
    @abstractmethod
    def __init__(
        self,
        verbose: bool = False,
        randomSeed: int = 12,
        pruneRmsThresh: Optional[float] = -1,
        remove_all_conformers: bool = True,
        ETversion: int = 2,
        useRandomCoords: bool = True,
        **kwargs,
    ):
        raise NotImplementedError

    @abstractmethod
    def embed_TS(
        self, mol_ts: Chem.Mol, mol: Chem.Mol, reacting_atoms, frozen_atoms, n, verbose
    ):
        """
        Embed a transition state
        """
        raise NotImplementedError


class BoundsMatrixEmbedder(BaseEmbedder):
    def __init__(
        self,
        verbose: bool = False,
        randomSeed: int = 12,
        pruneRmsThresh: Optional[float] = -1,
        remove_all_conformers: bool = True,
        ETversion: int = 2,
        useRandomCoords: bool = True,
        **kwargs,
    ):
        self.verbose = verbose
        self.randomSeed = randomSeed
        self.pruneRmsThresh = pruneRmsThresh
        self.remove_all_conformers = remove_all_conformers
        self.ETversion = ETversion
        self.useRandomCoords = useRandomCoords
        self.num_threads = kwargs.get("num_threads", 1)

    def embed_TS(self, mol_ts, mol, reacting_atoms, frozen_atoms, n=10, verbose=False):

        if not isinstance(mol, Chem.rdchem.Mol):
            raise Exception("Embedding: input for embedding is not a molecule!")

        bounds_matrix = get_bounds_matrix(
            mol_ts=mol_ts,
            new_mol=mol,
            reacting_atoms=reacting_atoms,
            frozen_atoms=frozen_atoms,
            verbose=verbose,
        )
        print_bounds_matrix_errors(bounds_matrix)

        params = AllChem.EmbedParameters()
        params.verbose = self.verbose
        params.ETversion = self.ETversion
        params.useMacrocycleTorsions = True
        params.useMacrocycle14config = True
        params.useSmallRingTorsions = True
        params.embedFragmentsSeparately = False
        params.clearConfs = False
        params.SetBoundsMat(bounds_matrix)
        params.trackFailures = True
        params.pruneRmsThresh = self.pruneRmsThresh
        params.randomSeed = self.randomSeed
        params.numThreads = self.num_threads
        if self.verbose:
            if self.useRandomCoords is False:
                print("Random Coordinates are not being used!")
            else:
                print("Random Coordinates are being used!")
        params.useRandomCoords = self.useRandomCoords

        chiral_check = min(n, 3)
        result = AllChem.EmbedMultipleConfs(mol, chiral_check, params)  # type: ignore
        error_counts = params.GetFailureCounts()
        if (
            len(result) < chiral_check / 2
            and error_counts[1] > params.maxIterations * chiral_check / 2
        ):
            if verbose:
                print(
                    "Erroneous initial conformers detected. Chiral Tags set to UNSPECIFIED"
                )
            for atom in mol.GetAtoms():
                atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            result = EmbedMultipleConfs(mol, n, params)
        elif (
            len(result) < chiral_check / 2
            and error_counts[6] > params.maxIterations * chiral_check / 2
        ):
            if verbose:
                print(
                    "Erroneous initial conformers detected. Enforce Chirality set to False"
                )
            params.enforceChirality = False
            result = EmbedMultipleConfs(mol, n, params)
        elif (
            len(result) < chiral_check / 2
            and error_counts[7] > params.maxIterations * chiral_check / 2
        ):
            if verbose:
                print(
                    "Erroneous initial conformers detected. Enforce Chirality set to False"
                )
            params.enforceChirality = False
            result = EmbedMultipleConfs(mol, n, params)
        elif chiral_check < n:
            rest_confs = n - chiral_check
            params.clearConfs = False
            result = EmbedMultipleConfs(mol, rest_confs, params)

        if mol.GetNumConformers() == 0:
            if self.useRandomCoords:
                print(
                    "Problem while generating the conformers... try a different approach"
                )
            else:
                print(
                    "Problem while generating the conformers... Random coordinates will be used."
                )
                params.useRandomCoords = True
                result = EmbedMultipleConfs(mol, n, params)

        error_counts = params.GetFailureCounts()
        if self.verbose:
            print("Embedding failure counts: ", error_counts)

        return result, error_counts


class CmapEmbedder(BaseEmbedder):
    def __init__(
        self,
        verbose: bool = False,
        randomSeed: int = 12,
        pruneRmsThresh: Optional[float] = -1,
        remove_all_conformers: bool = True,
        ETversion: int = 2,
        useRandomCoords: bool = True,
        **kwargs,
    ):
        self.verbose = verbose
        self.randomSeed = randomSeed
        self.pruneRmsThresh = pruneRmsThresh
        self.remove_all_conformers = remove_all_conformers
        self.ETversion = ETversion
        self.useRandomCoords = useRandomCoords
        self.num_threads = kwargs.get("num_threads", 1)

    def embed_TS(self, mol_ts, mol, reacting_atoms, frozen_atoms, n=10, verbose=False):

        if not isinstance(mol, Chem.rdchem.Mol):
            raise Exception("Embedding: input for embedding is not a molecule!")

        cmap = {
            frozen_atoms[i]: mol_ts.GetConformer().GetAtomPosition(frozen_atoms[i])
            for i in range(len(frozen_atoms))
        }

        params = AllChem.EmbedParameters()
        params.verbose = self.verbose
        params.ETversion = self.ETversion
        params.useMacrocycleTorsions = True
        params.useMacrocycle14config = True
        params.useSmallRingTorsions = True
        params.embedFragmentsSeparately = False
        params.clearConfs = False
        params.SetCoordMap(cmap)  # type: ignore
        params.trackFailures = True
        params.pruneRmsThresh = self.pruneRmsThresh
        params.randomSeed = self.randomSeed
        params.numThreads = self.num_threads

        if self.verbose:
            if self.useRandomCoords is False:
                print("Random Coordinates are not being used!")
            else:
                print("Random Coordinates are being used!")
        params.useRandomCoords = self.useRandomCoords

        chiral_check = min(n, 3)
        result = EmbedMultipleConfs(mol, chiral_check, params)
        error_counts = params.GetFailureCounts()
        if (
            len(result) < chiral_check / 2
            and error_counts[1] > params.maxIterations * chiral_check / 2
        ):
            if verbose:
                print(
                    "Erroneous initial conformers detected. Chiral Tags set to UNSPECIFIED"
                )
            for atom in mol.GetAtoms():
                atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            result = EmbedMultipleConfs(mol, n, params)
        elif (
            len(result) < chiral_check / 2
            and error_counts[6] > params.maxIterations * chiral_check / 2
        ):
            if verbose:
                print(
                    "Erroneous initial conformers detected. Enforce Chirality set to False"
                )
            params.enforceChirality = False
            result = EmbedMultipleConfs(mol, n, params)
        elif (
            len(result) < chiral_check / 2
            and error_counts[7] > params.maxIterations * chiral_check / 2
        ):
            if verbose:
                print(
                    "Erroneous initial conformers detected. Enforce Chirality set to False"
                )
            params.enforceChirality = False
            result = EmbedMultipleConfs(mol, n, params)
        elif chiral_check < n:
            rest_confs = n - chiral_check
            params.clearConfs = False
            result = EmbedMultipleConfs(mol, rest_confs, params)

        if mol.GetNumConformers() == 0:
            if self.useRandomCoords:
                print(
                    "Problem while generating the conformers... try a different approach"
                )
            else:
                print(
                    "Problem while generating the conformers... Random coordinates will be used."
                )
                params.useRandomCoords = True
                result = EmbedMultipleConfs(mol, n, params)

        error_counts = params.GetFailureCounts()
        if self.verbose:
            print("Embedding failure counts: ", error_counts)

        return result, error_counts
