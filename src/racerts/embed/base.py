"""The embedder interface."""

from abc import ABC, abstractmethod
from typing import Optional

from rdkit import Chem

from racerts.task import FrozenSet


class BaseEmbedder(ABC):
    """Embeds conformers into a molecule; subclasses implement embed."""

    @abstractmethod
    def embed(
        self, mol: Chem.Mol, reference: Optional[Chem.Mol], frozen: FrozenSet, n: int
    ):
        """
        Add n conformers to mol (a copy of the graph without conformers), keeping the
        hard and soft atoms of frozen at the positions of the reference (None for
        tasks without one). The return value is not used.

        An embedder that supports distance restraints takes them as a further
        argument, restraints=(): the Embed stage passes the DistanceRestraints of the
        embedding to embedders with that argument and raises for the others.
        """
        raise NotImplementedError
