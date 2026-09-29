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
        frozen atoms at the positions of the reference.
        """
        raise NotImplementedError
