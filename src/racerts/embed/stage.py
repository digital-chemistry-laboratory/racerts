"""The Embed stage."""

import logging
from typing import Optional

from rdkit import Chem
from rdkit.Chem import Descriptors

from racerts.pipeline import ConformerEnsemble

from .base import BaseEmbedder
from .dg import BoundsMatrixEmbedder, CmapEmbedder, DistanceGeometryEmbedder

logger = logging.getLogger(__name__)

DEFAULT_CONF_FACTOR = 80
EMBED_MODES = {"cmap": CmapEmbedder, "bounds": BoundsMatrixEmbedder}


def conformer_count(
    mol: Chem.Mol,
    number_of_conformers: int = -1,
    conf_factor: int = DEFAULT_CONF_FACTOR,
) -> int:
    """number_of_conformers, or for -1: rotatable bonds * conf_factor + 30."""
    if number_of_conformers == -1:
        return (
            Descriptors.NumRotatableBonds(mol)  # type: ignore[attr-defined]
        ) * conf_factor + 30
    return number_of_conformers


def default_embedder(
    task, seed: int, mode: str = "cmap", etkdg: Optional[bool] = None, **settings
) -> DistanceGeometryEmbedder:
    """
    The embedder for a task. When atoms are held at a reference, as in legacy racerts:
    plain distance geometry with the legacy chirality fallback. Otherwise (ground states)
    ETKDGv3 without the fallback, so the stereocentres of the input are kept.

    Args:
        mode: "cmap" (CmapEmbedder) or "bounds" (BoundsMatrixEmbedder).
        etkdg: Overrides the choice between ETKDGv3 and plain distance geometry.
        settings: Further arguments of the embedder, e.g. num_threads.
    """
    fixed = task.needs_reference
    return EMBED_MODES[mode](
        randomSeed=seed,
        etkdg=(not fixed) if etkdg is None else etkdg,
        chirality_fallback=fixed,
        **settings,
    )


def no_conformers_error(frozen) -> RuntimeError:
    hint = (
        "Check the reacting atoms and the TS geometry, or try another embedder."
        if frozen
        else "Try more conformers (n_conformers) or another embedder."
    )
    return RuntimeError(f"Embedding produced no conformers. {hint}")


class Embed:
    """
    Embeds conformers of the context's graph, with the frozen atoms of the task at the
    reference positions.

    Args:
        embedder: Any BaseEmbedder; default: default_embedder for the task, with
            the context's seed.
        n_conformers: Number of conformers to embed; -1 for the default count
            (rotatable bonds * conf_factor + 30).
        conf_factor: Conformers per rotatable bond for the default count.

    Embed starts an ensemble; it raises if it gets one. An embedder passed in keeps its
    own settings, including its seed.
    """

    name = "embed"

    def __init__(
        self,
        embedder: Optional[BaseEmbedder] = None,
        n_conformers: int = -1,
        conf_factor: int = DEFAULT_CONF_FACTOR,
    ):
        self.embedder = embedder
        self.n_conformers = n_conformers
        self.conf_factor = conf_factor

    def run(
        self, ctx, ensemble: Optional[ConformerEnsemble] = None
    ) -> ConformerEnsemble:
        if ensemble is not None:
            raise ValueError(
                "Embed starts an ensemble. To combine the conformers of two runs, "
                "merge their ensembles (ConformerEnsemble.merge)."
            )
        embedder = self.embedder
        if embedder is None:
            embedder = default_embedder(ctx.task, ctx.seed)
        mol = ctx.graph()
        n = conformer_count(mol, self.n_conformers, self.conf_factor)
        logger.info("Embedding %d conformers with %s.", n, type(embedder).__name__)
        embedder.embed(mol, ctx.reference, ctx.frozen, n)
        if mol.GetNumConformers() == 0:
            raise no_conformers_error(ctx.frozen)

        embedded = ConformerEnsemble(mol)
        provenance = {
            "embedder": type(embedder).__name__,
            "seed": getattr(embedder, "randomSeed", None),
        }
        if getattr(embedder, "etkdg", None) is not None:
            provenance["etkdg"] = embedder.etkdg
        embedded.add_provenance(**provenance)
        return embedded
