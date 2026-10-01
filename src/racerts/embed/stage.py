"""The Embed stage."""

import logging
from typing import Callable, Optional, Sequence, Union

from rdkit import Chem
from rdkit.Chem import Descriptors

from racerts.pipeline import ConformerEnsemble
from racerts.system.spec import rigid_body_dof
from racerts.task import FrozenSet

from .base import BaseEmbedder
from .dg import BoundsMatrixEmbedder, CmapEmbedder, DistanceGeometryEmbedder

logger = logging.getLogger(__name__)

DEFAULT_CONF_FACTOR = 80
EMBED_MODES = {"cmap": CmapEmbedder, "bounds": BoundsMatrixEmbedder}
COUNT_POLICIES = ("legacy", "fragments", "per_bond")

CountPolicy = Union[str, Callable[[Chem.Mol, Optional[FrozenSet]], int]]


def conformer_count(
    mol: Chem.Mol,
    number_of_conformers: int = -1,
    conf_factor: int = DEFAULT_CONF_FACTOR,
    policy: CountPolicy = "legacy",
    frozen: Optional[FrozenSet] = None,
) -> int:
    """
    number_of_conformers, or for -1 the count of the policy (n_rot: rotatable bonds):
    - "legacy": n_rot * conf_factor + 30 (legacy racerts);
    - "fragments": (n_rot + rigid-body freedom of the fragments that move relative to
      the frozen core, e.g. solvent molecules) * conf_factor + 30;
    - "per_bond": max(7, 10 * n_rot);
    - a callable policy(mol, frozen) that returns the count.
    """
    if number_of_conformers != -1:
        return number_of_conformers
    if callable(policy):
        return int(policy(mol, frozen))
    n_rot = Descriptors.NumRotatableBonds(mol)  # type: ignore[attr-defined]
    if policy == "legacy":
        return n_rot * conf_factor + 30
    if policy == "fragments":
        core = frozen.core if frozen else ()
        return (n_rot + rigid_body_dof(mol, core)) * conf_factor + 30
    if policy == "per_bond":
        return max(7, 10 * n_rot)
    raise ValueError(
        f"Unknown conformer count policy {policy!r}; use one of {COUNT_POLICIES} or a "
        "callable."
    )


def default_embedder(
    task,
    seed: int,
    mode: str = "cmap",
    etkdg: Optional[bool] = None,
    chirality_fallback: Union[bool, str] = "legacy",
    **settings,
) -> DistanceGeometryEmbedder:
    """
    The embedder for a task. When atoms are held at a reference, as in legacy racerts:
    plain distance geometry with a chirality fallback. Otherwise (ground states)
    ETKDGv3 without the fallback, so the stereocentres of the input are kept.

    Args:
        mode: "cmap" (CmapEmbedder) or "bounds" (BoundsMatrixEmbedder).
        etkdg: Overrides the choice between ETKDGv3 and plain distance geometry.
        chirality_fallback: The fallback for tasks with atoms held at a reference
            ("legacy" or "frozen_first", see DistanceGeometryEmbedder).
        settings: Further arguments of the embedder, e.g. num_threads.
    """
    fixed = task.needs_reference
    return EMBED_MODES[mode](
        randomSeed=seed,
        etkdg=(not fixed) if etkdg is None else etkdg,
        chirality_fallback=chirality_fallback if fixed else False,
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
        n_conformers: Number of conformers to embed; -1 for the count of the policy.
        conf_factor: Conformers per rotatable bond for the default count.
        count_policy: How -1 is counted (see conformer_count), default "legacy"
            (rotatable bonds * conf_factor + 30).
        references: With several reference geometries (conformers of the context's
            molecule, e.g. TSs from a TS search): "all", or a list of their conformer
            ids. The count is embedded for each; the provenance records "reference"
            (its conformer id), and Refine then refines each conformer against its
            reference. Default: the first conformer only.

    Embed starts an ensemble; it raises if it gets one. An embedder passed in keeps its
    own settings, including its seed.
    """

    name = "embed"

    def __init__(
        self,
        embedder: Optional[BaseEmbedder] = None,
        n_conformers: int = -1,
        conf_factor: int = DEFAULT_CONF_FACTOR,
        count_policy: CountPolicy = "legacy",
        references: Union[None, str, Sequence[int]] = None,
    ):
        if isinstance(references, str) and references != "all":
            raise ValueError("references must be None, 'all' or conformer ids.")
        self.embedder = embedder
        self.n_conformers = n_conformers
        self.conf_factor = conf_factor
        self.count_policy = count_policy
        self.references = references

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
        n = conformer_count(
            ctx.graph(),
            self.n_conformers,
            self.conf_factor,
            self.count_policy,
            ctx.frozen,
        )
        provenance = {
            "embedder": type(embedder).__name__,
            "seed": getattr(embedder, "randomSeed", None),
        }
        if getattr(embedder, "etkdg", None) is not None:
            provenance["etkdg"] = embedder.etkdg

        references = self._references(ctx)
        if references is None:
            logger.info("Embedding %d conformers with %s.", n, type(embedder).__name__)
            embedded = self._embed(embedder, ctx, ctx.reference, n)
            embedded.add_provenance(**provenance)
            return embedded

        ensemble = None
        for ref_id in references:
            logger.info(
                "Embedding %d conformers with %s from reference %d.",
                n,
                type(embedder).__name__,
                ref_id,
            )
            part = self._embed(embedder, ctx, ctx.reference_mol(ref_id), n, check=False)
            part.add_provenance(**provenance, reference=ref_id)
            ensemble = part if ensemble is None else ensemble.merge(part)
        if len(ensemble) == 0:
            raise no_conformers_error(ctx.frozen)
        return ensemble

    def _references(self, ctx) -> Optional[list]:
        if self.references is None:
            return None
        if ctx.reference is None:
            raise ValueError("Embed(references=...) needs reference geometries.")
        available = [conf.GetId() for conf in ctx.mol.GetConformers()]
        if self.references == "all":
            return available
        if not self.references:
            raise ValueError("references must name at least one conformer.")
        unknown = set(self.references) - set(available)
        if unknown:
            raise ValueError(
                f"No reference conformers with ids {sorted(unknown)} "
                f"(available: {available})."
            )
        return list(self.references)

    @staticmethod
    def _embed(embedder, ctx, reference, n, check=True) -> ConformerEnsemble:
        mol = ctx.graph()
        embedder.embed(mol, reference, ctx.frozen, n)
        if check and mol.GetNumConformers() == 0:
            raise no_conformers_error(ctx.frozen)
        return ConformerEnsemble(mol)
