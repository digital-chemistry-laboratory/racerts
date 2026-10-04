"""The Embed stage."""

import copy
import logging
from typing import Callable, Optional, Sequence, Union

from rdkit import Chem
from rdkit.Chem import Descriptors

from racerts.pipeline import ConformerEnsemble
from racerts.restraints.active import (
    EMBED_TARGET_HALF_WIDTH,
    keep_embedded_lengths,
    record_active_lengths,
    target_provenance,
    target_windows,
)
from racerts.restraints.model import accepts_restraints
from racerts.system.spec import rigid_body_dof
from racerts.task import FrozenSet
from racerts.utils.checks import is_integer

from .base import BaseEmbedder
from .bounds import InconsistentRestraints, windows_fit
from .dg import BoundsMatrixEmbedder, CmapEmbedder, DistanceGeometryEmbedder

logger = logging.getLogger(__name__)

DEFAULT_CONF_FACTOR = 80
DEFAULT_HINT_SHARE = 0.3
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
        core = frozen.core if frozen is not None else ()
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
    if mode not in EMBED_MODES:
        raise ValueError(
            f"Unknown embed mode {mode!r}; use one of {tuple(EMBED_MODES)}."
        )
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
        hint_share: With hints among the restraints (source "hint", e.g. graph
            hydrogen bonds): the share of the conformers embedded in hint batches, one
            per hint and one with all hints together (if their windows are
            compatible); the others are embedded without hints. Each batch has its own
            seed; the provenance records the hints of each conformer
            ("active_restraints").

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
        hint_share: float = DEFAULT_HINT_SHARE,
    ):
        if not is_integer(n_conformers):
            raise TypeError(f"n_conformers must be an integer, not {n_conformers!r}.")
        if n_conformers != -1 and n_conformers < 1:
            raise ValueError("n_conformers must be positive, or -1 for the default.")
        if isinstance(references, str) and references != "all":
            raise ValueError("references must be None, 'all' or conformer ids.")
        if not 0 <= hint_share <= 1:
            raise ValueError("hint_share must be between 0 and 1.")
        self.hint_share = hint_share
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
        restraints = ctx.restraints.for_stage("embed")
        if restraints:
            if not accepts_restraints(embedder.embed):
                raise ValueError(
                    f"{type(embedder).__name__}.embed takes no restraints, so it cannot "
                    "embed with distance restraints."
                )
            common = [r.label for r in restraints if r.source != "hint"]
            if common:
                provenance["restraints"] = common

        references = self._references(ctx)
        stratified = getattr(ctx.task, "stratify", 0)
        hinted = any(r.source == "hint" for r in restraints)
        windowed = getattr(ctx.task, "windowed", False)
        if references is None and not hinted and not stratified and not windowed:
            logger.info("Embedding %d conformers with %s.", n, type(embedder).__name__)
            embedded = self._embed(embedder, ctx, ctx.reference, n)
            embedded.add_provenance(**provenance)
            return embedded

        if references is not None and windowed:
            raise ValueError(
                "Active-bond windows take one reference geometry (their windows are "
                "built around it): they do not combine with Embed(references=...)."
            )
        ensemble = None
        targets = [(None, ctx.reference)] if references is None else [
            (ref_id, ctx.reference_mol(ref_id)) for ref_id in references
        ]  # fmt: skip
        batches = self._batches(ctx, n)
        for ref_id, reference in targets:
            for k, (count, extra, batch) in enumerate(batches):
                logger.info(
                    "Embedding %d conformers with %s%s%s.",
                    count,
                    type(embedder).__name__,
                    "" if ref_id is None else f" from reference {ref_id}",
                    f" ({batch})" if batch else "",
                )
                batch_embedder = _with_seed_offset(embedder, k)
                part = self._embed(
                    batch_embedder, ctx, reference, count, extra, check=False
                )
                part.add_provenance(
                    **{
                        **provenance,
                        "seed": getattr(batch_embedder, "randomSeed", None),
                    }
                )
                if ref_id is not None:
                    part.add_provenance(reference=ref_id)
                if batch:
                    part.add_provenance(**batch)
                ensemble = part if ensemble is None else ensemble.merge(part)
        if len(ensemble) == 0:
            raise no_conformers_error(ctx.frozen)
        record_active_lengths(ctx, ensemble)
        keep_embedded_lengths(ctx, ensemble)
        return ensemble

    def _batches(self, ctx, n):
        """
        (count, restraints, provenance) per batch: target batches (stratified active
        bonds), else the batch without hints first, then the hint batches.
        """
        restraints = ctx.restraints.for_stage("embed")
        hints = [r for r in restraints if r.source == "hint"]
        base = [r for r in restraints if r.source != "hint"]
        if getattr(ctx.task, "stratify", 0):
            if hints:
                raise ValueError("Hints and stratified active bonds do not combine.")
            targets = ctx.task.targets(ctx.mol)
            sizes = [
                n // len(targets) + (i < n % len(targets)) for i in range(len(targets))
            ]
            if n < len(targets):
                logger.warning(
                    "%d conformers for %d targets: only the first %d targets are "
                    "sampled.",
                    n,
                    len(targets),
                    n,
                )
            return [
                (
                    size,
                    target_windows(base, target, EMBED_TARGET_HALF_WIDTH),
                    target_provenance(target),
                )
                for size, target in zip(sizes, targets)
                if size
            ]
        if not hints:
            return [(n, base, {})]
        subsets = [[hint] for hint in hints]
        if len(hints) > 1 and windows_fit(
            ctx.graph(), ctx.reference, ctx.frozen, base + hints
        ):
            subsets.append(hints)
        n_hint = round(self.hint_share * n)
        if n_hint == 0:
            return [(n, base, {"active_restraints": []})]
        subsets = subsets[:n_hint]  # at least one conformer per hint batch
        size = n_hint // len(subsets)
        batches = []
        if n > size * len(subsets):
            batches.append((n - size * len(subsets), base, {"active_restraints": []}))
        return batches + [
            (size, base + subset, {"active_restraints": [h.label for h in subset]})
            for subset in subsets
        ]

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
    def _embed(
        embedder, ctx, reference, n, restraints=None, check=True
    ) -> ConformerEnsemble:
        mol = ctx.graph()
        if restraints is None:
            restraints = ctx.restraints.for_stage("embed")
        if restraints:
            try:
                embedder.embed(mol, reference, ctx.frozen, n, restraints=restraints)
            except InconsistentRestraints as error:
                if not getattr(ctx.task, "windowed", False):
                    raise
                raise InconsistentRestraints(
                    f"{error} With active-bond windows: try a narrower active_window "
                    "(the TS core cannot take this one)."
                ) from None
        else:
            embedder.embed(mol, reference, ctx.frozen, n)
        if check and mol.GetNumConformers() == 0:
            raise no_conformers_error(ctx.frozen)
        return ConformerEnsemble(mol)


def _with_seed_offset(embedder, k: int):
    """For batch k > 0, a copy of the embedder with another seed (if it has one)."""
    seed = getattr(embedder, "randomSeed", None)
    if k == 0 or seed is None or seed < 0:
        return embedder
    batch_embedder = copy.copy(embedder)
    batch_embedder.randomSeed = (seed + 7919 * k) % (2**31 - 1)  # RDKit: 31 bits
    return batch_embedder
