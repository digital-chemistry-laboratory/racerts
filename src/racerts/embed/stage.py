"""The Embed stage."""

import copy
import logging
from typing import Callable, Optional, Sequence, Union

import numpy as np
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
from racerts.restraints.model import OPTIONAL_SOURCES, accepts_restraints
from racerts.system.spec import rigid_body_dof
from racerts.task import FrozenSet
from racerts.utils.checks import is_integer

from .base import BaseEmbedder
from .bounds import InconsistentRestraints, windows_fit
from .dg import BoundsMatrixEmbedder, CmapEmbedder, DistanceGeometryEmbedder

logger = logging.getLogger(__name__)

DEFAULT_CONF_FACTOR = 80
DEFAULT_HINT_SHARE = 0.3
# RDKit attempts per conformer within which a hint has to embed. On a 146-atom peptide
# every hint that occurs in its ensemble embedded within 20; the ones that never embed
# run for minutes with RDKit's own limit (ten attempts per atom).
HINT_ATTEMPTS = 20
FRACTION_BATCHES = 10  # with restraint_fraction: batches that draw their restraints
STREAM = 11  # the random stream of those draws, apart from the embedding's
EMBED_MODES = {"cmap": CmapEmbedder, "bounds": BoundsMatrixEmbedder}
COUNT_POLICIES = ("legacy", "fragments", "per_bond")

CountPolicy = Union[str, Callable[[Chem.Mol, Optional[FrozenSet]], int]]


def conformer_count(
    mol: Chem.Mol,
    number_of_conformers: int = -1,
    conf_factor: int = DEFAULT_CONF_FACTOR,
    policy: CountPolicy = "legacy",
    frozen: Optional[FrozenSet] = None,
    fragments=None,
) -> int:
    """
    number_of_conformers, or for -1 the count of the policy (n_rot: rotatable bonds):
    - "legacy": n_rot * conf_factor + 30 (legacy racerts);
    - "fragments": (n_rot + rigid-body freedom of the fragments that move relative to
      the frozen core, e.g. solvent molecules; by their roles if fragments are given,
      see racerts.system.roles) * conf_factor + 30;
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
        return (n_rot + rigid_body_dof(mol, core, fragments)) * conf_factor + 30
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
        hint_attempts: A hint is a potential interaction: the embedding decides
            whether the molecule can have it. A conformer of a hint batch that does not
            embed within this many RDKit attempts is embedded without the hints
            instead, so the count stays; a hint batch that gives no conformer at all is
            dropped, which is logged and recorded ("dropped_hints"). The number of
            attempts, not the time, decides: the same seed gives the same ensemble on
            every machine. 0: RDKit's limit (ten attempts per atom).

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
        restraint_fraction: float = 1.0,
        hint_attempts: int = HINT_ATTEMPTS,
    ):
        if not is_integer(hint_attempts) or hint_attempts < 0:
            raise ValueError("hint_attempts must be a non-negative integer.")
        if not is_integer(n_conformers):
            raise TypeError(f"n_conformers must be an integer, not {n_conformers!r}.")
        if n_conformers != -1 and n_conformers < 1:
            raise ValueError("n_conformers must be positive, or -1 for the default.")
        if isinstance(references, str) and references != "all":
            raise ValueError("references must be None, 'all' or conformer ids.")
        if not 0 <= hint_share <= 1:
            raise ValueError("hint_share must be between 0 and 1.")
        if not 0 < restraint_fraction <= 1:
            raise ValueError("restraint_fraction must be in (0, 1].")
        self.hint_share = hint_share
        self.hint_attempts = int(hint_attempts)
        self.restraint_fraction = restraint_fraction
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
            ctx.fragments,
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
        fractional = self.restraint_fraction < 1 and any(
            r.source in OPTIONAL_SOURCES for r in restraints
        )
        if (
            references is None
            and not hinted
            and not stratified
            and not windowed
            and not fractional
        ):
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
                batch_embedder = _batch_embedder(embedder, k)
                parts = self._embed_batch(
                    batch_embedder, ctx, reference, count, extra, batch
                )
                for part, labels in parts:
                    part.add_provenance(
                        **{
                            **provenance,
                            "seed": getattr(batch_embedder, "randomSeed", None),
                        }
                    )
                    if ref_id is not None:
                        part.add_provenance(reference=ref_id)
                    if labels:
                        part.add_provenance(**labels)
                    ensemble = part if ensemble is None else ensemble.merge(part)
        if len(ensemble) == 0:
            raise no_conformers_error(ctx.frozen)
        record_active_lengths(ctx, ensemble)
        keep_embedded_lengths(ctx, ensemble)
        return ensemble

    def _embed_batch(self, embedder, ctx, reference, count, restraints, batch):
        """
        The conformers of one batch as (ensemble, provenance of the batch) parts. A
        batch without hints is one part. Hint batches get hint_attempts attempts per
        conformer; what does not embed within them is embedded without the hints, as
        a second part (see hint_attempts).
        """
        hints = [r.label for r in restraints or () if r.source == "hint"]
        if not hints or not self.hint_attempts:
            return [
                (self._embed(embedder, ctx, reference, count, restraints, False), batch)
            ]
        limited = copy.copy(embedder)
        limited.max_attempts = self.hint_attempts
        hinted = self._embed(limited, ctx, reference, count, restraints, check=False)
        missing = count - len(hinted)
        if not missing:
            return [(hinted, batch)]
        without = [r for r in restraints if r.source != "hint"]
        unhinted = {**batch, "active_restraints": []}
        if not len(hinted):
            unhinted["dropped_hints"] = hints
            logger.info(
                "The hint%s %s did not embed within %d attempts per conformer: "
                "dropped as not possible for this molecule; %d conformers are "
                "embedded without.",
                "s" if len(hints) > 1 else "",
                ", ".join(hints),
                self.hint_attempts,
                missing,
            )
        rest = self._embed(embedder, ctx, reference, missing, without, check=False)
        return (
            [(hinted, batch), (rest, unhinted)] if len(hinted) else [(rest, unhinted)]
        )

    def _batches(self, ctx, n):
        """
        (count, restraints, provenance) per batch, each embedded with its own seed: per
        target (stratified active bonds), the conformers without hints and then the
        hint batches; with restraint_fraction < 1, each of these in batches of about a
        tenth of n that take every optional restraint (OPTIONAL_SOURCES) with that
        probability, recorded as "restraint_subset".
        """
        restraints = ctx.restraints.for_stage("embed")
        hints = [r for r in restraints if r.source == "hint"]
        base = [r for r in restraints if r.source != "hint"]
        groups = [(n, base, {})]
        if getattr(ctx.task, "stratify", 0):
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
            groups = [
                (
                    size,
                    target_windows(base, target, EMBED_TARGET_HALF_WIDTH),
                    target_provenance(target),
                )
                for size, target in zip(sizes, targets)
                if size
            ]
        batches = []
        for size, group, where in groups:
            batches += [
                (count, chosen, {**where, **extra})
                for count, chosen, extra in self._hint_batches(ctx, size, group, hints)
            ]
        if self.restraint_fraction < 1:
            batches = self._fraction_batches(ctx, n, batches)
        return batches

    def _fraction_batches(self, ctx, n, batches):
        """The batches split into FRACTION_BATCHES batches (of n) that take every
        optional restraint with probability restraint_fraction."""
        if not any(r.source in OPTIONAL_SOURCES for _, rs, _ in batches for r in rs):
            return batches
        rng = np.random.default_rng(None if ctx.seed < 0 else [ctx.seed, STREAM])
        size = max(1, n // FRACTION_BATCHES)
        split = []
        for count, chosen, extra in batches:
            fixed = [r for r in chosen if r.source not in OPTIONAL_SOURCES]
            optional = [r for r in chosen if r.source in OPTIONAL_SOURCES]
            while count > 0:
                m = min(size, count)
                count -= m
                subset = [r for r in optional if rng.random() < self.restraint_fraction]
                labels = {"restraint_subset": [r.label for r in subset]}
                split.append((m, fixed + subset, {**extra, **labels}))
        return split

    def _hint_batches(self, ctx, n, base, hints):
        """The batch without hints first, then the hint batches."""
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
        ensemble = ConformerEnsemble(mol)
        widened = getattr(embedder, "widened", None)
        if widened:  # bounds widened to the reference (see reference_bounds)
            ensemble.add_provenance(widened_bounds=len(widened))
        return ensemble


def _batch_embedder(embedder, k: int):
    """
    The embedder of batch k: a copy with its own seed for k > 0 (if it has one), and
    with a seed per conformer. The legacy seeds embed the first three conformers of a
    call twice, which would repeat in every batch; legacy racerts has no batches, so
    there is no legacy result to keep.
    """
    seed = getattr(embedder, "randomSeed", None)
    own_seed = k > 0 and seed is not None and seed >= 0
    legacy_seeds = getattr(embedder, "sequential_seeds", True) is False
    if not (own_seed or legacy_seeds):
        return embedder
    batch_embedder = copy.copy(embedder)
    if own_seed:
        batch_embedder.randomSeed = (seed + 7919 * k) % (2**31 - 1)  # RDKit: 31 bits
    if legacy_seeds:
        batch_embedder.sequential_seeds = True
    return batch_embedder
