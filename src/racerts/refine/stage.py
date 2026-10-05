"""The Refine stage and the MMFF -> UFF fallback."""

import logging
from typing import Callable, Optional, Type

from racerts.pipeline import ConformerEnsemble
from racerts.pipeline.ensemble import PROVENANCE
from racerts.restraints.active import (
    REFINE_TARGET_HALF_WIDTH,
    WINDOW_SOURCES,
    record_active_lengths,
    release_targets,
    target_windows,
)
from racerts.restraints.model import accepts_restraints, applying, position_restraints
from racerts.system.stereo import stereo_anchors

from .base import BaseOptimizer
from .forcefield import MMFFOptimizer, UFFOptimizer

logger = logging.getLogger(__name__)

REFINE_BACKENDS = {"mmff": MMFFOptimizer, "uff": UFFOptimizer}


def energy_label(optimizer) -> str:
    """
    The name of an optimizer's energies (the molecule property "energy_method"): its
    method label if it has one (ASEOptimizer(method=...)), else its class name.
    """
    return getattr(optimizer, "method", None) or type(optimizer).__name__


def refine_with_fallback(
    optimizer: BaseOptimizer,
    run: Callable[[BaseOptimizer], object],
    fallback: Optional[Type[UFFOptimizer]] = UFFOptimizer,
    num_threads: Optional[int] = None,
) -> str:
    """
    Call run(optimizer). If an MMFFOptimizer fails (e.g. MMFF has no parameters for
    an element) and there is a fallback (a UFFOptimizer class), log a warning and call
    run with a fallback optimizer of the same settings (num_threads: default that of
    the MMFFOptimizer) instead. Other errors are passed on.

    Returns:
        str: The name of the energies (energy_label of the optimizer that produced
        them).
    """
    try:
        run(optimizer)
    except Exception as e:
        if fallback is None or not isinstance(optimizer, MMFFOptimizer):
            raise
        logger.warning(
            "%s failed (%s); falling back to UFF. UFF energies are less reliable for "
            "ranking conformers.",
            type(optimizer).__name__,
            e,
        )
        failed = optimizer
        optimizer = fallback(
            verbose=failed.verbose,
            conf_id_ref=failed.conf_id_ref,
            force_constant=failed.force_constant,
            num_threads=failed.num_threads if num_threads is None else num_threads,
        )
        # Set afterwards: legacy UFF subclasses take only the arguments above.
        for setting in ("converge", "anchor_free_energies"):
            if hasattr(failed, setting):
                setattr(optimizer, setting, getattr(failed, setting))
        run(optimizer)
    return energy_label(optimizer)


class Refine:
    """
    Refines the conformers in place, with the hard frozen atoms of the task as anchors,
    and records the optimizer as the molecule property "energy_method".

    Args:
        optimizer: Any BaseOptimizer; default MMFFOptimizer.
        fallback: Fall back to UFF if MMFF fails.
        anchors: Hold the hard frozen atoms at the reference (default). False
            releases them, e.g. for a saddle-point search from TS-like conformers
            (ASEOptimizer with Sella), and with them the windows of a TS with
            active-bond windows: its conformers then are no longer at their targets
            and compare as one group (the provenance keeps the targets as
            "released_targets"). Other distance restraints and the soft atoms still
            hold.
        stereo_anchors: Also hold the free substituents that alone set the
            configuration of a frozen stereocentre (racerts.system.stereo),
            which the "frozen_first" embedding places at the reference; the default
            pipeline sets it with that fallback. Otherwise refinement can turn such a
            substituent through to the other stereoisomer.
    """

    name = "refine"

    def __init__(
        self,
        optimizer: Optional[BaseOptimizer] = None,
        fallback: bool = True,
        anchors: bool = True,
        stereo_anchors: bool = False,
    ):
        self.optimizer = optimizer
        self.fallback = fallback
        self.anchors = anchors
        self.stereo_anchors = stereo_anchors

    def run(self, ctx, ensemble: ConformerEnsemble) -> ConformerEnsemble:
        optimizer = self.optimizer if self.optimizer is not None else MMFFOptimizer()
        if (
            self.anchors
            and getattr(ctx.task, "windowed", False)
            and not accepts_restraints(optimizer._refine)
        ):
            raise ValueError(
                f"In window mode the reacting atoms are held by restraints, which "
                f"{type(optimizer).__name__} does not take, while their neighbours stay "
                "fixed. Refine with an optimizer that takes restraints "
                "(racerts.refine.MMFFOptimizer, UFFOptimizer or ASEOptimizer), or "
                "search the saddle point freely: Refine(optimizer, anchors=False)."
            )
        anchors = ctx.frozen.hard if self.anchors else ()
        if self.anchors and self.stereo_anchors and ctx.reference is not None:
            extra = [i for i in stereo_anchors(ctx.mol, ctx.frozen) if i not in anchors]
            anchors = (*anchors, *extra)
        restraints = ctx.restraints.for_stage("refine")
        released = not self.anchors and getattr(ctx.task, "windowed", False)
        if released:
            restraints = [r for r in restraints if r.source not in WINDOW_SOURCES]
        # soft atoms are held at the reference conformer the optimizer aligns on
        conf_id_ref = getattr(optimizer, "conf_id_ref", -1)
        groups = _refine_groups(ctx, ensemble, restraints, conf_id_ref, not released)
        if conf_id_ref != -1 and len({id(reference) for reference, *_ in groups}) > 1:
            raise ValueError(
                "With several references (Embed(references=...)) each conformer is "
                "refined against its own: the conf_id_ref of the optimizer must be -1."
            )

        def refine(opt):
            if len(groups) == 1:
                reference, _, group_restraints = groups[0]
                return opt.refine(ensemble.mol, reference, anchors, group_restraints)
            # Several references (Embed(references=...)) or active-bond targets: each
            # group against its own reference and targets.
            for reference, conf_ids, group_restraints in groups:
                part = ensemble.filter(conf_ids)
                opt.refine(part.mol, reference, anchors, group_restraints)
                _write_back(ensemble, part, conf_ids)

        energy_method = refine_with_fallback(
            optimizer,
            refine,
            fallback=UFFOptimizer if self.fallback else None,
        )
        ensemble.mol.SetProp("energy_method", energy_method)
        record_active_lengths(ctx, ensemble)
        if released:
            release_targets(ensemble)
        return ensemble


def _write_back(ensemble: ConformerEnsemble, part: ConformerEnsemble, conf_ids) -> None:
    """Positions, energies and provenance of the conformers of part into ensemble;
    conformers that part lost are removed."""
    kept = set(part.conf_ids)
    for conf_id in conf_ids:
        target = ensemble.mol.GetConformer(conf_id)
        if conf_id not in kept:
            ensemble.mol.RemoveConformer(conf_id)
            continue
        source = part.mol.GetConformer(conf_id)
        for atom, position in enumerate(source.GetPositions()):
            target.SetAtomPosition(atom, position.tolist())
        for key in ("energy", PROVENANCE):
            target.ClearProp(key)
        if source.HasProp("energy"):
            target.SetDoubleProp("energy", source.GetDoubleProp("energy"))
        if source.HasProp(PROVENANCE):
            target.SetProp(PROVENANCE, source.GetProp(PROVENANCE))


def _refine_groups(ctx, ensemble, base, conf_id_ref=-1, hold_targets=True):
    """
    (reference, conformer ids, restraints) groups: by reference (Embed(references=
    ...)), by active-bond target of a windowed TS (held at +/-
    REFINE_TARGET_HALF_WIDTH, unless hold_targets is off) and by the optional
    restraints of the embedding batch (Embed restraint_fraction). The soft atoms of
    the task get position restraints at their reference positions.
    """
    groups = []
    soft = ctx.frozen.soft
    for reference, conf_ids in ctx.by_reference(ensemble):
        held = []  # soft atoms near their reference positions
        if soft and reference is not None:
            held = position_restraints(reference, soft, conf_id_ref)
        by_target = {}
        for conf_id in conf_ids:
            provenance = ensemble.provenance(conf_id)
            targets = provenance.get("active_bond_targets") if hold_targets else None
            subset = provenance.get("restraint_subset")
            key = (
                None if not targets else tuple(sorted(targets.items())),
                None if subset is None else tuple(sorted(subset)),
            )
            by_target.setdefault(key, []).append(conf_id)
        for (targets, subset), ids in by_target.items():
            distances = base
            if subset is not None:
                distances = applying(base, {"restraint_subset": subset})
            if targets is not None:
                target = {tuple(map(int, pair.split("-"))): t for pair, t in targets}
                distances = target_windows(
                    distances,
                    target,
                    REFINE_TARGET_HALF_WIDTH,
                    ctx.task.target_force_constant,
                )
            groups.append((reference, ids, [*distances, *held]))
    return groups
