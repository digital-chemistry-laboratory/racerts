"""The staged workflow: cheap filters, ranking at a higher level, Exploit, rescoring."""

from types import MappingProxyType
from typing import Optional, Sequence

from racerts.embed import Embed
from racerts.exploit import Exploit
from racerts.pipeline import Pipeline
from racerts.prune import PruneEnergy, PruneRMSD
from racerts.prune.energy import EnergyPruner
from racerts.refine import MMFFOptimizer, Refine
from racerts.validate import OVERLAP_FACTOR, AttackFace, Clash, Validate, gate

WINDOWS = (25.0, 8.0, 6.0)  # kcal/mol: cheap level, expensive level, final


class _WindowedTS:
    """Runs a stage only for a TS with active-bond windows and the stereo filter on, as
    the default pipeline runs AttackFace."""

    name = "attack_face"

    def __init__(self, stage):
        self.stage = stage

    def run(self, ctx, ensemble):
        task = ctx.task
        if getattr(task, "windowed", False) and getattr(task, "stereo_filter", False):
            return self.stage.run(ctx, ensemble)
        return ensemble


def _refine(stage_or_optimizer) -> Refine:
    if isinstance(stage_or_optimizer, Refine):
        return stage_or_optimizer
    return Refine(stage_or_optimizer)


def staged(
    expensive,
    cheap=None,
    windows: Sequence[float] = WINDOWS,
    embed: Optional[Embed] = None,
    exploit=MappingProxyType({}),
    rescore=None,
    clash_filter: Optional[float] = OVERLAP_FACTOR,
) -> Pipeline:
    """
    The staged workflow as a pipeline:

    1. embed (one Embed stage: its batches mix biased and unbiased settings), then drop
       conformers whose heavy atoms overlap;
    2. refine at the cheap level (default MMFF, converged), the validity gate
       (racerts.validate.gate), the forming-bond stereo filter for windowed TSs, and a
       loose energy window (windows[0]) with the RMSD pruning: this removes embedding
       artifacts and duplicates, while the cheap energies do not decide much;
    3. refine at the expensive level (an ASE calculator: xTB, an MLIP), the gate again,
       the ranking window (windows[1]) and the RMSD pruning;
    4. Exploit with the same refinement (unless exploit is None);
    5. the RMSD pruning of the whole, the final window (windows[2]), and optionally a
       Rescore stage (single points at a higher level or in the target solvent).

    For TSs the result goes to a TS optimization, e.g.
    Refine(ASEOptimizer(..., optimizer_cls=Sella), anchors=False) with
    Validate(ImaginaryModes(...), ReactionCore()).

    Args:
        expensive: The refinement of steps 3 and 4: a Refine stage, or an optimizer
            for one (an ASEOptimizer, e.g. with method="UMA-s-1p2").
        cheap: The refinement of step 2 (a Refine stage or an optimizer); default
            MMFFOptimizer(converge=True).
        windows: The energy windows (kcal/mol) of steps 2, 3 and 5.
        embed: The Embed stage; default Embed() with the defaults for the task.
        exploit: Settings of Exploit (a dict), an Exploit stage, or None for none.
        rescore: A Rescore stage at the end, or None.
        clash_filter: Drop embedded conformers with heavy atoms closer than this times
            their vdW sum (Clash; None: no filter). The default, 0.5, is below every
            lower bound of RDKit's distance geometry for these pairs, so it drops
            overlaps only: contacts at the edge of the bounds are relaxed by refinement
            (the gate after it checks clashes at 0.7), and dropping them can lose the
            conformer that is the lowest afterwards.
    """
    if clash_filter is not None and (
        isinstance(clash_filter, bool) or not isinstance(clash_filter, (int, float))
    ):
        raise TypeError(
            f"clash_filter is a factor of the vdW sum or None, not {clash_filter!r}."
        )
    if clash_filter is not None and not 0 < clash_filter <= 1:
        raise ValueError(f"clash_filter must be in (0, 1], not {clash_filter}.")
    windows = tuple(windows)
    if len(windows) != 3 or not all(w > 0 for w in windows):
        raise ValueError("windows needs three positive energies (kcal/mol).")
    cheap = _refine(cheap if cheap is not None else MMFFOptimizer(converge=True))
    expensive = _refine(expensive)
    stages = [embed if embed is not None else Embed()]
    if clash_filter is not None:
        stages.append(Validate(Clash(clash_filter), warn_above=0.3))
    stages += [
        cheap,
        gate(),
        _WindowedTS(Validate(AttackFace())),
        PruneEnergy(EnergyPruner(threshold=windows[0])),
        PruneRMSD(),
        expensive,
        gate(),
        PruneEnergy(EnergyPruner(threshold=windows[1])),
        PruneRMSD(),
    ]
    if exploit is not None:
        if not isinstance(exploit, Exploit):
            exploit = Exploit(expensive, **dict(exploit))
        stages.append(exploit)
    stages += [PruneRMSD(), PruneEnergy(EnergyPruner(threshold=windows[2]))]
    if rescore is not None:
        stages.append(rescore)
    return Pipeline(stages)
