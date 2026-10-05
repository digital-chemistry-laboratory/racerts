"""
The staged workflow: levels of refinement from cheap to expensive, each with its gate,
energy window and RMSD pruning, and optionally a search (Exploit).
"""

from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional, Union

from racerts.config import PipelineConfig
from racerts.embed import Embed
from racerts.exploit import Exploit
from racerts.pipeline import Pipeline
from racerts.prune import FamilySelector, PruneCount, PruneEnergy, PruneRMSD
from racerts.prune.energy import EnergyPruner
from racerts.refine import Refine, Rescore
from racerts.validate import OVERLAP_FACTOR, AttackFace, Clash, Validate, gate

CHEAP_WINDOW = 25.0  # kcal/mol: of the force-field level, which should not decide
WINDOW = 8.0  # kcal/mol: of a level whose energies rank
FINAL_WINDOW = 6.0  # kcal/mol: of the result
POOLS = ("family", "energy")


@dataclass(frozen=True)
class Level:
    """
    One level of the staged workflow: a refinement, the validity gate
    (racerts.validate.gate), the energy window and the RMSD pruning.

    Attributes:
        optimizer: The refinement: an optimizer or a Refine stage. None: the force
            field of the settings (the config of staged; MMFF, not converged).
        window: The energy window after the refinement (kcal/mol).
        exploit: Settings of an Exploit search at this level (a dict; {} for the
            defaults), or an Exploit stage; None: no search. Exploit needs an
            ASEOptimizer. The search takes the refinement, the ranking energy and
            the energy window of the level, unless its settings say otherwise.
        rank: A Rescore stage for the energy that the window, the RMSD pruning and
            Exploit use, if not the energy of the refinement: e.g. that energy plus a
            solvation correction (Rescore(batch=..., add=True)).
        pool: The most conformers that enter the level (default: all), which limits
            its cost.
        pool_by: How the pool is chosen: "family" (spread over structural families;
            FamilySelector) or "energy" (the lowest of the level before; PruneCount).
    """

    optimizer: Any = None
    window: float = WINDOW
    exploit: Union[None, Mapping, Exploit] = None
    rank: Optional[Rescore] = None
    pool: Optional[int] = None
    pool_by: str = "family"

    def __post_init__(self):
        if not self.window > 0:
            raise ValueError(f"window must be a positive energy, not {self.window}.")
        if self.pool_by not in POOLS:
            raise ValueError(f"pool_by must be one of {POOLS}, not {self.pool_by!r}.")

    def stages(self, config: PipelineConfig, faces: bool = False) -> list:
        """The stages of the level; faces: with the forming-bond stereo filter of
        windowed TSs after the gate (the first level)."""
        if self.optimizer is None:
            refine = config.refine_stage()
        elif isinstance(self.optimizer, Refine):
            refine = self.optimizer
        else:
            refine = Refine(self.optimizer)
        stages = []
        if self.pool is not None:
            by_family = self.pool_by == "family"
            stages.append(
                FamilySelector(self.pool) if by_family else PruneCount(self.pool)
            )
        stages += [refine, gate()]
        if faces:
            stages.append(_WindowedTS(Validate(AttackFace())))
        if self.rank is not None:
            stages.append(self.rank)
        stages += [PruneEnergy(EnergyPruner(threshold=self.window)), PruneRMSD()]
        if self.exploit is not None:
            exploit = self.exploit
            if not isinstance(exploit, Exploit):
                settings = {"rank": self.rank, "energy_window": self.window}
                exploit = Exploit(refine, **{**settings, **dict(exploit)})
            stages += [exploit, PruneRMSD()]
        return stages


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


class _DefaultEmbed:
    """The Embed stage of the settings, for the task and the seed of the context."""

    name = "embed"

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, ctx, ensemble=None):
        config = replace(self.config, seed=ctx.seed)
        return config.embed_stage(ctx.task).run(ctx, ensemble)


def staged(
    levels,
    embed: Optional[Embed] = None,
    config: Optional[PipelineConfig] = None,
    clash_filter: Optional[float] = OVERLAP_FACTOR,
    final_window: float = FINAL_WINDOW,
    rescore: Optional[Rescore] = None,
) -> Pipeline:
    """
    The staged workflow as a pipeline:

    1. embed (one Embed stage: its batches mix biased and unbiased settings), then drop
       conformers whose heavy atoms overlap;
    2. the levels in their order, from cheap to expensive. Each (Level): with pool, at
       most that many conformers; the refinement; the validity gate
       (racerts.validate.gate); after the first level the forming-bond stereo filter
       of windowed TSs; the ranking energy if it is another one (rank); the energy
       window and the RMSD pruning; with exploit, Exploit with the same refinement and
       ranking energy and the RMSD pruning of the whole;
    3. the final window, and optionally a Rescore stage (single points at a higher
       level or in the target solvent).

    For TSs the result goes to a saddle search with its checks:
    racerts.recipes.saddles.

    Args:
        levels: The levels in their order (Level; an optimizer or a Refine stage in
            the list stands for Level of it). A single optimizer or Refine stage
            instead of a list is the usual workflow with it,
            [Level(window=25), Level(optimizer, exploit={})]: the force field of the
            settings, which removes the artifacts of the embedding while its energies
            do not decide much, then the ranking and the search at the expensive
            level.
        embed: The Embed stage; default: that of the default pipeline (config).
        config: The settings of the default embedding and of the force-field level
            (default: PipelineConfig()); of no use if nothing takes them.
        clash_filter: Drop embedded conformers with heavy atoms closer than this times
            their vdW sum (Clash; None: no filter). The default, 0.5, is below every
            lower bound of RDKit's distance geometry for these pairs, so it drops
            overlaps only: contacts at the edge of the bounds are relaxed by refinement
            (the gate after it checks clashes at 0.7), and dropping them can lose the
            conformer that is the lowest afterwards.
        final_window: The energy window of the result (kcal/mol).
        rescore: A Rescore stage at the end, or None.
    """
    if not isinstance(levels, (list, tuple)):
        levels = [Level(window=CHEAP_WINDOW), Level(levels, exploit={})]
    levels = [lv if isinstance(lv, Level) else Level(lv) for lv in levels]
    if not levels:
        raise ValueError("staged needs at least one level.")
    uses_config = embed is None or any(level.optimizer is None for level in levels)
    if config is not None and not uses_config:
        raise ValueError("config is not used: every level and embed are given.")
    config = config if config is not None else PipelineConfig()
    if clash_filter is not None and (
        isinstance(clash_filter, bool) or not isinstance(clash_filter, (int, float))
    ):
        raise TypeError(
            f"clash_filter is a factor of the vdW sum or None, not {clash_filter!r}."
        )
    if clash_filter is not None and not 0 < clash_filter <= 1:
        raise ValueError(f"clash_filter must be in (0, 1], not {clash_filter}.")
    if not final_window > 0:
        raise ValueError("final_window must be a positive energy (kcal/mol).")
    stages = [embed if embed is not None else _DefaultEmbed(config)]
    if clash_filter is not None:
        stages.append(Validate(Clash(clash_filter), warn_above=0.3))
    for k, level in enumerate(levels):
        stages += level.stages(config, faces=k == 0)
    stages.append(PruneEnergy(EnergyPruner(threshold=final_window)))
    if rescore is not None:
        stages.append(rescore)
    return Pipeline(stages)
