"""Active bonds of windowed TSs: target windows and the recorded lengths."""

import logging

from .model import DistanceRestraint

logger = logging.getLogger(__name__)

OUTSIDE_TOLERANCE = 0.05  # A: distance geometry meets a window to about this
# Half widths (A) of the windows that hold an active bond at its target.
EMBED_TARGET_HALF_WIDTH = 0.01
REFINE_TARGET_HALF_WIDTH = 0.02
# The sources of the restraints that place the reacting atoms of a windowed TS.
WINDOW_SOURCES = ("active", "neighbor", "core", "target")


def target_windows(restraints, target, half_width, force_constant=None):
    """The distance restraints with the active bonds held at their targets +/-
    half_width."""
    held = []
    for r in restraints:
        if r.pair in target:
            t = target[r.pair]
            held.append(
                DistanceRestraint(
                    *r.pair,
                    t - half_width,
                    t + half_width,
                    force_constant=force_constant or r.force_constant,
                    source="target",
                )
            )
        else:
            held.append(r)
    return held


def target_provenance(target) -> dict:
    return {
        "active_bond_targets": {f"{a}-{b}": round(t, 4) for (a, b), t in target.items()}
    }


def release_targets(ensemble) -> None:
    """
    After a free refinement of a windowed TS: the conformers are no longer held at
    their targets, so they compare as one group (two targets can lead to the same
    saddle point). The provenance keeps the targets as "released_targets".
    """
    for conf_id in ensemble.conf_ids:
        targets = ensemble.provenance(conf_id).get("active_bond_targets")
        if targets:
            ensemble.add_provenance(
                conf_id, active_bond_targets=None, released_targets=targets
            )


def record_active_lengths(ctx, ensemble) -> None:
    """The lengths of the active bonds of a windowed TS in each provenance."""
    if not getattr(ctx.task, "windowed", False):
        return
    pairs = ctx.task.active_pairs(ctx.mol)  # of the reference
    for conf_id in ensemble.conf_ids:
        lengths = ctx.task.active_lengths(ensemble.mol, conf_id, pairs)
        ensemble.add_provenance(conf_id, active_bond_lengths=lengths)


def keep_embedded_lengths(ctx, ensemble) -> None:
    """
    Unstratified window mode: each conformer's embedded active-bond lengths become
    its targets, which refinement holds (a flat-bottom window would let the force
    field push every conformer to the same edge of the window). A length outside its
    window is moved to the edge; conformers embedded more than OUTSIDE_TOLERANCE
    outside are reported, and if all are, the frozen atoms cannot take the window.

    Raises:
        ValueError: If no conformer was embedded with its active bonds in the window.
    """
    if not getattr(ctx.task, "windowed", False) or getattr(ctx.task, "stratify", 0):
        return
    windows = {f"{a}-{b}": w for (a, b), w in ctx.task.active_windows(ctx.mol).items()}
    outside = {}
    for conf_id in ensemble.conf_ids:
        lengths = ensemble.provenance(conf_id)["active_bond_lengths"]
        targets = {
            pair: round(min(max(d, windows[pair][0]), windows[pair][1]), 4)
            for pair, d in lengths.items()
        }
        ensemble.add_provenance(conf_id, active_bond_targets=targets)
        off = {
            p: d for p, d in lengths.items() if abs(d - targets[p]) > OUTSIDE_TOLERANCE
        }
        if off:
            outside[conf_id] = off
    if not outside:
        return
    pair, length = next(iter(next(iter(outside.values())).items()))
    example = (
        f"bond {pair} at {length:.2f} A for the window "
        f"{windows[pair][0]:.2f}-{windows[pair][1]:.2f} A"
    )
    if len(outside) == len(ensemble):
        raise ValueError(
            f"No conformer was embedded with its active bonds in the window (e.g. "
            f"{example}): the frozen atoms cannot take it; try a narrower active_window."
        )
    logger.warning(
        "%d of %d conformers were embedded with an active bond more than %.2f A "
        "outside its window (e.g. %s); refinement holds them at the edge of the window.",
        len(outside),
        len(ensemble),
        OUTSIDE_TOLERANCE,
        example,
    )
