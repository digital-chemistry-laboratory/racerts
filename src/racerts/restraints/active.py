"""Active bonds of windowed TSs: target windows and the recorded lengths."""

from .model import DistanceRestraint


def target_windows(restraints, target, half_width, force_constant=None):
    """The restraints with the active bonds held at their targets +/- half_width."""
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
    Uniform window mode: each conformer's embedded active-bond lengths become its
    targets, which refinement holds (a flat-bottom window would let the force field
    push every conformer to the same edge of the window).
    """
    if not getattr(ctx.task, "windowed", False) or getattr(ctx.task, "stratify", 0):
        return
    windows = {f"{a}-{b}": w for (a, b), w in ctx.task.active_windows(ctx.mol).items()}
    for conf_id in ensemble.conf_ids:
        lengths = ensemble.provenance(conf_id)["active_bond_lengths"]
        targets = {
            pair: round(min(max(d, windows[pair][0]), windows[pair][1]), 4)
            for pair, d in lengths.items()
        }
        ensemble.add_provenance(conf_id, active_bond_targets=targets)
