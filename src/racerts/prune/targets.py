"""The targets of a windowed TS: conformers compare only within one."""

from typing import Callable, List, Sequence

from racerts.pipeline import ConformerEnsemble

LENGTH_BINS = 5  # unstratified active-bond windows are pruned per fifth of the window


def window_bins(ctx, ensemble) -> dict:
    """
    The window bin of every conformer of a windowed TS, whose energies and geometries
    are comparable only within it: its targets (stratified) or the fifths of the
    windows that its targets fall in (uniform). () for every conformer otherwise, and
    for conformers whose targets a free refinement released.
    """
    if ctx is None or not getattr(ctx.task, "windowed", False):
        return {conf_id: () for conf_id in ensemble.conf_ids}
    windows = {f"{a}-{b}": w for (a, b), w in ctx.task.active_windows(ctx.mol).items()}
    stratified = getattr(ctx.task, "stratify", 0)
    bins = {}
    for conf_id in ensemble.conf_ids:
        targets = ensemble.provenance(conf_id).get("active_bond_targets") or {}
        key = []
        for bond, t in sorted(targets.items()):
            if stratified or bond not in windows:
                key.append((bond, t))
            else:
                lo, hi = windows[bond]
                key.append(
                    (
                        bond,
                        min(int((t - lo) / (hi - lo) * LENGTH_BINS), LENGTH_BINS - 1),
                    )
                )
        bins[conf_id] = tuple(key)
    return bins


def target_groups(ctx, ensemble) -> List[List[int]]:
    """
    The conformer ids per target of the active bonds of a windowed TS: per target
    (stratified) or per fifth of the window (unstratified), in the order of the
    targets; else one group.
    """
    groups = {}
    for conf_id, key in window_bins(ctx, ensemble).items():
        groups.setdefault(key, []).append(conf_id)
    return [groups[key] for key in sorted(groups)] or [ensemble.conf_ids]


def round_robin(groups: Sequence[Sequence]) -> list:
    """
    The members of the groups in turns: the first of each group, then the second of
    each, ... (e.g. of the clusters of ClusterPruner.clusters, for a selection that
    is spread over them).
    """
    ranks = range(max(map(len, groups), default=0))
    return [group[rank] for rank in ranks for group in groups if rank < len(group)]


def in_turns(
    ctx, ensemble: ConformerEnsemble, ranked: Callable[[ConformerEnsemble], List[int]]
) -> List[int]:
    """
    All conformer ids, the targets of a windowed TS taking turns: the first of each
    target, then the second, ... ranked(part) orders the conformers of one target (an
    ensemble), best first. Without targets there is one group: ranked(ensemble).
    """
    groups = target_groups(ctx, ensemble)
    parts = [ensemble] if len(groups) == 1 else [ensemble.filter(g) for g in groups]
    return round_robin([ranked(part) for part in parts])
