"""Geometric validators: the frozen core."""

from typing import Dict

import numpy as np

from racerts.pipeline import ConformerEnsemble
from racerts.prune.rmsd import superpose


class FrozenCore:
    """
    Whether the frozen (hard) atoms of the task are still at the reference geometry:
    after the best superposition on them, no frozen atom may be further than
    tolerance (A) from its reference position.
    """

    name = "frozen_core"

    def __init__(self, tolerance: float = 1e-3):
        self.tolerance = tolerance

    def validate(self, ctx, ensemble: ConformerEnsemble) -> Dict[int, str]:
        atoms = list(ctx.frozen.hard)
        if not atoms or ctx.reference is None:
            return {}
        reasons = {}
        for reference, conf_ids in ctx.by_reference(ensemble):
            target = reference.GetConformer().GetPositions()[atoms]
            for conf_id in conf_ids:
                positions = ensemble.mol.GetConformer(conf_id).GetPositions()[atoms]
                moved = _max_deviation(positions, target)
                if moved > self.tolerance:
                    reasons[conf_id] = f"frozen atoms moved by up to {moved:.3f} A"
        return reasons


def _max_deviation(positions: np.ndarray, reference: np.ndarray) -> float:
    """Largest atom deviation after the best superposition (Kabsch, rotations only)."""
    q = reference - reference.mean(axis=0)
    return float(np.linalg.norm(superpose(positions, reference) - q, axis=1).max())
