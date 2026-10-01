"""Distance restraints: model, sources, embedding windows, flat-bottom refinement."""

import numpy as np
import pytest

from racerts.restraints import (
    DistanceRestraint,
    RestraintSet,
)

# ---- model ----


def test_a_restraint_is_a_window():
    r = DistanceRestraint.around(7, 2, 2.2)
    assert (r.first, r.second) == (2, 7)
    assert (r.lower, r.upper) == pytest.approx((1.95, 2.45))
    assert r.force_constant == 20.0 and r.stage == "both" and r.label == "user:2-7"
    positions = np.zeros((8, 3))
    positions[7] = [2.0, 0, 0]
    assert r.violation(positions) == 0
    positions[7] = [3.0, 0, 0]
    assert r.violation(positions) == pytest.approx(0.55)
    assert DistanceRestraint.around(0, 1, 0.1).lower == 0.0


@pytest.mark.parametrize(
    "args, error",
    [
        ((0, 0, 1.0, 2.0), ValueError),
        ((0, 1, 2.0, 1.0), ValueError),
        ((0, 1, -1.0, 1.0), ValueError),
        ((0, 1.5, 1.0, 2.0), TypeError),
        ((True, 1, 1.0, 2.0), TypeError),
    ],
)
def test_invalid_restraints_raise(args, error):
    with pytest.raises(error):
        DistanceRestraint(*args)
    with pytest.raises(ValueError, match="stage"):
        DistanceRestraint(0, 1, 1.0, 2.0, stage="late")


def test_restraint_sets():
    a = DistanceRestraint.around(0, 1, 2.0)
    b = DistanceRestraint.around(1, 0, 3.0, stage="embed", source="hbond")
    with pytest.raises(ValueError, match="Conflicting restraints for atom pair"):
        RestraintSet([a, b])
    merged = RestraintSet([a]).merge([b])
    assert list(merged) == [b]
    assert merged.for_stage("refine") == [] and merged.for_stage("embed") == [b]
    both = RestraintSet([a, DistanceRestraint.around(2, 3, 2.5, source="contact")])
    assert [r.pair for r in both.by_source("contact")] == [(2, 3)]
    assert len(both.without_pairs_within([0, 1])) == 1
    remapped = both.remap({0: 10, 1: 11, 2: 12})
    assert [r.pair for r in remapped] == [(10, 11)] and list(remapped)[
        0
    ].label == "user:10-11"
    assert RestraintSet.from_json(both.to_json()) == both
    sampled = [len(both.sample(np.random.default_rng(i), 0.5)) for i in range(200)]
    assert 0.4 < np.mean(sampled) / 2 < 0.6
