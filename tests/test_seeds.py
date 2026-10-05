"""
Seeds and random draws: racerts.utils.seeds. One function gives the seed of every use of
the user's seed, one class the random numbers that racerts draws itself.
"""

import copy

import numpy as np
import pytest

import racerts
from racerts.embed.stage import _batch_embedder
from racerts.utils import seeds


def test_a_derived_seed_is_a_fixed_function_of_the_seed_and_its_use():
    # A hash: no library version changes it. These values are fixed for good.
    assert seeds.derive(12) == 1051840539
    assert seeds.derive(12, "batch", 1) == 192700149
    assert seeds.derive(np.int64(12), "batch", np.int64(1)) == 192700149
    assert seeds.derive(12, "batch", 1) != seeds.derive(12, "batch", 2)
    assert seeds.derive(12, "batch", 1) != seeds.derive(12, "other", 1)
    # An RDKit seed with room for the seeds of the conformers after it.
    for seed in (0, 1, 12, 2**31, 10**12):
        assert 0 <= seeds.derive(seed, "x") < seeds.SEED_RANGE - seeds.STREAM_LENGTH
    # A negative seed asks for no reproducibility (RDKit's -1): it stays what it is.
    assert seeds.derive(-1) == -1 and seeds.derive(-1, "batch", 3) == -1
    with pytest.raises(TypeError, match="integer"):
        seeds.derive(1.5)


def test_the_streams_of_different_seeds_and_uses_do_not_fall_together():
    # seed + 7919 k made batch 1 of seed 12 the batch 0 of seed 7931.
    assert seeds.derive(12, "batch", 1) != seeds.derive(7931, "batch", 0)
    assert seeds.derive(12, "batch", 1) != seeds.derive(7931)
    # 100 seeds with 10 batches each: 1000 streams, none within 10000 conformers of
    # another.
    values = sorted(seeds.derive(s, "batch", k) for s in range(100) for k in range(10))
    assert min(b - a for a, b in zip(values, values[1:])) > 10000


def test_a_stream_draws_the_same_numbers_for_the_same_seed_and_use():
    first, again = seeds.Stream(12, "exploit"), seeds.Stream(12, "exploit")
    drawn = [first.random() for _ in range(3)]
    # Python's random.Random.random(), whose sequence is fixed across versions.
    assert drawn == [0.7014333017714215, 0.3059118228906078, 0.19888917168516196]
    assert [again.random() for _ in range(3)] == drawn
    assert seeds.Stream(12, "other").random() != drawn[0]
    assert seeds.Stream(13, "exploit").random() != drawn[0]
    # A negative seed: not reproducible.
    assert seeds.Stream(-1, "exploit").random() != seeds.Stream(-1, "exploit").random()


def test_the_draws_of_a_stream():
    stream = seeds.Stream(3, "test")
    assert all(0.0 <= stream.random() < 1.0 for _ in range(200))
    assert all(-2.0 <= stream.uniform(-2.0, 5.0) < 5.0 for _ in range(200))
    assert all(0.0 <= stream.uniform() < 1.0 for _ in range(20))
    assert {stream.integers(4) for _ in range(200)} == {0, 1, 2, 3}
    assert {stream.integers(2, 5) for _ in range(200)} == {2, 3, 4}
    assert all(isinstance(stream.integers(3), int) for _ in range(5))
    assert {stream.choice("abc") for _ in range(100)} == {"a", "b", "c"}
    with pytest.raises(ValueError, match="empty"):
        stream.integers(0)
    # Standard normal numbers, as an array of the size asked for.
    values = stream.normal(size=4000)
    assert values.shape == (4000,) and isinstance(stream.normal(), float)
    assert abs(values.mean()) < 0.05 and abs(values.std() - 1.0) < 0.05
    assert np.allclose(
        seeds.Stream(3, "n").normal(size=3), seeds.Stream(3, "n").normal(size=3)
    )


def test_a_batch_embeds_with_the_derived_seed_of_its_number():
    embedder = racerts.embed.CmapEmbedder(randomSeed=12)
    assert _batch_embedder(embedder, 0) is embedder
    for k in (1, 2, 7):
        batch = _batch_embedder(copy.copy(embedder), k)
        assert batch.randomSeed == seeds.derive(12, "batch", k)
    unseeded = racerts.embed.CmapEmbedder(randomSeed=-1)
    assert _batch_embedder(unseeded, 3).randomSeed == -1
