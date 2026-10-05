"""
Seeds and random draws. Everything random in racerts starts from the user's seed and
passes through this module: derive gives the seed of one use (a batch, a stage), Stream
the random numbers that racerts draws itself. Both are fixed by SHA-256 and by the
sequence of Python's random.Random.random(), which no version of Python, NumPy or RDKit
changes: a seed names the same seeds and draws everywhere.
"""

import hashlib
import math
import random
from typing import Optional, Sequence

import numpy as np

from racerts.utils.checks import is_integer

# RDKit seeds are 31-bit integers; a derived seed leaves room for this many seeds after
# it (the conformers of one embedding take consecutive seeds).
SEED_RANGE = 2**31 - 1
STREAM_LENGTH = 2**24


def derive(seed: int, *use) -> int:
    """
    The seed of one use of the user's seed. use names it with words and numbers (e.g.
    "batch", 3); without it the result is the first seed of the conformers of seed
    itself. A hash of the seed and the use (SHA-256), spread over the seed range: the
    streams of neighbouring seeds, and of different uses of one seed, are unrelated, and
    two of them overlap only by chance. A negative seed asks for no reproducibility
    (RDKit's -1) and is returned as it is.
    """
    if not is_integer(seed):
        raise TypeError(f"A seed must be an integer, not {seed!r}.")
    seed = int(seed)
    if seed < 0:
        return seed
    text = "/".join(str(part) for part in (seed, *use))
    digest = hashlib.sha256(text.encode()).digest()
    return int.from_bytes(digest[:8], "big") % (SEED_RANGE - STREAM_LENGTH)


class Stream:
    """
    The random numbers of one use of the user's seed (see derive): the same draws for
    the same seed and use, unseeded for a negative seed. Every draw is built from
    random.Random.random() alone.
    """

    def __init__(self, seed: int, *use):
        derived = derive(seed, *use)
        self._random = random.Random(None if derived < 0 else derived).random

    def random(self) -> float:
        """A number in [0, 1)."""
        return self._random()

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        """A number in [low, high)."""
        return low + (high - low) * self._random()

    def integers(self, low: int, high: Optional[int] = None) -> int:
        """An integer in [0, low), or in [low, high) with both given."""
        if high is None:
            low, high = 0, low
        if high <= low:
            raise ValueError(f"The range of integers [{low}, {high}) is empty.")
        return low + int(self._random() * (high - low))

    def choice(self, items: Sequence):
        """One of the items."""
        return items[self.integers(len(items))]

    def normal(self, size: Optional[int] = None):
        """A standard normal number, or an array of size of them (Box-Muller)."""
        values = [
            math.sqrt(-2.0 * math.log(1.0 - self._random()))
            * math.cos(2.0 * math.pi * self._random())
            for _ in range(1 if size is None else size)
        ]
        return values[0] if size is None else np.array(values)
