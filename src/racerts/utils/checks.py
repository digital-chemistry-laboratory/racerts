"""Checks of argument values."""

from numbers import Integral


def is_integer(value) -> bool:
    """Whether value is an integer (also a NumPy one); True and False are not."""
    return isinstance(value, Integral) and not isinstance(value, bool)
