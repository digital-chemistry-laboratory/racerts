"""Checks of argument values."""

from numbers import Integral


def is_integer(value) -> bool:
    """Whether value is an integer (also a NumPy one); True and False are not."""
    return isinstance(value, Integral) and not isinstance(value, bool)


def check_positive(**settings) -> None:
    """Raise ValueError for the first of the settings that is not positive."""
    for name, value in settings.items():
        if not value > 0:
            raise ValueError(f"{name} must be positive, not {value!r}.")


def check_count(name: str, value, least: int = 1) -> int:
    """value as an int; raises ValueError unless it is an integer of at least least."""
    if not is_integer(value) or value < least:
        raise ValueError(
            f"{name} must be an integer of at least {least}, not {value!r}."
        )
    return int(value)
