"""
attrs fields and validators for settings classes (racerts.config).

Every check raises ValueError naming the setting, with the section of its class as
prefix (the class attribute _section, e.g. "prune.rmsd_threshold"), so that a wrong
value in a config file points to its key. attrs runs the checks when an instance is
made and when a setting is changed.
"""

import numpy as np
from attrs import field

from racerts.utils.checks import is_integer


def setting(instance, attribute) -> str:
    """The name of a setting in error messages, e.g. "prune.rmsd_threshold"."""
    section = getattr(instance, "_section", "")
    return f"{section}.{attribute.name}" if section else attribute.name


# Converters: NumPy numbers (e.g. a seed from numpy.random) become Python numbers of
# their kind, which the config files can hold; other values are left to the checks.


def to_bool(value):
    return bool(value) if isinstance(value, np.bool_) else value


def to_int(value):
    return int(value) if is_integer(value) else value


def to_float(value):
    if is_integer(value) or isinstance(value, (float, np.floating)):
        return float(value)
    return value


def _is(kind, what: str):
    def check(instance, attribute, value):
        # True and False are ints: no bools as numbers, and no 0 or 1 as flags.
        if isinstance(value, bool) != (kind is bool) or not isinstance(value, kind):
            raise ValueError(
                f"{setting(instance, attribute)} must be {what}, not {value!r}."
            )

    return check


is_bool = _is(bool, "true or false")
is_int = _is(int, "an integer")
is_number = _is(float, "a number")  # after to_float
is_str = _is(str, "a string")


def must(test, requirement: str):
    """A check: test(value), or "<setting> must <requirement>."."""

    def check(instance, attribute, value):
        if not test(value):
            raise ValueError(f"{setting(instance, attribute)} must {requirement}.")

    return check


non_negative = must(lambda value: value >= 0, "not be negative")
positive = must(lambda value: value > 0, "be positive")


def one_of(choices):
    shown = sorted(choices) if isinstance(choices, (dict, set)) else list(choices)
    return must(lambda value: value in choices, f"be one of {shown}")


def current(instance, attribute, value, *names) -> list:
    """
    The values of the settings names, with value for the one being checked: on a
    change, attrs checks the new value before it is set.
    """
    return [
        value if name == attribute.name else getattr(instance, name) for name in names
    ]


# Settings with their checks; the type is checked first, so the others can compare.


def flag(default: bool, *checks):
    return field(default=default, converter=to_bool, validator=[is_bool, *checks])


def integer(default: int, *checks):
    return field(default=default, converter=to_int, validator=[is_int, *checks])


def number(default: float, *checks):
    return field(default=default, converter=to_float, validator=[is_number, *checks])


def choice(default: str, choices, *checks):
    return field(default=default, validator=[is_str, one_of(choices), *checks])
