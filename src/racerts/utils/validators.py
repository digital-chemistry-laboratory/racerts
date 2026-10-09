"""
attrs validators and fields for settings classes (racerts.config).

Every check raises ValueError naming the setting, with the section of its class as
prefix (the class attribute _section, e.g. "prune.rmsd_threshold"), so that a wrong
value in a config file points to its key. attrs runs the checks when an instance is
made and when a setting is changed.
"""

from attrs import field


def setting(instance, attribute) -> str:
    """The name of a setting in error messages, e.g. "prune.rmsd_threshold"."""
    section = getattr(instance, "_section", "")
    return f"{section}.{attribute.name}" if section else attribute.name


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
is_number = _is(float, "a number")  # after int_to_float
is_str = _is(str, "a string")


def non_negative(instance, attribute, value) -> None:
    if value < 0:
        raise ValueError(f"{setting(instance, attribute)} must not be negative.")


def positive(instance, attribute, value) -> None:
    if value <= 0:
        raise ValueError(f"{setting(instance, attribute)} must be positive.")


def positive_or(default, meaning: str):
    """Positive, or default (e.g. -1 for "default count")."""

    def check(instance, attribute, value):
        if value != default and value <= 0:
            raise ValueError(
                f"{setting(instance, attribute)} must be {default} ({meaning}) or > 0."
            )

    return check


def one_of(choices):
    def check(instance, attribute, value):
        if value not in choices:
            raise ValueError(
                f"{setting(instance, attribute)} must be one of {sorted(choices)}."
            )

    return check


def int_to_float(value):
    """Converter: ints to floats, other values (bools too) unchanged for the checks."""
    return float(value) if type(value) is int else value


# Settings with their checks; the type is checked first, so the others can compare.


def flag(default: bool):
    return field(default=default, validator=is_bool)


def integer(default: int, *checks):
    return field(default=default, validator=[is_int, *checks])


def number(default: float, *checks):
    return field(
        default=default, converter=int_to_float, validator=[is_number, *checks]
    )


def choice(default: str, choices):
    return field(default=default, validator=[is_str, one_of(choices)])
