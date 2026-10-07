"""Errors of racerts that a caller may want to tell apart."""


class NoConformersError(RuntimeError):
    """
    A step left no conformer: nothing could be embedded, every optimization or
    single point failed with an expected error, none converged, or no conformer
    passed a check. A RuntimeError, as these cases were raised before; an error of a
    calculation that is not expected (ASEOptimizer expected_errors) is raised as it
    is, and is not of this type.
    """
