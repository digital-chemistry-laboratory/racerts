"""Errors of racerts that a caller may want to tell apart."""

INCONSISTENT_RESTRAINTS = (
    "The distance restraints are inconsistent with each other or with the frozen "
    "atoms: triangle smoothing would have to move other bounds (e.g. stretch bonds)."
)


class RacerTSError(Exception):
    """
    The base of the errors below: racerts cannot do what was asked with this molecule,
    for a reason it names. A caller that goes through many molecules catches it and
    goes on. Not of this type: a call that is wrong whatever the molecule (a
    ValueError or TypeError alone), and what a calculator or RDKit raises.
    """


class NoConformersError(RacerTSError, RuntimeError):
    """
    A step left no conformer: nothing could be embedded, every optimization or
    single point failed with an expected error, none converged, or no conformer
    passed a check. A RuntimeError, as these cases were raised before; an error of a
    calculation that is not expected (ASEOptimizer expected_errors) is raised as it
    is, and is not of this type.
    """


class MoleculeError(RacerTSError, ValueError):
    """
    The molecule cannot be used as it was given: a SMILES that is not valid, a file
    that cannot be read, a geometry, a charge or a multiplicity that is not that of
    the SMILES, two endpoints that are not one reaction. A ValueError, as these cases
    were raised before.
    """


class InconsistentRestraints(RacerTSError, ValueError):
    """
    The restraints cannot be met together: windows that contradict each other, the
    distances that are fixed, or the frozen atoms.
    """

    def __init__(self, message: str = INCONSISTENT_RESTRAINTS):
        super().__init__(message)


class SwapError(RacerTSError, ValueError):
    """A swap that cannot be done as given (selector, fragment, valences, settings)."""
