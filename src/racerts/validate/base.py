"""The Validator protocol and the Validate stage."""

import logging
from collections import Counter
from typing import Callable, Dict, Optional, Protocol, Union, runtime_checkable

from rdkit import Chem

from racerts.errors import NoConformersError
from racerts.pipeline import ConformerEnsemble

logger = logging.getLogger(__name__)

ON_FAIL = ("drop", "flag")


@runtime_checkable
class Validator(Protocol):
    """
    Checks conformers. validate returns the reason for every conformer that fails;
    conformers that are not in the dict pass. It gets the whole ensemble, so that a
    validator can batch its work (e.g. on a GPU).
    """

    name: str

    def validate(self, ctx, ensemble: ConformerEnsemble) -> Dict[int, str]: ...


class FunctionValidator:
    """A Validator from a function check(mol, conf_id) -> reason, or None if it passes."""

    def __init__(self, check: Callable[[Chem.Mol, int], Optional[str]], name: str):
        self.check = check
        self.name = name

    def validate(self, ctx, ensemble: ConformerEnsemble) -> Dict[int, str]:
        reasons = {}
        for conf_id in ensemble.conf_ids:
            reason = self.check(ensemble.mol, conf_id)
            if reason is False:
                reason = "failed"
            if reason not in (None, True):
                reasons[conf_id] = str(reason)
        return reasons

    def __repr__(self) -> str:
        return f"validator({self.name})"


def validator(
    check: Callable[[Chem.Mol, int], Union[Optional[str], bool]],
    name: Optional[str] = None,
) -> FunctionValidator:
    """
    A Validator from a per-conformer function check(mol, conf_id) that returns None
    (or True) if the conformer passes, and the reason (or False) if it fails.
    """
    return FunctionValidator(check, name or getattr(check, "__name__", "validator"))


class Validate:
    """
    Runs validators on every conformer and records the result in its provenance
    ("validation": {validator name: "ok" or the reason}).

    Args:
        validators: Validator objects (see Validator; e.g. Connectivity, FrozenCore,
            ImaginaryModes, or validator(function)).
        on_fail: "drop" removes the conformers that fail any validator; "flag" keeps
            them with the reasons in their provenance.
        require_any: With "drop", raise NoConformersError if no conformer passes.
            Flagging removes nothing: it then logs a warning and returns the ensemble.
        warn_above: The share of failing conformers above which a warning is logged
            (default 0: any failure); below it, the failures are logged as information.
            The log counts the failures per validator.
    """

    name = "validate"

    def __init__(
        self,
        *validators: Validator,
        on_fail: str = "drop",
        require_any=True,
        warn_above: float = 0.0,
    ):
        if not validators:
            raise ValueError("Validate needs at least one validator.")
        for item in validators:
            if isinstance(item, type) or not isinstance(item, Validator):
                raise TypeError(
                    f"{item!r} is not a Validator (an object with a name and "
                    "validate(ctx, ensemble)); wrap functions with validator()."
                )
        names = [item.name for item in validators]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:  # the provenance records the results by name
            raise ValueError(
                f"Repeated validator names {repeated}: give each validator of a "
                "stage its own name."
            )
        if on_fail not in ON_FAIL:
            raise ValueError(f"on_fail must be one of {ON_FAIL}.")
        if not 0 <= warn_above <= 1:
            raise ValueError("warn_above must be a share between 0 and 1.")
        self.validators = validators
        self.on_fail = on_fail
        self.require_any = require_any
        self.warn_above = warn_above

    def run(self, ctx, ensemble: ConformerEnsemble) -> ConformerEnsemble:
        failed: Dict[int, Dict[str, str]] = {}
        results = {conf_id: {} for conf_id in ensemble.conf_ids}
        for item in self.validators:
            reasons = item.validate(ctx, ensemble)
            for conf_id in results:
                reason = reasons.get(conf_id)
                results[conf_id][item.name] = "ok" if reason is None else reason
                if reason is not None:
                    failed.setdefault(conf_id, {})[item.name] = reason
        for conf_id, result in results.items():
            previous = ensemble.provenance(conf_id).get("validation", {})
            ensemble.add_provenance(conf_id, validation={**previous, **result})

        if failed and len(failed) == len(results):
            conf_id, reasons = next(iter(failed.items()))
            none = (
                f"No conformer passed validation (e.g. conformer {conf_id}: "
                f"{_describe(reasons)})"
            )
            if self.on_fail == "flag":
                logger.warning(
                    "%s; all %d are kept with their reasons.", none, len(results)
                )
                return ensemble
            if self.require_any:
                raise NoConformersError(none + ".")
        if failed:
            counts = Counter(name for reasons in failed.values() for name in reasons)
            share = len(failed) / len(results)
            level = logging.WARNING if share > self.warn_above else logging.INFO
            hint = ""
            if self.warn_above > 0 and level == logging.WARNING:
                hint = (
                    f"; more than {self.warn_above:.0%}: check the charge, the "
                    "restraints or the hypotheses"
                )
            logger.log(
                level,
                "%d of %d conformers failed validation%s (%s)%s; e.g. conformer %d: %s",
                len(failed),
                len(results),
                " and are removed" if self.on_fail == "drop" else "",
                ", ".join(f"{name}: {count}" for name, count in counts.items()),
                hint,
                next(iter(failed)),
                _describe(next(iter(failed.values()))),
            )
            if self.on_fail == "drop":
                for conf_id in failed:
                    ensemble.mol.RemoveConformer(conf_id)
        return ensemble

    def __repr__(self) -> str:
        names = ", ".join(item.name for item in self.validators)
        return f"Validate({names}, on_fail={self.on_fail!r})"


def _describe(reasons: Dict[str, str]) -> str:
    return "; ".join(f"{name}: {reason}" for name, reason in reasons.items())
