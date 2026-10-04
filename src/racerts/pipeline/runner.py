"""Pipelines: stages that each take and return a ConformerEnsemble."""

import logging
import time
from typing import Iterable, Optional, Protocol, runtime_checkable

from .context import Context
from .ensemble import ConformerEnsemble

logger = logging.getLogger(__name__)


@runtime_checkable
class Stage(Protocol):
    """
    One step of a pipeline. run gets the context and the ensemble so far (None for
    the first stage) and returns the ensemble for the next stage.
    """

    name: str

    def run(
        self, ctx: Context, ensemble: Optional[ConformerEnsemble]
    ) -> ConformerEnsemble: ...


class Pipeline:
    """
    Runs its stages in order and logs the number of conformers and time of each.

    Stages may change the ensemble they get in place; an ensemble passed to run is
    copied first, so the caller's ensemble stays as it is.
    """

    def __init__(self, stages: Iterable[Stage]):
        self.stages = list(stages)

    def run(
        self, ctx: Context, ensemble: Optional[ConformerEnsemble] = None
    ) -> ConformerEnsemble:
        if ensemble is not None:
            ensemble = ensemble.copy()
        for stage in self.stages:
            start = time.perf_counter()
            ensemble = stage.run(ctx, ensemble)
            if not isinstance(ensemble, ConformerEnsemble):
                raise TypeError(
                    f"Stage {stage.name!r} returned {type(ensemble).__name__}, not the "
                    "ensemble."
                )
            logger.info(
                "%s: %d conformers (%.2f s)",
                stage.name,
                len(ensemble),
                time.perf_counter() - start,
            )
        if ensemble is None:
            raise ValueError("The pipeline has no stages.")
        return ensemble

    def __repr__(self) -> str:
        return f"Pipeline([{', '.join(stage.name for stage in self.stages)}])"
