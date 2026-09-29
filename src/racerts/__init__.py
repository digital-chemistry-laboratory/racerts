"""racerts: conformer ensembles with a frozen core, for transition states and more.

    import racerts

    ensemble = racerts.generate_ts("ts.xyz", [3, 4, 5], smiles="CCCCCC=C")
    ensemble.write_xyz("conformers.xyz")

The legacy racerts API (ConformerGenerator, the component registries and the legacy
module paths such as racerts.embedder or racerts.optimizer.ase) is kept in
racerts.compat and importable from here.
"""

from .api import generate, generate_gs, generate_ts
from .compat import (  # noqa: F401 (the legacy modules)
    ConformerGenerator,
    conformer_generator,
    embedder,
    embedders,
    mol_getter,
    mol_getters,
    optimizer,
    optimizers,
    pruner,
    pruners,
    visualizer,
)
from .config import EmbedConfig, PipelineConfig, PruneConfig, RefineConfig
from .embed import Embed
from .pipeline import ConformerEnsemble, ConformerRecord, Context, Pipeline, Stage
from .prune import PruneEnergy, PruneRMSD
from .refine import Refine
from .task import Constrained, FrozenSet, GroundState, Task, TransitionState

__all__ = [
    "ConformerEnsemble",
    "ConformerRecord",
    "Constrained",
    "Context",
    "Embed",
    "EmbedConfig",
    "FrozenSet",
    "GroundState",
    "Pipeline",
    "PipelineConfig",
    "PruneConfig",
    "PruneEnergy",
    "PruneRMSD",
    "Refine",
    "RefineConfig",
    "Stage",
    "Task",
    "TransitionState",
    "generate",
    "generate_gs",
    "generate_ts",
    "ConformerGenerator",
    "embedders",
    "mol_getters",
    "optimizers",
    "pruners",
]
