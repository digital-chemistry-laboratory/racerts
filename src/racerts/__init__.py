"""racerts: conformer ensembles with a frozen core, for transition states and more.

    import racerts

    ensemble = racerts.generate_ts("ts.xyz", [3, 4, 5], smiles="CCCCCC=C")
    ensemble.write_xyz("conformers.xyz")

The legacy racerts API (ConformerGenerator, the component registries and the legacy
module paths such as racerts.embedder or racerts.optimizer.ase) is kept in
racerts.compat and importable from here.
"""

from .api import generate, generate_gs, generate_ts, swap
from .compat import (
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
from .config import (
    EmbedConfig,
    PipelineConfig,
    PruneConfig,
    RefineConfig,
    RestraintConfig,
)
from .embed import Embed
from .exploit import Exploit
from .pipeline import ConformerEnsemble, ConformerRecord, Context, Pipeline, Stage
from .prune import PruneCluster, PruneCount, PruneEnergy, PruneRMSD
from .refine import Refine, Rescore
from .system.swap import Swap, SwapError, SwapResult, apply_swap
from .task import Constrained, FrozenSet, GroundState, Task, TransitionState
from .validate import Validate

__all__ = [
    "ConformerEnsemble",
    "ConformerRecord",
    "Constrained",
    "Context",
    "Embed",
    "EmbedConfig",
    "Exploit",
    "FrozenSet",
    "GroundState",
    "Pipeline",
    "PipelineConfig",
    "PruneCluster",
    "PruneConfig",
    "PruneCount",
    "PruneEnergy",
    "PruneRMSD",
    "Refine",
    "RefineConfig",
    "Rescore",
    "RestraintConfig",
    "Stage",
    "Swap",
    "SwapError",
    "SwapResult",
    "Task",
    "TransitionState",
    "Validate",
    "apply_swap",
    "generate",
    "generate_gs",
    "generate_ts",
    "swap",
    "ConformerGenerator",
    "embedders",
    "mol_getters",
    "optimizers",
    "pruners",
    "conformer_generator",
    "embedder",
    "mol_getter",
    "optimizer",
    "pruner",
    "visualizer",
]
