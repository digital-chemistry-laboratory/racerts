"""generate conformer ensembles for a task."""

from typing import Optional, Sequence, Union

from attrs import evolve
from rdkit import Chem

from racerts.config import PipelineConfig
from racerts.pipeline import ConformerEnsemble, Context, Pipeline
from racerts.system.build import BaseMolGetter, build_mol
from racerts.task import GroundState, Task, TransitionState
from racerts.utils.log import verbose_logging


def generate(
    mol: Chem.Mol,
    task: Task,
    config: Optional[PipelineConfig] = None,
    pipeline: Optional[Pipeline] = None,
    charge: Optional[int] = None,
    multiplicity: Optional[int] = None,
    verbose: bool = False,
) -> ConformerEnsemble:
    """
    Generate a conformer ensemble of mol for the task.

    Args:
        mol: The molecular graph with all hydrogens; its conformer (if any) is the
            reference geometry for the frozen atoms of the task.
        task: TransitionState, GroundState, Constrained or any Task.
        config: Settings of the default pipeline (default PipelineConfig()).
        pipeline: A custom pipeline instead of the default one. Of config, only the
            seed is then used, by stages that create their own components (e.g.
            Embed() without an embedder); components passed to stages keep their
            own settings.
        charge, multiplicity: Override the values of mol (see
            set_charge_and_multiplicity).
        verbose: Log progress (INFO) during the call.
    """
    config = config if config is not None else PipelineConfig()
    with verbose_logging(verbose):
        ctx = Context.create(
            mol, task, seed=config.seed, charge=charge, multiplicity=multiplicity
        )
        pipeline = pipeline if pipeline is not None else config.build(task)
        return pipeline.run(ctx)


def generate_ts(
    file_name: str,
    reacting_atoms: Sequence[int],
    charge: int = 0,
    smiles: Union[str, Sequence[str], None] = None,
    multiplicity: Optional[int] = None,
    frozen_atoms: Optional[Sequence[int]] = None,
    config: Optional[PipelineConfig] = None,
    pipeline: Optional[Pipeline] = None,
    mol_getter: Optional[BaseMolGetter] = None,
    auto_fallback: bool = True,
    verbose: bool = False,
) -> ConformerEnsemble:
    """
    A TS conformer ensemble from a TS geometry (xyz or sdf/mol file), with the reacting
    atoms and their neighbours kept at the TS geometry. With default settings the
    result equals ConformerGenerator().generate_conformers (legacy racerts).

    Args:
        file_name: The TS geometry.
        reacting_atoms: 0-based indices of the atoms whose bonds form or break.
        charge: Total charge.
        smiles: SMILES of the TS topology (list of fragments), recommended for xyz
            files; without, the bonds are perceived from the geometry.
        multiplicity: Spin multiplicity (default: the lowest for the electrons).
        frozen_atoms: Frozen atoms instead of the reacting atoms and neighbours.
        config, pipeline: See generate.
        mol_getter: The first method to build the graph (see build_mol).
        auto_fallback: As in legacy racerts: fall back to perceived bonds or
            connectivity if the graph cannot be built from the SMILES, and from MMFF
            to UFF (config.refine.fallback) in the default pipeline.
        verbose: Log progress (INFO) during the call.
    """
    if isinstance(smiles, str):
        smiles = [smiles]
    config = config if config is not None else PipelineConfig()
    if not auto_fallback:
        config = evolve(config, refine=evolve(config.refine, fallback=False))
    with verbose_logging(verbose):
        mol = build_mol(
            file_name,
            charge,
            list(reacting_atoms),
            input_smiles=list(smiles) if smiles else None,
            mol_getter=mol_getter,
            auto_fallback=auto_fallback,
        )
    if mol is None:
        raise ValueError(f"No valid mol object could be generated from {file_name}.")
    return generate(
        mol,
        TransitionState(reacting_atoms, frozen_atoms),
        config=config,
        pipeline=pipeline,
        charge=charge,
        multiplicity=multiplicity,
        verbose=verbose,
    )


def generate_gs(
    smiles: Union[str, Chem.Mol],
    charge: Optional[int] = None,
    multiplicity: Optional[int] = None,
    config: Optional[PipelineConfig] = None,
    pipeline: Optional[Pipeline] = None,
    verbose: bool = False,
) -> ConformerEnsemble:
    """
    A ground-state conformer ensemble (nothing frozen) from a SMILES or a Mol.
    Hydrogens are added; by default embedding uses ETKDGv3, and the stereocentres of
    the input are kept.
    """
    if not isinstance(smiles, (str, Chem.Mol)):
        raise TypeError(
            f"Expected SMILES string or RDKit Mol, got {type(smiles).__name__}: {smiles!r}"
        )
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else smiles
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol, addCoords=mol.GetNumConformers() > 0)  # a new Mol
    return generate(
        mol,
        GroundState(),
        config=config,
        pipeline=pipeline,
        charge=charge,
        multiplicity=multiplicity,
        verbose=verbose,
    )
