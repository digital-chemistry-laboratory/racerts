"""generate: conformer ensembles for a task (transition states, ground states, runs)."""

import logging
from dataclasses import replace
from typing import Optional, Sequence, Union

import numpy as np
from rdkit import Chem

from racerts.config import PipelineConfig
from racerts.pipeline import ConformerEnsemble, Context, Pipeline
from racerts.pipeline.runs import Runs, compare_runs, merge_runs
from racerts.restraints import RestraintSet
from racerts.system.build import BaseMolGetter, build_mol
from racerts.task import GroundState, Task, TransitionState
from racerts.utils.checks import is_integer
from racerts.utils.log import verbose_logging

logger = logging.getLogger(__name__)


def generate(
    mol: Chem.Mol,
    task: Task,
    config: Optional[PipelineConfig] = None,
    pipeline: Optional[Pipeline] = None,
    charge: Optional[int] = None,
    multiplicity: Optional[int] = None,
    verbose: bool = False,
    restraints: Optional[RestraintSet] = None,
) -> ConformerEnsemble:
    """
    Generate a conformer ensemble of mol for the task.

    Args:
        mol: The molecular graph with all hydrogens; its conformer (if any) is the
            reference geometry for the frozen atoms of the task.
        task: TransitionState, GroundState, Constrained or any Task.
        config: Settings of the default pipeline (default PipelineConfig()).
        pipeline: A custom pipeline instead of the default one. Of config, only the
            seed and the restraints are then used (the seed by stages that create
            their own components, e.g. Embed() without an embedder; components passed
            to stages keep their own settings).
        charge, multiplicity: Override the values of mol (see
            set_charge_and_multiplicity).
        verbose: Log progress (INFO) during the call.
        restraints: Distance restraints in addition to those of config.restraints
            (they win for the same atom pair).
    """
    config = config if config is not None else PipelineConfig()
    roles = config.restraints.roles or None
    with verbose_logging(verbose):
        ctx = Context.create(
            mol,
            task,
            seed=config.seed,
            charge=charge,
            multiplicity=multiplicity,
            roles=roles,
        )
        if config.restraints or restraints:
            combined = RestraintSet()
            if config.restraints:
                combined = config.restraints.build(ctx.mol, ctx.frozen, config.seed)
            ctx = ctx.with_restraints(combined.merge(restraints or ()))
        pipeline = pipeline if pipeline is not None else config.build(task)
        return pipeline.run(ctx)


def generate_runs(
    mol: Chem.Mol,
    task: Task,
    seeds: Sequence[int],
    config: Optional[PipelineConfig] = None,
    pipeline: Optional[Pipeline] = None,
    charge: Optional[int] = None,
    multiplicity: Optional[int] = None,
    verbose: bool = False,
    restraints: Optional[RestraintSet] = None,
    window: float = 2.0,
    temperature: float = 298.15,
    energy_tolerance: Optional[float] = None,
    rmsd: float = 0.25,
) -> Runs:
    """
    Independent runs of generate, one per seed, merged and compared: a search of a
    flexible system is converged when further runs find the same low conformers again.

    Returns the ensemble of every seed, their union without duplicates (each conformer
    with the run that found it lowest, "run", and all that found it, "found_by") and a
    ConvergenceReport: the ensemble free energy of each run above that of the union,
    the share of a run's population that another run found again, and what leaving one
    run out changes (report.converged(tolerance)); print(result.report) gives the table.

    Args:
        seeds: At least two different seeds; each replaces config.seed for one run.
        mol, task, config, pipeline, charge, multiplicity, verbose, restraints: As
            for generate. A custom pipeline must take its seed from the run (stages
            that create their own components do; a component built with a seed of
            its own gives every run the same conformers, which is refused).
        window, temperature, energy_tolerance, rmsd: Of the comparison (compare_runs,
            merge_runs).

    The energies of the runs must compare: not for active bonds that are still held at
    their targets (search the saddle points freely first).
    """
    seeds = list(seeds)
    if not all(is_integer(seed) for seed in seeds):
        raise TypeError(f"seeds must be integers, not {seeds!r}.")
    if len(seeds) < 2:
        raise ValueError("generate_runs needs at least two seeds.")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"generate_runs needs different seeds, not {seeds}.")
    config = config if config is not None else PipelineConfig()
    # With every atom frozen nothing is sampled: all seeds give the reference.
    sampled = len(set(task.frozen_atoms(mol).hard)) < mol.GetNumAtoms()
    ensembles = []
    for seed in seeds:
        ensemble = generate(
            mol,
            task,
            config=replace(config, seed=int(seed)),
            pipeline=pipeline,
            charge=charge,
            multiplicity=multiplicity,
            verbose=verbose,
            restraints=restraints,
        )
        for earlier, other in zip(seeds, ensembles):
            if sampled and _same_conformers(ensemble, other):
                raise ValueError(
                    f"Seeds {earlier} and {seed} gave the same conformers: the "
                    "pipeline does not take the seed of the run (a component with a "
                    "seed of its own?)."
                )
        ensembles.append(ensemble)
    settings = dict(energy_tolerance=energy_tolerance, rmsd=rmsd)
    seeds = [int(seed) for seed in seeds]
    report = compare_runs(
        ensembles, seeds, window=window, temperature=temperature, **settings
    )
    with verbose_logging(verbose):
        logger.info("%s", report)
    return Runs(seeds, ensembles, merge_runs(ensembles, seeds, **settings), report)


def _same_conformers(a: ConformerEnsemble, b: ConformerEnsemble) -> bool:
    if len(a) != len(b) or not len(a):
        return False
    first_a = a.mol.GetConformer(a.conf_ids[0]).GetPositions()
    first_b = b.mol.GetConformer(b.conf_ids[0]).GetPositions()
    return bool(np.allclose(first_a, first_b, atol=1e-8))


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
    restraints: Optional[RestraintSet] = None,
    active_window=None,
    active_bonds: Optional[Sequence[Sequence[int]]] = None,
    stratify: Optional[int] = None,
    neighbor_window: float = 0.10,
) -> ConformerEnsemble:
    """
    A TS conformer ensemble from a TS geometry (xyz or sdf/mol file), with the reacting
    atoms and their neighbours kept at the TS geometry. With
    config=PipelineConfig.legacy() the result equals
    ConformerGenerator().generate_conformers (legacy racerts).

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
        restraints: Distance restraints (see generate).
        active_window, active_bonds, stratify, neighbor_window: Sample the lengths
            of the forming bonds in a window (see TransitionState).
    """
    if isinstance(smiles, str):
        smiles = [smiles]
    config = config if config is not None else PipelineConfig()
    if not auto_fallback:
        config = replace(config, refine=replace(config.refine, fallback=False))
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
        TransitionState(
            reacting_atoms,
            frozen_atoms,
            active_bonds=active_bonds,
            active_window=active_window,
            neighbor_window=neighbor_window,
            stratify=stratify,
        ),
        config=config,
        pipeline=pipeline,
        charge=charge,
        multiplicity=multiplicity,
        verbose=verbose,
        restraints=restraints,
    )


def generate_gs(
    smiles: Union[str, Chem.Mol],
    charge: Optional[int] = None,
    multiplicity: Optional[int] = None,
    config: Optional[PipelineConfig] = None,
    pipeline: Optional[Pipeline] = None,
    verbose: bool = False,
    restraints: Optional[RestraintSet] = None,
) -> ConformerEnsemble:
    """
    A ground-state conformer ensemble (nothing frozen) from a SMILES or a Mol.
    Hydrogens are added; by default embedding uses ETKDGv3, and the stereocentres of
    the input are kept. Without a given multiplicity, a graph with radical electrons
    (e.g. "[CH2]", a triplet carbene) has 1 + their number (GroundState.multiplicity);
    otherwise the lowest multiplicity for the electrons.
    """
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
        restraints=restraints,
    )
