"""generate conformer ensembles for a task, and swap groups of a reference."""

import json
import logging
from dataclasses import replace
from typing import Optional, Sequence, Union

import numpy as np
from rdkit import Chem

from racerts.config import PipelineConfig
from racerts.embed import Embed
from racerts.embed.stage import default_embedder
from racerts.pipeline import ConformerEnsemble, Context, Pipeline
from racerts.restraints import RestraintSet
from racerts.system.build import BaseMolGetter, build_mol
from racerts.system.graph import radical_multiplicity
from racerts.system.spec import set_charge_and_multiplicity
from racerts.system.swap import Swap, SwapError, apply_swap
from racerts.task import Constrained, FrozenSet, GroundState, Task, TransitionState
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
            seed is then used, by stages that create their own components (e.g.
            Embed() without an embedder); components passed to stages keep their
            own settings.
        charge, multiplicity: Override the values of mol (see
            set_charge_and_multiplicity).
        verbose: Log progress (INFO) during the call.
        restraints: Distance restraints in addition to those of config.restraints
            (they win for the same atom pair).
    """
    config = config if config is not None else PipelineConfig()
    with verbose_logging(verbose):
        ctx = Context.create(
            mol, task, seed=config.seed, charge=charge, multiplicity=multiplicity
        )
        if config.restraints or restraints:
            combined = RestraintSet()
            if config.restraints:
                combined = config.restraints.build(ctx.mol, ctx.frozen, config.seed)
            ctx = Context.create(
                ctx.mol,
                task,
                seed=config.seed,
                restraints=combined.merge(restraints or ()),
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
    restraints: Optional[RestraintSet] = None,
    active_window=None,
    active_bonds: Optional[Sequence[Sequence[int]]] = None,
    stratify: int = 0,
    neighbor_window: float = 0.10,
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
    (e.g. "[CH2]", a triplet carbene) has 1 + their number; otherwise the lowest
    multiplicity for the electrons.
    """
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else smiles
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol, addCoords=mol.GetNumConformers() > 0)  # a new Mol
    if multiplicity is None and not mol.HasProp("multiplicity"):
        if radical_multiplicity(mol) > 1:
            multiplicity = radical_multiplicity(mol)
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


CONSERVE = ("hard", "soft", "free")
SWAP_ROUTES = ("dg",)
DEFAULT_SWAP_ROUTES = ("dg",)
CORE_DISTANCE_WARNING = 3  # bonds between an attachment and the frozen atoms


def swap(
    reference: Union[Chem.Mol, ConformerEnsemble],
    change: Swap,
    task: Optional[Task] = None,
    conserve: str = "soft",
    n_conformers: int = -1,
    routes: Optional[Sequence[str]] = None,
    hard: Sequence[int] = (),
    config: Optional[PipelineConfig] = None,
    charge: Optional[int] = None,
    multiplicity: Optional[int] = None,
    verbose: bool = False,
) -> ConformerEnsemble:
    """
    Conformers after a swap (racerts.system.swap.Swap): the reference with a group
    replaced, sampled around the kept geometry.

    Args:
        reference: The molecular graph with one or more conformers (e.g. an ensemble
            of TS conformers); each conformer is a reference.
        change: What is replaced by what (Swap).
        task: The task of the reference (reference atom indices), e.g. its
            TransitionState: its frozen atoms stay hard. An atom replaced by a
            fragment atom passes its role on (SwapResult.index_map).
        conserve: What happens to the kept atoms:
            - "hard": nothing is sampled; the fragment is grafted rigidly on each
              reference (catmlp's transfer; single attachments only);
            - "soft": the kept atoms outside the junction (see SwapResult) start at
              the reference and are held near it by position restraints (0.3 A, k 5
              kcal/(mol A^2)) in MMFF/UFF refinement; the junction and the new atoms
              are sampled;
            - "free": only the hard atoms of the task are held (full resampling).
        n_conformers: Distance-geometry conformers per reference (-1: the count of
            config.embed).
        routes: "dg" (distance geometry with the kept atoms mapped), the default.
            Each conformer records its route in the provenance.
        hard: Further atoms (reference indices) to hold fixed: kept ones, or removed
            ones whose replacement then stays at their position (e.g. the donor atoms
            of a ligand swap, SwapResult.positioned).
        config: Settings of the embedding, refinement and pruning; embedding uses the
            coordinate map and the "frozen_first" chirality fallback; swaps take no
            restraints (config.restraints raises).
        charge, multiplicity, verbose: See generate.

    The conformers are then refined, pruned by energy and duplicates as in the
    default pipeline.

    Raises:
        SwapError (a ValueError): for a swap or settings that do not work as given.
    """
    config = config if config is not None else PipelineConfig()
    mol = reference.mol if isinstance(reference, ConformerEnsemble) else reference
    if not isinstance(mol, Chem.Mol) or mol.GetNumConformers() == 0:
        raise SwapError("A swap needs the reference as a Mol with conformers.")
    if conserve not in CONSERVE:
        raise SwapError(f"conserve must be one of {CONSERVE}, not {conserve!r}.")
    if conserve == "hard" and (routes is not None or n_conformers != -1):
        raise SwapError("conserve='hard' samples nothing: no routes or n_conformers.")
    routes = DEFAULT_SWAP_ROUTES if routes is None else routes
    unknown = set(routes) - set(SWAP_ROUTES)
    if unknown or not routes:
        raise SwapError(f"routes must be taken from {SWAP_ROUTES}, not {list(routes)}.")
    if config.restraints:
        raise SwapError("Swaps take no restraints: config.restraints must be empty.")
    if config.embed.mode != "cmap":
        raise SwapError("Swaps embed with the coordinate map: embed.mode 'cmap'.")
    invalid = [
        i
        for i in hard
        if isinstance(i, bool)
        or not isinstance(i, (int, np.integer))
        or not 0 <= i < mol.GetNumAtoms()
    ]
    if invalid:
        raise SwapError(f"Invalid hard atoms {invalid}.")
    with verbose_logging(verbose):
        result = apply_swap(mol, change, seed=config.seed)
        index_map = result.index_map  # replaced atoms pass their role on
        try:  # an atom of the task that leaves without replacement
            base = task.remap(index_map) if task is not None else None
        except ValueError as error:
            raise SwapError(str(error)) from None
        if getattr(base, "windowed", False):
            raise SwapError(
                "Swaps of tasks with active-bond windows are not supported: swap, "
                "then generate the windowed ensemble from the new seed."
            )
        frozen = base.frozen_atoms(result.mol) if base is not None else FrozenSet()
        missing = [i for i in hard if i not in index_map]
        if missing:
            raise SwapError(f"The hard atoms {missing} leave in the swap.")
        held = tuple(dict.fromkeys((*frozen.hard, *(index_map[i] for i in hard))))
        unplaced = sorted(set(held) & set(result.new_atoms) - set(result.positioned))
        if unplaced:
            raise SwapError(
                f"The frozen atoms {unplaced} are new atoms without coordinates; hold "
                "only kept atoms, or atoms that replace one (SwapResult.positioned)."
            )
        _warn_near_core(result, held)
        if conserve == "hard":
            if not result.placed:
                raise SwapError(
                    "conserve='hard' needs a single attachment that replaces a bond."
                )
            grafted = ConformerEnsemble(Chem.Mol(result.mol))
            set_charge_and_multiplicity(grafted.mol, charge, multiplicity)
            _record_index_map(grafted.mol, index_map)
            for conf_id in grafted.conf_ids:  # each is the graft on that reference
                grafted.add_provenance(conf_id, route="graft", reference=conf_id)
            return grafted

        soft = []
        if conserve == "soft":
            skip = set(held) | set(result.junction)
            soft = [i for i in result.conserved if i not in skip]
        if held or soft:
            core = frozen.core if base is not None and frozen.hard else None
            new_task = Constrained(held, soft, core=core)
        else:
            new_task = GroundState()
        ctx = Context.create(
            result.mol,
            new_task,
            seed=config.seed,
            charge=charge,
            multiplicity=multiplicity,
        )
        parts = []
        if "dg" in routes:
            embedder = default_embedder(
                new_task,
                config.seed,
                mode="cmap",
                etkdg=config.embed.etkdg,
                chirality_fallback="frozen_first",
                useRandomCoords=config.embed.use_random_coords,
                sequential_seeds=config.embed.sequential_seeds,
                num_threads=config.num_threads,
            )
            several = new_task.needs_reference and result.mol.GetNumConformers() > 1
            embedded = Embed(
                embedder,
                n_conformers,
                config.embed.conf_factor,
                count_policy=config.embed.count_policy,
                references="all" if several else None,
            ).run(ctx)
            embedded.add_provenance(route="dg")
            parts.append(embedded)
        ensemble = parts[0]
        for part in parts[1:]:
            ensemble = ensemble.merge(part)
        stages = [s for s in config.build(new_task).stages if s.name != "embed"]
        ensemble = Pipeline(stages).run(ctx, ensemble)
        _record_index_map(ensemble.mol, index_map)
    return ensemble


def _record_index_map(mol: Chem.Mol, index_map) -> None:
    """The reference index -> new index map as the property "swap_index_map" (JSON),
    e.g. for the atom lists of a follow-up run."""
    mol.SetProp(
        "swap_index_map", json.dumps({str(i): int(k) for i, k in index_map.items()})
    )


def _warn_near_core(result, held) -> None:
    if not held:
        return
    distances = Chem.GetDistanceMatrix(result.mol)
    ends = {i for pair in result.attachments for i in pair}
    closest = int(min(distances[i, j] for i in ends for j in held))
    if closest < CORE_DISTANCE_WARNING:
        where = "at" if closest == 0 else f"{closest} bond(s) from"
        message = (
            f"The swap attaches {where} the frozen atoms: the new group may change "
            "the core (e.g. the TS)."
        )
        result.warnings.append(message)
        logger.warning(message)
