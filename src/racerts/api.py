"""generate conformer ensembles for a task, and swap groups of a reference."""

import json
import logging
from dataclasses import replace
from typing import Optional, Sequence, Union

import numpy as np
from rdkit import Chem

from racerts.config import PipelineConfig
from racerts.embed import Embed
from racerts.embed.rigid_attach import rigid_attach
from racerts.embed.stage import default_embedder
from racerts.pipeline import ConformerEnsemble, Context, Pipeline
from racerts.pipeline.runs import Runs, compare_runs, merge_runs
from racerts.restraints import RestraintSet
from racerts.system.build import BaseMolGetter, build_mol
from racerts.system.graph import radical_multiplicity
from racerts.system.spec import set_charge_and_multiplicity
from racerts.system.stereo import trans_in_small_rings
from racerts.system.swap import Swap, SwapError, apply_swap
from racerts.task import Constrained, FrozenSet, GroundState, Task, TransitionState
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
SWAP_ROUTES = ("dg", "rigid")
# The rigid poses are opt-in: at an equal TS-search budget they crowded out better
# starting points.
DEFAULT_SWAP_ROUTES = ("dg",)
RESIDUAL_CLASH_FACTOR = 0.8  # of the vdW sum, heavy atoms beyond three bonds
HYDROGEN_CLASH_FACTOR = 0.5  # of the vdW sum, pairs with a hydrogen
CORE_DISTANCE_WARNING = 3  # bonds between an attachment and the frozen atoms


def swap(
    reference: Union[Chem.Mol, ConformerEnsemble],
    change: Swap,
    task: Optional[Task] = None,
    conserve: str = "soft",
    n_conformers: int = -1,
    routes: Optional[Sequence[str]] = None,
    hard: Sequence[int] = (),
    n_rigid_conformers: int = 3,
    n_rotations: int = 12,
    config: Optional[PipelineConfig] = None,
    charge: Optional[int] = None,
    multiplicity: Optional[int] = None,
    verbose: bool = False,
    restraints: Optional[RestraintSet] = None,
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
              reference (single attachments only);
            - "soft": the kept atoms outside the junction (see SwapResult) start at
              the reference and are held near it by position restraints (0.3 A, k 5
              kcal/(mol A^2)) in MMFF/UFF refinement; the junction and the new atoms
              are sampled;
            - "free": only the hard atoms of the task are held (full resampling).
        n_conformers: Distance-geometry conformers per reference (-1: as
            config.embed.n_conformers, by default the count for the molecule).
        routes: "dg" (distance geometry with the kept atoms mapped) and "rigid"
            (fragment conformers turned about the attachment bond; single
            attachments that replace a bond only); default "dg". Each conformer
            records its route in the provenance.
        hard: Further atoms (reference indices) to hold fixed: kept ones, or removed
            ones whose replacement then stays at their position (e.g. the donor atoms
            of a ligand swap, SwapResult.positioned).
        n_rigid_conformers, n_rotations: Poses of the rigid route per reference.
        config: Settings of the embedding, refinement and pruning; embedding uses the
            coordinate map and the "frozen_first" chirality fallback. config.restraints
            must be empty: the restraints of the reference go into restraints.
        charge, multiplicity, verbose: See generate.
        restraints: Distance restraints of the reference (reference indices), e.g. its
            hydrogen bonds: those between kept atoms hold in embedding and refinement;
            the others are left out with a warning and listed in the molecule
            property "swap_lost_restraints" (JSON).

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
    if conserve == "hard" and (
        routes is not None or n_conformers != -1 or restraints is not None
    ):
        raise SwapError(
            "conserve='hard' samples nothing: no routes, n_conformers or restraints."
        )
    routes = DEFAULT_SWAP_ROUTES if routes is None else routes
    unknown = set(routes) - set(SWAP_ROUTES)
    if unknown or not routes:
        raise SwapError(f"routes must be taken from {SWAP_ROUTES}, not {list(routes)}.")
    if config.restraints:
        raise SwapError(
            "config.restraints must be empty for a swap; racerts.swap(restraints=...) "
            "carries the restraints of the reference over."
        )
    if config.embed.mode != "cmap":
        raise SwapError("Swaps embed with the coordinate map: embed.mode 'cmap'.")
    invalid = [i for i in hard if not (is_integer(i) and 0 <= i < mol.GetNumAtoms())]
    if invalid:
        raise SwapError(f"Invalid hard atoms {invalid}.")
    if "rigid" not in routes and (n_rigid_conformers, n_rotations) != (3, 12):
        raise SwapError("n_rigid_conformers and n_rotations need the rigid route.")
    with verbose_logging(verbose):
        result = apply_swap(mol, change, seed=config.seed, restraints=restraints)
        index_map = result.index_map  # replaced atoms pass their role on
        try:  # an atom of the task that leaves without replacement
            base = task.remap(index_map) if task is not None else None
        except ValueError as error:
            raise SwapError(str(error)) from None
        if getattr(base, "windowed", False):
            raise SwapError(
                "Swaps of tasks with active-bond windows are not supported: swap, "
                "then generate the windowed ensemble from the new reference."
            )
        frozen = base.frozen_atoms(result.mol) if base is not None else FrozenSet()
        missing = [i for i in hard if i not in index_map]
        if missing:
            raise SwapError(f"The hard atoms {missing} leave in the swap.")
        asked = [index_map[i] for i in hard]
        held = tuple(dict.fromkeys((*frozen.hard, *asked)))
        unplaced = sorted(set(held) & set(result.new_atoms) - set(result.positioned))
        if unplaced:
            raise SwapError(
                f"The frozen atoms {unplaced} are new atoms without coordinates; hold "
                "only kept atoms, or atoms that replace one (SwapResult.positioned)."
            )
        # The rest of a graft has coordinates, but its rotation about the new bond is
        # arbitrary: only the atoms that replace one are where the reference puts them.
        anchored = set(result.conserved) | set(result.replaced.values())
        loose = [i for i in held if i not in anchored]
        if set(loose) & set(asked):
            raise SwapError(
                f"The hard atoms {sorted(set(loose) & set(asked))} are new atoms whose "
                "position the reference does not define; hold only kept atoms, or "
                "atoms that replace one (SwapResult.replaced)."
            )
        if loose:  # e.g. the new neighbours of an atom that took a reacting atom's role
            logger.warning(
                "The frozen atoms %s of the task are new atoms whose position the "
                "reference does not define; they are sampled, not held.",
                loose,
            )
            held = tuple(i for i in held if i in anchored)
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
            _warn_residual_clash(grafted, held, grafted.conf_ids, result)
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
            restraints=result.restraints,
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
                reference_bounds=config.embed.reference_bounds,
                num_threads=config.num_threads,
            )
            several = new_task.needs_reference and result.mol.GetNumConformers() > 1
            stage = Embed(
                embedder,
                config.embed.n_conformers if n_conformers == -1 else n_conformers,
                config.embed.conf_factor,
                count_policy=config.embed.count_policy,
                references="all" if several else None,
            )
            try:
                embedded = stage.run(ctx)
            except RuntimeError as error:
                raise _strained(result.mol, error) or error from None
            embedded.add_provenance(route="dg")
            parts.append(embedded)
        if "rigid" in routes:
            if result.placed:
                parts.append(
                    rigid_attach(result, n_rigid_conformers, n_rotations, config.seed)
                )
            else:
                logger.info("No rigid route: it needs a single attachment.")
        if not parts:
            raise SwapError("No route applies to this swap; use routes=['dg'].")
        ensemble = parts[0]
        for part in parts[1:]:
            ensemble = ensemble.merge(part)
        stages = [s for s in config.build(new_task).stages if s.name != "embed"]
        for stage in stages:  # the swap embeds with the frozen_first fallback
            if stage.name == "refine":
                stage.stereo_anchors = True
        try:
            ensemble = Pipeline(stages).run(ctx, ensemble)
        except RuntimeError as error:
            raise _strained(result.mol, error) or error from None
        _record_index_map(ensemble.mol, index_map)
        if restraints is not None:
            ensemble.mol.SetProp(
                "swap_lost_restraints", json.dumps(result.lost_contacts)
            )
        _warn_residual_clash(ensemble, held, reference=result)
    return ensemble


def _strained(mol: Chem.Mol, error) -> Optional[SwapError]:
    """The error of a swap that no conformer survived, if the graph asks for a trans
    double bond in a small ring: the likely reason."""
    strained = trans_in_small_rings(mol)
    if not strained:
        return None
    listed = ", ".join(f"{a}={b} (ring of {n})" for a, b, n in strained)
    return SwapError(
        f"No conformer has the trans double bond {listed} that the swap asks for: "
        f"the embedding or the force field does not hold it ({error})"
    )


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
        logger.warning(
            "The swap attaches %s the frozen atoms: the new group may change the core "
            "(e.g. the TS).",
            where,
        )


def _warn_residual_clash(
    ensemble: ConformerEnsemble, held=(), conf_ids=None, reference=None
) -> None:
    """
    Warn if atoms more than three bonds apart clash in the conformers conf_ids
    (default: the lowest one): heavy atoms closer than RESIDUAL_CLASH_FACTOR times
    their vdW sum, pairs with a hydrogen closer than HYDROGEN_CLASH_FACTOR times it.
    Not checked: pairs of held atoms (they keep the reference distance, e.g. a forming
    bond). Pairs of kept atoms that are that close in the conformer's own reference
    (reference: a SwapResult; e.g. a coordination that the graph lacks) count as a
    clash only if they come more than 0.2 A closer.
    """
    if not len(ensemble):
        return
    mol = ensemble.mol
    if conf_ids is None:
        energies = ensemble.energies()
        found = energies.size and np.isfinite(energies).any()
        conf_ids = [ensemble.best() if found else ensemble.conf_ids[0]]
    table = Chem.GetPeriodicTable()
    numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
    radii = np.array([table.GetRvdw(int(z)) for z in numbers])
    factor = np.where(
        (numbers[:, None] == 1) | (numbers[None, :] == 1),
        HYDROGEN_CLASH_FACTOR,
        RESIDUAL_CLASH_FACTOR,
    )
    limit = factor * (radii[:, None] + radii[None, :])
    check = np.triu(Chem.GetDistanceMatrix(mol) > 3, k=1)
    held = sorted(set(held))
    if held:
        check[np.ix_(held, held)] = False
    kept = np.array(reference.conserved) if reference is not None else None
    for conf_id in conf_ids:
        positions = mol.GetConformer(conf_id).GetPositions()
        distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
        bound = limit
        if kept is not None:  # the reference this conformer comes from
            first = reference.mol.GetConformers()[0].GetId()
            ref_id = ensemble.provenance(conf_id).get("reference", first)
            x = reference.mol.GetConformer(ref_id).GetPositions()[kept]
            ref = np.linalg.norm(x[:, None] - x[None, :], axis=-1)
            bound = limit.copy()
            own = bound[np.ix_(kept, kept)]
            bound[np.ix_(kept, kept)] = np.where(ref < own, ref - 0.2, own)
        clashes = np.argwhere(check & (distances < bound))
        if len(clashes):
            i, j = clashes[0]
            logger.warning(
                "Conformer %d has a clash: atoms %d and %d are closer than %.2f A.",
                conf_id,
                i,
                j,
                bound[i, j],
            )
