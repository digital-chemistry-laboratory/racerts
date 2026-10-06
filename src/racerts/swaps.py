"""racerts.swap: conformers after a swap, sampled around the kept geometry."""

import json
import logging
from typing import Optional, Sequence, Union

import numpy as np
from rdkit import Chem

from racerts.config import PipelineConfig
from racerts.embed import Embed
from racerts.embed.rigid_attach import rigid_attach
from racerts.embed.stage import default_embedder
from racerts.pipeline import ConformerEnsemble, Context, Pipeline
from racerts.restraints import RestraintSet
from racerts.system.spec import set_charge_and_multiplicity
from racerts.system.stereo import trans_in_small_rings
from racerts.system.swap import Swap, SwapError, apply_swap
from racerts.task import Constrained, FrozenSet, GroundState, Task
from racerts.utils.checks import is_integer
from racerts.utils.log import verbose_logging
from racerts.validate.checks import clash_limits, first_clash

logger = logging.getLogger(__name__)

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
    for conf_id in conf_ids:
        reference_positions = None
        if reference is not None:  # the reference this conformer comes from
            first = reference.mol.GetConformers()[0].GetId()
            ref_id = ensemble.provenance(conf_id).get("reference", first)
            reference_positions = reference.mol.GetConformer(ref_id).GetPositions()
        pairs = clash_limits(
            mol,
            held,
            reference_positions,
            RESIDUAL_CLASH_FACTOR,
            hydrogen_factor=HYDROGEN_CLASH_FACTOR,
            reference_atoms=None if reference is None else reference.conserved,
        )
        clash = first_clash(mol.GetConformer(conf_id).GetPositions(), pairs)
        if clash:
            logger.warning("Conformer %d has a clash: %s.", conf_id, clash)
