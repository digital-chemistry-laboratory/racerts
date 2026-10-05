"""
The racerts command line.

    racerts ts FILE -r ATOM [ATOM ...] [-s SMILES ...]   TS conformers
    racerts gs SMILES                                    ground-state conformers
    racerts swap FILE --new "[*]CCCC" --old "[CH3][c:1]"  conformers after a swap
    racerts FILE [options]                               legacy (or: racerts run)
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import replace

from racerts.api import CONSERVE, SWAP_ROUTES, generate_gs, generate_ts, swap
from racerts.config import PipelineConfig
from racerts.embed import CHIRALITY_FALLBACK_MODES, COUNT_POLICIES, EMBED_MODES
from racerts.embed.bounds import REFERENCE_BOUNDS
from racerts.pipeline import Pipeline
from racerts.refine import REFINE_BACKENDS
from racerts.refine.forcefield import DIELECTRIC_MODELS
from racerts.restraints.export import DEFAULT_PATHS as EXPORT_PATHS
from racerts.restraints.export import FORMATS as EXPORT_FORMATS
from racerts.restraints.export import ExportRestraints
from racerts.system import GRAPH_METHODS, build_mol
from racerts.system.swap import MODES as SWAP_MODES
from racerts.system.swap import Swap
from racerts.task import TransitionState
from racerts.utils.log import cli_logging

SUBCOMMANDS = ("ts", "gs", "swap")
DEFAULT_OUTPUT = "conformer_ensemble.xyz"


def main(argv=None) -> None:
    """The console script (its return value becomes the exit status: None)."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in SUBCOMMANDS:
        run_subcommand(argv)
        return
    from racerts.compat import cli  # not at the top: it imports the helpers below

    cli.main(argv[1:] if argv and argv[0] == "run" else argv)


def _subcommand_parser() -> argparse.ArgumentParser:
    """
    The parser of the subcommands. Each gets only the options it uses: the pipeline
    options all three, the embedding and restraint options ts and gs.
    """
    parser = argparse.ArgumentParser(prog="racerts")
    sub = parser.add_subparsers(dest="command", required=True)
    defaults = PipelineConfig()
    _add_ts(sub, defaults)
    _add_gs(sub, defaults)
    _add_swap(sub, defaults)
    return parser


def _add_ts(sub, defaults) -> None:
    command = sub.add_parser(
        "ts",
        description="TS conformers: the reacting atoms and their neighbours stay at "
        "the TS geometry. With --legacy, the same ensemble as legacy racerts.",
    )
    command.add_argument(
        "filename", help="TS geometry (.xyz, or .sdf/.mol with bonds)."
    )
    command.add_argument(
        "-r",
        "--reacting-atoms",
        type=int,
        nargs="+",
        required=True,
        help="0-based indices of the atoms whose bonds form or break.",
    )
    command.add_argument(
        "-s",
        "--smiles",
        nargs="+",
        help="SMILES of the TS topology, one per fragment (recommended for xyz).",
    )
    command.add_argument(
        "--frozen-atoms",
        type=int,
        nargs="+",
        help="Frozen atoms instead of the reacting atoms and their neighbours.",
    )
    command.add_argument(
        "--graph",
        choices=list(GRAPH_METHODS),
        help="First method for the molecular graph (default: smiles).",
    )
    command.add_argument("-c", "--charge", type=int, default=0, help="Total charge.")
    command.add_argument(
        "--active-window",
        type=float,
        nargs="+",
        metavar="A",
        help="Sample the forming bonds in a window: DELTA (reference length +/- DELTA) "
        "or LO HI (A); writes <output>.active_bonds.csv next to the output.",
    )
    command.add_argument(
        "--active-bond",
        type=int,
        nargs=2,
        action="append",
        metavar=("I", "J"),
        help="An active bond (repeatable; default: the forming bonds of the "
        "reference).",
    )
    command.add_argument(
        "--stratify",
        type=int,
        default=None,
        metavar="K",
        help="K target lengths, the midpoints of K equal parts of the window "
        "(default: 5; 0: placed by the embedding, unevenly).",
    )
    command.add_argument(
        "--neighbor-window",
        type=float,
        help="+/- A for the reacting atoms to their neighbours, with a window "
        "(default 0.1).",
    )
    command.add_argument(
        "--chirality-fallback",
        choices=list(CHIRALITY_FALLBACK_MODES),
        help="When the frozen atoms contradict chiral tags: drop all tags (legacy) "
        "or those of the frozen atoms first (default "
        f"{defaults.embed.chirality_fallback}).",
    )
    _add_reference_bounds(command, defaults)
    _add_embedding_and_restraint_options(command, defaults)
    command.add_argument(
        "--keep-hbonds",
        action="store_true",
        default=None,
        help="Keep the hydrogen bonds of the reference geometry.",
    )
    command.add_argument(
        "--contact",
        nargs=2,
        type=int,
        action="append",
        metavar=("I", "J"),
        help="Keep the non-covalent contact I...J as in the reference geometry.",
    )
    command.add_argument(
        "--keep-fragments",
        action="store_true",
        default=None,
        help="Keep fragments without reacting atoms (e.g. solvent) at the core.",
    )
    command.add_argument(
        "--restraint-fraction",
        type=float,
        help="Probability with which an embedding batch takes each restraint of the "
        "reference geometry (--keep-hbonds, --contact, --keep-fragments; default "
        f"{defaults.restraints.fraction:g}).",
    )
    _add_pipeline_options(command, defaults)


def _add_gs(sub, defaults) -> None:
    command = sub.add_parser(
        "gs",
        description="Ground-state conformers (nothing frozen), embedded with ETKDGv3.",
    )
    command.add_argument("smiles", help="SMILES of the molecule.")
    command.add_argument(
        "-c", "--charge", type=int, default=None, help="Total charge (default: SMILES)."
    )
    _add_embedding_and_restraint_options(command, defaults)
    command.add_argument(
        "--link-fragments",
        action="store_true",
        default=None,
        help="Embed the fragments of a complex together.",
    )
    _add_pipeline_options(command, defaults)


def _add_swap(sub, defaults) -> None:
    command = sub.add_parser(
        "swap",
        description="Replace a group of the reference by a new fragment and sample "
        "the new atoms around the kept geometry (see racerts.swap).",
    )
    command.add_argument("filename", help="Reference geometry (.xyz, or .sdf/.mol).")
    command.add_argument(
        "--new",
        required=True,
        help="The new fragment: SMILES with a dummy per attachment ([*] or [*:1]).",
    )
    where = command.add_mutually_exclusive_group(required=True)
    where.add_argument(
        "--old", help="SMARTS of the group that leaves, the kept atom mapped ([c:1])."
    )
    where.add_argument(
        "--remove",
        type=int,
        nargs="*",
        metavar="I",
        help="Atoms that leave (their hydrogens too); none: an addition (--attach).",
    )
    where.add_argument(
        "--site", type=int, help="Map number of the terminal H or dummy that leaves."
    )
    where.add_argument(
        "--center",
        type=int,
        nargs=2,
        metavar=("ATOM", "GROUP"),
        help="The GROUP-th group bound to ATOM leaves (groups by lowest atom index).",
    )
    command.add_argument(
        "--attach",
        type=int,
        nargs=2,
        action="append",
        metavar=("DUMMY", "ATOM"),
        help="Dummy number and the atom it binds to (repeatable).",
    )
    command.add_argument(
        "--bond-type",
        nargs=2,
        action="append",
        metavar=("DUMMY", "TYPE"),
        help="single, double, triple or dative (from the fragment atom).",
    )
    command.add_argument(
        "--mode",
        choices=list(SWAP_MODES),
        default="append",
        help="append (default): new atoms take the slots of removed ones, the rest are "
        "appended; renumber: kept atoms first. Changed indices are logged.",
    )
    command.add_argument(
        "-s", "--smiles", nargs="+", help="SMILES of the reference, one per fragment."
    )
    command.add_argument("-c", "--charge", type=int, default=0, help="Total charge.")
    command.add_argument(
        "-r",
        "--reacting-atoms",
        type=int,
        nargs="+",
        help="For a TS reference: its reacting atoms (they and their neighbours stay).",
    )
    command.add_argument(
        "--conserve",
        choices=list(CONSERVE),
        default="soft",
        help="hard: graft only; soft: kept atoms restrained near the reference "
        "(default); free: only the TS core held.",
    )
    command.add_argument(
        "--routes",
        nargs="+",
        choices=list(SWAP_ROUTES),
        default=None,
        help="Distance geometry and/or rigid poses of the fragment (default dg).",
    )
    command.add_argument(
        "--hard", type=int, nargs="+", default=[], help="Further atoms held fixed."
    )
    _add_reference_bounds(command, defaults)
    _add_pipeline_options(command, defaults)


def _add_reference_bounds(command, defaults) -> None:
    """An option of ts and swap, which embed around a reference geometry."""
    command.add_argument(
        "--reference-bounds",
        choices=list(REFERENCE_BOUNDS),
        help="Distance bounds of the graph that exclude a distance of the reference: "
        "widened to it when embedding fails without (fallback), always, or never "
        f"(default {defaults.embed.reference_bounds}).",
    )


def _add_embedding_and_restraint_options(command, defaults) -> None:
    """Options of ts and gs: the embedding mode and the distance restraints."""
    command.add_argument(
        "--embed",
        choices=list(EMBED_MODES),
        help=f"Embedding mode (default {defaults.embed.mode}).",
    )
    command.add_argument(
        "--restraint",
        nargs=3,
        action="append",
        metavar=("I", "J", "DISTANCE"),
        help="Keep atoms I and J at DISTANCE (A) +/- the half width (repeatable).",
    )
    command.add_argument(
        "--hints",
        action="store_true",
        default=None,
        help="Hydrogen bonds from the graph as embedding windows in some batches.",
    )
    command.add_argument(
        "--restraint-half-width",
        type=float,
        help=f"Half width of restraint windows, A (default "
        f"{defaults.restraints.half_width:g}).",
    )
    command.add_argument(
        "--restraint-force-constant",
        type=float,
        help="Flat-bottom force constant, kcal/(mol A^2) (default "
        f"{defaults.restraints.force_constant:g}).",
    )
    command.add_argument(
        "--export-restraints",
        choices=list(EXPORT_FORMATS),
        help="Write the frozen atoms and restraints for xtb, CREST (--cinp), ORCA "
        "or as JSON, and stop before embedding.",
    )
    command.add_argument(
        "--export-to",
        help="File of --export-restraints (default: restraints.xcontrol, "
        ".inp or .json next to the output).",
    )


def _add_pipeline_options(command, defaults) -> None:
    """Options of every subcommand: the config, its common settings and the output."""
    command.add_argument(
        "--multiplicity",
        type=int,
        help="Spin multiplicity 2S+1 (default: the lowest for the electrons).",
    )
    command.add_argument(
        "--config",
        help="PipelineConfig as JSON or YAML; the options given here override it.",
    )
    command.add_argument(
        "--legacy",
        action="store_true",
        help="The settings of legacy racerts where the defaults changed since "
        "(--config and the options given here override them).",
    )
    command.add_argument(
        "-n",
        "--n-conformers",
        type=int,
        help="Conformers to embed (default: rotatable bonds * conf factor + 30).",
    )
    command.add_argument(
        "--conf-factor",
        type=int,
        help=f"Conformers per rotatable bond (default {defaults.embed.conf_factor}).",
    )
    command.add_argument(
        "--etkdg",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="ETKDGv3 instead of plain distance geometry (default: only for gs).",
    )
    command.add_argument(
        "--count-policy",
        choices=list(COUNT_POLICIES),
        help="How the default number of conformers is counted: legacy, fragments "
        "(adds the rigid-body freedom of fragments without frozen atoms) or per_bond "
        f"(10 per rotatable bond, at least 7; default {defaults.embed.count_policy}).",
    )
    command.add_argument(
        "--sequential-seeds",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="One seed per conformer (legacy racerts embeds its first 3 twice; "
        f"default {defaults.embed.sequential_seeds}).",
    )
    command.add_argument(
        "--refine",
        choices=list(REFINE_BACKENDS),
        help=f"Force field (default {defaults.refine.backend}).",
    )
    command.add_argument(
        "--converge",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Minimize until the energy stops dropping (legacy racerts stops "
        f"early next to the frozen atoms; default {defaults.refine.converge}).",
    )
    command.add_argument(
        "--anchor-free-energies",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Energies without the terms that hold the frozen atoms (default "
        f"{defaults.refine.anchor_free_energies}).",
    )
    command.add_argument(
        "--dielectric",
        nargs=2,
        metavar=("MODEL", "CONSTANT"),
        help=f"MMFF dielectric: model ({', '.join(DIELECTRIC_MODELS)}) and "
        "constant, e.g. 'distance 4' (default: constant 1).",
    )
    command.add_argument(
        "--no-fallback",
        action="store_true",
        help="Fail instead of falling back (graph: bonds, connectivity; MMFF: UFF).",
    )
    command.add_argument(
        "--seed", type=int, help=f"Random seed (default {defaults.seed})."
    )
    command.add_argument(
        "--num-threads",
        type=int,
        help=f"Threads (default {defaults.num_threads}).",
    )
    command.add_argument(
        "--energy-threshold",
        type=float,
        help=f"Energy window, kcal/mol (default {defaults.prune.energy_threshold:g}).",
    )
    command.add_argument(
        "--rmsd-threshold",
        type=float,
        help=f"Duplicate RMSD, A (default {defaults.prune.rmsd_threshold:g}).",
    )
    command.add_argument(
        "--check-stereo",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="After refinement, drop conformers whose stereo differs from the "
        f"graph (default {defaults.prune.check_stereo}).",
    )
    command.add_argument(
        "--rmsd-hydrogens",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include hydrogens in the duplicate RMSD (default "
        f"{defaults.prune.include_hs}).",
    )
    command.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output file (default {DEFAULT_OUTPUT}).",
    )
    command.add_argument(
        "--crest-energies",
        action="store_true",
        help="Write only the energy (Hartree) on each comment line, as CREST does.",
    )
    _add_verbose(command)


def _config_from_args(args) -> PipelineConfig:
    """
    The defaults (or those of legacy racerts), updated by the config file and by the
    options given on the command line (those that the subcommand has).
    """

    def given(name):
        return getattr(args, name, None)

    if args.config:
        config = PipelineConfig.from_file(args.config, legacy=args.legacy)
    else:
        config = PipelineConfig.legacy() if args.legacy else PipelineConfig()
    dielectric = {}
    if args.dielectric:
        model, constant = args.dielectric
        try:
            dielectric = dict(
                dielectric_model=model, dielectric_constant=float(constant)
            )
        except ValueError:
            raise ValueError(f"--dielectric: {constant!r} is not a number.") from None
    user = None
    if given("restraint"):
        try:
            user = [[int(i), int(j), float(d)] for i, j, d in args.restraint]
        except ValueError:
            raise ValueError(
                f"--restraint takes two atom indices and a distance: {args.restraint}"
            ) from None
    contacts = given("contact")
    return _replace(
        config,
        seed=args.seed,
        restraints=_replace(
            config.restraints,
            user=user,
            hbonds=given("keep_hbonds"),
            contacts=[list(pair) for pair in contacts] if contacts else None,
            keep_fragments=given("keep_fragments"),
            link_fragments=given("link_fragments"),
            hints=given("hints"),
            fraction=given("restraint_fraction"),
            half_width=given("restraint_half_width"),
            force_constant=given("restraint_force_constant"),
        ),
        num_threads=args.num_threads,
        embed=_replace(
            config.embed,
            n_conformers=args.n_conformers,
            conf_factor=args.conf_factor,
            mode=given("embed"),
            etkdg=args.etkdg,
            count_policy=args.count_policy,
            sequential_seeds=args.sequential_seeds,
            chirality_fallback=given("chirality_fallback"),
            reference_bounds=given("reference_bounds"),
        ),
        refine=_replace(
            config.refine,
            backend=args.refine,
            fallback=False if args.no_fallback else None,
            converge=args.converge,
            anchor_free_energies=args.anchor_free_energies,
            **dielectric,
        ),
        prune=_replace(
            config.prune,
            energy_threshold=args.energy_threshold,
            rmsd_threshold=args.rmsd_threshold,
            include_hs=args.rmsd_hydrogens,
            check_stereo=args.check_stereo,
        ),
    )


def _replace(obj, **changes):
    """dataclasses.replace with the changes that are not None (checked again)."""
    return replace(
        obj, **{key: value for key, value in changes.items() if value is not None}
    )


def run_subcommand(argv):
    """
    racerts ts, racerts gs and racerts swap. Wrong input (a ValueError or OSError,
    e.g. an invalid setting or a missing config file) ends with the message and exit
    status 2, not a traceback; -vv shows the traceback.
    """

    def run(parser, args):
        try:
            return _run_subcommand(parser, args)
        except (ValueError, OSError) as error:
            if args.verbose >= 2:
                raise
            parser.error(str(error))

    return _parse_and_run(_subcommand_parser(), argv, run)


def _run_subcommand(parser, args):
    if args.command == "swap":
        return _run_swap(parser, args)
    config = _config_from_args(args)
    pipeline = None
    if args.export_restraints:
        path = args.export_to or os.path.join(
            os.path.dirname(args.output), EXPORT_PATHS[args.export_restraints]
        )
        pipeline = Pipeline([ExportRestraints(args.export_restraints, path)])
    elif args.export_to:
        parser.error("--export-to needs --export-restraints.")

    if args.command == "ts":
        _check_file(parser, args.filename)
        window = args.active_window
        window_options = {}
        if args.neighbor_window is not None:
            window_options["neighbor_window"] = args.neighbor_window
        if window is None and (
            args.active_bond or args.stratify is not None or window_options
        ):
            parser.error(
                "--active-bond, --stratify and --neighbor-window need --active-window."
            )
        if window is not None:
            if len(window) not in (1, 2):
                parser.error("--active-window takes DELTA or LO HI.")
            window = window[0] if len(window) == 1 else tuple(window)
        ensemble = generate_ts(
            args.filename,
            args.reacting_atoms,
            charge=args.charge,
            smiles=args.smiles,
            multiplicity=args.multiplicity,
            frozen_atoms=args.frozen_atoms,
            config=config,
            mol_getter=GRAPH_METHODS[args.graph]() if args.graph else None,
            auto_fallback=not args.no_fallback,
            active_window=window,
            active_bonds=args.active_bond,
            stratify=args.stratify,
            **window_options,
            pipeline=pipeline,
        )
        if pipeline is not None:
            return ensemble
        if window is not None:
            stem = os.path.splitext(args.output)[0]
            ensemble.write_active_bonds(f"{stem}.active_bonds.csv")
    else:
        ensemble = generate_gs(
            args.smiles,
            charge=args.charge,
            multiplicity=args.multiplicity,
            config=config,
            pipeline=pipeline,
        )
        if pipeline is not None:
            return ensemble
    ensemble.write_xyz(args.output, use_energy=args.crest_energies)
    return ensemble


def _run_swap(parser, args):
    _check_file(parser, args.filename)
    config = _config_from_args(args)
    reacting = args.reacting_atoms or []
    mol = build_mol(
        args.filename,
        args.charge,
        reacting,
        input_smiles=args.smiles,
        auto_fallback=not args.no_fallback,
    )
    if mol is None:
        parser.error(f"No molecule could be built from {args.filename}.")
    change = Swap(
        args.new,
        site=args.site,
        remove_atoms=args.remove,
        center=args.center[0] if args.center else None,
        substructure=args.center[1] if args.center else None,
        old_fragment=args.old,
        attach_map=dict(args.attach) if args.attach else None,
        bond_types={int(n): kind for n, kind in args.bond_type}
        if args.bond_type
        else None,
        mode=args.mode,
    )
    if args.conserve == "hard" and (
        args.crest_energies or args.n_conformers or args.routes
    ):
        parser.error("--conserve hard samples nothing: no -n, --routes or energies.")
    ensemble = swap(
        mol,
        change,
        task=TransitionState(reacting) if reacting else None,
        conserve=args.conserve,
        routes=args.routes,
        hard=args.hard,
        config=config,
        multiplicity=args.multiplicity,
    )
    index_map = json.loads(ensemble.mol.GetProp("swap_index_map"))
    moved = {int(i): k for i, k in index_map.items() if int(i) != k}
    if moved:  # e.g. the -r atoms of a follow-up run
        logging.getLogger(__name__).warning(
            "Atom indices change in the swap (old: new): %s", moved
        )
    ensemble.write_xyz(args.output, use_energy=args.crest_energies)
    return ensemble


def _add_verbose(parser) -> None:
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Log progress to stderr (-v: INFO, -vv: DEBUG).",
    )


def _check_file(parser, path: str) -> None:
    if not os.path.isfile(path):
        parser.error(f"'{path}' does not exist or is not a valid file.")


def _parse_and_run(parser, argv, run):
    """Parse argv, then run(parser, args) with logging to stderr as asked (-v, -vv)."""
    args = parser.parse_args(argv)
    with cli_logging(args.verbose):
        return run(parser, args)


if __name__ == "__main__":
    main()
