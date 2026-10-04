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

from racerts.api import generate_gs, generate_ts, swap
from racerts.config import PipelineConfig
from racerts.embed import CHIRALITY_FALLBACK_MODES, COUNT_POLICIES, EMBED_MODES
from racerts.refine import REFINE_BACKENDS
from racerts.refine.forcefield import DIELECTRIC_MODELS
from racerts.system import GRAPH_METHODS, build_mol
from racerts.system.swap import Swap, SwapError
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
    parser = argparse.ArgumentParser(
        prog="racerts",
        description="Conformer ensembles with a frozen core. Without a subcommand, "
        "racerts runs the legacy command line (`racerts FILE -h`).",
        epilog="Remember to cite the racerts paper :)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ts = sub.add_parser(
        "ts",
        help="TS conformers from a TS geometry",
        description="TS conformers: the reacting atoms and their neighbours stay at "
        "the TS geometry. With default settings, the same ensemble as legacy racerts.",
    )
    ts.add_argument("filename", help="TS geometry (.xyz, or .sdf/.mol with bonds).")
    ts.add_argument(
        "-r",
        "--reacting-atoms",
        type=int,
        nargs="+",
        required=True,
        help="0-based indices of the atoms whose bonds form or break.",
    )
    ts.add_argument(
        "-s",
        "--smiles",
        nargs="+",
        help="SMILES of the TS topology, one per fragment (recommended for xyz).",
    )
    ts.add_argument(
        "--frozen-atoms",
        type=int,
        nargs="+",
        help="Frozen atoms instead of the reacting atoms and their neighbours.",
    )
    ts.add_argument(
        "--graph",
        choices=list(GRAPH_METHODS),
        help="First method for the molecular graph (default: smiles).",
    )
    ts.add_argument("-c", "--charge", type=int, default=0, help="Total charge.")
    ts.add_argument(
        "--active-window",
        type=float,
        nargs="+",
        metavar="A",
        help="Sample the forming bonds in a window: DELTA (seed length +/- DELTA) or "
        "LO HI (A); writes active_bonds.csv next to the output.",
    )
    ts.add_argument(
        "--active-bond",
        type=int,
        nargs=2,
        action="append",
        metavar=("I", "J"),
        help="An active bond (repeatable; default: the forming bonds of the seed).",
    )
    ts.add_argument(
        "--stratify",
        type=int,
        default=0,
        metavar="K",
        help="K target lengths evenly spaced in the window (default: anywhere).",
    )
    ts.add_argument(
        "--neighbor-window",
        type=float,
        default=0.10,
        help="+/- A for the reacting atoms to their neighbours (default 0.1).",
    )

    gs = sub.add_parser(
        "gs",
        help="ground-state conformers from a SMILES",
        description="Ground-state conformers (nothing frozen), embedded with ETKDGv3.",
    )
    gs.add_argument("smiles", help="SMILES of the molecule.")
    gs.add_argument(
        "-c", "--charge", type=int, default=None, help="Total charge (default: SMILES)."
    )

    swap = sub.add_parser(
        "swap",
        help="conformers after replacing a group of a reference geometry",
        description="Replace a group of the reference by a new fragment and sample "
        "the new atoms around the kept geometry (see racerts.swap).",
    )
    swap.add_argument("filename", help="Reference geometry (.xyz, or .sdf/.mol).")
    swap.add_argument(
        "--new",
        required=True,
        help="The new fragment: SMILES with a dummy per attachment ([*] or [*:1]).",
    )
    where = swap.add_mutually_exclusive_group(required=True)
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
    swap.add_argument(
        "--attach",
        type=int,
        nargs=2,
        action="append",
        metavar=("DUMMY", "ATOM"),
        help="Dummy number and the atom it binds to (repeatable).",
    )
    swap.add_argument(
        "--bond-type",
        nargs=2,
        action="append",
        metavar=("DUMMY", "TYPE"),
        help="single, double, triple or dative (from the fragment atom).",
    )
    swap.add_argument(
        "--mode",
        choices=["append", "renumber"],
        default="append",
        help="append (default): new atoms take the slots of removed ones, the rest are "
        "appended; renumber: kept atoms first. Changed indices are logged.",
    )
    swap.add_argument(
        "-s", "--smiles", nargs="+", help="SMILES of the reference, one per fragment."
    )
    swap.add_argument("-c", "--charge", type=int, default=0, help="Total charge.")
    swap.add_argument(
        "-r",
        "--reacting-atoms",
        type=int,
        nargs="+",
        help="For a TS reference: its reacting atoms (they and their neighbours stay).",
    )
    swap.add_argument(
        "--conserve",
        choices=["hard", "soft", "free"],
        default="soft",
        help="hard: graft only; soft: kept atoms restrained near the reference "
        "(default); free: only the TS core held.",
    )
    swap.add_argument(
        "--routes",
        nargs="+",
        choices=["dg", "rigid"],
        default=None,
        help="Distance geometry and/or rigid poses of the fragment (default dg).",
    )
    swap.add_argument(
        "--hard", type=int, nargs="+", default=[], help="Further atoms held fixed."
    )

    defaults = PipelineConfig()
    for command in (ts, gs, swap):
        command.add_argument(
            "--multiplicity",
            type=int,
            help="Spin multiplicity 2S+1 (default: the lowest for the electrons).",
        )
        command.add_argument(
            "--config",
            help="PipelineConfig as JSON or YAML; options below override it.",
        )
        command.add_argument(
            "--legacy",
            action="store_true",
            help="The settings of legacy racerts where the defaults changed since "
            "(--config and the options below override them).",
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
            "--embed",
            choices=list(EMBED_MODES),
            help=f"Embedding mode (default {defaults.embed.mode}).",
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
            "--chirality-fallback",
            choices=list(CHIRALITY_FALLBACK_MODES),
            help="When the frozen atoms contradict chiral tags: drop all tags (legacy) "
            "or those of the frozen atoms first (default "
            f"{defaults.embed.chirality_fallback}).",
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
            help="Fail instead of falling back (graph: bonds, connectivity; "
            "MMFF: UFF).",
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
            "--restraint",
            nargs=3,
            action="append",
            metavar=("I", "J", "DISTANCE"),
            help="Keep atoms I and J at DISTANCE (A) +/- the half width (repeatable).",
        )
        command.add_argument(
            "--keep-hbonds",
            action="store_true",
            default=None,
            help="Keep the hydrogen bonds of the input geometry (ts).",
        )
        command.add_argument(
            "--contact",
            nargs=2,
            type=int,
            action="append",
            metavar=("I", "J"),
            help="Keep the non-covalent contact I...J as in the input geometry (ts).",
        )
        command.add_argument(
            "--keep-fragments",
            action="store_true",
            default=None,
            help="Keep fragments without reacting atoms (e.g. solvent) at the core (ts).",
        )
        command.add_argument(
            "--link-fragments",
            action="store_true",
            default=None,
            help="Embed the fragments of a complex together (gs).",
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
    return parser


def _config_from_args(args) -> PipelineConfig:
    """
    The defaults (or those of legacy racerts), updated by the config file and by the
    options given on the command line.
    """
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
    if args.restraint:
        try:
            user = [[int(i), int(j), float(d)] for i, j, d in args.restraint]
        except ValueError:
            raise ValueError(
                f"--restraint takes two atom indices and a distance: {args.restraint}"
            ) from None
    return _replace(
        config,
        seed=args.seed,
        restraints=_replace(
            config.restraints,
            user=user,
            hbonds=args.keep_hbonds,
            contacts=[list(pair) for pair in args.contact] if args.contact else None,
            keep_fragments=args.keep_fragments,
            link_fragments=args.link_fragments,
            hints=args.hints,
            half_width=args.restraint_half_width,
            force_constant=args.restraint_force_constant,
        ),
        num_threads=args.num_threads,
        embed=_replace(
            config.embed,
            n_conformers=args.n_conformers,
            conf_factor=args.conf_factor,
            mode=args.embed,
            etkdg=args.etkdg,
            count_policy=args.count_policy,
            sequential_seeds=args.sequential_seeds,
            chirality_fallback=args.chirality_fallback,
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
    """racerts ts / racerts gs."""
    return _parse_and_run(_subcommand_parser(), argv, _run_subcommand)


def _run_subcommand(parser, args):
    if args.command == "swap":
        return _run_swap(parser, args)
    config = _config_from_args(args)

    if args.command == "ts":
        _check_file(parser, args.filename)
        window = args.active_window
        if window is None and (args.active_bond or args.stratify):
            parser.error("--active-bond and --stratify need --active-window.")
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
            neighbor_window=args.neighbor_window,
        )
        if window is not None:
            write_active_bonds(
                ensemble, os.path.join(os.path.dirname(args.output), "active_bonds.csv")
            )
    else:
        ensemble = generate_gs(
            args.smiles,
            charge=args.charge,
            multiplicity=args.multiplicity,
            config=config,
        )
    ensemble.write_xyz(args.output, use_energy=args.crest_energies)
    return ensemble


SWAP_UNUSED = (
    "restraint", "keep_hbonds", "contact", "keep_fragments", "link_fragments",
    "hints", "restraint_half_width", "restraint_force_constant", "embed",
    "chirality_fallback",
)  # fmt: skip


def _run_swap(parser, args):
    unused = [f"--{n.replace('_', '-')}" for n in SWAP_UNUSED if getattr(args, n)]
    if unused:
        parser.error(f"racerts swap does not take {', '.join(unused)}.")
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
    try:
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
    except ValueError as error:
        parser.error(str(error))
    if args.conserve == "hard" and (
        args.crest_energies or args.n_conformers or args.routes
    ):
        parser.error("--conserve hard samples nothing: no -n, --routes or energies.")
    try:  # settings the swap does not take, e.g. from a config file
        ensemble = swap(
            mol,
            change,
            task=TransitionState(reacting) if reacting else None,
            conserve=args.conserve,
            n_conformers=config.embed.n_conformers,
            routes=args.routes,
            hard=args.hard,
            config=config,
            multiplicity=args.multiplicity,
        )
    except SwapError as error:  # the input, not a failure of the run
        parser.error(str(error))
    index_map = json.loads(ensemble.mol.GetProp("swap_index_map"))
    moved = {int(i): k for i, k in index_map.items() if int(i) != k}
    if moved:  # e.g. the -r atoms of a follow-up run
        logging.getLogger(__name__).warning(
            "Atom indices change in the swap (old: new): %s", moved
        )
    ensemble.write_xyz(args.output, use_energy=args.crest_energies)
    return ensemble


def write_active_bonds(ensemble, path: str) -> None:
    """A CSV: conformer id, energy (kcal/mol), target and length of each active bond."""
    bonds, rows = [], []
    for conf_id in ensemble.conf_ids:
        provenance = ensemble.provenance(conf_id)
        lengths = provenance.get("active_bond_lengths", {})
        targets = provenance.get("active_bond_targets", {})
        bonds = bonds or sorted(lengths)
        energy = ensemble.energy(conf_id)
        cells = [str(conf_id), "" if energy is None else f"{energy:.4f}"]
        for bond in bonds:
            target = targets.get(bond)
            cells += ["" if target is None else f"{target:.4f}", f"{lengths[bond]:.4f}"]
        rows.append(",".join(cells))
    header = ["conf_id", "energy_kcal_mol"]
    for bond in bonds:
        header += [f"target_{bond}", f"length_{bond}"]
    with open(path, "w") as handle:
        handle.write("\n".join([",".join(header), *rows]) + "\n")


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
