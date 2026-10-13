"""
The racerts command line.

    racerts ts FILE -r ATOM [ATOM ...] [-s SMILES ...]   TS conformers
    racerts gs SMILES | FILE                             ground-state conformers
    racerts constrained FILE --hard ATOM [ATOM ...]      conformers around held atoms
    racerts FILE [options]                               legacy (or: racerts run)
"""

import argparse
import logging
import os
import sys

from attrs import evolve

from racerts.api import generate, generate_gs, generate_ts
from racerts.config import PipelineConfig
from racerts.embed import EMBED_MODES
from racerts.refine import REFINE_BACKENDS
from racerts.system import GRAPH_METHODS, build_mol
from racerts.task import Constrained
from racerts.utils.log import cli_logging

logger = logging.getLogger(__name__)

SUBCOMMANDS = ("ts", "gs", "constrained")
STRUCTURE_FILES = (".xyz", ".sdf", ".mol")
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
        "--frozen-atoms",
        type=int,
        nargs="+",
        help="Frozen atoms instead of the reacting atoms and their neighbours.",
    )

    constrained = sub.add_parser(
        "constrained",
        help="conformers around atoms held at the geometry of a file",
        description="Conformers of a structure in which the --hard atoms stay where "
        "they are in the file, e.g. a binding motif or a core.",
    )
    constrained.add_argument(
        "filename", help="Structure (.xyz, or .sdf/.mol with bonds)."
    )
    constrained.add_argument(
        "--hard",
        type=int,
        nargs="+",
        required=True,
        help="0-based indices of the atoms held at the geometry of the file.",
    )

    for command, graph in ((ts, "TS topology"), (constrained, "molecule")):
        command.add_argument(
            "-s",
            "--smiles",
            nargs="+",
            help=f"SMILES of the {graph}, one per fragment (recommended for xyz).",
        )
        command.add_argument(
            "--graph",
            choices=list(GRAPH_METHODS),
            help="First method for the molecular graph (default: smiles).",
        )
        command.add_argument(
            "-c", "--charge", type=int, default=0, help="Total charge."
        )

    gs = sub.add_parser(
        "gs",
        help="ground-state conformers from a SMILES or a structure file",
        description="Ground-state conformers (nothing frozen), embedded with ETKDGv3. "
        "From a file, the configuration (stereocentres, double bonds) is that of its "
        "geometry.",
    )
    gs.add_argument(
        "molecule",
        help="SMILES of the molecule, or a structure file (.xyz, or .sdf/.mol with "
        "bonds).",
    )
    gs.add_argument(
        "-s",
        "--smiles",
        nargs="+",
        help="For an xyz file: SMILES of the molecule, one per fragment.",
    )
    gs.add_argument(
        "--graph",
        choices=list(GRAPH_METHODS),
        help="For an xyz file: method for the molecular graph (default: smiles with "
        "-s, then the perceived bonds; connect, the connectivity without bond "
        "orders, only if given here).",
    )
    gs.add_argument(
        "-c",
        "--charge",
        type=int,
        default=None,
        help="Total charge (default: from the SMILES; 0 for a file).",
    )

    defaults = PipelineConfig()
    for command in (ts, gs, constrained):
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
            "--refine",
            choices=list(REFINE_BACKENDS),
            help=f"Force field (default {defaults.refine.backend}).",
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
    """The config file (or the defaults) with the options given on the command line."""
    config = PipelineConfig.from_file(args.config) if args.config else PipelineConfig()
    return _replace(
        config,
        seed=args.seed,
        num_threads=args.num_threads,
        embed=_replace(
            config.embed,
            n_conformers=args.n_conformers,
            conf_factor=args.conf_factor,
            mode=args.embed,
            etkdg=args.etkdg,
        ),
        refine=_replace(
            config.refine,
            backend=args.refine,
            fallback=False if args.no_fallback else None,
        ),
        prune=_replace(
            config.prune,
            energy_threshold=args.energy_threshold,
            rmsd_threshold=args.rmsd_threshold,
        ),
    )


def _replace(obj, **changes):
    """attrs.evolve with the changes that are not None (checked again)."""
    return evolve(
        obj, **{key: value for key, value in changes.items() if value is not None}
    )


def run_subcommand(argv):
    """racerts ts / racerts gs / racerts constrained."""
    return _parse_and_run(_subcommand_parser(), argv, _run_subcommand)


def _run_subcommand(parser, args):
    config = _config_from_args(args)

    if args.command == "ts":
        _check_file(parser, args.filename)
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
        )
    elif args.command == "constrained":
        _check_file(parser, args.filename)
        mol = build_mol(
            args.filename,
            args.charge,
            [],  # no reacting atoms: the graph of the file as it is
            input_smiles=args.smiles,
            mol_getter=GRAPH_METHODS[args.graph]() if args.graph else None,
            auto_fallback=not args.no_fallback,
        )
        if mol is None:
            parser.error(f"No molecule could be built from '{args.filename}'.")
        ensemble = generate(
            mol,
            Constrained(args.hard),
            config=config,
            multiplicity=args.multiplicity,
        )
    else:
        molecule = args.molecule
        if molecule.lower().endswith(STRUCTURE_FILES):
            _check_file(parser, molecule)
            molecule = _molecule_of_file(parser, args)
        ensemble = generate_gs(
            molecule,
            charge=args.charge,
            multiplicity=args.multiplicity,
            config=config,
        )
    ensemble.write_xyz(args.output, use_energy=args.crest_energies)
    return ensemble


def _molecule_of_file(parser, args):
    """
    The molecule of a structure file for a ground-state search. The graph of an xyz
    file comes from the SMILES (-s) if given, then from the perceived bonds, and from
    its connectivity alone only if --graph connect asks for it: without bond orders
    the embedder gets another molecule, and here no frozen atoms hold any of it in
    place (racerts ts and racerts constrained fall back to it with a warning).
    """
    charge = 0 if args.charge is None else args.charge
    methods = [args.graph or ("smiles" if args.smiles else "bonds")]
    if methods == ["smiles"] and not args.no_fallback:
        methods.append("bonds")
    if not args.molecule.lower().endswith(".xyz"):
        methods = methods[:1]  # a file with bonds keeps its graph
    failure = None
    for method in methods:
        if failure is not None:
            logger.warning("The graph from %s failed: %s", methods[0], failure)
        try:
            mol = build_mol(
                args.molecule,
                charge,
                [],  # no reacting atoms: the graph of the file as it is
                input_smiles=args.smiles,
                mol_getter=GRAPH_METHODS[method](),
                auto_fallback=False,
            )
        except Exception as error:  # whatever the method raises, as in build_mol
            failure = error
            continue
        if mol is not None:
            return mol
        failure = "the file could not be read"
    parser.error(
        f"No molecular graph with bond orders from '{args.molecule}' ({failure}). "
        "Give the SMILES (-s) or the charge (-c), use a file with bonds (.sdf, "
        ".mol), or ask for the connectivity alone (--graph connect)."
    )


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
