"""
The racerts command line.

    racerts ts FILE -r ATOM [ATOM ...] [-s SMILES ...]   TS conformers
    racerts gs SMILES                                    ground-state conformers
    racerts FILE [options]                               legacy (or: racerts run)
"""

import argparse
import os
import sys
from dataclasses import replace

from racerts.api import generate_gs, generate_ts
from racerts.config import PipelineConfig
from racerts.embed import EMBED_MODES
from racerts.embed.stage import COUNT_POLICIES
from racerts.refine import REFINE_BACKENDS
from racerts.refine.forcefield import DIELECTRIC_MODELS
from racerts.system import GRAPH_METHODS
from racerts.utils.log import cli_logging

SUBCOMMANDS = ("ts", "gs")
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

    gs = sub.add_parser(
        "gs",
        help="ground-state conformers from a SMILES",
        description="Ground-state conformers (nothing frozen), embedded with ETKDGv3.",
    )
    gs.add_argument("smiles", help="SMILES of the molecule.")
    gs.add_argument(
        "-c", "--charge", type=int, default=None, help="Total charge (default: SMILES)."
    )

    defaults = PipelineConfig()
    for command in (ts, gs):
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
            "--count-policy",
            choices=list(COUNT_POLICIES),
            help="How the default number of conformers is counted: legacy, fragments "
            "(adds the rigid-body freedom of fragments without frozen atoms) or catmlp "
            f"(default {defaults.embed.count_policy}).",
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
            choices=["legacy", "frozen_first"],
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
    """The config file (or the defaults) with the options given on the command line."""
    config = PipelineConfig.from_file(args.config) if args.config else PipelineConfig()
    dielectric = {}
    if args.dielectric:
        model, constant = args.dielectric
        try:
            dielectric = dict(
                dielectric_model=model, dielectric_constant=float(constant)
            )
        except ValueError:
            raise ValueError(f"--dielectric: {constant!r} is not a number.") from None
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
    else:
        ensemble = generate_gs(
            args.smiles,
            charge=args.charge,
            multiplicity=args.multiplicity,
            config=config,
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
