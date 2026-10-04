"""The legacy racerts command line: racerts FILE [options] (also racerts run FILE ...)."""

import argparse

from racerts.cli import DEFAULT_OUTPUT, _add_verbose, _check_file, _parse_and_run

from .conformer_generator import DEFAULT_CONF_FACTOR, ConformerGenerator
from .embedder import embedders
from .mol_getter import mol_getters
from .optimizer import optimizers
from .pruner import EnergyPruner, RMSDPruner


def main(argv=None) -> None:
    """The legacy racerts command line (argv: default sys.argv[1:])."""
    parser = argparse.ArgumentParser(
        prog="racerts",
        description="Rapid conformer ensemble generation for transition states. "
        "This is the legacy racerts command line (also `racerts run ...`); see "
        "`racerts ts -h`, `racerts gs -h` and `racerts swap -h` for the subcommands.",
        epilog="Remember to cite the racerts paper :)",
    )

    ##### ---- Input -----

    parser.add_argument(
        "filename",
        type=str,
        help="xyz or sdf/mol filename containing a single transition state conformer.",
    )
    parser.add_argument(
        "-c",
        "--charge",
        type=int,
        required=False,
        default=0,
        help="Molecule total charge.",
    )
    parser.add_argument(
        "-mult",
        "--multiplicity",
        type=int,
        default=None,
        help="Spin multiplicity 2S+1 (default: the lowest for the number of electrons).",
    )

    parser.add_argument(
        "-atoms",
        "--reacting_atoms",
        type=int,
        nargs="+",
        required=False,
        default=[],
        help="List of reacting atom indices (e.g., --reacting_atoms 1 2 3).",
    )
    parser.add_argument(
        "-frozen",
        "--frozen_atoms",
        type=int,
        nargs="+",
        default=[],
        help="List of frozen atom indices (e.g., --frozen_atoms 4 5 6).",
    )
    parser.add_argument(
        "-smiles",
        "--input_smiles",
        type=str,
        nargs="+",
        default=[],
        help="List of input SMILES strings (e.g., --input_smiles 'C1=CC=CC=C1' 'O=C=O').",
    )

    parser.add_argument(
        "-n",
        "--number_of_conformers",
        type=int,
        default=-1,
        help="Number of conformers to generate (default: -1).",
    )
    parser.add_argument(
        "-cf",
        "--conf_factor",
        type=int,
        default=DEFAULT_CONF_FACTOR,
        help=f"Conformer factor (default: {DEFAULT_CONF_FACTOR}).",
    )

    # ------ Conformer generation setup -------

    parser.add_argument(
        "-m",
        "--mol",
        action="store",
        choices=["smiles", "bonds", "connect"],
        required=False,
        help="RDKit method to create the mol object: 'smiles', 'bonds' 'connect'. The default is to use the SMILES template, if available, else try to use determineBonds ('bonds') or as last option determineConnectivity ('connect').",
    )
    parser.add_argument(
        "-e",
        "--embed",
        action="store",
        choices=["dm", "cmap"],
        required=False,
        help="Method for the RDKit-based embedding step: either using a distance matrix ('dm') or a coordinate mapping ('cmap'), as the default.",
    )
    parser.add_argument(
        "-ff",
        "--ff",
        action="store",
        choices=["mmff", "uff"],
        required=False,
        help="Method for the RDKit-based force-field optimization step: either MMFF ('mmff') or UFF ('uff'). The default tries the MMFF and at failure uses the UFF.",
    )
    parser.add_argument(
        "-o",
        "--output",
        action="store",
        type=str,
        required=False,
        help="Output filename.",
    )
    parser.add_argument(
        "--out_energies",
        action="store_true",
        required=False,
        help="Write only the energy (in Hartree) on each comment line, as in CREST ensembles, instead of the default extended XYZ line with energy (eV), charge and spin.",
    )
    _add_verbose(parser)

    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of threads to use (default 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12,
        help="Random seed for reproducibility (default 12).",
    )
    parser.add_argument(
        "--no_fallback",
        action="store_false",
        dest="fallback",
        default=True,
        help="Disable fallback to other methods (default: enabled).",
    )

    # ------ Mol Getter setup -------

    parser.add_argument(
        "--no_assignbonds",
        action="store_false",
        dest="assignBonds",
        default=True,
        help="Flag to control if bonds should be assigned when using the 'bonds' mol getter (default True).",
    )
    parser.add_argument(
        "--disallow_charged_fragments",
        action="store_false",
        dest="allowChargedFragments",
        default=True,
        help="Flag to control if charged fragments are allowed when using the 'bonds' mol getter (default True).",
    )

    # ------ Embedder setup -------

    parser.add_argument(
        "--no_random_coords",
        action="store_false",
        dest="useRandomCoords",
        default=True,
        help="Disable the use of random coordinates if embedding fails (default: enabled).",
    )

    # ------ Optimizer setup -------
    parser.add_argument(
        "--force_constant",
        type=float,
        default=1e6,
        help="Force constant for the distance constraints during the force-field optimization (default 1e6).",
    )
    # ------ Pruner setup -------

    parser.add_argument(
        "-rmsd",
        "--rmsd_thres",
        type=float,
        default=0.125,
        help="Threshold for rmsd pruning (default 0.125 A).",
    )
    parser.add_argument(
        "-rmsd_hs",
        "--rmsd_include_hs",
        action="store_true",
        help="Flag to control if hydrogen atoms are included for RMSD calculations (default false).",
    )

    parser.add_argument(
        "--no_filter_energies",
        action="store_false",
        default=True,
        help="Disable energy filtering during RMSD pruning (default: enabled).",
    )
    parser.add_argument(
        "--no_filter_rotations",
        action="store_false",
        default=True,
        help="Disable rotatable bond filtering during RMSD pruning (default: enabled).",
    )
    parser.add_argument(
        "--rmsd_energy",
        type=float,
        default=0.1,
        help="Energy threshold in kcal/mol for RMSD pruning to determine if two structures are likely different (default 0.1 kcal/mol).",
    )
    parser.add_argument(
        "--rmsd_rot_fraction",
        type=float,
        default=0.03,
        help="Rotatable bond fraction threshold for RMSD pruning to determine if two structures are likely different (default 0.03).",
    )
    parser.add_argument(
        "-rmsd_match",
        "--rmsd_max_matches",
        type=int,
        default=10000,
        help="Maximum number of substructure matches to consider for calculating RMSDs (default 10000).",
    )

    # ------ Energy Pruner setup -------

    parser.add_argument(
        "-energy",
        "--energy_thres",
        type=float,
        default=20.0,
        help="Energy threshold in kcal/mol for pruning (default 20.0 kcal/mol).",
    )

    ##### ---- Parse args -----

    _parse_and_run(parser, argv, _run)


def _run(parser, args):
    verbose = bool(args.verbose)

    cg = ConformerGenerator(
        verbose=verbose,
        randomSeed=args.seed,
        num_threads=args.num_threads,
    )

    cg.energy_pruner = EnergyPruner(
        threshold=args.energy_thres,
        verbose=verbose,
    )
    cg.rmsd_pruner = RMSDPruner(
        threshold=args.rmsd_thres,
        verbose=verbose,
        num_threads=args.num_threads,
        include_hs=args.rmsd_include_hs,
        filter_energies=args.no_filter_energies,
        filter_rotations=args.no_filter_rotations,
        energy_threshold=args.rmsd_energy,
        rot_fraction_threshold=args.rmsd_rot_fraction,
        maxMatches=args.rmsd_max_matches,
    )

    if args.mol:
        cg.mol_getter = mol_getters[args.mol](
            assignBonds=getattr(args, "assignBonds", True),
            allowChargedFragments=getattr(args, "allowChargedFragments", True),
        )

    if args.embed:
        cg.embedder = embedders[args.embed](
            verbose=verbose,
            num_threads=args.num_threads,
            randomSeed=args.seed,
            useRandomCoords=args.useRandomCoords,
        )
    else:  # keep the default embedder, but apply its flag
        cg.embedder.useRandomCoords = args.useRandomCoords

    if args.ff:
        cg.optimizer = optimizers[args.ff](
            verbose=verbose,
            num_threads=args.num_threads,
            force_constant=args.force_constant,
        )
    else:  # keep the default optimizer, but apply its flag
        cg.optimizer.force_constant = args.force_constant

    _check_file(parser, args.filename)
    cg.generate_conformers(
        args.filename,
        args.charge,
        args.reacting_atoms,
        frozen_atoms=args.frozen_atoms,
        input_smiles=args.input_smiles,
        number_of_conformers=args.number_of_conformers,
        conf_factor=args.conf_factor,
        auto_fallback=args.fallback,
        multiplicity=args.multiplicity,
    )
    cg.write_xyz(args.output or DEFAULT_OUTPUT, use_energy=args.out_energies)
