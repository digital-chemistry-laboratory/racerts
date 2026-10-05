# Staged workflow

The staged workflow refines conformers in levels, from cheap to expensive, and can
search around the best ones at a level before the final ranking.
`racerts.recipes.staged` builds it as a pipeline:

```python
import racerts
from racerts.recipes import Level, staged
from racerts.refine import ASEOptimizer

uma = ASEOptimizer(uma_calculator, method="UMA-s-1p2", fmax=0.02)  # any ASE calculator
pipeline = staged(uma, config=racerts.PipelineConfig(embed={"n_conformers": 100}))
ensemble = racerts.generate_ts("ts.xyz", [3, 4, 5], smiles="CCCCCC=C", pipeline=pipeline)
```

One optimizer stands for the usual two levels. Written out:

```python
pipeline = staged([
    Level(window=25),         # the force field of the settings
    Level(uma, exploit={}),   # window 8 kcal/mol, with a search
])
```

| Step | Stages | Purpose |
| --- | --- | --- |
| 1 | `Embed`, `Validate(Clash(0.5))` | one embedding: its batches mix biased (restraints, hints, active-bond targets) and unbiased settings, and every conformer records its batch; conformers with overlapping heavy atoms are dropped |
| 2, per level | with `pool`: `FamilySelector`; then `Refine`, the gate, `AttackFace` (first level, windowed TSs), with `rank`: the ranking energy, `PruneEnergy` (the window of the level), `PruneRMSD` | the first level removes embedding artifacts and duplicates, with a loose window, since force-field energies should not decide populations; the later levels rank |
| 3, with `exploit` | `Exploit`, `PruneRMSD` | Monte Carlo around the best conformers, with the refinement and the ranking energy of the level ([Exploit](pipeline.md#exploit)) |
| 4 | `PruneEnergy` (6 kcal/mol), optionally `Rescore` | the final ensemble; rescoring (a higher level, the target solvent) restores the populations after the biasing upstream |

A `Level` has:

- **`optimizer`:** the refinement, an optimizer or a `Refine` stage. Without one it is
  the refinement of the default pipeline (MMFF, not converged: it only has to remove
  the artifacts of the embedding). A level with a search needs an `ASEOptimizer` (xTB,
  an MLIP).
- **`window`:** the energy window after the refinement in kcal/mol (default 8).
- **`exploit`:** the settings of `Exploit` as a dict (`{}`: its defaults); without it
  the level has no search.
- **`rank`:** a `Rescore` stage for the energy that the level ranks by, when that is
  not the energy of its refinement (see below).
- **`pool`, `pool_by`:** the most conformers that enter the level (default: all),
  chosen by structural family (`"family"`, the default: the best of each cluster, then
  the second best, ...) or by the energy of the level before (`"energy"`). A cheap
  level often misranks a flexible system, and Exploit searches only around what it
  gets.

The other arguments of `staged`:

- **`embed`:** the `Embed` stage; default: that of the default pipeline.
- **`config`:** the [settings](pipeline.md#settings) of the default embedding and of a
  level without an optimizer, e.g. `PipelineConfig(embed={"n_conformers": 100})`.
- **`final_window`:** the energy window of the result (6 kcal/mol).
- **`rescore`:** a `Rescore` stage at the end.
- **`clash_filter`:** step 1 drops conformers with heavy atoms closer than this times
  their vdW sum (default 0.5; `None`: no filter). Distance geometry may place atoms four
  bonds apart at their lower bound, 0.7 times the vdW sum (0.555 for some), and
  refinement relaxes such contacts, so only overlaps are dropped here; the gate after each
  refinement checks clashes at 0.7. A filter at 0.7 here can drop the conformer that is
  the lowest after refinement.

## A level in between

A force field misranks the conformers of a flexible system, and it does not sample ring
conformations that distance geometry missed. A semiempirical level between the two
helps with both:

```python
from tblite.ase import TBLite

def gfn2_calculator():
    return TBLite(method="GFN2-xTB", verbosity=0)

gfn2 = ASEOptimizer(gfn2_calculator, method="GFN2-xTB", fmax=0.05, max_steps=300,
                    num_workers=None)   # all CPUs
pipeline = staged([
    Level(window=25),
    Level(gfn2, window=15, exploit={"max_optimizations": 400}),
    Level(uma, exploit={}, pool=30),
])
```

- The level refines what passed the force field, with the gate, its energy window and
  the RMSD pruning.
- Its `exploit` runs `Exploit` there: ring flips and torsion moves cost little at this
  level, and the pool for the expensive level then draws on the minima they find.
- An `ASEOptimizer` with a calculator factory and `num_workers` runs in worker processes,
  one single-threaded calculator each. The factory must be a function of a module (a
  lambda cannot be sent to the workers), and a script that starts workers needs the
  `if __name__ == "__main__":` guard. The force-field level has `refine.num_workers` for
  the same.

## The ranking energy

The energy that optimizes the geometries need not be the one that ranks them. A common
case: an MLIP in the gas phase for the geometries, and a solvation term from a cheaper
method on top. The `rank` of a level is a `Rescore` stage that follows its refinement,
also inside its `Exploit`; the window, the duplicates, Exploit's parents and its stop
then use its energies:

```python
from tblite.ase import TBLite
from racerts import Rescore

def alpb(structures):
    """E(GFN2, ALPB) - E(GFN2, gas phase) of each structure, in eV."""
    corrections = []
    for atoms in structures:
        energies = []
        for solvation in (None, ("alpb", "dioxane")):
            atoms.calc = TBLite(method="GFN2-xTB", solvation=solvation, verbosity=0,
                                charge=atoms.info["charge"],
                                multiplicity=atoms.info["multiplicity"])
            energies.append(atoms.get_potential_energy())
        corrections.append(energies[1] - energies[0])
    return corrections

solvated = Rescore(batch=alpb, method="ALPB(dioxane)", add=True)
pipeline = staged([Level(window=25), Level(uma, exploit={}, rank=solvated)])
```

- `Rescore(..., add=True)` adds its single points to the energies (a correction);
  without `add` it replaces them (single points at another level).
- The method of the result is recorded (`UMA-s-1p2+ALPB(dioxane)`), and each conformer
  keeps the energy of the refinement (`previous_energy`) and the correction
  (`energy_correction`) in its provenance.
- Whether a candidate of Exploit is a minimum is still judged with the energies of the
  refiner, the surface it was optimized on.
- Which energy to rank by is a question for a benchmark against the level of the final
  energies; the correction can reorder conformers by several kcal/mol.
- Free energies: `Rescore` asks a calculator for energies only, so one that returns a
  free energy (the energy plus a thermal correction from a Hessian of its own) ranks
  by free energy without further code, e.g.
  `Rescore(thermal_correction, method="G(GFN2-xTB)", add=True)` at the end of a
  pipeline. It belongs after a free refinement to tight forces (about 0.01 eV/Å; at the
  default 0.05 the corrections are off by as much as they differ between conformers)
  and costs a Hessian per conformer. The energy windows before it should be the wanted
  free-energy window plus about 2 kcal/mol: inside an ensemble the thermal correction
  varies by 1 to 2 kcal/mol.

## The gate

After each refinement, `racerts.validate.gate()` drops the conformers that are no longer
valid:

- frozen atoms that moved by more than 0.1 Å (`FrozenCore`; MMFF/UFF leave them up to
  0.09 Å from the reference);
- changed bonds or stereo (`Connectivity`);
- clashing heavy atoms (`Clash`);
- restraints broken by more than 0.5 Å (`RestraintViolation`).

A conformer that fails is dropped, not reset to its input geometry. Its input is no
minimum at the new level, and a structure that breaks at the expensive level often has
no basin there. The log counts the failures per check, and warns when more than 30 %
fail: that points to a wrong charge, restraint or hypothesis rather than to single bad
conformers.

## Restraints at the expensive level

ASE calculators take the restraints like MMFF/UFF do: `ASEOptimizer` adds flat-bottom
terms for distance windows and soft atoms to its calculator, and the energies it stores
leave them out.

## Transition states

The active-bond windows of a TS act in embedding only:

- **Refinement:** every refinement holds each conformer's active bonds at its embedded
  length or target ([how closely](active-bonds.md)). `Exploit` keeps each child's
  targets.
- **Comparing energies:** energies at different held lengths are not comparable, so the
  energy windows and the RMSD pruning work per window bin.
- **Output:** the result is a set of TS-like structures for a TS optimization (next
  section).

## From TS-like conformers to transition states

`racerts.recipes.saddles` is the saddle search with its checks, as a pipeline for the
result of `staged` (or of the default pipeline). It runs in a context of the reference
molecule, the one the TS-like conformers were made from: its geometry is what the
checks compare with (a context made from the ensemble would take the first conformer
of the ensemble for the reference).

```python
from sella import Sella
from racerts.recipes import saddles
from racerts.system import build_mol

mol = build_mol("ts.xyz", 0, [3, 4, 5], input_smiles=["CCCCCC=C"])  # the reference
task = racerts.TransitionState([3, 4, 5])
ts_like = racerts.generate(mol, task, pipeline=staged(uma))

search = ASEOptimizer(uma_calculator, optimizer_cls=Sella,
                      optimizer_kwargs={"order": 1, "internal": True},
                      fmax=0.01, max_steps=300, method="UMA-s-1p2")
ctx = racerts.Context.create(mol, task)
transition_states = saddles(search, pool=20).run(ctx, ts_like)
```

| Step | Stages | Purpose |
| --- | --- | --- |
| 1 | with `pool`: `FamilySelector` | at most that many searches, spread over structural families (`pool_by="energy"`: the lowest) |
| 2 | `Refine(search, anchors=False)` | the free saddle search: the frozen atoms and the windows of the task are released |
| 3 | `Validate(Converged(), ReactionMode(calculator), ReactionCore(), Connectivity())` | keep the transition states of the reaction (below) |
| 4 | `PruneRMSD` | several TS-like conformers can end in one saddle point |

A converged search with one imaginary mode is not yet a TS of the reaction:

- `Converged`: a search that stopped at its step limit is no stationary point.
- `ReactionMode`: exactly one imaginary mode, and it moves an active bond (the stretch
  of the bond along the mode, for a displacement of unit length over all atoms: 1.41 for
  two atoms alone, about 0 for a rotor elsewhere; at least `min_stretch`, 0.3). The
  provenance records `imaginary_frequency` and `mode_stretch`.
- `ReactionCore`: the distances between the reacting atoms are those of the reference
  within 0.5 Å; another saddle of the same atoms (a proton already transferred) fails.
- `Connectivity`: the bonds and the stereo of the graph outside the reacting atoms.

The Hessians are finite differences of the forces of `calculator` (default: that of the
search), 6 N force calls per conformer. Two arguments take code from outside:

- `hessian=`: a function of ASE `Atoms` that returns the Hessian (eV/Å², 3N × 3N)
  instead, e.g. an analytical Hessian of the calculator or one by automatic
  differentiation: `saddles(search, hessian=my_hessian)`. The same argument exists on
  `ImaginaryModes` and `ReactionMode`.
- `checks=`: further validators after the four above, e.g. one that follows the
  imaginary mode downhill to both sides and compares the two minima with the expected
  ones (a function per conformer becomes a validator with `racerts.validate.validator`).

In Cartesian coordinates a search can leave a symmetric start for another saddle point;
internal coordinates (`"internal": True` for Sella) are more robust, and the checks drop
what went wrong.

## Cost

- **Calculators on a GPU:** pass a calculator instance and `num_workers=1`. A calculator
  factory makes a new calculator for every structure when it runs without worker
  processes.
- **Budget:** `Exploit` stops when it rarely finds new minima, and at the latest after
  `max_optimizations`. The stage reports its numbers in `stats`
  (`next(s for s in pipeline.stages if s.name == "exploit").stats`).
