# Plug-in points

Calculators, optimizers and checks from other packages go into the stages unchanged:

- **Calculators:** `ASEOptimizer(calculator=...)` and `Rescore(calculator)` take an ASE
  calculator or a callable that returns one (a factory: one calculator per worker
  process, or per conformer without workers). Wrapping calculators work as they are,
  e.g. a bias potential such as AFIR.
- **Optimizers:** `ASEOptimizer(optimizer_cls=..., optimizer_kwargs=...)` takes any class
  with the ASE optimizer interface, e.g. `sella.Sella` with `{"order": 1}` for saddle
  points. Every conformer records `converged`, `n_steps` and `wall_time` in its
  provenance; `drop_unconverged=True` removes the unconverged ones.
- **A relaxation of another package:** a subclass of `ASEOptimizer` that overrides
  `_relax(tasks, config, reference, num_workers)` hands the structures (ASE `Atoms`
  with charge, multiplicity and the frozen atoms as constraints) to any code that
  relaxes them, e.g. an optimizer that takes several structures at once on a GPU, and
  returns a `racerts.refine.Outcome` for each. Everything around that step stays:
  the conversion from and to conformers, the energies, the provenance, the failures,
  and `Refine`, `Exploit` and the staged recipe take the optimizer as any
  `ASEOptimizer`. An override that only brings its own calculators, or runs several
  structures at a time, calls `racerts.refine.optimize_one(calculator, config, task)`
  for each: the relaxation of one structure as racerts does it, restraints
  included. An optimizer that shares nothing with it subclasses `BaseOptimizer`
  ([components](pipeline.md)).
- **Warm-up:** `prepare(calculator, reference_atoms)` is called once for every
  calculator before its first conformer, with the reference geometry (e.g. for
  calculators that take their topology from the first geometry they see).
- **Batch energies:** `Rescore(batch=fn)`, with `fn(list_of_atoms)` returning energies
  in eV.
- **Validators:** any object with a `name` and `validate(ctx, ensemble)`, returning the
  reason for each conformer that fails; `racerts.validate.validator(fn)` turns a
  function `fn(mol, conf_id)` into one. The validators of one `Validate` stage need
  distinct names.

Built-in validators (`racerts.validate`):

- `Connectivity()`: the bonds perceived from the geometry and the specified stereo are
  those of the graph (bonds between reacting atoms exempt); `IdentityFilter()` drops
  the conformers that fail it.
- `FrozenCore(tolerance)`: the frozen atoms are at the reference.
- `ImaginaryModes(calculator, expected=1)`: finite-difference frequencies, for
  stationary points.
- `ReactionMode(calculator)`: one imaginary mode that moves an active bond of the TS
  (a rotor of a loose complex is a first-order saddle point too), and `Converged()`:
  see [the saddle search](workflow.md#from-ts-like-conformers-to-transition-states).
- `ReactionCore(tolerance=0.5)`: the distances between the reacting atoms are those of
  the reference TS within tolerance (Å). After a free saddle search it tells a TS of the
  reaction from other saddles of the same atoms, which pass the two checks above.
- `AttackFace()`: for [active-bond windows](active-bonds.md).
- `Clash(factor=0.7)`: no heavy atoms more than three bonds apart (or in different
  fragments) closer than factor × their vdW sum. Pairs of hard and core atoms keep the
  reference geometry (e.g. a forming bond) and are not checked; pairs that are close in
  the reference count only if they come 0.2 Å closer.
- `RestraintViolation(tolerance=0.5)`: no restraint of the refinement (distance windows,
  soft atoms) violated by more than tolerance (Å): a contact that broke, not the small
  excess that flat-bottom terms allow.

`racerts.validate.gate()` combines `FrozenCore`, `Connectivity`, `Clash` and
`RestraintViolation` into the validity gate after a refinement: it drops the conformers
that fail and warns when more than 30 % fail (`Validate(..., warn_above=0.3)`), which
points to a wrong charge, restraint or hypothesis rather than to single bad conformers.
Its `FrozenCore` allows 0.1 Å: MMFF and UFF hold the frozen atoms with stiff springs,
which leave them a few hundredths of an Å from the reference.

From TS-like conformers to transition states with GFN2-xTB (tblite) and Sella:

```python
from sella import Sella
from tblite.ase import TBLite
from racerts.refine import ASEOptimizer
from racerts.validate import ImaginaryModes, ReactionCore

def gfn2():
    return TBLite(method="GFN2-xTB", verbosity=0)

saddle = ASEOptimizer(gfn2, optimizer_cls=Sella, optimizer_kwargs={"order": 1}, fmax=0.005)
pipeline = racerts.Pipeline([
    racerts.Embed(),
    racerts.Refine(),
    racerts.PruneEnergy(),
    racerts.PruneRMSD(),
    racerts.Rescore(gfn2, method="GFN2-xTB"),
    racerts.PruneCount(5),
    racerts.Refine(saddle, anchors=False, fallback=False),
    racerts.Validate(ImaginaryModes(gfn2, expected=1), ReactionCore()),
])
ensemble = racerts.generate_ts("sn2.xyz", [0, 1, 2], charge=-1, smiles="CCl.[Cl-]",
                               pipeline=pipeline)
```
