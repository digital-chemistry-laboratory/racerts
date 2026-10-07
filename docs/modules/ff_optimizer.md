# Optimizers (`racerts.refine`)

Optimizers for refining TS-constrained conformers.

Constructor parameters:

- `conf_id_ref: int = -1` : reference conformer ID on the TS molecule
- `force_constant: float = 1e6` : force constant of distance constraints to TS reference points

Further settings (`converge`, `energies_without_anchors`, the MMFF dielectric) are described
with the [settings](../settings.md).

<a id="mmff_optimizer"></a>
### MMFFOptimizer
Optimizes each conformer with MMFF94:
>1. Align to the TS reference conformer (on `align_indices`)
>2. Create RDKit force-field extra points at the TS coordinates for each `align_indices` atom and add distance constraints with `force_constant`
>3. Minimize and set the conformer `energy` property from `ff.CalcEnergy()` in kcal/mol
>4. Re-align to the TS reference

<a id="uff_optimizer"></a>
### UFFOptimizer
Same flow as `MMFFOptimizer` using the RDKit implementation of UFF.


!!! note "MMFF vs UFF"
    - MMFF: Good general-purpose performance, if MMFF parameters are available.
    - UFF: Use if no MMFF parameters available or as fallback.

<a id="ase_optimizer"></a>
### ASEOptimizer (optional dependency)

`ASEOptimizer` uses the [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/):
>1. Align to the TS reference conformer on `align_indices`
>2. Convert each RDKit conformer to ASE `Atoms`
>3. Apply `FixAtoms(indices=align_indices)` to freeze these atoms
>4. Optimize with an ASE optimizer (`BFGS` by default) and the provided ASE calculator (`calculator=...`, instance or callable)
>5. Write positions back to RDKit and set conformer `energy` in kcal/mol (`eV * 23.06054783061903`)
>6. Re-align to the TS reference

Charge and multiplicity are the `charge` and `multiplicity` arguments, else the values stored on the molecule by [`generate_conformers`](./conformer_generator.md#generate_conformers), else the sum of the formal charges and the lowest multiplicity for the number of electrons. They reach the calculator in two forms: in `atoms.info` (`charge`, `spin` = multiplicity, as read by UMA/fairchem) and as initial charges and magnetic moments (whose sums tblite and xtb-python read). Calculators that take charge and multiplicity as their own arguments (e.g. ASE's `ORCA(charge=..., mult=...)`, PySCF) need them passed there.
A conformer whose calculation fails (e.g. SCF not converged) is left without an energy and is dropped when pruning; if all conformers fail, an error is raised.

!!! note "Index convention"
    `align_indices` follow standard Python indexing (0-based), consistent with RDKit and ASE atom indices.

Install ASE support with:

```bash
pip install "racerts[ase]"
pip install "racerts[xtb]"  # also tblite, for GFN2-xTB
```

Example with a calculator instance:

```python
from ase.calculators.lj import LennardJones
from racerts.optimizer import ASEOptimizer

optimizer = ASEOptimizer(
    calculator=LennardJones(),
    fmax=0.05,
    max_steps=100,
)
```

Example with a calculator callable:

```python
from ase.calculators.lj import LennardJones
from racerts.optimizer import ASEOptimizer

optimizer = ASEOptimizer(
    calculator=LennardJones,
    num_workers=4,
)
```
