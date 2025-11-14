# optimizer

Force-field optimizers for refining TS-constrained conformers.

Constructor parameters:

- `conf_id_ref: int = -1` : reference conformer ID on the TS molecule
- `force_constant: float = 1e6` : force constant of distance constraints to TS reference points

<a id="mmff_optimizer"></a>
### MMFFOptimizer
Optimizes each conformer with MMFF94:
>1. Align to the TS reference conformer (on `align_indices`)
>2. Create RDKit force-field extra points at the TS coordinates for each `align_indices` atom and add distance constraints with `force_constant`
>3. Minimize and set the conformer `energy` property from `ff.CalcEnergy()` in kcal/mol
>4. Re-align to the TS reference

<a id="uff_optimizer"></a>
### UFFOptimizer
Same flow as `MMFF_optimizer` using the RDKit implementation of UFF.


!!! note MMFF vs UFF
    - MMFF: Good general-purpose performance, if MMFF parameters are available.
    - UFF: Use if no MMFF parameters available or as fallback.