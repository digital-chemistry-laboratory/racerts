# Active-bond windows

TS conformers of ionic or organocatalytic reactions do not all share the forming-bond
lengths of the seed. With `active_window`, racerts samples them in a window instead of
keeping the seed lengths (legacy racerts), and records them for the TS optimization that
follows (e.g. a constrained optimization at the recorded length, then the saddle search).

```python
ensemble = racerts.generate_ts(
    "aldol_ts.xyz", [0, 10, 11, 12, 19], smiles=["OC(=O)[C@@H]1CCCN1C(C)=C", "O=Cc1ccccc1"],
    active_window=(2.0, 2.9), active_bonds=[(10, 12)], stratify=5,
)
print(ensemble.summary())  # ...; active bond 10-12: min 2.04, median 2.48, max 2.92 A
```

`racerts ts ... --active-window 2.0 2.9 --active-bond 10 12 --stratify 5` does the same
and writes `active_bonds.csv` (conformer, energy, target and length of each active bond).

What changes with a window (`TransitionState(..., active_window=...)`):
- **Active bonds:** by default the forming bonds, i.e. pairs of reacting atoms that the
  graph does not bond and that are closer than 1.6 × the sum of their covalent radii,
  except 1,3-pairs at an angle of 80° or more (e.g. ring atoms 2.4 Å apart; a narrow
  angle is a three-membered TS such as a reductive elimination) (for `TransitionState.from_endpoints`: the bonds that form or
  break); or `active_bonds`. Without any, window mode raises.
- **Windows:** a number `d` gives the seed length ± d; `(lo, hi)` gives absolute bounds
  for every active bond. A window that reaches below 0.9 × the covalent bond length is
  warned about; one that the TS core cannot take raises (use a narrower window).
- **What stays fixed:** only the neighbours of the reacting atoms that are not reacting
  themselves stay at the seed geometry. The reacting atoms are placed by windows:
  - the active bonds;
  - their bonded neighbours at ± `neighbor_window` (0.1 Å);
  - an embedding-only window for their other distances in the core, which RDKit's default
    (van der Waals) bounds would otherwise forbid, e.g. O···O = 2.5 Å in a proton
    transfer.
- **Sampling:** with `stratify=0`, distance geometry draws the lengths in the window;
  with `stratify=k`, the embedding batches cycle through k target lengths evenly spaced
  in the window.
- **Refinement:** MMFF/UFF holds each conformer at its target (stratified) or at its
  embedded length (uniform), within ± 0.02 Å (k = 10⁴ kcal/(mol Å²)). A flat-bottom
  window would not do: the force field's repulsion between the unbonded atoms pushes
  every conformer to the upper edge. ASE calculators take no restraints, so
  `Refine(ASEOptimizer(...))` with anchors raises in window mode; refine with MMFF/UFF
  first, or search the saddle point freely (`Refine(optimizer, anchors=False)`).
- **Pruning:** the energy window and the duplicate RMSD apply per target (stratified) or
  per fifth of the window (uniform). Energies at different constrained lengths are not
  comparable: on the aldol TS, MMFF puts 2.0 Å about 66 kcal/mol above 2.9 Å.
- **Attack face:** conformers whose partner approaches a reacting atom from the other
  face than in the seed are dropped after refinement (`AttackFace`, a validator;
  `stereo_filter=False` turns it off). A face counts where the partner lies at least
  0.3 Å from the plane of the atom's other neighbours, in the seed and, to be flagged,
  on the other side in the conformer.
- **Provenance:** `active_bond_targets` and `active_bond_lengths` ("i-j": Å) for every
  conformer; `ConformerEnsemble.summary()` gives min, median and max per bond.

After the TS optimization of windowed conformers, check the TS with
`ReactionCore()` besides the imaginary mode: from stretched or compressed active bonds, a
free saddle search can end on a saddle of another step (with UMA: a proton already on
its acceptor, 1.0 Å off the reacting-atom distances of the TS, 12 kcal/mol lower), which
an imaginary mode and the connectivity outside the reacting atoms do not reveal.
