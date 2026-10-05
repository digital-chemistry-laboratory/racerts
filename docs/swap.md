# Swaps

A swap replaces a group of a reference geometry by a new fragment and samples the new
atoms around the kept geometry: e.g. a methyl of a catalyst by a butyl, a hydrogen of a
TS by a substituent, a ligand of a complex by another.

```python
import racerts

ensemble = racerts.swap(
    mol,                                   # graph with one or more conformers
    racerts.Swap("[*:1]CCCC", old_fragment="[CH3][c:1]"),
    n_conformers=30,
)
```

## What is replaced

`Swap(new_fragment, <selector>, attach_map=None, bond_types=None, mode="append")`.
The new fragment is a SMILES with one dummy per attachment (`[*]`, or `[*:1]`,
`[*:2]`, ...); its hydrogens are added. One selector names what leaves:

| Selector | Leaves | The fragment binds to |
| --- | --- | --- |
| `site=n` | the terminal H or dummy with map number n (`racerts.system.swap.label_hydrogen` labels the only H of an atom) | its neighbour |
| `remove_atoms=[...]` | these atoms and their hydrogens; `[]` with `attach_map`: an addition | the atoms where bonds were cut (by index, in dummy order) |
| `center=c, substructure=k` | the k-th group bound to atom c (groups by their lowest atom index) | c |
| `old_fragment="[CH3][c:1]"` | the unmapped atoms of a SMARTS match and everything bound to them beyond the mapped atoms; the group must be unique (if it matches in several ways, the first is used, with a warning; `attach_map` chooses) | the mapped atom of the dummy's number |

- `attach_map={dummy: atom}` sets the attachments explicitly, e.g. a bidentate ligand
  with both dummies on the metal.
- `bond_types={dummy: "dative"}` (or `"single"`, `"double"`, `"triple"`): a dative
  bond goes from the fragment atom to the kept atom. Without it, a phosphine bound by a
  single bond becomes P(V) with an extra H.
- `mode="append"`: fragment atoms take the slots of removed atoms (the first fragment
  atom takes the slot of the atom it replaces) and the rest are appended, so the kept
  atoms keep their indices when the fragment has at least as many atoms as leave;
  otherwise the unused slots close up (`SwapResult.ref_to_new`; the CLI logs changed
  indices). `mode="renumber"` puts the kept atoms first.
- Stereo, in two rules:
    1. What the graphs specify is carried over: the tetrahedral chiral tags and the
       double-bond stereo of the kept atoms and of the fragment, by the order of their
       neighbours, so a swap never inverts them (other tags, e.g. square planar, are
       dropped when the order changes).
    2. What the graphs leave open takes the configuration of the reference geometry
       where that defines it: the atoms around the centre or double bond are kept or
       replace a removed atom (for a centre one neighbour may be new if three are
       kept: a phosphine made a phosphine oxide). This holds for stereo that the swap
       creates (CH₂ → CH(R), cis/trans on a ring, E/Z of a double bond: which
       hydrogen is replaced chooses it) and for stereo of the reference that its
       graph does not tag (a SMILES without stereo on a geometry that has one).
       Otherwise, e.g. around the rest of a grafted fragment, it stays unspecified
       and both configurations are generated. A warning names a centre whose tag is
       lost this way, and reference conformers that disagree.
- Charge: that of the reference, changed by the formal charges of the fragment and of
  the atoms that leave; the multiplicity of the reference, if it has one.
- A kept atom may not lose bond order (e.g. C=O replaced by C–F would leave the carbon
  an unplaced hydrogen short): this raises.

A swap that cannot be done as given (no or several matches, a fragment that is not
one piece, valences that do not work out, settings a swap does not take) raises
`SwapError`, a `ValueError`. `apply_swap(mol, swap)` returns a `SwapResult` with the
new molecule (every reference conformer, same IDs), the kept atoms (`conserved`,
`ref_to_new`), `new_atoms`, the `junction` (kept atoms at an attachment and their kept
neighbours) and diagnostics. For a single attachment that replaces a bond, the fragment
is grafted rigidly (along the removed bond; `placed`), at the removed bond's length
scaled by the covalent radii, so that a bond stretched in a TS (a leaving group) stays
stretched.

## Sampling (`racerts.swap`)

`conserve` sets what happens to the kept atoms:

| conserve | kept atoms | new atoms |
| --- | --- | --- |
| `"hard"` | as in the reference | the graft only (one conformer per reference, checked for clashes) |
| `"soft"` (default) | start at the reference; held within 0.3 Å by position restraints (k = 5 kcal/(mol Å²)) in MMFF/UFF; the junction is free | sampled |
| `"free"` | only the frozen atoms of `task` held | sampled with everything else |

## Rings

Rings are not special: a swap removes atoms and binds the fragment where bonds were
cut, one dummy per cut bond. So ring atoms can leave if the fragment closes every cut.

```python
# a ring atom replaced, a ring made larger: one CH2 of cyclohexane
racerts.swap(ring, racerts.Swap("[*:1]S[*:2]", remove_atoms=[0]))    # thiane
racerts.swap(ring, racerts.Swap("[*:1]CC[*:2]", remove_atoms=[0]))   # cycloheptane

# a fused ring opened: the three ring carbons of bicyclo[7.1.0]decane become a chain,
# which gives cyclodecane (by the atoms, or by a pattern that maps both ends)
racerts.swap(bicycle, racerts.Swap("[*:1]CCC[*:2]", remove_atoms=[4, 5, 6]))
racerts.swap(bicycle, racerts.Swap("[*:1]CCC[*:2]", old_fragment="[C:1]C1CC1[C:2]"))

# a ring closed on a double bond: one hydrogen leaves on each carbon of C=C
racerts.swap(alkene, racerts.Swap("[*:1]CCCCC[*:2]", remove_atoms=[h_a, h_b]))
```

- A pattern over part of a ring maps the ring atoms that stay at both ends
  (`[C:1]...[C:2]`); with one end mapped, what leaves would run round the ring.
- Which hydrogens leave chooses cis or trans of a ring closed on a double bond, by the
  rule for created stereo: the two hydrogens on the same side give the cis ring.
- RDKit's graphs have cis and trans only for double bonds in rings of eight atoms and
  more. racerts keeps the configuration of smaller rings as well: it is set on the
  bond with its two reference atoms, embedded, and checked on every conformer.
- A trans double bond in a small ring is strained, and whether it holds depends on
  the method that refines it: MMFF holds it in a ring of seven atoms (31 kcal/mol
  above cis) and of eight (16 kcal/mol), not of six. If no conformer has the
  configuration, the swap raises and names the bond; it does not return cis.

Two limits of the sampling:

- A double bond in a strained ring or bridge can embed with the stereo of the graph,
  only twisted, and relax into the other isomer in the force field. The stereo check of
  the pipeline (`prune.check_stereo`, on by default, off with the legacy settings)
  removes such conformers.
- With `conserve="soft"`, embedding can end without a conformer two bonds from a
  cis-substituted double bond whose E/Z is in the graph (distance geometry bounds the
  cis 1,4-distance below the one of real geometries, where the kept atoms are held);
  `conserve="free"` or `"hard"` work.

Routes (`routes`, default `["dg"]`): `"dg"`, distance geometry with the kept atoms
placed by the coordinate map; `"rigid"` (opt-in), conformers of the fragment turned
about the attachment bond (12 steps), without poses whose heavy atoms clash with the
kept ones. The provenance records the route (and the reference conformer). Then the
default refinement and pruning run.

With `task` (in reference indices, e.g. the `TransitionState` of a TS), its frozen
atoms stay hard; an atom that the fragment replaces passes its role on (a leaving group
Cl swapped for Br stays a reacting atom). Further new atoms that the task would freeze
(the neighbours of such an atom) have no position from the reference: they are sampled,
with a warning. A swap at or next to the frozen atoms is warned about: the new group
may change the TS.

Restraints of the reference (`restraints=`, a `RestraintSet` in reference indices, e.g.
its hydrogen bonds) carry over where both atoms are kept, and hold in embedding and
refinement; the others are left out with a warning (`SwapResult.lost_contacts`, the
molecule property `swap_lost_restraints`). A contact to an atom that a fragment atom
replaces does not carry over: its distance would not fit.

Ligand swaps: with several attachments (e.g. a bidentate ligand), a fragment atom that
replaces an atom of the same element (a donor) starts at its position
(`SwapResult.positioned`). Holding the metal, its other ligands and the old donors
(`hard=[...]`, reference indices) keeps the coordination geometry, which neither
distance geometry nor UFF knows: dmpe → dppe on cis-PdCl₂ keeps P–Pd–P at 90° and the
square plane, and samples the phenyl groups. An added ligand (`remove_atoms=[]`) has no
position to start from.

For example, on 2-methylbiphenyl → 2-butylbiphenyl `"soft"` keeps the biphenyl skeleton
within 0.13 Å RMSD of the reference while the butyl group is sampled; `"free"` also
turns the ring torsion.

On the command line:

```bash
$ racerts swap ts.xyz -s "CCl" "[Cl-]" -c -1 -r 0 1 2 --new "[*]CCCCO" --remove 3
$ racerts swap ref.xyz -s "Cc1ccccc1-c1ccccc1" --new "[*:1]CCCC" --old "[CH3][c:1]"
```

Options: `--new`, one of `--old`, `--remove`, `--site`, `--center ATOM GROUP`;
`--attach DUMMY ATOM`, `--bond-type DUMMY TYPE`, `--mode`, `-r/--reacting-atoms`,
`--conserve`, `--routes`, `--hard`, and the options of `racerts ts` for the conformer
count, refinement and pruning (a swap embeds with the coordinate map and the
`frozen_first` fallback, and takes no restraints from the command line). The new indices
of the reference atoms are in the molecule property `swap_index_map`; the CLI logs
those that change.
