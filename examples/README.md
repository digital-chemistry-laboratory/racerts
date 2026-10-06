# Examples

Each script runs on its own in a few seconds and writes its files to the working directory:

    python transition_state.py

| script | shows |
|---|---|
| `transition_state.py` | `generate_ts`: a TS geometry (`ts.xyz`), its reacting atoms and its SMILES give a conformer ensemble |
| `ground_state.py` | `generate_gs` from a SMILES; settings as a `PipelineConfig`, written to a file for the command line |
| `restraints.py` | hydrogen-bond hints: more conformers with the contact, and a lower minimum |
| `swap.py` | `swap`: a methyl group of a reference ensemble becomes a butyl group |
| `other_levels.py` | a staged pipeline that refines with GFN2-xTB (needs `tblite`) |
| `example.ipynb` | the legacy racerts API (`ConformerGenerator`) step by step |

The same from the command line:

    racerts ts ts.xyz --reacting-atoms 7 8 22 --smiles "C/[NH+]=C(OC)/c1ccccc1.COS(=O)(=O)[O-]" -n 30
    racerts gs "OC(=O)[C@@H]1CCCN1" --seed 7 -n 50 --energy-threshold 10

The [documentation](https://digital-chemistry-laboratory.github.io/racerts/) has the details.
