<p align="center">
  <img src="https://raw.githubusercontent.com/digital-chemistry-laboratory/racerts/main/docs/img/logo.png" alt="racerTS logo" width="450">
</p>

<p align="center"><b>Ra</b>pid <b>C</b>onformer <b>E</b>nsembles with <b>R</b>DKit for <b>T</b>ransition <b>S</b>tates</p>

<p align="center">
  <a href="https://doi.org/10.1021/acs.jcim.5c02794"><img src="https://img.shields.io/badge/paper-JCIM%202026-blue" alt="Paper"></a>
  <a href="https://pypi.org/project/racerts/"><img src="https://img.shields.io/pypi/v/racerts.svg" alt="PyPI"></a>
  <a href="https://anaconda.org/conda-forge/racerts"><img src="https://img.shields.io/conda/vn/conda-forge/racerts.svg" alt="conda-forge"></a>
  <a href="https://github.com/digital-chemistry-laboratory/racerts/actions/workflows/test.yml"><img src="https://img.shields.io/github/actions/workflow/status/digital-chemistry-laboratory/racerts/test.yml?label=tests" alt="Tests"></a>
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT license">
</p>

<p align="center">Conformer ensembles of transition states, and of ground states and complexes, via constrained distance geometry</p>

# Installation

```shell
$ pip install racerts
```

# Usage

A transition-state geometry, the atoms whose bonds form or break and a SMILES of the bonds
give an ensemble of transition-state conformers:

```python
import racerts

ensemble = racerts.generate_ts(
    "ts.xyz",  # examples/ts.xyz
    reacting_atoms=[7, 8, 22],  # 0-based
    smiles="C/[NH+]=C(OC)/c1ccccc1.COS(=O)(=O)[O-]",
)
print(ensemble.summary())
ensemble.write_xyz("ensemble.xyz")
```

Ground states, settings, restraints and swaps go through the same pipeline:

```python
racerts.generate_gs("OC(=O)[C@@H]1CCCN1")  # from a SMILES, nothing held fixed

config = racerts.PipelineConfig(embed={"n_conformers": 100}, restraints={"hints": True})
diol = racerts.generate_gs("OCCCCCO", config=config)  # with hydrogen-bond hints

change = racerts.Swap("[*:1]CCCC", old_fragment="[CH3][c:1]")  # methyl to butyl
racerts.swap(racerts.generate_gs("Cc1ccccc1-c1ccccc1"), change)  # samples the new part
```

From the command line:

```console
$ racerts ts ts.xyz --reacting-atoms 7 8 22 --smiles "C/[NH+]=C(OC)/c1ccccc1.COS(=O)(=O)[O-]"
$ racerts gs "OC(=O)[C@@H]1CCCN1"
$ racerts ts.xyz --charge 0 --reacting_atoms 7 8 22   # legacy form and settings (as racerts ts --legacy)
```

[`examples/`](examples) has these as scripts that run in seconds. The
[documentation](https://digital-chemistry-laboratory.github.io/racerts/) covers tasks and
pipelines, restraints, active-bond windows, swaps, other levels of theory through ASE, and
the settings; [`CHANGELOG.md`](CHANGELOG.md) lists what is new. The legacy racerts API
(`ConformerGenerator`) keeps working, with the results of legacy racerts.

# Cite this work
If you use racerTS, please cite [our paper in *J. Chem. Inf. Model.*](https://doi.org/10.1021/acs.jcim.5c02794):

```
@article{schmid_rapid_2026,
	author = {Schmid, Stefan P. and Seng, Henrik and Kläy, Thibault and Jorner, Kjell},
	title = {Rapid Generation of Transition-State Conformer Ensembles via Constrained Distance Geometry},
	journal = {Journal of Chemical Information and Modeling},
	volume = {66},
	number = {5},
	pages = {2777-2790},
	year = {2026},
	doi = {10.1021/acs.jcim.5c02794},
}
```

# Data availability

All data to reproduce the study can be found on [Zenodo](https://doi.org/10.5281/zenodo.17610186).

# License
MIT License

Copyright &copy; 2025 ETH Zürich
