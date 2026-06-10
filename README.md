<img src="https://raw.githubusercontent.com/digital-chemistry-laboratory/racerts/main/docs/img/logo.png" alt="Logo" width="1000">

**Ra**pid **C**onformer **E**nsembles with **R**DKit for **T**ransition **S**tates

<img src="https://raw.githubusercontent.com/digital-chemistry-laboratory/racerts/main/docs/img/TOC.png" alt="TOC" width="400">

# Installation

```shell
$ pip install racerts
```

# Usage

racerTS can be imported as a Python module that is easily integrated into
workflows for transition state conformer ensemble generation.
For further information, see the separate [documentation](https://digital-chemistry-laboratory.github.io/racerts/).

```shell
>>> from racerts import ConformerGenerator
>>> cg = ConformerGenerator()
>>> ts_conformers_mol = cg.generate_conformers(file_name="example.xyz", charge=0, reacting_atoms=[2,3,4])
>>> cg.write_xyz("ensemble.xyz")
```

It can also be accessed via a command line interface.

```console
$ racerts example.xyz --charge 0 --reacting_atoms 2 3 4
```

# Cite this work
If you use racerTS in one of your works, please make sure to cite it:

```
@article{doi:10.1021/acs.jcim.5c02794,
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
