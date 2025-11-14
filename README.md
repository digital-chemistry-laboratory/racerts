<img src="https://raw.githubusercontent.com/digital-chemistry-laboratory/racerts/main/docs/img/logo.png" alt="Logo" width="1000">

**R**apid and **A**ccurate **C**onformer **E**nsembles with **R**DKit for **T**ransition **S**tates

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


# License
MIT License

Copyright &copy; 2025 ETH Zürich