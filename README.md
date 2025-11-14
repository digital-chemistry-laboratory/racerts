<img src="https://github.com/digital-chemistry-laboratory/racerts/blob/8074daa0ae89513eec6f441e67a84e792608a0c1/docs/img/TOC.png" alt="Logo" width="1000">

**R**apid and **A**ccurate **C**onformer **E**nsembles with **R**DKit for **T**ransition **S**tates

<img src="https://github.com/digital-chemistry-laboratory/racerts/blob/8074daa0ae89513eec6f441e67a84e792608a0c1/docs/img/TOC.png" alt="TOC" width="400">

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