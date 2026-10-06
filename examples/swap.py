"""A swap: a group of a reference ensemble is replaced, and only the new part is sampled."""

import racerts

reference = racerts.generate_gs(
    "Cc1ccccc1-c1ccccc1", config=racerts.PipelineConfig(embed={"n_conformers": 20})
)  # 2-methylbiphenyl

# The methyl group leaves and a butyl group comes: the atom with the map number stays, and
# the dummy atom of the fragment with the same number marks where it binds.
change = racerts.Swap("[*:1]CCCC", old_fragment="[CH3][c:1]")
ensemble = racerts.swap(reference, change, n_conformers=20)

print("reference:", reference.summary())
print("swapped:  ", ensemble.summary())
ensemble.write_xyz("butylbiphenyl.xyz")
