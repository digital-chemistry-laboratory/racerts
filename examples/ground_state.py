"""A ground-state ensemble from a SMILES, with settings."""

import racerts

config = racerts.PipelineConfig(
    seed=7,
    embed={"n_conformers": 50},
    prune={"energy_threshold": 10.0},  # kcal/mol above the lowest conformer
)
ensemble = racerts.generate_gs("OC(=O)[C@@H]1CCCN1", config=config)  # proline

print(ensemble.summary())
best = ensemble.best()
print(f"lowest conformer: {best}, {ensemble.energy(best):.2f} kcal/mol")
ensemble.write_xyz("proline.xyz")
config.to_file(
    "settings.json"
)  # the same settings for: racerts gs ... --config settings.json
