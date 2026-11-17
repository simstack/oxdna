# oxDNA sticky-end thermodynamics

SimStack nodes for one pair of complementary sticky ends. Part I (MD) reports the binding energy split into the oxDNA2 terms and by location. Part II (umbrella-sampled VMMC) reports the free-energy profile, bound fractions against temperature, and melting temperatures corrected for box volume and concentration.

The host tree has no nested package directory. `pyproject.toml` installs it for local tests. The image copies `pyproject.docker` to `pyproject.toml` and installs the same tree. `vendor/` is a local oxDNA tarball and is not part of the repository.

## Nodes

| node | step |
|---|---|
| `fuse` | I.2 mutual-trap MD |
| `production_md` | I.3 production MD and checks |
| `pair_energies` | I.4 DNAnalysis and the energy decomposition |
| `fragment` | II.1 fragment, unbound start, initial weights |
| `umbrella_round` | II.2 one VMMC round |
| `update_weights` | II.2 weight update |
| `umbrella_tune` | II.2 repeat round + update until convergence or `max_rounds` |
| `umbrella_production` | II.3 production replicas and checks |
| `umbrella_analysis` | II.4 fractions, melting temperatures, F(n) |

Inputs are an `OxdnaSystem` model: topology, configuration, sequence-parameter file, sequences, temperature, salt, concentration, and the run lengths. `md_backend` is `CPU` or `CUDA`. `stop_on_qc_fail` decides whether a failed `qc.json` stops the node. oxDNA and DNAnalysis are taken from `PATH`. oxDNA 3.6.0 is refused for VMMC.

The YLs6 nanostar-linker files live in `data/example_YLs6/`. `testing/yls6.py` builds the production settings (CUDA, the original step counts) and a short CPU smoke system.

## Image

```bash
docker build -t oxdna:latest .
```

The image compiles oxDNA v3.8.1 with CUDA off and installs the SimStack package from `ww/new-features`. `config.toml` points the local resource at `oxdna:latest`.

A gitignored `simstack.toml` is required before calling nodes (`await context.initialize()` with no arguments). Then:

```bash
uv run create_model_table --dir models
uv run create_node_table --dir nodes
uv run python -m oxdna.testing.yls6_smoke
```

Do not use oxDNA 3.6.0 for VMMC with DNA2.
