import asyncio
import shutil
from pathlib import Path

from simstack.core.context import context
from simstack.models import Parameters

from oxdna.nodes.workflow import (
    fragment,
    fuse,
    pair_energies,
    production_md,
    umbrella_analysis,
    umbrella_production,
    umbrella_round,
    update_weights,
)
from oxdna.testing.yls6 import yls6_smoke_settings, yls6_system
from oxdna.util.archive import unpack_directory
from oxdna.util.oxdna_io import read_json


def _assert_qc(result, folder):
    qc_files = list(Path(folder).rglob("qc.json"))
    if not qc_files:
        raise AssertionError(f"{result.step} did not write qc.json")
    for path in qc_files:
        data = read_json(path)
        if "status" not in data:
            raise AssertionError(f"{path} has no status")
    if not result.oxdna_version:
        raise AssertionError(f"{result.step} did not record an oxDNA version")
    if not result.seq_dep_sha256:
        raise AssertionError(f"{result.step} did not record the sequence-file checksum")


def _assert_logs(folder):
    logs = list(Path(folder).rglob("oxdna.log"))
    if not logs:
        raise AssertionError("oxDNA log is missing")
    for path in logs:
        text = path.read_text(errors="replace")
        if "END OF THE SIMULATION" not in text:
            raise AssertionError(f"{path} did not finish")
        if "oxDNA2_sequence_dependent_parameters.txt" not in text:
            raise AssertionError(f"{path} did not mention the sequence-parameter file")


async def _save_inputs(system):
    await context.db.save(system.topology)
    await context.db.save(system.configuration)
    await context.db.save(system.seq_dep_file)
    await context.db.save(system)


async def main():
    await context.initialize()
    system = yls6_system(yls6_smoke_settings())
    await _save_inputs(system)
    parameters = Parameters(resource="local", in_docker=True, force_rerun=True)
    work = Path.cwd() / "yls6_smoke_inspect"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()

    fusion = await fuse(system, parameters=parameters)
    fusion_dir = unpack_directory(fusion.folder, work / "fuse")
    _assert_qc(fusion, fusion_dir)
    _assert_logs(fusion_dir)
    print(f"fuse {fusion.qc_status} version {fusion.oxdna_version}")

    md = await production_md(system, fusion, parameters=parameters)
    md_dir = unpack_directory(md.folder, work / "md")
    _assert_qc(md, md_dir)
    _assert_logs(md_dir)
    print(f"production_md {md.qc_status}")

    energies = await pair_energies(system, md, parameters=parameters)
    energy_dir = unpack_directory(energies.folder, work / "energies")
    _assert_qc(energies, energy_dir)
    if not (energy_dir / "bond_energy.csv").is_file():
        raise AssertionError("bond_energy.csv was not written")
    print(f"pair_energies {energies.qc_status}")

    frag = await fragment(system, fusion, parameters=parameters)
    frag_dir = unpack_directory(frag.folder, work / "fragment")
    _assert_qc(frag, frag_dir)
    print(f"fragment {frag.qc_status}")

    rnd = await umbrella_round(system, frag, parameters=parameters)
    round_dir = unpack_directory(rnd.folder, work / "round")
    _assert_qc(rnd, round_dir)
    _assert_logs(round_dir)
    print(f"umbrella_round {rnd.qc_status}")

    weights = await update_weights(system, rnd, parameters=parameters)
    weights_dir = unpack_directory(weights.folder, work / "weights")
    _assert_qc(weights, weights_dir)
    if not (weights_dir / "convergence.json").is_file():
        raise AssertionError("convergence.json was not written")
    print(f"update_weights {weights.qc_status}")

    production = await umbrella_production(system, weights, parameters=parameters)
    prod_dir = unpack_directory(production.folder, work / "production")
    _assert_qc(production, prod_dir)
    _assert_logs(prod_dir)
    print(f"umbrella_production {production.qc_status}")

    analysis = await umbrella_analysis(system, production, parameters=parameters)
    analysis_dir = unpack_directory(analysis.folder, work / "analysis")
    _assert_qc(analysis, analysis_dir)
    if not (analysis_dir / "results.json").is_file():
        raise AssertionError("results.json was not written")
    print(f"umbrella_analysis {analysis.qc_status}")
    print("YLs6 smoke chain finished")


if __name__ == "__main__":
    asyncio.run(main())
