from pathlib import Path

from simstack.core.context import context
from simstack.core.node import node
from simstack.models import FileStack

from oxdna.models.system import OxdnaConfigurationResult, OxdnaFolderResult, OxdnaSystem
from oxdna.part1_md import check_md, decompose, prepare_fusion, prepare_md
from oxdna.part1_md.pair_energies import main as run_pair_energies
from oxdna.part2_umbrella import (
    analyse_umbrella,
    check_umbrella,
    prepare_fragment,
    prepare_production,
)
from oxdna.part2_umbrella.update_weights import main as apply_weight_update
from oxdna.util.archive import pack_directory, unpack_directory
from oxdna.util.execution import (
    QcFailed,
    finish_qc,
    log_problems,
    parse_weight_overrides,
    read_qc,
    require_vmmc_version,
    run_oxdna,
    run_oxdna_replicas,
)
from oxdna.util.oxdna_io import read_json, write_json
from oxdna.util.system_dict import materialize_system


async def _saved(path):
    stack = FileStack.from_local_file(path, in_memory=True)
    await context.db.save(stack)
    return stack


async def _as_stack(value):
    if isinstance(value, FileStack):
        return value
    found = await context.db.find_one(FileStack, FileStack.id == value)
    if found is None:
        raise ValueError(f"FileStack {value} was not found")
    return found


async def _configuration_result(system, folder, qc, configuration):
    folder_file = pack_directory(folder)
    await context.db.save(folder_file)
    configuration_file = await _saved(configuration)
    return OxdnaConfigurationResult(
        system_name=system["name"],
        step=str(qc.get("step", "")),
        qc_status=str(qc["status"]),
        problems=[str(item) for item in (qc.get("problems") or [])],
        warnings=[str(item) for item in (qc.get("warnings") or [])],
        oxdna_version=str(qc["oxdna_version"]),
        seq_dep_sha256=str(qc["seq_dep_sha256"]),
        folder=folder_file,
        configuration=configuration_file,
    )


async def _folder_result(system, folder, qc):
    folder_file = pack_directory(folder)
    await context.db.save(folder_file)
    return OxdnaFolderResult(
        system_name=system["name"],
        step=str(qc.get("step", "")),
        qc_status=str(qc["status"]),
        problems=[str(item) for item in (qc.get("problems") or [])],
        warnings=[str(item) for item in (qc.get("warnings") or [])],
        oxdna_version=str(qc["oxdna_version"]),
        seq_dep_sha256=str(qc["seq_dep_sha256"]),
        folder=folder_file,
    )


def _seq_dep(system, folder):
    return folder / Path(system["oxdna"]["seq_dep_file"]).name


async def _weight_overrides(model, work):
    if not model.use_weight_overrides:
        return None
    stack = await _as_stack(model.weight_overrides)
    path = stack.get(local_dir=work / "overrides")
    return parse_weight_overrides(path)


def _fail_if_log(folder, seq_dep_name, node_runner):
    problems = log_problems(folder / "oxdna.log", seq_dep_name)
    if problems:
        return node_runner.fail("; ".join(problems))
    return None


@node
async def fuse(system: OxdnaSystem, **kwargs):
    """
    I.2 Fusion MD. Mutual traps pull the designed sticky-end pairs together.

    SimstackResult:
        result (OxdnaConfigurationResult): Fused configuration and the run folder.
    """
    node_runner = kwargs.get("node_runner")
    work = Path.cwd() / "fuse"
    try:
        spec = materialize_system(system, work / "inputs")
        out = work / "fusion"
        prepare_fusion.main(spec, out)
        code = run_oxdna(out)
        if code != 0:
            return node_runner.fail(f"oxDNA exited {code}")
        failed = _fail_if_log(out, _seq_dep(spec, out).name, node_runner)
        if failed is not None:
            return failed
        if not (out / "last_conf.dat").is_file():
            return node_runner.fail("last_conf.dat was not written")
        qc = finish_qc(out, _seq_dep(spec, out), spec["stop_on_qc_fail"])
        return await _configuration_result(spec, out, qc, out / "last_conf.dat")
    except QcFailed as exc:
        return node_runner.fail(str(exc))
    except Exception as exc:
        node_runner.error(str(exc))
        return node_runner.fail(str(exc))


@node
async def production_md(system: OxdnaSystem, fusion: OxdnaConfigurationResult, **kwargs):
    """
    I.3 Production MD without traps, then the completion and pair checks.

    SimstackResult:
        result (OxdnaConfigurationResult): Production start configuration and the run folder.
    """
    node_runner = kwargs.get("node_runner")
    work = Path.cwd() / "production_md"
    try:
        spec = materialize_system(system, work / "inputs")
        fused = await _as_stack(fusion.configuration)
        fused_path = fused.get(local_dir=work / "fused")
        out = work / "md"
        prepare_md.main(spec, fused_path, out)
        prepare_qc = read_qc(out)
        if spec["stop_on_qc_fail"] and prepare_qc.get("status") == "fail":
            return node_runner.fail("; ".join(prepare_qc.get("problems") or ["prepare_md failed"]))
        code = run_oxdna(out)
        if code != 0:
            return node_runner.fail(f"oxDNA exited {code}")
        failed = _fail_if_log(out, _seq_dep(spec, out).name, node_runner)
        if failed is not None:
            return failed
        check_md.main(spec, out)
        qc = finish_qc(out, _seq_dep(spec, out), spec["stop_on_qc_fail"])
        return await _configuration_result(spec, out, qc, out / "start.dat")
    except QcFailed as exc:
        return node_runner.fail(str(exc))
    except Exception as exc:
        node_runner.error(str(exc))
        return node_runner.fail(str(exc))


@node
async def pair_energies(system: OxdnaSystem, md: OxdnaConfigurationResult, **kwargs):
    """
    I.4 DNAnalysis pair energies and the binding-energy decomposition.

    SimstackResult:
        result (OxdnaFolderResult): Run folder with bond_energy.csv and qc.json.
    """
    node_runner = kwargs.get("node_runner")
    work = Path.cwd() / "pair_energies"
    try:
        spec = materialize_system(system, work / "inputs")
        folder_stack = await _as_stack(md.folder)
        run = unpack_directory(folder_stack, work / "unpacked")
        code = run_pair_energies(spec, run)
        if code != 0:
            return node_runner.fail(f"DNAnalysis exited {code}")
        decompose.main(spec, run)
        qc = finish_qc(run, _seq_dep(spec, run), spec["stop_on_qc_fail"])
        return await _folder_result(spec, run, qc)
    except QcFailed as exc:
        return node_runner.fail(str(exc))
    except Exception as exc:
        node_runner.error(str(exc))
        return node_runner.fail(str(exc))


@node
async def fragment(system: OxdnaSystem, fusion: OxdnaConfigurationResult, **kwargs):
    """
    II.1 Cut the sticky-end fragment, separate the partners, and write initial weights.

    SimstackResult:
        result (OxdnaFolderResult): Round folder with one run_* directory per replica.
    """
    node_runner = kwargs.get("node_runner")
    work = Path.cwd() / "fragment"
    try:
        spec = materialize_system(system, work / "inputs")
        fused = await _as_stack(fusion.configuration)
        fused_path = fused.get(local_dir=work / "fused")
        out = work / "rounds"
        prepare_fragment.main(spec, fused_path, out)
        qc = finish_qc(out, spec["_dir"] / spec["oxdna"]["seq_dep_file"], spec["stop_on_qc_fail"])
        return await _folder_result(spec, out, qc)
    except QcFailed as exc:
        return node_runner.fail(str(exc))
    except Exception as exc:
        node_runner.error(str(exc))
        return node_runner.fail(str(exc))


@node
async def umbrella_round(system: OxdnaSystem, fragment_result: OxdnaFolderResult, **kwargs):
    """
    II.2 One VMMC weight-tuning round for every run_* folder.

    SimstackResult:
        result (OxdnaFolderResult): The same round folder after oxDNA.
    """
    node_runner = kwargs.get("node_runner")
    work = Path.cwd() / "umbrella_round"
    try:
        require_vmmc_version()
        spec = materialize_system(system, work / "inputs")
        folder_stack = await _as_stack(fragment_result.folder)
        folder = unpack_directory(folder_stack, work / "unpacked")
        seq_name = Path(spec["oxdna"]["seq_dep_file"]).name
        outcome = run_oxdna_replicas(folder, seq_name)
        if outcome["failures"]:
            detail = "; ".join(f"{item['run']}: {item['detail']}" for item in outcome["failures"])
            return node_runner.fail(detail)
        write_json(folder / "qc.json", {
            "step": "umbrella_round",
            "status": "ok",
            "problems": [],
            "warnings": [],
            "replicas": [item["run"] for item in outcome["results"]],
        })
        qc = finish_qc(folder, folder / "run_1" / seq_name, spec["stop_on_qc_fail"])
        return await _folder_result(spec, folder, qc)
    except QcFailed as exc:
        return node_runner.fail(str(exc))
    except Exception as exc:
        node_runner.error(str(exc))
        return node_runner.fail(str(exc))


@node
async def update_weights(system: OxdnaSystem, round_result: OxdnaFolderResult, **kwargs):
    """
    II.2 Suggest new umbrella weights from the round that just finished.

    The node accepts the suggestion. An accepted weights file replaces individual
    states when use_weight_overrides is true.

    SimstackResult:
        result (OxdnaFolderResult): Round folder with the new wfile.txt and convergence.json.
    """
    node_runner = kwargs.get("node_runner")
    work = Path.cwd() / "update_weights"
    try:
        spec = materialize_system(system, work / "inputs")
        folder_stack = await _as_stack(round_result.folder)
        folder = unpack_directory(folder_stack, work / "unpacked")
        overrides = await _weight_overrides(system, work)
        umbrella = spec["umbrella"]
        apply_weight_update(
            folder,
            umbrella["weight_cap"],
            accept=True,
            min_transitions=umbrella["min_transitions"],
            overrides=overrides,
        )
        convergence = read_json(folder / "convergence.json")
        warnings = [] if convergence["converged"] else ["weight tuning has not converged"]
        write_json(folder / "qc.json", {
            "step": "update_weights",
            "status": "ok" if convergence["converged"] else "warn",
            "problems": [],
            "warnings": warnings,
            "converged": convergence["converged"],
        })
        seq_name = Path(spec["oxdna"]["seq_dep_file"]).name
        qc = finish_qc(folder, folder / "run_1" / seq_name, spec["stop_on_qc_fail"])
        return await _folder_result(spec, folder, qc)
    except QcFailed as exc:
        return node_runner.fail(str(exc))
    except Exception as exc:
        node_runner.error(str(exc))
        return node_runner.fail(str(exc))


@node
async def umbrella_tune(system: OxdnaSystem, fragment_result: OxdnaFolderResult, **kwargs):
    """
    II.2 Repeat a VMMC round and a weight update until convergence or max_rounds.

    SimstackResult:
        result (OxdnaFolderResult): Round folder with the weights of the last round.
    """
    node_runner = kwargs.get("node_runner")
    work = Path.cwd() / "umbrella_tune"
    try:
        require_vmmc_version()
        spec = materialize_system(system, work / "inputs")
        folder_stack = await _as_stack(fragment_result.folder)
        folder = unpack_directory(folder_stack, work / "unpacked")
        umbrella = spec["umbrella"]
        if umbrella["max_rounds"] < 1:
            raise ValueError(f"max_rounds must be >= 1, got {umbrella['max_rounds']}")
        seq_name = Path(spec["oxdna"]["seq_dep_file"]).name
        overrides = await _weight_overrides(system, work)
        converged = False
        for _ in range(int(umbrella["max_rounds"])):
            outcome = run_oxdna_replicas(folder, seq_name)
            if outcome["failures"]:
                detail = "; ".join(f"{item['run']}: {item['detail']}" for item in outcome["failures"])
                return node_runner.fail(detail)
            apply_weight_update(
                folder,
                umbrella["weight_cap"],
                accept=True,
                min_transitions=umbrella["min_transitions"],
                overrides=overrides,
            )
            converged = bool(read_json(folder / "convergence.json")["converged"])
            if converged:
                break
        warnings = [] if converged else [f"not converged after {umbrella['max_rounds']} rounds"]
        write_json(folder / "qc.json", {
            "step": "umbrella_tune",
            "status": "ok" if converged else "warn",
            "problems": [],
            "warnings": warnings,
            "converged": converged,
            "max_rounds": umbrella["max_rounds"],
        })
        qc = finish_qc(folder, folder / "run_1" / seq_name, spec["stop_on_qc_fail"])
        return await _folder_result(spec, folder, qc)
    except QcFailed as exc:
        return node_runner.fail(str(exc))
    except Exception as exc:
        node_runner.error(str(exc))
        return node_runner.fail(str(exc))


@node
async def umbrella_production(system: OxdnaSystem, rounds: OxdnaFolderResult, **kwargs):
    """
    II.3 Production VMMC replicas at fixed weights, then the transition checks.

    SimstackResult:
        result (OxdnaFolderResult): Production folder with one run_* per replica.
    """
    node_runner = kwargs.get("node_runner")
    work = Path.cwd() / "umbrella_production"
    try:
        require_vmmc_version()
        spec = materialize_system(system, work / "inputs")
        folder_stack = await _as_stack(rounds.folder)
        round_folder = unpack_directory(folder_stack, work / "rounds")
        out = work / "production"
        prepare_production.main(spec, round_folder, out)
        seq_name = Path(spec["oxdna"]["seq_dep_file"]).name
        outcome = run_oxdna_replicas(out, seq_name)
        if outcome["failures"]:
            detail = "; ".join(f"{item['run']}: {item['detail']}" for item in outcome["failures"])
            return node_runner.fail(detail)
        check_umbrella.main(spec, out)
        qc = finish_qc(out, out / "run_1" / seq_name, spec["stop_on_qc_fail"])
        return await _folder_result(spec, out, qc)
    except QcFailed as exc:
        return node_runner.fail(str(exc))
    except Exception as exc:
        node_runner.error(str(exc))
        return node_runner.fail(str(exc))


@node
async def umbrella_analysis(system: OxdnaSystem, production: OxdnaFolderResult, **kwargs):
    """
    II.4 Bound fractions, melting temperatures and the free-energy profile.

    SimstackResult:
        result (OxdnaFolderResult): Production folder including results.json.
    """
    node_runner = kwargs.get("node_runner")
    work = Path.cwd() / "umbrella_analysis"
    try:
        spec = materialize_system(system, work / "inputs")
        folder_stack = await _as_stack(production.folder)
        folder = unpack_directory(folder_stack, work / "unpacked")
        analyse_umbrella.main(spec, folder)
        results = folder / "results.json"
        if not results.is_file():
            return node_runner.fail("results.json was not written")
        write_json(folder / "qc.json", {
            "step": "umbrella_analysis",
            "status": "ok",
            "problems": [],
            "warnings": [],
        })
        seq_name = Path(spec["oxdna"]["seq_dep_file"]).name
        qc = finish_qc(folder, folder / "run_1" / seq_name, spec["stop_on_qc_fail"])
        return await _folder_result(spec, folder, qc)
    except QcFailed as exc:
        return node_runner.fail(str(exc))
    except Exception as exc:
        node_runner.error(str(exc))
        return node_runner.fail(str(exc))
