import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from oxdna.util.oxdna_io import oxdna_version, read_json, refuse_vmmc_360, stamp_qc, write_json


class QcFailed(Exception):
    pass


def log_problems(log_path, seq_dep_name):
    path = Path(log_path)
    if not path.is_file():
        return ["oxDNA log is missing"]
    text = path.read_text(errors="replace")
    problems = []
    if "END OF THE SIMULATION" not in text:
        problems.append("oxDNA log does not contain END OF THE SIMULATION")
    if seq_dep_name not in text:
        problems.append(f"oxDNA log does not mention {seq_dep_name}")
    return problems


def run_oxdna(folder, executable="oxDNA"):
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    result = subprocess.run(
        [executable, "input"],
        cwd=folder,
        capture_output=True,
        text=True,
        env=env,
    )
    (Path(folder) / "oxdna.log").write_text(f"{result.stdout or ''}{result.stderr or ''}")
    return result.returncode


def _replica_dirs(folder):
    folder = Path(folder)
    runs = sorted(folder.glob("run_*"), key=lambda path: int(path.name.split("_")[1]))
    if not runs:
        raise ValueError(f"no run_* folders in {folder}")
    return runs


def run_oxdna_replicas(folder, seq_dep_name, executable="oxDNA"):
    """Run oxDNA in every run_* folder. A leftover energy.dat is a failure."""
    if not seq_dep_name:
        raise ValueError("seq_dep_name is required")

    def one(run):
        if (run / "energy.dat").is_file():
            return {"run": run.name, "status": "fail", "detail": "unprocessed energy.dat"}
        code = run_oxdna(run, executable)
        problems = log_problems(run / "oxdna.log", seq_dep_name)
        if code != 0:
            problems.append(f"oxDNA exited {code}")
        status = "fail" if problems else "ok"
        return {"run": run.name, "status": status, "detail": "; ".join(problems)}

    runs = _replica_dirs(folder)
    with ThreadPoolExecutor(max_workers=len(runs)) as pool:
        results = list(pool.map(one, runs))
    failures = [item for item in results if item["status"] != "ok"]
    return {"results": results, "failures": failures}


def require_vmmc_version(executable="oxDNA"):
    version = oxdna_version(executable)
    refuse_vmmc_360(version)
    return version


def finish_qc(folder, seq_dep, stop_on_qc_fail):
    qc_path = Path(folder) / "qc.json"
    if not qc_path.is_file():
        write_json(qc_path, {
            "step": "missing",
            "status": "fail",
            "problems": ["qc.json was not written"],
            "warnings": [],
        })
    data = stamp_qc(qc_path, seq_dep)
    if stop_on_qc_fail and data.get("status") == "fail":
        problems = data.get("problems") or ["qc failed"]
        raise QcFailed("; ".join(str(item) for item in problems))
    return data


def read_qc(folder):
    return read_json(Path(folder) / "qc.json")


def parse_weight_overrides(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"weight override must be 'n dist value', got {line!r}")
        rows.append((int(parts[0]), int(parts[1]), float(parts[2])))
    if not rows:
        raise ValueError(f"weight override file {path} is empty")
    return rows
