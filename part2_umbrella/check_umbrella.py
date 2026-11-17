#!/usr/bin/env python
"""II.3  Checks of finished production runs (all run_* folders of one system).

Usage:  called with a system dict and FOLDER.
Per replica: completed (log ends normally), full transitions between unbound (n = 0) and fully bound
(n = N) from energy.dat, biased share of unbound lines, states never visited (last_hist.dat), and
ln(phi) = ln(bound / unbound) at the simulation temperature.
Per system: standard error of ln(phi) over the replicas, and 2 / sqrt(total transitions) as the
precision expected from the number of transitions.
Writes FOLDER/qc_per_run.csv and FOLDER/qc.json (fail: a run incomplete or a replica never unbinds;
warn: a state never visited or fewer than 5 transitions in a replica).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from oxdna.util.oxdna_io import write_json


def read_hist(path):
    """last_hist.dat -> (n, dist, biased counts, unbiased columns: simulation T first)."""
    h = np.loadtxt(path, comments="#")
    return h[:, 0].astype(int), h[:, 1].astype(int), h[:, 2], h[:, 3:]


def main(s, folder):
    folder = Path(folder)
    rows = []
    for run in sorted(folder.glob("run_*"), key=lambda r: int(r.name.split("_")[1])):
        log = run / "oxdna.log"
        completed = log.exists() and "END OF THE SIMULATION" in log.read_text()
        if not (run / "energy.dat").is_file() or not (run / "last_hist.dat").is_file():
            rows.append({"run": run.name, "completed": False, "energy_lines": 0,
                         "full_transitions": 0, "unbound_share_biased": np.nan,
                         "unvisited_states": "missing", "ln_phi": np.nan})
            continue
        e = pd.read_csv(run / "energy.dat", sep=r"\s+", header=None).to_numpy()
        if e.size == 0:
            rows.append({"run": run.name, "completed": completed, "energy_lines": 0,
                         "full_transitions": 0, "unbound_share_biased": np.nan,
                         "unvisited_states": "missing", "ln_phi": np.nan})
            continue
        n = e[:, -3].astype(int)
        n_hist, dist, biased, unbiased = read_hist(run / "last_hist.dat")
        n_full = n_hist.max()
        ends = n[(n == 0) | (n == n_full)]
        weights = {(int(a), int(b)): float(w) for a, b, w in np.loadtxt(run / "wfile.txt", ndmin=2)}
        unvisited = [(int(a), int(b)) for a, b, c in zip(n_hist, dist, biased) if c == 0 and weights.get((a, b), 0) > 0]
        bound, free = unbiased[(n_hist >= 1) & (dist == 0), 0].sum(), unbiased[n_hist == 0, 0].sum()
        rows.append({"run": run.name, "completed": completed, "energy_lines": len(n),
                     "full_transitions": int(np.count_nonzero(np.diff(ends == 0))),
                     "unbound_share_biased": float((n == 0).mean()), "unvisited_states": str(unvisited),
                     "ln_phi": float(np.log(bound / free)) if bound > 0 and free > 0 else np.nan})
    q = pd.DataFrame(rows)
    q.to_csv(folder / "qc_per_run.csv", index=False)

    problems, warnings = [], []
    if not q.completed.all():
        problems.append(f"incomplete runs: {list(q.run[~q.completed])}")
    if (q.full_transitions == 0).any():
        problems.append(f"replicas without a full transition: {list(q.run[q.full_transitions == 0])}")
    if (q.unvisited_states != "[]").any():
        warnings.append("some states were never visited")
    if (q.full_transitions < 5).any():
        warnings.append(f"fewer than 5 full transitions in {list(q.run[q.full_transitions < 5])}")
    se = float(q.ln_phi.std(ddof=1) / np.sqrt(len(q))) if len(q) > 1 else np.nan
    out = {"step": "check_umbrella", "status": "fail" if problems else ("warn" if warnings else "ok"),
           "problems": problems, "warnings": warnings, "replicas": len(q),
           "total_full_transitions": int(q.full_transitions.sum()), "se_ln_phi": se,
           "expected_se_from_transitions": float(2 / np.sqrt(max(q.full_transitions.sum(), 1)))}
    write_json(folder / "qc.json", out)
    print(f"{s['name']}: {out['status']} | {len(q)} replicas | transitions {list(q.full_transitions)} | "
          f"ln phi {q.ln_phi.mean():.2f} +- {se:.2f}" + "".join(f" | {w}" for w in problems + warnings))
    return 1 if problems else 0


if __name__ == "__main__":
    raise ValueError("check_umbrella takes an OxdnaSystem dict, not a yaml path")
