#!/usr/bin/env python
"""II.3  Production inputs with the final weights of the tuning rounds.

Usage:  called with a system dict, ROUNDS_FOLDER and OUT_FOLDER.
Takes the fragment files and the current wfile.txt from ROUNDS_FOLDER/run_1 (all replicas must carry the
same weights) and writes OUT_FOLDER/run_1..run_<production_replicas> with new seeds and the production
step count. The weights stay fixed. Warns if ROUNDS_FOLDER/convergence.json does not say converged.
"""
import json
import shutil
import sys
from pathlib import Path

from oxdna.util.oxdna_io import copy_seq_dep, write_json
from oxdna.part2_umbrella.prepare_fragment import vmmc_input

FILES = ["fragment.top", "fragment_unbound.dat", "op.txt", "junction_pairs.op"]


def main(s, rounds, out):
    u = s["umbrella"]
    rounds, out = Path(rounds), Path(out)
    weights = (rounds / "run_1" / "wfile.txt").read_text()
    for r in rounds.glob("run_*"):
        assert (r / "wfile.txt").read_text() == weights, f"{r}: different weights from run_1"
    conv = rounds / "convergence.json"
    converged = conv.exists() and json.loads(conv.read_text())["converged"]
    if not converged:
        print("WARNING: the weight tuning is not marked as converged (convergence.json)")
    for r in range(1, int(u["production_replicas"]) + 1):
        d = out / f"run_{r}"
        d.mkdir(parents=True, exist_ok=True)
        for f in FILES:
            shutil.copy(rounds / "run_1" / f, d / f)
        (d / "wfile.txt").write_text(weights)
        (d / "input").write_text(vmmc_input(s, int(u["seed"]) + 1000 + r, u["production_steps"],
                                            u["round_equilibration_steps"], max(int(u["production_steps"]) // 400, 1)))
        copy_seq_dep(s, d)
    write_json(out / "qc.json", {"step": "prepare_production", "status": "ok" if converged else "warn",
                                 "weights_converged": converged})
    print(f"{s['name']}: {u['production_replicas']} production replicas of {int(u['production_steps'])} steps")


if __name__ == "__main__":
    raise ValueError("prepare_production takes an OxdnaSystem dict, not a yaml path")
