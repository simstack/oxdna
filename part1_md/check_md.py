#!/usr/bin/env python
"""I.3  Checks of a finished production MD run.

Usage:  called with a system dict and RUN_FOLDER.
Reads:  input, energy_split.dat, sticky_bonds.dat, junction_bonds.dat (if present), oxdna.log (if present)
Writes: qc.json with status ok / warn / fail and
  * completed: last step written equals 'steps' in the input (and the log ends normally, if present)
  * stationarity: shift of the mean potential energy between the discarded part and the rest, in
    standard deviations (warn above 0.5)
  * pairs: mean number of designed pairs formed, fraction of frames with all formed, frames unbound
  * junction: fraction of records with the branch-migration pair formed
"""
import re
import sys
from pathlib import Path

import numpy as np

from oxdna.util.oxdna_io import read_op, mean_sem, write_json, TERMS


def read_hb_list(path):
    """hb_list output -> (steps, list of sets of formed pairs)."""
    steps, bonds = [], []
    for line in Path(path).read_text().splitlines():
        if "#" in line:
            steps.append(int(line.split("#")[0]))
            bonds.append(set())
        elif line.strip():
            a, b = map(int, line.split())
            bonds[-1].add((min(a, b), max(a, b)))
    return np.array(steps), bonds


def main(s, run):
    run = Path(run)
    problems, warnings, out = [], [], {"step": "check_md"}
    energy_path = run / "energy_split.dat"
    if not energy_path.is_file() or energy_path.stat().st_size == 0:
        out.update({"status": "fail", "problems": ["energy_split.dat is missing or empty"], "warnings": [],
                    "completed": False})
        write_json(run / "qc.json", out)
        return 1
    steps_total = int(re.search(r"^steps = (\d+)", (run / "input").read_text(), re.M).group(1))

    e = np.loadtxt(energy_path)
    if e.ndim == 1:
        e = e.reshape(1, -1)
    step, pot = e[:, 0].astype(int), e[:, 1:1 + len(TERMS)].sum(axis=1)
    out["last_step"] = int(step[-1])
    log = run / "oxdna.log"
    out["completed"] = bool(step[-1] >= steps_total and
                            (not log.exists() or "END OF THE SIMULATION" in log.read_text()))
    if not out["completed"]:
        problems.append(f"run incomplete: last step {step[-1]} of {steps_total}")

    keep = step >= s["md"]["discard_fraction"] * steps_total
    if int(keep.sum()) < 2 or int((~keep).sum()) < 1 or float(pot[keep].std()) == 0.0:
        shift = float("nan")
        problems.append("not enough energy samples to check stationarity")
        out["potential_energy"] = None
    else:
        shift = (pot[~keep].mean() - pot[keep].mean()) / pot[keep].std()
        if abs(shift) > 0.5:
            warnings.append(f"potential energy shifts by {shift:.2f} sd after the cut: equilibration may be too short")
        mean, sem, tau = mean_sem(pot[keep])
        out["potential_energy"] = {"mean": mean, "sem": sem, "tau_samples": tau}
    out["stationarity_shift_sd"] = shift

    designed = set(read_op(run / "sticky_pairs.op"))
    bonds_path = run / "sticky_bonds.dat"
    if not bonds_path.is_file():
        problems.append("sticky_bonds.dat is missing")
        n = np.array([])
    else:
        b_step, bonds = read_hb_list(bonds_path)
        n = np.array([len(b & designed) for b, st in zip(bonds, b_step) if st >= s["md"]["discard_fraction"] * steps_total])
    if len(n) == 0:
        problems.append("no sticky-bond records after the equilibration cut")
        out["pairs"] = {"designed": len(designed), "mean_formed": None,
                        "fraction_all_formed": None, "fraction_unbound": None}
    else:
        out["pairs"] = {"designed": len(designed), "mean_formed": float(n.mean()),
                        "fraction_all_formed": float((n == len(designed)).mean()),
                        "fraction_unbound": float((n == 0).mean())}
        if (n == 0).any():
            warnings.append(f"the sticky ends are fully unbound in {(n == 0).mean():.1%} of the records")

    if (run / "junction_bonds.dat").exists():
        junction = read_op(run / "junction_pairs.op")
        j_step, j_bonds = read_hb_list(run / "junction_bonds.dat")
        keep_j = j_step >= s["md"]["discard_fraction"] * steps_total
        if len(junction) > 1:
            migrated = np.array([tuple(sorted(junction[1])) in b for b in j_bonds])[keep_j]
            out["junction_migrated_fraction"] = float(migrated.mean())

    out["status"] = "fail" if problems else ("warn" if warnings else "ok")
    out["problems"], out["warnings"] = problems, warnings
    write_json(run / "qc.json", out)
    formed = "n/a" if len(n) == 0 else f"{n.mean():.2f}"
    shift_text = "n/a" if shift != shift else f"{shift:+.2f}"
    print(f"{s['name']}: {out['status']} | completed {out['completed']} | shift {shift_text} sd | "
          f"pairs formed {formed}/{len(designed)}" + "".join(f" | {w}" for w in problems + warnings))
    return 1 if problems else 0


if __name__ == "__main__":
    raise ValueError("check_md takes an OxdnaSystem dict, not a yaml path")
