#!/usr/bin/env python
"""II.2  Weight update after one round, for the run_* folders of one system:

  1. for every replica that ran (run folder with an energy.dat), take n, dist and the weight from the
     last three columns of energy.dat; every line is used, none is skipped;
  2. histogram of the states, each line counted with 1/weight, normalised within the replica;
  3. average of these probabilities over the replicas;
  4. raw new weight = 1 / averaged probability;
  5. rescale: 1/p sets its own overall scale, so the raw weights are multiplied by one factor that makes
     them agree with the old weights on average over the visited states (geometric mean, each state
     counted by its number of energy.dat lines);
  6. cap: each rescaled weight is limited to between old/cap and old*cap (cap = 20 unless given).
States that no replica visited keep their old weight and are marked in the printout. Capped states are
marked too, with the uncapped value shown in the column "rescaled 1/p".

The suggestions are printed first. Then you can change any weight by hand before anything is written:
  <Enter>            accept the weights as shown and write them
  n dist value       set one weight, e.g.  3 0 5000   (repeat as often as needed; the table is reprinted)
  q                  quit without writing anything

Without --yes the suggestions are shown and wait for input; with --yes they are accepted as shown
(for unattended workflows).

Convergence: convergence.json in the system folder reports, for the round just analysed, the states
never visited and the number of full transitions between unbound (n = 0) and fully bound (n = N) in
each replica. converged = true when every state was visited and every replica made at least
--min-transitions full transitions (default 2). Then use the current wfile.txt for production.

When the weights are accepted, in every run folder of the system:
  wfile.txt     -> wfile_round<k>.txt       (the weights this round ran with)
  energy.dat    -> energy_round<k>.dat
  last_hist.dat -> last_hist_round<k>.dat
  oxdna.log     -> oxdna_round<k>.log,  last_conf.dat -> last_conf_round<k>.dat
and the new wfile.txt is written into every run folder. The start configuration is not touched, so
the next round starts from the same unbound configuration. A summary table weights_round<k>.txt is
written in the system folder.

Usage:  python update_weights.py FOLDER [--cap 20] [--yes] [--min-transitions 2]
"""
import shutil
import sys
from pathlib import Path

import numpy as np

from oxdna.util.oxdna_io import write_json


def read_weights(path):
    rows = [l.split() for l in Path(path).read_text().splitlines() if l.strip()]
    return [(int(r[0]), int(r[1])) for r in rows], np.array([float(r[2]) for r in rows])


def full_transitions(n, n_full):
    """Number of changes between n = 0 and n = n_full, ignoring the states in between."""
    ends = n[(n == 0) | (n == n_full)]
    return int(np.count_nonzero(np.diff(ends == 0)))


def table(states, visits, p, old, rescaled, new, flagged, capped):
    lines = [f"{'n':>3s} {'dist':>4s} {'visits':>9s} {'p unbiased':>12s} {'old weight':>13s} "
             f"{'rescaled 1/p':>13s} {'new weight':>13s}"]
    for i, s in enumerate(states):
        if old[i] == 0:                          # impossible state, always 0
            continue
        mark = "  <- not visited, old weight kept" if flagged[i] else ("  <- capped" if capped[i] else "")
        lines.append(f"{s[0]:3d} {s[1]:4d} {visits[i]:9d} {p[i]:12.4e} {old[i]:13.6g} "
                     f"{rescaled[i]:13.6g} {new[i]:13.6g}{mark}")
    return "\n".join(lines)


def main(system, cap, accept=False, min_transitions=2, overrides=None):
    if cap is None or cap <= 1:
        raise ValueError(f"weight cap must be > 1, got {cap!r}")
    if min_transitions is None:
        raise ValueError("min_transitions is required")
    folder = Path(system)
    runs = sorted(folder.glob("run_*"), key=lambda r: int(r.name.split("_")[1]))
    used = [r for r in runs if (r / "energy.dat").exists()]
    assert used, f"no energy.dat in any run folder of {folder}"
    states, old = read_weights(runs[0] / "wfile.txt")
    index = {s: i for i, s in enumerate(states)}

    n_full = max(s[0] for s in states)
    probabilities, visits, transitions = [], np.zeros(len(states), dtype=int), {}
    for run in used:
        _, w_run = read_weights(run / "wfile.txt")
        assert np.allclose(w_run, old), f"{run}: wfile.txt differs from {runs[0]}"
        e = np.loadtxt(run / "energy.dat", ndmin=2)
        n, dist, w = e[:, -3].astype(int), e[:, -2].astype(int), e[:, -1]
        unbiased = np.zeros(len(states))
        for a, b, c in zip(n, dist, w):
            unbiased[index[(a, b)]] += 1.0 / c
            visits[index[(a, b)]] += 1
        probabilities.append(unbiased / unbiased.sum())
        transitions[run.name] = full_transitions(n, n_full)
    p = np.mean(probabilities, axis=0)

    flagged = (p == 0) & (old > 0)
    seen = (p > 0) & (old > 0)
    if not np.any(seen):
        scale = 1.0
        rescaled = old.copy()
        new = old.copy()
        capped = np.zeros(len(old), dtype=bool)
    else:
        raw = np.where(seen, 1.0 / np.where(seen, p, 1.0), 0.0)
        scale = np.exp(np.sum(visits[seen] * np.log(old[seen] / raw[seen])) / visits[seen].sum())
        rescaled = np.where(seen, raw * scale, old)          # not-visited states: old weight
        new = np.clip(rescaled, old / cap, old * cap)
        capped = seen & ~np.isclose(new, rescaled)
        new[old == 0] = 0.0
        rescaled[old == 0] = 0.0

    print(f"{system}: {len(used)} replica(s) used ({', '.join(r.name for r in used)}), "
          f"{visits.sum()} energy lines in total | rescale factor {scale:.4g} | cap x{cap:g}")
    unvisited = [states[i] for i in np.where(flagged)[0]]
    converged = not unvisited and min(transitions.values()) >= min_transitions
    write_json(folder / "convergence.json", {"converged": converged, "unvisited_states": unvisited,
                                             "full_transitions": transitions, "min_transitions": min_transitions})
    print(f"full transitions per replica: {transitions} | unvisited states: {unvisited or 'none'} | "
          f"converged: {converged}")
    if overrides:
        for a, b, value in overrides:
            i = index[(int(a), int(b))]
            if old[i] == 0:
                raise ValueError(f"state {(int(a), int(b))} is impossible and keeps weight 0")
            new[i] = float(value)
            capped[i] = False
        accept = True
    while True:
        print(table(states, visits, p, old, rescaled, new, flagged, capped))
        if accept:
            break
        answer = input("Enter = accept and write | 'n dist value' = set a weight | q = quit: ").strip()
        if answer == "":
            break
        if answer.lower() == "q":
            print("nothing written")
            return
        try:
            a, b, value = answer.split()
            i = index[(int(a), int(b))]
            assert old[i] > 0, "this state is impossible and keeps weight 0"
            new[i] = float(value)
            capped[i] = False                       # set by hand
        except (ValueError, KeyError, AssertionError) as err:
            print(f"not understood ({err}); give e.g.  3 0 5000")

    k = len(list(runs[0].glob("wfile_round*.txt"))) + 1
    for run in runs:
        shutil.copy(run / "wfile.txt", run / f"wfile_round{k}.txt")
        for f, archived in [("energy.dat", f"energy_round{k}.dat"), ("last_hist.dat", f"last_hist_round{k}.dat"),
                            ("oxdna.log", f"oxdna_round{k}.log"), ("last_conf.dat", f"last_conf_round{k}.dat")]:
            if (run / f).exists():
                shutil.move(run / f, run / archived)
        (run / "wfile.txt").write_text("".join(f"{s[0]} {s[1]} {v:.6g}\n" for s, v in zip(states, new)))
    (folder / f"weights_round{k}.txt").write_text(
        f"rescale factor {scale:.6g}, cap x{cap:g}\n" + table(states, visits, p, old, rescaled, new, flagged, capped) + "\n")
    print(f"round {k} archived; new wfile.txt written to {len(runs)} run folders")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--cap" not in args or "--min-transitions" not in args:
        raise ValueError("update_weights requires --cap and --min-transitions")
    opts = {}
    for key, caster in (("--cap", float), ("--min-transitions", int)):
        i = args.index(key)
        opts[key] = caster(args[i + 1])
        del args[i:i + 2]
    accept = "--yes" in args
    args = [a for a in args if a != "--yes"]
    main(args[0].rstrip("/"), opts["--cap"], accept, opts["--min-transitions"])
