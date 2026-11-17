#!/usr/bin/env python
"""I.4  Binding energy between the two partners, split by term and by location.

Usage:  called with a system dict and RUN_FOLDER.
Reads:  all_pair_energies.dat (DNAnalysis), energy_split.dat, topology.top, prepare.json
Per frame: E_bind = sum over all pairs (i in partner A, j in partner B) of the pair energy, split into
  terms:      FENE ... DH (the backbone terms are zero between different strands)
  locations:  duplex  - sticky end A with sticky end B (incl. the first duplex base of B),
              nick A  - contacts of the 3' end of A's duplex partner (faces the 5' end of B),
              nick B  - contacts of the 3' end of B's duplex partner (faces sticky end A),
              other   - all remaining pairs between the partners.
Check:  the pair energies summed over all pairs reproduce energy_split.dat term by term.
Writes: bond_energy.csv (mean, sem, tau in frames), bond_energy_per_frame.csv, qc.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from oxdna.util.oxdna_io import read_topology, sticky_ids, partner_strands, mean_sem, write_json, TERMS


def read_pair_energies(path):
    """-> (pair rows with a frame column, steps of the frames)."""
    steps, sizes, count = [], [], 0
    with open(path) as f:
        for line in f:
            if line.startswith("#id1"):
                if steps:
                    sizes.append(count)
                steps.append(int(line.rsplit("=", 1)[1]))
                count = 0
            elif line.strip() and not line.startswith("#"):
                count += 1
    sizes.append(count)
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None, names=["i", "j"] + TERMS + ["total"])
    df["frame"] = np.repeat(np.arange(len(steps)), sizes)
    return df, np.array(steps)


def main(s, run):
    run = Path(run)
    if not (run / "all_pair_energies.dat").is_file():
        write_json(run / "qc.json", {"step": "decompose", "status": "fail",
                                     "problems": ["all_pair_energies.dat is missing"], "warnings": []})
        return 1
    top = read_topology(run / "topology.top")
    geo = json.loads((run / "prepare.json").read_text())
    a_ids, b_ids = sticky_ids(s, top)
    b_region = b_ids + [top["strands"][s["partner_b"]["sticky_strand"]][len(b_ids)]]
    in_a, in_b = partner_strands(s, top)

    pe, steps = read_pair_energies(run / "all_pair_energies.dat")
    i, j = pe["i"].to_numpy(), pe["j"].to_numpy()
    cross = (in_a[i] & in_b[j]) | (in_b[i] & in_a[j])
    has = lambda ids: np.isin(i, ids) | np.isin(j, ids)
    duplex = cross & has(a_ids) & has(b_region)
    nick_a = cross & ~duplex & has([geo["duplex_a_3end"]])
    nick_b = cross & ~duplex & ~nick_a & has([geo["duplex_b_3end"]])
    other = cross & ~duplex & ~nick_a & ~nick_b

    nf = len(steps)
    frame_sum = lambda mask: pe.loc[mask].groupby("frame")[TERMS].sum().reindex(range(nf), fill_value=0.0)
    bind = frame_sum(cross)
    per_frame = pd.DataFrame({"step": steps, "E_bind": bind.sum(axis=1),
                              **{f"E_bind {t}": bind[t] for t in TERMS if t not in ("FENE", "BEXC", "STCK")},
                              **{f"E_bind {k}": frame_sum(m).sum(axis=1)
                                 for k, m in [("duplex", duplex), ("nick A", nick_a), ("nick B", nick_b), ("other", other)]}})

    split = pd.DataFrame(np.loadtxt(run / "energy_split.dat")[:, :1 + len(TERMS)], columns=["step"] + TERMS).set_index("step")
    total = frame_sum(np.ones(len(pe), bool)).to_numpy() / len(top["base"])
    common = np.isin(steps, split.index)
    deviation = float(np.abs(total[common] - split.loc[steps[common], TERMS].to_numpy()).max()) if common.any() else np.nan
    backbone_cross = float(np.abs(pe.loc[cross, ["FENE", "BEXC", "STCK"]].to_numpy()).max())

    rows = []
    for c in per_frame.columns[1:]:
        m, e, tau = mean_sem(per_frame[c])
        rows.append({"quantity": c, "mean": m, "sem": e, "tau_frames": tau})
    table = pd.DataFrame(rows)
    table.to_csv(run / "bond_energy.csv", index=False)
    per_frame.to_csv(run / "bond_energy_per_frame.csv", index=False)

    problems = []
    if not deviation < 1e-3:
        problems.append(f"pair energies do not reproduce energy_split.dat (max deviation {deviation:.2e}, "
                        f"{common.sum()} frames compared)")
    write_json(run / "qc.json", {"step": "decompose", "status": "fail" if problems else "ok", "problems": problems,
                                 "frames": nf, "frames_compared": int(common.sum()),
                                 "max_deviation_from_energy_split": deviation, "backbone_terms_between_partners": backbone_cross})
    print(f"{s['name']}: {nf} frames | E_bind {table.loc[0, 'mean']:.3f} +- {table.loc[0, 'sem']:.3f} | "
          f"check vs energy_split: max deviation {deviation:.1e} over {common.sum()} frames")
    return 1 if problems else 0


if __name__ == "__main__":
    raise ValueError("decompose takes an OxdnaSystem dict, not a yaml path")
