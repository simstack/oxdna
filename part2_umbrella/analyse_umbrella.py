#!/usr/bin/env python
"""II.4  Bound fractions, melting temperatures and free-energy profile from the production runs.

Usage:  called with a system dict and FOLDER.
From the unbiased columns of each replica's last_hist.dat (simulation temperature plus the reweighted
temperatures), per replica, then mean +- standard error over replicas:

  phi(T)      bound / unbound, bound = n >= 1, unbound = n = 0 (both distance states)
  f_box       = phi / (1 + phi): one fragment of each kind in the simulation box
  f_bulk      both partners at concentration c each: f / (1 - f)^2 = Phi = phi c / c_box,
              c_box = 1 / (N_A V) for the box volume V (one molecule of each kind per box)
  T(f)        temperature where f_box or f_bulk falls through each target fraction, interpolated between
              the reweighted temperatures; if it does not cross within them, from a straight line of
              ln(phi) against 1/T through the temperatures within far_from_simulation_C (marked);
              results further than far_from_simulation_C from the simulation temperature are flagged
  F(n)        free energy of n formed designed pairs at the simulation temperature, kT, relative to the
              bound basin (n >= 1); barrier = F(1); ln_tau = mean first-passage time of unbinding
              N -> ... -> 1 -> (0, 0) on a nearest-neighbour walk (attempt-rate units)
  ln_K        ln of the association constant at 1 M standard state, ln(phi / c_box); dG_kT = -ln_K

Writes FOLDER/results.json, results_per_replica.csv, bound_fraction_vs_T.csv, free_energy_profile.csv
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from oxdna.util.oxdna_io import write_json

N_A, LENGTH_M = 6.02214076e23, 0.8518e-9          # oxDNA length unit in m


def read_ln_phi(run):
    """(temperatures in C, ln phi, simulation temperature, unbiased p at T_sim with n and dist)."""
    path = run / "last_hist.dat"
    header = path.read_text().splitlines()[0]
    t_extra = [float(v) * 3000 - 273.15 for v in header.split("Ts:")[1].split()] if "Ts:" in header else []
    t_sim = float(re.search(r"^T = ([\d.]+)C", (run / "input").read_text(), re.M).group(1))
    h = np.loadtxt(path, comments="#")
    n, dist = h[:, 0].astype(int), h[:, 1].astype(int)
    temps = np.array([t_sim] + t_extra)
    cols = h[:, 3:3 + len(temps)]
    with np.errstate(divide="ignore", invalid="ignore"):
        ln_phi = np.log(cols[(n >= 1) & (dist == 0)].sum(0)) - np.log(cols[n == 0].sum(0))
    order = np.argsort(temps)
    return temps[order], ln_phi[order], t_sim, (n, dist, cols[:, 0] / cols[:, 0].sum())


def bulk_fraction(ln_phi, c, c_box):
    Phi = np.exp(ln_phi) * c / c_box
    a = 1 + 1 / (2 * Phi)
    return a - np.sqrt(a * a - 1)


def crossing(temps, f, target):
    above = f >= target
    i = np.where(above[:-1] & ~above[1:])[0]
    if len(i) == 0:
        return np.nan
    i = i[0]
    return temps[i] + (target - f[i]) * (temps[i + 1] - temps[i]) / (f[i + 1] - f[i])


def from_line(temps, ln_phi, t_sim, window, ln_phi_target):
    near = (np.abs(temps - t_sim) <= window) & np.isfinite(ln_phi)
    if int(near.sum()) < 2:
        return np.nan
    slope, intercept = np.polyfit(1 / (temps[near] + 273.15), ln_phi[near], 1)
    return slope / (ln_phi_target - intercept) - 273.15


def log_mfpt(F):
    """ln of sum_{i < M} sum_{j <= i} exp(F_i - F_j), states F[0] (start) .. F[-1] (absorbing), in kT."""
    F = np.asarray(F, float)
    terms = np.array([F[i] - F[j] for i in range(len(F) - 1) for j in range(i + 1)])
    return float(terms.max() + np.log(np.exp(terms - terms.max()).sum()))


def sem(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def main(s, folder):
    u = s["umbrella"]
    folder = Path(folder)
    c, window = s["concentration_M"], u["far_from_simulation_C"]
    c_box = 1 / (N_A * (u["box"] * LENGTH_M) ** 3 * 1e3)
    rows, curves, profiles = [], [], []
    runs = sorted(folder.glob("run_*"), key=lambda r: int(r.name.split("_")[1]))
    for run in runs:
        temps, ln_phi, t_sim, (n, dist, p) = read_ln_phi(run)
        if abs(t_sim - s["temperature_C"]) >= 1e-6:
            raise ValueError(f"{run}: input temperature {t_sim} != system temperature {s['temperature_C']}")
        f_box, f_bulk = 1 / (1 + np.exp(-ln_phi)), bulk_fraction(ln_phi, c, c_box)
        k = int(np.argmin(np.abs(temps - t_sim)))
        row = {"run": run.name, "ln_phi": ln_phi[k], "f_box": f_box[k], "f_bulk": f_bulk[k],
               "ln_K": ln_phi[k] - np.log(c_box)}
        row["dG_kT"] = -row["ln_K"]
        for target in u["target_fractions"]:
            for label, f, lp in (("box", f_box, np.log(target / (1 - target))),
                                 ("bulk", f_bulk, np.log(target / (1 - target) ** 2 * c_box / c))):
                t = crossing(temps, f, target)
                row[f"T_{label}_f{target}_from_line"] = bool(np.isnan(t))
                row[f"T_{label}_f{target}"] = from_line(temps, ln_phi, t_sim, window, lp) if np.isnan(t) else t
        n_full = n[dist == 0].max()
        p_bound = np.array([p[(n == m) & (dist == 0)].sum() for m in range(1, n_full + 1)])
        p_00 = p[(n == 0) & (dist == 0)].sum()
        with np.errstate(divide="ignore"):
            F = -np.log(p_bound) + np.log(p_bound.sum())        # relative to the bound basin
        row["barrier_kT"] = F[0]
        row["ln_tau"] = log_mfpt(list(F[::-1]) + [-np.log(p_00) + np.log(p_bound.sum())])
        rows.append(row)
        curves.append(pd.DataFrame({"run": run.name, "T_C": temps, "ln_phi": ln_phi, "f_box": f_box, "f_bulk": f_bulk}))
        profiles.append(pd.DataFrame({"run": run.name, "n": np.arange(1, n_full + 1), "F_kT": F}))

    per = pd.DataFrame(rows)
    per.to_csv(folder / "results_per_replica.csv", index=False)
    cv = pd.concat(curves).groupby("T_C")[["ln_phi", "f_box", "f_bulk"]].agg(["mean", sem])
    cv.columns = [f"{a} {b}" for a, b in cv.columns]
    cv.to_csv(folder / "bound_fraction_vs_T.csv")
    pf = pd.concat(profiles).groupby("n")["F_kT"].agg(["mean", sem])
    pf.to_csv(folder / "free_energy_profile.csv")

    values = [col for col in per.columns if col != "run" and not col.endswith("_from_line")]
    results = {"system": s["name"], "temperature_C": s["temperature_C"], "salt_M": s["salt_M"],
               "concentration_M": c, "c_box_M": c_box, "replicas": len(per)}
    for col in values:
        results[col] = {"mean": float(per[col].mean()), "sem": sem(per[col])}
        if col.startswith("T_"):
            results[col]["far_from_simulation"] = bool(abs(per[col].mean() - s["temperature_C"]) > window)
            results[col]["replicas_from_line"] = int(per[f"{col}_from_line"].sum())
    write_json(folder / "results.json", results)
    show = ", ".join(f"{k} {v['mean']:.2f} +- {v['sem']:.2f}" for k, v in results.items()
                     if isinstance(v, dict) and k in ("f_bulk", "T_bulk_f0.5", "T_bulk_f0.71", "barrier_kT", "ln_K"))
    print(f"{s['name']}: {len(per)} replicas | {show}")


if __name__ == "__main__":
    raise ValueError("analyse_umbrella takes an OxdnaSystem dict, not a yaml path")
