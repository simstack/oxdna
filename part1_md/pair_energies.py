#!/usr/bin/env python
"""I.4  Pair energies of every saved frame after the equilibration cut, with DNAnalysis.

Usage:  called with a system dict and RUN_FOLDER.
Writes RUN_FOLDER/input_analysis and runs DNAnalysis there, which writes all_pair_energies.dat:
one block per frame, one line per interacting pair with the eight oxDNA2 terms.
"""
import subprocess
import sys
from pathlib import Path

from oxdna.util.oxdna_io import fill_template, model_values


def main(s, run, write_only=False, dnanalysis="DNAnalysis"):
    run = Path(run)
    trajectory = run / "trajectory.dat"
    if not trajectory.is_file():
        raise ValueError(f"trajectory.dat is missing in {run}")
    with open(trajectory) as f:
        frames = sum(1 for line in f if line.startswith("t = "))
    skip = int(round(s["md"]["discard_fraction"] * frames))
    (run / "input_analysis").write_text(fill_template("input_analysis.txt", {
        **model_values(s, s["temperature_C"]), "TOPOLOGY": "topology.top", "SKIP": skip}))
    print(f"{s['name']}: {frames} frames, {skip} skipped")
    if write_only:
        return 0
    r = subprocess.run([dnanalysis, "input_analysis"], cwd=run, capture_output=True, text=True)
    (run / "dnanalysis.log").write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        print(f"DNAnalysis failed, see {run / 'dnanalysis.log'}")
    return r.returncode


if __name__ == "__main__":
    raise ValueError("pair_energies takes an OxdnaSystem dict, not a yaml path")
