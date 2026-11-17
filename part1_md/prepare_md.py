#!/usr/bin/env python
"""I.3  Production MD input for the fused pair (no external forces).

Usage:  called with a system dict, FUSED_CONF and OUT_FOLDER.
        FUSED_CONF: last_conf.dat of the fusion run (or an already fused structure, e.g. of the
        parent of a mutant: the sequences in the system dict are applied to the topology).
Writes: topology.top, start.dat, sticky_pairs.op, junction_pairs.op, input, the sequence-parameter
        file, prepare.json (pairs, nicks, junction), qc.json.  Then run oxDNA in OUT_FOLDER.
Checks: every backbone bond intact and all designed pairs but at most one formed in the start
        configuration (after the box change).
"""
import sys
from pathlib import Path

import numpy as np

from oxdna.util.oxdna_io import (system_file, read_topology, write_topology, apply_sequences,
                                 designed_pairs, read_configuration, write_configuration, pair_formed,
                                 junction_geometry, longest_backbone_bond, bond_op, fill_template,
                                 model_values, copy_seq_dep, write_json)


def main(s, fused_conf, out):
    m = s["md"]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    top = read_topology(system_file(s, s["structure"]["topology"]))
    changed = apply_sequences(s, top)
    pairs = designed_pairs(s, top)
    _, conf = read_configuration(fused_conf)
    box = np.ones(3) * float(m["box"])

    longest = longest_backbone_bond(top, conf, box)
    formed = [pair_formed(conf, box, i, j) for i, j in pairs]
    geo = junction_geometry(s, top, conf, box)
    problems = []
    for tag in ("a", "b"):
        dist = geo[f"duplex_{tag}_distance"]
        if dist >= 0.75:
            problems.append(f"duplex {tag} is not paired (hydrogen-bond distance {dist:.2f})")
    if longest > 0.95:
        problems.append(f"backbone bond of {longest:.2f} in the start configuration")
    if sum(formed) < len(pairs) - 1:
        problems.append(f"only {sum(formed)} of {len(pairs)} designed pairs formed")

    junction = [geo["junction_body"]] + ([geo["junction_migration"]] if geo["junction_migration"] else [])
    write_topology(top, out / "topology.top")
    write_configuration(out / "start.dat", conf, box)
    (out / "sticky_pairs.op").write_text(bond_op(pairs, "sticky_end"))
    (out / "junction_pairs.op").write_text(bond_op(junction, "junction"))
    (out / "input").write_text(fill_template("input_md.txt", {
        **model_values(s, s["temperature_C"]), "TOPOLOGY": "topology.top", "CONF": "start.dat",
        "EQUIL": int(m["equilibration_steps"]), "STEPS": int(m["steps"]),
        "CONF_INTERVAL": int(m["print_conf_interval"]), "PRINT_EVERY": int(m["print_every"])}))
    copy_seq_dep(s, out)
    write_json(out / "prepare.json", {"designed_pairs": pairs, "pairs_formed_at_start": int(sum(formed)),
                                      "sequence_changes": changed, "longest_backbone_bond": longest, **geo})
    write_json(out / "qc.json", {"step": "prepare_md", "status": "fail" if problems else "ok", "problems": problems})
    print(f"{s['name']}: {sum(formed)}/{len(pairs)} designed pairs formed, nick A {geo['nick_a']}, "
          f"nick B {geo['nick_b']}, junction {junction}" + (f" | PROBLEMS: {problems}" if problems else ""))
    return 1 if problems else 0


if __name__ == "__main__":
    raise ValueError("prepare_md takes an OxdnaSystem dict, not a yaml path")
