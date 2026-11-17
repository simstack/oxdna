#!/usr/bin/env python
"""I.2  Fusion MD input: mutual traps pull the two sticky ends into the designed duplex.

Usage:  called with a system dict and OUT_FOLDER
Writes: topology.top, start.dat, mutual_trap.txt, sticky_pairs.op, input, the sequence-parameter file,
        prepare.json, qc.json.  Then run oxDNA in OUT_FOLDER (oxDNA input > oxdna.log).
The fused structure is OUT_FOLDER/last_conf.dat.
"""
import sys
from pathlib import Path

from oxdna.util.oxdna_io import (system_file, read_topology, write_topology, apply_sequences,
                                 designed_pairs, read_configuration, write_configuration, hb_distance,
                                 bond_op, fill_template, model_values, copy_seq_dep, write_json)


def main(s, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    top = read_topology(system_file(s, s["structure"]["topology"]))
    changed = apply_sequences(s, top)
    pairs = designed_pairs(s, top)
    box, conf = read_configuration(system_file(s, s["structure"]["configuration"]))
    assert len(conf) == len(top["base"]), "topology and configuration differ in length"

    m = s["md"]
    traps = "".join(f"{{\ntype = mutual_trap\nparticle = {p}\nref_particle = {q}\n"
                    f"stiff = {m["trap_stiffness"]}\nr0 = {m["trap_r0"]}\n}}\n\n"
                    for i, j in pairs for p, q in ((i, j), (j, i)))
    write_topology(top, out / "topology.top")
    write_configuration(out / "start.dat", conf, box)
    (out / "mutual_trap.txt").write_text(traps)
    (out / "sticky_pairs.op").write_text(bond_op(pairs, "sticky_end"))
    (out / "input").write_text(fill_template("input_fusion.txt", {
        **model_values(s, m["fusion_temperature_C"]), "TOPOLOGY": "topology.top", "CONF": "start.dat",
        "STEPS": int(m["fusion_steps"]), "CONF_INTERVAL": int(m["print_conf_interval"])}))
    copy_seq_dep(s, out)

    start_distance = min(hb_distance(conf, box, i, j) for i, j in pairs)
    write_json(out / "prepare.json", {"designed_pairs": pairs, "sequence_changes": changed,
                                      "box": box.tolist(), "closest_designed_pair_at_start": start_distance})
    write_json(out / "qc.json", {"step": "prepare_fusion", "status": "ok", "n_designed_pairs": len(pairs),
                                 "n_traps": 2 * len(pairs)})
    print(f"{s['name']}: {len(pairs)} designed pairs, {2 * len(pairs)} traps, "
          f"closest designed pair {start_distance:.2f} apart at the start"
          + (f"; sequence changes: {changed}" if changed else ""))


if __name__ == "__main__":
    raise ValueError("prepare_fusion takes an OxdnaSystem dict, not a yaml path")
