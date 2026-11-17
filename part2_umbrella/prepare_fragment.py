#!/usr/bin/env python
"""II.1  Fragment for umbrella sampling, unbound start, order parameter and initial weights.

Usage:  called with a system dict, FUSED_CONF and OUT_FOLDER.
        FUSED_CONF: a fused (bound) configuration of the full structure, e.g. the fusion last_conf.dat.

The fragment keeps four strands:
  1  strand of sticky end A: the sticky end + duplex_bp_a nucleotides of its duplex
  2  its duplex partner: the duplex_bp_a nucleotides at its 3' end (3' end = nick A)
  3  strand of sticky end B: the sticky end + duplex_bp_b nucleotides of its duplex
  4  its duplex partner: the duplex_bp_b nucleotides at its 3' end (3' end = nick B)
so both nicks and the junction next to B's duplex are kept. Strands 3-4 are then moved rigidly half a
box away from strands 1-2: every run starts unbound.

Order parameter (op.txt): number of designed pairs formed (bond) and a minimum-distance coordinate
over the same pairs with one interface at contact_distance: states (n, 0) for n = 0..N, and (0, 1)
for "far apart". States (n >= 1, 1) cannot occur and get weight 0.

Writes OUT_FOLDER/run_1..run_<round_replicas>/ with fragment.top, fragment_unbound.dat, op.txt,
junction_pairs.op, wfile.txt, input (one weight-tuning round) and the sequence-parameter file;
OUT_FOLDER/fragment.json and qc.json.
"""
import sys
from pathlib import Path

import numpy as np

from oxdna.util.oxdna_io import (COMPLEMENT, system_file, read_topology, apply_sequences, designed_pairs,
                                 read_configuration, write_configuration, pair_formed, hb_distance,
                                 junction_geometry, longest_backbone_bond, bond_op, fill_template,
                                 model_values, copy_seq_dep, write_json)

DIRECTIONS = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, -1, 0), (1, 0, -1),
              (0, 1, -1), (1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)]


def cut(s, top, conf, box, geo):
    """Fragment strands (old ids, 5'->3'), classic topology text and centred coordinates."""
    u = s["umbrella"]
    la, lb = len(s["partner_a"]["sticky_sequence"]), len(s["partner_b"]["sticky_sequence"])
    sa, sb = top["strands"][s["partner_a"]["sticky_strand"]], top["strands"][s["partner_b"]["sticky_strand"]]
    pa, pb = top["strands"][geo["duplex_a_strand"]], top["strands"][geo["duplex_b_strand"]]
    strands = [sa[:la + u["duplex_bp_a"]], pa[-u["duplex_bp_a"]:], sb[:lb + u["duplex_bp_b"]], pb[-u["duplex_bp_b"]:]]
    for x, y, tag in ((strands[0][la:], strands[1], "A"), (strands[2][lb:], strands[3], "B")):
        sx, sy = "".join(top["base"][x]), "".join(top["base"][y])
        assert sx == "".join(COMPLEMENT[c] for c in reversed(sy)), f"duplex {tag} is not complementary: {sx} / {sy}"

    order = [i for st in strands for i in reversed(st)]          # classic format: each strand 3' -> 5'
    new_id = {old: new for new, old in enumerate(order)}
    lines = [f"{len(order)} {len(strands)}"]
    for k, st in enumerate(strands, start=1):
        listed = list(reversed(st))
        for pos, old in enumerate(listed):
            n3 = new_id[listed[pos - 1]] if pos > 0 else -1
            n5 = new_id[listed[pos + 1]] if pos < len(listed) - 1 else -1
            lines.append(f"{k} {top['base'][old]} {n3} {n5}")

    xyz, start, ref = conf[order].copy(), 0, None                # undo periodic wrapping strand by strand
    for st in strands:
        block = slice(start, start + len(st))
        com = xyz[block, :3].mean(axis=0)
        if ref is None:
            ref = com
        else:
            xyz[block, :3] -= box * np.round((com - ref) / box)
        start += len(st)
    xyz[:, :3] += u["box"] / 2 - xyz[:, :3].mean(axis=0)
    xyz[:, 9:15] = 0.0
    frag_strand = np.repeat([1, 2, 3, 4], [len(x) for x in strands])
    return strands, new_id, "\n".join(lines) + "\n", xyz, frag_strand


def unbind(xyz, frag_strand, box):
    """Move strands 3-4 half a box away from strands 1-2, along the direction with the largest gap."""
    half_a, half_b = frag_strand <= 2, frag_strand >= 3
    best = None
    for v in map(np.array, DIRECTIONS):
        moved = xyz.copy()
        moved[half_b, :3] += xyz[half_a, :3].mean(0) + 0.5 * box * v / np.abs(v).max() - xyz[half_b, :3].mean(0)
        d = moved[half_a, None, :3] - moved[None, half_b, :3]
        d -= box * np.round(d / box)
        gap = float(np.linalg.norm(d, axis=-1).min())
        if best is None or gap > best[1]:
            best = (moved, gap)
    return best


def initial_weights(u, n_pairs):
    w = u["initial_weights"]
    bound = list(w["bound"])[:n_pairs] + [1.0] * max(0, n_pairs - len(w["bound"]))
    rows = [(0, 0, w["unbound_near"])] + [(n, 0, bound[n - 1]) for n in range(1, n_pairs + 1)]
    rows += [(0, 1, w["unbound_far"])] + [(n, 1, 0.0) for n in range(1, n_pairs + 1)]
    return "".join(f"{n} {d} {v:.6g}\n" for n, d, v in rows)


def extrapolation(u, t_sim):
    lo, hi, step = u["extrapolate_C"]
    temps = [t for t in np.arange(lo, hi + step / 2, step) if abs(t - t_sim) > 1e-6]
    return ", ".join(f"{t:g}C" for t in temps)


def vmmc_input(s, seed, steps, equilibration, conf_interval):
    u = s["umbrella"]
    return fill_template("input_vmmc.txt", {
        **model_values(s, s["temperature_C"]), "SEED": seed, "EXTRAPOLATE": extrapolation(u, s["temperature_C"]),
        "EQUIL": int(equilibration), "STEPS": int(steps), "PRINT_EVERY": int(u["print_every"]),
        "CONF_INTERVAL": int(conf_interval)})


def main(s, fused_conf, out):
    u = s["umbrella"]
    out = Path(out)
    top = read_topology(system_file(s, s["structure"]["topology"]))
    changed = apply_sequences(s, top)
    pairs = designed_pairs(s, top)
    box, conf = read_configuration(fused_conf)
    geo = junction_geometry(s, top, conf, box)

    strands, new_id, top_text, xyz, frag_strand = cut(s, top, conf, box, geo)
    frag_box = np.ones(3) * u["box"]
    extent = float(np.ptp(xyz[:, :3], axis=0).max())
    frag_pairs = [(new_id[i], new_id[j]) for i, j in pairs]
    formed = sum(pair_formed(xyz, frag_box, i, j) for i, j in frag_pairs)
    moved, gap = unbind(xyz, frag_strand, u["box"])
    d_sticky = min(hb_distance(moved, frag_box, i, j) for i, j in frag_pairs)
    junction = [geo["junction_body"]] + ([geo["junction_migration"]] if geo["junction_migration"] else [])
    frag_junction = [(new_id[i], new_id[j]) for i, j in junction]

    problems = []
    for tag in ("a", "b"):
        dist = geo[f"duplex_{tag}_distance"]
        if dist >= 0.75:
            problems.append(f"duplex {tag} is not paired (hydrogen-bond distance {dist:.2f})")
    if extent > u["box"] / 2:
        problems.append(f"fragment extent {extent:.1f} too large for box {u['box']}")
    if formed < len(pairs) - 1:
        problems.append(f"only {formed} of {len(pairs)} designed pairs formed in the fused structure")
    if gap < 3.0 or d_sticky < u["contact_distance"]:
        problems.append(f"unbound start not separated enough (gap {gap:.2f}, sticky pairs {d_sticky:.2f})")

    op = bond_op(frag_pairs, "sticky_bonds") + bond_op(frag_pairs, "sticky_distance").replace(
        "order_parameter = bond", "order_parameter = mindistance").replace("\n}", f"\ninterfaces = {u['contact_distance']}\n}}")
    for r in range(1, int(u["round_replicas"]) + 1):
        d = out / f"run_{r}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "fragment.top").write_text(top_text)
        write_configuration(d / "fragment_unbound.dat", moved, frag_box)
        (d / "op.txt").write_text(op)
        (d / "junction_pairs.op").write_text(bond_op(frag_junction, "junction"))
        (d / "wfile.txt").write_text(initial_weights(u, len(pairs)))
        (d / "input").write_text(vmmc_input(s, int(u["seed"]) + r, u["round_steps"],
                                            u["round_equilibration_steps"], u["round_steps"]))
        copy_seq_dep(s, d)
    frag_top = read_topology(out / "run_1" / "fragment.top")
    write_json(out / "fragment.json", {
        "nucleotides": len(new_id), "designed_pairs": frag_pairs, "pairs_formed_before_unbinding": formed,
        "junction_pairs": frag_junction, "sequence_changes": changed, "extent": extent, "gap_after_unbinding": gap,
        "closest_designed_pair_after_unbinding": d_sticky,
        "nick_a": (new_id[geo["nick_a"][0]], new_id[geo["nick_a"][1]]),
        "nick_b": (new_id[geo["nick_b"][0]], new_id[geo["nick_b"][1]]),
        "longest_backbone_bond": longest_backbone_bond(frag_top, moved, frag_box),
        "strand_sequences": ["".join(top["base"][x]) for x in strands]})
    write_json(out / "qc.json", {"step": "prepare_fragment", "status": "fail" if problems else "ok", "problems": problems})
    print(f"{s['name']}: fragment {len(new_id)} nt, {len(pairs)} designed pairs ({formed} formed before unbinding), "
          f"halves {gap:.1f} apart after unbinding" + (f" | PROBLEMS: {problems}" if problems else ""))
    return 1 if problems else 0


if __name__ == "__main__":
    raise ValueError("prepare_fragment takes an OxdnaSystem dict, not a yaml path")
