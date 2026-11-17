"""Shared helpers: system file, oxDNA topology/configuration, designed pairs, order-parameter files.

Conventions
-----------
* Topology: classic oxDNA format. Each line is ``strand base n3 n5``; n5 = -1 marks a strand's 5' end.
  ``strands`` below always lists nucleotide ids from 5' to 3'.
* Both sticky ends sit at the 5' end of their strands (``sticky_strand`` in the system file).
  Sticky end A is the longer (or equally long) one. Base k of B (5' -> 3') pairs base len_A - 1 - k of A,
  so the 5'-terminal base of B pairs the last base of A next to A's duplex. Only complementary
  positions are designed pairs, so a mismatch (mutant) simply drops out.
"""
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np

COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G"}
TERMS = ["FENE", "BEXC", "STCK", "NEXC", "HB", "CRSTCK", "CXSTCK", "DH"]
OXDNA_VERSION_FILE = Path("/usr/local/share/oxdna/VERSION")


def require_system_keys(system, keys):
    missing = [key for key in keys if key not in system]
    if missing:
        raise ValueError(f"system dict is missing {missing}")
    return system


def system_file(system, relpath):
    p = Path(relpath)
    return p if p.is_absolute() else system["_dir"] / p


# ------------------------------------------------------------------ topology
def read_topology(path):
    """-> dict with base, strand, n3, n5 (arrays) and strands {strand: ids 5'->3'}."""
    lines = [l for l in Path(path).read_text().splitlines() if l.strip()]
    n = int(lines[0].split()[0])
    rows = [l.split() for l in lines[1:1 + n]]
    top = {"base": np.array([r[1] for r in rows]), "strand": np.array([int(r[0]) for r in rows]),
           "n3": np.array([int(r[2]) for r in rows]), "n5": np.array([int(r[3]) for r in rows])}
    strands = {}
    for s in np.unique(top["strand"]):
        five = [i for i in np.where(top["strand"] == s)[0] if top["n5"][i] == -1]
        assert len(five) == 1, f"strand {s} has {len(five)} 5' ends"
        order, cur = [], five[0]
        while cur != -1:
            order.append(int(cur))
            cur = top["n3"][cur]
        strands[int(s)] = order
    top["strands"] = strands
    return top


def write_topology(top, path):
    lines = [f"{len(top['base'])} {len(top['strands'])}"]
    lines += [f"{s} {b} {a} {c}" for s, b, a, c in zip(top["strand"], top["base"], top["n3"], top["n5"])]
    Path(path).write_text("\n".join(lines) + "\n")


def apply_sequences(system, top):
    """Set the sticky-end bases of the topology to the sequences in the system file.

    The sequence is applied to sticky_strand and to every strand in other_sticky_strands (the other
    sticky ends of the same molecule, which carry the same sequence, e.g. the second end of a linker).

    Returns the changed positions; a non-empty list means this system is a mutant of the structure it
    was built from. Only the topology changes, never the configuration.
    """
    changed = []
    for p in ("partner_a", "partner_b"):
        for strand in [system[p]["sticky_strand"]] + list(system[p].get("other_sticky_strands", [])):
            ids = top["strands"][strand]
            for k, (i, b) in enumerate(zip(ids, system[p]["sticky_sequence"])):
                if top["base"][i] != b:
                    changed.append(f"{p} strand {strand} position {k}: {top['base'][i]} -> {b}")
                    top["base"][i] = b
    return changed


def sticky_ids(system, top):
    """Nucleotide ids (5'->3') of sticky end A and of sticky end B."""
    a = top["strands"][system["partner_a"]["sticky_strand"]][:len(system["partner_a"]["sticky_sequence"])]
    b = top["strands"][system["partner_b"]["sticky_strand"]][:len(system["partner_b"]["sticky_sequence"])]
    return a, b


def designed_pairs(system, top):
    """Designed sticky-end pairs (id in A, id in B), complementary positions only."""
    a, b = sticky_ids(system, top)
    pairs = [(a[len(a) - 1 - k], b[k]) for k in range(len(b))]
    return [(i, j) for i, j in pairs if COMPLEMENT[top["base"][i]] == top["base"][j]]


def partner_strands(system, top):
    """Boolean masks: nucleotides of partner A and of partner B."""
    return (np.isin(top["strand"], system["partner_a"]["strands"]),
            np.isin(top["strand"], system["partner_b"]["strands"]))


# ------------------------------------------------------------------ configuration
def read_configuration(path):
    """-> (box edge array, (N, 15) array)."""
    lines = Path(path).read_text().splitlines()
    box = np.array([float(v) for v in lines[1].split("=")[1].split()[:3]])
    conf = np.array([[float(v) for v in l.split()] for l in lines[3:] if l.strip()])
    return box, conf


def write_configuration(path, conf, box, t=0):
    b = " ".join(f"{v:g}" for v in np.ones(3) * box)
    body = "\n".join(" ".join(f"{v:.6f}" for v in row) for row in conf)
    Path(path).write_text(f"t = {t}\nb = {b}\nE = 0 0 0\n{body}\n")


def backbone_sites(conf):
    """oxDNA2 backbone sites: r - 0.34 a1 + 0.3408 a2, with a2 = a3 x a1."""
    a1, a3 = conf[:, 3:6], conf[:, 6:9]
    return conf[:, 0:3] - 0.34 * a1 + 0.3408 * np.cross(a3, a1)


def longest_backbone_bond(top, conf, box):
    """Largest backbone-site distance between bonded nucleotides (minimum image).
    The FENE bond diverges at 1.0025 oxDNA units, so a value near 1 means a broken structure."""
    b = backbone_sites(conf)
    i = np.array([k for k, j in enumerate(top["n3"]) if j != -1])
    d = b[i] - b[top["n3"][i]]
    d -= box * np.round(d / box)
    return float(np.linalg.norm(d, axis=1).max())


def hb_distance(conf, box, i, j):
    hb = conf[:, 0:3] + 0.4 * conf[:, 3:6]
    d = hb[i] - hb[j]
    d -= box * np.round(d / box)
    return float(np.linalg.norm(d))


def pair_formed(conf, box, i, j):
    """Geometric base-pair test: hydrogen-bonding sites within 0.75 and bases antiparallel."""
    return hb_distance(conf, box, i, j) < 0.75 and conf[i, 3:6] @ conf[j, 3:6] < -0.5


def nearest_partner(conf, box, top, i):
    """Complementary nucleotide on another strand with the closest hydrogen-bonding site to i."""
    hb = conf[:, 0:3] + 0.4 * conf[:, 3:6]
    d = hb - hb[i]
    d -= box * np.round(d / box)
    dist = np.linalg.norm(d, axis=1)
    dist[(top["strand"] == top["strand"][i]) | (top["base"] != COMPLEMENT[top["base"][i]])] = np.inf
    j = int(np.argmin(dist))
    return j, float(dist[j])


def junction_geometry(system, top, conf, box):
    """Duplex partners next to the sticky duplex, the two nicks and the junction pairs.

    A's strand continues past its sticky end into a duplex; the 3' end of the partner strand of that
    duplex faces the 5'-terminal base of B (nick A). Likewise the 3' end of B's duplex partner is the
    designed partner of B[len_B] and faces A[len_A - len_B] (nick B); when A is longer, a flap of
    len_A - len_B nucleotides of A sits in this nick. Junction: B[len_B] pairs its designed partner
    (body) or, if complementary, A[len_A - len_B - 1] (one-nucleotide branch migration).
    """
    a_ids, b_ids = sticky_ids(system, top)
    la, lb = len(a_ids), len(b_ids)
    out = {}
    for tag, p, l in (("a", "partner_a", la), ("b", "partner_b", lb)):
        strand = top["strands"][system[p]["sticky_strand"]]
        j, dist = nearest_partner(conf, box, top, strand[l + 1])       # one base into the duplex
        three = top["strands"][int(top["strand"][j])][-1]
        if COMPLEMENT[top["base"][three]] != top["base"][strand[l]]:
            raise ValueError(f"duplex {tag}: 3' end {three} is not the partner of {strand[l]}")
        out[f"duplex_{tag}_strand"] = int(top["strand"][j])
        out[f"duplex_{tag}_3end"] = int(three)
        out[f"duplex_{tag}_distance"] = dist
    strand_b = top["strands"][system["partner_b"]["sticky_strand"]]
    out["nick_a"] = (out["duplex_a_3end"], b_ids[0])
    out["nick_b"] = (out["duplex_b_3end"], a_ids[la - lb])
    out["junction_body"] = (strand_b[lb], out["duplex_b_3end"])
    k = la - lb - 1
    out["junction_migration"] = ((strand_b[lb], a_ids[k]) if k >= 0 and
                                 COMPLEMENT[top["base"][strand_b[lb]]] == top["base"][a_ids[k]] else None)
    return out


# ------------------------------------------------------------------ order-parameter files
def bond_op(pairs, name):
    lines = "\n".join(f"pair{k} = {i}, {j}" for k, (i, j) in enumerate(pairs, start=1))
    return f"{{\norder_parameter = bond\nname = {name}\n{lines}\n}}\n"


def read_op(path):
    """'pairN = i, j' lines of an order-parameter file -> list of sorted (i, j)."""
    return [tuple(sorted(int(x) for x in l.split("=")[1].split(",")))
            for l in Path(path).read_text().splitlines() if l.strip().startswith("pair")]


# ------------------------------------------------------------------ input files
def fill_template(name, values):
    """templates/<name> with every {KEY} replaced by values[KEY]."""
    text = (Path(__file__).resolve().parents[1] / "templates" / name).read_text()
    for k, v in values.items():
        text = text.replace("{" + k + "}", str(v))
    left = [w.split("}")[0] for w in text.split("{")[1:] if "}" in w and w.split("}")[0].isupper()]
    assert not left, f"unfilled fields in {name}: {left}"
    return text


def cuda_block(backend):
    if backend == "CUDA":
        return (
            "backend_precision = mixed\n"
            "use_edge = 1\n"
            "edge_n_forces = 1\n"
            "CUDA_list = verlet\n"
            "CUDA_sort_every = 0\n"
        )
    if backend == "CPU":
        return ""
    raise ValueError(f"md_backend must be CPU or CUDA, got {backend!r}")


def model_values(system, temperature_c):
    o = system["oxdna"]
    backend = system["md"]["md_backend"]
    return {"T_C": f"{temperature_c:g}", "SALT": system["salt_M"],
            "SEQ_DEP_KEY": o["seq_dep_key"], "SEQ_DEP_FILE": Path(o["seq_dep_file"]).name,
            "BACKEND": backend, "CUDA_BLOCK": cuda_block(backend)}


def copy_seq_dep(system, folder):
    src = system_file(system, system["oxdna"]["seq_dep_file"])
    (Path(folder) / src.name).write_bytes(src.read_bytes())


# ------------------------------------------------------------------ statistics and output
def tau_int(x, c=5.0):
    """Integrated autocorrelation time in samples (Sokal's self-consistent window)."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    if n < 4 or x.std() == 0:
        return np.nan
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= acf[0]
    tau = 1.0
    for m in range(1, n):
        tau = 1 + 2 * acf[1:m + 1].sum()
        if m >= c * tau:
            break
    return max(tau, 1.0)


def mean_sem(x):
    """(mean, standard error corrected for autocorrelation, tau in samples)."""
    x = np.asarray(x, float)
    tau = tau_int(x)
    tau = 1.0 if np.isnan(tau) else tau
    return float(x.mean()), float(x.std(ddof=1) * np.sqrt(2 * tau / len(x))), float(tau)


def write_json(path, data):
    conv = lambda v: v.item() if hasattr(v, "item") else str(v)
    Path(path).write_text(json.dumps(data, indent=1, default=conv) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def file_sha256(path):
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def oxdna_version(executable="oxDNA"):
    """Version reported by the oxDNA binary, or the image VERSION file."""
    result = subprocess.run([executable], capture_output=True, text=True)
    text = f"{result.stdout or ''}\n{result.stderr or ''}"
    match = re.search(r"\b(\d+\.\d+\.\d+)\b", text)
    if match:
        return match.group(1)
    if OXDNA_VERSION_FILE.is_file():
        version = OXDNA_VERSION_FILE.read_text().strip()
        if re.fullmatch(r"\d+\.\d+\.\d+", version):
            return version
        raise ValueError(f"oxDNA VERSION file is {version!r}")
    raise ValueError(f"{executable} did not report a version. Output: {text[:800]}")


def refuse_vmmc_360(version):
    if version == "3.6.0":
        raise ValueError("oxDNA 3.6.0 must not be used for VMMC with DNA2")


def stamp_qc(qc_path, seq_dep, executable="oxDNA"):
    """Record the oxDNA version and the sequence-file checksum on an existing qc.json."""
    data = read_json(qc_path)
    data["oxdna_version"] = oxdna_version(executable)
    data["seq_dep_sha256"] = file_sha256(seq_dep)
    write_json(qc_path, data)
    return data
