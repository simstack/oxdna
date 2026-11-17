import copy
from pathlib import Path

from oxdna.models.system import OxdnaSystem

_TOP_KEYS = (
    "name",
    "structure",
    "partner_a",
    "partner_b",
    "temperature_C",
    "salt_M",
    "concentration_M",
    "seq_dep_key",
    "seq_dep_file",
    "stop_on_qc_fail",
    "md",
    "umbrella",
)
_PARTNER_KEYS = ("strands", "sticky_strand", "other_sticky_strands", "sticky_sequence")
_MD_KEYS = (
    "md_backend",
    "fusion_temperature_C",
    "fusion_steps",
    "trap_stiffness",
    "trap_r0",
    "box",
    "equilibration_steps",
    "steps",
    "print_conf_interval",
    "print_every",
    "discard_fraction",
)
_UMBRELLA_KEYS = (
    "duplex_bp_a",
    "duplex_bp_b",
    "box",
    "contact_distance",
    "initial_weights",
    "seed",
    "round_steps",
    "round_replicas",
    "round_equilibration_steps",
    "weight_cap",
    "min_transitions",
    "production_steps",
    "production_replicas",
    "print_every",
    "extrapolate_from_C",
    "extrapolate_to_C",
    "extrapolate_step_C",
    "target_fractions",
    "far_from_simulation_C",
    "max_rounds",
)
_WEIGHT_KEYS = ("unbound_far", "unbound_near", "bound")


def _missing(mapping, keys, label):
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"{label} is missing {missing}")


def build_system_dict(settings, directory):
    """Copy explicit settings into the dict the oxDNA scripts read.

    ``directory`` is the folder that holds the topology, configuration and
    sequence-parameter file named by the settings. Relative paths resolve there.
    """
    if not isinstance(settings, dict):
        raise ValueError(f"settings must be a dict, got {type(settings).__name__}")
    _missing(settings, _TOP_KEYS, "system settings")
    system = copy.deepcopy(settings)
    _missing(system["structure"], ("topology", "configuration"), "structure")
    for label in ("partner_a", "partner_b"):
        _missing(system[label], _PARTNER_KEYS, label)
        sequence = system[label]["sticky_sequence"]
        if not isinstance(sequence, str) or sequence == "" or sequence != sequence.upper():
            raise ValueError(f"{label}.sticky_sequence must be a non-empty uppercase DNA sequence")
    if len(system["partner_a"]["sticky_sequence"]) < len(system["partner_b"]["sticky_sequence"]):
        raise ValueError("partner_a must carry the longer (or equally long) sticky end")
    _missing(system["md"], _MD_KEYS, "md")
    backend = system["md"]["md_backend"]
    if backend not in ("CPU", "CUDA"):
        raise ValueError(f"md_backend must be CPU or CUDA, got {backend!r}")
    _missing(system["umbrella"], _UMBRELLA_KEYS, "umbrella")
    _missing(system["umbrella"]["initial_weights"], _WEIGHT_KEYS, "initial_weights")
    umbrella = system["umbrella"]
    umbrella["extrapolate_C"] = [
        umbrella["extrapolate_from_C"],
        umbrella["extrapolate_to_C"],
        umbrella["extrapolate_step_C"],
    ]
    if not isinstance(system["stop_on_qc_fail"], bool):
        raise ValueError("stop_on_qc_fail must be a bool")
    system["oxdna"] = {
        "seq_dep_key": system["seq_dep_key"],
        "seq_dep_file": system["seq_dep_file"],
    }
    system["_dir"] = Path(directory)
    return system


def settings_from_model(model: OxdnaSystem):
    """Read every scientific field off an OxdnaSystem. File names are filled by the caller."""
    def partner(value):
        return {
            "strands": list(value.strands),
            "sticky_strand": value.sticky_strand,
            "other_sticky_strands": list(value.other_sticky_strands),
            "sticky_sequence": value.sticky_sequence,
        }

    md = model.md
    umbrella = model.umbrella
    weights = umbrella.initial_weights
    return {
        "name": model.name,
        "partner_a": partner(model.partner_a),
        "partner_b": partner(model.partner_b),
        "temperature_C": model.temperature_C,
        "salt_M": model.salt_M,
        "concentration_M": model.concentration_M,
        "seq_dep_key": model.seq_dep_key,
        "stop_on_qc_fail": model.stop_on_qc_fail,
        "use_weight_overrides": model.use_weight_overrides,
        "md": {
            "md_backend": md.md_backend.value if hasattr(md.md_backend, "value") else md.md_backend,
            "fusion_temperature_C": md.fusion_temperature_C,
            "fusion_steps": md.fusion_steps,
            "trap_stiffness": md.trap_stiffness,
            "trap_r0": md.trap_r0,
            "box": md.box,
            "equilibration_steps": md.equilibration_steps,
            "steps": md.steps,
            "print_conf_interval": md.print_conf_interval,
            "print_every": md.print_every,
            "discard_fraction": md.discard_fraction,
        },
        "umbrella": {
            "duplex_bp_a": umbrella.duplex_bp_a,
            "duplex_bp_b": umbrella.duplex_bp_b,
            "box": umbrella.box,
            "contact_distance": umbrella.contact_distance,
            "initial_weights": {
                "unbound_far": weights.unbound_far,
                "unbound_near": weights.unbound_near,
                "bound": list(weights.bound),
            },
            "seed": umbrella.seed,
            "round_steps": umbrella.round_steps,
            "round_replicas": umbrella.round_replicas,
            "round_equilibration_steps": umbrella.round_equilibration_steps,
            "weight_cap": umbrella.weight_cap,
            "min_transitions": umbrella.min_transitions,
            "production_steps": umbrella.production_steps,
            "production_replicas": umbrella.production_replicas,
            "print_every": umbrella.print_every,
            "extrapolate_from_C": umbrella.extrapolate_from_C,
            "extrapolate_to_C": umbrella.extrapolate_to_C,
            "extrapolate_step_C": umbrella.extrapolate_step_C,
            "target_fractions": list(umbrella.target_fractions),
            "far_from_simulation_C": umbrella.far_from_simulation_C,
            "max_rounds": umbrella.max_rounds,
        },
    }


def materialize_system(model: OxdnaSystem, folder: Path):
    """Write the three input files and return the script dict."""
    folder.mkdir(parents=True, exist_ok=True)
    topology = model.topology.get(local_dir=folder)
    configuration = model.configuration.get(local_dir=folder)
    seq_dep = model.seq_dep_file.get(local_dir=folder)
    settings = settings_from_model(model)
    settings["structure"] = {"topology": topology.name, "configuration": configuration.name}
    settings["seq_dep_file"] = seq_dep.name
    return build_system_dict(settings, folder)
