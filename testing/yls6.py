import copy
from pathlib import Path

from simstack.models import FileStack

from oxdna.models.system import (
    OxdnaMd,
    OxdnaPartner,
    OxdnaSystem,
    OxdnaUmbrella,
    OxdnaUmbrellaWeights,
)


def example_dir():
    return Path(__file__).resolve().parents[1] / "data" / "example_YLs6"


def yls6_production_settings():
    """Scientific YLs6 settings. Every value is set here; nothing is filled in later."""
    return {
        "name": "YLs6",
        "structure": {"topology": "YLs6.top", "configuration": "YLs6.dat"},
        "partner_a": {
            "strands": [1, 2, 3],
            "sticky_strand": 3,
            "other_sticky_strands": [1, 2],
            "sticky_sequence": "CGATTGACTCTC",
        },
        "partner_b": {
            "strands": [4, 5],
            "sticky_strand": 4,
            "other_sticky_strands": [5],
            "sticky_sequence": "GAGAGT",
        },
        "temperature_C": 14.3,
        "salt_M": 0.118,
        "concentration_M": 6.0e-4,
        "seq_dep_key": "seq_dep_file",
        "seq_dep_file": "oxDNA2_sequence_dependent_parameters.txt",
        "stop_on_qc_fail": True,
        "use_weight_overrides": False,
        "md": {
            "md_backend": "CUDA",
            "fusion_temperature_C": 37.0,
            "fusion_steps": 10000000,
            "trap_stiffness": 0.2,
            "trap_r0": 1.2,
            "box": 100.0,
            "equilibration_steps": 100000,
            "steps": 50000000,
            "print_conf_interval": 50000,
            "print_every": 1000,
            "discard_fraction": 0.1,
        },
        "umbrella": {
            "duplex_bp_a": 12,
            "duplex_bp_b": 12,
            "box": 30.0,
            "contact_distance": 1.5,
            "initial_weights": {
                "unbound_far": 1.65,
                "unbound_near": 26228.51,
                "bound": [599763.11, 96780.58, 3412.92, 606.67, 26.91, 2.81, 1.5],
            },
            "seed": 6000,
            "round_steps": 100000,
            "round_replicas": 3,
            "round_equilibration_steps": 10000,
            "weight_cap": 20.0,
            "min_transitions": 2,
            "production_steps": 4000000,
            "production_replicas": 6,
            "print_every": 10,
            "extrapolate_from_C": 0.0,
            "extrapolate_to_C": 80.0,
            "extrapolate_step_C": 2.0,
            "target_fractions": [0.5, 0.7071],
            "far_from_simulation_C": 6.0,
            "max_rounds": 20,
        },
    }


def yls6_smoke_settings():
    """YLs6 structure and sequences with short CPU runs for a local container test."""
    settings = copy.deepcopy(yls6_production_settings())
    settings["stop_on_qc_fail"] = False
    settings["md"]["md_backend"] = "CPU"
    settings["md"]["fusion_steps"] = 2000
    settings["md"]["equilibration_steps"] = 200
    settings["md"]["steps"] = 2000
    settings["md"]["print_conf_interval"] = 500
    settings["md"]["print_every"] = 100
    settings["umbrella"]["round_steps"] = 2000
    settings["umbrella"]["round_replicas"] = 1
    settings["umbrella"]["round_equilibration_steps"] = 100
    settings["umbrella"]["production_steps"] = 2000
    settings["umbrella"]["production_replicas"] = 1
    settings["umbrella"]["max_rounds"] = 1
    settings["umbrella"]["min_transitions"] = 1
    return settings


def _partner(data):
    return OxdnaPartner(
        strands=data["strands"],
        sticky_strand=data["sticky_strand"],
        other_sticky_strands=data["other_sticky_strands"],
        sticky_sequence=data["sticky_sequence"],
    )


def _md(data):
    return OxdnaMd(
        md_backend=data["md_backend"],
        fusion_temperature_C=data["fusion_temperature_C"],
        fusion_steps=data["fusion_steps"],
        trap_stiffness=data["trap_stiffness"],
        trap_r0=data["trap_r0"],
        box=data["box"],
        equilibration_steps=data["equilibration_steps"],
        steps=data["steps"],
        print_conf_interval=data["print_conf_interval"],
        print_every=data["print_every"],
        discard_fraction=data["discard_fraction"],
    )


def _umbrella(data):
    weights = data["initial_weights"]
    return OxdnaUmbrella(
        duplex_bp_a=data["duplex_bp_a"],
        duplex_bp_b=data["duplex_bp_b"],
        box=data["box"],
        contact_distance=data["contact_distance"],
        initial_weights=OxdnaUmbrellaWeights(
            unbound_far=weights["unbound_far"],
            unbound_near=weights["unbound_near"],
            bound=weights["bound"],
        ),
        seed=data["seed"],
        round_steps=data["round_steps"],
        round_replicas=data["round_replicas"],
        round_equilibration_steps=data["round_equilibration_steps"],
        weight_cap=data["weight_cap"],
        min_transitions=data["min_transitions"],
        production_steps=data["production_steps"],
        production_replicas=data["production_replicas"],
        print_every=data["print_every"],
        extrapolate_from_C=data["extrapolate_from_C"],
        extrapolate_to_C=data["extrapolate_to_C"],
        extrapolate_step_C=data["extrapolate_step_C"],
        target_fractions=data["target_fractions"],
        far_from_simulation_C=data["far_from_simulation_C"],
        max_rounds=data["max_rounds"],
    )


def yls6_system(settings) -> OxdnaSystem:
    root = example_dir()
    structure = settings["structure"]
    return OxdnaSystem(
        name=settings["name"],
        topology=FileStack.from_local_file(root / structure["topology"], in_memory=True),
        configuration=FileStack.from_local_file(root / structure["configuration"], in_memory=True),
        seq_dep_file=FileStack.from_local_file(root / settings["seq_dep_file"], in_memory=True),
        seq_dep_key=settings["seq_dep_key"],
        partner_a=_partner(settings["partner_a"]),
        partner_b=_partner(settings["partner_b"]),
        temperature_C=settings["temperature_C"],
        salt_M=settings["salt_M"],
        concentration_M=settings["concentration_M"],
        md=_md(settings["md"]),
        umbrella=_umbrella(settings["umbrella"]),
        stop_on_qc_fail=settings["stop_on_qc_fail"],
        use_weight_overrides=settings["use_weight_overrides"],
    )
