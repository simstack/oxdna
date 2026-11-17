import json

import pytest
from pydantic import ValidationError

from oxdna.models.system import OxdnaSystem
from oxdna.part1_md import prepare_fusion
from oxdna.testing.yls6 import example_dir, yls6_production_settings, yls6_smoke_settings
from oxdna.util.oxdna_io import designed_pairs, read_topology
from oxdna.util.system_dict import build_system_dict


def test_production_settings_match_the_yls6_parameters():
    settings = yls6_production_settings()
    assert settings["name"] == "YLs6"
    assert settings["partner_a"]["sticky_sequence"] == "CGATTGACTCTC"
    assert settings["partner_b"]["sticky_sequence"] == "GAGAGT"
    assert settings["partner_a"]["strands"] == [1, 2, 3]
    assert settings["partner_b"]["sticky_strand"] == 4
    assert settings["temperature_C"] == 14.3
    assert settings["salt_M"] == 0.118
    assert settings["concentration_M"] == 6.0e-4
    assert settings["seq_dep_key"] == "seq_dep_file"
    assert settings["md"]["md_backend"] == "CUDA"
    assert settings["md"]["fusion_steps"] == 10000000
    assert settings["md"]["steps"] == 50000000
    assert settings["md"]["trap_stiffness"] == 0.2
    assert settings["md"]["trap_r0"] == 1.2
    assert settings["umbrella"]["initial_weights"]["bound"][0] == 599763.11
    assert settings["umbrella"]["production_replicas"] == 6
    assert settings["umbrella"]["target_fractions"] == [0.5, 0.7071]
    assert settings["stop_on_qc_fail"] is True
    assert settings["umbrella"]["max_rounds"] == 20
    assert settings["umbrella"]["min_transitions"] == 2


def test_smoke_settings_keep_the_sequences_and_shorten_the_runs():
    smoke = yls6_smoke_settings()
    production = yls6_production_settings()
    assert smoke["partner_a"] == production["partner_a"]
    assert smoke["partner_b"] == production["partner_b"]
    assert smoke["temperature_C"] == production["temperature_C"]
    assert smoke["salt_M"] == production["salt_M"]
    assert smoke["md"]["md_backend"] == "CPU"
    assert smoke["md"]["fusion_steps"] == 2000
    assert smoke["md"]["steps"] == 2000
    assert smoke["umbrella"]["round_replicas"] == 1
    assert smoke["umbrella"]["production_replicas"] == 1
    assert smoke["stop_on_qc_fail"] is False


def test_missing_system_field_raises():
    settings = yls6_production_settings()
    del settings["temperature_C"]
    with pytest.raises(ValueError, match="temperature_C"):
        build_system_dict(settings, example_dir())
    with pytest.raises(ValidationError):
        OxdnaSystem(name="YLs6")


def test_prepare_fusion_writes_traps_for_the_designed_pairs(tmp_path):
    data = example_dir()
    system = build_system_dict(yls6_production_settings(), data)
    topology = read_topology(data / "YLs6.top")
    pairs = designed_pairs(system, topology)
    assert len(pairs) == 6
    out = tmp_path / "fusion"
    prepare_fusion.main(system, out)
    traps = (out / "mutual_trap.txt").read_text()
    assert traps.count("type = mutual_trap") == 2 * len(pairs)
    text = (out / "input").read_text()
    assert "backend = CUDA" in text
    assert "seq_dep_file = oxDNA2_sequence_dependent_parameters.txt" in text
    qc = json.loads((out / "qc.json").read_text())
    assert qc["status"] == "ok"
    assert qc["n_designed_pairs"] == len(pairs)
    assert qc["n_traps"] == 2 * len(pairs)


def test_cpu_backend_omits_cuda_keys(tmp_path):
    system = build_system_dict(yls6_smoke_settings(), example_dir())
    out = tmp_path / "fusion"
    prepare_fusion.main(system, out)
    text = (out / "input").read_text()
    assert "backend = CPU" in text
    assert "CUDA_list" not in text
    assert "CUDA_sort_every" not in text
