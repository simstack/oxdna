from enum import Enum
from typing import Optional

from odmantic import EmbeddedModel, Model, Reference
from pydantic import model_validator
from simstack.models import FileStack, simstack_model
from simstack.util.cleaned_json_schema import cleaned_json_schema
from simstack.util.generate_ui_schema import generate_ui_schema


class OxdnaBackend(str, Enum):
    CPU = "CPU"
    CUDA = "CUDA"


@simstack_model
class OxdnaPartner(EmbeddedModel):
    field_name: str = "OxdnaPartner"
    strands: list[int]
    sticky_strand: int
    other_sticky_strands: list[int]
    sticky_sequence: str


@simstack_model
class OxdnaMd(EmbeddedModel):
    field_name: str = "OxdnaMd"
    md_backend: OxdnaBackend
    fusion_temperature_C: float
    fusion_steps: int
    trap_stiffness: float
    trap_r0: float
    box: float
    equilibration_steps: int
    steps: int
    print_conf_interval: int
    print_every: int
    discard_fraction: float


@simstack_model
class OxdnaUmbrellaWeights(EmbeddedModel):
    field_name: str = "OxdnaUmbrellaWeights"
    unbound_far: float
    unbound_near: float
    bound: list[float]


@simstack_model
class OxdnaUmbrella(EmbeddedModel):
    field_name: str = "OxdnaUmbrella"
    duplex_bp_a: int
    duplex_bp_b: int
    box: float
    contact_distance: float
    initial_weights: OxdnaUmbrellaWeights
    seed: int
    round_steps: int
    round_replicas: int
    round_equilibration_steps: int
    weight_cap: float
    min_transitions: int
    production_steps: int
    production_replicas: int
    print_every: int
    extrapolate_from_C: float
    extrapolate_to_C: float
    extrapolate_step_C: float
    target_fractions: list[float]
    far_from_simulation_C: float
    max_rounds: int


@simstack_model
class OxdnaSystem(Model):
    field_name: str = "OxdnaSystem"
    name: str
    topology: FileStack = Reference()
    configuration: FileStack = Reference()
    seq_dep_file: FileStack = Reference()
    seq_dep_key: str
    partner_a: OxdnaPartner
    partner_b: OxdnaPartner
    temperature_C: float
    salt_M: float
    concentration_M: float
    md: OxdnaMd
    umbrella: OxdnaUmbrella
    stop_on_qc_fail: bool
    use_weight_overrides: bool
    weight_overrides: Optional[FileStack] = None

    @model_validator(mode="before")
    @classmethod
    def sync_weight_overrides(cls, data):
        if not isinstance(data, dict):
            return data
        if "use_weight_overrides" not in data:
            raise ValueError("use_weight_overrides is required")
        if data.get("use_weight_overrides"):
            if data.get("weight_overrides") is None:
                raise ValueError("weight_overrides is required when use_weight_overrides is true")
        else:
            data["weight_overrides"] = None
        return data

    @classmethod
    def json_schema(cls, recursive=True):
        schema = cleaned_json_schema(cls)
        schema["title"] = cls.__name__
        props = schema["properties"]
        overrides = props.pop("weight_overrides", None)
        schema.setdefault("dependencies", {})["use_weight_overrides"] = {
            "oneOf": [
                {"properties": {"use_weight_overrides": {"const": False}}},
                {
                    "properties": {
                        "use_weight_overrides": {"const": True},
                        "weight_overrides": overrides,
                    }
                },
            ]
        }
        return schema

    @classmethod
    def ui_schema(cls):
        ui = generate_ui_schema(cls)
        ui["field_name"] = {"ui:widget": "hidden"}
        ui["use_weight_overrides"] = {
            "ui:widget": "checkbox",
            "ui:title": "Use an accepted weights file",
        }
        ui.setdefault("weight_overrides", {})["ui:condition"] = {"use_weight_overrides": True}
        return ui


@simstack_model
class OxdnaConfigurationResult(Model):
    field_name: str = "OxdnaConfigurationResult"
    system_name: str
    step: str
    qc_status: str
    problems: list[str]
    warnings: list[str]
    oxdna_version: str
    seq_dep_sha256: str
    folder: FileStack = Reference()
    configuration: FileStack = Reference()


@simstack_model
class OxdnaFolderResult(Model):
    field_name: str = "OxdnaFolderResult"
    system_name: str
    step: str
    qc_status: str
    problems: list[str]
    warnings: list[str]
    oxdna_version: str
    seq_dep_sha256: str
    folder: FileStack = Reference()
