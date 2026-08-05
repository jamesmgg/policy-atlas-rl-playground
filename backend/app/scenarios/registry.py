from __future__ import annotations

from .classic import CLASSIC_SPECS
from .driving import DRIVING_SPECS
from .spec import ScenarioSpec

SCENARIOS: dict[str, ScenarioSpec] = {
    spec.id: spec for spec in (*DRIVING_SPECS, *CLASSIC_SPECS)
}

DEFAULT_SCENARIO = "apex-gp"


def get_spec(scenario_id: str) -> ScenarioSpec:
    try:
        return SCENARIOS[scenario_id]
    except KeyError:
        raise KeyError(f"unknown scenario: {scenario_id}") from None


def list_specs() -> list[ScenarioSpec]:
    return list(SCENARIOS.values())
