import sys
import warnings
from pathlib import Path
from typing import Optional

project_root = Path(__file__).resolve().parent.parent.parent / "src"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest

from parrhesia.actions import (
    DEFAULT_AUTO_RIPPLE_TIME_BINS,
    DFRegulation,
    DFRegulationPlan,
)
from parrhesia.flow_agent35.regen.types import (
    PredictedImprovement,
    Proposal,
    Window,
)


def _basic_plan() -> DFRegulationPlan:
    plan = DFRegulationPlan()
    plan.add(
        DFRegulation.from_flights(
            id="r1",
            tv_id="TV1",
            window_from="08:00",
            window_to="09:00",
            flights=["A1", "B2"],
            allowed_rate_per_hour=20,
        )
    )
    plan.add(
        DFRegulation.from_flights(
            id="r2",
            tv_id="TV2",
            window_from="10:00",
            window_to="11:00",
            flights=["C3", "D4"],
            allowed_rate_per_hour=15,
        )
    )
    plan.add(
        DFRegulation.from_flights(
            id="r3",
            tv_id="TV1",
            window_from="07:30",
            window_to="09:30",
            flights=["A1", "E5"],
            allowed_rate_per_hour=18,
        )
    )
    return plan


def test_metrics_and_payload_union_windows() -> None:
    plan = _basic_plan()

    assert plan.number_of_regulations() == 3
    assert plan.number_of_flows() == 3
    assert plan.number_of_flights_affected() == 5

    payload = plan.to_base_eval_payload(auto_ripple_time_bins=0, warn_on_auto_fallback=False)

    assert set(payload["flows"].keys()) == {"0", "1", "2"}
    assert payload["flows"]["0"] == ["A1", "B2"]
    assert payload["flows"]["2"] == ["A1", "E5"]

    targets = payload["targets"]
    assert targets["TV1"]["from"] == "07:30"
    assert targets["TV1"]["to"] == "09:30"
    assert targets["TV2"] == {"from": "10:00", "to": "11:00"}
    assert payload["auto_ripple_time_bins"] == 0


def test_explicit_ripples_override_auto() -> None:
    plan = _basic_plan()
    ripple_spec = {"TVX": {"from": "06:00", "to": "07:00"}}
    with warnings.catch_warnings(record=True) as caught:
        payload = plan.to_base_eval_payload(ripples=ripple_spec, warn_on_auto_fallback=True)
    assert not caught
    assert "ripples" in payload and payload["ripples"] == ripple_spec
    assert "auto_ripple_time_bins" not in payload


def test_auto_ripple_fallback_warns_once() -> None:
    plan = _basic_plan()
    with pytest.warns(UserWarning):
        payload = plan.to_base_eval_payload()
    assert payload["auto_ripple_time_bins"] == DEFAULT_AUTO_RIPPLE_TIME_BINS

    with warnings.catch_warnings(record=True) as caught:
        payload_again = plan.to_base_eval_payload()
    assert not caught
    assert payload_again["auto_ripple_time_bins"] == DEFAULT_AUTO_RIPPLE_TIME_BINS


def test_auto_ripple_explicit_override() -> None:
    plan = _basic_plan()
    with warnings.catch_warnings(record=True) as caught:
        payload = plan.to_base_eval_payload(auto_ripple_time_bins=4)
    assert not caught
    assert payload["auto_ripple_time_bins"] == 4


@pytest.mark.parametrize("value, expected", [("0", 0), (-3, 0), ("5", 5)])
def test_auto_ripple_coercion(value: object, expected: int) -> None:
    plan = _basic_plan()
    with warnings.catch_warnings(record=True) as caught:
        payload = plan.to_base_eval_payload(auto_ripple_time_bins=value, warn_on_auto_fallback=False)
    assert not caught
    assert payload["auto_ripple_time_bins"] == expected


@pytest.mark.parametrize(
    "ripples_arg, metadata, expect_ripples, expect_auto",
    [
        ({}, None, False, True),
        (None, {"auto_ripple_time_bins": 7}, False, True),
        (None, {"ripples": {"TVY": {"from": "05:00", "to": "06:00"}}}, True, False),
    ],
)
def test_ripples_and_metadata_precedence(
    ripples_arg: Optional[dict],
    metadata: Optional[dict],
    expect_ripples: bool,
    expect_auto: bool,
) -> None:
    plan = _basic_plan()
    plan.metadata = metadata
    with warnings.catch_warnings(record=True) as caught:
        payload = plan.to_base_eval_payload(ripples=ripples_arg)
    if metadata and "ripples" in metadata:
        assert not caught
    if expect_ripples:
        assert "ripples" in payload
    else:
        assert "ripples" not in payload
    if expect_auto:
        assert "auto_ripple_time_bins" in payload
    else:
        assert "auto_ripple_time_bins" not in payload


def test_invalid_auto_value_triggers_fallback() -> None:
    plan = _basic_plan()
    with pytest.warns(UserWarning):
        payload = plan.to_base_eval_payload(auto_ripple_time_bins="abc")
    assert payload["auto_ripple_time_bins"] == DEFAULT_AUTO_RIPPLE_TIME_BINS


def test_empty_ripples_fallback() -> None:
    plan = _basic_plan()
    with pytest.warns(UserWarning):
        payload = plan.to_base_eval_payload(ripples={})
    assert payload["auto_ripple_time_bins"] == DEFAULT_AUTO_RIPPLE_TIME_BINS


def test_from_payload_round_trip() -> None:
    plan = _basic_plan()
    payload = plan.to_base_eval_payload(auto_ripple_time_bins=2, warn_on_auto_fallback=False)
    width = max(1, len(str(plan.number_of_regulations() - 1)))
    flow_to_tv = {f"{idx:0{width}d}": reg.tv_id for idx, reg in enumerate(plan.regulations)}
    allowed = {f"{idx:0{width}d}": reg.allowed_rate_per_hour for idx, reg in enumerate(plan.regulations)}
    reconstructed = DFRegulationPlan.from_payload(
        payload,
        metadata={"flow_to_tv": flow_to_tv, "allowed_rates": allowed},
    )

    assert reconstructed.number_of_regulations() == plan.number_of_regulations()
    assert reconstructed.number_of_flights_affected() == plan.number_of_flights_affected()
    assert reconstructed.to_base_eval_payload(auto_ripple_time_bins=0, warn_on_auto_fallback=False)[
        "targets"
    ] == plan.to_base_eval_payload(auto_ripple_time_bins=0, warn_on_auto_fallback=False)["targets"]


def test_from_payload_requires_metadata_when_tv_ambiguous() -> None:
    payload = {
        "flows": {"0": ["A1"], "1": ["B2"], "2": ["C3"]},
        "targets": {
            "TV1": {"from": "07:00", "to": "08:00"},
            "TV2": {"from": "09:00", "to": "10:00"},
        },
    }
    with pytest.raises(ValueError):
        DFRegulationPlan.from_payload(payload)

    plan = DFRegulationPlan.from_payload(
        payload,
        metadata={"flow_to_tv": {"0": "TV1", "1": "TV1", "2": "TV2"}},
    )
    assert plan.number_of_regulations() == 3


def test_from_proposal_builds_regulations() -> None:
    proposal = Proposal(
        hotspot_id="HOT1",
        controlled_volume="TV_CTRL",
        window=Window(start_bin=18, end_bin=19),
        flows_info=[
            {"flow_id": 0, "control_tv_id": "TV_CTRL", "R_i": 18, "r0_i": 25, "lambda_cut_i": 7},
            {"flow_id": 1, "control_tv_id": "TV_ALT", "R_i": 20, "r0_i": 28, "lambda_cut_i": 4},
        ],
        predicted_improvement=PredictedImprovement(1.0, 2.0),
        diagnostics={"time_bin_minutes": 30},
        target_cells=[("TV_CTRL", 18)],
        ripple_cells=[("TV_ALT", 12)],
        target_tvs=["TV_CTRL"],
        ripple_tvs=["TV_ALT"],
    )
    flights_by_flow = {
        0: [{"flight_id": "A1"}, {"flight_id": "B2"}],
        "1": ["C3", "D4"],
    }
    plan = DFRegulationPlan.from_proposal(proposal, flights_by_flow)

    assert plan.number_of_regulations() == 2
    targets = plan.to_base_eval_payload(auto_ripple_time_bins=0, warn_on_auto_fallback=False)["targets"]
    assert targets["TV_CTRL"] == {"from": "09:00", "to": "10:00"}
    assert any("baseline_rate_per_hour" in (reg.metadata or {}) for reg in plan.regulations)
    assert plan.metadata["proposal_hotspot_id"] == "HOT1"
