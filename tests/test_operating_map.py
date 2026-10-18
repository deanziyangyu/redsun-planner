from __future__ import annotations

import pytest

from planner.acquisition_space import AcquisitionEnvelope, AxisThreshold
from planner.operating_map import (
    broom_trace,
    build_operating_payload,
    simulation_operating_payload,
    snap_traces,
)
from planner.simulation import (
    FocusedSpec,
    MockTarget,
    SearchImagingSpec,
    StageProfile,
    SurveySpec,
    simulate_search_imaging,
)


def _envelope() -> AcquisitionEnvelope:
    return AcquisitionEnvelope(
        velocity_um_s=AxisThreshold(10.0, 100.0),
        exposure_s=AxisThreshold(0.05, 0.1),
        width_um=AxisThreshold(50.0, 150.0),
    )


def _spec() -> SearchImagingSpec:
    return SearchImagingSpec(
        field_size_um=(400.0, 200.0),
        survey=SurveySpec((200.0, 100.0), 0.1, 0.02, 1, 0.06, 0.05, 0.1, 0.25),
        focused=FocusedSpec(2.0, 1, 1, 1, 0.06, 0.1, 0.1, 0.25, (64.0, 64.0)),
        stage=StageProfile(1000.0, 10000.0, 0.01),
        targets=(MockTarget("a", (100.0, 50.0)),),
    )


def test_snap_traces_include_survey_and_focused_points() -> None:
    spec = _spec()
    report = simulate_search_imaging(spec)

    traces = snap_traces(report, _envelope())

    assert [trace.name for trace in traces] == [
        "dual snap survey",
        "dual snap focused",
        "dual snap mode switch",
    ]
    survey = traces[0].points
    assert survey[0].label == "survey tile"
    assert survey[0].corner == "001"
    assert survey[1].label == "survey equivalent"
    focused = traces[1].points[0]
    assert focused.label == "focused revisit"


def test_broom_trace_ramps_from_dwell_to_cruise_and_back() -> None:
    trace = broom_trace(StageProfile(100.0, 1000.0), 100.0, 2.0, 0.02, _envelope())

    assert trace.points[0].label == "dwell at sample"
    assert trace.points[0].velocity_um_s == pytest.approx(50.0)
    assert trace.points[9].label == "cruise between samples"
    assert trace.points[9].velocity_um_s == pytest.approx(100.0)
    assert trace.points[-1].label == "dwell at next sample"
    assert trace.points[-1].velocity_um_s == pytest.approx(50.0)
    assert len(trace.points) == 19


def test_operating_payload_reports_ranges_thresholds_and_traces() -> None:
    spec = _spec()
    report = simulate_search_imaging(spec)
    broom = broom_trace(StageProfile(1000.0, 10000.0), 100.0, 2.0, 0.02)
    payload = build_operating_payload(
        snap_traces(report, _envelope()) + (broom,), _envelope()
    )

    assert payload["has_envelope"] is True
    assert payload["thresholds"]["velocity"] == [10.0, 100.0]
    assert payload["ranges"]["velocity"][0] == 0.0
    assert payload["ranges"]["velocity"][1] >= 100.0
    assert len(payload["traces"]) == 4
    assert all("points" in trace for trace in payload["traces"])


def test_simulation_operating_payload_without_envelope_has_no_corners() -> None:
    spec = _spec()
    report = simulate_search_imaging(spec)

    payload = simulation_operating_payload(report)

    assert payload["has_envelope"] is False
    assert payload["thresholds"] is None
    assert all(
        point["corner"] is None
        for trace in payload["traces"]
        for point in trace["points"]
    )
