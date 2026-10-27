"""Hardware-free tests for the stepped onset plan.

The plan is exercised through a real Bluesky RunEngine with a simulated
combined-XY stage and a fake triggerable camera; the band densities come from a
plain generator that returns the next declared pair.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from bluesky import RunEngine
from ophyd.status import Status

from planner import (
    OnsetPolicyConfig,
    SteppedOnsetFrameRecord,
    SteppedOnsetSpec,
    stepped_onset_plan,
)
from planner.intensity_events import IntensityRiseConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence


class _FakeCamera:
    """Triggerable camera with no device dependency."""

    name = "stepped_test_camera"

    def __init__(self) -> None:
        self.frames = 0

    def trigger(self) -> Status:
        """Count one requested frame and finish immediately."""
        self.frames += 1
        status = Status()
        status.set_finished()
        return status

    async def read(self) -> dict[str, dict[str, Any]]:
        """Return one scalar frame identifier."""
        return {"immediate": {"value": self.frames, "timestamp": 0.0}}

    async def describe(self) -> dict[str, dict[str, Any]]:
        """Describe the fake frame identifier."""
        return {"immediate": {"source": "fake", "dtype": "integer", "shape": []}}


class _FakeStage:
    """Combined-XY movable that records the last requested position."""

    name = "stepped_test_stage"
    parent = None

    def __init__(self) -> None:
        self.value = (0.0, 0.0)

    def set(self, value: tuple[float, float]) -> Status:
        """Accept one XY pair and finish immediately."""
        self.value = (float(value[0]), float(value[1]))
        status = Status()
        status.set_finished()
        return status

    async def read(self) -> dict[str, dict[str, Any]]:
        """Return the current XY pair."""
        return {"value": {"value": self.value, "timestamp": 0.0}}

    async def describe(self) -> dict[str, dict[str, Any]]:
        """Describe the XY pair."""
        return {"value": {"source": "fake", "dtype": "array", "shape": [2]}}


def _bands(
    values: Sequence[tuple[float | None, float | None]],
) -> Callable[[int], Generator[Any, None, tuple[float | None, float | None]]]:
    """Return a host-style band-density plan over the declared pairs."""

    def read(index: int) -> Generator[Any, None, tuple[float | None, float | None]]:
        if index < 0:  # pragma: no cover - keeps this a generator plan
            yield None
        return values[min(index, len(values) - 1)]

    return read


def _policy(**overrides: Any) -> OnsetPolicyConfig:
    values: dict[str, Any] = {
        "direction": "increasing_x",
        "sensor_height_px": 120,
        "band_fraction": 0.17,
        "coarse_step_um": 5.0,
        "dense_step_um": 1.0,
        "dense_budget_um": 10.0,
        "rise": IntensityRiseConfig(
            baseline_window_frames=5,
            baseline_min_samples=5,
            persistence_frames=2,
            release_persistence_frames=2,
            enter_relative_rise=0.2,
            exit_relative_rise=0.08,
        ),
    }
    values.update(overrides)
    return OnsetPolicyConfig(**values)


def _spec(**overrides: Any) -> SteppedOnsetSpec:
    values: dict[str, Any] = {
        "start_xy_um": (-60.0, 0.0),
        "end_xy_um": (60.0, 0.0),
        "stage_envelope_xy_um": ((-100.0, -5.0), (100.0, 5.0)),
        "policy": _policy(),
        "max_frames": 80,
    }
    values.update(overrides)
    return SteppedOnsetSpec(**values)


def _run(
    spec: SteppedOnsetSpec,
    bands: Sequence[tuple[float | None, float | None]],
    *,
    on_frame: Callable[[SteppedOnsetFrameRecord], None] | None = None,
) -> tuple[Any, list[tuple[str, dict[str, Any]]]]:
    """Run the plan and return its result plus the emitted documents."""
    documents: list[tuple[str, dict[str, Any]]] = []
    holder: dict[str, Any] = {}
    engine = RunEngine({})
    engine.subscribe(lambda name, doc: documents.append((name, doc)))

    def run_plan() -> Generator[Any, None, None]:
        holder["result"] = yield from stepped_onset_plan(
            spec,
            stage_xy=_FakeStage(),
            camera=_FakeCamera(),
            band_densities_of_frame=_bands(bands),
            on_frame=on_frame,
        )

    engine(run_plan())
    return holder["result"], documents


#: baseline, then a leading rise, then a later trailing rise, then quiet.
_FEATURE_BANDS = (
    [(100.0, 100.0)] * 8
    + [(300.0, 100.0)] * 4
    + [(300.0, 300.0)] * 3
    + [(100.0, 100.0)] * 20
)


def test_the_plan_advances_coarse_then_dense_then_coarse() -> None:
    result, _documents = _run(_spec(), _FEATURE_BANDS)

    reasons = [record.reason for record in result.records]
    assert reasons[0] == "advance_coarse"
    assert "advance_dense" in reasons
    assert "released_by_trailing_band" in reasons
    assert result.stopped_reason == "completed"
    assert result.truncated is False
    assert result.records[0].x_um == pytest.approx(-60.0)
    assert all(record.y_um == 0.0 for record in result.records)

    coarse = {
        record.requested_step_um
        for record in result.records
        if record.reason == "advance_coarse"
    }
    dense = {
        record.requested_step_um
        for record in result.records
        if record.reason == "advance_dense"
    }
    assert coarse == {5.0}
    assert dense == {1.0}
    assert result.records[-1].dense_used_um <= 10.0


def test_a_decreasing_traverse_walks_negative_x() -> None:
    spec = _spec(
        start_xy_um=(60.0, 0.0),
        end_xy_um=(-60.0, 0.0),
        policy=_policy(direction="decreasing_x"),
    )
    result, _documents = _run(spec, _FEATURE_BANDS)
    positions = [record.x_um for record in result.records]
    assert positions[0] == pytest.approx(60.0)
    assert all(right < left for left, right in zip(positions, positions[1:]))
    assert result.stopped_reason == "completed"


def test_max_frames_truncates_and_says_so() -> None:
    result, _documents = _run(_spec(max_frames=3), _FEATURE_BANDS)
    assert result.frame_count == 3
    assert result.truncated is True
    assert result.stopped_reason == "max_frames"


def test_missing_bands_hold_then_block_without_moving() -> None:
    result, _documents = _run(_spec(), [(None, None)] * 6)
    assert result.stopped_reason == "blocked_missing_observations"
    assert result.truncated is False
    assert all(record.requested_step_um == 0.0 for record in result.records)
    assert {record.x_um for record in result.records} == {-60.0}
    assert result.records[-1].reason == "blocked_missing_observations"


def test_a_zero_budget_exits_after_the_onset_frame() -> None:
    result, _documents = _run(
        _spec(policy=_policy(dense_budget_um=0.0)), _FEATURE_BANDS
    )
    exhausted = [record for record in result.records if record.budget_exhausted]
    assert len(exhausted) == 1
    assert all(record.requested_step_um != 1.0 for record in result.records)


def test_every_frame_is_reported_to_the_host() -> None:
    collected: list[SteppedOnsetFrameRecord] = []
    result, documents = _run(
        _spec(max_frames=5), _FEATURE_BANDS, on_frame=collected.append
    )
    assert len(collected) == 5
    assert [record.frame_index for record in collected] == list(range(5))
    assert [record.frame_index for record in result.records] == list(range(5))
    start = next(doc for name, doc in documents if name == "start")
    assert start["planner_stepped_onset_policy_id"] == result.spec.policy.policy_id


def test_the_record_serializes() -> None:
    result, _documents = _run(_spec(max_frames=3), _FEATURE_BANDS)
    payload = result.records[0].to_dict()
    assert json.loads(json.dumps(payload))["reason"] == "advance_coarse"


def test_spec_validation() -> None:
    with pytest.raises(ValueError, match="hold y constant"):
        _spec(end_xy_um=(60.0, 1.0))
    with pytest.raises(ValueError, match="must differ along x"):
        _spec(end_xy_um=(-60.0, 0.0))
    with pytest.raises(ValueError, match="does not match"):
        _spec(policy=_policy(direction="decreasing_x"))
    with pytest.raises(ValueError, match="leaves the declared stage envelope"):
        _spec(start_xy_um=(-200.0, 0.0))
    with pytest.raises(ValueError, match="max_frames"):
        _spec(max_frames=0)
    with pytest.raises(TypeError, match="OnsetPolicyConfig"):
        _spec(policy=object())
    with pytest.raises(ValueError, match="nonzero axis-aligned box"):
        _spec(stage_envelope_xy_um=((-100.0, -5.0), (-100.0, 5.0)))


def test_plan_argument_validation() -> None:
    engine = RunEngine({})
    bands = _bands(_FEATURE_BANDS)

    with pytest.raises(ValueError, match="stage_xy is required"):
        engine(
            stepped_onset_plan(
                _spec(),
                stage_xy=None,
                camera=_FakeCamera(),
                band_densities_of_frame=bands,
            )
        )
    with pytest.raises(ValueError, match="camera is required"):
        engine(
            stepped_onset_plan(
                _spec(),
                stage_xy=_FakeStage(),
                camera=None,
                band_densities_of_frame=bands,
            )
        )
    with pytest.raises(TypeError, match="callable plan"):
        engine(
            stepped_onset_plan(
                _spec(),
                stage_xy=_FakeStage(),
                camera=_FakeCamera(),
                band_densities_of_frame=None,  # type: ignore[arg-type]
            )
        )
    with pytest.raises(ValueError, match="cannot override"):
        engine(
            stepped_onset_plan(
                _spec(),
                stage_xy=_FakeStage(),
                camera=_FakeCamera(),
                band_densities_of_frame=bands,
                md={"planner_scan_mode": "other"},
            )
        )
