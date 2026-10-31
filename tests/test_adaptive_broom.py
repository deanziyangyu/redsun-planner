"""Hardware-free tests for the adaptive broom raster plan.

The plan is exercised through a real Bluesky RunEngine with simulated stage
axes and a fake triggerable camera. The intensity source is a plain generator
that returns the next declared value, which is exactly the host contract the
acquisition application binds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from bluesky import RunEngine
from ophyd.sim import SynAxis
from ophyd.status import Status

from planner import (
    AdaptiveBroomFrameRecord,
    AdaptiveBroomSpec,
    adaptive_broom_plan,
    get_plan_block,
)
from planner.speed_policy import SpeedBlendConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Generator


class _FakeCamera:
    """Triggerable, readable camera with no device dependency."""

    name = "adaptive_test_camera"

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


def _intensity_plan(
    values: list[float],
) -> Callable[[int], Generator[Any, None, float]]:
    """Return a host-style intensity plan over the declared values."""

    def read(index: int) -> Generator[Any, None, float]:
        if index == 0:  # pragma: no cover - keeps this a generator plan
            yield None
        return values[min(index - 1, len(values) - 1)]

    return read


def _spec(
    *,
    field_um: tuple[float, float] = (4.0, 2.0),
    fast: float = 20.0,
    slow: float = 5.0,
    period: float = 0.1,
    ceiling: float = 2000.0,
    max_frames: int = 200,
) -> AdaptiveBroomSpec:
    """Return a small spec whose fast step is two micrometres."""
    return AdaptiveBroomSpec(
        origin_xy_um=(0.0, 0.0),
        field_size_um=field_um,
        stage_envelope_xy_um=((-5.0, -5.0), (50.0, 50.0)),
        row_pitch_um=2.0,
        frame_period_s=period,
        slow_speed_um_s=slow,
        fast_speed_um_s=fast,
        stage_max_speed_um_s=ceiling,
        max_frames=max_frames,
        config=SpeedBlendConfig(smoothing_frames=1, max_acceleration_um_s2=10_000),
    )


def _run(
    values: list[float], spec: AdaptiveBroomSpec | None = None
) -> tuple[Any, list[tuple[str, dict[str, Any]]]]:
    """Run the plan on simulated axes and return its result and documents."""
    x = SynAxis(name="adaptive_x", value=0.0, egu="um")
    y = SynAxis(name="adaptive_y", value=0.0, egu="um")
    camera = _FakeCamera()
    documents: list[tuple[str, dict[str, Any]]] = []
    holder: dict[str, Any] = {}
    engine = RunEngine({})
    engine.subscribe(lambda name, doc: documents.append((name, doc)))

    def run_plan() -> Generator[Any, None, None]:
        holder["result"] = yield from adaptive_broom_plan(
            _spec() if spec is None else spec,
            stage_x=x,
            stage_y=y,
            camera=camera,
            intensity_of_frame=_intensity_plan(values),
        )

    engine(run_plan())
    return holder["result"], documents


def test_a_flat_field_runs_at_the_fast_speed_everywhere() -> None:
    result, _documents = _run([0.0])

    speeds = [record.commanded_speed_um_s for record in result.records]
    assert speeds
    assert set(speeds) == {20.0}
    assert all(record.step_um == pytest.approx(2.0) for record in result.records)


def test_the_speed_sequence_changes_when_the_intensity_signal_changes() -> None:
    flat, _ = _run([0.0])
    risen, _ = _run([0.0, 1.0, 1.0, 1.0])

    flat_speeds = [record.commanded_speed_um_s for record in flat.records]
    risen_speeds = [record.commanded_speed_um_s for record in risen.records]
    assert flat_speeds[0] == risen_speeds[0] == 20.0
    assert min(risen_speeds) == 5.0
    assert risen_speeds[1:3] == [5.0, 5.0]
    assert risen_speeds != flat_speeds[: len(risen_speeds)]
    # A slower request takes smaller steps, so the same rows need more frames.
    assert risen.frame_count > flat.frame_count


def test_commanded_speed_never_leaves_its_bounds_or_the_stage_ceiling() -> None:
    result, _documents = _run([0.0, 0.5, 1.0, 0.8, 0.0])

    for record in result.records:
        assert 5.0 <= record.commanded_speed_um_s <= 20.0
        assert record.commanded_speed_um_s <= result.spec.stage_max_speed_um_s
        assert record.step_um == pytest.approx(
            record.commanded_speed_um_s * result.spec.frame_period_s
        )


def test_the_run_records_exactly_one_speed_per_frame() -> None:
    result, documents = _run([0.0, 0.2, 0.4])

    events = [doc for name, doc in documents if name == "event"]
    recorded = [
        doc["data"]["adaptive_speed_um_s"]
        for doc in events
        if "adaptive_speed_um_s" in doc["data"]
    ]
    assert len(recorded) == result.frame_count
    assert len(result.records) == result.frame_count
    assert [record.frame_index for record in result.records] == list(
        range(1, result.frame_count + 1)
    )
    assert all(
        isinstance(record, AdaptiveBroomFrameRecord) for record in result.records
    )
    assert recorded == [record.commanded_speed_um_s for record in result.records]


def test_frame_count_is_capped_and_reported_as_truncated() -> None:
    result, _documents = _run([0.0], _spec(field_um=(20.0, 2.0), max_frames=3))

    assert result.frame_count == 3
    assert result.truncated is True


def test_spec_refuses_a_field_that_leaves_its_declared_envelope() -> None:
    with pytest.raises(ValueError, match="field leaves its declared stage envelope"):
        AdaptiveBroomSpec(
            origin_xy_um=(0.0, 0.0),
            field_size_um=(300.0, 120.0),
            stage_envelope_xy_um=((-100.0, -100.0), (100.0, 100.0)),
            row_pitch_um=100.0,
            frame_period_s=0.1,
            slow_speed_um_s=30.0,
            fast_speed_um_s=150.0,
            stage_max_speed_um_s=2000.0,
        )


def test_spec_refuses_a_fast_speed_above_the_stage_ceiling() -> None:
    with pytest.raises(ValueError, match="declared maximum speed"):
        _spec(fast=2500.0, ceiling=2000.0)


def test_the_plan_block_is_registered() -> None:
    block = get_plan_block("planner.adaptive_broom")

    assert block.plan_stub is adaptive_broom_plan
    assert block.resource_requirements


def _braking_spec(
    *, dwell: int = 2, threshold: float | None = None
) -> AdaptiveBroomSpec:
    """Return a spec whose brake holds each slow frame for *dwell* frames."""
    return AdaptiveBroomSpec(
        origin_xy_um=(0.0, 0.0),
        field_size_um=(8.0, 2.0),
        stage_envelope_xy_um=((-5.0, -5.0), (50.0, 50.0)),
        row_pitch_um=2.0,
        frame_period_s=0.1,
        slow_speed_um_s=5.0,
        fast_speed_um_s=20.0,
        stage_max_speed_um_s=2000.0,
        brake_dwell_frames=dwell,
        brake_interest_threshold=threshold,
        config=SpeedBlendConfig(smoothing_frames=1, max_acceleration_um_s2=10_000),
    )


def test_the_brake_holds_every_slow_frame_without_advancing() -> None:
    result, _documents = _run([0.0, 1.0, 1.0], _braking_spec(dwell=2))

    held = [record for record in result.records if record.dwell_index]
    assert result.brake_count >= 1
    assert held
    for record in held:
        assert record.commanded_speed_um_s == 0.0
        assert record.step_um == 0.0
        assert record.dwell_index >= 1
    # Every held frame shares its point with the frame that took the brake, and holds only
    # follow a frame the policy had already slowed.
    by_index = {record.frame_index: record for record in result.records}
    for record in held:
        brake_frame = by_index[record.frame_index - record.dwell_index]
        assert (record.x_um, record.y_um) == (brake_frame.x_um, brake_frame.y_um)
        assert brake_frame.commanded_speed_um_s == result.spec.slow_speed_um_s


def test_a_held_run_records_more_frames_and_holds_none_when_disabled() -> None:
    flat_with_brake, _ = _run([1.0], _braking_spec(dwell=2))
    flat_without, _ = _run([1.0], _spec())

    assert flat_with_brake.brake_count > 0
    assert flat_without.brake_count == 0
    assert all(record.dwell_index == 0 for record in flat_without.records)
    assert flat_with_brake.frame_count > flat_without.frame_count


def test_the_hold_is_capped_by_the_frame_budget() -> None:
    result, _documents = _run([1.0], _braking_spec(dwell=5))

    capped = AdaptiveBroomSpec(
        origin_xy_um=(0.0, 0.0),
        field_size_um=(8.0, 2.0),
        stage_envelope_xy_um=((-5.0, -5.0), (50.0, 50.0)),
        row_pitch_um=2.0,
        frame_period_s=0.1,
        slow_speed_um_s=5.0,
        fast_speed_um_s=20.0,
        stage_max_speed_um_s=2000.0,
        max_frames=4,
        brake_dwell_frames=5,
    )
    truncated, _ = _run([1.0], capped)
    assert truncated.truncated is True
    assert truncated.frame_count == 4
    assert result.frame_count > 4


def test_a_brake_threshold_below_the_policy_fast_threshold_is_refused() -> None:
    with pytest.raises(ValueError, match="interest_fast_threshold"):
        _braking_spec(threshold=0.01)


def test_the_brake_defaults_to_the_policy_slow_threshold() -> None:
    spec = _braking_spec()

    assert spec.brake_interest_threshold == pytest.approx(
        spec.config.interest_slow_threshold
    )


def test_the_frame_table_reports_the_hold_state() -> None:
    result, documents = _run([1.0], _braking_spec(dwell=1))

    events = [doc for name, doc in documents if name == "event"]
    dwells = [
        doc["data"]["adaptive_dwell_index"]
        for doc in events
        if "adaptive_dwell_index" in doc["data"]
    ]
    assert len(dwells) == result.frame_count
    assert dwells == [record.dwell_index for record in result.records]
    brake = documents[0][1]["planner_adaptive_brake"]
    assert brake["dwell_frames"] == 1
    assert brake["kind"] == "step-cancel"


class _CountingAxis(SynAxis):
    """Axis that records how many times it was commanded."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.sets = 0

    def set(self, value: float) -> Any:
        """Count the command and forward it."""
        self.sets += 1
        return super().set(value)


def test_a_single_row_raster_commands_the_idle_axis_once() -> None:
    x = _CountingAxis(name="counted_x", value=0.0, egu="um")
    y = _CountingAxis(name="counted_y", value=0.0, egu="um")
    camera = _FakeCamera()
    spec = AdaptiveBroomSpec(
        origin_xy_um=(0.0, 0.0),
        field_size_um=(8.0, 1.0),
        stage_envelope_xy_um=((-5.0, -5.0), (50.0, 50.0)),
        row_pitch_um=2.0,
        frame_period_s=0.1,
        slow_speed_um_s=5.0,
        fast_speed_um_s=20.0,
        stage_max_speed_um_s=2000.0,
        max_frames=100,
        config=SpeedBlendConfig(smoothing_frames=1, max_acceleration_um_s2=10_000),
    )
    holder: dict[str, Any] = {}

    def run_plan() -> Generator[Any, None, None]:
        holder["result"] = yield from adaptive_broom_plan(
            spec,
            stage_x=x,
            stage_y=y,
            camera=camera,
            intensity_of_frame=_intensity_plan([0.0]),
        )

    RunEngine({})(run_plan())
    result = holder["result"]

    # y never changes inside one row, so it is commanded once and x once per frame: a
    # restated target on a host that skips it would leave the client waiting forever.
    assert y.sets == 1
    assert x.sets == result.frame_count
