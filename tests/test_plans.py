from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np
import pytest
from bluesky import RunEngine

from planner import ReplayFrame, build_grid_recipe, plans, replay_dataset_plan
from planner.bounded_scanning import (
    ConfirmedBoundary,
    CornerObservation,
    plan_broom_scan,
)
from planner.contracts import AcquisitionSettings
from planner.plans import hand_eye_calibration_plan, push_broom_plan, snap_scan_plan
from planner.replay import _ReplayFrameReadable

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


class _ReplaySource:
    """Small lazy source used to run a hardware-free document replay."""

    source_id = "synthetic-dataset:frames-280-281"
    source_metadata: ClassVar[dict[str, Any]] = {
        "source_reference": "synthetic-test-data",
        "source_time_origin": "OME DeltaT seconds",
        "source_frame_selection": [280, 281],
    }

    def iter_frames(self) -> Iterator[ReplayFrame]:
        yield ReplayFrame(
            source_frame_id="280",
            source_time_s=3.5,
            images={"image": np.full((2, 3), 12, dtype=np.uint16)},
            source_context={"source_stage_x_um": 100.0, "source_roi": "0,0,3,2"},
        )
        yield ReplayFrame(
            source_frame_id="281",
            source_time_s=3.52,
            images={"image": np.full((2, 3), 24, dtype=np.uint16)},
            source_context={"source_stage_x_um": 101.0, "source_roi": "0,0,3,2"},
        )


def _install_bluesky_stubs(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, tuple[Any, ...], dict[str, Any]]]:
    calls = []

    def message(command: str, *args: Any, **kwargs: Any) -> Iterator[Any]:
        calls.append((command, args, kwargs))
        yield (command, args, kwargs)

    def wrapper(name: str, **wrapper_kwargs: Any) -> Callable[[Any], Any]:
        def decorate(plan: Any) -> Any:
            def wrapped(*args: Any, **kwargs: Any) -> Iterator[Any]:
                yield from message(f"{name}_start", **wrapper_kwargs)
                yield from plan(*args, **kwargs)
                yield from message(f"{name}_stop", **wrapper_kwargs)

            return wrapped

        return decorate

    bps = types.SimpleNamespace(
        mv=lambda *args, **kwargs: message("mv", *args, **kwargs),
        trigger_and_read=lambda *args, **kwargs: message(
            "trigger_and_read", *args, **kwargs
        ),
        prepare=lambda *args, **kwargs: message("prepare", *args, **kwargs),
        kickoff=lambda *args, **kwargs: message("kickoff", *args, **kwargs),
        complete=lambda *args, **kwargs: message("complete", *args, **kwargs),
        wait=lambda *args, **kwargs: message("wait", *args, **kwargs),
        collect=lambda *args, **kwargs: message("collect", *args, **kwargs),
        stop=lambda *args, **kwargs: message("stop", *args, **kwargs),
        declare_stream=lambda *args, **kwargs: message(
            "declare_stream", *args, **kwargs
        ),
        collect_while_completing=lambda *args, **kwargs: message(
            "collect_while_completing", *args, **kwargs
        ),
    )
    bpp = types.SimpleNamespace(
        run_decorator=lambda *, md: wrapper("run", md=md),
        stage_decorator=lambda devices: wrapper("stage", devices=devices),
        finalize_wrapper=lambda plan, final_plan: _finalize_stub(plan, final_plan),
    )
    monkeypatch.setattr(plans, "bps", bps)
    monkeypatch.setattr(plans, "bpp", bpp)
    return calls


def _finalize_stub(plan: Any, final_plan: Any) -> Iterator[Any]:
    try:
        yield from plan
    finally:
        yield from final_plan()


def test_snap_plan_moves_then_emits_one_capture_per_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_bluesky_stubs(monkeypatch)
    recipe = build_grid_recipe(
        (10, 20), specimen_id="s", rows=1, columns=2, step_xy_um=(5, 7)
    )
    messages = list(
        snap_scan_plan(
            recipe,
            stage_x="x",
            stage_y="y",
            camera="camera",
            readables=("x_readback", "y_readback"),
        )
    )
    names = [item[0] for item in messages]
    assert names == [
        "stage_start",
        "run_start",
        "mv",
        "trigger_and_read",
        "mv",
        "trigger_and_read",
        "run_stop",
        "stage_stop",
    ]
    start_md = next(call[2]["md"] for call in calls if call[0] == "run_start")
    assert start_md["planner_position_ids"] == [
        position.position_id for position in recipe.positions
    ]
    assert start_md["planner_event_position_ids"] == start_md["planner_position_ids"]
    assert start_md["planner_calibration_id"] == recipe.calibration_id
    assert calls[3][1][0] == ["camera", "x_readback", "y_readback"]


def test_dataset_replay_emits_source_context_through_a_bluesky_run_engine() -> None:
    """Replay frames as ordinary documents without binding physical devices."""
    source = _ReplaySource()
    docs = []
    engine = RunEngine()
    engine.subscribe(lambda name, doc: docs.append((name, doc)))

    engine(replay_dataset_plan(source))

    start = next(doc for name, doc in docs if name == "start")
    events = [doc for name, doc in docs if name == "event"]
    assert start["planner_workflow"] == "dataset-frame-replay-v1"
    assert start["planner_replay_source_id"] == source.source_id
    assert start["planner_replay_source_metadata"] == source.source_metadata
    assert start["planner_simulation_only"] is True
    assert start["planner_hardware_commands_issued"] is False
    assert len(events) == 2
    assert [event["data"]["source_frame_id"] for event in events] == ["280", "281"]
    assert [event["data"]["source_time_s"] for event in events] == [3.5, 3.52]
    assert [event["data"]["source_stage_x_um"] for event in events] == [100.0, 101.0]
    assert [int(event["data"]["image"].sum()) for event in events] == [72, 144]


def test_dataset_replay_descriptor_declares_units_for_every_field() -> None:
    """Label the time field in seconds and leave unknown units explicitly empty."""
    readable = _ReplayFrameReadable(next(_ReplaySource().iter_frames()))
    described = readable.describe()
    assert described["source_time_s"]["units"] == "s"
    numeric = {
        key: field for key, field in described.items() if key != "source_frame_id"
    }
    assert all("units" in field for field in numeric.values())
    assert "units" not in described["source_frame_id"]
    assert described["image"]["units"] == ""


def test_dataset_replay_rejects_run_metadata_override_before_run_opens() -> None:
    """Keep caller metadata from overriding planner-owned run declarations."""
    with pytest.raises(ValueError, match="cannot override"):
        plan = replay_dataset_plan(
            _ReplaySource(), md={"planner_hardware_commands_issued": True}
        )
        next(iter(plan))


def test_snap_plan_accepts_one_atomic_xy_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_bluesky_stubs(monkeypatch)
    recipe = build_grid_recipe(
        (10, 20), specimen_id="s", rows=1, columns=2, step_xy_um=(5, 7)
    )
    messages = list(
        snap_scan_plan(
            recipe,
            stage_xy="xy",
            camera="camera",
            readables=("xy_readback",),
        )
    )
    moves = [call for call in calls if call[0] == "mv"]
    assert len(moves) == len(recipe.positions) == 2
    assert moves[0][1] == ("xy", (10.0, 20.0))
    assert moves[1][1] == ("xy", (15.0, 20.0))
    assert [item[0] for item in messages].count("trigger_and_read") == 2


def test_snap_plan_prepares_one_camera_event_before_each_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_bluesky_stubs(monkeypatch)
    recipe = build_grid_recipe(
        (10, 20),
        specimen_id="s",
        rows=1,
        columns=2,
        step_xy_um=(5, 7),
        settings=AcquisitionSettings(frames=2),
    )

    def prepare_one_event(camera: Any) -> Iterator[Any]:
        yield from plans.bps.prepare(camera, number_of_events=1, wait=True)  # type: ignore[attr-defined]

    messages = list(
        snap_scan_plan(
            recipe,
            stage_x="x",
            stage_y="y",
            camera="camera",
            prepare_camera=prepare_one_event,
        )
    )

    names = [item[0] for item in messages]
    assert names.count("prepare") == 4
    assert names.count("trigger_and_read") == 4
    for index, name in enumerate(names):
        if name == "trigger_and_read":
            assert names[index - 1] == "prepare"
    assert all(
        call[2]["number_of_events"] == 1 for call in calls if call[0] == "prepare"
    )


def test_snap_plan_uses_host_capture_callback_for_streamable_camera(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_bluesky_stubs(monkeypatch)
    recipe = build_grid_recipe(
        (10, 20), specimen_id="s", rows=1, columns=2, step_xy_um=(5, 7)
    )

    def capture_one(camera: Any, first_capture: bool) -> Iterator[Any]:
        if first_capture:
            yield from plans.bps.declare_stream(camera, name="primary", collect=True)  # type: ignore[attr-defined]
        yield from plans.bps.kickoff(camera, wait=True)  # type: ignore[attr-defined]
        yield from plans.bps.complete(camera, wait=True)  # type: ignore[attr-defined]
        yield from plans.bps.collect(camera, stream=True)  # type: ignore[attr-defined]

    def prepare_capture(camera: Any, number_of_frames: int) -> Iterator[Any]:
        yield from plans.bps.prepare(  # type: ignore[attr-defined]
            camera, number_of_events=number_of_frames, wait=True
        )

    list(
        snap_scan_plan(
            recipe,
            stage_x="x",
            stage_y="y",
            camera="camera",
            prepare_capture=prepare_capture,
            capture_one=capture_one,
        )
    )

    names = [call[0] for call in calls]
    assert names.count("declare_stream") == 1
    assert names.count("kickoff") == len(recipe.positions)
    assert names.count("complete") == len(recipe.positions)
    assert names.count("collect") == len(recipe.positions)
    assert "trigger_and_read" not in names
    prepares = [call for call in calls if call[0] == "prepare"]
    assert len(prepares) == 1
    assert prepares[0][2]["number_of_events"] == len(recipe.positions)


def test_hand_eye_plan_emits_reference_probe_return_sequence_and_readbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_bluesky_stubs(monkeypatch)
    list(
        hand_eye_calibration_plan(
            anchor_xy_um=(100.0, 200.0),
            stage_x="x",
            stage_y="y",
            camera="camera",
            x_readback="x_rb",
            y_readback="y_rb",
            coordinate_frame="stage-xy",
            x_probe_um=10.0,
            y_probe_um=20.0,
            mode_context={"profile": "demo"},
            registration_roi_yxhw_px=(0, 128, 768, 768),
            registration_preprocessing="percentile_clahe",
        )
    )
    moves = [call[1] for call in calls if call[0] == "mv"]
    reads = [call for call in calls if call[0] == "trigger_and_read"]
    assert moves == [
        ("x", 100.0, "y", 200.0),
        ("x", 100.0, "y", 200.0),
        ("x", 110.0, "y", 200.0),
        ("x", 100.0, "y", 200.0),
        ("x", 90.0, "y", 200.0),
        ("x", 100.0, "y", 200.0),
        ("x", 100.0, "y", 220.0),
        ("x", 100.0, "y", 200.0),
        ("x", 100.0, "y", 180.0),
        ("x", 100.0, "y", 200.0),
    ]
    assert len(reads) == 10
    assert reads[0][1][0] == ["camera", "x_rb", "y_rb"]
    start_md = next(call[2]["md"] for call in calls if call[0] == "run_start")
    assert start_md["planner_workflow"] == "stage-camera-hand-eye-calibration-v1"
    assert start_md["planner_hand_eye_registration"] == {
        "roi_yxhw_px": [0, 128, 768, 768],
        "preprocessing": "percentile_clahe",
    }
    assert (
        start_md["planner_calibration_mode_context"]["registration_roi_yxhw_px"]
        == "0,128,768,768"
    )
    assert start_md["planner_calibration_capture_labels"] == [
        "reference",
        "reference_repeat",
        "probe_x_positive",
        "return_x_positive",
        "probe_x_negative",
        "return_x_negative",
        "probe_y_positive",
        "return_y_positive",
        "probe_y_negative",
        "return_y_negative",
    ]


def test_hand_eye_plan_runs_with_ophyd_simulated_devices() -> None:
    bluesky = pytest.importorskip("bluesky")
    ophyd_sim = pytest.importorskip("ophyd.sim")
    np = pytest.importorskip("numpy")
    x = ophyd_sim.SynAxis(name="cal_x", value=100, egu="um")
    y = ophyd_sim.SynAxis(name="cal_y", value=200, egu="um")
    camera = ophyd_sim.SynSignal(
        func=lambda: np.ones((32, 32), dtype=np.uint16), name="cal_camera"
    )
    documents = []
    engine = bluesky.RunEngine({})
    engine.subscribe(lambda name, doc: documents.append((name, doc)))

    engine(
        hand_eye_calibration_plan(
            anchor_xy_um=(100, 200),
            stage_x=x,
            stage_y=y,
            camera=camera,
            x_readback=x.readback,
            y_readback=y.readback,
            coordinate_frame="simulated-stage-xy",
            mode_context={"profile": "ophyd-sim"},
        )
    )

    start = next(doc for name, doc in documents if name == "start")
    events = [doc for name, doc in documents if name == "event"]
    stop = next(doc for name, doc in documents if name == "stop")
    assert start["planner_calibration_capture_labels"][0] == "reference"
    assert len(events) == 10
    assert [event["data"]["cal_x"] for event in events] == [
        100,
        100,
        110,
        100,
        90,
        100,
        100,
        100,
        100,
        100,
    ]
    assert [event["data"]["cal_y"] for event in events] == [
        200,
        200,
        200,
        200,
        200,
        200,
        210,
        200,
        190,
        200,
    ]
    assert stop["exit_status"] == "success"


def test_push_broom_plan_prepares_and_collects_each_strip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_bluesky_stubs(monkeypatch)
    first = CornerObservation((0, 0), True, "a", (0, 0))
    opposite = CornerObservation((30, 20), True, "b", (30, 20))
    boundary = ConfirmedBoundary.from_opposite_corners(
        first, opposite, coordinate_frame="stage-xy", boundary_id="test-boundary"
    )
    recipe = plan_broom_scan(boundary, (10, 10), 0.5)
    detectors = ("position_encoder",)
    messages = list(
        push_broom_plan(recipe, flyer="combined_flyer", detectors=detectors)
    )
    names = [item[0] for item in messages]
    assert names.count("prepare") == len(recipe.strips)
    assert names.count("kickoff") == len(recipe.strips)
    assert names.count("collect_while_completing") == len(recipe.strips)
    assert names.count("declare_stream") == 1
    declaration = next(call for call in calls if call[0] == "declare_stream")
    assert declaration[1] == ("combined_flyer", "position_encoder")
    assert declaration[2] == {"name": "primary", "collect": True}
    collected = [call[1][1] for call in calls if call[0] == "collect_while_completing"]
    assert all(items[0] == "combined_flyer" for items in collected)
    assert all(items[1] == "position_encoder" for items in collected)
    assert all(not call[2].get("wait", True) for call in calls if call[0] == "kickoff")
    start_md = next(call[2]["md"] for call in calls if call[0] == "run_start")
    assert start_md["planner_scan_mode"] == "push-broom"
    assert start_md["planner_expected_strip_count"] == len(recipe.strips)


def test_push_broom_plan_calls_host_progress_callback_for_each_strip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_bluesky_stubs(monkeypatch)
    boundary = ConfirmedBoundary.from_opposite_corners(
        CornerObservation((0, 0), True, "a", (0, 0)),
        CornerObservation((30, 20), True, "b", (30, 20)),
        coordinate_frame="stage-xy",
        boundary_id="progress-test",
    )
    recipe = plan_broom_scan(boundary, (10, 10), 0.5)
    progress_calls: list[tuple[Any, int, int]] = []

    def report_progress(
        status: Any, strip_index: int, strip_count: int
    ) -> Iterator[Any]:
        progress_calls.append((status, strip_index, strip_count))
        yield ("test_progress", status, strip_index, strip_count)

    messages = list(
        push_broom_plan(
            recipe,
            flyer="geometry-flyer",
            stream_writers=("camera-stream",),
            progress_callback=report_progress,
        )
    )

    assert len(progress_calls) == len(recipe.strips)
    assert [call[1] for call in progress_calls] == list(
        range(1, len(recipe.strips) + 1)
    )
    assert all(call[2] == len(recipe.strips) for call in progress_calls)
    assert sum(call[0] == "complete" for call in calls) == len(recipe.strips)
    assert sum(call[0] == "wait" for call in calls) == len(recipe.strips)
    assert sum(message[0] == "test_progress" for message in messages) == len(
        recipe.strips
    )


def test_plans_reject_missing_positions_and_reserved_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_bluesky_stubs(monkeypatch)
    recipe = build_grid_recipe(
        (0, 0), specimen_id="s", rows=1, columns=1, step_xy_um=(1, 1)
    )
    with pytest.raises(ValueError, match="cannot override"):
        list(
            snap_scan_plan(
                recipe,
                stage_x="x",
                stage_y="y",
                camera="camera",
                md={"planner_scan_mode": "override"},
            )
        )
    with pytest.raises(TypeError, match="BroomScanPlan"):
        list(push_broom_plan(cast("Any", recipe), flyer="not-a-flyer"))


def test_push_broom_collects_flyer_events_with_real_run_engine() -> None:
    bluesky = pytest.importorskip("bluesky")
    pytest.importorskip("ophyd")
    from ophyd.sim import SynAxis  # noqa: PLC0415
    from ophyd.status import DeviceStatus  # noqa: PLC0415

    class FakeFlyer(SynAxis):  # type: ignore[misc]
        def __init__(self, name: str) -> None:
            super().__init__(name=name)
            self._frame_index = 0

        def prepare(self, *, strip: Any) -> Any:
            self._strip = strip
            status = DeviceStatus(self, call_stop_on_failure=False)
            status.set_finished()
            return status

        def kickoff(self) -> Any:
            status = DeviceStatus(self, call_stop_on_failure=False)
            status.set_finished()
            return status

        def complete(self) -> Any:
            status = DeviceStatus(self, call_stop_on_failure=False)
            status.set_finished()
            return status

        def describe_collect(self) -> Any:
            return {
                "frame": {
                    "source": "SIM:frame",
                    "dtype": "integer",
                    "shape": [],
                }
            }

        def collect(self) -> Any:
            self._frame_index += 1
            yield {
                "data": {"frame": self._frame_index},
                "timestamps": {"frame": 1.0},
                "time": 1.0,
            }

    first = CornerObservation((0, 0), True, "a", (0, 0))
    opposite = CornerObservation((30, 20), True, "b", (30, 20))
    boundary = ConfirmedBoundary.from_opposite_corners(
        first, opposite, coordinate_frame="stage-xy", boundary_id="engine-boundary"
    )
    recipe = plan_broom_scan(boundary, (10, 10), 0.5)
    flyer = FakeFlyer(name="flyer")
    documents = []
    engine = bluesky.RunEngine({})
    engine.subscribe(lambda name, doc: documents.append((name, doc)))

    engine(push_broom_plan(recipe, flyer=flyer))

    start = next(doc for name, doc in documents if name == "start")
    events = [doc for name, doc in documents if name == "event"]
    event_pages = [doc for name, doc in documents if name == "event_page"]
    stop = next(doc for name, doc in documents if name == "stop")
    assert start["planner_expected_strip_count"] == len(recipe.strips)
    collected_frames = [event["data"]["frame"] for event in events]
    collected_frames.extend(
        frame for page in event_pages for frame in page["data"]["frame"]
    )
    assert len(collected_frames) == len(recipe.strips)
    assert collected_frames == list(range(1, len(recipe.strips) + 1))
    assert stop["exit_status"] == "success"
