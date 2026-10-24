"""Bluesky plans over validated, backend-independent scan recipes.

Device bindings and physical limits remain the execution host's job. A
push-broom adapter must implement Bluesky's preparable/flyable/collectable
contracts and emit each frame with stage readbacks and acquisition timing.
"""

from __future__ import annotations

# Host plan widgets resolve these plan annotations at runtime.
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import chain
from math import isfinite
from typing import Any
from uuid import uuid4

from bluesky import plan_stubs as bps
from bluesky import preprocessors as bpp
from bluesky.utils import MsgGenerator

from .bounded_scanning import BroomScanPlan, SnapScanPlan
from .hand_eye import hand_eye_capture_labels
from .observed_coverage import ObservedSnapGridFit
from .protocols import LiveFlyer, XYMotor
from .replay import ReplayFrame, ReplaySource, _ReplayFrameReadable
from .scanning import ScanPosition, StageScanRecipe


def _metadata(
    md: Mapping[str, Any] | None, required: Mapping[str, Any]
) -> dict[str, Any]:
    result = dict(md or {})
    reserved = set(required).intersection(result)
    if reserved:
        names = ", ".join(sorted(reserved))
        raise ValueError(f"metadata cannot override planner fields: {names}")
    result.update(required)
    return result


def prepare_live_stream(
    detectors: Sequence[LiveFlyer],
    trigger_info: Any,
    stream_name: str,
    *,
    collect: bool = True,
    declare: bool = True,
) -> MsgGenerator[Any]:
    """Prepare live flyers and optionally declare their event stream.

    Staging remains the caller's responsibility so one host can stage several
    device groups under its own resource policy. ``trigger_info`` is passed
    through to the Bluesky preparable protocol without importing a host type.
    """
    if not detectors:
        raise ValueError("at least one live detector is required")
    if not stream_name:
        raise ValueError("stream_name must not be empty")
    for detector in detectors:
        yield from bps.prepare(detector, trigger_info, wait=True)
    if declare:
        yield from bps.declare_stream(*detectors, name=stream_name, collect=collect)


def teardown_live_stream(
    detectors: Sequence[LiveFlyer],
    stream_name: str,
    *,
    flush_period_s: float = 0.5,
    on_complete: Callable[[LiveFlyer, Any], Iterable[Any]] | None = None,
) -> MsgGenerator[Any]:
    """Complete live flyers while collecting written assets, then unstage.

    ``on_complete`` lets an execution host attach progress reporting without
    bringing its presenter framework into this package.
    """
    _validate_live_stream_inputs(detectors, stream_name, flush_period_s)
    group = uuid4().hex
    for detector in detectors:
        status = yield from bps.complete(detector, wait=False, group=group)
        if on_complete is not None:
            yield from on_complete(detector, status)
    done = False
    while not done:
        done = yield from bps.wait(
            group=group, timeout=flush_period_s, error_on_timeout=False
        )
        yield from bps.collect(*detectors, name=stream_name)
    yield from bps.unstage_all(*detectors)


def collect_while_waiting_plan(
    plan: Iterable[Any],
    detectors: Sequence[LiveFlyer],
    stream_name: str,
    *,
    flush_period_s: float = 0.5,
    on_flush: Callable[[LiveFlyer], Iterable[Any]] | None = None,
) -> MsgGenerator[Any]:
    """Collect live flyers each time a wrapped plan sleeps for a flush period.

    The elapsed value is accumulated from requested Bluesky sleep messages.
    An optional host callback can wait on device counters and update GUI
    progress after each collection.
    """
    _validate_live_stream_inputs(detectors, stream_name, flush_period_s)
    slept_s = 0.0

    def flush() -> Iterable[Any]:
        yield from bps.collect(*detectors, name=stream_name)
        if on_flush is not None:
            for detector in detectors:
                yield from on_flush(detector)

    def collect_after_sleep(msg: Any) -> tuple[None, Iterable[Any] | None]:
        nonlocal slept_s
        if msg.command != "sleep":
            return None, None
        slept_s += msg.args[0]
        if slept_s < flush_period_s:
            return None, None
        slept_s = 0.0
        return None, flush()

    yield from bpp.plan_mutator(plan, collect_after_sleep)


def live_stream(
    detectors: Sequence[LiveFlyer],
    stream_name: str,
    *,
    parent: str,
    median_scan: str | None = None,
    until: Iterable[Any] | None = None,
    flush_period_s: float = 0.5,
    on_complete: Callable[[LiveFlyer, Any], Iterable[Any]] | None = None,
    on_flush: Callable[[LiveFlyer], Iterable[Any]] | None = None,
) -> MsgGenerator[Any]:
    """Write one capture from already-prepared live flyers as a nested run.

    The host owns detector staging/preparation, the action UI, resource locks
    and any hardware policy. ``until`` can keep the capture open while another
    plan runs; assets are periodically collected during its sleeps. This is a
    Bluesky generator and does not create or require a RunEngine.
    """
    _validate_live_stream_inputs(detectors, stream_name, flush_period_s)
    if not isinstance(parent, str) or not parent.strip():
        raise ValueError("parent must be a nonempty run UID")
    if median_scan is not None and (
        not isinstance(median_scan, str) or not median_scan.strip()
    ):
        raise ValueError("median_scan must be a nonempty run UID when supplied")
    uid = yield from bps.open_run(
        md={"purpose": "capture", "parent": parent, "median_scan": median_scan}
    )
    yield from bps.declare_stream(*detectors, name=stream_name, collect=True)
    yield from bps.kickoff_all(*detectors, wait=True)
    if until is not None:
        yield from collect_while_waiting_plan(
            until,
            detectors,
            stream_name,
            flush_period_s=flush_period_s,
            on_flush=on_flush,
        )
    yield from teardown_live_stream(
        detectors,
        stream_name,
        flush_period_s=flush_period_s,
        on_complete=on_complete,
    )
    yield from bps.close_run()
    return uid


def live_median_scan(
    detectors: Sequence[LiveFlyer],
    motor: XYMotor,
    step_um: float,
    frames_per_side: int,
    *,
    parent: str | None = None,
    purpose: str = "median_scan",
    stream_name: str = "median_scan",
    on_frame: Callable[[int, int, Any, float], Iterable[Any]] | None = None,
) -> MsgGenerator[Any]:
    """Acquire a square-motion frame stack for a host's temporal median.

    The square visits +X, +Y, -X and -Y for ``frames_per_side`` samples on
    each leg, closing at its starting position. Frames are emitted with the
    detector buffers and motor readback in a Bluesky event. The host owns the
    median calculation, run-key routing, progress UI and any device policy.
    """
    if not detectors:
        raise ValueError("at least one live detector is required")
    if (
        isinstance(step_um, bool)
        or not isinstance(step_um, (int, float))
        or not isfinite(float(step_um))
        or step_um <= 0
    ):
        raise ValueError("step_um must be finite and positive")
    if (
        isinstance(frames_per_side, bool)
        or not isinstance(frames_per_side, int)
        or frames_per_side < 1
    ):
        raise ValueError("frames_per_side must be a positive integer")
    if not purpose or not stream_name:
        raise ValueError("purpose and stream_name must not be empty")
    if parent is not None and (not isinstance(parent, str) or not parent.strip()):
        raise ValueError("parent must be a nonempty run UID when supplied")
    axes = motor.axis
    if not {"x", "y"}.issubset(axes):
        raise TypeError("motor must expose x and y movable axes")
    x, y = axes["x"], axes["y"]
    square = [
        (axis, direction)
        for axis, direction in (
            (x, float(step_um)),
            (y, float(step_um)),
            (x, -float(step_um)),
            (y, -float(step_um)),
        )
        for _ in range(frames_per_side)
    ]
    uid = yield from bps.open_run(md={"purpose": purpose, "parent": parent})
    for frame_index, (axis, direction) in enumerate(square, start=1):
        for detector in detectors:
            yield from bps.trigger(detector, wait=True)
        yield from bps.create(name=stream_name)
        for detector in detectors:
            yield from bps.read(detector.buffer)
        yield from bps.read(motor)
        yield from bps.save()
        if on_frame is not None:
            yield from on_frame(frame_index, len(square), axis, direction)
        yield from bps.mvr(axis, direction)
    yield from bps.close_run()
    return uid


def _validate_live_stream_inputs(
    detectors: Sequence[LiveFlyer], stream_name: str, flush_period_s: float
) -> None:
    """Validate shared live-stream inputs before emitting Bluesky messages."""
    if not detectors:
        raise ValueError("at least one live detector is required")
    if not stream_name:
        raise ValueError("stream_name must not be empty")
    if (
        isinstance(flush_period_s, bool)
        or not isinstance(flush_period_s, (int, float))
        or not isfinite(float(flush_period_s))
        or flush_period_s <= 0
    ):
        raise ValueError("flush_period_s must be finite and positive")


def hand_eye_calibration_plan(
    *,
    anchor_xy_um: tuple[float, float],
    stage_xy: Any | None = None,
    stage_x: Any | None = None,
    stage_y: Any | None = None,
    camera: Any,
    x_readback: Any | None = None,
    y_readback: Any | None = None,
    x_probe_um: float = 10.0,
    y_probe_um: float = 10.0,
    repeats_per_direction: int = 1,
    grid_shape_xy: tuple[int, int] | None = None,
    grid_spacing_xy_um: tuple[float, float] | None = None,
    mode_context: Mapping[str, str],
    readables: Sequence[Any] = (),
    coordinate_frame: str,
    registration_roi_yxhw_px: tuple[int, int, int, int] | None = None,
    registration_preprocessing: str = "none",
    capture_one: Callable[[Any, Sequence[Any], str], Iterable[Any]] | None = None,
    md: Mapping[str, Any] | None = None,
    stream_name: str = "calibration",
) -> MsgGenerator[Any]:
    """Acquire a bounded hand-eye sequence using probes or a dense XY grid.

    The host supplies a preflighted anchor and device objects. In probe mode,
    it captures signed X/Y endpoints and returns to the anchor after each
    probe. In grid mode, it captures a centered odd-sized lattice in snake
    order, then returns to the anchor for a closure frame. Both modes record
    actual stage readbacks with every image event. Return targets use the
    preflight anchor in stage units; this plan does not derive a return command
    from raw encoder-count deltas. Device identity, physical limits, settling
    semantics and profile preflight remain the host's responsibility; this
    plan is not authorization to move.

    Registration and fitting happen after collection with
    :func:`planner.hand_eye.fit_hand_eye`, using measured readback deltas and
    direct registered feature motion. ``capture_one`` lets an execution host
    add per-image behavior such as shutter sequencing while keeping
    device-specific commands outside the planner package.
    """
    if len(anchor_xy_um) != 2 or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in anchor_xy_um
    ):
        raise ValueError("anchor_xy_um must be a finite XY pair in micrometres")
    for name, value in (("x_probe_um", x_probe_um), ("y_probe_um", y_probe_um)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")
    if (grid_shape_xy is None) != (grid_spacing_xy_um is None):
        raise ValueError("grid shape and spacing must be provided together")
    grid_enabled = grid_shape_xy is not None
    if grid_enabled:
        assert grid_shape_xy is not None and grid_spacing_xy_um is not None
        if (
            len(grid_shape_xy) != 2
            or any(
                isinstance(v, bool) or not isinstance(v, int) or v < 3 or v % 2 == 0
                or v > 15
                for v in grid_shape_xy
            )
            or grid_shape_xy[0] * grid_shape_xy[1] > 225
        ):
            raise ValueError("grid_shape_xy must be odd dimensions from 3 to 15")
        if (
            len(grid_spacing_xy_um) != 2
            or any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                or v <= 0
                for v in grid_spacing_xy_um
            )
        ):
            raise ValueError("grid_spacing_xy_um must contain positive finite XY steps")
        if repeats_per_direction != 1:
            raise ValueError("grid captures do not combine with probe repeats")
        columns, rows = grid_shape_xy
        sx, sy = (float(v) for v in grid_spacing_xy_um)
        capture_labels = (
            "reference",
            *(
                f"grid_y{row:02d}_x{column:02d}"
                for row in range(rows)
                for column in (
                    range(columns)
                    if row % 2 == 0
                    else range(columns - 1, -1, -1)
                )
            ),
            "anchor_return",
        )
    else:
        capture_labels = hand_eye_capture_labels(repeats_per_direction)
    if not coordinate_frame:
        raise ValueError("coordinate_frame is required")
    if not mode_context or any(
        not isinstance(key, str) or not key or not isinstance(value, str) or not value
        for key, value in mode_context.items()
    ):
        raise ValueError("mode_context must map nonempty strings to nonempty strings")
    if not stream_name:
        raise ValueError("stream_name must not be empty")
    if registration_roi_yxhw_px is not None and (
        len(registration_roi_yxhw_px) != 4
        or any(
            isinstance(v, bool) or not isinstance(v, int)
            for v in registration_roi_yxhw_px
        )
        or registration_roi_yxhw_px[0] < 0
        or registration_roi_yxhw_px[1] < 0
        or registration_roi_yxhw_px[2] < 16
        or registration_roi_yxhw_px[3] < 16
    ):
        raise ValueError("registration ROI must be nonnegative YXHW with H,W >= 16")
    if registration_preprocessing not in {"none", "percentile_clahe"}:
        raise ValueError(
            "registration_preprocessing must be 'none' or 'percentile_clahe'"
        )
    registration = {
        "roi_yxhw_px": list(registration_roi_yxhw_px)
        if registration_roi_yxhw_px
        else None,
        "preprocessing": registration_preprocessing,
    }
    calibration_context = dict(mode_context)
    registration_identity = {
        "registration_roi_yxhw_px": (
            ",".join(str(v) for v in registration_roi_yxhw_px)
            if registration_roi_yxhw_px
            else "full-frame"
        ),
        "registration_preprocessing": registration_preprocessing,
    }
    for identity_key, identity_value in registration_identity.items():
        if (
            identity_key in calibration_context
            and calibration_context[identity_key] != identity_value
        ):
            raise ValueError(
                f"mode_context {identity_key!r} disagrees with registration configuration"
            )
        calibration_context[identity_key] = identity_value
    if stage_xy is None:
        if stage_x is None or stage_y is None:
            raise ValueError("provide one combined XY stage or both scalar stage axes")
        if x_readback is None or y_readback is None:
            raise ValueError("scalar stage axes require both XY readbacks")
        if stage_x is stage_y or x_readback is y_readback:
            raise ValueError("X and Y stage axes/readbacks must be distinct")
        if any(
            device is camera for device in (stage_x, stage_y, x_readback, y_readback)
        ):
            raise ValueError("camera and stage/readback devices must be distinct")
    else:
        if (
            stage_x is not None
            or stage_y is not None
            or x_readback is not None
            or y_readback is not None
        ):
            raise ValueError(
                "combined XY stage cannot be combined with scalar axes/readbacks"
            )
        if stage_xy is camera:
            raise ValueError("camera and XY stage devices must be distinct")

    ax, ay = float(anchor_xy_um[0]), float(anchor_xy_um[1])
    schedule = [
        ("reference", ax, ay),
        ("reference_repeat", ax, ay),
    ]
    probe_schedule = (
        ("probe_x_positive", "return_x_positive", ax + x_probe_um, ay),
        ("probe_x_negative", "return_x_negative", ax - x_probe_um, ay),
        ("probe_y_positive", "return_y_positive", ax, ay + y_probe_um),
        ("probe_y_negative", "return_y_negative", ax, ay - y_probe_um),
    )
    for repeat_index in range(1, repeats_per_direction + 1):
        suffix = f"_repeat_{repeat_index:02d}" if repeats_per_direction > 1 else ""
        for probe_label, return_label, x_target, y_target in probe_schedule:
            schedule.extend(
                (
                    (f"{probe_label}{suffix}", x_target, y_target),
                    (f"{return_label}{suffix}", ax, ay),
                )
            )
    if grid_enabled:
        schedule = [("reference", ax, ay)]
        center_x, center_y = (columns - 1) / 2, (rows - 1) / 2
        for row in range(rows):
            column_order = (
                range(columns)
                if row % 2 == 0
                else range(columns - 1, -1, -1)
            )
            for column in column_order:
                schedule.append(
                    (
                        "grid_y{:02d}_x{:02d}".format(row, column),
                        ax + (column - center_x) * sx,
                        ay + (row - center_y) * sy,
                    )
                )
        schedule.append(("anchor_return", ax, ay))

    planner_md = _metadata(
        md,
        {
            "planner_schema": "redsun-planner-run-v1",
            "planner_workflow": (
                "stage-camera-hand-eye-grid-calibration-v1"
                if grid_enabled
                else "stage-camera-hand-eye-calibration-v1"
            ),
            "planner_coordinate_frame": coordinate_frame,
            "planner_calibration_mode_context": dict(
                sorted(calibration_context.items())
            ),
            "planner_hand_eye_registration": registration,
            "planner_anchor_xy_um": [ax, ay],
            "planner_probe_offsets_um": {
                "x": float(x_probe_um),
                "y": float(y_probe_um),
            },
            "planner_calibration_probe_repeats": repeats_per_direction,
            "planner_calibration_capture_labels": list(capture_labels),
            "planner_hand_eye_grid": (
                {
                    "shape_xy": [columns, rows],
                    "spacing_xy_um": [sx, sy],
                    "traversal": "snake",
                    "centered_on_anchor": True,
                }
                if grid_enabled
                else None
            ),
        },
    )
    detectors = (
        [camera, stage_xy] if stage_xy is not None else [camera, x_readback, y_readback]
    )
    detectors.extend(d for d in readables if all(d is not known for known in detectors))

    def body() -> Iterable[Any]:
        for _label, x_target, y_target in schedule:
            # One bounded XY point move per probe/return; wait for the host
            # device's normal completion semantics before acquiring an image.
            if stage_xy is not None:
                yield from bps.mv(stage_xy, (x_target, y_target))
            else:
                yield from bps.mv(stage_x, x_target, stage_y, y_target)
            if capture_one is None:
                yield from bps.trigger_and_read(detectors, name=stream_name)
            else:
                yield from capture_one(camera, detectors, stream_name)

    run_wrapped = bpp.run_decorator(md=planner_md)(body)
    staged = bpp.stage_decorator([camera])(run_wrapped)
    yield from staged()


def snap_scan_plan(
    recipe: StageScanRecipe | SnapScanPlan | ObservedSnapGridFit,
    *,
    stage_xy: Any | None = None,
    stage_x: Any | None = None,
    stage_y: Any | None = None,
    camera: Any,
    readables: Sequence[Any] = (),
    prepare_camera: Callable[[Any], Iterable[Any]] | None = None,
    prepare_capture: Callable[[Any, int], Iterable[Any]] | None = None,
    capture_one: Callable[[Any, bool], Iterable[Any]] | None = None,
    md: Mapping[str, Any] | None = None,
    stream_name: str = "primary",
) -> MsgGenerator[Any]:
    """Move to each validated XY point, trigger/read, and emit one run.

    The executor must preflight the recipe against actual limits and adapter
    capabilities before passing it to a RunEngine. ``readables`` should
    include stage readback signals when the motor objects themselves do not
    expose readable positions. Event order corresponds to ``position_ids`` in
    the run metadata; run subscribers should preserve event sequence numbers.
    Camera preparation is supplied by the host adapter, which can translate
    snaps into the detector framework's finite-trigger contract without
    coupling this reusable plan to a particular device library. ``prepare_camera``
    runs once before each trigger/read pair. Streamable detectors can instead
    use ``prepare_capture`` once with the total frame count and ``capture_one``
    for each frame, allowing one external capture resource to span the scan.
    Camera configuration is otherwise a separate adapter-owned plan or preflight
    step.
    An :class:`ObservedSnapGridFit` uses its explicit pixel-to-stage transform
    and embeds observed-coverage provenance; it does not assert a
    ``ConfirmedBoundary`` or specimen boundary.
    """
    if isinstance(recipe, StageScanRecipe):
        positions = tuple(recipe.positions)
        identity: dict[str, Any] = {
            "planner_recipe_id": recipe.recipe_id,
            "planner_recipe_version": recipe.version,
            "planner_specimen_id": recipe.specimen_id,
            "planner_coordinate_frame": recipe.coordinate_frame,
            "planner_calibration_id": recipe.calibration_id,
            "planner_phase": recipe.phase.value,
        }
        settings: dict[str, Any] = {
            "exposure_ms": recipe.settings.exposure_ms,
            "frames_per_position": recipe.settings.frames,
            "channel_id": recipe.settings.channel_id,
            "roi_origin": recipe.settings.roi_origin,
            "binning": recipe.settings.binning,
        }
        scan_mode = "snap"
    elif isinstance(recipe, SnapScanPlan):
        positions = tuple(
            ScanPosition(
                point.position_id, point.x_um, point.y_um, point.row, point.column
            )
            for point in recipe.positions
        )
        identity = {
            "planner_recipe_id": recipe.boundary_id,
            "planner_recipe_version": "bounded-snap-v1",
            "planner_coordinate_frame": recipe.coordinate_frame,
            "planner_boundary_id": recipe.boundary_id,
            "planner_footprint_xy_um": list(recipe.footprint_xy_um),
            "planner_overlap": recipe.overlap,
            "planner_coverage_claim": recipe.coverage_claim,
        }
        settings = {}
        scan_mode = "snap"
    elif isinstance(recipe, ObservedSnapGridFit):
        if not recipe.within_envelope:
            raise ValueError(
                "observed-coverage snap fit contains positions outside its declared stage envelope"
            )
        positions = tuple(
            ScanPosition(
                tile.position_id,
                tile.center_stage_xy_um[0],
                tile.center_stage_xy_um[1],
                tile.row,
                tile.column,
            )
            for tile in recipe.positions
        )
        identity = {
            "planner_recipe_id": recipe.coverage.coverage_id,
            "planner_recipe_version": "observed-snap-v1",
            "planner_coordinate_frame": recipe.coverage.stage_coordinate_frame,
            "planner_scan_boundary_type": "registered-observed-coverage",
            "planner_source_run_id": recipe.coverage.source_run_id,
            "planner_registration_id": recipe.coverage.registration_id,
            "planner_calibration_id": recipe.coverage.calibration_id,
            "planner_coverage_fit": recipe.to_record(),
        }
        settings = {}
        scan_mode = "snap"
    else:
        raise TypeError(
            "recipe must be a StageScanRecipe, SnapScanPlan, or ObservedSnapGridFit"
        )
    if not positions:
        raise ValueError("snap recipe has no positions")
    if not stream_name:
        raise ValueError("stream_name must not be empty")
    if stage_xy is None and (stage_x is None or stage_y is None):
        raise ValueError(
            "provide one combined stage_xy device or both stage_x and stage_y"
        )
    if stage_xy is not None and (stage_x is not None or stage_y is not None):
        raise ValueError(
            "stage_xy cannot be combined with separate stage_x/stage_y devices"
        )
    position_ids = [position.position_id for position in positions]
    if len(set(position_ids)) != len(position_ids):
        raise ValueError("position IDs must be unique")
    frames_per_position = int(settings.get("frames_per_position", 1))
    total_frames = len(positions) * frames_per_position
    event_position_ids = [
        position.position_id
        for position in positions
        for _ in range(frames_per_position)
    ]
    event_frame_indices = [
        frame_index
        for _position in positions
        for frame_index in range(frames_per_position)
    ]
    planner_md = _metadata(
        md,
        {
            "planner_schema": "redsun-planner-run-v1",
            "planner_scan_mode": scan_mode,
            "planner_position_ids": position_ids,
            "planner_event_position_ids": event_position_ids,
            "planner_event_frame_indices": event_frame_indices,
            "planner_expected_event_count": len(event_position_ids),
            **identity,
            **settings,
        },
    )

    def body() -> Iterable[Any]:
        capture_started = False
        if prepare_capture is not None:
            yield from prepare_capture(camera, total_frames)
        for position in positions:
            if stage_xy is not None:
                yield from bps.mv(stage_xy, (position.x_um, position.y_um))
            else:
                yield from bps.mv(stage_x, position.x_um, stage_y, position.y_um)
            for _frame_index in range(frames_per_position):
                if capture_one is not None:
                    yield from capture_one(camera, not capture_started)
                    capture_started = True
                else:
                    if prepare_camera is not None:
                        yield from prepare_camera(camera)
                    yield from bps.trigger_and_read(
                        [camera, *readables], name=stream_name
                    )

    run_wrapped = bpp.run_decorator(md=planner_md)(body)
    staged = bpp.stage_decorator([camera])(run_wrapped)
    yield from staged()


def push_broom_plan(
    recipe: BroomScanPlan,
    *,
    flyer: Any,
    detectors: Sequence[Any] = (),
    stream_writers: Sequence[Any] = (),
    md: Mapping[str, Any] | None = None,
    stream_name: str = "primary",
    progress_callback: Callable[[Any, int, int], Iterable[Any]] | None = None,
    validate_strip: Callable[[Any, int, int], None] | None = None,
    post_scan: Callable[[], Iterable[Any]] | None = None,
) -> MsgGenerator[Any]:
    """Prepare and collect each continuous strip from Bluesky devices.

    With no ``stream_writers``, ``flyer`` and ``detectors`` contribute inline
    event data to one stream. When a camera stores frames as external stream
    assets, pass an event-only geometry ``flyer`` and the camera writer(s) in
    ``stream_writers``. The plan collects geometry events first, then the
    external asset documents; each geometry event carries the camera's
    resource-local zero-based store index. The event's RunEngine ``seq_num``
    selects the matching ``stream_datum.seq_nums`` range, and that datum's
    ``indices`` range resolves the resource-local image index.
    This split is required because Bluesky's asset-writer collect path does not
    also emit that object's ``collect()`` events.

    ``progress_callback``, when provided with external ``stream_writers``, is
    called with the flyer's completion status and the one-based strip index and
    total strip count. The host may use the status watcher to report frame
    progress while the strip completes. If completion or monitoring fails, the
    plan still collects that strip's available geometry and asset documents
    before propagating the failure. ``validate_strip``, when provided, runs
    after a strip's geometry and asset documents have been collected; the host
    can then reject a domain-level terminal result without losing that evidence.
    ``post_scan``, when provided, runs after all strips have completed and been
    validated, but before the run closes. It is skipped if a strip fails or the
    run is interrupted, so a host can use it for successful-run actions such as
    returning the stage to its recorded anchor.

    ``flyer.prepare(strip=...)`` must configure a bounded strip without
    starting motion, and its event stream must include frame identity, image
    store index, stage pose, exposure interval and clock provenance. Each
    stream writer must declare its external data key and emit matching stream
    datum indices. The executor still owns limits, shutter/profile policy,
    resource locks and cancellation/outcome-unknown handling.
    """
    if not isinstance(recipe, BroomScanPlan):
        raise TypeError("recipe must be a BroomScanPlan")
    if not recipe.strips:
        raise ValueError("broom recipe has no strips")
    if not stream_name:
        raise ValueError("stream_name must not be empty")
    if stream_writers and detectors:
        raise ValueError(
            "use either inline detectors or external stream_writers, not both"
        )
    if progress_callback is not None and not stream_writers:
        raise ValueError("progress_callback requires external stream_writers")
    if validate_strip is not None and not stream_writers:
        raise ValueError("validate_strip requires external stream_writers")
    planner_md = _metadata(
        md,
        {
            "planner_schema": "redsun-planner-run-v1",
            "planner_scan_mode": "push-broom",
            "planner_recipe_id": recipe.boundary_id,
            "planner_recipe_version": "bounded-broom-v1",
            "planner_coordinate_frame": recipe.coordinate_frame,
            "planner_boundary_id": recipe.boundary_id,
            "planner_footprint_xy_um": list(recipe.footprint_xy_um),
            "planner_overlap": recipe.overlap,
            "planner_coverage_claim": recipe.coverage_claim,
            "planner_expected_strip_count": len(recipe.strips),
        },
    )

    collectables = [flyer]
    collectables.extend(detector for detector in detectors if detector is not flyer)
    writer_streams = tuple(
        (writer, f"{stream_name}_assets_{index:02d}")
        for index, writer in enumerate(stream_writers)
    )

    active_strip = False
    current_strip_collected = True

    def collect_strip() -> Iterable[Any]:
        yield from bps.collect(flyer, name=stream_name)
        for writer, writer_stream_name in writer_streams:
            yield from bps.collect(writer, name=writer_stream_name)

    def body() -> Iterable[Any]:
        nonlocal active_strip, current_strip_collected
        if writer_streams:
            # Keep geometry events distinct from external image stream datums.
            yield from bps.declare_stream(flyer, name=stream_name, collect=True)
            for writer, writer_stream_name in writer_streams:
                yield from bps.declare_stream(
                    writer, name=writer_stream_name, collect=True
                )
        else:
            # Bluesky requires a collect descriptor before the first collect.
            yield from bps.declare_stream(*collectables, name=stream_name, collect=True)
        for strip_index, strip in enumerate(recipe.strips, start=1):
            yield from bps.prepare(flyer, strip=strip, wait=True)
            yield from bps.kickoff(flyer, wait=False)
            active_strip = True
            current_strip_collected = False
            if writer_streams:
                group = uuid4().hex
                status = yield from bps.complete(flyer, group=group, wait=False)
                if progress_callback is not None:
                    yield from progress_callback(
                        status, strip_index, len(recipe.strips)
                    )
                yield from bps.wait(group=group)
                yield from collect_strip()
                current_strip_collected = True
                active_strip = False
                if validate_strip is not None:
                    validate_strip(flyer, strip_index, len(recipe.strips))
            else:
                yield from bps.collect_while_completing(
                    [flyer], collectables, stream_name=stream_name
                )
            current_strip_collected = True
            active_strip = False
        if post_scan is not None:
            yield from post_scan()

    def collect_interrupted_strip() -> Iterable[Any]:
        """Stop an interrupted fly and collect its accepted partial prefix."""
        if active_strip and not current_strip_collected:
            yield from bps.stop(flyer)
            if writer_streams:
                yield from collect_strip()
            else:
                yield from bps.collect(flyer, *collectables[1:], name=stream_name)

    def body_with_finalizer() -> Iterable[Any]:
        yield from bpp.finalize_wrapper(body(), collect_interrupted_strip)

    run_wrapped = bpp.run_decorator(md=planner_md)(body_with_finalizer)
    staged = bpp.stage_decorator([flyer])(run_wrapped)
    yield from staged()


def replay_dataset_plan(
    source: ReplaySource,
    *,
    interval_s: float = 0.0,
    md: Mapping[str, Any] | None = None,
    stream_name: str = "primary",
) -> MsgGenerator[Any]:
    """Emit a saved dataset as normal Bluesky image/provenance events.

    The source is iterated lazily and one image frame is held at a time. Event
    documents carry the original frame ID/time and caller-provided scalar
    source context; the RunEngine event time records when replay emitted each
    frame. This plan makes no device calls and records that replay is
    simulation-only. Source-specific TIFF/Zarr reading and all host services,
    presentation and storage decisions stay in the consuming application.

    Parameters
    ----------
    source
        Streaming source that exposes stable identity/metadata and produces
        descriptor-consistent :class:`ReplayFrame` records.
    interval_s
        Wall-clock spacing between replayed events. It does not replace or
        alter timestamps embedded in the source frames.
    md
        Additional run metadata. Planner-owned fields cannot be overridden.
    stream_name
        Bluesky event stream name for the replay observations.
    """
    if not isinstance(source.source_id, str) or not source.source_id.strip():
        raise ValueError("replay source_id must be a nonempty string")
    if not stream_name:
        raise ValueError("stream_name must not be empty")
    if (
        isinstance(interval_s, bool)
        or not isinstance(interval_s, (int, float))
        or not isfinite(interval_s)
        or interval_s < 0
    ):
        raise ValueError("interval_s must be finite and nonnegative")
    source_metadata = dict(source.source_metadata)
    required_metadata = {
        "purpose": "dataset_replay",
        "planner_schema": "redsun-planner-run-v1",
        "planner_workflow": "dataset-frame-replay-v1",
        "planner_replay_source_id": source.source_id,
        "planner_replay_source_metadata": source_metadata,
        "planner_replay_pacing_interval_s": float(interval_s),
        "planner_simulation_only": True,
        "planner_hardware_commands_issued": False,
        "planner_execution_authorized": False,
    }
    replay_md = _metadata(md, required_metadata)
    try:
        json.dumps(replay_md, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "replay run metadata must be finite JSON-compatible data"
        ) from error

    frames = iter(source.iter_frames())
    first = next(frames, None)
    if first is None:
        raise ValueError("replay source contains no frames")
    if not isinstance(first, ReplayFrame):
        raise TypeError("replay source must yield ReplayFrame records")

    readable = _ReplayFrameReadable(first)
    yield from bps.open_run(md=replay_md)
    for index, frame in enumerate(chain((first,), frames)):
        if not isinstance(frame, ReplayFrame):
            raise TypeError("replay source must yield ReplayFrame records")
        readable.update(frame)
        if index and interval_s:
            yield from bps.sleep(float(interval_s))
        yield from bps.create(name=stream_name)
        yield from bps.read(readable)
        yield from bps.save()
    yield from bps.close_run()
