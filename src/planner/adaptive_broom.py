"""Adaptive broom raster plan with per-frame intensity-derived speed.

The plan walks a bounded rectangular field row by row. After each frame the host
supplies one nonnegative intensity measure, the existing
:class:`planner.speed_policy.SpeedBlendController` turns it into a bounded,
slew-limited speed request, and the plan advances to the next frame by
``commanded speed * frame period``. A flat field therefore runs at the fast
speed and covers the row in coarse steps; an intensity rise slows the request
and tightens the along-row sampling.

This module is hardware-free. It opens no device, computes no image statistics
and makes no claim about a controller's real acceleration. The host binds three
things: :func:`adaptive_broom_plan` asks for each frame's intensity through
``intensity_of_frame``, sends the commanded speed through ``command_speed``, and
receives one :class:`AdaptiveBroomFrameRecord` per frame through ``on_frame`` and
the plan's return value.

An optional brake turns a slow request into a hold: when the smoothed interest
reaches the brake threshold the raster stops advancing for
``brake_dwell_frames`` frames and keeps imaging the same field, then continues
along the row. That is the only brake a staged raster can honestly offer - the
stage is already at rest between frames, so the plan cancels the step it was
about to take rather than decelerating a moving stage. A controller that can
change velocity inside a traverse needs its own measured brake budget; see the
acquisition host's live-brake evidence.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence  # noqa: TC003
from dataclasses import dataclass
from math import ceil, floor, isfinite
from typing import Any

from bluesky import plan_stubs as bps
from bluesky import preprocessors as bpp
from bluesky.utils import MsgGenerator  # noqa: TC002

from .speed_policy import SpeedBlendConfig, SpeedBlendController

Point = tuple[float, float]
Box = tuple[Point, Point]

#: Reserved planner metadata keys; a host may not override them.
_RESERVED_METADATA = (
    "planner_schema",
    "planner_scan_mode",
    "planner_workflow",
    "planner_adaptive_spec",
    "planner_adaptive_frame_period_s",
    "planner_adaptive_speed_bounds_um_s",
    "planner_adaptive_stage_max_speed_um_s",
    "planner_adaptive_intensity_scale",
    "planner_adaptive_row_count",
    "planner_adaptive_max_frames",
    "planner_adaptive_brake",
)


def _finite(value: object, name: str, *, positive: bool) -> float:
    """Return a finite real number, rejecting booleans and out-of-range values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    if positive and number <= 0:
        raise ValueError(f"{name} must be positive")
    if not positive and number < 0:
        raise ValueError(f"{name} must be nonnegative")
    return number


def _signed(value: object, name: str) -> float:
    """Return a finite real number, rejecting booleans and non-finite values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _point(value: Any, name: str) -> Point:
    """Return a validated finite XY pair in micrometres."""
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must be a finite XY pair in micrometres")
    return (
        _signed(value[0], f"{name}[0]"),
        _signed(value[1], f"{name}[1]"),
    )


def _positive_pair(value: Any, name: str) -> tuple[float, float]:
    """Return a validated pair of finite positive micrometre lengths."""
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must contain two finite lengths")
    return (
        _finite(value[0], f"{name}[0]", positive=True),
        _finite(value[1], f"{name}[1]", positive=True),
    )


@dataclass(frozen=True)
class AdaptiveBroomSpec:
    """One bounded adaptive broom raster and its speed policy bounds.

    ``origin_xy_um`` is the first frame's stage position and the field extends
    positively from it. The complete field rectangle must fit inside
    ``stage_envelope_xy_um``, an absolute stage-coordinate box. ``slow`` and
    ``fast`` are the intensity-policy bounds, and ``fast`` may not exceed
    ``stage_max_speed_um_s``.

    ``brake_dwell_frames`` enables the hold: zero keeps the plain raster, and a
    positive count holds each slow frame for that many extra frames at the same
    point. ``brake_interest_threshold`` defaults to the policy's slow threshold,
    so the hold begins exactly where the policy has already asked for the slow
    speed; a declared value must not be below the policy's fast threshold,
    because a brake that fires while the policy still wants search speed would
    spend frames on background.
    """

    origin_xy_um: Point
    field_size_um: tuple[float, float]
    stage_envelope_xy_um: Box
    row_pitch_um: float
    frame_period_s: float
    slow_speed_um_s: float
    fast_speed_um_s: float
    stage_max_speed_um_s: float
    intensity_scale: float = 1.0
    max_frames: int = 20000
    brake_dwell_frames: int = 0
    brake_interest_threshold: float | None = None
    config: SpeedBlendConfig = SpeedBlendConfig()

    def __post_init__(self) -> None:
        """Validate the raster, its declared envelope and the speed bounds."""
        object.__setattr__(
            self, "origin_xy_um", _point(self.origin_xy_um, "origin_xy_um")
        )
        object.__setattr__(
            self, "field_size_um", _positive_pair(self.field_size_um, "field_size_um")
        )
        if (
            not isinstance(self.stage_envelope_xy_um, (tuple, list))
            or len(self.stage_envelope_xy_um) != 2
        ):
            raise ValueError(
                "stage_envelope_xy_um must contain two opposite XY corners"
            )
        lower = _point(self.stage_envelope_xy_um[0], "stage_envelope_xy_um[0]")
        upper = _point(self.stage_envelope_xy_um[1], "stage_envelope_xy_um[1]")
        if lower[0] >= upper[0] or lower[1] >= upper[1]:
            raise ValueError(
                "stage_envelope_xy_um corners must define a nonzero axis-aligned box"
            )
        object.__setattr__(self, "stage_envelope_xy_um", (lower, upper))
        for name in ("row_pitch_um", "frame_period_s", "intensity_scale"):
            object.__setattr__(
                self, name, _finite(getattr(self, name), name, positive=True)
            )
        for name in (
            "slow_speed_um_s",
            "fast_speed_um_s",
            "stage_max_speed_um_s",
        ):
            object.__setattr__(
                self, name, _finite(getattr(self, name), name, positive=True)
            )
        if not self.slow_speed_um_s <= self.fast_speed_um_s:
            raise ValueError("slow_speed_um_s must not exceed fast_speed_um_s")
        if not self.fast_speed_um_s <= self.stage_max_speed_um_s:
            raise ValueError(
                "fast_speed_um_s must not exceed the stage's declared maximum speed"
            )
        if (
            isinstance(self.max_frames, bool)
            or not isinstance(self.max_frames, int)
            or self.max_frames < 1
        ):
            raise ValueError("max_frames must be a positive integer")
        if not isinstance(self.config, SpeedBlendConfig):
            raise TypeError("config must be a SpeedBlendConfig")
        if (
            isinstance(self.brake_dwell_frames, bool)
            or not isinstance(self.brake_dwell_frames, int)
            or self.brake_dwell_frames < 0
        ):
            raise ValueError("brake_dwell_frames must be a nonnegative integer")
        if self.brake_interest_threshold is not None:
            threshold = _finite(
                self.brake_interest_threshold,
                "brake_interest_threshold",
                positive=False,
            )
            if threshold > 1.0:
                raise ValueError("brake_interest_threshold must not exceed 1")
            if threshold < self.config.interest_fast_threshold:
                raise ValueError(
                    "brake_interest_threshold must not be below the policy's "
                    "interest_fast_threshold, which still requests search speed"
                )
            object.__setattr__(self, "brake_interest_threshold", threshold)
        if self.brake_dwell_frames and self.brake_interest_threshold is None:
            object.__setattr__(
                self, "brake_interest_threshold", self.config.interest_slow_threshold
            )
        origin_x, origin_y = self.origin_xy_um
        width, height = self.field_size_um
        if (
            origin_x < lower[0]
            or origin_y < lower[1]
            or origin_x + width > upper[0]
            or origin_y + height > upper[1]
        ):
            raise ValueError(
                "field leaves its declared stage envelope: "
                f"field ({origin_x}, {origin_y}) + ({width}, {height}) um is outside "
                f"({lower[0]}, {lower[1]})..({upper[0]}, {upper[1]}) um"
            )

    @property
    def row_count(self) -> int:
        """Number of raster rows between the field's lower and upper Y edges."""
        _, height = self.field_size_um
        return max(1, int(floor(height / self.row_pitch_um)) + 1)

    @property
    def frame_count_bounds(self) -> tuple[int, int]:
        """Estimated minimum and maximum frame counts over the raster.

        The count is not fixed: it follows the commanded speed and, when the
        brake is enabled, the held frames it adds. These bounds bracket the
        along-row step by the fast and slow speeds and reserve the declared hold
        for every slow frame.
        """
        width, _ = self.field_size_um
        fast_step = self.fast_speed_um_s * self.frame_period_s
        slow_step = self.slow_speed_um_s * self.frame_period_s
        per_row_fast = max(1, int(ceil(width / fast_step)) + 1)
        per_row_slow = max(1, int(ceil(width / slow_step)) + 1)
        hold = 1 + self.brake_dwell_frames
        return (
            self.row_count * per_row_fast,
            min(self.max_frames, self.row_count * per_row_slow * hold),
        )


@dataclass(frozen=True)
class AdaptiveBroomFrameRecord:
    """One frame's position, intensity evidence and commanded speed.

    ``dwell_index`` is zero for a frame that advances the raster and 1..N for the
    held frames of a brake, whose ``step_um`` is zero because the stage stays
    where the braking frame left it.
    """

    frame_index: int
    row: int
    column: int
    x_um: float
    y_um: float
    intensity: float
    interest: float
    blend_fraction: float
    desired_speed_um_s: float
    commanded_speed_um_s: float
    step_um: float
    dwell_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe per-frame record."""
        return {
            "frame_index": self.frame_index,
            "row": self.row,
            "column": self.column,
            "x_um": self.x_um,
            "y_um": self.y_um,
            "intensity": self.intensity,
            "interest": self.interest,
            "blend_fraction": self.blend_fraction,
            "desired_speed_um_s": self.desired_speed_um_s,
            "commanded_speed_um_s": self.commanded_speed_um_s,
            "step_um": self.step_um,
            "dwell_index": self.dwell_index,
        }


@dataclass(frozen=True)
class AdaptiveBroomResult:
    """Completed or truncated adaptive raster and its per-frame records."""

    spec: AdaptiveBroomSpec
    records: tuple[AdaptiveBroomFrameRecord, ...]
    rows: int
    truncated: bool
    brake_count: int = 0

    @property
    def frame_count(self) -> int:
        """Number of frames the run actually recorded."""
        return len(self.records)


class _FrameRecordReadable:
    """Expose the current adaptive frame record as inline Bluesky event data."""

    name = "adaptive_broom_frame"
    parent = None

    _KEYS = (
        "adaptive_frame_index",
        "adaptive_row",
        "adaptive_column",
        "adaptive_intensity",
        "adaptive_interest",
        "adaptive_blend_fraction",
        "adaptive_desired_speed_um_s",
        "adaptive_speed_um_s",
        "adaptive_step_um",
        "adaptive_dwell_index",
    )

    def __init__(self) -> None:
        self._record: AdaptiveBroomFrameRecord | None = None

    def update(self, record: AdaptiveBroomFrameRecord) -> None:
        """Set the record the next read reports."""
        self._record = record

    async def read(self) -> dict[str, dict[str, Any]]:
        """Return the current record's scalar readings."""
        record = self._record
        if record is None:
            raise RuntimeError("adaptive broom frame record is not set")
        values = (
            record.frame_index,
            record.row,
            record.column,
            record.intensity,
            record.interest,
            record.blend_fraction,
            record.desired_speed_um_s,
            record.commanded_speed_um_s,
            record.step_um,
            record.dwell_index,
        )
        return {
            key: {"value": value, "timestamp": 0.0}
            for key, value in zip(self._KEYS, values)
        }

    async def describe(self) -> dict[str, dict[str, Any]]:
        """Describe the fixed per-frame record keys."""
        return {
            key: {"source": "planner.adaptive_broom", "dtype": "number", "shape": []}
            for key in self._KEYS
        }


def _metadata(
    md: Mapping[str, Any] | None, required: Mapping[str, Any]
) -> dict[str, Any]:
    result = dict(md or {})
    reserved = set(_RESERVED_METADATA).intersection(result)
    if reserved:
        names = ", ".join(sorted(reserved))
        raise ValueError(f"metadata cannot override planner fields: {names}")
    result.update(required)
    return result


def adaptive_broom_plan(
    spec: AdaptiveBroomSpec,
    *,
    stage_xy: Any | None = None,
    stage_x: Any | None = None,
    stage_y: Any | None = None,
    camera: Any,
    intensity_of_frame: Callable[[int], MsgGenerator[float]],
    command_speed: Callable[[float], Iterable[Any]] | None = None,
    on_frame: Callable[[AdaptiveBroomFrameRecord], None] | None = None,
    readables: Sequence[Any] = (),
    md: Mapping[str, Any] | None = None,
    stream_name: str = "adaptive_broom",
) -> MsgGenerator[AdaptiveBroomResult]:
    """Raster a bounded field and set each frame's speed from its intensity.

    The host supplies ``intensity_of_frame(index)`` as a Bluesky plan that
    returns one finite nonnegative intensity measure (for example the mean of
    the camera's live buffer). ``command_speed(speed_um_s)`` is called after
    every frame with the bounded request; it is where a host applies a speed
    port, and it may be omitted when the host only records the value.
    ``on_frame`` receives each record as it is produced, so a host can retain
    partial evidence when a later frame fails.

    The run advances by ``commanded_speed_um_s * frame_period_s``, so the same
    request both bounds the policy and sets the along-row sampling. Because the
    last frame of a row lands within one step of the edge, the plan claims row
    coverage within that step, not exact edge coverage.

    With ``spec.brake_dwell_frames`` the plan also holds: when the smoothed
    interest reaches ``spec.brake_interest_threshold`` it keeps imaging the same
    field for that many frames, each with a zero step and a zero speed request,
    before it advances. The hold is a cancelled step, not a deceleration - the
    stage is at rest between frames on this raster - and the result counts the
    holds in ``brake_count``.
    """
    if not isinstance(spec, AdaptiveBroomSpec):
        raise TypeError("spec must be an AdaptiveBroomSpec")
    if stage_xy is None and (stage_x is None or stage_y is None):
        raise ValueError(
            "provide one combined stage_xy device or both stage_x and stage_y"
        )
    if stage_xy is not None and (stage_x is not None or stage_y is not None):
        raise ValueError(
            "stage_xy cannot be combined with separate stage_x/stage_y devices"
        )
    if not callable(intensity_of_frame):
        raise TypeError("intensity_of_frame must be a callable plan")
    if command_speed is not None and not callable(command_speed):
        raise TypeError("command_speed must be a callable plan")
    if on_frame is not None and not callable(on_frame):
        raise TypeError("on_frame must be callable")
    if not stream_name:
        raise ValueError("stream_name must not be empty")

    planner_md = _metadata(
        md,
        {
            "planner_schema": "redsun-planner-run-v1",
            "planner_scan_mode": "adaptive-broom",
            "planner_workflow": "adaptive-broom-intensity-speed-v1",
            "planner_adaptive_spec": {
                "origin_xy_um": list(spec.origin_xy_um),
                "field_size_um": list(spec.field_size_um),
                "stage_envelope_xy_um": [
                    list(spec.stage_envelope_xy_um[0]),
                    list(spec.stage_envelope_xy_um[1]),
                ],
                "row_pitch_um": spec.row_pitch_um,
                "frame_period_s": spec.frame_period_s,
                "slow_speed_um_s": spec.slow_speed_um_s,
                "fast_speed_um_s": spec.fast_speed_um_s,
                "stage_max_speed_um_s": spec.stage_max_speed_um_s,
                "intensity_scale": spec.intensity_scale,
                "max_frames": spec.max_frames,
                "brake_dwell_frames": spec.brake_dwell_frames,
                "brake_interest_threshold": spec.brake_interest_threshold,
                "config": {
                    "interest_fast_threshold": spec.config.interest_fast_threshold,
                    "interest_slow_threshold": spec.config.interest_slow_threshold,
                    "smoothing_frames": spec.config.smoothing_frames,
                    "minimum_motion_confidence": (
                        spec.config.minimum_motion_confidence
                    ),
                    "max_acceleration_um_s2": spec.config.max_acceleration_um_s2,
                },
            },
            "planner_adaptive_frame_period_s": spec.frame_period_s,
            "planner_adaptive_speed_bounds_um_s": [
                spec.slow_speed_um_s,
                spec.fast_speed_um_s,
            ],
            "planner_adaptive_stage_max_speed_um_s": spec.stage_max_speed_um_s,
            "planner_adaptive_intensity_scale": spec.intensity_scale,
            "planner_adaptive_row_count": spec.row_count,
            "planner_adaptive_max_frames": spec.max_frames,
            "planner_adaptive_brake": (
                None
                if not spec.brake_dwell_frames
                else {
                    "kind": "step-cancel",
                    "dwell_frames": spec.brake_dwell_frames,
                    "interest_threshold": spec.brake_interest_threshold,
                }
            ),
        },
    )

    controller = SpeedBlendController(
        slow_speed_um_s=spec.slow_speed_um_s,
        fast_speed_um_s=spec.fast_speed_um_s,
        sample_interval_s=spec.frame_period_s,
        config=spec.config,
        initial_speed_um_s=spec.fast_speed_um_s,
    )
    record_readable = _FrameRecordReadable()
    records: list[AdaptiveBroomFrameRecord] = []
    origin_x, origin_y = spec.origin_xy_um
    width, height = spec.field_size_um
    right = origin_x + width
    frame_index = 0
    truncated = False
    brake_count = 0
    last_x: float | None = None
    last_y: float | None = None
    brake_threshold = spec.brake_interest_threshold

    def emit(record: AdaptiveBroomFrameRecord) -> Iterable[Any]:
        record_readable.update(record)
        records.append(record)
        if on_frame is not None:
            on_frame(record)
        yield from bps.create(name=stream_name)
        yield from bps.read(camera)
        for readable in readables:
            yield from bps.read(readable)
        yield from bps.read(record_readable)
        yield from bps.save()

    def move(x_um: float, y_um: float) -> Iterable[Any]:
        # Only the axes this move changes are commanded. A host stage may treat a restated
        # target as a no-op and skip it, and then a residual of one controller count on the
        # idle axis is never corrected: measured 2026-10-31 on the native ASI stage, a raster
        # whose y stayed at 2517.14 um hung for the 60 s move timeout on the nineteenth frame
        # because y read back 2517.32 um and no command would ever change it.
        nonlocal last_x, last_y
        if stage_xy is not None:
            yield from bps.mv(stage_xy, (x_um, y_um))
            last_x, last_y = x_um, y_um
            return
        y_changed = last_y is None or y_um != last_y
        if y_changed:
            yield from bps.mv(stage_x, x_um, stage_y, y_um)
            last_y = y_um
        else:
            yield from bps.mv(stage_x, x_um)
        last_x = x_um

    def checked_intensity(raw: object) -> float:
        """Validate one host-supplied intensity measure."""
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError(
                "intensity_of_frame must return a real number, not "
                f"{type(raw).__name__}"
            )
        intensity = float(raw)
        if not isfinite(intensity) or intensity < 0:
            raise ValueError("intensity_of_frame values must be finite and nonnegative")
        return intensity

    def body() -> Iterable[Any]:
        nonlocal frame_index, truncated, brake_count
        for row in range(spec.row_count):
            y_um = origin_y + row * spec.row_pitch_um
            if y_um > origin_y + height + 1e-9:
                break
            reverse = row % 2 == 1
            x_um = right if reverse else origin_x
            column = 0
            while origin_x - 1e-9 <= x_um <= right + 1e-9:
                if frame_index >= spec.max_frames:
                    truncated = True
                    return
                yield from move(x_um, y_um)
                yield from bps.trigger(camera, wait=True)
                frame_index += 1
                intensity = checked_intensity(
                    (yield from intensity_of_frame(frame_index))
                )
                interest = min(1.0, intensity / spec.intensity_scale)
                proposal = controller.update(interest)
                commanded = min(
                    spec.stage_max_speed_um_s,
                    max(
                        spec.slow_speed_um_s,
                        min(proposal.slew_limited_speed_um_s, spec.fast_speed_um_s),
                    ),
                )
                if command_speed is not None:
                    yield from command_speed(commanded)
                step_um = commanded * spec.frame_period_s
                record = AdaptiveBroomFrameRecord(
                    frame_index=frame_index,
                    row=row,
                    column=column,
                    x_um=x_um,
                    y_um=y_um,
                    intensity=intensity,
                    interest=interest,
                    blend_fraction=proposal.blend_fraction,
                    desired_speed_um_s=proposal.desired_speed_um_s,
                    commanded_speed_um_s=commanded,
                    step_um=step_um,
                )
                yield from emit(record)
                if (
                    spec.brake_dwell_frames
                    and brake_threshold is not None
                    and proposal.smoothed_interest >= brake_threshold
                ):
                    # The brake latches this point: the raster does not advance, and every
                    # held frame commands a zero step so the host's speed port cannot move it
                    # either. The policy decision is latched with it, because re-evaluating it
                    # inside the hold would let the hold shorten itself.
                    brake_count += 1
                    for dwell in range(1, spec.brake_dwell_frames + 1):
                        if frame_index >= spec.max_frames:
                            truncated = True
                            return
                        yield from bps.trigger(camera, wait=True)
                        frame_index += 1
                        held_intensity = checked_intensity(
                            (yield from intensity_of_frame(frame_index))
                        )
                        if command_speed is not None:
                            yield from command_speed(0.0)
                        yield from emit(
                            AdaptiveBroomFrameRecord(
                                frame_index=frame_index,
                                row=row,
                                column=column,
                                x_um=x_um,
                                y_um=y_um,
                                intensity=held_intensity,
                                interest=min(
                                    1.0, held_intensity / spec.intensity_scale
                                ),
                                blend_fraction=proposal.blend_fraction,
                                desired_speed_um_s=proposal.desired_speed_um_s,
                                commanded_speed_um_s=0.0,
                                step_um=0.0,
                                dwell_index=dwell,
                            )
                        )
                column += 1
                x_um += -step_um if reverse else step_um
        return

    run_wrapped = bpp.run_decorator(md=planner_md)(body)
    yield from run_wrapped()
    return AdaptiveBroomResult(
        spec=spec,
        records=tuple(records),
        rows=spec.row_count,
        truncated=truncated,
        brake_count=brake_count,
    )


__all__ = [
    "AdaptiveBroomFrameRecord",
    "AdaptiveBroomResult",
    "AdaptiveBroomSpec",
    "adaptive_broom_plan",
]
