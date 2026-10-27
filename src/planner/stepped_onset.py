"""Stepped onset plan: the parity path over the causal band policy.

This is the executable form of :class:`planner.onset_policy.OnsetPolicy` for a
point-move stage, and it is the stepped reference that Q04's session binding
must reproduce before any continuous-speed mode is considered. It opens no
device: a host supplies a stage, a triggerable camera and one band-density plan,
and receives one record per frame.

The plan walks a fixed-y traverse in the declared x direction. Each frame's
leading and trailing band densities go through the policy, which returns the
next coarse, dense or zero (hold) step; the plan moves that far and triggers the
next frame. The dense budget is the policy's own, so the plan cannot spend more
than the declaration allows. A hold is a real outcome (missing evidence), so it
emits a record and consumes the frame budget rather than silently advancing.

    stepped_onset_plan(spec, stage_xy=stage, camera=camera,
                       band_densities_of_frame=bands)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence  # noqa: TC003
from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal

from bluesky import plan_stubs as bps
from bluesky import preprocessors as bpp
from bluesky.utils import MsgGenerator  # noqa: TC002

from .onset_policy import OnsetObservation, OnsetPolicy, OnsetPolicyConfig

Point = tuple[float, float]
Box = tuple[Point, Point]

#: Reserved planner metadata keys; a host may not override them.
_RESERVED_METADATA = (
    "planner_schema",
    "planner_scan_mode",
    "planner_workflow",
    "planner_stepped_onset_spec",
    "planner_stepped_onset_policy_id",
    "planner_stepped_onset_max_frames",
)


def _finite(value: object, name: str) -> float:
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
    return (_finite(value[0], f"{name}[0]"), _finite(value[1], f"{name}[1]"))


@dataclass(frozen=True)
class SteppedOnsetSpec:
    """One bounded stepped traverse and the policy that drives it.

    ``start_xy_um`` and ``end_xy_um`` share a y coordinate: the policy models a
    single-direction traverse, so a diagonal start would be a different motion.
    Both points must lie inside ``stage_envelope_xy_um``, and the policy's
    direction must agree with the start-to-end sign.
    """

    start_xy_um: Point
    end_xy_um: Point
    stage_envelope_xy_um: Box
    policy: OnsetPolicyConfig
    max_frames: int = 4000

    def __post_init__(self) -> None:
        """Validate the traverse, its envelope and the policy agreement."""
        object.__setattr__(self, "start_xy_um", _point(self.start_xy_um, "start_xy_um"))
        object.__setattr__(self, "end_xy_um", _point(self.end_xy_um, "end_xy_um"))
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
        if not isinstance(self.policy, OnsetPolicyConfig):
            raise TypeError("policy must be an OnsetPolicyConfig")
        start_x, start_y = self.start_xy_um
        end_x, end_y = self.end_xy_um
        if start_y != end_y:
            raise ValueError("a stepped onset traverse must hold y constant")
        if start_x == end_x:
            raise ValueError("start and end must differ along x")
        direction: Literal["increasing_x", "decreasing_x"] = (
            "increasing_x" if end_x > start_x else "decreasing_x"
        )
        if self.policy.direction != direction:
            raise ValueError(
                f"policy direction {self.policy.direction!r} does not match a "
                f"{direction} traverse"
            )
        for name, point in (
            ("start_xy_um", self.start_xy_um),
            ("end_xy_um", self.end_xy_um),
        ):
            x, y = point
            if x < lower[0] or x > upper[0] or y < lower[1] or y > upper[1]:
                raise ValueError(f"{name} leaves the declared stage envelope")
        if (
            isinstance(self.max_frames, bool)
            or not isinstance(self.max_frames, int)
            or self.max_frames < 1
        ):
            raise ValueError("max_frames must be a positive integer")

    @property
    def direction(self) -> Literal["increasing_x", "decreasing_x"]:
        """Return the traverse direction implied by the endpoints."""
        return (
            "increasing_x"
            if self.end_xy_um[0] > self.start_xy_um[0]
            else "decreasing_x"
        )

    @property
    def span_um(self) -> float:
        """Return the along-x traverse length."""
        return abs(self.end_xy_um[0] - self.start_xy_um[0])


@dataclass(frozen=True)
class SteppedOnsetFrameRecord:
    """One frame's position and the policy decision that chose the next one."""

    frame_index: int
    x_um: float
    y_um: float
    leading_density: float | None
    trailing_density: float | None
    state: str
    reason: str
    requested_step_um: float
    dense_used_um: float
    dense_remaining_um: float
    onset: bool
    release: bool
    budget_exhausted: bool
    relative_rise: float | None
    trailing_relative_rise: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe per-frame record."""
        return {
            "frame_index": self.frame_index,
            "x_um": self.x_um,
            "y_um": self.y_um,
            "leading_density": self.leading_density,
            "trailing_density": self.trailing_density,
            "state": self.state,
            "reason": self.reason,
            "requested_step_um": self.requested_step_um,
            "dense_used_um": self.dense_used_um,
            "dense_remaining_um": self.dense_remaining_um,
            "onset": self.onset,
            "release": self.release,
            "budget_exhausted": self.budget_exhausted,
            "relative_rise": self.relative_rise,
            "trailing_relative_rise": self.trailing_relative_rise,
        }


@dataclass(frozen=True)
class SteppedOnsetResult:
    """Completed, truncated or blocked stepped traverse and its records."""

    spec: SteppedOnsetSpec
    records: tuple[SteppedOnsetFrameRecord, ...]
    truncated: bool
    stopped_reason: str

    @property
    def frame_count(self) -> int:
        """Return the number of frames the plan recorded."""
        return len(self.records)


class _FrameRecordReadable:
    """Expose the current stepped-onset record as inline Bluesky event data."""

    name = "stepped_onset_frame"
    parent = None

    _KEYS = (
        "stepped_frame_index",
        "stepped_x_um",
        "stepped_y_um",
        "stepped_leading_density",
        "stepped_trailing_density",
        "stepped_state",
        "stepped_reason",
        "stepped_requested_step_um",
        "stepped_dense_used_um",
        "stepped_dense_remaining_um",
        "stepped_onset",
        "stepped_release",
    )

    def __init__(self) -> None:
        self._record: SteppedOnsetFrameRecord | None = None

    def update(self, record: SteppedOnsetFrameRecord) -> None:
        """Set the record the next read reports."""
        self._record = record

    async def read(self) -> dict[str, dict[str, Any]]:
        """Return the current record's scalar readings."""
        record = self._record
        if record is None:
            raise RuntimeError("stepped onset frame record is not set")
        values = (
            record.frame_index,
            record.x_um,
            record.y_um,
            record.leading_density,
            record.trailing_density,
            record.state,
            record.reason,
            record.requested_step_um,
            record.dense_used_um,
            record.dense_remaining_um,
            record.onset,
            record.release,
        )
        return {
            key: {"value": value, "timestamp": 0.0}
            for key, value in zip(self._KEYS, values)
        }

    async def describe(self) -> dict[str, dict[str, Any]]:
        """Describe the fixed per-frame record keys."""
        return {
            key: {"source": "planner.stepped_onset", "dtype": "number", "shape": []}
            for key in self._KEYS
        }


def _metadata(
    md: Mapping[str, Any] | None, required: Mapping[str, Any]
) -> dict[str, Any]:
    """Merge host metadata, refusing to override reserved planner keys."""
    result = dict(md or {})
    reserved = set(_RESERVED_METADATA).intersection(result)
    if reserved:
        names = ", ".join(sorted(reserved))
        raise ValueError(f"metadata cannot override planner fields: {names}")
    result.update(required)
    return result


def _density(value: Any, name: str) -> float | None:
    """Return a band density or None, rejecting bools and negative values."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number or None")
    number = float(value)
    if not isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def stepped_onset_plan(
    spec: SteppedOnsetSpec,
    *,
    stage_xy: Any,
    camera: Any,
    band_densities_of_frame: Callable[
        [int], MsgGenerator[tuple[float | None, float | None]]
    ],
    on_frame: Callable[[SteppedOnsetFrameRecord], None] | None = None,
    readables: Sequence[Any] = (),
    md: Mapping[str, Any] | None = None,
    stream_name: str = "stepped_onset",
    trigger_camera: bool = True,
) -> MsgGenerator[SteppedOnsetResult]:
    """Run one bounded stepped traverse under the causal band policy.

    ``band_densities_of_frame(index)`` is a Bluesky plan returning the leading
    and trailing band densities for the frame just acquired. Either may be
    ``None`` when the host could not measure it; the policy then holds and the
    plan records a zero step instead of advancing. Reaching ``max_frames`` or a
    blocked-observation outcome sets ``truncated`` and a ``stopped_reason``.

    With ``trigger_camera=False`` the plan does not trigger the camera: a host
    that runs a free-running finite capture reads the frame it wants itself in
    ``band_densities_of_frame`` and is responsible for the settle before that
    read. The camera is still read into each event, so the recorded evidence is
    the same shape either way.
    """
    if not isinstance(spec, SteppedOnsetSpec):
        raise TypeError("spec must be a SteppedOnsetSpec")
    if stage_xy is None:
        raise ValueError("stage_xy is required")
    if camera is None:
        raise ValueError("camera is required")
    if not callable(band_densities_of_frame):
        raise TypeError("band_densities_of_frame must be a callable plan")
    if on_frame is not None and not callable(on_frame):
        raise TypeError("on_frame must be callable")
    if not isinstance(trigger_camera, bool):
        raise TypeError("trigger_camera must be a boolean")
    if not stream_name:
        raise ValueError("stream_name must not be empty")

    policy = OnsetPolicy(spec.policy)
    planner_md = _metadata(
        md,
        {
            "planner_schema": "redsun-planner-run-v1",
            "planner_scan_mode": "stepped-onset",
            "planner_workflow": "stepped-onset-band-policy-v1",
            "planner_stepped_onset_spec": {
                "start_xy_um": list(spec.start_xy_um),
                "end_xy_um": list(spec.end_xy_um),
                "stage_envelope_xy_um": [
                    list(spec.stage_envelope_xy_um[0]),
                    list(spec.stage_envelope_xy_um[1]),
                ],
                "direction": spec.direction,
                "policy": spec.policy.as_dict(),
                "max_frames": spec.max_frames,
            },
            "planner_stepped_onset_policy_id": spec.policy.policy_id,
            "planner_stepped_onset_max_frames": spec.max_frames,
            "planner_stepped_onset_trigger_camera": trigger_camera,
        },
    )
    record_readable = _FrameRecordReadable()
    records: list[SteppedOnsetFrameRecord] = []
    start_x, y_um = spec.start_xy_um
    end_x = spec.end_xy_um[0]
    sign = 1.0 if spec.direction == "increasing_x" else -1.0
    x_um = start_x
    frame_index = 0
    truncated = False
    stopped_reason = "completed"

    def emit(record: SteppedOnsetFrameRecord) -> Iterable[Any]:
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

    def body() -> Iterable[Any]:
        nonlocal x_um, frame_index, truncated, stopped_reason
        while True:
            if frame_index >= spec.max_frames:
                truncated = True
                stopped_reason = "max_frames"
                return
            yield from bps.mv(stage_xy, (x_um, y_um))
            if trigger_camera:
                yield from bps.trigger(camera, wait=True)
            densities = yield from band_densities_of_frame(frame_index)
            if not isinstance(densities, (tuple, list)) or len(densities) != 2:
                raise TypeError(
                    "band_densities_of_frame must return a (leading, trailing) pair"
                )
            leading = _density(densities[0], "leading density")
            trailing = _density(densities[1], "trailing density")
            proposal = policy.update(OnsetObservation(frame_index, leading, trailing))
            step = proposal.step_um or 0.0
            record = SteppedOnsetFrameRecord(
                frame_index=frame_index,
                x_um=x_um,
                y_um=y_um,
                leading_density=leading,
                trailing_density=trailing,
                state=proposal.state,
                reason=proposal.reason,
                requested_step_um=step,
                dense_used_um=proposal.dense_used_um,
                dense_remaining_um=proposal.dense_remaining_um,
                onset=proposal.onset,
                release=proposal.release,
                budget_exhausted=proposal.budget_exhausted,
                relative_rise=proposal.relative_rise,
                trailing_relative_rise=proposal.trailing_relative_rise,
            )
            yield from emit(record)
            frame_index += 1
            if proposal.reason == "blocked_missing_observations":
                stopped_reason = "blocked_missing_observations"
                return
            if step <= 0:
                # A hold re-triggers at the same x and is bounded by max_frames.
                continue
            next_x = x_um + sign * step
            if sign > 0 and next_x > end_x + 1e-9:
                stopped_reason = "completed"
                return
            if sign < 0 and next_x < end_x - 1e-9:
                stopped_reason = "completed"
                return
            x_um = next_x

    run_wrapped = bpp.run_decorator(md=planner_md)(body)
    yield from run_wrapped()
    return SteppedOnsetResult(
        spec=spec,
        records=tuple(records),
        truncated=truncated,
        stopped_reason=stopped_reason,
    )


__all__ = [
    "SteppedOnsetFrameRecord",
    "SteppedOnsetResult",
    "SteppedOnsetSpec",
    "stepped_onset_plan",
]
