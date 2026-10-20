"""Typed, feasible operating-point schedules and Bluesky orchestration."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from math import hypot, isfinite
from types import MappingProxyType
from typing import Any, Literal

from bluesky import preprocessors as bpp
from bluesky.utils import MsgGenerator

from .acquisition_space import OperatingCorner


def _finite(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not isfinite(result) or (positive and result <= 0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _json_object(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    _validate_json_value(value, name)
    try:
        result = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite JSON-compatible values") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{name} must serialize to an object")
    return result


def _validate_json_value(value: Any, name: str) -> None:
    """Reject Python containers or keys that JSON encoding would coerce."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{name} object keys must be strings")
            _validate_json_value(item, name)
    elif isinstance(value, list):
        for item in value:
            _validate_json_value(item, name)
    elif value is None or isinstance(value, (str, bool, int)):
        return
    elif isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{name} numbers must be finite")
    else:
        raise ValueError(f"{name} must contain only JSON values")


def _freeze_json(value: Any) -> Any:
    """Recursively freeze a JSON-compatible value."""
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    """Return a mutable JSON-ready copy of a recursively frozen value."""
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _require_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    """Reject incomplete and unknown fields in a serialized record."""
    actual = set(value)
    missing = expected - actual
    unknown = actual - expected
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(sorted(map(str, missing))))
        if unknown:
            details.append("unknown " + ", ".join(sorted(map(str, unknown))))
        raise ValueError(f"{name} fields are invalid: {'; '.join(details)}")


@dataclass(frozen=True, slots=True)
class OperatingPointSegment:
    """One constant-mode acquisition interval selected by the framework."""

    segment_id: str
    epoch_id: str
    corner: OperatingCorner
    velocity_um_s: float
    direction_xy: tuple[float, float]
    path_start_xy_um: tuple[float, float]
    path_end_xy_um: tuple[float, float]
    length_um: float
    acceleration_um_s2: float
    exposure_s: float
    frame_period_s: float
    detection_width_um: float
    sensor_shape_yx: tuple[int, int]
    sensor_roi_xywh: tuple[int, int, int, int]
    binning_yx: tuple[int, int]
    trigger_mode: str
    metrics_snapshot_id: str
    thresholds_used: Mapping[str, Any]
    utility_score: float
    measurement_feasible: bool
    measurement_reasons: tuple[str, ...] = ()
    slit_axis: Literal["x", "y"] | None = None
    slit_start_px: int | None = None
    slit_width_px: int | None = None
    slit_policy_id: str | None = None
    predicted_coverage: float | None = None
    predicted_cost: float | None = None
    calibration_id: str | None = None
    pose_solution_id: str | None = None

    def __post_init__(self) -> None:
        """Reject incomplete identities, malformed geometry and invalid units."""
        for name in ("segment_id", "epoch_id", "trigger_mode", "metrics_snapshot_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if not isinstance(self.corner, OperatingCorner):
            raise TypeError("corner must be an OperatingCorner")
        for name in (
            "direction_xy", "path_start_xy_um", "path_end_xy_um", "sensor_shape_yx",
            "sensor_roi_xywh", "binning_yx", "measurement_reasons",
        ):
            if not isinstance(getattr(self, name), tuple):
                raise TypeError(f"{name} must be a tuple")
        for name in (
            "velocity_um_s", "length_um", "acceleration_um_s2", "exposure_s",
            "frame_period_s", "detection_width_um",
        ):
            _finite(getattr(self, name), name, positive=True)
        if len(self.direction_xy) != 2:
            raise ValueError("direction_xy must contain two components")
        dx = _finite(self.direction_xy[0], "direction_xy.x")
        dy = _finite(self.direction_xy[1], "direction_xy.y")
        if abs(hypot(dx, dy) - 1.0) > 1e-6:
            raise ValueError("direction_xy must be a normalized unit vector")
        if len(self.path_start_xy_um) != 2 or len(self.path_end_xy_um) != 2:
            raise ValueError("path endpoints must each contain x and y")
        start_x, start_y = (
            _finite(self.path_start_xy_um[0], "path_start_xy_um.x"),
            _finite(self.path_start_xy_um[1], "path_start_xy_um.y"),
        )
        end_x, end_y = (
            _finite(self.path_end_xy_um[0], "path_end_xy_um.x"),
            _finite(self.path_end_xy_um[1], "path_end_xy_um.y"),
        )
        path_dx, path_dy = end_x - start_x, end_y - start_y
        path_length = hypot(path_dx, path_dy)
        if abs(path_length - self.length_um) > max(1e-6, self.length_um * 1e-6):
            raise ValueError("path endpoints must agree with length_um")
        if path_length <= 0 or abs(path_dx / path_length - dx) > 1e-6 or abs(
            path_dy / path_length - dy
        ) > 1e-6:
            raise ValueError("direction_xy must point from path_start to path_end")
        if len(self.sensor_shape_yx) != 2 or any(
            isinstance(v, bool) or not isinstance(v, int) or v < 1
            for v in self.sensor_shape_yx
        ):
            raise ValueError("sensor_shape_yx must contain positive integer dimensions")
        if len(self.sensor_roi_xywh) != 4 or any(
            isinstance(v, bool) or not isinstance(v, int) or v < 0
            for v in self.sensor_roi_xywh
        ) or min(self.sensor_roi_xywh[2:]) < 1:
            raise ValueError("sensor_roi_xywh must be a valid x,y,width,height")
        if len(self.binning_yx) != 2 or any(
            isinstance(v, bool) or not isinstance(v, int) or v < 1
            for v in self.binning_yx
        ):
            raise ValueError("binning_yx must contain two positive integers")
        if self.exposure_s > self.frame_period_s:
            raise ValueError("exposure_s cannot exceed frame_period_s")
        if not isinstance(self.measurement_feasible, bool):
            raise ValueError("measurement_feasible must be a boolean")
        if any(
            not isinstance(reason, str) or not reason.strip()
            for reason in self.measurement_reasons
        ):
            raise ValueError("measurement_reasons must contain nonempty strings")
        if self.measurement_feasible and self.measurement_reasons:
            raise ValueError("a feasible measurement cannot carry rejection reasons")
        _finite(self.utility_score, "utility_score")
        object.__setattr__(
            self,
            "thresholds_used",
            _freeze_json(_json_object(self.thresholds_used, "thresholds_used")),
        )
        for name in ("predicted_coverage", "predicted_cost"):
            value = getattr(self, name)
            if value is not None:
                _finite(value, name)
        for name in ("calibration_id", "pose_solution_id", "slit_policy_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a nonempty string when supplied")
        slit_fields = (
            self.slit_axis,
            self.slit_start_px,
            self.slit_width_px,
            self.slit_policy_id,
        )
        if any(value is not None for value in slit_fields):
            if self.slit_axis not in {"x", "y"}:
                raise ValueError(
                    "slit_axis must be 'x' or 'y' when slit settings are used"
                )
            if (
                isinstance(self.slit_start_px, bool)
                or not isinstance(self.slit_start_px, int)
                or self.slit_start_px < 0
            ):
                raise ValueError("slit_start_px must be a nonnegative integer")
            if (
                isinstance(self.slit_width_px, bool)
                or not isinstance(self.slit_width_px, int)
                or self.slit_width_px < 1
            ):
                raise ValueError("slit_width_px must be a positive integer")
            if self.slit_policy_id is None:
                raise ValueError("slit_policy_id is required with slit settings")
        x, y, width, height = self.sensor_roi_xywh
        sensor_height, sensor_width = self.sensor_shape_yx
        if x + width > sensor_width or y + height > sensor_height:
            raise ValueError("sensor_roi_xywh exceeds sensor_shape_yx")
        bin_y, bin_x = self.binning_yx
        if x % bin_x or y % bin_y or width % bin_x or height % bin_y:
            raise ValueError("sensor ROI origin and dimensions must align with binning")
        if self.slit_axis is not None:
            delivered_width = width // bin_x
            delivered_height = height // bin_y
            axis_limit = delivered_width if self.slit_axis == "x" else delivered_height
            if self.slit_start_px + self.slit_width_px > axis_limit:
                raise ValueError("virtual slit exceeds the delivered sensor ROI")

    @property
    def expected_observations(self) -> float:
        """Return expected samples per location, ``W / (v * T_f)``."""
        return self.detection_width_um / (self.velocity_um_s * self.frame_period_s)

    @property
    def expected_residence_s(self) -> float:
        """Return expected location residence, ``W / v``."""
        return self.detection_width_um / self.velocity_um_s

    @property
    def modeled_blur_um(self) -> float:
        """Return exposure travel, ``v * T_e``."""
        return self.velocity_um_s * self.exposure_s

    @property
    def camera_mode(
        self,
    ) -> tuple[tuple[int, int], tuple[int, int, int, int], float, str]:
        """Return the camera settings whose changes require a new epoch."""
        return (
            self.binning_yx,
            self.sensor_roi_xywh,
            self.exposure_s,
            self.trigger_mode,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the request plus its derived dwell and blur values."""
        return {
            "segment_id": self.segment_id,
            "epoch_id": self.epoch_id,
            "corner": self.corner.value,
            "velocity_um_s": self.velocity_um_s,
            "direction_xy": list(self.direction_xy),
            "path_start_xy_um": list(self.path_start_xy_um),
            "path_end_xy_um": list(self.path_end_xy_um),
            "length_um": self.length_um,
            "acceleration_um_s2": self.acceleration_um_s2,
            "exposure_s": self.exposure_s,
            "frame_period_s": self.frame_period_s,
            "detection_width_um": self.detection_width_um,
            "sensor_shape_yx": list(self.sensor_shape_yx),
            "sensor_roi_xywh": list(self.sensor_roi_xywh),
            "binning_yx": list(self.binning_yx),
            "trigger_mode": self.trigger_mode,
            "metrics_snapshot_id": self.metrics_snapshot_id,
            "thresholds_used": _thaw_json(self.thresholds_used),
            "utility_score": self.utility_score,
            "measurement_feasible": self.measurement_feasible,
            "measurement_reasons": list(self.measurement_reasons),
            "slit_axis": self.slit_axis,
            "slit_start_px": self.slit_start_px,
            "slit_width_px": self.slit_width_px,
            "slit_policy_id": self.slit_policy_id,
            "predicted_coverage": self.predicted_coverage,
            "predicted_cost": self.predicted_cost,
            "calibration_id": self.calibration_id,
            "pose_solution_id": self.pose_solution_id,
            "expected_n_obs": self.expected_observations,
            "expected_t_obs_s": self.expected_residence_s,
            "modeled_blur_um": self.modeled_blur_um,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OperatingPointSegment:
        """Parse a complete serialized segment and verify its derived metrics."""
        if not isinstance(value, Mapping):
            raise TypeError("segment value must be a mapping")
        expected = {
            "segment_id",
            "epoch_id",
            "corner",
            "velocity_um_s",
            "direction_xy",
            "path_start_xy_um",
            "path_end_xy_um",
            "length_um",
            "acceleration_um_s2",
            "exposure_s",
            "frame_period_s",
            "detection_width_um",
            "sensor_shape_yx",
            "sensor_roi_xywh",
            "binning_yx",
            "trigger_mode",
            "metrics_snapshot_id",
            "thresholds_used",
            "utility_score",
            "measurement_feasible",
            "measurement_reasons",
            "slit_axis",
            "slit_start_px",
            "slit_width_px",
            "slit_policy_id",
            "predicted_coverage",
            "predicted_cost",
            "calibration_id",
            "pose_solution_id",
            "expected_n_obs",
            "expected_t_obs_s",
            "modeled_blur_um",
        }
        _require_keys(value, expected, "operating-point segment")
        fields = dict(value)
        for name in (
            "direction_xy",
            "path_start_xy_um",
            "path_end_xy_um",
            "sensor_shape_yx",
            "sensor_roi_xywh",
            "binning_yx",
            "measurement_reasons",
        ):
            if not isinstance(fields[name], (list, tuple)):
                raise TypeError(f"{name} must be a JSON array")
            fields[name] = tuple(fields[name])
        fields["corner"] = OperatingCorner(fields["corner"])
        fields.pop("expected_n_obs")
        fields.pop("expected_t_obs_s")
        fields.pop("modeled_blur_um")
        segment = cls(**fields)
        for key, actual in (
            ("expected_n_obs", segment.expected_observations),
            ("expected_t_obs_s", segment.expected_residence_s),
            ("modeled_blur_um", segment.modeled_blur_um),
        ):
            recorded = _finite(value[key], key)
            if abs(recorded - actual) > max(1e-9, abs(actual) * 1e-9):
                raise ValueError(f"serialized {key} does not match segment inputs")
        return segment


@dataclass(frozen=True, slots=True)
class OperatingPointRequest:
    """Ordered operating-point decision to lower into a Bluesky run."""

    run_id: str
    recipe_id: str
    recipe_version: str
    software_version: str
    coordinate_frame: str
    decision_timestamp_utc: str
    segments: tuple[OperatingPointSegment, ...]

    def __post_init__(self) -> None:
        """Require stable run provenance and uniquely identified segments."""
        for name in (
            "run_id", "recipe_id", "recipe_version", "software_version",
            "coordinate_frame", "decision_timestamp_utc",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if not isinstance(self.segments, tuple) or any(
            not isinstance(segment, OperatingPointSegment) for segment in self.segments
        ):
            raise TypeError("segments must be a tuple of OperatingPointSegment records")
        try:
            timestamp = datetime.fromisoformat(
                self.decision_timestamp_utc.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ValueError(
                "decision_timestamp_utc must be an ISO-8601 timestamp"
            ) from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() != timezone.utc.utcoffset(
            timestamp
        ):
            raise ValueError("decision_timestamp_utc must include a UTC offset")
        if not self.segments:
            raise ValueError("operating-point request requires at least one segment")
        ids = [segment.segment_id for segment in self.segments]
        if len(ids) != len(set(ids)):
            raise ValueError("segment_id values must be unique within a request")
        seen_epochs: dict[
            str, tuple[tuple[int, int], tuple[int, int, int, int], float, str]
        ] = {}
        closed_epochs: set[str] = set()
        for index, current in enumerate(self.segments):
            mode = current.camera_mode
            if current.epoch_id in seen_epochs:
                if seen_epochs[current.epoch_id] != mode:
                    raise ValueError(
                        "an epoch_id cannot be reused for a different camera mode"
                    )
                if current.epoch_id in closed_epochs:
                    raise ValueError("an epoch_id cannot be reused after it has ended")
            if index:
                previous = self.segments[index - 1]
                if previous.epoch_id != current.epoch_id:
                    closed_epochs.add(previous.epoch_id)
                elif previous.camera_mode != mode:
                    raise ValueError(
                        "camera ROI, binning, exposure or trigger changes require "
                        "a new epoch"
                    )
            seen_epochs[current.epoch_id] = mode

    def to_dict(self) -> dict[str, Any]:
        """Return the ordered request as JSON-ready data."""
        return {
            "schema": "redsun-planner-operating-point-request-v1",
            "run_id": self.run_id,
            "recipe_id": self.recipe_id,
            "recipe_version": self.recipe_version,
            "software_version": self.software_version,
            "coordinate_frame": self.coordinate_frame,
            "decision_timestamp_utc": self.decision_timestamp_utc,
            "segments": [segment.to_dict() for segment in self.segments],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OperatingPointRequest:
        """Parse a versioned request from JSON-compatible mapping data."""
        if not isinstance(value, Mapping):
            raise TypeError("request value must be a mapping")
        expected = {
            "schema",
            "run_id",
            "recipe_id",
            "recipe_version",
            "software_version",
            "coordinate_frame",
            "decision_timestamp_utc",
            "segments",
        }
        _require_keys(value, expected, "operating-point request")
        if value["schema"] != "redsun-planner-operating-point-request-v1":
            raise ValueError("unsupported operating-point request schema")
        segments = value["segments"]
        if not isinstance(segments, (list, tuple)):
            raise TypeError("segments must be a JSON array")
        return cls(
            run_id=value["run_id"],
            recipe_id=value["recipe_id"],
            recipe_version=value["recipe_version"],
            software_version=value["software_version"],
            coordinate_frame=value["coordinate_frame"],
            decision_timestamp_utc=value["decision_timestamp_utc"],
            segments=tuple(OperatingPointSegment.from_dict(item) for item in segments),
        )


@dataclass(frozen=True, slots=True)
class OperatingPointCapabilities:
    """Host-declared physical limits required before plan generation."""

    sensor_shape_yx: tuple[int, int]
    max_velocity_um_s: float
    max_acceleration_um_s2: float
    min_exposure_s: float
    max_exposure_s: float
    min_frame_period_s: float
    supported_trigger_modes: tuple[str, ...]
    stage_bounds_xy_um: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        """Validate the declared capability snapshot."""
        for name in (
            "sensor_shape_yx",
            "supported_trigger_modes",
            "stage_bounds_xy_um",
        ):
            if not isinstance(getattr(self, name), tuple):
                raise TypeError(f"{name} must be a tuple")
        if len(self.sensor_shape_yx) != 2 or any(
            isinstance(v, bool) or not isinstance(v, int) or v < 1
            for v in self.sensor_shape_yx
        ):
            raise ValueError("sensor_shape_yx must contain positive integer dimensions")
        for name in (
            "max_velocity_um_s", "max_acceleration_um_s2", "min_exposure_s",
            "max_exposure_s", "min_frame_period_s",
        ):
            _finite(getattr(self, name), name, positive=True)
        if self.min_exposure_s > self.max_exposure_s:
            raise ValueError("min_exposure_s cannot exceed max_exposure_s")
        if len(self.stage_bounds_xy_um) != 4:
            raise ValueError("stage_bounds_xy_um must be xmin,xmax,ymin,ymax")
        xmin, xmax, ymin, ymax = (
            _finite(value, "stage_bounds_xy_um") for value in self.stage_bounds_xy_um
        )
        if xmin >= xmax or ymin >= ymax:
            raise ValueError("stage bounds must have positive width and height")
        if not self.supported_trigger_modes or any(
            not isinstance(mode, str) or not mode.strip()
            for mode in self.supported_trigger_modes
        ):
            raise ValueError("supported_trigger_modes must be nonempty strings")
        if len(set(self.supported_trigger_modes)) != len(self.supported_trigger_modes):
            raise ValueError("supported_trigger_modes must be unique")

    def to_dict(self) -> dict[str, Any]:
        """Return the immutable host capability snapshot as JSON-ready data."""
        return {
            "schema": "redsun-planner-operating-point-capabilities-v1",
            "sensor_shape_yx": list(self.sensor_shape_yx),
            "max_velocity_um_s": self.max_velocity_um_s,
            "max_acceleration_um_s2": self.max_acceleration_um_s2,
            "min_exposure_s": self.min_exposure_s,
            "max_exposure_s": self.max_exposure_s,
            "min_frame_period_s": self.min_frame_period_s,
            "supported_trigger_modes": list(self.supported_trigger_modes),
            "stage_bounds_xy_um": list(self.stage_bounds_xy_um),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OperatingPointCapabilities:
        """Parse a versioned host capability snapshot."""
        if not isinstance(value, Mapping):
            raise TypeError("capability value must be a mapping")
        expected = {
            "schema",
            "sensor_shape_yx",
            "max_velocity_um_s",
            "max_acceleration_um_s2",
            "min_exposure_s",
            "max_exposure_s",
            "min_frame_period_s",
            "supported_trigger_modes",
            "stage_bounds_xy_um",
        }
        _require_keys(value, expected, "operating-point capabilities")
        if value["schema"] != "redsun-planner-operating-point-capabilities-v1":
            raise ValueError("unsupported operating-point capabilities schema")
        for name in (
            "sensor_shape_yx", "supported_trigger_modes", "stage_bounds_xy_um"
        ):
            if not isinstance(value[name], (list, tuple)):
                raise TypeError(f"{name} must be a JSON array")
        return cls(
            sensor_shape_yx=tuple(value["sensor_shape_yx"]),
            max_velocity_um_s=value["max_velocity_um_s"],
            max_acceleration_um_s2=value["max_acceleration_um_s2"],
            min_exposure_s=value["min_exposure_s"],
            max_exposure_s=value["max_exposure_s"],
            min_frame_period_s=value["min_frame_period_s"],
            supported_trigger_modes=tuple(value["supported_trigger_modes"]),
            stage_bounds_xy_um=tuple(value["stage_bounds_xy_um"]),
        )


@dataclass(frozen=True, slots=True)
class OperatingPointDecision:
    """Physical and measurement feasibility result for one request."""

    accepted: bool
    request_id: str
    segment_reasons: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        """Validate and freeze the per-segment feasibility result."""
        if not isinstance(self.accepted, bool):
            raise ValueError("accepted must be a boolean")
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id must be a nonempty string")
        if not isinstance(self.segment_reasons, Mapping):
            raise TypeError("segment_reasons must be a mapping")
        reasons = dict(self.segment_reasons)
        for segment_id, items in reasons.items():
            if not isinstance(segment_id, str) or not segment_id.strip():
                raise ValueError("segment_reasons keys must be nonempty strings")
            if not isinstance(items, tuple) or any(
                not isinstance(item, str) or not item.strip() for item in items
            ):
                raise ValueError("segment reasons must be tuples of nonempty strings")
        if self.accepted and any(reasons.values()):
            raise ValueError("accepted decision cannot contain rejection reasons")
        if not self.accepted and not any(reasons.values()):
            raise ValueError("rejected decision must contain at least one reason")
        object.__setattr__(self, "segment_reasons", MappingProxyType(reasons))

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic feasibility decision."""
        return {
            "schema": "redsun-planner-operating-point-decision-v1",
            "accepted": self.accepted,
            "request_id": self.request_id,
            "segment_reasons": {
                key: list(value) for key, value in self.segment_reasons.items()
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OperatingPointDecision:
        """Parse a versioned feasibility decision with strict reason lists."""
        if not isinstance(value, Mapping):
            raise TypeError("decision value must be a mapping")
        expected = {"schema", "accepted", "request_id", "segment_reasons"}
        _require_keys(value, expected, "operating-point decision")
        if value["schema"] != "redsun-planner-operating-point-decision-v1":
            raise ValueError("unsupported operating-point decision schema")
        raw_reasons = value["segment_reasons"]
        if not isinstance(raw_reasons, Mapping):
            raise TypeError("segment_reasons must be a mapping")
        reasons: dict[str, tuple[str, ...]] = {}
        for segment_id, items in raw_reasons.items():
            if not isinstance(items, (list, tuple)):
                raise TypeError("each segment reason collection must be a JSON array")
            reasons[segment_id] = tuple(items)
        return cls(
            accepted=value["accepted"],
            request_id=value["request_id"],
            segment_reasons=reasons,
        )


def assess_operating_point_request(
    request: OperatingPointRequest,
    capabilities: OperatingPointCapabilities,
) -> OperatingPointDecision:
    """Check measurement verdicts and physical limits without commanding devices."""
    if not isinstance(request, OperatingPointRequest):
        raise TypeError("request must be an OperatingPointRequest")
    if not isinstance(capabilities, OperatingPointCapabilities):
        raise TypeError("capabilities must be OperatingPointCapabilities")
    reasons_by_segment: dict[str, tuple[str, ...]] = {}
    previous: OperatingPointSegment | None = None
    for segment in request.segments:
        reasons = list(segment.measurement_reasons)
        if not segment.measurement_feasible and not reasons:
            reasons.append("measurement model rejected this operating point")
        if segment.sensor_shape_yx != capabilities.sensor_shape_yx:
            reasons.append("sensor shape differs from the capability snapshot")
        if segment.velocity_um_s > capabilities.max_velocity_um_s:
            reasons.append("requested velocity exceeds the declared stage limit")
        if segment.acceleration_um_s2 > capabilities.max_acceleration_um_s2:
            reasons.append("requested acceleration exceeds the declared stage limit")
        if not (
            capabilities.min_exposure_s
            <= segment.exposure_s
            <= capabilities.max_exposure_s
        ):
            reasons.append("exposure is outside the declared camera range")
        if segment.frame_period_s < capabilities.min_frame_period_s:
            reasons.append("frame period is faster than the declared camera limit")
        if segment.trigger_mode not in capabilities.supported_trigger_modes:
            reasons.append("trigger mode is not supported by the declared camera")
        xmin, xmax, ymin, ymax = capabilities.stage_bounds_xy_um
        for label, (x, y) in (
            ("start", segment.path_start_xy_um),
            ("end", segment.path_end_xy_um),
        ):
            if not xmin <= x <= xmax or not ymin <= y <= ymax:
                reasons.append(f"path {label} is outside the declared stage bounds")
        if (
            previous is not None
            and previous.camera_mode != segment.camera_mode
            and previous.epoch_id == segment.epoch_id
        ):
            reasons.append("camera mode changed without opening a new epoch")
        reasons_by_segment[segment.segment_id] = tuple(dict.fromkeys(reasons))
        previous = segment
    accepted = not any(reasons_by_segment.values())
    return OperatingPointDecision(accepted, request.run_id, reasons_by_segment)


def operating_point_schedule_plan(
    request: OperatingPointRequest,
    capabilities: OperatingPointCapabilities,
    *,
    configure_segment: Callable[[OperatingPointSegment], Iterable[Any]],
    acquire_segment: Callable[[OperatingPointSegment], Iterable[Any]],
    md: Mapping[str, Any] | None = None,
) -> MsgGenerator[Any]:
    """Lower an accepted schedule into one run with ordered host plan callbacks.

    The host supplies Bluesky plans for camera/stage configuration and segment
    acquisition. Settings are applied once at segment boundaries; this function
    neither creates devices nor issues commands. It serializes the exact request
    and both feasibility and measurement decisions in run metadata.
    """
    if not callable(configure_segment) or not callable(acquire_segment):
        raise TypeError("configure_segment and acquire_segment must be callable")
    decision = assess_operating_point_request(request, capabilities)
    if not decision.accepted:
        raise ValueError(f"operating-point request rejected: {decision.to_dict()}")
    required = {
        "planner_schema": "redsun-planner-run-v1",
        "planner_workflow": "operating-point-schedule-v1",
        "planner_request": request.to_dict(),
        "planner_decision": decision.to_dict(),
        "planner_capabilities": capabilities.to_dict(),
        "planner_host_callbacks_required": True,
    }
    metadata = dict(md or {})
    conflicts = set(required).intersection(metadata)
    if conflicts:
        raise ValueError(
            "metadata cannot override planner fields: " + ", ".join(sorted(conflicts))
        )
    metadata.update(required)
    try:
        json.dumps(metadata, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("run metadata must be finite JSON-compatible data") from exc

    def body() -> Iterable[Any]:
        for segment in request.segments:
            yield from configure_segment(segment)
            yield from acquire_segment(segment)

    yield from bpp.run_decorator(md=metadata)(body())


__all__ = [
    "OperatingPointCapabilities",
    "OperatingPointDecision",
    "OperatingPointRequest",
    "OperatingPointSegment",
    "assess_operating_point_request",
    "operating_point_schedule_plan",
]
