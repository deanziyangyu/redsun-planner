"""Replay-safe association of calibrated ROI events with stage motion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from math import isfinite

from numpy import generic as numpy_scalar

from .braking import CalibratedRoiBrakingAssessment, assess_calibrated_roi_braking
from .geometry import Calibration
from .motion_state import StageMotionEstimate


@dataclass(frozen=True)
class RoiIntensityEvent:
    """An intensity event with detector geometry and mapped host-clock time.

    `event_time_monotonic_ns` is the event time after mapping the source clock
    into host ``time.monotonic_ns``. The mapping identifier and uncertainty
    must be retained with evidence; this record does not claim hardware sync.
    """

    event_id: str
    trace_id: str
    trace_sample_index: int
    event_time_monotonic_ns: int
    time_mapping_id: str
    time_mapping_uncertainty_ns: int
    motion_request_id: str
    detector_roi_yxhw: tuple[int, int, int, int]
    signal_class: str
    threshold_policy_id: str

    def __post_init__(self) -> None:
        """Check the declared fields for consistency."""
        if not all(
            (
                self.event_id,
                self.trace_id,
                self.time_mapping_id,
                self.motion_request_id,
                self.signal_class,
                self.threshold_policy_id,
            )
        ):
            raise ValueError("event and provenance identifiers are required")
        for name, value in (
            ("trace_sample_index", self.trace_sample_index),
            ("event_time_monotonic_ns", self.event_time_monotonic_ns),
            ("time_mapping_uncertainty_ns", self.time_mapping_uncertainty_ns),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.trace_sample_index < 0 or self.event_time_monotonic_ns < 0:
            raise ValueError("sample index and event time must be nonnegative")
        if self.time_mapping_uncertainty_ns < 0:
            raise ValueError("time mapping uncertainty must be nonnegative")
        if len(self.detector_roi_yxhw) != 4 or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in self.detector_roi_yxhw
        ):
            raise ValueError("detector_roi_yxhw must contain integer y, x, h, w")
        y, x, height, width = self.detector_roi_yxhw
        if min(y, x) < 0 or min(height, width) <= 0:
            raise ValueError(
                "detector ROI requires nonnegative origin and positive size"
            )


@dataclass(frozen=True)
class RoiMotionDecisionRecord:
    """Auditable joint evidence for one ROI event and braking assessment."""

    event: RoiIntensityEvent
    calibration_id: str
    calibration_snapshot: Calibration
    sensor_shape_yx: tuple[int, int]
    detail_roi_yxhw: tuple[int, int, int, int]
    motion_fit: StageMotionEstimate
    motion_at_event: StageMotionEstimate
    motion_sample_age_at_event_s: float
    extrapolation_from_fit_s: float
    event_to_command_latency_s: float
    maximum_event_sample_age_s: float
    detail_speed_um_s: float
    deceleration_um_s2: float
    uncertainty_margin_um: float
    confidence_sigma: float
    braking: CalibratedRoiBrakingAssessment

    def to_record(self) -> dict[str, object]:
        """Return a JSON-compatible versioned record for the host recorder."""

        def normalize(value: object) -> object:
            if isinstance(value, numpy_scalar):
                return value.item()
            if isinstance(value, Mapping):
                return {str(key): normalize(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [normalize(item) for item in value]
            return value

        return normalize(
            {
                "record_type": "roi_motion_decision",
                "schema_version": 3,
                **asdict(self),
            }
        )  # type: ignore[return-value]


def assess_roi_event_transition(
    *,
    event: RoiIntensityEvent,
    motion: StageMotionEstimate,
    sensor_shape_yx: tuple[int, int],
    detail_roi_yxhw: tuple[int, int, int, int],
    calibration: Calibration,
    detail_speed_um_s: float,
    deceleration_um_s2: float,
    event_to_command_latency_s: float,
    uncertainty_margin_um: float,
    max_event_sample_age_s: float,
    confidence_sigma: float = 2.0,
) -> RoiMotionDecisionRecord:
    """Associate one ROI event with a fresh motion fit and assess braking.

    The motion fit is referenced to its newest controller readback. Its
    constant-acceleration estimate is propagated to the mapped event time.
    Clock-mapping uncertainty is conservatively added to fit uncertainty and
    the sample-age bound. `event_to_command_latency_s` starts at the event
    timestamp; target-arrival-to-event delay is a separate quantity that must
    be estimated from labeled replay. This pure calculation issues no device
    commands and is not a real-time guarantee.
    """
    if not isinstance(event, RoiIntensityEvent):
        raise TypeError("event must be a RoiIntensityEvent")
    if not isinstance(motion, StageMotionEstimate):
        raise TypeError("motion must be a StageMotionEstimate")
    if not isinstance(calibration, Calibration):
        raise TypeError("calibration must be a Calibration")
    if event.motion_request_id != motion.request_id:
        raise ValueError("ROI event and stage fit belong to different motion requests")
    if not motion.valid:
        raise ValueError(f"motion estimate is not usable: {motion.status}")
    if motion.evaluation_time_ns is None:
        raise ValueError("motion fit must retain its evaluation_time_ns")
    if (
        len(motion.observation_event_ids) != motion.sample_count
        or not motion.observation_event_ids
        or len(set(motion.observation_event_ids)) != motion.sample_count
    ):
        raise ValueError("motion fit must retain unique source readback event IDs")
    if not isfinite(max_event_sample_age_s) or max_event_sample_age_s <= 0:
        raise ValueError("max_event_sample_age_s must be finite and positive")
    if not isfinite(event_to_command_latency_s) or event_to_command_latency_s < 0:
        raise ValueError("event_to_command_latency_s must be finite and nonnegative")

    newest_ns = motion.window_end_ns
    if event.event_time_monotonic_ns < newest_ns:
        raise ValueError("ROI event precedes the stage-motion fit reference time")
    freshness_reference_ns = (
        motion.evaluation_time_ns
        if motion.evaluation_time_ns is not None
        else newest_ns
    )
    if event.event_time_monotonic_ns < freshness_reference_ns:
        raise ValueError("ROI event precedes the motion fit freshness evaluation")
    extrapolation_s = (event.event_time_monotonic_ns - newest_ns) * 1e-9
    mapping_uncertainty_s = event.time_mapping_uncertainty_ns * 1e-9
    post_evaluation_s = (
        event.event_time_monotonic_ns - freshness_reference_ns
    ) * 1e-9
    event_age_s = (
        motion.maximum_sample_age_s + post_evaluation_s + mapping_uncertainty_s
    )
    if event_age_s > max_event_sample_age_s:
        raise ValueError("stage observation is too old at the ROI event time")

    vx, vy = motion.current_velocity_xy_um_s
    ax, ay = motion.current_acceleration_xy_um_s2
    velocity_se = motion.velocity_standard_error_xy_um_s
    acceleration_se = motion.acceleration_standard_error_xy_um_s2
    propagated_velocity = (vx + ax * extrapolation_s, vy + ay * extrapolation_s)
    propagated_velocity_se = (
        velocity_se[0]
        + extrapolation_s * acceleration_se[0]
        + (abs(ax) + acceleration_se[0]) * mapping_uncertainty_s,
        velocity_se[1]
        + extrapolation_s * acceleration_se[1]
        + (abs(ay) + acceleration_se[1]) * mapping_uncertainty_s,
    )
    position_at_fit = motion.current_position_xy_um
    position_se_at_fit = motion.position_uncertainty_xy_um
    if (position_at_fit is None) != (position_se_at_fit is None):
        raise ValueError(
            "motion fit must provide both current position and its uncertainty"
        )
    position_at_event = None
    position_uncertainty_at_event = None
    if position_at_fit is not None and position_se_at_fit is not None:
        fit_bracket_s = motion.maximum_half_bracket_s
        position_at_event = (
            position_at_fit[0] + vx * extrapolation_s + 0.5 * ax * extrapolation_s**2,
            position_at_fit[1] + vy * extrapolation_s + 0.5 * ay * extrapolation_s**2,
        )
        position_uncertainty_at_event = (
            position_se_at_fit[0]
            + extrapolation_s * velocity_se[0]
            + 0.5 * extrapolation_s**2 * acceleration_se[0]
            + (
                abs(vx)
                + velocity_se[0]
                + (abs(ax) + acceleration_se[0]) * extrapolation_s
            )
            * mapping_uncertainty_s
            + (abs(vx) + velocity_se[0]) * fit_bracket_s
            + 0.5 * (abs(ax) + acceleration_se[0]) * fit_bracket_s**2,
            position_se_at_fit[1]
            + extrapolation_s * velocity_se[1]
            + 0.5 * extrapolation_s**2 * acceleration_se[1]
            + (
                abs(vy)
                + velocity_se[1]
                + (abs(ay) + acceleration_se[1]) * extrapolation_s
            )
            * mapping_uncertainty_s
            + (abs(vy) + velocity_se[1]) * fit_bracket_s
            + 0.5 * (abs(ay) + acceleration_se[1]) * fit_bracket_s**2,
        )
    motion_at_event = replace(
        motion,
        current_position_xy_um=position_at_event,
        position_uncertainty_xy_um=position_uncertainty_at_event,
        current_velocity_xy_um_s=propagated_velocity,
        velocity_standard_error_xy_um_s=propagated_velocity_se,
        maximum_sample_age_s=event_age_s,
        evaluation_time_ns=event.event_time_monotonic_ns,
    )
    braking = assess_calibrated_roi_braking(
        sensor_shape_yx=sensor_shape_yx,
        detector_roi_yxhw=event.detector_roi_yxhw,
        detail_roi_yxhw=detail_roi_yxhw,
        calibration=calibration,
        motion=motion_at_event,
        detail_speed_um_s=detail_speed_um_s,
        deceleration_um_s2=deceleration_um_s2,
        detection_latency_s=event_to_command_latency_s,
        uncertainty_margin_um=uncertainty_margin_um,
        confidence_sigma=confidence_sigma,
    )
    return RoiMotionDecisionRecord(
        event=event,
        calibration_id=calibration.calibration_id,
        calibration_snapshot=calibration,
        sensor_shape_yx=sensor_shape_yx,
        detail_roi_yxhw=detail_roi_yxhw,
        motion_fit=motion,
        motion_at_event=motion_at_event,
        motion_sample_age_at_event_s=event_age_s,
        extrapolation_from_fit_s=extrapolation_s,
        event_to_command_latency_s=event_to_command_latency_s,
        maximum_event_sample_age_s=max_event_sample_age_s,
        detail_speed_um_s=detail_speed_um_s,
        deceleration_um_s2=deceleration_um_s2,
        uncertainty_margin_um=uncertainty_margin_um,
        confidence_sigma=confidence_sigma,
        braking=braking,
    )
