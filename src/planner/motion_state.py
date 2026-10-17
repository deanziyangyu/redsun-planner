"""Estimate stage-axis motion from timestamp-bracketed position observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import hypot, isfinite

import numpy as np


def _as_int(value: object, label: str) -> int:
    """Read an integer field from a parsed record, accepting a numeric string."""
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"{label} must be an integer")


@dataclass(frozen=True)
class StagePositionObservation:
    """One raw XY controller readback with a host-clock uncertainty bracket."""

    event_id: str
    sequence: int
    request_id: str
    position_xy_um: tuple[float, float]
    midpoint_estimate_ns: int
    call_bracket_ns: tuple[int, int]

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> StagePositionObservation:
        """Parse one `stage_position_sample` JSONL record from LudlMotionHost."""
        if record.get("record_type") != "stage_position_sample":
            raise ValueError("record_type must be stage_position_sample")
        if (
            record.get("schema_version") != 1
            or record.get("device") != "XYStage"
            or record.get("axes") != ["x", "y"]
            or record.get("source") != "controller_readback"
            or record.get("units") != "um"
        ):
            raise ValueError("unsupported stage-position schema/device/source/units")
        timestamps = record.get("timestamps")
        position = record.get("position_um")
        if not isinstance(timestamps, Mapping) or not isinstance(position, Mapping):
            raise TypeError("sample requires timestamps and position_um mappings")
        bracket = timestamps.get("host_monotonic_bracket_ns")
        midpoint = timestamps.get("host_monotonic_midpoint_estimate_ns")
        if timestamps.get("host_clock_domain") != "time.monotonic_ns":
            raise ValueError("sample must use the host time.monotonic_ns clock domain")
        if not isinstance(bracket, list) or len(bracket) != 2:
            raise ValueError("sample requires host_monotonic_bracket_ns")
        try:
            return cls(
                event_id=str(record["event_id"]),
                sequence=_as_int(record["sequence"], "sequence"),
                request_id=str(record["request_id"]),
                position_xy_um=(float(position["x"]), float(position["y"])),
                midpoint_estimate_ns=_as_int(midpoint, "midpoint estimate"),
                call_bracket_ns=(int(bracket[0]), int(bracket[1])),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid stage-position sample record: {exc}") from exc

    def __post_init__(self) -> None:
        """Check the declared fields for consistency."""
        if not self.event_id or not self.request_id:
            raise ValueError("event_id and request_id are required")
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise ValueError("sequence must be a nonnegative integer")
        if len(self.position_xy_um) != 2 or not all(
            isfinite(float(value)) for value in self.position_xy_um
        ):
            raise ValueError("position_xy_um must contain two finite values")
        if len(self.call_bracket_ns) != 2:
            raise ValueError("call_bracket_ns must contain start and end")
        start_ns, end_ns = self.call_bracket_ns
        if start_ns > self.midpoint_estimate_ns or end_ns < self.midpoint_estimate_ns:
            raise ValueError("midpoint estimate must lie inside its call bracket")
        if end_ns < start_ns:
            raise ValueError("call bracket end precedes start")


@dataclass(frozen=True)
class StageMotionEstimate:
    """Local fit at the newest observation time, with source readback IDs.

    Velocity and acceleration standard errors are stored independently for X
    and Y. The estimate does not include cross-axis covariance.

    `observation_event_ids` reference the raw stage-position journal entries
    used by the fit. Empty IDs are permitted for legacy/manual estimates, but
    ROI-event decision records require the full source references.
    """

    request_id: str
    current_velocity_xy_um_s: tuple[float, float]
    current_acceleration_xy_um_s2: tuple[float, float]
    velocity_standard_error_xy_um_s: tuple[float, float]
    acceleration_standard_error_xy_um_s2: tuple[float, float]
    window_start_ns: int
    window_end_ns: int
    maximum_sample_age_s: float
    maximum_gap_s: float
    maximum_half_bracket_s: float
    position_fit_rms_um: float
    sample_count: int
    valid: bool
    status: str
    evaluation_time_ns: int | None = None
    observation_event_ids: tuple[str, ...] = ()


def estimate_stage_motion(
    observations: Sequence[StagePositionObservation],
    *,
    position_uncertainty_um: float,
    evaluation_time_ns: int,
    window_samples: int = 9,
    max_sample_age_s: float = 0.1,
    max_gap_s: float = 0.2,
    max_timing_uncertainty_fraction: float = 0.25,
    max_fit_rms_um: float | None = None,
) -> StageMotionEstimate:
    """Fit recent XY positions to ``p(t)=p0+v*t+0.5*a*t²``.

    The estimator reports controller-readback motion in the host monotonic
    clock domain. It does not claim encoder or exposure synchronization. A
    finite positive `position_uncertainty_um` is mandatory so the fit cannot
    silently treat quantized positions as exact. Per-axis coefficient errors
    use each axis's regression residual variance with this supplied uncertainty
    as a floor. Timing brackets, sequence, gaps and fit residual are surfaced
    and can make the estimate unusable.
    """
    if (
        isinstance(window_samples, bool)
        or not isinstance(window_samples, int)
        or window_samples < 5
    ):
        raise ValueError("window_samples must be an integer of at least 5")
    if not isfinite(position_uncertainty_um) or position_uncertainty_um <= 0:
        raise ValueError("position_uncertainty_um must be finite and positive")
    if not isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("max_gap_s must be finite and positive")
    if not isfinite(max_sample_age_s) or max_sample_age_s <= 0:
        raise ValueError("max_sample_age_s must be finite and positive")
    if isinstance(evaluation_time_ns, bool) or not isinstance(evaluation_time_ns, int):
        raise TypeError("evaluation_time_ns must be a monotonic timestamp in ns")
    if (
        not isfinite(max_timing_uncertainty_fraction)
        or max_timing_uncertainty_fraction <= 0
    ):
        raise ValueError("max_timing_uncertainty_fraction must be positive")
    if max_fit_rms_um is not None and (
        not isfinite(max_fit_rms_um) or max_fit_rms_um <= 0
    ):
        raise ValueError("max_fit_rms_um must be finite and positive")
    if len(observations) < window_samples:
        raise ValueError(f"at least {window_samples} observations are required")

    window = tuple(observations[-window_samples:])
    request_ids = {sample.request_id for sample in window}
    if len(request_ids) != 1:
        raise ValueError("all observations in a fit window must share one request_id")
    if any(
        current.sequence <= previous.sequence
        or current.midpoint_estimate_ns <= previous.midpoint_estimate_ns
        for previous, current in pairwise(window)
    ):
        raise ValueError("sample sequence and monotonic time must strictly increase")
    if evaluation_time_ns < window[-1].midpoint_estimate_ns:
        raise ValueError("evaluation time precedes the newest observation")

    times_ns = np.asarray(
        [sample.midpoint_estimate_ns for sample in window], dtype=np.float64
    )
    positions = np.asarray(
        [sample.position_xy_um for sample in window], dtype=np.float64
    )
    times_s = (times_ns - times_ns[-1]) * 1e-9
    intervals_s = np.diff(times_s)
    maximum_gap_s = float(np.max(intervals_s))
    sample_age_s = (evaluation_time_ns - window[-1].midpoint_estimate_ns) * 1e-9
    bracket_half_widths_s = np.asarray(
        [
            max(
                sample.midpoint_estimate_ns - sample.call_bracket_ns[0],
                sample.call_bracket_ns[1] - sample.midpoint_estimate_ns,
            )
            * 1e-9
            for sample in window
        ],
        dtype=np.float64,
    )
    maximum_half_bracket_s = float(np.max(bracket_half_widths_s))
    typical_interval_s = float(np.median(intervals_s))
    timing_fraction = maximum_half_bracket_s / typical_interval_s
    design = np.column_stack((np.ones(window_samples), times_s, 0.5 * times_s**2))
    coefficients, _, rank, _ = np.linalg.lstsq(design, positions, rcond=None)
    residuals = positions - design @ coefficients
    fit_rms_um = float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))
    # Estimate residual variance per axis so a noisy axis is not made
    # artificially precise by averaging its residuals with the other axis.
    # The supplied readback uncertainty remains a per-observation floor.
    residual_degrees_of_freedom = max(1, window_samples - design.shape[1])
    residual_variance_xy = np.sum(residuals**2, axis=0) / residual_degrees_of_freedom
    sigma2_xy = np.maximum(position_uncertainty_um**2, residual_variance_xy)
    coefficient_covariance_base = np.linalg.pinv(design.T @ design)
    velocity_se_xy = (
        float(np.sqrt(max(0.0, coefficient_covariance_base[1, 1] * sigma2_xy[0]))),
        float(np.sqrt(max(0.0, coefficient_covariance_base[1, 1] * sigma2_xy[1]))),
    )
    acceleration_se_xy = (
        float(np.sqrt(max(0.0, coefficient_covariance_base[2, 2] * sigma2_xy[0]))),
        float(np.sqrt(max(0.0, coefficient_covariance_base[2, 2] * sigma2_xy[1]))),
    )
    velocity = coefficients[1]
    acceleration = coefficients[2]

    deltas = np.diff(positions, axis=0)
    significant = np.linalg.norm(deltas, axis=1) > 2 * position_uncertainty_um
    significant_deltas = deltas[significant]
    has_reversal = any(
        float(np.dot(left, right)) < 0 for left, right in pairwise(significant_deltas)
    )

    status = "valid"
    if rank < 3:
        status = "ill_conditioned_time_basis"
    elif sample_age_s > max_sample_age_s:
        status = "motion_estimate_stale"
    elif maximum_gap_s > max_gap_s:
        status = "sample_gap_exceeds_limit"
    elif has_reversal:
        status = "motion_reversal_in_fit_window"
    elif timing_fraction > max_timing_uncertainty_fraction:
        status = "timing_bracket_too_wide"
    elif max_fit_rms_um is not None and fit_rms_um > max_fit_rms_um:
        status = "trajectory_fit_residual_exceeds_limit"
    elif hypot(float(velocity[0]), float(velocity[1])) <= 3 * sum(velocity_se_xy):
        status = "motion_below_fit_resolution"

    return StageMotionEstimate(
        request_id=window[-1].request_id,
        current_velocity_xy_um_s=(float(velocity[0]), float(velocity[1])),
        current_acceleration_xy_um_s2=(
            float(acceleration[0]),
            float(acceleration[1]),
        ),
        velocity_standard_error_xy_um_s=velocity_se_xy,
        acceleration_standard_error_xy_um_s2=acceleration_se_xy,
        window_start_ns=window[0].midpoint_estimate_ns,
        window_end_ns=window[-1].midpoint_estimate_ns,
        maximum_sample_age_s=sample_age_s,
        maximum_gap_s=maximum_gap_s,
        maximum_half_bracket_s=maximum_half_bracket_s,
        position_fit_rms_um=fit_rms_um,
        sample_count=window_samples,
        valid=status == "valid",
        status=status,
        evaluation_time_ns=evaluation_time_ns,
        observation_event_ids=tuple(sample.event_id for sample in window),
    )
