"""Estimate stage-axis motion from timestamp-bracketed position observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import hypot, isfinite
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class ConstantVelocityMotionModel:
    """Declared linear stage motion over one explicit clock interval.

    The anchor is the stage position at ``start_time_s`` and velocity is in
    the same coordinate frame. This is a commanded/modelled trajectory, not a
    measured readback. Callers must supply a matching clock domain and
    error-bound inputs before using it to associate exposures.

    Parameters
    ----------
    motion_request_id
        Identifier for the command that declared this segment.
    anchor_id, calibration_id
        Durable identifiers for the start-position observation and XY scale
        calibration used to express the model in micrometres.
    clock_id
        Clock domain shared by the motion interval and associated exposures.
    coordinate_frame
        Coordinate-frame name for the anchor and velocity vector.
    start_time_s, end_time_s
        Closed interval in seconds over which constant velocity is declared.
    anchor_xy_um, velocity_xy_um_s
        XY anchor at the segment start and declared XY velocity.
    anchor_error_bound_xy_um, velocity_error_bound_xy_um_s
        Per-axis nonnegative absolute error bounds. The anchor bound includes
        timing, readback and calibration error at the segment start.
    """

    motion_request_id: str
    anchor_id: str
    calibration_id: str
    clock_id: str
    coordinate_frame: str
    start_time_s: float
    end_time_s: float
    anchor_xy_um: tuple[float, float]
    velocity_xy_um_s: tuple[float, float]
    anchor_error_bound_xy_um: tuple[float, float]
    velocity_error_bound_xy_um_s: tuple[float, float]

    def __post_init__(self) -> None:
        """Validate identity, interval and declared model parameters."""
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (
                self.motion_request_id,
                self.anchor_id,
                self.calibration_id,
                self.clock_id,
                self.coordinate_frame,
            )
        ):
            raise ValueError(
                "motion_request_id, anchor_id, calibration_id, clock_id and "
                "coordinate_frame are required"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            for value in (self.start_time_s, self.end_time_s)
        ):
            raise ValueError("motion interval times must be finite")
        if self.end_time_s <= self.start_time_s:
            raise ValueError("motion end_time_s must follow start_time_s")
        for name in (
            "anchor_xy_um",
            "velocity_xy_um_s",
            "anchor_error_bound_xy_um",
            "velocity_error_bound_xy_um_s",
        ):
            values = getattr(self, name)
            if not isinstance(values, tuple) or len(values) != 2:
                raise ValueError(f"{name} must contain an X/Y pair")
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                for value in values
            ):
                raise ValueError(f"{name} must contain finite numeric values")
        if any(value < 0 for value in self.anchor_error_bound_xy_um):
            raise ValueError("anchor error bounds must be nonnegative")
        if any(value < 0 for value in self.velocity_error_bound_xy_um_s):
            raise ValueError("velocity error bounds must be nonnegative")


@dataclass(frozen=True)
class ModeledStagePosition:
    """Predicted position with a conservative bound and full provenance."""

    position_xy_um: tuple[float, float]
    position_error_bound_xy_um: tuple[float, float]
    time_s: float
    clock_id: str
    coordinate_frame: str
    motion_request_id: str
    anchor_id: str
    calibration_id: str
    source: Literal["commanded_constant_velocity_model"] = (
        "commanded_constant_velocity_model"
    )


def model_exposure_position(
    model: ConstantVelocityMotionModel,
    *,
    exposure_start_s: float,
    exposure_end_s: float,
    clock_id: str,
    clock_mapping_uncertainty_s: float,
) -> ModeledStagePosition:
    """Predict stage position at an exposure midpoint from a declared model.

    Parameters
    ----------
    model
        Declared constant-velocity segment and its anchor uncertainties.
    exposure_start_s, exposure_end_s
        Exposure interval in seconds in ``clock_id``.
    clock_id
        Clock-domain identifier shared by the exposure and motion model.
    clock_mapping_uncertainty_s
        Nonnegative timing uncertainty in seconds for the exposure-to-model
        clock association.

    The exposure midpoint must lie inside the model interval and use the same
    clock identifier. The per-axis position-error bound sums the absolute
    anchor, velocity and clock-mapping contributions, including their
    first-order cross term. This avoids assuming the uncertainty sources are
    statistically independent.

    Notes
    -----
    The result remains model-derived even if its inputs are based on a stage
    readback. It is not a substitute for an exposure-synchronized pose sample.
    """
    if not isinstance(model, ConstantVelocityMotionModel):
        raise TypeError("model must be a ConstantVelocityMotionModel")
    if not isinstance(clock_id, str) or clock_id != model.clock_id:
        raise ValueError("exposure and motion model must use the same clock_id")
    if (
        isinstance(exposure_start_s, bool)
        or isinstance(exposure_end_s, bool)
        or not isinstance(exposure_start_s, (int, float))
        or not isinstance(exposure_end_s, (int, float))
        or not isfinite(exposure_start_s)
        or not isfinite(exposure_end_s)
        or exposure_end_s <= exposure_start_s
    ):
        raise ValueError("exposure interval must be finite and ordered")
    midpoint = (exposure_start_s + exposure_end_s) / 2.0
    if not model.start_time_s <= midpoint <= model.end_time_s:
        raise ValueError("exposure midpoint must lie inside the motion model interval")
    if (
        isinstance(clock_mapping_uncertainty_s, bool)
        or not isinstance(clock_mapping_uncertainty_s, (int, float))
        or not isfinite(clock_mapping_uncertainty_s)
        or clock_mapping_uncertainty_s < 0
    ):
        raise ValueError("clock_mapping_uncertainty_s must be finite and nonnegative")
    elapsed = midpoint - model.start_time_s
    positions = tuple(
        anchor + velocity * elapsed
        for anchor, velocity in zip(model.anchor_xy_um, model.velocity_xy_um_s)
    )
    uncertainties = tuple(
        float(
            anchor_error_bound
            + elapsed * velocity_error_bound
            + abs(velocity) * clock_mapping_uncertainty_s
            + velocity_error_bound * clock_mapping_uncertainty_s
        )
        for anchor_error_bound, velocity_error_bound, velocity in zip(
            model.anchor_error_bound_xy_um,
            model.velocity_error_bound_xy_um_s,
            model.velocity_xy_um_s,
        )
    )
    return ModeledStagePosition(
        position_xy_um=(float(positions[0]), float(positions[1])),
        position_error_bound_xy_um=(
            float(uncertainties[0]),
            float(uncertainties[1]),
        ),
        time_s=midpoint,
        clock_id=model.clock_id,
        coordinate_frame=model.coordinate_frame,
        motion_request_id=model.motion_request_id,
        anchor_id=model.anchor_id,
        calibration_id=model.calibration_id,
    )


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
        if (
            not isinstance(self.event_id, str)
            or not self.event_id.strip()
            or not isinstance(self.request_id, str)
            or not self.request_id.strip()
        ):
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
class StageAxisPositionObservation:
    """One controller axis read with its own timestamp and uncertainty bracket."""

    event_id: str
    sequence: int
    request_id: str
    axis: Literal["x", "y"]
    position_um: float
    midpoint_estimate_ns: int
    call_bracket_ns: tuple[int, int]

    def __post_init__(self) -> None:
        """Validate identity, coordinate, and clock fields."""
        if not self.event_id or not self.request_id:
            raise ValueError("event_id and request_id are required")
        if self.axis not in ("x", "y"):
            raise ValueError("axis must be 'x' or 'y'")
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise ValueError("sequence must be a nonnegative integer")
        if isinstance(self.position_um, bool) or not isfinite(self.position_um):
            raise ValueError("position_um must be finite")
        if isinstance(self.midpoint_estimate_ns, bool) or not isinstance(
            self.midpoint_estimate_ns, int
        ):
            raise ValueError("midpoint_estimate_ns must be an integer")
        if self.midpoint_estimate_ns < 0:
            raise ValueError("midpoint_estimate_ns must be nonnegative")
        if (
            not isinstance(self.call_bracket_ns, tuple)
            or len(self.call_bracket_ns) != 2
        ):
            raise ValueError("call_bracket_ns must contain start and end")
        start_ns, end_ns = self.call_bracket_ns
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in self.call_bracket_ns
        ):
            raise ValueError("call_bracket_ns values must be integers")
        if end_ns < start_ns or not start_ns <= self.midpoint_estimate_ns <= end_ns:
            raise ValueError(
                "midpoint estimate must lie inside an ordered call bracket"
            )


@dataclass(frozen=True)
class StageMotionEstimate:
    """Local fit at the newest observation time, with source readback IDs.

    Velocity and acceleration standard errors are stored independently for X
    and Y. The estimate does not include cross-axis covariance.

    ``current_position_xy_um`` and its per-axis fit uncertainty refer to the
    fit's ``window_end_ns`` reference time. They are absent only for manually
    built legacy estimates that do not carry a position fit.

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
    current_position_xy_um: tuple[float, float] | None = None
    position_uncertainty_xy_um: tuple[float, float] | None = None


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
    position_se_xy = (
        float(np.sqrt(max(0.0, coefficient_covariance_base[0, 0] * sigma2_xy[0]))),
        float(np.sqrt(max(0.0, coefficient_covariance_base[0, 0] * sigma2_xy[1]))),
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
        current_position_xy_um=(float(coefficients[0, 0]), float(coefficients[0, 1])),
        position_uncertainty_xy_um=position_se_xy,
    )


def estimate_stage_motion_from_axes(
    observations: Sequence[StageAxisPositionObservation],
    *,
    position_uncertainty_um: float,
    evaluation_time_ns: int,
    window_samples_per_axis: int = 9,
    max_sample_age_s: float = 0.1,
    max_gap_s: float = 0.2,
    max_timing_uncertainty_fraction: float = 0.25,
    max_fit_rms_um: float | None = None,
) -> StageMotionEstimate:
    """Fit independent X/Y read streams at one explicit monotonic time.

    Separate axis reads are never paired into synthetic XY observations. Each
    axis is fit independently at the latest timestamp observed on either axis;
    the older stream is extrapolated only to that common fit time. Freshness is
    evaluated at `evaluation_time_ns`, which must not precede the common fit
    time. Both axes need recent, well-bracketed observations from one request.
    """
    if (
        isinstance(window_samples_per_axis, bool)
        or not isinstance(window_samples_per_axis, int)
        or window_samples_per_axis < 5
    ):
        raise ValueError("window_samples_per_axis must be an integer of at least 5")
    if not isfinite(position_uncertainty_um) or position_uncertainty_um <= 0:
        raise ValueError("position_uncertainty_um must be finite and positive")
    if not isfinite(max_sample_age_s) or max_sample_age_s <= 0:
        raise ValueError("max_sample_age_s must be finite and positive")
    if not isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("max_gap_s must be finite and positive")
    if (
        not isfinite(max_timing_uncertainty_fraction)
        or max_timing_uncertainty_fraction <= 0
    ):
        raise ValueError("max_timing_uncertainty_fraction must be finite and positive")
    if max_fit_rms_um is not None and (
        not isfinite(max_fit_rms_um) or max_fit_rms_um <= 0
    ):
        raise ValueError("max_fit_rms_um must be finite and positive")
    if isinstance(evaluation_time_ns, bool) or not isinstance(evaluation_time_ns, int):
        raise TypeError("evaluation_time_ns must be an integer monotonic timestamp")

    request_ids = {sample.request_id for sample in observations}
    if len(request_ids) != 1:
        raise ValueError("all axis observations must share one motion request_id")
    selected: dict[str, tuple[StageAxisPositionObservation, ...]] = {}
    for axis in ("x", "y"):
        stream = tuple(sample for sample in observations if sample.axis == axis)
        if len(stream) < window_samples_per_axis:
            raise ValueError(
                f"at least {window_samples_per_axis} {axis.upper()} observations are required"
            )
        window = stream[-window_samples_per_axis:]
        if any(
            current.sequence <= previous.sequence
            or current.midpoint_estimate_ns <= previous.midpoint_estimate_ns
            for previous, current in pairwise(window)
        ):
            raise ValueError(
                f"{axis.upper()} sample sequence and time must strictly increase"
            )
        if evaluation_time_ns < window[-1].midpoint_estimate_ns:
            raise ValueError("evaluation time precedes a latest axis observation")
        selected[axis] = window

    common_fit_time_ns = max(
        window[-1].midpoint_estimate_ns for window in selected.values()
    )
    if evaluation_time_ns < common_fit_time_ns:
        raise ValueError("evaluation time precedes the common X/Y fit time")

    velocities: list[float] = []
    accelerations: list[float] = []
    velocity_errors: list[float] = []
    acceleration_errors: list[float] = []
    positions_at_fit: list[float] = []
    position_errors: list[float] = []
    ages: list[float] = []
    gaps: list[float] = []
    half_brackets: list[float] = []
    rms_values: list[float] = []
    statuses: list[str] = []
    source_ids: list[str] = []
    starts: list[int] = []
    for axis in ("x", "y"):
        window = selected[axis]
        offsets_ns = np.asarray(
            [sample.midpoint_estimate_ns - common_fit_time_ns for sample in window],
            dtype=np.float64,
        )
        positions = np.asarray(
            [sample.position_um for sample in window], dtype=np.float64
        )
        times_s = offsets_ns * 1e-9
        intervals = (
            np.asarray(
                [
                    current.midpoint_estimate_ns - previous.midpoint_estimate_ns
                    for previous, current in pairwise(window)
                ],
                dtype=np.float64,
            )
            * 1e-9
        )
        age_s = (evaluation_time_ns - window[-1].midpoint_estimate_ns) * 1e-9
        half_widths = np.asarray(
            [
                max(
                    sample.midpoint_estimate_ns - sample.call_bracket_ns[0],
                    sample.call_bracket_ns[1] - sample.midpoint_estimate_ns,
                )
                * 1e-9
                for sample in window
            ]
        )
        typical_interval = float(np.median(intervals))
        timing_fraction = float(np.max(half_widths)) / typical_interval
        design = np.column_stack(
            (np.ones(window_samples_per_axis), times_s, 0.5 * times_s**2)
        )
        coefficients, _, rank, _ = np.linalg.lstsq(design, positions, rcond=None)
        residuals = positions - design @ coefficients
        rms = float(np.sqrt(np.mean(residuals**2)))
        dof = max(1, window_samples_per_axis - design.shape[1])
        variance = max(position_uncertainty_um**2, float(np.sum(residuals**2) / dof))
        covariance = np.linalg.pinv(design.T @ design)
        velocities.append(float(coefficients[1]))
        accelerations.append(float(coefficients[2]))
        positions_at_fit.append(float(coefficients[0]))
        position_errors.append(float(np.sqrt(max(0.0, covariance[0, 0] * variance))))
        velocity_errors.append(float(np.sqrt(max(0.0, covariance[1, 1] * variance))))
        acceleration_errors.append(
            float(np.sqrt(max(0.0, covariance[2, 2] * variance)))
        )
        ages.append(age_s)
        gaps.append(float(np.max(intervals)))
        half_brackets.append(float(np.max(half_widths)))
        rms_values.append(rms)
        starts.append(window[0].midpoint_estimate_ns)
        source_ids.extend(sample.event_id for sample in window)
        deltas = np.diff(positions)
        significant = np.abs(deltas) > 2 * position_uncertainty_um
        significant_deltas = deltas[significant]
        if any(left * right < 0 for left, right in pairwise(significant_deltas)):
            statuses.append(f"{axis}_motion_reversal_in_fit_window")
        if rank < 3:
            statuses.append(f"{axis}_ill_conditioned_time_basis")
        elif age_s > max_sample_age_s:
            statuses.append(f"{axis}_motion_estimate_stale")
        elif float(np.max(intervals)) > max_gap_s:
            statuses.append(f"{axis}_sample_gap_exceeds_limit")
        elif timing_fraction > max_timing_uncertainty_fraction:
            statuses.append(f"{axis}_timing_bracket_too_wide")
        elif max_fit_rms_um is not None and rms > max_fit_rms_um:
            statuses.append(f"{axis}_trajectory_fit_residual_exceeds_limit")

    vx, vy = velocities
    if hypot(vx, vy) <= 3 * sum(velocity_errors):
        statuses.append("motion_below_fit_resolution")
    status = ";".join(statuses) if statuses else "valid"
    return StageMotionEstimate(
        request_id=next(iter(request_ids)),
        current_velocity_xy_um_s=(vx, vy),
        current_acceleration_xy_um_s2=(accelerations[0], accelerations[1]),
        velocity_standard_error_xy_um_s=(velocity_errors[0], velocity_errors[1]),
        acceleration_standard_error_xy_um_s2=(
            acceleration_errors[0],
            acceleration_errors[1],
        ),
        window_start_ns=min(starts),
        window_end_ns=common_fit_time_ns,
        maximum_sample_age_s=max(ages),
        maximum_gap_s=max(gaps),
        maximum_half_bracket_s=max(half_brackets),
        position_fit_rms_um=float(hypot(*rms_values)),
        sample_count=len(source_ids),
        valid=not statuses,
        status=status,
        evaluation_time_ns=evaluation_time_ns,
        observation_event_ids=tuple(source_ids),
        current_position_xy_um=(positions_at_fit[0], positions_at_fit[1]),
        position_uncertainty_xy_um=(position_errors[0], position_errors[1]),
    )
