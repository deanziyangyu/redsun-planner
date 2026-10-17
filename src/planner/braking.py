"""Geometry-aware feasibility of search-to-detail speed transitions."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite

from .geometry import Calibration
from .motion_state import StageMotionEstimate


@dataclass(frozen=True)
class RoiBrakingAssessment:
    """Lead-distance budget for one calibrated detector/detail geometry."""

    available_lead_um: float
    required_latency_distance_um: float
    required_braking_distance_um: float
    required_lead_um: float
    margin_um: float
    speed_at_command_um_s: float
    transition_time_s: float
    cross_track_gap_um: float
    cross_track_intervals_overlap: bool
    feasible: bool


@dataclass(frozen=True)
class CalibratedRoiBrakingAssessment:
    """Braking result linked to the calibration and measured motion window."""

    assessment: RoiBrakingAssessment
    calibration_id: str
    request_id: str
    feature_motion_direction_stage_xy: tuple[float, float]


def assess_calibrated_roi_braking(
    *,
    sensor_shape_yx: tuple[int, int],
    detector_roi_yxhw: tuple[int, int, int, int],
    detail_roi_yxhw: tuple[int, int, int, int],
    calibration: Calibration,
    motion: StageMotionEstimate,
    detail_speed_um_s: float,
    deceleration_um_s2: float,
    detection_latency_s: float,
    uncertainty_margin_um: float,
    confidence_sigma: float = 2.0,
) -> CalibratedRoiBrakingAssessment:
    """Project calibrated ROI geometry against measured, bracketed stage state.

    `Calibration.pixel_to_stage` maps native sensor pixels to stage-plane
    micrometres; its ROI origin and binning are applied by `image_to_stage`.
    A stationary specimen appears to move opposite the stage velocity in the
    image. The returned feasibility therefore uses the measured XY velocity
    and acceleration projected onto that feature-motion direction. Available
    lead runs from the detector ROI's downstream edge to the detail ROI's entry
    edge. Feasibility also requires overlapping cross-track intervals, so at
    least one point-feature path could pass through both ROIs. An integrated
    detector event does not localize its source within the detector ROI; this
    overlap is a possibility check, not proof that the triggering feature takes
    that path. The caller must supply physical deceleration and a positive
    distance reserve for affine calibration uncertainty and unmodeled
    controller effects. Fitted velocity and acceleration errors are expanded
    by `confidence_sigma` in the hazardous direction before distance
    calculations. This is not evidence that an adapter can change speed
    during a dispatched move.
    """
    if not isinstance(calibration, Calibration):
        raise TypeError("calibration must be a planner.geometry.Calibration")
    if not isinstance(motion, StageMotionEstimate):
        raise TypeError("motion must be a StageMotionEstimate")
    if not motion.valid:
        raise ValueError(f"motion estimate is not usable: {motion.status}")
    if not isfinite(uncertainty_margin_um) or uncertainty_margin_um <= 0:
        raise ValueError("uncertainty_margin_um must be finite and positive")
    if not isfinite(confidence_sigma) or confidence_sigma <= 0:
        raise ValueError("confidence_sigma must be finite and positive")
    height, width = sensor_shape_yx
    if (
        isinstance(height, bool)
        or isinstance(width, bool)
        or not isinstance(height, int)
        or not isinstance(width, int)
        or height <= 0
        or width <= 0
    ):
        raise ValueError("sensor_shape_yx must contain positive integer dimensions")

    def checked_roi(
        name: str, roi: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        if len(roi) != 4 or any(
            isinstance(value, bool) or not isinstance(value, int) for value in roi
        ):
            raise ValueError(f"{name} must contain integer y, x, height, width")
        y, x, roi_height, roi_width = roi
        if min(y, x) < 0 or min(roi_height, roi_width) <= 0:
            raise ValueError(f"{name} must have a nonnegative origin and positive size")
        if y + roi_height > height or x + roi_width > width:
            raise ValueError(f"{name} must fit inside sensor_shape_yx")
        return roi

    detector_roi_yxhw = checked_roi("detector_roi_yxhw", detector_roi_yxhw)
    detail_roi_yxhw = checked_roi("detail_roi_yxhw", detail_roi_yxhw)
    vx, vy = motion.current_velocity_xy_um_s
    ax, ay = motion.current_acceleration_xy_um_s2
    speed = hypot(vx, vy)
    if speed <= 0:
        raise ValueError("measured stage velocity must be nonzero")
    # Feature motion in the detector is opposite translation of a stationary
    # specimen under the stage/camera calibration.
    feature_direction = (-vx / speed, -vy / speed)

    def stage_point(x: float, y: float) -> tuple[float, float]:
        return calibration.image_to_stage(x, y, stage_xy_um=(0.0, 0.0))

    y, x, roi_height, roi_width = detector_roi_yxhw
    corners_px = (
        (x - 0.5, y - 0.5),
        (x + roi_width - 0.5, y - 0.5),
        (x - 0.5, y + roi_height - 0.5),
        (x + roi_width - 0.5, y + roi_height - 0.5),
    )
    detector_corners = tuple(stage_point(px, py) for px, py in corners_px)
    detector_downstream_boundary = max(
        detector_corners,
        key=lambda point: (
            point[0] * feature_direction[0] + point[1] * feature_direction[1]
        ),
    )
    detail_y, detail_x, detail_height, detail_width = detail_roi_yxhw
    detail_corners_px = (
        (detail_x - 0.5, detail_y - 0.5),
        (detail_x + detail_width - 0.5, detail_y - 0.5),
        (detail_x - 0.5, detail_y + detail_height - 0.5),
        (detail_x + detail_width - 0.5, detail_y + detail_height - 0.5),
    )
    detail_corners = tuple(stage_point(px, py) for px, py in detail_corners_px)
    detail_entry_boundary = min(
        detail_corners,
        key=lambda point: (
            point[0] * feature_direction[0] + point[1] * feature_direction[1]
        ),
    )
    available_lead = sum(
        (detail_entry_boundary[index] - detector_downstream_boundary[index])
        * feature_direction[index]
        for index in range(2)
    )
    cross_track_direction = (-feature_direction[1], feature_direction[0])

    def cross_track_interval(
        corners: tuple[tuple[float, float], ...],
    ) -> tuple[float, float]:
        projections = tuple(
            point[0] * cross_track_direction[0] + point[1] * cross_track_direction[1]
            for point in corners
        )
        return min(projections), max(projections)

    detector_cross_track = cross_track_interval(detector_corners)
    detail_cross_track = cross_track_interval(detail_corners)
    cross_track_gap = max(
        0.0,
        detector_cross_track[0] - detail_cross_track[1],
        detail_cross_track[0] - detector_cross_track[1],
    )
    cross_track_overlap = cross_track_gap <= 1e-9
    feature_acceleration = -(ax * feature_direction[0] + ay * feature_direction[1])
    # The estimator currently exports per-axis errors, not the XY covariance.
    # Use an L1 projection bound until covariance is measured; RSS would
    # incorrectly assume independent axis errors and could understate risk.
    velocity_sigma = sum(
        abs(feature_direction[index]) * motion.velocity_standard_error_xy_um_s[index]
        for index in range(2)
    )
    acceleration_sigma = sum(
        abs(feature_direction[index])
        * motion.acceleration_standard_error_xy_um_s2[index]
        for index in range(2)
    )
    # Use the selected confidence bound in the hazardous direction: higher
    # speed and acceleration toward the detection/detail path require more lead.
    speed_for_braking = speed + confidence_sigma * velocity_sigma
    acceleration_for_braking = (
        feature_acceleration + confidence_sigma * acceleration_sigma
    )
    latency = detection_latency_s
    if not isfinite(latency) or latency < 0:
        raise ValueError("detection_latency_s must be finite and nonnegative")
    if not isfinite(detail_speed_um_s) or detail_speed_um_s < 0:
        raise ValueError("detail_speed_um_s must be finite and nonnegative")
    if detail_speed_um_s > speed:
        raise ValueError("detail_speed_um_s cannot exceed measured stage speed")
    if not isfinite(deceleration_um_s2) or deceleration_um_s2 <= 0:
        raise ValueError("deceleration_um_s2 must be finite and positive")

    latency_interval = latency
    if acceleration_for_braking < 0:
        latency_interval = min(latency, speed_for_braking / -acceleration_for_braking)
    latency_distance = max(
        0.0,
        speed_for_braking * latency_interval
        + 0.5 * acceleration_for_braking * latency_interval**2,
    )
    speed_at_command = max(0.0, speed_for_braking + acceleration_for_braking * latency)
    braking_distance = max(
        0.0,
        (speed_at_command**2 - detail_speed_um_s**2) / (2 * deceleration_um_s2),
    )
    has_downshift = speed_at_command > detail_speed_um_s
    transition_time = (
        latency + (speed_at_command - detail_speed_um_s) / deceleration_um_s2
        if has_downshift
        else 0.0
    )
    required_lead = latency_distance + braking_distance + uncertainty_margin_um
    assessment = RoiBrakingAssessment(
        available_lead_um=available_lead,
        required_latency_distance_um=latency_distance,
        required_braking_distance_um=braking_distance,
        required_lead_um=required_lead,
        margin_um=available_lead - required_lead,
        speed_at_command_um_s=speed_at_command,
        transition_time_s=transition_time,
        cross_track_gap_um=cross_track_gap,
        cross_track_intervals_overlap=cross_track_overlap,
        feasible=available_lead >= required_lead and cross_track_overlap,
    )
    return CalibratedRoiBrakingAssessment(
        assessment=assessment,
        calibration_id=calibration.calibration_id,
        request_id=motion.request_id,
        feature_motion_direction_stage_xy=feature_direction,
    )


def assess_roi_braking(
    *,
    sensor_shape_yx: tuple[int, int],
    detector_roi_yxhw: tuple[int, int, int, int],
    detail_roi_yxhw: tuple[int, int, int, int],
    pixel_size_um_yx: tuple[float, float],
    feature_motion_direction_yx: tuple[float, float],
    current_speed_um_s: float,
    current_acceleration_um_s2: float,
    detail_speed_um_s: float,
    deceleration_um_s2: float,
    detection_latency_s: float,
) -> RoiBrakingAssessment:
    """Assess whether an ROI event leaves enough lead for a speed ramp.

    ROI coordinates are ``(y, x, height, width)`` in sensor pixels. The
    calibrated feature-motion direction is a unit vector in physical image
    coordinates (positive Y/X follows increasing image coordinates). The
    model measures available lead from the detector ROI's downstream edge to
    the detail ROI's entry edge. It requires overlapping cross-track
    projections, so some point-feature path could pass through both ROIs;
    integrated detector events do not localize their source and this is only a
    possibility check.

    During detector/transport latency, the current signed acceleration is held
    constant to estimate distance and speed at the time a deceleration command
    can take effect. The required distance then uses constant-deceleration
    kinematics. This does not model jerk,
    speed-controller update quantization, settling, or uncertain calibration;
    callers should reserve additional margin for those effects.
    """
    height, width = sensor_shape_yx
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (height, width)
    ):
        raise ValueError("sensor_shape_yx must contain positive integer dimensions")

    def checked_roi(
        name: str, roi: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        if len(roi) != 4 or any(
            isinstance(value, bool) or not isinstance(value, int) for value in roi
        ):
            raise ValueError(f"{name} must contain integer y, x, height, width")
        y, x, roi_height, roi_width = roi
        if min(y, x) < 0 or min(roi_height, roi_width) <= 0:
            raise ValueError(f"{name} must have a nonnegative origin and positive size")
        if y + roi_height > height or x + roi_width > width:
            raise ValueError(f"{name} must fit inside sensor_shape_yx")
        return roi

    detector_roi_yxhw = checked_roi("detector_roi_yxhw", detector_roi_yxhw)
    detail_roi_yxhw = checked_roi("detail_roi_yxhw", detail_roi_yxhw)
    sy, sx = pixel_size_um_yx
    if not all(isfinite(value) and value > 0 for value in (sy, sx)):
        raise ValueError("pixel_size_um_yx values must be finite and positive")
    dy, dx = feature_motion_direction_yx
    direction_norm = hypot(dy, dx)
    if not isfinite(direction_norm) or abs(direction_norm - 1.0) > 1e-6:
        raise ValueError("feature_motion_direction_yx must be a finite unit vector")
    if not all(
        isfinite(value) and value >= 0
        for value in (current_speed_um_s, detail_speed_um_s, detection_latency_s)
    ):
        raise ValueError(
            "speeds and detection_latency_s must be finite and nonnegative"
        )
    if detail_speed_um_s > current_speed_um_s:
        raise ValueError("detail_speed_um_s cannot exceed current_speed_um_s")
    if not isfinite(current_acceleration_um_s2):
        raise ValueError("current_acceleration_um_s2 must be finite")
    if not isfinite(deceleration_um_s2) or deceleration_um_s2 <= 0:
        raise ValueError("deceleration_um_s2 must be finite and positive")

    y, x, roi_height, roi_width = detector_roi_yxhw
    # Pixel coordinates refer to pixel centers; the detector ROI boundary is
    # half a pixel beyond the first/last center.
    corners_px = (
        (y - 0.5, x - 0.5),
        (y - 0.5, x + roi_width - 0.5),
        (y + roi_height - 0.5, x - 0.5),
        (y + roi_height - 0.5, x + roi_width - 0.5),
    )
    corners_um = tuple((py * sy, px * sx) for py, px in corners_px)
    detector_downstream_boundary = max(
        corners_um, key=lambda point: point[0] * dy + point[1] * dx
    )

    detail_y, detail_x, detail_height, detail_width = detail_roi_yxhw
    detail_corners_um = tuple(
        (py * sy, px * sx)
        for px, py in (
            (detail_x - 0.5, detail_y - 0.5),
            (detail_x + detail_width - 0.5, detail_y - 0.5),
            (detail_x - 0.5, detail_y + detail_height - 0.5),
            (detail_x + detail_width - 0.5, detail_y + detail_height - 0.5),
        )
    )
    detail_entry_boundary = min(
        detail_corners_um,
        key=lambda point: point[0] * dy + point[1] * dx,
    )
    available = (detail_entry_boundary[0] - detector_downstream_boundary[0]) * dy + (
        detail_entry_boundary[1] - detector_downstream_boundary[1]
    ) * dx
    cross_track_direction = (-dx, dy)

    def cross_track_interval(
        corners: tuple[tuple[float, float], ...],
    ) -> tuple[float, float]:
        projections = tuple(
            point[0] * cross_track_direction[0] + point[1] * cross_track_direction[1]
            for point in corners
        )
        return min(projections), max(projections)

    detector_cross_track = cross_track_interval(corners_um)
    detail_cross_track = cross_track_interval(detail_corners_um)
    cross_track_gap = max(
        0.0,
        detector_cross_track[0] - detail_cross_track[1],
        detail_cross_track[0] - detector_cross_track[1],
    )
    cross_track_overlap = cross_track_gap <= 1e-9
    latency_interval_s = detection_latency_s
    if current_acceleration_um_s2 < 0:
        latency_interval_s = min(
            detection_latency_s,
            current_speed_um_s / -current_acceleration_um_s2,
        )
    latency_distance = max(
        0.0,
        current_speed_um_s * latency_interval_s
        + 0.5 * current_acceleration_um_s2 * latency_interval_s**2,
    )
    speed_at_command = max(
        0.0,
        current_speed_um_s + current_acceleration_um_s2 * detection_latency_s,
    )
    has_downshift = speed_at_command > detail_speed_um_s
    braking_distance = max(
        0.0,
        (speed_at_command**2 - detail_speed_um_s**2) / (2 * deceleration_um_s2),
    )
    required = latency_distance + braking_distance
    transition_time = (
        detection_latency_s
        + (speed_at_command - detail_speed_um_s) / deceleration_um_s2
        if has_downshift
        else 0.0
    )
    return RoiBrakingAssessment(
        available_lead_um=available,
        required_latency_distance_um=latency_distance,
        required_braking_distance_um=braking_distance,
        required_lead_um=required,
        margin_um=available - required,
        speed_at_command_um_s=speed_at_command,
        transition_time_s=transition_time,
        cross_track_gap_um=cross_track_gap,
        cross_track_intervals_overlap=cross_track_overlap,
        feasible=available >= required and cross_track_overlap,
    )
