"""Pure detector ROI placement from calibrated stage and image motion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import hypot, isfinite
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class LeadingEdgeRoiProposal:
    """Axis-aligned detector ROI proposed for the leading image-motion edge."""

    calibration_id: str
    roi_yxhw: tuple[int, int, int, int]
    image_motion_direction_px_xy: tuple[float, float]
    selected_readout_edge: str
    dominant_axis_fraction: float
    approximation: str
    binning_xy: tuple[int, int]
    direction_semantics: str

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready geometry with its approximation disclosed."""
        return asdict(self)


def calibrated_leading_edge_roi(
    *,
    calibration_id: str,
    pixel_to_stage_um_per_native_px: Any,
    binning_xy: tuple[int, int],
    stage_travel_xy_um: tuple[float, float],
    frame_shape_yx: tuple[int, int],
    depth_px: int,
) -> LeadingEdgeRoiProposal:
    """Propose a rectangular ROI at the leading image edge for a stage move.

    ``pixel_to_stage_um_per_native_px`` is the accepted hand-eye 2×2 affine
    linear component, mapping native sensor pixel displacement ``(x, y)`` to
    stage displacement ``(x, y)`` in micrometres. ``stage_travel_xy_um`` is
    the planned stage displacement. A stationary specimen appears to move
    opposite the stage, so the inverse affine is applied to the negative stage
    vector to predict image-content displacement in delivered pixels. The
    proposal takes a band at the dominant signed image-motion edge.

    The current detector ROI is axis-aligned. For a rotated camera this helper
    chooses the dominant readout axis and reports the fraction of motion along
    it. It does not claim an exact oblique leading edge, transform an ROI to
    physical stage coordinates, configure a camera, or validate detection
    performance. The host must use a passed, mode-matched calibration and
    review the proposal before using it in an acquisition recipe.
    """
    if not isinstance(calibration_id, str) or not calibration_id.strip():
        raise ValueError("calibration_id must be a nonempty string")
    matrix = np.asarray(pixel_to_stage_um_per_native_px, dtype=np.float64)
    if matrix.shape != (2, 2) or not np.isfinite(matrix).all():
        raise ValueError("pixel_to_stage matrix must be a finite 2 by 2 matrix")
    determinant = float(np.linalg.det(matrix))
    if not isfinite(determinant) or abs(determinant) <= 1e-12:
        raise ValueError("pixel_to_stage matrix must be invertible")
    if (
        not isinstance(binning_xy, tuple)
        or len(binning_xy) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in binning_xy
        )
    ):
        raise ValueError("binning_xy must contain two positive integers")
    if (
        not isinstance(stage_travel_xy_um, tuple)
        or len(stage_travel_xy_um) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            for value in stage_travel_xy_um
        )
    ):
        raise ValueError("stage_travel_xy_um must contain two finite values")
    travel_x, travel_y = (float(value) for value in stage_travel_xy_um)
    if hypot(travel_x, travel_y) <= 0:
        raise ValueError("stage_travel_xy_um must be nonzero")
    if (
        not isinstance(frame_shape_yx, tuple)
        or len(frame_shape_yx) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in frame_shape_yx
        )
    ):
        raise ValueError("frame_shape_yx must contain two positive integers")
    if isinstance(depth_px, bool) or not isinstance(depth_px, int) or depth_px < 1:
        raise ValueError("depth_px must be a positive integer")
    frame_height, frame_width = frame_shape_yx
    if depth_px > max(frame_height, frame_width):
        raise ValueError("depth_px cannot exceed the larger frame dimension")

    pixel_to_stage_delivered = matrix @ np.diag(binning_xy)
    image_motion_xy = np.linalg.solve(
        pixel_to_stage_delivered,
        -np.asarray((travel_x, travel_y), dtype=np.float64),
    )
    norm = float(np.linalg.norm(image_motion_xy))
    if not isfinite(norm) or norm <= 0:
        raise ValueError("calibration produced no finite image-motion direction")
    dominant_axis = int(np.argmax(np.abs(image_motion_xy)))
    dominant_fraction = float(abs(image_motion_xy[dominant_axis]) / norm)

    if dominant_axis == 0:
        width = min(depth_px, frame_width)
        leading_right = image_motion_xy[0] > 0
        x = frame_width - width if leading_right else 0
        roi = (0, x, frame_height, width)
        edge = "right" if leading_right else "left"
    else:
        height = min(depth_px, frame_height)
        leading_bottom = image_motion_xy[1] > 0
        y = frame_height - height if leading_bottom else 0
        roi = (y, 0, height, frame_width)
        edge = "bottom" if leading_bottom else "top"

    return LeadingEdgeRoiProposal(
        calibration_id=calibration_id.strip(),
        roi_yxhw=roi,
        image_motion_direction_px_xy=(
            float(image_motion_xy[0] / norm),
            float(image_motion_xy[1] / norm),
        ),
        selected_readout_edge=edge,
        dominant_axis_fraction=dominant_fraction,
        approximation="full-span rectangular band on dominant calibrated sensor axis",
        binning_xy=binning_xy,
        direction_semantics="stationary specimen image motion opposite stage travel",
    )


__all__ = ["LeadingEdgeRoiProposal", "calibrated_leading_edge_roi"]
