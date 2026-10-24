"""Hardware-independent camera/stage hand-eye fit contracts.

Image motion is the observed motion of sample features. Mosaic placement is
the inverse mapping used to place a moving image into the reference frame.
This module consumes measured stage deltas and registered image shifts; it
does not acquire images or issue device commands.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


def _point2(row: Iterable[float]) -> tuple[float, float]:
    """Return an XY pair from a two-element row."""
    values = tuple(float(value) for value in row)
    if len(values) != 2:
        raise ValueError("expected an XY pair")
    return (values[0], values[1])


def _matrix2(
    matrix: Iterable[Iterable[float]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return a 2x2 matrix from a two-row nested sequence."""
    rows = tuple(_point2(row) for row in matrix)
    if len(rows) != 2:
        raise ValueError("expected a 2x2 matrix")
    return (rows[0], rows[1])


Matrix2 = tuple[tuple[float, float], tuple[float, float]]
Point2 = tuple[float, float]

CAPTURE_LABELS = (
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
)


def hand_eye_capture_labels(repeats_per_direction: int = 1) -> tuple[str, ...]:
    """Return the event labels for one hand-eye run with repeated probes.

    Parameters
    ----------
    repeats_per_direction
        Number of complete signed X/Y probe cycles. One preserves the original
        capture labels; multiple cycles suffix each probe and return label with
        its one-based repeat number.
    """
    if (
        isinstance(repeats_per_direction, bool)
        or not isinstance(repeats_per_direction, int)
        or repeats_per_direction < 1
    ):
        raise ValueError("repeats_per_direction must be a positive integer")
    labels = list(CAPTURE_LABELS[:2])
    for repeat_index in range(1, repeats_per_direction + 1):
        suffix = f"_repeat_{repeat_index:02d}" if repeats_per_direction > 1 else ""
        for probe_label, return_label in (
            ("probe_x_positive", "return_x_positive"),
            ("probe_x_negative", "return_x_negative"),
            ("probe_y_positive", "return_y_positive"),
            ("probe_y_negative", "return_y_negative"),
        ):
            labels.extend((f"{probe_label}{suffix}", f"{return_label}{suffix}"))
    return tuple(labels)


@dataclass(frozen=True)
class HandEyePair:
    """One measured readback delta and its registered feature displacement."""

    pair_id: str
    stage_delta_xy_um: Point2
    content_motion_xy_px: Point2
    registration_score: float
    feature_motion_xy_px: Point2 | None = None
    ransac_inlier_count: int | None = None
    ransac_inlier_fraction: float | None = None
    ransac_median_residual_px: float | None = None

    def __post_init__(self) -> None:
        """Check the declared fields for consistency."""
        if not self.pair_id:
            raise ValueError("pair_id is required")
        for label, value in (
            ("stage delta", self.stage_delta_xy_um),
            ("content motion", self.content_motion_xy_px),
        ):
            if len(value) != 2 or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not isfinite(item)
                for item in value
            ):
                raise ValueError(f"{label} must be a finite XY pair")
        if (
            isinstance(self.registration_score, bool)
            or not isinstance(self.registration_score, (int, float))
            or not isfinite(self.registration_score)
            or not -1 <= self.registration_score <= 1
        ):
            raise ValueError("registration_score must be finite and in [-1, 1]")
        if self.feature_motion_xy_px is not None and (
            len(self.feature_motion_xy_px) != 2
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not isfinite(item)
                for item in self.feature_motion_xy_px
            )
        ):
            raise ValueError("feature motion must be a finite XY pair")
        metrics = (
            self.ransac_inlier_count,
            self.ransac_inlier_fraction,
            self.ransac_median_residual_px,
        )
        if any(metric is not None for metric in metrics) and any(
            metric is None for metric in metrics
        ):
            raise ValueError("all RANSAC quality metrics must be supplied together")
        if self.ransac_inlier_count is not None:
            fraction = self.ransac_inlier_fraction
            residual_px = self.ransac_median_residual_px
            if (
                isinstance(self.ransac_inlier_count, bool)
                or not isinstance(self.ransac_inlier_count, int)
                or self.ransac_inlier_count < 0
                or fraction is None
                or not isfinite(fraction)
                or not 0 <= fraction <= 1
                or residual_px is None
                or not isfinite(residual_px)
                or residual_px < 0
            ):
                raise ValueError("invalid RANSAC quality metrics")


@dataclass(frozen=True)
class HandEyeClosure:
    """Measured stage and image return error for one probe's anchor return."""

    label: str
    stage_error_xy_um: Point2
    image_displacement_xy_px: Point2

    def __post_init__(self) -> None:
        """Check the declared fields for consistency."""
        if not self.label.startswith("return_"):
            raise ValueError("closure label must begin with 'return_'")
        for name, value in (
            ("stage_error_xy_um", self.stage_error_xy_um),
            ("image_displacement_xy_px", self.image_displacement_xy_px),
        ):
            if len(value) != 2 or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not isfinite(item)
                for item in value
            ):
                raise ValueError(f"{name} must be a finite XY pair")


@dataclass(frozen=True)
class HandEyeQualityCriteria:
    """Acceptance thresholds applied to hand-eye calibration evidence."""

    minimum_ransac_inliers: int = 30
    minimum_inlier_fraction: float = 0.35
    maximum_feature_residual_px: float = 2.5
    minimum_ecc_score: float = 0.4
    maximum_ecc_feature_disagreement_px: float = 3.0
    maximum_return_stage_error_um: float = 1.0
    maximum_return_image_displacement_px: float = 3.0

    def __post_init__(self) -> None:
        """Check the declared fields for consistency."""
        if (
            isinstance(self.minimum_ransac_inliers, bool)
            or not isinstance(self.minimum_ransac_inliers, int)
            or self.minimum_ransac_inliers < 1
        ):
            raise ValueError("minimum_ransac_inliers must be a positive integer")
        if not 0 <= self.minimum_inlier_fraction <= 1:
            raise ValueError("minimum_inlier_fraction must be in [0, 1]")
        if not -1 <= self.minimum_ecc_score <= 1:
            raise ValueError("minimum_ecc_score must be in [-1, 1]")
        for name, value in (
            ("maximum_feature_residual_px", self.maximum_feature_residual_px),
            (
                "maximum_ecc_feature_disagreement_px",
                self.maximum_ecc_feature_disagreement_px,
            ),
            ("maximum_return_stage_error_um", self.maximum_return_stage_error_um),
            (
                "maximum_return_image_displacement_px",
                self.maximum_return_image_displacement_px,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class HandEyeQualityResult:
    """Outcome of the hand-eye quality gates and the checks that failed."""

    passed: bool
    failed_checks: tuple[str, ...]


@dataclass(frozen=True)
class HandEyeFit:
    """Fitted readback-to-image transforms and fit diagnostics."""

    calibration_id: str
    coordinate_frame: str
    mode_context: tuple[tuple[str, str], ...]
    stage_units: str
    image_units: str
    content_motion_px_per_stage_unit: Matrix2
    mosaic_placement_px_per_stage_unit: Matrix2
    pair_ids: tuple[str, ...]
    residuals_xy_px: tuple[Point2, ...]
    rmse_px: float
    covariance: tuple[tuple[float, ...], ...]


def fit_hand_eye(
    pairs: Sequence[HandEyePair],
    *,
    coordinate_frame: str,
    minimum_registration_score: float = 0.4,
    minimum_pairs: int = 4,
    maximum_residual_px: float | None = 2.5,
    mode_context: Mapping[str, str] | None = None,
) -> HandEyeFit:
    """Fit ``Δimage_content = A @ Δstage_readback`` by least squares.

    Inputs must contain positive and negative observed motion on both stage
    axes. ECC callers must pass the direct content-motion translation, not
    the moving-to-reference placement translation. The returned placement
    matrix is ``-A`` for the translation-only convention.
    """
    if not coordinate_frame:
        raise ValueError("coordinate_frame is required")
    if mode_context is not None and any(
        not isinstance(key, str) or not key or not isinstance(value, str) or not value
        for key, value in mode_context.items()
    ):
        raise ValueError("mode_context must map nonempty strings to nonempty strings")
    context = tuple(sorted((mode_context or {}).items()))
    if minimum_pairs < 4 or len(pairs) < minimum_pairs:
        raise ValueError(f"at least {minimum_pairs} registration pairs are required")
    if len({pair.pair_id for pair in pairs}) != len(pairs):
        raise ValueError("pair_id values must be unique")
    if not -1 <= minimum_registration_score <= 1:
        raise ValueError("minimum_registration_score must be in [-1, 1]")
    selected = [p for p in pairs if p.registration_score >= minimum_registration_score]
    if len(selected) < minimum_pairs:
        raise ValueError("too few pairs meet the registration score threshold")

    stage = np.asarray([p.stage_delta_xy_um for p in selected], dtype=np.float64)
    image = np.asarray([p.content_motion_xy_px for p in selected], dtype=np.float64)
    for axis, name in enumerate(("X", "Y")):
        if not (np.any(stage[:, axis] > 0) and np.any(stage[:, axis] < 0)):
            raise ValueError(
                f"probe pairs must include positive and negative {name} motion"
            )
    if np.linalg.matrix_rank(stage) < 2:
        raise ValueError("stage probe deltas do not span both axes")

    # B has stage rows and image columns, hence A = B.T maps stage XY to image XY.
    coefficients, _residual_sum, _rank, _singular = np.linalg.lstsq(
        stage, image, rcond=None
    )
    predicted = stage @ coefficients
    residuals = image - predicted
    norms = np.linalg.norm(residuals, axis=1)
    if maximum_residual_px is not None:
        if not isfinite(maximum_residual_px) or maximum_residual_px <= 0:
            raise ValueError("maximum_residual_px must be finite and positive")
        if np.any(norms > maximum_residual_px):
            raise ValueError(
                "one or more registration residuals exceed the configured limit"
            )
    matrix = coefficients.T
    placement = -matrix
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    dof = max(1, len(selected) - 2)
    variance = float(np.sum(np.square(residuals)) / dof)
    stage_cov = np.linalg.pinv(stage.T @ stage)
    # Flattened (image-x/stage-x, image-x/stage-y, image-y/stage-x, image-y/stage-y).
    covariance = np.kron(np.eye(2), stage_cov * variance)
    canonical = {
        "coordinate_frame": coordinate_frame,
        "mode_context": context,
        "stage_units": "um",
        "image_units": "px",
        "matrix": matrix.tolist(),
        "pair_ids": [p.pair_id for p in selected],
    }
    calibration_id = (
        "hand-eye-"
        + sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
    )
    return HandEyeFit(
        calibration_id=calibration_id,
        coordinate_frame=coordinate_frame,
        mode_context=context,
        stage_units="um",
        image_units="px",
        content_motion_px_per_stage_unit=_matrix2(matrix),
        mosaic_placement_px_per_stage_unit=_matrix2(placement),
        pair_ids=tuple(p.pair_id for p in selected),
        residuals_xy_px=tuple(_point2(row) for row in residuals),
        rmse_px=rmse,
        covariance=tuple(tuple(float(v) for v in row) for row in covariance),
    )


def matches_hand_eye_mode(fit: HandEyeFit, mode_context: Mapping[str, str]) -> bool:
    """Return whether a calibration was fitted for the exact supplied mode."""
    if any(
        not isinstance(k, str) or not k or not isinstance(v, str) or not v
        for k, v in mode_context.items()
    ):
        raise ValueError("mode_context must map nonempty strings to nonempty strings")
    return fit.mode_context == tuple(sorted(mode_context.items()))


def evaluate_hand_eye_quality(
    pairs: Sequence[HandEyePair],
    closures: Sequence[HandEyeClosure],
    criteria: HandEyeQualityCriteria | None = None,
) -> HandEyeQualityResult:
    """Evaluate RANSAC/ECC agreement and stage/image return-closure gates."""
    if criteria is None:
        criteria = HandEyeQualityCriteria()
    if len({pair.pair_id for pair in pairs}) != len(pairs):
        raise ValueError("pair_id values must be unique")
    if len({closure.label for closure in closures}) != len(closures):
        raise ValueError("closure labels must be unique")
    required_closures = {
        "return_x_positive",
        "return_x_negative",
        "return_y_positive",
        "return_y_negative",
    }
    failed: list[str] = []
    if len(pairs) < 4:
        failed.append("fewer than four signed axis registration pairs")
    if pairs:
        stage_deltas = np.asarray([pair.stage_delta_xy_um for pair in pairs])
        for axis, name in enumerate(("X", "Y")):
            if not np.any(stage_deltas[:, axis] > 0):
                failed.append(f"missing positive {name} registration pair")
            if not np.any(stage_deltas[:, axis] < 0):
                failed.append(f"missing negative {name} registration pair")
    available_closures = {closure.label for closure in closures}
    for required in sorted(required_closures):
        if not any(
            label == required or label.startswith(f"{required}_repeat_")
            for label in available_closures
        ):
            failed.append(f"missing closure evidence: {required}")
    for pair in pairs:
        prefix = f"{pair.pair_id}: "
        if pair.registration_score < criteria.minimum_ecc_score:
            failed.append(prefix + "ECC score below threshold")
        if pair.feature_motion_xy_px is None or pair.ransac_inlier_count is None:
            failed.append(prefix + "feature registration evidence missing")
            continue
        fraction = pair.ransac_inlier_fraction
        residual_px = pair.ransac_median_residual_px
        if fraction is None or residual_px is None:
            failed.append(prefix + "feature registration evidence missing")
            continue
        if pair.ransac_inlier_count < criteria.minimum_ransac_inliers:
            failed.append(prefix + "RANSAC inlier count below threshold")
        if fraction < criteria.minimum_inlier_fraction:
            failed.append(prefix + "RANSAC inlier fraction below threshold")
        if residual_px > criteria.maximum_feature_residual_px:
            failed.append(prefix + "RANSAC residual above threshold")
        disagreement = float(
            np.linalg.norm(
                np.asarray(pair.content_motion_xy_px)
                - np.asarray(pair.feature_motion_xy_px)
            )
        )
        if disagreement > criteria.maximum_ecc_feature_disagreement_px:
            failed.append(prefix + "ECC/RANSAC motion disagreement above threshold")
    for closure in closures:
        if (
            np.linalg.norm(closure.stage_error_xy_um)
            > criteria.maximum_return_stage_error_um
        ):
            failed.append(closure.label + ": stage return error above threshold")
        if (
            np.linalg.norm(closure.image_displacement_xy_px)
            > criteria.maximum_return_image_displacement_px
        ):
            failed.append(closure.label + ": image return displacement above threshold")
    return HandEyeQualityResult(not failed, tuple(failed))
