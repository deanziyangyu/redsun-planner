from __future__ import annotations

import numpy as np
import pytest

from planner import (
    HandEyeClosure,
    HandEyePair,
    evaluate_hand_eye_quality,
    fit_hand_eye,
    matches_hand_eye_mode,
)


def test_fit_hand_eye_preserves_camera_rotation_and_transform_conventions() -> None:
    # Stage X mostly maps to image down; stage Y mostly maps to image right.
    matrix = np.asarray([[-0.05, 3.82], [3.85, 0.03]])
    deltas = [
        (10.0, 0.0),
        (-10.0, 0.0),
        (0.0, 10.0),
        (0.0, -10.0),
        (10.0, 10.0),
        (-10.0, -10.0),
    ]
    pairs = [
        HandEyePair(
            str(index),
            delta,
            tuple(matrix @ np.asarray(delta)),
            0.99,
        )
        for index, delta in enumerate(deltas)
    ]

    context = {"camera": "demo-camera", "binning": "2x2", "roi": "full"}
    fit = fit_hand_eye(pairs, coordinate_frame="ix83-stage-xy", mode_context=context)

    np.testing.assert_allclose(fit.content_motion_px_per_stage_unit, matrix, atol=1e-10)
    np.testing.assert_allclose(
        fit.mosaic_placement_px_per_stage_unit, -matrix, atol=1e-10
    )
    assert fit.rmse_px < 1e-10
    assert len(fit.calibration_id) == len("hand-eye-") + 16
    assert len(fit.covariance) == 4
    assert matches_hand_eye_mode(fit, context)
    assert not matches_hand_eye_mode(fit, {"camera": "demo-camera", "binning": "1x1"})


def test_fit_hand_eye_requires_both_directions_and_good_registration() -> None:
    pairs = [
        HandEyePair("xp", (10, 0), (0, 38), 0.9),
        HandEyePair("xn", (-10, 0), (0, -38), 0.9),
        HandEyePair("yp", (0, 10), (38, 0), 0.9),
        HandEyePair("yn", (0, -10), (-38, 0), 0.1),
    ]
    with pytest.raises(ValueError, match="too few pairs"):
        fit_hand_eye(pairs, coordinate_frame="stage", minimum_registration_score=0.4)
    with pytest.raises(ValueError, match="positive and negative Y"):
        fit_hand_eye(
            pairs[:3] + [HandEyePair("xp2", (10, 0), (0, 38), 0.9)],
            coordinate_frame="stage",
        )


def test_hand_eye_quality_gates_registration_agreement_and_return_closure() -> None:
    matrix = np.asarray([[-0.05, 3.82], [3.85, 0.03]])
    deltas = [(10, 0), (-10, 0), (0, 10), (0, -10)]
    pairs = [
        HandEyePair(
            str(index),
            delta,
            tuple(matrix @ np.asarray(delta)),
            0.9,
            feature_motion_xy_px=tuple(matrix @ np.asarray(delta)),
            ransac_inlier_count=50,
            ransac_inlier_fraction=0.6,
            ransac_median_residual_px=1.2,
        )
        for index, delta in enumerate(deltas)
    ]
    closures = [
        HandEyeClosure(label, (0.2, -0.1), (0.5, 0.4))
        for label in (
            "return_x_positive",
            "return_x_negative",
            "return_y_positive",
            "return_y_negative",
        )
    ]
    assert evaluate_hand_eye_quality(pairs, closures).passed
    assert not evaluate_hand_eye_quality([], closures).passed

    bad_closures = [
        HandEyeClosure(label, (1.2, 0), (0, 0))
        if label == "return_x_positive"
        else closure
        for label, closure in zip(
            (
                "return_x_positive",
                "return_x_negative",
                "return_y_positive",
                "return_y_negative",
            ),
            closures,
        )
    ]
    result = evaluate_hand_eye_quality(pairs, bad_closures)
    assert not result.passed
    assert (
        "return_x_positive: stage return error above threshold" in result.failed_checks
    )
