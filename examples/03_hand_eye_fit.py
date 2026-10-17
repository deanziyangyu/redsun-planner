"""Fit a stage-to-camera transform from measured pairs, then gate its quality.

The inputs are numbers a host already measured: how far the stage moved, how far
the image content moved, and what the stage and image read at the return point.
Nothing here touches a device or an image.
"""

from __future__ import annotations

from planner import HandEyeClosure, HandEyePair, evaluate_hand_eye_quality, fit_hand_eye

# True relation used here: content_px = 0.5 * stage_um, independent per axis.
STAGE_STEP_UM = 10.0
SCALE_PX_PER_UM = 0.5
pairs = tuple(
    HandEyePair(
        pair_id=f"probe-{name}",
        stage_delta_xy_um=(dx, dy),
        content_motion_xy_px=(SCALE_PX_PER_UM * dx, SCALE_PX_PER_UM * dy),
        registration_score=0.9,
        feature_motion_xy_px=(SCALE_PX_PER_UM * dx, SCALE_PX_PER_UM * dy),
        ransac_inlier_count=220,
        ransac_inlier_fraction=0.91,
        ransac_median_residual_px=0.4,
    )
    for name, (dx, dy) in (
        ("plus-x", (STAGE_STEP_UM, 0.0)),
        ("minus-x", (-STAGE_STEP_UM, 0.0)),
        ("plus-y", (0.0, STAGE_STEP_UM)),
        ("minus-y", (0.0, -STAGE_STEP_UM)),
    )
)

fit = fit_hand_eye(pairs, coordinate_frame="stage_xy_um")
print("calibration id", fit.calibration_id)
print("content-motion matrix (px per um):")
for row in fit.content_motion_px_per_stage_unit:
    print(f"  [{row[0]:+.4f} {row[1]:+.4f}]")
print(f"fit rmse {fit.rmse_px:.3f} px")
print("placement matrix is its negation, for the translation-only convention")

# Return-closure observations: how far off the stage and the image were when the
# stage came back to the reference point.
closures = tuple(
    HandEyeClosure(label, (0.15, 0.10), (0.4, 0.3))
    for label in (
        "return_x_positive",
        "return_x_negative",
        "return_y_positive",
        "return_y_negative",
    )
)

# The gates are an explicit, caller-owned decision, judged against your criteria.
result = evaluate_hand_eye_quality(pairs, closures)
print("quality passed:", result.passed)
for check in result.failed_checks:
    print("  failed:", check)
