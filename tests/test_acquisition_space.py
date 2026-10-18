import json
from dataclasses import asdict
from math import sqrt

import pytest

import planner
from planner import (
    CORNERS,
    EXAMPLE_TRAJECTORIES,
    AcquisitionAxis,
    AcquisitionEnvelope,
    AcquisitionMetrics,
    AcquisitionPoint,
    AxisBand,
    AxisThreshold,
    ExampleTrajectory,
    FeasibilityReason,
    MeasurementLimits,
    MeasurementVerdict,
    OperatingCorner,
    SearchUtility,
    acquisition_space,
    assess_measurement,
    example_trajectories,
    metrics_of,
    rank_by_utility,
)


def _point(
    *,
    velocity: float = 10.0,
    exposure: float = 0.001,
    width: float = 10.0,
    frame_period: float = 0.01,
    pixel_size: float = 0.1,
    resolution: float = 0.1,
    height: float = 10.0,
) -> AcquisitionPoint:
    return AcquisitionPoint(
        velocity_um_s=velocity,
        exposure_s=exposure,
        width_um=width,
        frame_period_s=frame_period,
        pixel_size_um=pixel_size,
        resolution_um=resolution,
        height_um=height,
    )


_ENVELOPE = AcquisitionEnvelope(
    velocity_um_s=AxisThreshold(10.0, 100.0),
    exposure_s=AxisThreshold(0.01, 0.1),
    width_um=AxisThreshold(5.0, 50.0),
)


@pytest.mark.parametrize(
    ("pixels", "residence_ms", "observations"),
    [
        (32, 20.8, 2.08),
        (128, 83.2, 8.32),
        (512, 332.8, 33.28),
        (2048, 1331.2, 133.12),
    ],
)
def test_illustrative_roi_table(
    pixels: int, residence_ms: float, observations: float
) -> None:
    point = _point(
        velocity=100.0,
        exposure=0.004,
        width=pixels * 0.065,
        frame_period=0.01,
        pixel_size=0.065,
        resolution=0.065,
        height=133.12,
    )
    metrics = metrics_of(point)
    assert point.pixel_size_um == pytest.approx(0.065)
    assert point.width_um == pytest.approx(pixels * 0.065)
    assert metrics.residence_s == pytest.approx(residence_ms / 1000.0, rel=1e-3)
    assert metrics.observations == pytest.approx(observations, rel=1e-3)


@pytest.mark.parametrize("pixels", [32, 128, 512, 2048])
def test_residence_throughput_identity(pixels: int) -> None:
    point = _point(
        velocity=100.0,
        exposure=0.004,
        width=pixels * 0.065,
        frame_period=0.01,
        pixel_size=0.065,
        resolution=0.065,
        height=133.12,
    )
    metrics = metrics_of(point)
    assert metrics.residence_s is not None
    assert metrics.residence_s * metrics.area_throughput_um2_s == pytest.approx(
        point.width_um * point.height_um, rel=1e-12
    )
    assert metrics.area_throughput_um2_s == pytest.approx(
        point.velocity_um_s * point.height_um, rel=1e-12
    )


def test_beta_gamma_duty_and_overlap_relations() -> None:
    point = _point(
        velocity=12.5,
        exposure=0.002,
        width=8.0,
        frame_period=0.01,
        pixel_size=0.1,
        resolution=0.2,
        height=6.0,
    )
    metrics = point.metrics()
    assert metrics.motion_blur_um == pytest.approx(12.5 * 0.002)
    assert metrics.interframe_shift_um == pytest.approx(12.5 * 0.01)
    assert metrics.beta == pytest.approx(metrics.motion_blur_um / 0.2)
    assert metrics.gamma == pytest.approx(metrics.interframe_shift_um / 8.0)
    assert metrics.duty == pytest.approx(0.002 / 0.01)
    assert metrics.gamma is not None
    assert metrics.overlap == pytest.approx(1.0 / metrics.gamma)
    assert metrics.observations == metrics.overlap
    assert metrics.motion_blur_um / point.width_um == pytest.approx(
        metrics.gamma * metrics.duty
    )


def test_effective_sigma_includes_box_motion_kernel() -> None:
    point = _point(
        velocity=50.0,
        exposure=0.01,
        width=10.0,
        frame_period=0.02,
        pixel_size=0.1,
        resolution=0.2,
        height=5.0,
    )
    expected = sqrt(0.2**2 + (50.0 * 0.01) ** 2 / 12.0)
    metrics = point.metrics()
    assert metrics.effective_sigma_um == pytest.approx(expected)
    assert metrics.blur_ratio == pytest.approx(expected / 0.2)
    assert metrics.beta == pytest.approx(0.5 / 0.2)


def test_stationary_stage_returns_none_not_zero_or_infinity() -> None:
    point = _point(velocity=0.0, exposure=0.005, width=10.0, resolution=0.2)
    metrics = point.metrics()
    assert metrics.motion_blur_um == 0.0
    assert metrics.interframe_shift_um == 0.0
    assert metrics.residence_s is None
    assert metrics.observations is None
    assert metrics.gamma is None
    assert metrics.overlap is None
    assert metrics.area_throughput_um2_s == 0.0
    assert metrics.beta == 0.0
    assert metrics.duty == pytest.approx(0.5)
    assert metrics.effective_sigma_um == pytest.approx(0.2)
    assert metrics.blur_ratio == pytest.approx(1.0)
    assert metrics_of(point) == point.metrics()


def test_point_to_dict_uses_asdict() -> None:
    point = _point()
    assert point.to_dict() == asdict(point)


def test_point_validation_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        _point(velocity=-1.0)
    with pytest.raises(ValueError, match="positive"):
        _point(exposure=0.0)
    with pytest.raises(ValueError, match="positive"):
        _point(width=0.0)
    with pytest.raises(ValueError, match="positive"):
        _point(frame_period=0.0)
    with pytest.raises(ValueError, match="positive"):
        _point(pixel_size=0.0)
    with pytest.raises(ValueError, match="positive"):
        _point(resolution=0.0)
    with pytest.raises(ValueError, match="positive"):
        _point(height=-1.0)
    with pytest.raises(ValueError, match="finite"):
        _point(resolution=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        _point(height=float("inf"))
    with pytest.raises(ValueError, match="exceed"):
        _point(exposure=0.02, frame_period=0.01)


def test_axis_threshold_requires_positive_ordered_bounds() -> None:
    with pytest.raises(ValueError, match="positive"):
        AxisThreshold(0.0, 1.0)
    with pytest.raises(ValueError, match="less than"):
        AxisThreshold(1.0, 1.0)
    with pytest.raises(ValueError, match="less than"):
        AxisThreshold(2.0, 1.0)
    with pytest.raises(ValueError, match="finite"):
        AxisThreshold(float("nan"), 1.0)


def test_limits_validation_rejects_bad_bounds() -> None:
    with pytest.raises(ValueError, match="max_blur_ratio"):
        MeasurementLimits(max_blur_ratio=0.0, max_gamma=1.0)
    with pytest.raises(ValueError, match="max_gamma"):
        MeasurementLimits(max_blur_ratio=2.0, max_gamma=-1.0)
    with pytest.raises(ValueError, match="min_observations"):
        MeasurementLimits(max_blur_ratio=2.0, max_gamma=1.0, min_observations=0.0)
    with pytest.raises(ValueError, match="min_duty"):
        MeasurementLimits(max_blur_ratio=2.0, max_gamma=1.0, min_duty=-0.1)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        MeasurementLimits(max_blur_ratio=2.0, max_gamma=1.0, max_duty=1.1)
    with pytest.raises(ValueError, match="must not exceed"):
        MeasurementLimits(max_blur_ratio=2.0, max_gamma=1.0, min_duty=0.8, max_duty=0.2)


def test_metrics_validation_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        AcquisitionMetrics(
            motion_blur_um=0.0,
            interframe_shift_um=0.0,
            residence_s=None,
            observations=None,
            area_throughput_um2_s=0.0,
            beta=0.0,
            gamma=None,
            duty=float("nan"),
            overlap=None,
            effective_sigma_um=0.1,
            blur_ratio=1.0,
        )


_EXPECTED_TAXONOMY = {
    "000": (
        "Localized high-speed sampling",
        "Fast time series of a narrow region",
        "Not intrinsically line-confocal",
    ),
    "001": (
        "High-speed widefield timelapse",
        "Single-FOV live-cell dynamics",
        "Standard widefield regime",
    ),
    "010": (
        "Localized photon integration",
        "Long-exposure, narrow-ROI fluorescence measurement",
        "Not TDI if stage is stationary",
    ),
    "011": (
        "Widefield photon integration",
        "Low-signal stationary widefield imaging",
        "Intensity/SNR prioritized over sampling rate",
    ),
    "100": (
        "Motion-frozen push-broom scanning",
        "Continuous strip-based imaging",
        "Short exposure relative to displacement",
    ),
    "101": (
        "Motion-frozen panoramic sampling",
        "Rapid snapshot search or widefield panorama",
        "Search coverage may differ from gap-free mapping",
    ),
    "110": (
        "Motion-integrated strip imaging",
        "Registered multi-frame accumulation; deconvolution",
        "Digital TDI is a processing strategy, not the corner itself",
    ),
    "111": (
        "Motion-integrated panoramic imaging",
        "Coarse searching; computational deblurring",
        "Long-exposure blur can reduce target detectability",
    ),
}


def test_corners_are_complete_and_carry_documented_taxonomy() -> None:
    assert len(OperatingCorner) == 8
    assert {corner.value for corner in OperatingCorner} == {
        "000",
        "001",
        "010",
        "011",
        "100",
        "101",
        "110",
        "111",
    }
    assert set(CORNERS) == set(OperatingCorner)
    assert len(CORNERS) == 8
    for code, text in _EXPECTED_TAXONOMY.items():
        row = CORNERS[OperatingCorner(code)]
        assert row.corner is OperatingCorner(code)
        assert (row.regime, row.example, row.qualification) == text


def test_corner_digits_adjacency_and_changed_axes() -> None:
    corner = OperatingCorner("101")
    assert corner.digits() == (1, 0, 1)
    assert corner.digit(AcquisitionAxis.VELOCITY) == 1
    assert corner.digit(AcquisitionAxis.EXPOSURE) == 0
    assert corner.is_high(AcquisitionAxis.WIDTH)
    assert corner.is_low(AcquisitionAxis.EXPOSURE)
    assert corner.is_adjacent(OperatingCorner("001"))
    assert not corner.is_adjacent(OperatingCorner("011"))
    assert corner.hamming_distance(OperatingCorner("011")) == 2
    assert corner.changed_axes(OperatingCorner("001")) == (AcquisitionAxis.VELOCITY,)
    assert corner.changed_axes(OperatingCorner("011")) == (
        AcquisitionAxis.VELOCITY,
        AcquisitionAxis.EXPOSURE,
    )
    assert not OperatingCorner("000").is_adjacent(OperatingCorner("000"))


def test_example_trajectories_are_documented_corner_moves() -> None:
    trajectories = example_trajectories()
    assert trajectories is EXAMPLE_TRAJECTORIES
    assert len(trajectories) == 4
    assert [
        (trajectory.source.value, trajectory.target.value)
        for trajectory in trajectories
    ] == [("101", "001"), ("101", "011"), ("100", "110"), ("101", "111")]
    assert all(trajectory.intent for trajectory in trajectories)
    # The framework's 101 -> 011 example changes v and T_e; the others are
    # one-axis moves. Keep that documented exception visible rather than
    # pretending every example is Hamming-adjacent.
    assert [trajectory.is_adjacent for trajectory in trajectories] == [
        True,
        False,
        True,
        True,
    ]
    assert trajectories[1].changed_axes == (
        AcquisitionAxis.VELOCITY,
        AcquisitionAxis.EXPOSURE,
    )
    with pytest.raises(ValueError, match="different"):
        ExampleTrajectory(
            OperatingCorner.C101,
            OperatingCorner.C101,
            "same corner",
        )
    with pytest.raises(ValueError, match="intent"):
        ExampleTrajectory(OperatingCorner.C101, OperatingCorner.C001, " ")


def test_envelope_bands_include_low_and_high_boundaries() -> None:
    assert _ENVELOPE.band(AcquisitionAxis.VELOCITY, 5.0) is AxisBand.LOW
    assert _ENVELOPE.band(AcquisitionAxis.VELOCITY, 10.0) is AxisBand.LOW
    assert _ENVELOPE.band(AcquisitionAxis.VELOCITY, 50.0) is AxisBand.BETWEEN
    assert _ENVELOPE.band(AcquisitionAxis.VELOCITY, 100.0) is AxisBand.HIGH
    assert _ENVELOPE.band(AcquisitionAxis.VELOCITY, 200.0) is AxisBand.HIGH
    assert _ENVELOPE.band(AcquisitionAxis.EXPOSURE, 0.001) is AxisBand.LOW
    assert _ENVELOPE.band(AcquisitionAxis.EXPOSURE, 0.05) is AxisBand.BETWEEN
    assert _ENVELOPE.band(AcquisitionAxis.EXPOSURE, 0.2) is AxisBand.HIGH
    assert _ENVELOPE.band(AcquisitionAxis.WIDTH, 1.0) is AxisBand.LOW
    assert _ENVELOPE.band(AcquisitionAxis.WIDTH, 10.0) is AxisBand.BETWEEN
    assert _ENVELOPE.band(AcquisitionAxis.WIDTH, 100.0) is AxisBand.HIGH


@pytest.mark.parametrize(
    "code", ["000", "001", "010", "011", "100", "101", "110", "111"]
)
def test_envelope_classifies_each_corner(code: str) -> None:
    point = _point(
        velocity=200.0 if code[0] == "1" else 1.0,
        exposure=0.2 if code[1] == "1" else 0.001,
        width=100.0 if code[2] == "1" else 1.0,
        frame_period=0.5,
    )
    assert _ENVELOPE.corner(point) is OperatingCorner(code)


def test_envelope_returns_none_when_any_axis_is_between() -> None:
    point = _point(velocity=50.0, exposure=0.001, width=1.0, frame_period=0.5)
    assert _ENVELOPE.corner(point) is None


def test_blur_ceiling_rejects_only_blur() -> None:
    point = _point(velocity=10.0, exposure=0.01, resolution=0.01)
    limits = MeasurementLimits(max_blur_ratio=2.0, max_gamma=10.0)
    verdict = assess_measurement(point, limits)
    assert verdict.feasible is False
    assert verdict.reasons == (FeasibilityReason.BLUR,)


def test_gap_ceiling_rejects_only_gap() -> None:
    point = _point(
        velocity=10.0,
        exposure=0.0001,
        width=1.0,
        frame_period=0.1,
        resolution=0.1,
    )
    limits = MeasurementLimits(max_blur_ratio=1.1, max_gamma=0.5)
    verdict = assess_measurement(point, limits)
    assert verdict.feasible is False
    assert verdict.reasons == (FeasibilityReason.GAP,)


def test_minimum_observations_rejects_only_too_few() -> None:
    point = _point(velocity=1.0, exposure=0.001, width=10.0, resolution=0.1)
    limits = MeasurementLimits(
        max_blur_ratio=1.1, max_gamma=1.0, min_observations=2000.0
    )
    verdict = assess_measurement(point, limits)
    assert verdict.feasible is False
    assert verdict.reasons == (FeasibilityReason.TOO_FEW_OBSERVATIONS,)


def test_duty_bounds_reject_only_duty() -> None:
    point = _point(
        velocity=1.0,
        exposure=0.05,
        width=10.0,
        frame_period=0.1,
        resolution=0.1,
    )
    limits = MeasurementLimits(
        max_blur_ratio=1.1,
        max_gamma=1.0,
        min_duty=0.6,
        max_duty=0.9,
    )
    verdict = assess_measurement(point, limits)
    assert verdict.feasible is False
    assert verdict.reasons == (FeasibilityReason.DUTY,)


def test_stationary_stage_skips_velocity_defined_checks() -> None:
    point = _point(velocity=0.0, exposure=0.005, width=10.0, resolution=0.1)
    limits = MeasurementLimits(
        max_blur_ratio=1.0,
        max_gamma=0.001,
        min_observations=1000.0,
        min_duty=0.4,
        max_duty=0.6,
    )
    verdict = assess_measurement(point, limits)
    assert verdict.feasible is True
    assert verdict.reasons == ()
    assert verdict.metrics.gamma is None
    assert verdict.metrics.observations is None


def test_feasibility_limits_are_inclusive_at_the_boundary() -> None:
    point = _point(velocity=10.0, exposure=0.002, width=10.0, resolution=0.2)
    metrics = point.metrics()
    assert metrics.gamma is not None
    assert metrics.observations is not None
    limits = MeasurementLimits(
        max_blur_ratio=metrics.blur_ratio,
        max_gamma=metrics.gamma,
        min_observations=metrics.observations,
        min_duty=metrics.duty,
        max_duty=metrics.duty,
    )
    assert assess_measurement(point, limits).feasible is True


def test_feasibility_accumulates_every_violated_reason() -> None:
    point = _point(
        velocity=100.0,
        exposure=0.01,
        width=0.5,
        frame_period=0.01,
        resolution=0.1,
        height=1.0,
    )
    limits = MeasurementLimits(
        max_blur_ratio=2.0,
        max_gamma=1.0,
        min_observations=2.0,
        min_duty=0.1,
        max_duty=0.8,
    )
    verdict = assess_measurement(point, limits)
    assert verdict.feasible is False
    assert verdict.reasons == (
        FeasibilityReason.BLUR,
        FeasibilityReason.GAP,
        FeasibilityReason.TOO_FEW_OBSERVATIONS,
        FeasibilityReason.DUTY,
    )


def test_verdict_rejects_inconsistent_state() -> None:
    metrics = _point().metrics()
    with pytest.raises(ValueError, match="exactly"):
        MeasurementVerdict(True, (FeasibilityReason.BLUR,), metrics)
    with pytest.raises(ValueError, match="repeat"):
        MeasurementVerdict(
            False,
            (FeasibilityReason.BLUR, FeasibilityReason.BLUR),
            metrics,
        )


def test_rank_by_utility_orders_feasible_and_keeps_rejected() -> None:
    low = _point(velocity=50.0, exposure=0.0001, resolution=1.0, height=1.0)
    middle = _point(velocity=100.0, exposure=0.0001, resolution=1.0, height=1.0)
    high = _point(velocity=200.0, exposure=0.0001, resolution=1.0, height=1.0)
    rejected = _point(
        velocity=1000.0,
        exposure=0.01,
        frame_period=0.01,
        resolution=0.1,
        height=10.0,
    )
    limits = MeasurementLimits(max_blur_ratio=2.0, max_gamma=1.0)
    ranked = rank_by_utility([low, rejected, high, middle], SearchUtility(), limits)
    assert len(ranked) == 4
    assert [scored.point for scored in ranked] == [high, middle, low, rejected]
    assert [scored.score for scored in ranked] == [200.0, 100.0, 50.0, 10000.0]
    assert [scored.feasible for scored in ranked] == [True, True, True, False]
    assert ranked[-1].verdict.reasons == (FeasibilityReason.BLUR,)


def test_rank_without_limits_keeps_all_points_ranked() -> None:
    low = _point(velocity=50.0, height=1.0)
    high = _point(velocity=200.0, height=1.0)
    rejected_with_limits = _point(
        velocity=1000.0, exposure=0.01, resolution=0.1, height=10.0
    )
    ranked = rank_by_utility([low, rejected_with_limits, high], SearchUtility())
    assert all(scored.feasible for scored in ranked)
    assert [scored.point for scored in ranked] == [
        rejected_with_limits,
        high,
        low,
    ]


def test_rank_preserves_rejected_input_order() -> None:
    first = _point(velocity=1000.0, exposure=0.01, resolution=0.1, height=10.0)
    second = _point(velocity=2000.0, exposure=0.01, resolution=0.1, height=20.0)
    feasible = _point(velocity=10.0, height=1.0)
    limits = MeasurementLimits(max_blur_ratio=2.0, max_gamma=1.0)
    ranked = rank_by_utility([first, feasible, second], SearchUtility(), limits)
    assert ranked[0].point is feasible
    assert [scored.point for scored in ranked[1:]] == [first, second]
    assert all(not scored.feasible for scored in ranked[1:])


def test_rank_accepts_a_structural_utility() -> None:
    class WidthUtility:
        def score(self, point: AcquisitionPoint, metrics: AcquisitionMetrics) -> float:
            return point.width_um

    narrow = _point(velocity=10.0, width=2.0)
    wide = _point(velocity=10.0, width=20.0)
    ranked = rank_by_utility([narrow, wide], WidthUtility())
    assert [scored.point for scored in ranked] == [wide, narrow]


def test_rank_rejects_nonfinite_utility_scores() -> None:
    class NanUtility:
        def score(self, point: AcquisitionPoint, metrics: AcquisitionMetrics) -> float:
            return float("nan")

    with pytest.raises(ValueError, match="utility score"):
        rank_by_utility([_point()], NanUtility())


def test_records_round_trip_through_json_without_nan_or_infinity() -> None:
    moving = _point(velocity=100.0, exposure=0.004, resolution=0.065, height=133.12)
    stationary = _point(velocity=0.0, exposure=0.005, width=10.0, resolution=0.2)
    blur_point = _point(velocity=10.0, exposure=0.01, resolution=0.01)
    blur_limits = MeasurementLimits(max_blur_ratio=2.0, max_gamma=10.0)
    ranked = rank_by_utility(
        [moving, stationary],
        SearchUtility(),
        MeasurementLimits(max_blur_ratio=10.0, max_gamma=10.0),
    )
    payload = {
        "point": moving.to_dict(),
        "metrics": metrics_of(moving).to_dict(),
        "stationary": metrics_of(stationary).to_dict(),
        "verdict": assess_measurement(blur_point, blur_limits).to_dict(),
        "ranked": [scored.to_dict() for scored in ranked],
    }
    encoded = json.dumps(payload, allow_nan=False)
    assert "NaN" not in encoded
    assert "Infinity" not in encoded
    decoded = json.loads(encoded)
    assert decoded == payload
    assert decoded["stationary"]["residence_s"] is None
    assert decoded["stationary"]["observations"] is None
    assert decoded["stationary"]["gamma"] is None
    assert decoded["stationary"]["overlap"] is None
    assert decoded["verdict"]["reasons"] == ["blur"]


def test_public_surface_reexports_every_model_symbol() -> None:
    assert len(set(acquisition_space.__all__)) == len(acquisition_space.__all__)
    assert set(acquisition_space.__all__) <= set(planner.__all__)
    assert len(set(planner.__all__)) == len(planner.__all__)
    for name in acquisition_space.__all__:
        assert getattr(planner, name) is getattr(acquisition_space, name)
