from __future__ import annotations

import pytest

from planner.speed_policy import (
    SpeedBlendConfig,
    SpeedBlendController,
    blend_intensity_to_speed,
    blend_interest,
)


def test_intensity_blends_from_fast_search_to_slow_imaging() -> None:
    curve = blend_intensity_to_speed(
        [0.05, 0.15, 0.25],
        [1.0, 1.0, 1.0],
        slow_speed_um_s=50,
        fast_speed_um_s=100,
        sample_interval_s=0.02,
        config=SpeedBlendConfig(smoothing_frames=1, max_acceleration_um_s2=10000),
    )
    assert curve.blend_fraction == pytest.approx((0, 0.5, 1))
    assert curve.desired_speed_um_s == pytest.approx((100, 75, 50))
    assert curve.slew_limited_speed_um_s == pytest.approx((50, 75, 50))


def test_filter_is_causal_and_limits_speed_change_per_sample() -> None:
    policy = SpeedBlendConfig(smoothing_frames=3, max_acceleration_um_s2=1000)
    prefix = blend_intensity_to_speed(
        [0.02, 0.12],
        [1, 1],
        slow_speed_um_s=50,
        fast_speed_um_s=100,
        sample_interval_s=0.02,
        config=policy,
    )
    extended = blend_intensity_to_speed(
        [0.02, 0.12, 1, 0],
        [1, 1, 1, 1],
        slow_speed_um_s=50,
        fast_speed_um_s=100,
        sample_interval_s=0.02,
        config=policy,
    )
    assert extended.smoothed_interest[:2] == prefix.smoothed_interest
    assert extended.desired_speed_um_s[:2] == prefix.desired_speed_um_s
    max_delta = policy.max_acceleration_um_s2 * 0.02
    assert all(
        abs(right - left) <= max_delta + 1e-12
        for left, right in zip(
            extended.slew_limited_speed_um_s,
            extended.slew_limited_speed_um_s[1:],
        )
    )


def test_low_motion_confidence_caps_desired_speed_at_zero() -> None:
    curve = blend_intensity_to_speed(
        [0.0, 0.0, 0.0],
        [1.0, 0.1, 0.1],
        slow_speed_um_s=50,
        fast_speed_um_s=100,
        sample_interval_s=0.1,
        config=SpeedBlendConfig(smoothing_frames=1, max_acceleration_um_s2=500),
    )
    assert curve.desired_speed_um_s == pytest.approx((100, 0, 0))
    assert curve.slew_limited_speed_um_s == pytest.approx((50, 0, 0))


def test_explicit_initial_speed_supports_fast_search_start() -> None:
    curve = blend_intensity_to_speed(
        [0.0, 0.3],
        [1.0, 1.0],
        slow_speed_um_s=50,
        fast_speed_um_s=200,
        sample_interval_s=0.02,
        config=SpeedBlendConfig(smoothing_frames=1, max_acceleration_um_s2=10_000),
        initial_speed_um_s=200,
    )
    assert curve.slew_limited_speed_um_s == pytest.approx((200, 50))


@pytest.mark.parametrize("initial_speed", [None, 200.0])
def test_streaming_controller_matches_batch_curve_for_every_prefix(
    initial_speed: float | None,
) -> None:
    config = SpeedBlendConfig(
        smoothing_frames=3,
        minimum_motion_confidence=0.25,
        max_acceleration_um_s2=1000,
    )
    interests = [0.02, 0.12, 0.7, 0.9, 0.3, 0.0]
    confidences = [1.0, 1.0, 0.8, 0.0, 0.5, 1.0]
    controller = SpeedBlendController(
        slow_speed_um_s=50,
        fast_speed_um_s=200,
        sample_interval_s=0.02,
        config=config,
        initial_speed_um_s=initial_speed,
    )

    for index in range(len(interests)):
        proposal = controller.update(interests[index], confidences[index])
        prefix = blend_intensity_to_speed(
            interests[: index + 1],
            confidences[: index + 1],
            slow_speed_um_s=50,
            fast_speed_um_s=200,
            sample_interval_s=0.02,
            config=config,
            initial_speed_um_s=initial_speed,
        )
        assert proposal.smoothed_interest == pytest.approx(prefix.smoothed_interest[-1])
        assert proposal.blend_fraction == pytest.approx(prefix.blend_fraction[-1])
        assert proposal.desired_speed_um_s == pytest.approx(
            prefix.desired_speed_um_s[-1]
        )
        assert proposal.slew_limited_speed_um_s == pytest.approx(
            prefix.slew_limited_speed_um_s[-1]
        )


def test_streaming_controller_rejects_invalid_observations() -> None:
    controller = SpeedBlendController(
        slow_speed_um_s=50,
        fast_speed_um_s=200,
        sample_interval_s=0.02,
    )
    with pytest.raises(ValueError, match="interest values"):
        controller.update(float("nan"))
    with pytest.raises(ValueError, match="motion_confidence"):
        controller.update(0.1, 1.1)


def test_inputs_and_policy_settings_are_validated() -> None:
    with pytest.raises(ValueError, match="lengths must match"):
        blend_intensity_to_speed(
            [0.1], [], slow_speed_um_s=1, fast_speed_um_s=2, sample_interval_s=0.1
        )
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        blend_interest([1.1])
    with pytest.raises(ValueError, match="thresholds"):
        SpeedBlendConfig(interest_fast_threshold=0.3, interest_slow_threshold=0.2)
