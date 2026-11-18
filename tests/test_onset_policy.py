"""Hardware-free tests for the causal band-onset policy.

Every case feeds the policy a synthetic sensor sequence or a hand-written
density trace. Nothing here reads a recorded frame or touches a device, so a
pass establishes the calculation only.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from planner import (
    BandRows,
    OnsetObservation,
    OnsetPolicy,
    OnsetPolicyConfig,
    band_rows,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from planner import OnsetProposal

_HEIGHT = 120
_WIDTH = 40


def _bar_frames(
    *,
    direction: str,
    steps: int = 30,
    speed_rows: float = 8.0,
    starts: Sequence[float] = (-60.0,),
    half_width: float = 10.0,
    background: float = 100.0,
    amplitude: float = 500.0,
    drift_per_frame: float = 0.0,
) -> list[np.ndarray]:
    """Return frames with bright bars crossing the sensor in the travel direction.

    Content moves toward increasing image rows for increasing stage x, so an
    increasing-x target starts above the sensor and travels down; a
    decreasing-x target does the reverse.
    """
    rows = np.arange(_HEIGHT, dtype=float)
    frames = []
    for index in range(steps):
        frame = np.full(
            (_HEIGHT, _WIDTH), background + drift_per_frame * index, dtype=float
        )
        for start in starts:
            if direction == "increasing_x":
                centre = start + speed_rows * index
            else:
                centre = _HEIGHT - 1.0 - start - speed_rows * index
            frame[np.abs(rows - centre) <= half_width, :] += amplitude
        frames.append(frame)
    return frames


def _run_dual_band(
    config: OnsetPolicyConfig, frames: Sequence[np.ndarray]
) -> list[OnsetProposal]:
    """Feed both declared bands of every synthetic frame through one policy."""
    geometry = config.band_rows
    leading = [
        float(frame[geometry.leading[0] : geometry.leading[1]].mean())
        for frame in frames
    ]
    trailing = [
        float(frame[geometry.trailing[0] : geometry.trailing[1]].mean())
        for frame in frames
    ]
    policy = OnsetPolicy(config)
    return [
        policy.update(OnsetObservation(index, lead, trail))
        for index, (lead, trail) in enumerate(zip(leading, trailing))
    ]


def _run_whole_frame(
    config: OnsetPolicyConfig, frames: Sequence[np.ndarray]
) -> list[OnsetProposal]:
    """Feed each synthetic frame's whole-frame mean through one policy."""
    policy = OnsetPolicy(config)
    return [
        policy.update(OnsetObservation(index, float(frame.mean())))
        for index, frame in enumerate(frames)
    ]


def _whole_frame_config(**overrides: object) -> OnsetPolicyConfig:
    """Return a whole-frame policy config with the test defaults."""
    values: dict[str, object] = {
        "trigger": "whole_frame",
        "release": "signal_fall",
        "coarse_step_um": 10.0,
        "dense_step_um": 2.0,
        "dense_budget_um": 120.0,
    }
    values.update(overrides)
    return OnsetPolicyConfig(**values)  # type: ignore[arg-type]


def _config(**overrides: object) -> OnsetPolicyConfig:
    """Return a band policy config sized to the synthetic sensor."""
    values: dict[str, object] = {"sensor_height_px": _HEIGHT}
    values.update(overrides)
    return OnsetPolicyConfig(**values)  # type: ignore[arg-type]


def test_band_rows_are_direction_aware_and_match_the_measured_strip() -> None:
    increasing = band_rows(1200, 0.17, "increasing_x")
    decreasing = band_rows(1200, 0.17, "decreasing_x")

    assert increasing == BandRows(leading=(0, 204), trailing=(996, 1200))
    assert decreasing == BandRows(leading=(996, 1200), trailing=(0, 204))
    # The 2026-10-26 strip ran with decreasing x and fired on the high-row band
    # first, which is the leading edge only under the decreasing layout.
    assert decreasing.leading[0] == 996

    half = band_rows(1000, 0.5, "increasing_x")
    assert half == BandRows(leading=(0, 500), trailing=(500, 1000))

    for height, fraction in ((0, 0.17), (1200, 0.0), (1200, 0.75), (120, -0.1)):
        with pytest.raises(ValueError):
            band_rows(height, fraction, "increasing_x")
    with pytest.raises(ValueError):
        band_rows(1200, 0.17, "sideways")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        BandRows(leading=(10, 10), trailing=(0, 5))
    with pytest.raises(ValueError):
        BandRows(leading=(-1, 10), trailing=(0, 5))


@pytest.mark.parametrize("direction", ["increasing_x", "decreasing_x"])
def test_each_direction_attacks_and_releases_on_its_own_bands(direction: str) -> None:
    config = _config(direction=direction)
    proposals = _run_dual_band(config, _bar_frames(direction=direction))

    onsets = [index for index, item in enumerate(proposals) if item.onset]
    releases = [index for index, item in enumerate(proposals) if item.release]
    assert len(onsets) == 1
    assert len(releases) == 1
    assert onsets[0] < releases[0]
    assert proposals[onsets[0]].state == "dense"
    assert proposals[onsets[0]].step_um == config.dense_step_um
    assert proposals[releases[0]].step_um == config.coarse_step_um
    assert proposals[releases[0]].reason == "released_by_trailing_band"

    before = proposals[: onsets[0]]
    after = proposals[releases[0] + 1 :]
    assert all(item.reason == "advance_coarse" for item in before)
    assert all(item.reason == "advance_coarse" for item in after)
    assert all(item.step_um == config.coarse_step_um for item in before + after)

    dense = [item for item in proposals if item.reason == "advance_dense"]
    assert len(dense) >= 3
    assert all(item.step_um == config.dense_step_um for item in dense)
    assert proposals[-1].dense_used_um == len(dense) * config.dense_step_um
    assert proposals[-1].dense_remaining_um == pytest.approx(
        config.dense_budget_um - len(dense) * config.dense_step_um
    )


def test_the_wrong_direction_layout_does_not_fire_on_the_same_frames() -> None:
    frames = _bar_frames(direction="increasing_x", steps=20)
    correct = _run_dual_band(_config(direction="increasing_x"), frames)
    wrong = _run_dual_band(_config(direction="decreasing_x"), frames)

    assert any(item.onset for item in correct[:16])
    # Under the wrong layout the band the target reaches first is the far edge,
    # so nothing can fire before the target has crossed the whole sensor.
    assert not any(item.onset for item in wrong[:16])


def test_isolated_targets_produce_one_window_each() -> None:
    config = _config(dense_budget_um=1000.0)
    # Two targets separated by more than one sensor height travel the same pass.
    frames = _bar_frames(direction="increasing_x", steps=90, starts=(-60.0, -260.0))
    proposals = _run_dual_band(config, frames)

    onsets = [index for index, item in enumerate(proposals) if item.onset]
    releases = [index for index, item in enumerate(proposals) if item.release]
    assert len(onsets) == 2
    assert len(releases) == 2
    assert onsets[0] < releases[0] < onsets[1] < releases[1]
    assert proposals[releases[0]].state == "approach"


def test_adjacent_targets_keep_one_merged_window() -> None:
    config = _config()
    # A five-row gap between two bars is narrower than the band depth, so the
    # leading band never falls back to baseline between them.
    frames = _bar_frames(direction="increasing_x", steps=30, starts=(-60.0, -85.0))
    proposals = _run_dual_band(config, frames)

    assert sum(item.onset for item in proposals) == 1
    assert sum(item.release for item in proposals) == 1


@pytest.mark.parametrize(
    ("background", "amplitude"),
    [(100.0, 0.0), (3000.0, 0.0)],
    ids=["flat-dark", "flat-bright"],
)
def test_flat_negatives_never_trigger(background: float, amplitude: float) -> None:
    config = _config()
    frames = _bar_frames(
        direction="increasing_x", steps=40, background=background, amplitude=amplitude
    )
    proposals = _run_dual_band(config, frames)

    assert not any(item.onset for item in proposals)
    assert not any(item.release for item in proposals)
    assert {item.reason for item in proposals} == {"advance_coarse"}
    assert proposals[-1].state == "approach"


def test_slow_baseline_drift_does_not_fake_an_onset() -> None:
    config = _config(dense_budget_um=1000.0)
    # 0.5 % per frame is far below the 20 % enter threshold, and the trailing
    # median tracks it, so only the real bar may fire.
    frames = _bar_frames(
        direction="increasing_x", steps=90, starts=(-260.0,), drift_per_frame=0.5
    )
    proposals = _run_dual_band(config, frames)

    assert sum(item.onset for item in proposals) == 1
    onset = next(index for index, item in enumerate(proposals) if item.onset)
    assert onset > 20


def test_missing_observations_hold_then_block_and_recover() -> None:
    config = _config(max_missing_observations=3)
    frames = _bar_frames(direction="increasing_x", steps=60, starts=(-260.0,))
    geometry = config.band_rows
    leading = [
        float(frame[geometry.leading[0] : geometry.leading[1]].mean())
        for frame in frames
    ]
    trailing = [
        float(frame[geometry.trailing[0] : geometry.trailing[1]].mean())
        for frame in frames
    ]

    policy = OnsetPolicy(config)
    proposals = []
    for index, (lead, trail) in enumerate(zip(leading, trailing)):
        if 3 <= index <= 8:
            proposals.append(policy.update(OnsetObservation(index, None)))
        else:
            proposals.append(policy.update(OnsetObservation(index, lead, trail)))

    held = proposals[3:9]
    assert [item.reason for item in held[:3]] == ["hold_missing_observations"] * 3
    assert [item.reason for item in held[3:]] == ["blocked_missing_observations"] * 3
    assert all(item.step_um == 0.0 for item in held)
    assert all(item.state == "approach" for item in held)
    assert all(item.dense_used_um == 0.0 for item in held)
    assert sum(item.onset for item in proposals) == 1
    assert next(index for index, item in enumerate(proposals) if item.onset) > 30

    # A missing trailing band is equally a hold while the policy is dense.
    dense_config = _config()
    dense_policy = OnsetPolicy(dense_config)
    geometry = dense_config.band_rows
    held_while_dense = False
    for index, frame in enumerate(frames):
        lead = float(frame[geometry.leading[0] : geometry.leading[1]].mean())
        trail = float(frame[geometry.trailing[0] : geometry.trailing[1]].mean())
        was_dense = dense_policy.state == "dense"
        proposal = dense_policy.update(
            OnsetObservation(index, lead, None if was_dense else trail)
        )
        if was_dense:
            assert proposal.step_um == 0.0
            assert proposal.reason in {
                "hold_missing_observations",
                "blocked_missing_observations",
            }
            held_while_dense = True
            break
    assert held_while_dense


def test_dense_budget_is_cumulative_and_persists_after_exhaustion() -> None:
    config = _config(dense_budget_um=10.0, dense_step_um=2.0)
    proposals = _run_dual_band(
        config, _bar_frames(direction="increasing_x", speed_rows=6.0)
    )

    dense = [item for item in proposals if item.reason == "advance_dense"]
    assert len(dense) == 5
    assert proposals[-1].dense_used_um == pytest.approx(10.0)
    assert proposals[-1].dense_remaining_um == pytest.approx(0.0)

    exhausted = [item for item in proposals if item.budget_exhausted]
    assert len(exhausted) == 1
    assert exhausted[0].reason == "dense_budget_exhausted"
    assert exhausted[0].state == "exit"
    assert exhausted[0].step_um == config.coarse_step_um

    after = proposals[proposals.index(exhausted[0]) + 1 :]
    assert after
    assert all(item.state == "exit" for item in after)
    assert all(item.reason == "after_dense_budget_exhausted" for item in after)
    assert not any(item.step_um == config.dense_step_um for item in after)


def test_zero_budget_exhausts_on_the_onset_without_a_dense_step() -> None:
    config = _config(dense_budget_um=0.0)
    proposals = _run_dual_band(config, _bar_frames(direction="increasing_x"))

    exhausted = [item for item in proposals if item.budget_exhausted]
    assert len(exhausted) == 1
    assert exhausted[0].dense_used_um == 0.0
    assert not any(item.reason == "advance_dense" for item in proposals)


def test_stepped_and_continuous_requests_stay_distinct() -> None:
    frames = _bar_frames(direction="increasing_x")
    stepped_config = _config()
    continuous_config = _config(mode="continuous", frame_period_s=0.5)
    stepped = _run_dual_band(stepped_config, frames)
    continuous = _run_dual_band(continuous_config, frames)

    assert all(item.mode == "stepped" and item.step_um is not None for item in stepped)
    assert all(item.speed_um_s is None for item in stepped)
    assert all(
        item.mode == "continuous" and item.speed_um_s is not None for item in continuous
    )
    assert all(item.step_um is None for item in continuous)
    assert [item.state for item in stepped] == [item.state for item in continuous]
    assert [item.reason for item in stepped] == [item.reason for item in continuous]
    assert continuous[0].speed_um_s == pytest.approx(
        stepped[0].step_um / continuous_config.frame_period_s  # type: ignore[operator]
    )


def test_whole_frame_mode_attacks_on_the_rise_and_finishes_on_the_fall() -> None:
    config = _whole_frame_config()
    proposals = _run_whole_frame(
        config, _bar_frames(direction="decreasing_x", steps=40)
    )

    onsets = [index for index, item in enumerate(proposals) if item.onset]
    releases = [index for index, item in enumerate(proposals) if item.release]
    assert len(onsets) == 1
    assert len(releases) == 1
    assert onsets[0] < releases[0]
    assert proposals[releases[0]].reason == "released_by_signal_fall"
    assert proposals[releases[0]].step_um == config.coarse_step_um
    with pytest.raises(ValueError):
        _ = config.band_rows


def test_policy_identity_is_deterministic_and_covers_the_thresholds() -> None:
    base = OnsetPolicyConfig()
    same = OnsetPolicyConfig()
    retuned = OnsetPolicyConfig(dense_step_um=3.0)

    assert base.policy_id == same.policy_id
    assert base.policy_id != retuned.policy_id
    assert base.policy_id.startswith("onset-v1-")
    payload = base.as_dict()
    assert payload["schema"] == "planner-onset-policy-v1"
    assert json.loads(json.dumps(payload)) == payload
    assert payload["band_rows"] == {"leading": [0, 204], "trailing": [996, 1200]}
    assert _config().as_dict()["band_rows"] == {
        "leading": [0, 20],
        "trailing": [100, 120],
    }

    proposals = _run_dual_band(
        _config(), _bar_frames(direction="increasing_x", steps=20)
    )
    record = proposals[-1].to_dict()
    assert record["policy_id"] == _config().policy_id
    assert json.loads(json.dumps(record)) == record
    assert record["schema"] == "planner-onset-policy-v1"


def test_config_and_observation_validation() -> None:
    with pytest.raises(ValueError, match="requires release"):
        OnsetPolicyConfig(trigger="whole_frame")
    with pytest.raises(ValueError, match="requires release"):
        OnsetPolicyConfig(release="signal_fall")
    with pytest.raises(ValueError, match="sensor_height_px"):
        OnsetPolicyConfig(sensor_height_px=None)
    with pytest.raises(ValueError, match="frame_period_s is required"):
        OnsetPolicyConfig(mode="continuous")
    with pytest.raises(ValueError, match="frame_period_s belongs"):
        OnsetPolicyConfig(frame_period_s=0.5)
    with pytest.raises(ValueError, match="dense_step_um"):
        OnsetPolicyConfig(dense_step_um=20.0)
    with pytest.raises(ValueError, match="dense_budget_um"):
        OnsetPolicyConfig(dense_budget_um=-1.0)
    with pytest.raises(ValueError, match="band_fraction"):
        OnsetPolicyConfig(band_fraction=0.7)
    with pytest.raises(TypeError, match="IntensityRiseConfig"):
        OnsetPolicyConfig(rise=object())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="frame_index"):
        OnsetObservation(-1, 1.0)
    with pytest.raises(ValueError, match="frame_density"):
        OnsetObservation(0, float("nan"))
    with pytest.raises(TypeError, match="frame_density"):
        OnsetObservation(0, True)
    with pytest.raises(TypeError, match="OnsetPolicyConfig"):
        OnsetPolicy(config=object())  # type: ignore[arg-type]

    policy = OnsetPolicy(OnsetPolicyConfig())
    policy.update(OnsetObservation(4, 100.0, 100.0))
    with pytest.raises(ValueError, match="strictly increasing"):
        policy.update(OnsetObservation(4, 100.0, 100.0))
    with pytest.raises(TypeError, match="OnsetObservation"):
        policy.update(object())  # type: ignore[arg-type]
