"""Pure causal intensity-to-speed blending for sample-search trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True)
class SpeedBlendConfig:
    """Policy thresholds and virtual slew limits, expressed in physical units."""

    interest_fast_threshold: float = 0.05
    interest_slow_threshold: float = 0.25
    smoothing_frames: int = 3
    minimum_motion_confidence: float = 0.25
    max_acceleration_um_s2: float = 5000.0

    def __post_init__(self) -> None:
        """Check the declared fields for consistency."""
        if not (
            isfinite(self.interest_fast_threshold)
            and isfinite(self.interest_slow_threshold)
            and 0 <= self.interest_fast_threshold < self.interest_slow_threshold <= 1
        ):
            raise ValueError("interest thresholds must satisfy 0 <= fast < slow <= 1")
        if (
            isinstance(self.smoothing_frames, bool)
            or not isinstance(self.smoothing_frames, int)
            or self.smoothing_frames < 1
        ):
            raise ValueError("smoothing_frames must be a positive integer")
        if not (
            isfinite(self.minimum_motion_confidence)
            and 0 <= self.minimum_motion_confidence <= 1
        ):
            raise ValueError("minimum_motion_confidence must be in [0, 1]")
        if not (
            isfinite(self.max_acceleration_um_s2) and self.max_acceleration_um_s2 > 0
        ):
            raise ValueError("max_acceleration_um_s2 must be finite and positive")


@dataclass(frozen=True)
class SpeedCurve:
    """Causal smoothed interest and requested/slew-limited speed samples."""

    smoothed_interest: tuple[float, ...]
    blend_fraction: tuple[float, ...]
    desired_speed_um_s: tuple[float, ...]
    slew_limited_speed_um_s: tuple[float, ...]


_DEFAULT_SPEED_BLEND_CONFIG = SpeedBlendConfig()


def _finite_values(values: Iterable[float], label: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if any(not isfinite(value) for value in result):
        raise ValueError(f"{label} values must be finite")
    return result


def _causal_mean(values: tuple[float, ...], window: int) -> tuple[float, ...]:
    smoothed: list[float] = []
    for index, value in enumerate(values):
        start = max(0, index - window + 1)
        smoothed.append(sum(values[start : index + 1]) / (index - start + 1))
    return tuple(smoothed)


def blend_interest(
    interest: Iterable[float], config: SpeedBlendConfig = _DEFAULT_SPEED_BLEND_CONFIG
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return causal smoothed intensity and normalized slow-mode blend."""
    raw = _finite_values(interest, "interest")
    if any(not 0 <= value <= 1 for value in raw):
        raise ValueError("interest values must be in [0, 1]")
    smoothed = _causal_mean(raw, config.smoothing_frames)
    span = config.interest_slow_threshold - config.interest_fast_threshold
    blend = tuple(
        min(1.0, max(0.0, (value - config.interest_fast_threshold) / span))
        for value in smoothed
    )
    return smoothed, blend


def blend_intensity_to_speed(
    interest: Iterable[float],
    motion_confidence: Iterable[float],
    *,
    slow_speed_um_s: float,
    fast_speed_um_s: float,
    sample_interval_s: float,
    config: SpeedBlendConfig = _DEFAULT_SPEED_BLEND_CONFIG,
    initial_speed_um_s: float | None = None,
) -> SpeedCurve:
    """Map causal intensity evidence to speed with a bounded slew model.

    Low interest selects fast search; increasing interest smoothly transitions
    to slow imaging. Low motion confidence sets the desired speed to zero.
    ``max_acceleration_um_s2`` limits sample-to-sample speed changes in this
    mathematical policy only. Device acceleration and braking limits must be
    measured and enforced by the instrument adapter.
    """
    raw = _finite_values(interest, "interest")
    confidence = _finite_values(motion_confidence, "motion_confidence")
    if len(raw) != len(confidence):
        raise ValueError("interest and motion_confidence lengths must match")
    if any(not 0 <= value <= 1 for value in raw):
        raise ValueError("interest values must be in [0, 1]")
    if any(not 0 <= value <= 1 for value in confidence):
        raise ValueError("motion_confidence values must be in [0, 1]")
    if not (
        isfinite(slow_speed_um_s)
        and isfinite(fast_speed_um_s)
        and 0 < slow_speed_um_s <= fast_speed_um_s
    ):
        raise ValueError("speeds must be finite and satisfy 0 < slow <= fast")
    if not isfinite(sample_interval_s) or sample_interval_s <= 0:
        raise ValueError("sample_interval_s must be finite and positive")
    if initial_speed_um_s is not None and (
        not isfinite(initial_speed_um_s)
        or not 0 <= initial_speed_um_s <= fast_speed_um_s
    ):
        raise ValueError("initial_speed_um_s must be finite and within [0, fast_speed]")

    smoothed, blend = blend_interest(raw, config)
    desired = tuple(
        0.0
        if confidence[index] < config.minimum_motion_confidence
        else fast_speed_um_s
        - (fast_speed_um_s - slow_speed_um_s) * (3 * fraction**2 - 2 * fraction**3)
        for index, fraction in enumerate(blend)
    )
    bounded: list[float] = []
    if desired:
        max_delta = config.max_acceleration_um_s2 * sample_interval_s
        if initial_speed_um_s is None:
            # Preserve the legacy policy's conservative slow-speed first sample.
            bounded.append(
                slow_speed_um_s
                if confidence[0] >= config.minimum_motion_confidence
                else 0.0
            )
            targets = desired[1:]
        else:
            bounded.append(
                min(
                    initial_speed_um_s + max_delta,
                    max(max(0.0, initial_speed_um_s - max_delta), desired[0]),
                )
            )
            targets = desired[1:]
        for target in targets:
            previous = bounded[-1]
            bounded.append(
                min(previous + max_delta, max(max(0.0, previous - max_delta), target))
            )
    return SpeedCurve(smoothed, blend, desired, tuple(bounded))
