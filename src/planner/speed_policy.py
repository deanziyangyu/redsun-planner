"""Pure causal intensity-to-speed blending for sample-search trajectories."""

from __future__ import annotations

from collections import deque
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


@dataclass(frozen=True)
class SpeedProposal:
    """One causal speed proposal produced from the observations so far."""

    interest: float
    smoothed_interest: float
    blend_fraction: float
    desired_speed_um_s: float
    slew_limited_speed_um_s: float


class SpeedBlendController:
    """Produce causal, acceleration-limited speed proposals one sample at a time.

    This pure policy is suitable for a Bluesky plan or a streaming observer to
    evaluate. It does not issue device commands and its acceleration bound is a
    model input, not a claim about physical stage behavior.
    """

    def __init__(
        self,
        *,
        slow_speed_um_s: float,
        fast_speed_um_s: float,
        sample_interval_s: float,
        config: SpeedBlendConfig = _DEFAULT_SPEED_BLEND_CONFIG,
        initial_speed_um_s: float | None = None,
    ) -> None:
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
            raise ValueError("initial speed must be finite and within [0, fast_speed]")
        self.slow_speed_um_s = slow_speed_um_s
        self.fast_speed_um_s = fast_speed_um_s
        self.sample_interval_s = sample_interval_s
        self.config = config
        self._initial_speed_um_s = initial_speed_um_s
        self._history: deque[float] = deque(maxlen=config.smoothing_frames)
        self._previous_speed_um_s: float | None = None

    def update(self, interest: float, motion_confidence: float = 1.0) -> SpeedProposal:
        """Return the speed proposal for one new intensity observation."""
        if not isfinite(interest) or not 0 <= interest <= 1:
            raise ValueError("interest values must be finite and in [0, 1]")
        if not isfinite(motion_confidence) or not 0 <= motion_confidence <= 1:
            raise ValueError("motion_confidence values must be finite and in [0, 1]")
        self._history.append(interest)
        smoothed = sum(self._history) / len(self._history)
        span = self.config.interest_slow_threshold - self.config.interest_fast_threshold
        fraction = min(
            1.0,
            max(
                0.0,
                (smoothed - self.config.interest_fast_threshold) / span,
            ),
        )
        if motion_confidence < self.config.minimum_motion_confidence:
            desired = 0.0
        else:
            smoothstep = 3 * fraction**2 - 2 * fraction**3
            desired = (
                self.fast_speed_um_s
                - (self.fast_speed_um_s - self.slow_speed_um_s) * smoothstep
            )

        max_delta = self.config.max_acceleration_um_s2 * self.sample_interval_s
        if self._previous_speed_um_s is None and self._initial_speed_um_s is None:
            limited = (
                self.slow_speed_um_s
                if motion_confidence >= self.config.minimum_motion_confidence
                else 0.0
            )
        else:
            previous = (
                self._initial_speed_um_s
                if self._previous_speed_um_s is None
                else self._previous_speed_um_s
            )
            assert previous is not None
            limited = min(
                previous + max_delta,
                max(max(0.0, previous - max_delta), desired),
            )
        self._previous_speed_um_s = limited
        return SpeedProposal(interest, smoothed, fraction, desired, limited)


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

    controller = SpeedBlendController(
        slow_speed_um_s=slow_speed_um_s,
        fast_speed_um_s=fast_speed_um_s,
        sample_interval_s=sample_interval_s,
        config=config,
        initial_speed_um_s=initial_speed_um_s,
    )
    proposals = tuple(
        controller.update(value, confidence[index]) for index, value in enumerate(raw)
    )
    return SpeedCurve(
        tuple(sample.smoothed_interest for sample in proposals),
        tuple(sample.blend_fraction for sample in proposals),
        tuple(sample.desired_speed_um_s for sample in proposals),
        tuple(sample.slew_limited_speed_um_s for sample in proposals),
    )
