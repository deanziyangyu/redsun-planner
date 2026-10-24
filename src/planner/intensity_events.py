"""Causal ROI-intensity rise detection for search-to-detail transitions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import isfinite
from statistics import median
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True)
class IntensityRiseConfig:
    """Trailing-baseline thresholds for an integration-independent ROI trace."""

    baseline_window_frames: int = 9
    baseline_min_samples: int = 5
    persistence_frames: int = 2
    release_persistence_frames: int = 2
    enter_relative_rise: float = 0.2
    exit_relative_rise: float = 0.08
    minimum_absolute_rise: float = 0.0
    baseline_floor: float = 1e-6

    def __post_init__(self) -> None:
        """Check the declared fields for consistency."""
        for name in (
            "baseline_window_frames",
            "baseline_min_samples",
            "persistence_frames",
            "release_persistence_frames",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.baseline_min_samples > self.baseline_window_frames:
            raise ValueError(
                "baseline_min_samples cannot exceed baseline_window_frames"
            )
        if not (
            isfinite(self.enter_relative_rise)
            and isfinite(self.exit_relative_rise)
            and 0 <= self.exit_relative_rise < self.enter_relative_rise
        ):
            raise ValueError("relative-rise thresholds must satisfy 0 <= exit < enter")
        if not isfinite(self.minimum_absolute_rise) or self.minimum_absolute_rise < 0:
            raise ValueError("minimum_absolute_rise must be finite and nonnegative")
        if not isfinite(self.baseline_floor) or self.baseline_floor <= 0:
            raise ValueError("baseline_floor must be finite and positive")


@dataclass(frozen=True)
class IntensityEventTrace:
    """Per-sample baseline, rise, active-state and transition records."""

    trailing_baseline: tuple[float | None, ...]
    relative_rise: tuple[float | None, ...]
    active: tuple[bool, ...]
    onset: tuple[bool, ...]
    release: tuple[bool, ...]


def detect_intensity_rise(
    intensity_density: Iterable[float],
    config: IntensityRiseConfig | None = None,
) -> IntensityEventTrace:
    """Detect a persistent rise above a trailing baseline, without future data.

    The supplied signal should be the integrated-positive-excess value (or
    another consistently normalized scalar) from one declared, fixed-geometry
    detector ROI. Candidate samples are held out of the baseline while a rise is being
    confirmed; the baseline is frozen while an event is active. This keeps the
    event independent of virtual TDI/integration and prevents a bright feature
    from immediately adapting away its own signal.
    """
    if config is None:
        config = IntensityRiseConfig()
    values = tuple(float(value) for value in intensity_density)
    if any(not isfinite(value) or value < 0 for value in values):
        raise ValueError("intensity_density values must be finite and nonnegative")

    history: deque[float] = deque(maxlen=config.baseline_window_frames)
    baselines: list[float | None] = []
    rises: list[float | None] = []
    active_samples: list[bool] = []
    onsets: list[bool] = []
    releases: list[bool] = []
    active = False
    candidate_count = 0
    release_count = 0

    for value in values:
        baseline = (
            median(history) if len(history) >= config.baseline_min_samples else None
        )
        onset = False
        release = False
        rise: float | None = None

        if baseline is None:
            history.append(value)
        else:
            scale = max(abs(baseline), config.baseline_floor)
            delta = value - baseline
            rise = delta / scale
            if active:
                exit_limit = max(
                    config.minimum_absolute_rise,
                    config.exit_relative_rise * scale,
                )
                release_count = release_count + 1 if delta <= exit_limit else 0
                if release_count >= config.release_persistence_frames:
                    active = False
                    release = True
                    candidate_count = 0
                    release_count = 0
                    history.clear()
                    history.append(value)
            else:
                enter_limit = max(
                    config.minimum_absolute_rise,
                    config.enter_relative_rise * scale,
                )
                if delta >= enter_limit:
                    candidate_count += 1
                    if candidate_count >= config.persistence_frames:
                        active = True
                        onset = True
                        candidate_count = 0
                        release_count = 0
                else:
                    candidate_count = 0
                    history.append(value)

        baselines.append(baseline)
        rises.append(rise)
        active_samples.append(active)
        onsets.append(onset)
        releases.append(release)

    return IntensityEventTrace(
        trailing_baseline=tuple(baselines),
        relative_rise=tuple(rises),
        active=tuple(active_samples),
        onset=tuple(onsets),
        release=tuple(releases),
    )
