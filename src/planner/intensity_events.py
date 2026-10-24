"""Causal ROI-intensity rise detection for search-to-detail transitions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import isfinite
from statistics import median
from typing import TYPE_CHECKING, Literal

from .speed_policy import SpeedBlendConfig, SpeedBlendController, SpeedProposal

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True)
class IntensityRoiSpec:
    """Delivered-frame ROI and scalar-signal meaning for one intensity trace."""

    trace_id: str
    sensor_shape_yx: tuple[int, int]
    roi_yxhw: tuple[int, int, int, int]
    signal_metric: str
    roi_mode: Literal["camera_readout", "software_slice"] = "software_slice"

    def __post_init__(self) -> None:
        """Validate sensor geometry and trace identity."""
        if (
            not isinstance(self.trace_id, str)
            or not self.trace_id.strip()
            or not isinstance(self.signal_metric, str)
            or not self.signal_metric.strip()
        ):
            raise ValueError("trace_id and signal_metric are required")
        if not isinstance(self.roi_mode, str) or self.roi_mode not in {
            "camera_readout",
            "software_slice",
        }:
            raise ValueError("roi_mode must be camera_readout or software_slice")
        if (
            not isinstance(self.sensor_shape_yx, tuple)
            or len(self.sensor_shape_yx) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in self.sensor_shape_yx
            )
        ):
            raise ValueError(
                "sensor_shape_yx must contain positive integer height and width"
            )
        if (
            not isinstance(self.roi_yxhw, tuple)
            or len(self.roi_yxhw) != 4
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in self.roi_yxhw
            )
        ):
            raise ValueError("roi_yxhw must contain integer y, x, height, width")
        y, x, height, width = self.roi_yxhw
        sensor_height, sensor_width = self.sensor_shape_yx
        if min(y, x) < 0 or min(height, width) < 1:
            raise ValueError("ROI origin must be nonnegative and ROI extent positive")
        if y + height > sensor_height or x + width > sensor_width:
            raise ValueError("roi_yxhw must fit within sensor_shape_yx")


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
    """Per-sample event records, optionally bound to their source ROI."""

    trailing_baseline: tuple[float | None, ...]
    relative_rise: tuple[float | None, ...]
    active: tuple[bool, ...]
    onset: tuple[bool, ...]
    release: tuple[bool, ...]
    roi: IntensityRoiSpec | None = None


@dataclass(frozen=True)
class IntensityEventSample:
    """One causal detector update for a single delivered ROI signal."""

    trailing_baseline: float | None
    relative_rise: float | None
    active: bool
    onset: bool
    release: bool


class IntensityRiseDetector:
    """Maintain trailing-baseline event state across delivered frames.

    Construct a new detector when ROI geometry, signal meaning, or acquisition
    epoch changes. ``update`` uses only the current and prior samples, so a
    host can evaluate a speed proposal as each frame arrives.
    """

    def __init__(self, config: IntensityRiseConfig | None = None) -> None:
        self.config = IntensityRiseConfig() if config is None else config
        if not isinstance(self.config, IntensityRiseConfig):
            raise TypeError("config must be an IntensityRiseConfig")
        self._history: deque[float] = deque(maxlen=self.config.baseline_window_frames)
        self._active = False
        self._candidate_count = 0
        self._release_count = 0

    def update(self, intensity_density: float) -> IntensityEventSample:
        """Consume one finite, nonnegative ROI signal and return its decision."""
        if isinstance(intensity_density, bool):
            raise TypeError("intensity_density must be a numeric signal, not bool")
        value = float(intensity_density)
        if not isfinite(value) or value < 0:
            raise ValueError("intensity_density must be finite and nonnegative")

        baseline = (
            median(self._history)
            if len(self._history) >= self.config.baseline_min_samples
            else None
        )
        onset = False
        release = False
        rise: float | None = None

        if baseline is None:
            self._history.append(value)
        else:
            scale = max(abs(baseline), self.config.baseline_floor)
            delta = value - baseline
            rise = delta / scale
            if self._active:
                exit_limit = max(
                    self.config.minimum_absolute_rise,
                    self.config.exit_relative_rise * scale,
                )
                self._release_count = (
                    self._release_count + 1 if delta <= exit_limit else 0
                )
                if self._release_count >= self.config.release_persistence_frames:
                    self._active = False
                    release = True
                    self._candidate_count = 0
                    self._release_count = 0
                    self._history.clear()
                    self._history.append(value)
            else:
                enter_limit = max(
                    self.config.minimum_absolute_rise,
                    self.config.enter_relative_rise * scale,
                )
                if delta >= enter_limit:
                    self._candidate_count += 1
                    if self._candidate_count >= self.config.persistence_frames:
                        self._active = True
                        onset = True
                        self._candidate_count = 0
                        self._release_count = 0
                else:
                    self._candidate_count = 0
                    self._history.append(value)

        return IntensityEventSample(
            trailing_baseline=baseline,
            relative_rise=rise,
            active=self._active,
            onset=onset,
            release=release,
        )


def detect_intensity_rise(
    intensity_density: Iterable[float],
    config: IntensityRiseConfig | None = None,
    *,
    roi: IntensityRoiSpec | None = None,
) -> IntensityEventTrace:
    """Detect a persistent rise above a trailing baseline, without future data.

    The supplied signal should be the integrated-positive-excess value (or
    another consistently normalized scalar) from one declared, fixed-geometry
    detector ROI. Candidate samples are held out of the baseline while a rise is
    being confirmed; the baseline is frozen while an event is active. This keeps the
    event independent of virtual TDI/integration and prevents a bright feature
    from immediately adapting away its own signal.

    ``roi`` associates the complete result with a rectangle in the delivered
    frame and a host-defined signal metric. Its mode distinguishes the camera
    readout ROI from a software slice. The scalar sequence may have come from a leading
    edge, an interior slit or the full sensor, and may use any consistent
    integration or normalization method. Use a new ``trace_id`` when the ROI or
    signal meaning changes. Omitting ``roi`` retains the legacy scalar-only form.
    """
    if config is None:
        config = IntensityRiseConfig()
    if roi is not None and not isinstance(roi, IntensityRoiSpec):
        raise TypeError("roi must be an IntensityRoiSpec")
    raw_values = tuple(intensity_density)
    if any(isinstance(value, bool) for value in raw_values):
        raise TypeError("intensity_density must be numeric signals, not bool")
    values = tuple(float(value) for value in raw_values)
    if any(not isfinite(value) or value < 0 for value in values):
        raise ValueError("intensity_density values must be finite and nonnegative")

    detector = IntensityRiseDetector(config)
    samples = [detector.update(value) for value in values]

    return IntensityEventTrace(
        trailing_baseline=tuple(sample.trailing_baseline for sample in samples),
        relative_rise=tuple(sample.relative_rise for sample in samples),
        active=tuple(sample.active for sample in samples),
        onset=tuple(sample.onset for sample in samples),
        release=tuple(sample.release for sample in samples),
        roi=roi,
    )


class RoiIntensitySpeedPolicy:
    """Compose ROI rise detection with an optional virtual speed proposal.

    The host supplies one scalar signal from its configured ROI. This policy
    owns the causal detector and its intensity-to-speed mapping; it does not
    inspect image data or issue device commands. Construct a new instance when
    the ROI, signal meaning, or acquisition epoch changes.
    """

    def __init__(
        self,
        *,
        rise_config: IntensityRiseConfig | None = None,
        slow_speed_um_s: float | None = None,
        fast_speed_um_s: float | None = None,
        interest_rise_saturation: float | None = None,
        sample_interval_s: float,
        motion_confidence: float = 1.0,
        max_acceleration_um_s2: float = 5000.0,
        initial_speed_um_s: float | None = None,
    ) -> None:
        """Create an epoch-scoped detector and optional speed controller."""
        self.detector = IntensityRiseDetector(rise_config)
        if not isfinite(sample_interval_s) or sample_interval_s <= 0:
            raise ValueError("sample_interval_s must be finite and positive")
        if not isfinite(motion_confidence) or not 0 <= motion_confidence <= 1:
            raise ValueError("motion_confidence must be finite and in [0, 1]")
        configured = (
            slow_speed_um_s,
            fast_speed_um_s,
            interest_rise_saturation,
        )
        if any(value is not None for value in configured) and any(
            value is None for value in configured
        ):
            raise ValueError(
                "slow speed, fast speed and interest rise saturation are required together"
            )
        if slow_speed_um_s is None:
            if initial_speed_um_s is not None:
                raise ValueError("initial speed requires configured speed bounds")
            self._speed_controller: SpeedBlendController | None = None
            self._interest_rise_saturation: float | None = None
            self._motion_confidence = 1.0
            return
        assert fast_speed_um_s is not None
        assert interest_rise_saturation is not None
        if not isfinite(interest_rise_saturation) or interest_rise_saturation <= 0:
            raise ValueError("interest_rise_saturation must be finite and positive")
        self._interest_rise_saturation = float(interest_rise_saturation)
        self._motion_confidence = float(motion_confidence)
        self._speed_controller = SpeedBlendController(
            slow_speed_um_s=float(slow_speed_um_s),
            fast_speed_um_s=float(fast_speed_um_s),
            sample_interval_s=sample_interval_s,
            config=SpeedBlendConfig(max_acceleration_um_s2=max_acceleration_um_s2),
            initial_speed_um_s=initial_speed_um_s,
        )

    def update_intensity(self, intensity_density: float) -> IntensityEventSample:
        """Update the rise detector from one scalar ROI signal."""
        return self.detector.update(intensity_density)

    def propose_speed(self, event: IntensityEventSample) -> SpeedProposal | None:
        """Map one detector update to an optional acceleration-limited proposal."""
        if not isinstance(event, IntensityEventSample):
            raise TypeError("event must be an IntensityEventSample")
        if self._speed_controller is None:
            return None
        assert self._interest_rise_saturation is not None
        interest = min(
            1.0,
            max(0.0, (event.relative_rise or 0.0) / self._interest_rise_saturation),
        )
        return self._speed_controller.update(interest, self._motion_confidence)
