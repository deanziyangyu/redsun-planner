"""Pure acquisition-space model for stage velocity, exposure and detection width.

The model classifies operating points against a caller-declared useful range,
computes the framework's kinematic and dimensionless metrics, rejects
measurement settings with typed feasibility reasons, and ranks feasible points
with an application utility. It emits no device commands and no Bluesky plan
messages; plan lowering stays in :mod:`planner.plans`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite, sqrt
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from typing import Any


class AxisBand(str, Enum):
    """Position of one axis value relative to the declared useful range."""

    LOW = "low"
    HIGH = "high"
    BETWEEN = "between"


class AcquisitionAxis(str, Enum):
    """Axis identifiers in the framework's ``(v, T_e, W)`` order."""

    VELOCITY = "v"
    EXPOSURE = "Te"
    WIDTH = "W"


_AXIS_INDEX: Mapping[AcquisitionAxis, int] = MappingProxyType(
    {
        AcquisitionAxis.VELOCITY: 0,
        AcquisitionAxis.EXPOSURE: 1,
        AcquisitionAxis.WIDTH: 2,
    }
)


class OperatingCorner(str, Enum):
    """One of eight limiting regimes, coded by the ``(v, T_e, W)`` low/high bits."""

    C000 = "000"
    C001 = "001"
    C010 = "010"
    C011 = "011"
    C100 = "100"
    C101 = "101"
    C110 = "110"
    C111 = "111"

    def digit(self, axis: AcquisitionAxis) -> int:
        """Return the low/high digit (0 or 1) for one axis."""
        return int(self.value[_AXIS_INDEX[axis]])

    def is_low(self, axis: AcquisitionAxis) -> bool:
        """Return whether the axis digit is low (0)."""
        return self.digit(axis) == 0

    def is_high(self, axis: AcquisitionAxis) -> bool:
        """Return whether the axis digit is high (1)."""
        return self.digit(axis) == 1

    def digits(self) -> tuple[int, int, int]:
        """Return the three digits in ``(v, T_e, W)`` order."""
        return (int(self.value[0]), int(self.value[1]), int(self.value[2]))

    def hamming_distance(self, other: OperatingCorner) -> int:
        """Return the number of axes on which two corners differ."""
        return sum(left != right for left, right in zip(self.value, other.value))

    def is_adjacent(self, other: OperatingCorner) -> bool:
        """Return whether two corners differ on exactly one axis."""
        return self.hamming_distance(other) == 1

    def changed_axes(self, other: OperatingCorner) -> tuple[AcquisitionAxis, ...]:
        """Return the axes whose digit differs, in ``(v, T_e, W)`` order."""
        return tuple(
            axis for axis in AcquisitionAxis if self.digit(axis) != other.digit(axis)
        )


@dataclass(frozen=True)
class CornerRegime:
    """One row of the documented eight-corner acquisition taxonomy.

    Attributes
    ----------
    corner
        Operating corner this row describes.
    regime
        Descriptive name of the operating regime.
    example
        Representative application or reconstruction.
    qualification
        Important qualification or caveat.
    """

    corner: OperatingCorner
    regime: str
    example: str
    qualification: str

    def __post_init__(self) -> None:
        """Check the declared fields for consistency."""
        if not isinstance(self.corner, OperatingCorner):
            raise TypeError("corner must be an OperatingCorner")
        for name in ("regime", "example", "qualification"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")


_CORNER_ROWS: tuple[CornerRegime, ...] = (
    CornerRegime(
        OperatingCorner.C000,
        "Localized high-speed sampling",
        "Fast time series of a narrow region",
        "Not intrinsically line-confocal",
    ),
    CornerRegime(
        OperatingCorner.C001,
        "High-speed widefield timelapse",
        "Single-FOV live-cell dynamics",
        "Standard widefield regime",
    ),
    CornerRegime(
        OperatingCorner.C010,
        "Localized photon integration",
        "Long-exposure, narrow-ROI fluorescence measurement",
        "Not TDI if stage is stationary",
    ),
    CornerRegime(
        OperatingCorner.C011,
        "Widefield photon integration",
        "Low-signal stationary widefield imaging",
        "Intensity/SNR prioritized over sampling rate",
    ),
    CornerRegime(
        OperatingCorner.C100,
        "Motion-frozen push-broom scanning",
        "Continuous strip-based imaging",
        "Short exposure relative to displacement",
    ),
    CornerRegime(
        OperatingCorner.C101,
        "Motion-frozen panoramic sampling",
        "Rapid snapshot search or widefield panorama",
        "Search coverage may differ from gap-free mapping",
    ),
    CornerRegime(
        OperatingCorner.C110,
        "Motion-integrated strip imaging",
        "Registered multi-frame accumulation; deconvolution",
        "Digital TDI is a processing strategy, not the corner itself",
    ),
    CornerRegime(
        OperatingCorner.C111,
        "Motion-integrated panoramic imaging",
        "Coarse searching; computational deblurring",
        "Long-exposure blur can reduce target detectability",
    ),
)


def _build_corners() -> Mapping[OperatingCorner, CornerRegime]:
    by_corner = {row.corner: row for row in _CORNER_ROWS}
    if len(_CORNER_ROWS) != len(OperatingCorner) or set(by_corner) != set(
        OperatingCorner
    ):
        raise ValueError("CORNERS must carry exactly one row for every corner")
    return MappingProxyType(by_corner)


CORNERS: Mapping[OperatingCorner, CornerRegime] = _build_corners()


@dataclass(frozen=True)
class ExampleTrajectory:
    """A documented single transition between two named acquisition corners.

    Three of the framework's examples change exactly one axis. ``101 -> 011``
    changes both velocity and exposure, so a documented trajectory does not
    imply Hamming adjacency; use :meth:`OperatingCorner.is_adjacent` when a
    one-axis move is the requirement.
    """

    source: OperatingCorner
    target: OperatingCorner
    intent: str

    def __post_init__(self) -> None:
        """Check the declared fields for consistency."""
        if not isinstance(self.source, OperatingCorner) or not isinstance(
            self.target, OperatingCorner
        ):
            raise TypeError("trajectory endpoints must be OperatingCorner values")
        if self.source is self.target:
            raise ValueError("trajectory endpoints must be different corners")
        if not isinstance(self.intent, str) or not self.intent.strip():
            raise ValueError("intent must be a nonempty string")

    @property
    def changed_axes(self) -> tuple[AcquisitionAxis, ...]:
        """Return the axes that differ between the two corners."""
        return self.source.changed_axes(self.target)

    @property
    def is_adjacent(self) -> bool:
        """Return whether the transition changes exactly one axis."""
        return self.source.is_adjacent(self.target)


def _build_example_trajectories() -> tuple[ExampleTrajectory, ...]:
    trajectories = (
        ExampleTrajectory(
            OperatingCorner.C101,
            OperatingCorner.C001,
            "Fast area search -> stationary high-speed timelapse after target detection.",
        ),
        ExampleTrajectory(
            OperatingCorner.C101,
            OperatingCorner.C011,
            "Search -> stationary sensitive imaging of a weak target.",
        ),
        ExampleTrajectory(
            OperatingCorner.C100,
            OperatingCorner.C110,
            "Motion-frozen line scan -> registered temporal accumulation for SNR.",
        ),
        ExampleTrajectory(
            OperatingCorner.C101,
            OperatingCorner.C111,
            "Short-exposure widefield scan -> motion-integrated panorama, "
            "with different reconstruction requirements.",
        ),
    )
    if len(trajectories) != 4 or len(
        {(trajectory.source, trajectory.target) for trajectory in trajectories}
    ) != len(trajectories):
        raise ValueError("EXAMPLE_TRAJECTORIES must list four distinct moves")
    two_axis = [trajectory for trajectory in trajectories if not trajectory.is_adjacent]
    if [
        (trajectory.source.value, trajectory.target.value) for trajectory in two_axis
    ] != [("101", "011")]:
        raise ValueError("only the documented 101 -> 011 example changes two axes")
    return trajectories


EXAMPLE_TRAJECTORIES: tuple[ExampleTrajectory, ...] = _build_example_trajectories()


def example_trajectories() -> tuple[ExampleTrajectory, ...]:
    """Return the four documented acquisition-space example trajectories."""
    return EXAMPLE_TRAJECTORIES


def _finite_real(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _finite_number(value: float, name: str, *, positive: bool) -> float:
    number = _finite_real(value, name)
    if positive and number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    if not positive and number < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def _finite_fraction(value: float, name: str) -> float:
    number = _finite_number(value, name, positive=False)
    if number > 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return number


@dataclass(frozen=True)
class AcquisitionPoint:
    """One operating point in object-space acquisition units.

    ``pixel_size_um`` is the object-space pixel pitch (the framework's ``s``),
    not the physical detector pitch, so ``width_um`` and ``height_um`` are
    already specimen-plane micrometres. The point is inert data: it does not
    assert that a camera, stage or calibration can realize it.
    """

    velocity_um_s: float
    exposure_s: float
    width_um: float
    frame_period_s: float
    pixel_size_um: float
    resolution_um: float
    height_um: float

    def __post_init__(self) -> None:
        """Check the declared fields for consistency."""
        object.__setattr__(
            self,
            "velocity_um_s",
            _finite_number(self.velocity_um_s, "velocity_um_s", positive=False),
        )
        for name in (
            "exposure_s",
            "width_um",
            "frame_period_s",
            "pixel_size_um",
            "resolution_um",
            "height_um",
        ):
            object.__setattr__(
                self, name, _finite_number(getattr(self, name), name, positive=True)
            )
        if self.exposure_s > self.frame_period_s:
            raise ValueError("exposure_s must not exceed frame_period_s")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary of the declared values."""
        return asdict(self)

    def metrics(self) -> AcquisitionMetrics:
        """Return the derived metrics for this point."""
        return metrics_of(self)


@dataclass(frozen=True)
class AxisThreshold:
    """Low/high band thresholds for one acquisition axis, in its declared unit.

    ``low_max`` is the largest value still classified low and ``high_min`` is
    the smallest value classified high. Values strictly between are
    :attr:`AxisBand.BETWEEN`.
    """

    low_max: float
    high_min: float

    def __post_init__(self) -> None:
        """Check the declared fields for consistency."""
        low_max = _finite_number(self.low_max, "low_max", positive=True)
        high_min = _finite_number(self.high_min, "high_min", positive=True)
        if low_max >= high_min:
            raise ValueError("low_max must be less than high_min")
        object.__setattr__(self, "low_max", low_max)
        object.__setattr__(self, "high_min", high_min)


@dataclass(frozen=True)
class AcquisitionEnvelope:
    """Caller-declared useful range per axis, split into low/high/between bands.

    The digits are relative to this declaration, not universal thresholds.
    """

    velocity_um_s: AxisThreshold
    exposure_s: AxisThreshold
    width_um: AxisThreshold

    def __post_init__(self) -> None:
        """Check the declared fields for consistency."""
        for name in ("velocity_um_s", "exposure_s", "width_um"):
            if not isinstance(getattr(self, name), AxisThreshold):
                raise TypeError(f"{name} must be an AxisThreshold")

    def _threshold(self, axis: AcquisitionAxis) -> AxisThreshold:
        thresholds: Mapping[AcquisitionAxis, AxisThreshold] = {
            AcquisitionAxis.VELOCITY: self.velocity_um_s,
            AcquisitionAxis.EXPOSURE: self.exposure_s,
            AcquisitionAxis.WIDTH: self.width_um,
        }
        return thresholds[axis]

    def band(self, axis: AcquisitionAxis, value: float) -> AxisBand:
        """Classify one axis value as low, high or between."""
        if not isinstance(axis, AcquisitionAxis):
            raise TypeError("axis must be an AcquisitionAxis")
        number = _finite_number(value, "value", positive=False)
        threshold = self._threshold(axis)
        if number <= threshold.low_max:
            return AxisBand.LOW
        if number >= threshold.high_min:
            return AxisBand.HIGH
        return AxisBand.BETWEEN

    def corner(self, point: AcquisitionPoint) -> OperatingCorner | None:
        """Return the point's corner, or ``None`` when any axis is between."""
        if not isinstance(point, AcquisitionPoint):
            raise TypeError("point must be an AcquisitionPoint")
        bands = (
            self.band(AcquisitionAxis.VELOCITY, point.velocity_um_s),
            self.band(AcquisitionAxis.EXPOSURE, point.exposure_s),
            self.band(AcquisitionAxis.WIDTH, point.width_um),
        )
        if AxisBand.BETWEEN in bands:
            return None
        digits = "".join("1" if band is AxisBand.HIGH else "0" for band in bands)
        return OperatingCorner(digits)


@dataclass(frozen=True)
class AcquisitionMetrics:
    """Derived kinematic and dimensionless metrics for one acquisition point.

    ``residence_s``, ``observations``, ``gamma`` and ``overlap`` are ``None``
    for a stationary stage because their definitions divide by velocity. They
    are never reported as zero or infinity. ``observations`` and ``overlap``
    coincide by definition: the overlap measure ``eta`` is also the number of
    successive observations per location.
    """

    motion_blur_um: float
    interframe_shift_um: float
    residence_s: float | None
    observations: float | None
    area_throughput_um2_s: float
    beta: float
    gamma: float | None
    duty: float
    overlap: float | None
    effective_sigma_um: float
    blur_ratio: float

    def __post_init__(self) -> None:
        """Check the declared fields for consistency and JSON safety."""
        for name in (
            "motion_blur_um",
            "interframe_shift_um",
            "area_throughput_um2_s",
            "beta",
        ):
            object.__setattr__(
                self,
                name,
                _finite_number(getattr(self, name), name, positive=False),
            )
        for name in ("duty", "effective_sigma_um", "blur_ratio"):
            object.__setattr__(
                self, name, _finite_number(getattr(self, name), name, positive=True)
            )
        if self.duty > 1:
            raise ValueError("duty must be in (0, 1]")
        for name in ("residence_s", "observations", "gamma", "overlap"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self, name, _finite_number(value, name, positive=False)
                )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary; undefined stationary metrics are null."""
        return asdict(self)


def metrics_of(point: AcquisitionPoint) -> AcquisitionMetrics:
    """Compute the framework metrics for one acquisition point.

    For a stationary stage, quantities defined by division by velocity are
    ``None``. Area throughput, beta, duty, effective sigma and the blur ratio
    remain meaningful because they are defined at ``v = 0``.
    """
    if not isinstance(point, AcquisitionPoint):
        raise TypeError("point must be an AcquisitionPoint")
    velocity = point.velocity_um_s
    delta0 = point.resolution_um
    motion_blur = velocity * point.exposure_s
    interframe_shift = velocity * point.frame_period_s
    moving = velocity > 0
    observations = point.width_um / interframe_shift if moving else None
    effective_sigma = sqrt(delta0 * delta0 + motion_blur * motion_blur / 12)
    return AcquisitionMetrics(
        motion_blur_um=motion_blur,
        interframe_shift_um=interframe_shift,
        residence_s=point.width_um / velocity if moving else None,
        observations=observations,
        area_throughput_um2_s=velocity * point.height_um,
        beta=motion_blur / delta0,
        gamma=interframe_shift / point.width_um if moving else None,
        duty=point.exposure_s / point.frame_period_s,
        overlap=observations,
        effective_sigma_um=effective_sigma,
        blur_ratio=effective_sigma / delta0,
    )


@dataclass(frozen=True)
class MeasurementLimits:
    """Declared ceilings and bounds for measurement feasibility.

    A value exactly at a ceiling or bound is feasible. ``min_observations``,
    ``min_duty`` and ``max_duty`` are optional. Duty bounds are fractions of
    the frame period and lie in ``[0, 1]``.
    """

    max_blur_ratio: float
    max_gamma: float
    min_observations: float | None = None
    min_duty: float | None = None
    max_duty: float | None = None

    def __post_init__(self) -> None:
        """Check the declared fields for consistency."""
        object.__setattr__(
            self,
            "max_blur_ratio",
            _finite_number(self.max_blur_ratio, "max_blur_ratio", positive=True),
        )
        object.__setattr__(
            self,
            "max_gamma",
            _finite_number(self.max_gamma, "max_gamma", positive=True),
        )
        if self.min_observations is not None:
            object.__setattr__(
                self,
                "min_observations",
                _finite_number(
                    self.min_observations, "min_observations", positive=True
                ),
            )
        min_duty = self.min_duty
        if min_duty is not None:
            min_duty = _finite_fraction(min_duty, "min_duty")
            object.__setattr__(self, "min_duty", min_duty)
        max_duty = self.max_duty
        if max_duty is not None:
            max_duty = _finite_fraction(max_duty, "max_duty")
            object.__setattr__(self, "max_duty", max_duty)
        if min_duty is not None and max_duty is not None and min_duty > max_duty:
            raise ValueError("min_duty must not exceed max_duty")


class FeasibilityReason(str, Enum):
    """Typed reason a measurement setting is rejected."""

    BLUR = "blur"
    GAP = "gap"
    TOO_FEW_OBSERVATIONS = "too_few_observations"
    DUTY = "duty"


@dataclass(frozen=True)
class MeasurementVerdict:
    """Feasibility decision for one metrics record, with every violated reason."""

    feasible: bool
    reasons: tuple[FeasibilityReason, ...]
    metrics: AcquisitionMetrics

    def __post_init__(self) -> None:
        """Check the declared fields for consistency."""
        if not isinstance(self.feasible, bool):
            raise TypeError("feasible must be a bool")
        if not isinstance(self.reasons, tuple):
            raise TypeError("reasons must be a tuple")
        if any(not isinstance(reason, FeasibilityReason) for reason in self.reasons):
            raise TypeError("reasons must contain FeasibilityReason values")
        if len(set(self.reasons)) != len(self.reasons):
            raise ValueError("reasons must not repeat")
        if self.feasible != (not self.reasons):
            raise ValueError("feasible must be true exactly when no reason is present")
        if not isinstance(self.metrics, AcquisitionMetrics):
            raise TypeError("metrics must be an AcquisitionMetrics")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary with reason values as strings."""
        return {
            "feasible": self.feasible,
            "reasons": [reason.value for reason in self.reasons],
            "metrics": self.metrics.to_dict(),
        }


def _assess_metrics(
    metrics: AcquisitionMetrics, limits: MeasurementLimits
) -> MeasurementVerdict:
    reasons: list[FeasibilityReason] = []
    if metrics.blur_ratio > limits.max_blur_ratio:
        reasons.append(FeasibilityReason.BLUR)
    if metrics.gamma is not None and metrics.gamma > limits.max_gamma:
        reasons.append(FeasibilityReason.GAP)
    if (
        metrics.observations is not None
        and limits.min_observations is not None
        and metrics.observations < limits.min_observations
    ):
        reasons.append(FeasibilityReason.TOO_FEW_OBSERVATIONS)
    duty_violated = (
        limits.min_duty is not None and metrics.duty < limits.min_duty
    ) or (limits.max_duty is not None and metrics.duty > limits.max_duty)
    if duty_violated:
        reasons.append(FeasibilityReason.DUTY)
    return MeasurementVerdict(not reasons, tuple(reasons), metrics)


def assess_measurement(
    point: AcquisitionPoint, limits: MeasurementLimits
) -> MeasurementVerdict:
    """Assess every declared feasibility constraint for one acquisition point."""
    if not isinstance(point, AcquisitionPoint):
        raise TypeError("point must be an AcquisitionPoint")
    if not isinstance(limits, MeasurementLimits):
        raise TypeError("limits must be a MeasurementLimits")
    return _assess_metrics(metrics_of(point), limits)


class Utility(Protocol):
    """Application-specific objective over an acquisition point and its metrics."""

    def score(self, point: AcquisitionPoint, metrics: AcquisitionMetrics) -> float:
        """Return a finite, higher-is-better score."""
        ...


@dataclass(frozen=True)
class SearchUtility:
    """Rank feasible points by ideal unique-area throughput.

    The score is ``Phi_A = v * H``: searched specimen area per second. This is
    the framework's search objective and nothing more. It deliberately does not
    invent a detection probability, sensitivity or false-negative model: the
    blur, gap, observation-count and duty constraints belong to
    :func:`assess_measurement`, which :func:`rank_by_utility` applies when
    ``limits`` is supplied. The throughput is idealized and ignores acceleration,
    turnaround, overlap,
    unusable edge pixels, autofocus and camera timing, so a high score is a
    ceiling rather than a prediction of realized search performance.
    """

    def score(self, point: AcquisitionPoint, metrics: AcquisitionMetrics) -> float:
        """Return ideal searched area per second, in square micrometres."""
        return metrics.area_throughput_um2_s


@dataclass(frozen=True)
class ScoredPoint:
    """One point with its metrics, feasibility verdict and utility score."""

    point: AcquisitionPoint
    metrics: AcquisitionMetrics
    verdict: MeasurementVerdict
    score: float

    def __post_init__(self) -> None:
        """Check the declared fields for consistency."""
        if not isinstance(self.point, AcquisitionPoint):
            raise TypeError("point must be an AcquisitionPoint")
        if not isinstance(self.metrics, AcquisitionMetrics):
            raise TypeError("metrics must be an AcquisitionMetrics")
        if not isinstance(self.verdict, MeasurementVerdict):
            raise TypeError("verdict must be a MeasurementVerdict")
        if self.verdict.metrics != self.metrics:
            raise ValueError("verdict and scored metrics must agree")
        object.__setattr__(self, "score", _finite_real(self.score, "score"))

    @property
    def feasible(self) -> bool:
        """Return whether the point passed every declared measurement limit."""
        return self.verdict.feasible

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary of the point, metrics, verdict and score."""
        return {
            "point": self.point.to_dict(),
            "metrics": self.metrics.to_dict(),
            "verdict": self.verdict.to_dict(),
            "score": self.score,
        }


def rank_by_utility(
    points: Iterable[AcquisitionPoint],
    utility: Utility,
    limits: MeasurementLimits | None = None,
) -> tuple[ScoredPoint, ...]:
    """Score points and return every input, feasible points first and best-first.

    When ``limits`` is ``None`` every point is feasible. Otherwise each point
    receives :func:`assess_measurement`'s verdict. Feasible points are sorted by
    descending score, with input order breaking ties. Rejected points follow in
    input order and keep their verdict and score, so they remain distinguishable
    rather than being silently dropped; a rejected score is informational and
    must not be selected as if the point were feasible.
    """
    if limits is not None and not isinstance(limits, MeasurementLimits):
        raise TypeError("limits must be a MeasurementLimits or None")
    feasible: list[ScoredPoint] = []
    rejected: list[ScoredPoint] = []
    for point in points:
        metrics = metrics_of(point)
        verdict = (
            MeasurementVerdict(True, (), metrics)
            if limits is None
            else _assess_metrics(metrics, limits)
        )
        score = _finite_real(utility.score(point, metrics), "utility score")
        scored = ScoredPoint(point, metrics, verdict, score)
        (feasible if verdict.feasible else rejected).append(scored)
    feasible.sort(key=lambda item: -item.score)
    return (*feasible, *rejected)


__all__ = [
    "CORNERS",
    "EXAMPLE_TRAJECTORIES",
    "AcquisitionAxis",
    "AcquisitionEnvelope",
    "AcquisitionMetrics",
    "AcquisitionPoint",
    "AxisBand",
    "AxisThreshold",
    "CornerRegime",
    "ExampleTrajectory",
    "FeasibilityReason",
    "MeasurementLimits",
    "MeasurementVerdict",
    "OperatingCorner",
    "ScoredPoint",
    "SearchUtility",
    "Utility",
    "assess_measurement",
    "example_trajectories",
    "metrics_of",
    "rank_by_utility",
]
