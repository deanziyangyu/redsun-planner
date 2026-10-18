"""Timing model for an adaptive broom scan with per-sample speed modulation.

A broom raster covers the field row by row. Around every declared sample the
stage target speed is ``fov_um / dwell_s`` so the sample remains inside the
field of view for the requested dwell; between samples it returns to the travel
speed. A forward/backward pass enforces the acceleration limit, so the stage
decelerates before a slow zone and accelerates after it. The report also
estimates two regular-broom baselines over the same path: a constant cruise at
the travel speed and a constant scan at the dwell speed. Their difference from
the adaptive run isolates what the speed modulation adds or saves.

The model is an estimate on declared values. It opens no device and asserts no
controller calibration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, hypot, isclose, isfinite, sqrt
from typing import TYPE_CHECKING, Any

import numpy as np

from .simulation import MockTarget, StageProfile

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


def _finite(value: float, name: str, *, positive: bool) -> float:
    """Return a finite number, rejecting booleans and out-of-range values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    if positive and number <= 0:
        raise ValueError(f"{name} must be positive")
    if not positive and number < 0:
        raise ValueError(f"{name} must be nonnegative")
    return number


def _pair(
    value: tuple[float, float], name: str, *, positive: bool
) -> tuple[float, float]:
    """Return a validated 2-tuple of finite numbers."""
    if len(value) != 2:
        raise ValueError(f"{name} must contain two values")
    first, second = value
    return (
        _finite(first, f"{name}[0]", positive=positive),
        _finite(second, f"{name}[1]", positive=positive),
    )


@dataclass(frozen=True)
class BroomSpec:
    """One adaptive broom raster over a field with declared mock samples.

    ``row_pitch_um`` sets the row spacing; ``fov_um`` is the square swath. The
    dwell speed is ``fov_um / dwell_s``. ``exposure_s``, ``readout_s`` and
    ``overhead_s`` describe the camera budget but do not change the stage time
    in this model; they are retained for the report's acquisition context.
    """

    field_size_um: tuple[float, float]
    fov_um: float
    row_pitch_um: float
    dwell_s: float
    stage: StageProfile
    exposure_s: float
    readout_s: float = 0.0
    overhead_s: float = 0.0
    samples: tuple[MockTarget, ...] = ()
    profile_step_um: float = 2.0
    command_delay_s: float = 0.0

    def __post_init__(self) -> None:
        """Check the declaration and sample positions."""
        object.__setattr__(
            self,
            "field_size_um",
            _pair(self.field_size_um, "field_size_um", positive=True),
        )
        for name in ("fov_um", "row_pitch_um", "dwell_s", "exposure_s"):
            object.__setattr__(
                self, name, _finite(getattr(self, name), name, positive=True)
            )
        for name in ("readout_s", "overhead_s", "command_delay_s"):
            object.__setattr__(
                self, name, _finite(getattr(self, name), name, positive=False)
            )
        object.__setattr__(
            self,
            "profile_step_um",
            _finite(self.profile_step_um, "profile_step_um", positive=True),
        )
        if not isinstance(self.stage, StageProfile):
            raise TypeError("stage must be a StageProfile")
        if not isinstance(self.samples, tuple):
            raise TypeError("samples must be a tuple of MockTarget records")
        width, height = self.field_size_um
        seen: set[str] = set()
        for sample in self.samples:
            if not isinstance(sample, MockTarget):
                raise TypeError("samples must contain MockTarget records")
            if sample.target_id in seen:
                raise ValueError(f"duplicate sample id {sample.target_id!r}")
            seen.add(sample.target_id)
            x, y = sample.center_xy_um
            if not (0 <= x <= width and 0 <= y <= height):
                raise ValueError(
                    f"sample {sample.target_id} center is outside the field"
                )

    @property
    def row_count(self) -> int:
        """Number of raster rows covering the field height."""
        _, height = self.field_size_um
        return max(1, ceil(height / self.row_pitch_um))

    @property
    def dwell_velocity_um_s(self) -> float:
        """Scan speed that keeps a point inside the swath for ``dwell_s``."""
        return self.fov_um / self.dwell_s

    @property
    def field_area_um2(self) -> float:
        """Field area in square micrometres."""
        width, height = self.field_size_um
        return width * height


@dataclass(frozen=True)
class BroomReport:
    """Timing comparison of an adaptive broom against two regular baselines."""

    samples: int
    rows: int
    field_area_um2: float
    adaptive_s: float
    regular_cruise_s: float
    regular_dwell_s: float
    slow_s: float
    transition_s: float
    fast_s: float
    delay_s: float
    dwell_nominal_s: float
    added_vs_cruise_s: float
    saved_vs_dwell_s: float
    min_dwell_s: float
    mean_dwell_s: float
    throughput_um2_s: float
    samples_per_hour: float

    def __post_init__(self) -> None:
        """Check the counters and timing fields."""
        for name in ("samples", "rows"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        for name in (
            "field_area_um2",
            "adaptive_s",
            "regular_cruise_s",
            "regular_dwell_s",
            "slow_s",
            "transition_s",
            "fast_s",
            "delay_s",
            "dwell_nominal_s",
            "added_vs_cruise_s",
            "saved_vs_dwell_s",
            "min_dwell_s",
            "mean_dwell_s",
            "throughput_um2_s",
            "samples_per_hour",
        ):
            object.__setattr__(
                self, name, _finite(getattr(self, name), name, positive=False)
            )
        if not isclose(
            self.added_vs_cruise_s,
            self.adaptive_s - self.regular_cruise_s,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "added_vs_cruise_s must equal adaptive_s - regular_cruise_s"
            )
        if not isclose(
            self.saved_vs_dwell_s,
            self.regular_dwell_s - self.adaptive_s,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("saved_vs_dwell_s must equal regular_dwell_s - adaptive_s")
        if self.adaptive_s > 0 and not isclose(
            self.throughput_um2_s,
            self.field_area_um2 / self.adaptive_s,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("throughput_um2_s must equal field_area_um2 / adaptive_s")
        if (
            self.samples > 0
            and self.adaptive_s > 0
            and not isclose(
                self.samples_per_hour,
                self.samples * 3600.0 / self.adaptive_s,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        ):
            raise ValueError("samples_per_hour must equal samples * 3600 / adaptive_s")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary."""
        return asdict(self)


def _rest_to_rest(distance_um: float, *, v_max: float, accel: float) -> float:
    """Return a rest-to-rest trapezoidal or triangular move time."""
    distance = abs(float(distance_um))
    if distance == 0:
        return 0.0
    cruise_um = v_max * v_max / accel
    if distance >= cruise_um:
        return distance / v_max + v_max / accel
    return 2 * sqrt(distance / accel)


def _speed_profile(
    sample_xs: Sequence[float],
    *,
    start_um: float,
    length_um: float,
    fov_um: float,
    v_dwell: float,
    v_max: float,
    accel: float,
    step_um: float,
    target_speed: float | None,
) -> NDArray[np.float64]:
    """Return an acceleration-limited speed profile along one row."""
    count = round(length_um / step_um) + 1
    target = np.full(count, v_max if target_speed is None else target_speed)
    if target_speed is None:
        for x in sample_xs:
            low = max(start_um, x - fov_um / 2)
            high = min(start_um + length_um, x + fov_um / 2)
            target[
                int((low - start_um) / step_um) : int((high - start_um) / step_um) + 1
            ] = v_dwell
    speed = target.astype(float)
    speed[0] = 0.0
    speed[-1] = 0.0
    for index in range(count - 1):
        speed[index + 1] = min(
            speed[index + 1], sqrt(speed[index] ** 2 + 2 * accel * step_um)
        )
    for index in range(count - 2, -1, -1):
        speed[index] = min(
            speed[index], sqrt(speed[index + 1] ** 2 + 2 * accel * step_um)
        )
    return speed


def _row_metrics(
    speed: NDArray[np.float64],
    sample_xs: Sequence[float],
    *,
    start_um: float,
    length_um: float,
    fov_um: float,
    v_dwell: float,
    v_max: float,
    step_um: float,
) -> tuple[float, float, float, float, list[float]]:
    """Return row time, slow, transition and fast time, and per-sample dwell."""
    segment_speed = (speed[:-1] + speed[1:]) / 2
    segment_time = step_um / segment_speed
    slow = float(np.sum(segment_time[segment_speed <= v_dwell * 1.02]))
    fast = float(np.sum(segment_time[segment_speed >= v_max * 0.98]))
    total = float(np.sum(segment_time))
    transition = max(0.0, total - slow - fast)
    positions = start_um + np.arange(len(segment_time)) * step_um
    dwells = []
    for x in sample_xs:
        inside = (positions >= max(start_um, x - fov_um / 2)) & (
            positions < min(start_um + length_um, x + fov_um / 2)
        )
        dwells.append(float(np.sum(segment_time[inside])))
    return total, slow, transition, fast, dwells


def simulate_broom_scan(spec: BroomSpec) -> BroomReport:
    """Estimate an adaptive broom raster and its regular-broom baselines.

    ``regular_cruise`` runs the same raster at the travel speed with no slow
    zones; ``regular_dwell`` runs it at the dwell speed everywhere. The dynamic
    run slows only at the declared samples, so ``added_vs_cruise`` is the cost
    of adding the two-second dwell to a regular fast broom and
    ``saved_vs_dwell`` is what speeding up between samples saves relative to a
    regular broom that dwells everywhere.
    """
    if not isinstance(spec, BroomSpec):
        raise TypeError("spec must be a BroomSpec")
    width, _height = spec.field_size_um
    start_um = -spec.fov_um / 2
    length_um = width + spec.fov_um
    rows: list[list[float]] = [[] for _ in range(spec.row_count)]
    for sample in spec.samples:
        x, y = sample.center_xy_um
        rows[min(spec.row_count - 1, int(y / spec.row_pitch_um))].append(x)
    preposition = _rest_to_rest(
        hypot(start_um, spec.row_pitch_um / 2),
        v_max=spec.stage.max_velocity_um_s,
        accel=spec.stage.acceleration_um_s2,
    )
    turn_time = _rest_to_rest(
        spec.row_pitch_um,
        v_max=spec.stage.max_velocity_um_s,
        accel=spec.stage.acceleration_um_s2,
    )
    dynamic = preposition
    cruise = preposition
    dwell = preposition
    slow_total = transition_total = fast_total = 0.0
    dwells: list[float] = []
    for row_index, sample_xs in enumerate(rows):
        ordered = sorted(sample_xs)
        common = {
            "start_um": start_um,
            "length_um": length_um,
            "fov_um": spec.fov_um,
            "v_dwell": spec.dwell_velocity_um_s,
            "v_max": spec.stage.max_velocity_um_s,
            "accel": spec.stage.acceleration_um_s2,
            "step_um": spec.profile_step_um,
        }
        adaptive_speed = _speed_profile(ordered, target_speed=None, **common)
        cruise_speed = _speed_profile(
            ordered,
            target_speed=spec.stage.max_velocity_um_s,
            **common,
        )
        dwell_speed = _speed_profile(
            ordered,
            target_speed=spec.dwell_velocity_um_s,
            **common,
        )
        row_dynamic, slow, transition, fast, row_dwells = _row_metrics(
            adaptive_speed,
            ordered,
            start_um=start_um,
            length_um=length_um,
            fov_um=spec.fov_um,
            v_dwell=spec.dwell_velocity_um_s,
            v_max=spec.stage.max_velocity_um_s,
            step_um=spec.profile_step_um,
        )
        row_cruise = float(
            np.sum(spec.profile_step_um / ((cruise_speed[:-1] + cruise_speed[1:]) / 2))
        )
        row_dwell = float(
            np.sum(spec.profile_step_um / ((dwell_speed[:-1] + dwell_speed[1:]) / 2))
        )
        dynamic += row_dynamic
        cruise += row_cruise
        dwell += row_dwell
        slow_total += slow
        transition_total += transition
        fast_total += fast
        dwells.extend(row_dwells)
        if row_index < spec.row_count - 1:
            dynamic += turn_time
            cruise += turn_time
            dwell += turn_time
    samples = len(spec.samples)
    delay = samples * 2 * spec.command_delay_s
    dynamic += delay
    nominal = samples * spec.dwell_s
    return BroomReport(
        samples=samples,
        rows=spec.row_count,
        field_area_um2=spec.field_area_um2,
        adaptive_s=dynamic,
        regular_cruise_s=cruise,
        regular_dwell_s=dwell,
        slow_s=slow_total,
        transition_s=transition_total,
        fast_s=fast_total,
        delay_s=delay,
        dwell_nominal_s=nominal,
        added_vs_cruise_s=dynamic - cruise,
        saved_vs_dwell_s=dwell - dynamic,
        min_dwell_s=min(dwells) if dwells else 0.0,
        mean_dwell_s=(sum(dwells) / len(dwells)) if dwells else 0.0,
        throughput_um2_s=spec.field_area_um2 / dynamic if dynamic > 0 else 0.0,
        samples_per_hour=samples * 3600.0 / dynamic if dynamic > 0 else 0.0,
    )


__all__ = ["BroomReport", "BroomSpec", "simulate_broom_scan"]
