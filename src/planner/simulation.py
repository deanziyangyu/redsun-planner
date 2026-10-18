"""Deterministic mock simulation of a two-round search-and-image workflow.

Round one places overlapping snap tiles over a declared specimen field; round
two revisits mock target positions for focused imaging. The simulator estimates
stage travel, settle, exposure, readout and per-position overhead from declared
values, then reports area throughput and the equivalent continuous operating
points. It reads no pixels, opens no device and makes no claim about a real
detector's recall: the targets are declared mock positions, not detections.

The two rounds carry independent pixel-size and resolution fields, so a survey
may run at the same magnification as the focused round. A lower-magnification
overview is one supported choice, not an assumption of the model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import ceil, exp, hypot, isclose, isfinite, sqrt
from random import Random
from typing import TYPE_CHECKING, Any

from .acquisition_space import (
    AcquisitionEnvelope,
    AcquisitionMetrics,
    AcquisitionPoint,
    MeasurementLimits,
    MeasurementVerdict,
    OperatingCorner,
    assess_measurement,
    metrics_of,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


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
class StageProfile:
    """Declared linear stage motion used to estimate each move.

    ``travel_time_s`` applies a symmetric trapezoidal profile: a move long
    enough to reach ``max_velocity_um_s`` cruises, a shorter move is triangular.
    ``settle_s`` is added once per nonzero move before the exposure starts.
    """

    max_velocity_um_s: float
    acceleration_um_s2: float
    settle_s: float = 0.0

    def __post_init__(self) -> None:
        """Check the declared motion values."""
        object.__setattr__(
            self,
            "max_velocity_um_s",
            _finite(self.max_velocity_um_s, "max_velocity_um_s", positive=True),
        )
        object.__setattr__(
            self,
            "acceleration_um_s2",
            _finite(self.acceleration_um_s2, "acceleration_um_s2", positive=True),
        )
        object.__setattr__(
            self, "settle_s", _finite(self.settle_s, "settle_s", positive=False)
        )

    def travel_time_s(self, distance_um: float) -> float:
        """Return the trapezoidal move time for one distance, excluding settle."""
        if isinstance(distance_um, bool) or not isinstance(distance_um, (int, float)):
            raise TypeError("distance_um must be a real number")
        distance = abs(float(distance_um))
        if not isfinite(distance):
            raise ValueError("distance_um must be finite")
        if distance == 0:
            return 0.0
        velocity = self.max_velocity_um_s
        acceleration = self.acceleration_um_s2
        cruise_distance = velocity * velocity / acceleration
        if distance >= cruise_distance:
            return distance / velocity + velocity / acceleration
        return 2 * sqrt(distance / acceleration)

    def positioning_time_s(self, distance_um: float) -> float:
        """Return the move plus settle time; a zero move does not settle."""
        if isinstance(distance_um, bool) or not isinstance(distance_um, (int, float)):
            raise TypeError("distance_um must be a real number")
        distance = abs(float(distance_um))
        if not isfinite(distance):
            raise ValueError("distance_um must be finite")
        if distance == 0:
            return 0.0
        return self.travel_time_s(distance) + self.settle_s


@dataclass(frozen=True)
class MockTarget:
    """One declared mock target location in field coordinates."""

    target_id: str
    center_xy_um: tuple[float, float]

    def __post_init__(self) -> None:
        """Check the identifier and center."""
        if not isinstance(self.target_id, str) or not self.target_id.strip():
            raise ValueError("target_id must be a nonempty string")
        object.__setattr__(
            self,
            "center_xy_um",
            _pair(self.center_xy_um, "center_xy_um", positive=False),
        )


def mock_targets(
    field_size_um: tuple[float, float],
    count: int,
    *,
    seed: int = 0,
    margin_um: float = 0.0,
) -> tuple[MockTarget, ...]:
    """Place ``count`` deterministic uniform mock targets inside a field."""
    width, height = _pair(field_size_um, "field_size_um", positive=True)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    margin = _finite(margin_um, "margin_um", positive=False)
    if margin * 2 >= min(width, height):
        raise ValueError("margin_um leaves no room for targets")
    generator = Random(seed)
    return tuple(
        MockTarget(
            f"target-{index:03d}",
            (
                generator.uniform(margin, width - margin),
                generator.uniform(margin, height - margin),
            ),
        )
        for index in range(count)
    )


@dataclass(frozen=True)
class SurveySpec:
    """Round-one snap-tile settings for the overlapping survey."""

    tile_size_um: tuple[float, float]
    overlap: float
    exposure_s: float
    frames: int
    readout_s: float
    overhead_s: float
    pixel_size_um: float
    resolution_um: float

    def __post_init__(self) -> None:
        """Check the declared survey values."""
        object.__setattr__(
            self,
            "tile_size_um",
            _pair(self.tile_size_um, "tile_size_um", positive=True),
        )
        overlap = _finite(self.overlap, "overlap", positive=False)
        if overlap >= 1:
            raise ValueError("overlap must be less than 1")
        object.__setattr__(self, "overlap", overlap)
        object.__setattr__(
            self, "exposure_s", _finite(self.exposure_s, "exposure_s", positive=True)
        )
        if (
            isinstance(self.frames, bool)
            or not isinstance(self.frames, int)
            or self.frames < 1
        ):
            raise ValueError("frames must be a positive integer")
        object.__setattr__(
            self, "readout_s", _finite(self.readout_s, "readout_s", positive=False)
        )
        object.__setattr__(
            self, "overhead_s", _finite(self.overhead_s, "overhead_s", positive=False)
        )
        object.__setattr__(
            self,
            "pixel_size_um",
            _finite(self.pixel_size_um, "pixel_size_um", positive=True),
        )
        object.__setattr__(
            self,
            "resolution_um",
            _finite(self.resolution_um, "resolution_um", positive=True),
        )


@dataclass(frozen=True)
class FocusedSpec:
    """Round-two focused acquisition settings, applied to every target."""

    exposure_s: float
    frames: int
    channels: int
    z_planes: int
    readout_s: float
    overhead_s: float
    pixel_size_um: float
    resolution_um: float
    roi_size_um: tuple[float, float]

    def __post_init__(self) -> None:
        """Check the declared focused values."""
        object.__setattr__(
            self, "exposure_s", _finite(self.exposure_s, "exposure_s", positive=True)
        )
        for name in ("frames", "channels", "z_planes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        object.__setattr__(
            self, "readout_s", _finite(self.readout_s, "readout_s", positive=False)
        )
        object.__setattr__(
            self, "overhead_s", _finite(self.overhead_s, "overhead_s", positive=False)
        )
        object.__setattr__(
            self,
            "pixel_size_um",
            _finite(self.pixel_size_um, "pixel_size_um", positive=True),
        )
        object.__setattr__(
            self,
            "resolution_um",
            _finite(self.resolution_um, "resolution_um", positive=True),
        )
        object.__setattr__(
            self, "roi_size_um", _pair(self.roi_size_um, "roi_size_um", positive=True)
        )


@dataclass(frozen=True)
class SurveyTile:
    """One survey tile: its grid position and stage-coordinate center."""

    tile_id: str
    row: int
    column: int
    center_xy_um: tuple[float, float]


@dataclass(frozen=True)
class SearchImagingSpec:
    """One mock search-and-image run: a field, two rounds and mock targets.

    Round one covers ``field_size_um`` with overlapping snap tiles in serpentine
    order starting near the field origin. Round two visits every declared target
    in nearest-neighbour order from the last survey position. The specification
    is inert: it names no device and asserts no stage envelope.
    """

    field_size_um: tuple[float, float]
    survey: SurveySpec
    focused: FocusedSpec
    stage: StageProfile
    targets: tuple[MockTarget, ...] = ()
    origin_xy_um: tuple[float, float] = (0.0, 0.0)
    review_s: float = 0.0
    bleach_time_constant_s: float | None = None

    def __post_init__(self) -> None:
        """Check the field, targets and origin."""
        object.__setattr__(
            self,
            "field_size_um",
            _pair(self.field_size_um, "field_size_um", positive=True),
        )
        object.__setattr__(
            self,
            "origin_xy_um",
            _pair(self.origin_xy_um, "origin_xy_um", positive=False),
        )
        object.__setattr__(
            self, "review_s", _finite(self.review_s, "review_s", positive=False)
        )
        if self.bleach_time_constant_s is not None:
            object.__setattr__(
                self,
                "bleach_time_constant_s",
                _finite(
                    self.bleach_time_constant_s,
                    "bleach_time_constant_s",
                    positive=True,
                ),
            )
        if not isinstance(self.survey, SurveySpec):
            raise TypeError("survey must be a SurveySpec")
        if not isinstance(self.focused, FocusedSpec):
            raise TypeError("focused must be a FocusedSpec")
        if not isinstance(self.stage, StageProfile):
            raise TypeError("stage must be a StageProfile")
        if not isinstance(self.targets, tuple):
            raise TypeError("targets must be a tuple of MockTarget records")
        width, height = self.field_size_um
        seen: set[str] = set()
        for target in self.targets:
            if not isinstance(target, MockTarget):
                raise TypeError("targets must contain MockTarget records")
            if target.target_id in seen:
                raise ValueError(f"duplicate target id {target.target_id!r}")
            seen.add(target.target_id)
            x, y = target.center_xy_um
            if not (0 <= x <= width and 0 <= y <= height):
                raise ValueError(
                    f"target {target.target_id} center {target.center_xy_um} is outside the field"
                )

    def survey_tiles(self) -> tuple[SurveyTile, ...]:
        """Return the serpentine survey tiles in acquisition order."""
        width, height = self.field_size_um
        tile_width, tile_height = self.survey.tile_size_um
        xs = _axis_centers(width, tile_width, self.survey.overlap)
        ys = _axis_centers(height, tile_height, self.survey.overlap)
        tiles: list[SurveyTile] = []
        for row, y in enumerate(ys):
            columns = (
                tuple(range(len(xs) - 1, -1, -1)) if row % 2 else tuple(range(len(xs)))
            )
            for column in columns:
                tiles.append(
                    SurveyTile(
                        f"tile-r{row:03d}-c{column:03d}",
                        row,
                        column,
                        (xs[column], y),
                    )
                )
        return tuple(tiles)

    def survey_centers(self) -> tuple[tuple[float, float], ...]:
        """Return the serpentine survey tile centers in acquisition order."""
        return tuple(tile.center_xy_um for tile in self.survey_tiles())

    def overscan_um(self) -> tuple[float, float]:
        """Return how far the last tile edge extends past the field on each axis."""
        width, height = self.field_size_um
        tile_width, tile_height = self.survey.tile_size_um
        xs = _axis_centers(width, tile_width, self.survey.overlap)
        ys = _axis_centers(height, tile_height, self.survey.overlap)
        return (
            max(0.0, xs[-1] + tile_width / 2 - width),
            max(0.0, ys[-1] + tile_height / 2 - height),
        )


def _axis_centers(extent: float, tile: float, overlap: float) -> tuple[float, ...]:
    """Return tile centers covering ``extent`` with at most ``step`` spacing."""
    if tile >= extent:
        return (extent / 2,)
    step = tile * (1 - overlap)
    count = max(1, ceil((extent - tile) / step) + 1)
    return tuple(tile / 2 + index * step for index in range(count))


@dataclass(frozen=True)
class RoundReport:
    """Estimated time breakdown for one round."""

    positions: int
    distance_um: float
    travel_s: float
    settle_s: float
    exposure_s: float
    readout_s: float
    overhead_s: float
    total_s: float

    def __post_init__(self) -> None:
        """Check the counters and the summed total."""
        if isinstance(self.positions, bool) or not isinstance(self.positions, int):
            raise TypeError("positions must be an integer")
        if self.positions < 0:
            raise ValueError("positions must be nonnegative")
        for name in (
            "distance_um",
            "travel_s",
            "settle_s",
            "exposure_s",
            "readout_s",
            "overhead_s",
            "total_s",
        ):
            object.__setattr__(
                self, name, _finite(getattr(self, name), name, positive=False)
            )
        summed = (
            self.travel_s
            + self.settle_s
            + self.exposure_s
            + self.readout_s
            + self.overhead_s
        )
        if not isclose(summed, self.total_s, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("total_s must equal the summed time components")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class ScanMetrics:
    """Reporting metrics for one sub-scan of the dual-scan routine.

    ``area_um2`` is the survey field for the primary scan and the summed target
    ROI area for the focused scan. Fractions of the sub-scan duration are in
    ``[0, 1]``; an empty focused scan reports zeros.
    """

    scan_id: str
    positions: int
    area_um2: float
    duration_s: float
    throughput_um2_s: float
    seconds_per_position: float
    travel_fraction: float
    exposure_fraction: float

    def __post_init__(self) -> None:
        """Check the metrics and their internal consistency."""
        if not isinstance(self.scan_id, str) or not self.scan_id.strip():
            raise ValueError("scan_id must be a nonempty string")
        if isinstance(self.positions, bool) or not isinstance(self.positions, int):
            raise TypeError("positions must be an integer")
        if self.positions < 0:
            raise ValueError("positions must be nonnegative")
        for name in (
            "area_um2",
            "duration_s",
            "throughput_um2_s",
            "seconds_per_position",
        ):
            object.__setattr__(
                self, name, _finite(getattr(self, name), name, positive=False)
            )
        for name in ("travel_fraction", "exposure_fraction"):
            value = _finite(getattr(self, name), name, positive=False)
            if value > 1:
                raise ValueError(f"{name} must be in [0, 1]")
            object.__setattr__(self, name, value)
        if self.duration_s > 0:
            if not isclose(
                self.throughput_um2_s,
                self.area_um2 / self.duration_s,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError("throughput_um2_s must equal area_um2 / duration_s")
        elif self.throughput_um2_s != 0:
            raise ValueError("an empty sub-scan has zero throughput")
        if self.positions > 0:
            if not isclose(
                self.seconds_per_position,
                self.duration_s / self.positions,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "seconds_per_position must equal duration_s / positions"
                )
        elif self.seconds_per_position != 0:
            raise ValueError("an empty sub-scan has zero seconds per position")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class AveragedMetrics:
    """Unweighted means across the nonempty sub-scans plus the combined run.

    The means average only sub-scans that acquired at least one position, so an
    empty focused round does not halve the survey values. ``combined`` figures
    use the summed sub-scan duration and the sum of the surveyed field area and
    the focused target ROI area; that is acquired image area, not unique
    specimen area.
    """

    mean_throughput_um2_s: float
    mean_seconds_per_position: float
    mean_travel_fraction: float
    mean_exposure_fraction: float
    combined_acquired_area_um2: float
    combined_throughput_um2_s: float
    total_duration_s: float

    def __post_init__(self) -> None:
        """Check the averages and fractions."""
        for name in (
            "mean_throughput_um2_s",
            "mean_seconds_per_position",
            "combined_acquired_area_um2",
            "combined_throughput_um2_s",
            "total_duration_s",
        ):
            object.__setattr__(
                self, name, _finite(getattr(self, name), name, positive=False)
            )
        for name in ("mean_travel_fraction", "mean_exposure_fraction"):
            value = _finite(getattr(self, name), name, positive=False)
            if value > 1:
                raise ValueError(f"{name} must be in [0, 1]")
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class PhotobleachingReport:
    """Pre-focus exposure each sample received from the overview.

    ``cost`` is the prior overview exposure divided by the focused exposure, so
    it is relative to 1: 0 means no preview exposure, 1 means the preview
    delivered as much light as the focused scan. It does not depend on a decay
    model. When ``bleach_time_constant_s`` is declared, ``retained`` is the
    log-linear brightness fraction ``exp(-prior_exposure_s / tau)`` before the
    focused scan (1 = unbleached) and ``loss_fraction`` is ``1 - retained``.
    """

    samples: int
    preview_exposure_s: float
    mean_cover_count: float
    prior_exposure_s_min: float
    prior_exposure_s_mean: float
    prior_exposure_s_max: float
    focused_exposure_s: float
    cost_min: float
    cost_mean: float
    cost_max: float
    bleach_time_constant_s: float | None = None
    retained_min: float | None = None
    retained_mean: float | None = None
    retained_max: float | None = None
    loss_fraction_min: float | None = None
    loss_fraction_mean: float | None = None
    loss_fraction_max: float | None = None

    def __post_init__(self) -> None:
        """Check the counters, fractions and optional decay fields."""
        if isinstance(self.samples, bool) or not isinstance(self.samples, int):
            raise TypeError("samples must be an integer")
        if self.samples < 0:
            raise ValueError("samples must be nonnegative")
        for name in (
            "preview_exposure_s",
            "mean_cover_count",
            "prior_exposure_s_min",
            "prior_exposure_s_mean",
            "prior_exposure_s_max",
            "focused_exposure_s",
            "cost_min",
            "cost_mean",
            "cost_max",
        ):
            object.__setattr__(
                self, name, _finite(getattr(self, name), name, positive=False)
            )
        if self.bleach_time_constant_s is not None:
            object.__setattr__(
                self,
                "bleach_time_constant_s",
                _finite(
                    self.bleach_time_constant_s,
                    "bleach_time_constant_s",
                    positive=True,
                ),
            )
        for name in ("retained_min", "retained_mean", "retained_max"):
            value = getattr(self, name)
            if value is not None:
                number = _finite(value, name, positive=False)
                if number > 1:
                    raise ValueError(f"{name} must be in [0, 1]")
                object.__setattr__(self, name, number)
        for name in ("loss_fraction_min", "loss_fraction_mean", "loss_fraction_max"):
            value = getattr(self, name)
            if value is not None:
                number = _finite(value, name, positive=False)
                if number > 1:
                    raise ValueError(f"{name} must be in [0, 1]")
                object.__setattr__(self, name, number)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class SimulationReport:
    """Estimated throughput, timing and equivalent operating points."""

    field_area_um2: float
    targets: int
    survey: RoundReport
    focused: RoundReport
    review_s: float
    total_s: float
    survey_throughput_um2_s: float
    end_to_end_throughput_um2_s: float
    targets_per_hour: float
    survey_order_ids: tuple[str, ...]
    focused_order_ids: tuple[str, ...]
    overscan_um: tuple[float, float]
    survey_acquisition: AcquisitionPoint
    survey_acquisition_metrics: AcquisitionMetrics
    survey_equivalent: AcquisitionPoint
    survey_equivalent_metrics: AcquisitionMetrics
    focused_point: AcquisitionPoint
    focused_metrics: AcquisitionMetrics
    survey_scan: ScanMetrics
    focused_scan: ScanMetrics
    averaged: AveragedMetrics
    photobleaching: PhotobleachingReport
    survey_acquisition_verdict: MeasurementVerdict | None = None
    focused_verdict: MeasurementVerdict | None = None
    survey_acquisition_corner: OperatingCorner | None = None
    survey_equivalent_corner: OperatingCorner | None = None
    focused_corner: OperatingCorner | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary, including derived points and verdicts."""
        return {
            "field_area_um2": self.field_area_um2,
            "targets": self.targets,
            "survey": self.survey.to_dict(),
            "focused": self.focused.to_dict(),
            "review_s": self.review_s,
            "total_s": self.total_s,
            "survey_throughput_um2_s": self.survey_throughput_um2_s,
            "end_to_end_throughput_um2_s": self.end_to_end_throughput_um2_s,
            "targets_per_hour": self.targets_per_hour,
            "survey_order_ids": list(self.survey_order_ids),
            "focused_order_ids": list(self.focused_order_ids),
            "overscan_um": list(self.overscan_um),
            "sub_scans": {
                "survey": self.survey_scan.to_dict(),
                "focused": self.focused_scan.to_dict(),
            },
            "averaged": self.averaged.to_dict(),
            "photobleaching": self.photobleaching.to_dict(),
            "survey_acquisition": self.survey_acquisition.to_dict(),
            "survey_acquisition_metrics": self.survey_acquisition_metrics.to_dict(),
            "survey_equivalent": self.survey_equivalent.to_dict(),
            "survey_equivalent_metrics": self.survey_equivalent_metrics.to_dict(),
            "focused_point": self.focused_point.to_dict(),
            "focused_metrics": self.focused_metrics.to_dict(),
            "survey_acquisition_verdict": _verdict_dict(
                self.survey_acquisition_verdict
            ),
            "focused_verdict": _verdict_dict(self.focused_verdict),
            "survey_acquisition_corner": _corner_value(self.survey_acquisition_corner),
            "survey_equivalent_corner": _corner_value(self.survey_equivalent_corner),
            "focused_corner": _corner_value(self.focused_corner),
        }


def _verdict_dict(verdict: MeasurementVerdict | None) -> dict[str, Any] | None:
    """Return a verdict as a JSON-safe dictionary, or ``None``."""
    return None if verdict is None else verdict.to_dict()


def _corner_value(corner: OperatingCorner | None) -> str | None:
    """Return a corner code as a string, or ``None``."""
    return None if corner is None else corner.value


def _focused_order(
    targets: Iterable[MockTarget], start_xy_um: tuple[float, float]
) -> tuple[MockTarget, ...]:
    """Order targets greedily by nearest neighbour from ``start_xy_um``."""
    remaining = list(targets)
    ordered: list[MockTarget] = []
    current = start_xy_um
    while remaining:
        candidate = min(
            remaining,
            key=lambda target: (
                hypot(
                    target.center_xy_um[0] - current[0],
                    target.center_xy_um[1] - current[1],
                ),
                target.target_id,
            ),
        )
        ordered.append(candidate)
        remaining.remove(candidate)
        current = candidate.center_xy_um
    return tuple(ordered)


def _survey_round(
    spec: SearchImagingSpec,
) -> tuple[RoundReport, tuple[float, float]]:
    """Estimate the survey round and return its final stage position."""
    tiles = spec.survey_tiles()
    stage = spec.stage
    survey = spec.survey
    previous = spec.origin_xy_um
    distance = travel = settle = 0.0
    for tile in tiles:
        step = hypot(
            tile.center_xy_um[0] - previous[0], tile.center_xy_um[1] - previous[1]
        )
        distance += step
        travel += stage.travel_time_s(distance_um=step)
        if step > 0:
            settle += stage.settle_s
        previous = tile.center_xy_um
    frames = survey.frames
    exposure = survey.exposure_s * frames * len(tiles)
    readout = survey.readout_s * frames * len(tiles)
    overhead = survey.overhead_s * len(tiles)
    return (
        RoundReport(
            positions=len(tiles),
            distance_um=distance,
            travel_s=travel,
            settle_s=settle,
            exposure_s=exposure,
            readout_s=readout,
            overhead_s=overhead,
            total_s=travel + settle + exposure + readout + overhead,
        ),
        tiles[-1].center_xy_um if tiles else spec.origin_xy_um,
    )


def _focused_round(
    spec: SearchImagingSpec, start_xy_um: tuple[float, float]
) -> tuple[RoundReport, tuple[MockTarget, ...]]:
    """Estimate the focused round and return the visit order."""
    ordered = _focused_order(spec.targets, start_xy_um)
    stage = spec.stage
    focused = spec.focused
    previous = start_xy_um
    distance = travel = settle = 0.0
    for target in ordered:
        step = hypot(
            target.center_xy_um[0] - previous[0], target.center_xy_um[1] - previous[1]
        )
        distance += step
        travel += stage.travel_time_s(distance_um=step)
        if step > 0:
            settle += stage.settle_s
        previous = target.center_xy_um
    frames = focused.frames * focused.channels * focused.z_planes
    return (
        RoundReport(
            positions=len(ordered),
            distance_um=distance,
            travel_s=travel,
            settle_s=settle,
            exposure_s=focused.exposure_s * frames * len(ordered),
            readout_s=focused.readout_s * frames * len(ordered),
            overhead_s=focused.overhead_s * len(ordered),
            total_s=travel
            + settle
            + (focused.exposure_s + focused.readout_s) * frames * len(ordered)
            + focused.overhead_s * len(ordered),
        ),
        ordered,
    )


def _stats(values: list[float]) -> tuple[float, float, float]:
    """Return ``(min, mean, max)`` with zeros for an empty sequence."""
    if not values:
        return 0.0, 0.0, 0.0
    return min(values), sum(values) / len(values), max(values)


def _photobleaching_report(spec: SearchImagingSpec) -> PhotobleachingReport:
    """Count overview exposures on every sample and derive its bleach cost."""
    tiles = spec.survey_tiles()
    tile_width, tile_height = spec.survey.tile_size_um
    counts: list[int] = []
    for target in spec.targets:
        x, y = target.center_xy_um
        counts.append(
            sum(
                1
                for tile in tiles
                if abs(tile.center_xy_um[0] - x) <= tile_width / 2
                and abs(tile.center_xy_um[1] - y) <= tile_height / 2
            )
        )
    preview_exposure = spec.survey.exposure_s * spec.survey.frames
    focused_exposure = spec.focused.exposure_s * spec.focused.frames
    prior = [count * preview_exposure for count in counts]
    cost = [value / focused_exposure for value in prior]
    prior_min, prior_mean, prior_max = _stats(prior)
    cost_min, cost_mean, cost_max = _stats(cost)
    mean_count = sum(counts) / len(counts) if counts else 0.0
    report = PhotobleachingReport(
        samples=len(counts),
        preview_exposure_s=preview_exposure,
        mean_cover_count=mean_count,
        prior_exposure_s_min=prior_min,
        prior_exposure_s_mean=prior_mean,
        prior_exposure_s_max=prior_max,
        focused_exposure_s=focused_exposure,
        cost_min=cost_min,
        cost_mean=cost_mean,
        cost_max=cost_max,
    )
    tau = spec.bleach_time_constant_s
    if tau is None:
        return report
    retained = [exp(-value / tau) for value in prior]
    loss = [1.0 - value for value in retained]
    retained_min, retained_mean, retained_max = _stats(retained)
    loss_min, loss_mean, loss_max = _stats(loss)
    return replace(
        report,
        bleach_time_constant_s=tau,
        retained_min=retained_min,
        retained_mean=retained_mean,
        retained_max=retained_max,
        loss_fraction_min=loss_min,
        loss_fraction_mean=loss_mean,
        loss_fraction_max=loss_max,
    )


def simulate_search_imaging(
    spec: SearchImagingSpec,
    *,
    envelope: AcquisitionEnvelope | None = None,
    limits: MeasurementLimits | None = None,
) -> SimulationReport:
    """Simulate one mock two-round run and return its throughput report.

    Round one is reported twice. ``survey_acquisition`` is the configured snap
    tile (stationary exposure), while ``survey_equivalent`` is the continuous
    operating point whose ideal area throughput equals the simulated survey
    throughput; its motion-blur terms are a modelling convenience, not a claim
    about a snap exposure. ``focused_point`` is the stationary focused
    acquisition. Corners need ``envelope``; feasibility verdicts need ``limits``.
    """
    if not isinstance(spec, SearchImagingSpec):
        raise TypeError("spec must be a SearchImagingSpec")
    width, height = spec.field_size_um
    field_area = width * height
    survey, last_center = _survey_round(spec)
    focused, order = _focused_round(spec, last_center)
    total = survey.total_s + spec.review_s + focused.total_s
    survey_throughput = field_area / survey.total_s if survey.total_s > 0 else 0.0
    end_to_end = field_area / total if total > 0 else 0.0
    tile_width, tile_height = spec.survey.tile_size_um
    survey_period = survey.total_s / survey.positions if survey.positions else 0.0
    survey_point = AcquisitionPoint(
        velocity_um_s=0.0,
        exposure_s=spec.survey.exposure_s,
        width_um=tile_width,
        frame_period_s=max(survey_period, spec.survey.exposure_s),
        pixel_size_um=spec.survey.pixel_size_um,
        resolution_um=spec.survey.resolution_um,
        height_um=tile_height,
    )
    equivalent_velocity = survey_throughput / tile_height if tile_height > 0 else 0.0
    equivalent_point = AcquisitionPoint(
        velocity_um_s=equivalent_velocity,
        exposure_s=spec.survey.exposure_s,
        width_um=tile_width,
        frame_period_s=max(survey_period, spec.survey.exposure_s),
        pixel_size_um=spec.survey.pixel_size_um,
        resolution_um=spec.survey.resolution_um,
        height_um=tile_height,
    )
    focused_period = focused.total_s / focused.positions if focused.positions else 0.0
    focused_point = AcquisitionPoint(
        velocity_um_s=0.0,
        exposure_s=spec.focused.exposure_s,
        width_um=spec.focused.roi_size_um[0],
        frame_period_s=max(focused_period, spec.focused.exposure_s),
        pixel_size_um=spec.focused.pixel_size_um,
        resolution_um=spec.focused.resolution_um,
        height_um=spec.focused.roi_size_um[1],
    )
    survey_metrics = metrics_of(survey_point)
    equivalent_metrics = metrics_of(equivalent_point)
    focused_metrics = metrics_of(focused_point)
    survey_verdict = (
        None if limits is None else assess_measurement(survey_point, limits)
    )
    focused_verdict = (
        None if limits is None else assess_measurement(focused_point, limits)
    )
    survey_scan = ScanMetrics(
        scan_id="survey",
        positions=survey.positions,
        area_um2=field_area,
        duration_s=survey.total_s,
        throughput_um2_s=survey_throughput,
        seconds_per_position=(
            survey.total_s / survey.positions if survey.positions else 0.0
        ),
        travel_fraction=(
            survey.travel_s / survey.total_s if survey.total_s > 0 else 0.0
        ),
        exposure_fraction=(
            survey.exposure_s / survey.total_s if survey.total_s > 0 else 0.0
        ),
    )
    focused_area = (
        len(spec.targets) * spec.focused.roi_size_um[0] * spec.focused.roi_size_um[1]
    )
    focused_scan = ScanMetrics(
        scan_id="focused",
        positions=focused.positions,
        area_um2=focused_area,
        duration_s=focused.total_s,
        throughput_um2_s=(
            focused_area / focused.total_s if focused.total_s > 0 else 0.0
        ),
        seconds_per_position=(
            focused.total_s / focused.positions if focused.positions else 0.0
        ),
        travel_fraction=(
            focused.travel_s / focused.total_s if focused.total_s > 0 else 0.0
        ),
        exposure_fraction=(
            focused.exposure_s / focused.total_s if focused.total_s > 0 else 0.0
        ),
    )
    active = tuple(scan for scan in (survey_scan, focused_scan) if scan.positions > 0)
    averaged = AveragedMetrics(
        mean_throughput_um2_s=(
            sum(scan.throughput_um2_s for scan in active) / len(active)
            if active
            else 0.0
        ),
        mean_seconds_per_position=(
            sum(scan.seconds_per_position for scan in active) / len(active)
            if active
            else 0.0
        ),
        mean_travel_fraction=(
            sum(scan.travel_fraction for scan in active) / len(active)
            if active
            else 0.0
        ),
        mean_exposure_fraction=(
            sum(scan.exposure_fraction for scan in active) / len(active)
            if active
            else 0.0
        ),
        combined_acquired_area_um2=field_area + focused_area,
        combined_throughput_um2_s=(
            (field_area + focused_area) / (survey.total_s + focused.total_s)
            if survey.total_s + focused.total_s > 0
            else 0.0
        ),
        total_duration_s=total,
    )
    photobleaching = _photobleaching_report(spec)
    return SimulationReport(
        field_area_um2=field_area,
        targets=len(spec.targets),
        survey=survey,
        focused=focused,
        review_s=spec.review_s,
        total_s=total,
        survey_throughput_um2_s=survey_throughput,
        end_to_end_throughput_um2_s=end_to_end,
        targets_per_hour=(0.0 if total == 0 else len(spec.targets) * 3600.0 / total),
        survey_order_ids=tuple(tile.tile_id for tile in spec.survey_tiles()),
        focused_order_ids=tuple(target.target_id for target in order),
        overscan_um=spec.overscan_um(),
        survey_acquisition=survey_point,
        survey_acquisition_metrics=survey_metrics,
        survey_equivalent=equivalent_point,
        survey_equivalent_metrics=equivalent_metrics,
        focused_point=focused_point,
        focused_metrics=focused_metrics,
        survey_scan=survey_scan,
        focused_scan=focused_scan,
        averaged=averaged,
        photobleaching=photobleaching,
        survey_acquisition_verdict=survey_verdict,
        focused_verdict=focused_verdict,
        survey_acquisition_corner=(
            None if envelope is None else envelope.corner(survey_point)
        ),
        survey_equivalent_corner=(
            None if envelope is None else envelope.corner(equivalent_point)
        ),
        focused_corner=None if envelope is None else envelope.corner(focused_point),
    )


__all__ = [
    "AveragedMetrics",
    "FocusedSpec",
    "MockTarget",
    "PhotobleachingReport",
    "RoundReport",
    "ScanMetrics",
    "SearchImagingSpec",
    "SimulationReport",
    "StageProfile",
    "SurveySpec",
    "SurveyTile",
    "mock_targets",
    "simulate_search_imaging",
]
