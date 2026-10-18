"""Command-line entry point for simulation, validation and spec comparison.

The commands read the JSON specification documented in ``docs/search_imaging.md``
and ``README.md``, build the hardware-free :class:`planner.simulation.
SearchImagingSpec`, and report the deterministic estimate. No device is opened
and no RunEngine is created. Exit codes are stable: ``0`` success or valid,
``1`` a semantically invalid specification, ``2`` an unreadable, unparseable or
unknown-key specification.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import ceil, isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .acquisition_space import (
    AcquisitionAxis,
    AcquisitionEnvelope,
    AcquisitionPoint,
    AxisBand,
    AxisThreshold,
    MeasurementLimits,
)
from .broom import BroomSpec, simulate_broom_scan
from .operating_map import broom_trace, build_operating_payload, snap_traces
from .report_html import write_comparison_html, write_simulation_html
from .simulation import (
    FocusedSpec,
    MockTarget,
    RoundReport,
    ScanMetrics,
    SearchImagingSpec,
    SimulationReport,
    StageProfile,
    SurveySpec,
    mock_targets,
    simulate_search_imaging,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

MAX_FRAMES_PER_POSITION = 10_000
MAX_SURVEY_TILES = 10_000

_EXIT_OK = 0
_EXIT_INVALID = 1
_EXIT_PARSE = 2

_SPEC_KEYS = {
    "field_size_um",
    "survey",
    "focused",
    "stage",
    "targets",
    "origin_xy_um",
    "review_s",
    "bleach_time_constant_s",
    "envelope",
    "limits",
}
_SURVEY_KEYS = {
    "tile_size_um",
    "overlap",
    "exposure_s",
    "frames",
    "readout_s",
    "overhead_s",
    "pixel_size_um",
    "resolution_um",
}
_FOCUSED_KEYS = {
    "exposure_s",
    "frames",
    "channels",
    "z_planes",
    "readout_s",
    "overhead_s",
    "pixel_size_um",
    "resolution_um",
    "roi_size_um",
}
_STAGE_KEYS = {"max_velocity_um_s", "acceleration_um_s2", "settle_s"}
_ENVELOPE_KEYS = {"velocity_um_s", "exposure_s", "width_um"}
_LIMITS_KEYS = {
    "max_blur_ratio",
    "max_gamma",
    "min_observations",
    "min_duty",
    "max_duty",
}
_METRIC_KEYS = (
    ("tile_count", "tile count", 0),
    ("target_count", "target count", 0),
    ("survey_time_s", "survey time s", 3),
    ("focused_time_s", "focused time s", 3),
    ("total_time_s", "total time s", 3),
    ("survey_throughput_um2_s", "survey throughput um2/s", 1),
    ("focused_throughput_um2_s", "focused throughput um2/s", 1),
    ("mean_sub_scan_throughput_um2_s", "mean sub-scan throughput um2/s", 1),
    ("combined_throughput_um2_s", "combined throughput um2/s", 1),
    ("survey_seconds_per_position", "survey s/position", 3),
    ("focused_seconds_per_position", "focused s/position", 3),
    ("end_to_end_throughput_um2_s", "end-to-end throughput um2/s", 1),
    ("targets_per_hour", "targets/hour", 1),
)
_CORNER_KEYS = (
    ("survey_configured", "survey configured"),
    ("survey_equivalent", "survey equivalent"),
    ("focused", "focused"),
)
_ABSENT = object()


class SpecParseError(ValueError):
    """Raised when a JSON spec is unreadable, malformed or has unknown keys."""


class SpecValidationError(ValueError):
    """Raised when a parsed spec declares semantically invalid values."""


@dataclass(frozen=True)
class _SpecInput:
    """A parsed specification plus the optional checks and canonical form."""

    spec: SearchImagingSpec
    envelope: AcquisitionEnvelope | None
    limits: MeasurementLimits | None
    canonical: dict[str, Any]


@dataclass(frozen=True)
class _ValidationOutcome:
    """The simulated report and the validation findings for one spec."""

    report: SimulationReport
    issues: tuple[str, ...]
    warnings: tuple[str, ...]
    tile_rows: int
    tile_columns: int
    overscan_um: tuple[float, float]


def _mapping(value: Any, path: str) -> dict[str, Any]:
    """Return a JSON object at ``path`` or raise a parse error."""
    if not isinstance(value, dict):
        raise SpecParseError(f"{path} must be a JSON object")
    return value


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise SpecParseError(f"{path} has unknown key(s): {', '.join(unknown)}")


def _require(value: Mapping[str, Any], keys: Sequence[str], path: str) -> None:
    missing = [key for key in keys if key not in value]
    if missing:
        raise SpecParseError(f"{path} is missing required key(s): {', '.join(missing)}")


def _number(value: Any, path: str) -> float:
    """Return a finite JSON number or raise a parse error."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpecParseError(f"{path} must be a number")
    number = float(value)
    if not isfinite(number):
        raise SpecParseError(f"{path} must be finite")
    return number


def _integer(value: Any, path: str) -> int:
    """Return a JSON integer or raise a parse error."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpecParseError(f"{path} must be an integer")
    return value


def _pair(value: Any, path: str) -> tuple[float, float]:
    """Return a two-number JSON array or raise a parse error."""
    if not isinstance(value, list) or len(value) != 2:
        raise SpecParseError(f"{path} must be a two-number array")
    return (_number(value[0], f"{path}[0]"), _number(value[1], f"{path}[1]"))


def _pairs(value: Any, path: str) -> tuple[tuple[float, float], ...]:
    """Return an array of two-number arrays or raise a parse error."""
    if not isinstance(value, list):
        raise SpecParseError(f"{path} must be an array of two-number arrays")
    return tuple(_pair(item, f"{path}[{index}]") for index, item in enumerate(value))


def _json_ready(value: Any) -> Any:
    """Return tuples as lists and mappings with string keys, recursively."""
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _parse_survey(value: Any) -> SurveySpec:
    """Build a survey round specification from its JSON object."""
    payload = _mapping(value, "survey")
    _reject_unknown(payload, _SURVEY_KEYS, "survey")
    _require(payload, sorted(_SURVEY_KEYS), "survey")
    tile_size = _pair(payload["tile_size_um"], "survey.tile_size_um")
    exposure = _number(payload["exposure_s"], "survey.exposure_s")
    frames = _integer(payload["frames"], "survey.frames")
    readout = _number(payload["readout_s"], "survey.readout_s")
    overhead = _number(payload["overhead_s"], "survey.overhead_s")
    pixel_size = _number(payload["pixel_size_um"], "survey.pixel_size_um")
    resolution = _number(payload["resolution_um"], "survey.resolution_um")
    overlap = _number(payload["overlap"], "survey.overlap")
    try:
        return SurveySpec(
            tile_size_um=tile_size,
            overlap=overlap,
            exposure_s=exposure,
            frames=frames,
            readout_s=readout,
            overhead_s=overhead,
            pixel_size_um=pixel_size,
            resolution_um=resolution,
        )
    except (TypeError, ValueError) as error:
        raise SpecValidationError(f"survey: {error}") from error


def _parse_focused(value: Any) -> FocusedSpec:
    """Build a focused round specification from its JSON object."""
    payload = _mapping(value, "focused")
    _reject_unknown(payload, _FOCUSED_KEYS, "focused")
    _require(payload, sorted(_FOCUSED_KEYS), "focused")
    roi_size = _pair(payload["roi_size_um"], "focused.roi_size_um")
    exposure = _number(payload["exposure_s"], "focused.exposure_s")
    frames = _integer(payload["frames"], "focused.frames")
    channels = _integer(payload["channels"], "focused.channels")
    z_planes = _integer(payload["z_planes"], "focused.z_planes")
    readout = _number(payload["readout_s"], "focused.readout_s")
    overhead = _number(payload["overhead_s"], "focused.overhead_s")
    pixel_size = _number(payload["pixel_size_um"], "focused.pixel_size_um")
    resolution = _number(payload["resolution_um"], "focused.resolution_um")
    try:
        return FocusedSpec(
            exposure_s=exposure,
            frames=frames,
            channels=channels,
            z_planes=z_planes,
            readout_s=readout,
            overhead_s=overhead,
            pixel_size_um=pixel_size,
            resolution_um=resolution,
            roi_size_um=roi_size,
        )
    except (TypeError, ValueError) as error:
        raise SpecValidationError(f"focused: {error}") from error


def _parse_stage(value: Any) -> StageProfile:
    """Build a stage profile from its JSON object."""
    payload = _mapping(value, "stage")
    _reject_unknown(payload, _STAGE_KEYS, "stage")
    _require(payload, sorted(_STAGE_KEYS), "stage")
    velocity = _number(payload["max_velocity_um_s"], "stage.max_velocity_um_s")
    acceleration = _number(payload["acceleration_um_s2"], "stage.acceleration_um_s2")
    settle = _number(payload["settle_s"], "stage.settle_s")
    try:
        return StageProfile(velocity, acceleration, settle)
    except (TypeError, ValueError) as error:
        raise SpecValidationError(f"stage: {error}") from error


def _parse_envelope(value: Any) -> AcquisitionEnvelope:
    """Build a declared acquisition envelope from its JSON object."""
    payload = _mapping(value, "envelope")
    _reject_unknown(payload, _ENVELOPE_KEYS, "envelope")
    _require(payload, sorted(_ENVELOPE_KEYS), "envelope")
    try:
        return AcquisitionEnvelope(
            velocity_um_s=AxisThreshold(
                *_pair(payload["velocity_um_s"], "envelope.velocity_um_s")
            ),
            exposure_s=AxisThreshold(
                *_pair(payload["exposure_s"], "envelope.exposure_s")
            ),
            width_um=AxisThreshold(*_pair(payload["width_um"], "envelope.width_um")),
        )
    except (TypeError, ValueError) as error:
        raise SpecValidationError(f"envelope: {error}") from error


def _parse_limits(value: Any) -> MeasurementLimits:
    """Build declared measurement limits from their JSON object."""
    payload = _mapping(value, "limits")
    _reject_unknown(payload, _LIMITS_KEYS, "limits")
    _require(payload, ["max_blur_ratio", "max_gamma"], "limits")
    max_blur_ratio = _number(payload["max_blur_ratio"], "limits.max_blur_ratio")
    max_gamma = _number(payload["max_gamma"], "limits.max_gamma")
    min_observations = (
        _number(payload["min_observations"], "limits.min_observations")
        if "min_observations" in payload
        else None
    )
    min_duty = (
        _number(payload["min_duty"], "limits.min_duty")
        if "min_duty" in payload
        else None
    )
    max_duty = (
        _number(payload["max_duty"], "limits.max_duty")
        if "max_duty" in payload
        else None
    )
    try:
        return MeasurementLimits(
            max_blur_ratio=max_blur_ratio,
            max_gamma=max_gamma,
            min_observations=min_observations,
            min_duty=min_duty,
            max_duty=max_duty,
        )
    except (TypeError, ValueError) as error:
        raise SpecValidationError(f"limits: {error}") from error


def _parse_targets(
    value: Any, field_size_um: tuple[float, float]
) -> tuple[tuple[MockTarget, ...], dict[str, Any]]:
    """Build mock targets and their canonical diff form from the JSON object."""
    payload = _mapping(value, "targets")
    if "positions_xy_um" in payload:
        _reject_unknown(payload, {"positions_xy_um"}, "targets")
        positions = _pairs(payload["positions_xy_um"], "targets.positions_xy_um")
        targets = tuple(
            MockTarget(f"target-{index:03d}", center)
            for index, center in enumerate(positions)
        )
        canonical: dict[str, Any] = {
            "count": len(targets),
            "positions_xy_um": [[x, y] for x, y in positions],
        }
        return targets, canonical
    _reject_unknown(payload, {"count", "seed", "margin_um"}, "targets")
    _require(payload, ["count"], "targets")
    count = _integer(payload["count"], "targets.count")
    seed = _integer(payload["seed"], "targets.seed") if "seed" in payload else 0
    margin = (
        _number(payload["margin_um"], "targets.margin_um")
        if "margin_um" in payload
        else 0.0
    )
    try:
        targets = mock_targets(field_size_um, count, seed=seed, margin_um=margin)
    except (TypeError, ValueError) as error:
        raise SpecValidationError(f"targets: {error}") from error
    return targets, {"count": count, "seed": seed, "margin_um": margin}


def _canonical(
    spec: SearchImagingSpec,
    envelope: AcquisitionEnvelope | None,
    limits: MeasurementLimits | None,
    targets_canonical: dict[str, Any],
) -> dict[str, Any]:
    """Return the canonical form compared by ``diff``."""
    return {
        "field_size_um": list(spec.field_size_um),
        "survey": _json_ready(asdict(spec.survey)),
        "focused": _json_ready(asdict(spec.focused)),
        "stage": _json_ready(asdict(spec.stage)),
        "targets": targets_canonical,
        "origin_xy_um": list(spec.origin_xy_um),
        "review_s": spec.review_s,
        "bleach_time_constant_s": spec.bleach_time_constant_s,
        "envelope": (
            None
            if envelope is None
            else {
                "velocity_um_s": [
                    envelope.velocity_um_s.low_max,
                    envelope.velocity_um_s.high_min,
                ],
                "exposure_s": [
                    envelope.exposure_s.low_max,
                    envelope.exposure_s.high_min,
                ],
                "width_um": [
                    envelope.width_um.low_max,
                    envelope.width_um.high_min,
                ],
            }
        ),
        "limits": (None if limits is None else _json_ready(asdict(limits))),
    }


def _load_envelope(path: Path) -> AcquisitionEnvelope:
    """Read a declared acquisition envelope from JSON."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise SpecParseError(f"cannot read {path}: {error}") from error
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise SpecParseError(f"{path}: invalid JSON: {error.msg}") from error
    root = _mapping(payload, "envelope")
    return _parse_envelope(root)


def _load_spec(path: Path) -> _SpecInput:
    """Read, parse and validate one JSON specification."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise SpecParseError(f"cannot read {path}: {error}") from error
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise SpecParseError(
            f"{path}: invalid JSON at line {error.lineno} column {error.colno}: "
            f"{error.msg}"
        ) from error
    root = _mapping(payload, "spec")
    _reject_unknown(root, _SPEC_KEYS, "spec")
    _require(root, ["field_size_um", "survey", "focused", "stage", "targets"], "spec")
    field_size = _pair(root["field_size_um"], "field_size_um")
    survey = _parse_survey(root["survey"])
    focused = _parse_focused(root["focused"])
    stage = _parse_stage(root["stage"])
    targets, targets_canonical = _parse_targets(root["targets"], field_size)
    origin = (
        _pair(root["origin_xy_um"], "origin_xy_um")
        if "origin_xy_um" in root
        else (0.0, 0.0)
    )
    review = _number(root["review_s"], "review_s") if "review_s" in root else 0.0
    bleach_time_constant = (
        _number(root["bleach_time_constant_s"], "bleach_time_constant_s")
        if "bleach_time_constant_s" in root
        else None
    )
    if bleach_time_constant is not None and bleach_time_constant <= 0:
        raise SpecValidationError("bleach_time_constant_s must be positive")
    envelope = _parse_envelope(root["envelope"]) if "envelope" in root else None
    limits = _parse_limits(root["limits"]) if "limits" in root else None
    try:
        spec = SearchImagingSpec(
            field_size_um=field_size,
            survey=survey,
            focused=focused,
            stage=stage,
            targets=targets,
            origin_xy_um=origin,
            review_s=review,
            bleach_time_constant_s=bleach_time_constant,
        )
    except (TypeError, ValueError) as error:
        raise SpecValidationError(f"spec: {error}") from error
    return _SpecInput(
        spec=spec,
        envelope=envelope,
        limits=limits,
        canonical=_canonical(spec, envelope, limits, targets_canonical),
    )


def _simulate(item: _SpecInput) -> SimulationReport:
    """Run the deterministic simulation for a parsed specification."""
    try:
        return simulate_search_imaging(
            item.spec, envelope=item.envelope, limits=item.limits
        )
    except (TypeError, ValueError) as error:
        raise SpecValidationError(f"spec: {error}") from error


def _axis_value(point: AcquisitionPoint, axis: AcquisitionAxis) -> float:
    """Return one operating-point coordinate for the requested axis."""
    if axis is AcquisitionAxis.VELOCITY:
        return point.velocity_um_s
    if axis is AcquisitionAxis.EXPOSURE:
        return point.exposure_s
    return point.width_um


def _validate_spec(item: _SpecInput) -> _ValidationOutcome:
    """Check geometry, frame counts, envelope classification and limits."""
    spec = item.spec
    report = _simulate(item)
    tiles = spec.survey_tiles()
    rows = max(tile.row for tile in tiles) + 1
    columns = max(tile.column for tile in tiles) + 1
    overscan = spec.overscan_um()
    issues: list[str] = []
    warnings: list[str] = []
    if len(tiles) > MAX_SURVEY_TILES:
        issues.append(
            f"survey grid has {len(tiles)} tiles, above the {MAX_SURVEY_TILES} position limit"
        )
    focused_frames = spec.focused.frames * spec.focused.channels * spec.focused.z_planes
    if spec.survey.frames > MAX_FRAMES_PER_POSITION:
        issues.append(
            "survey frames per position "
            f"{spec.survey.frames} exceed {MAX_FRAMES_PER_POSITION}"
        )
    if focused_frames > MAX_FRAMES_PER_POSITION:
        issues.append(
            "focused frames per target "
            f"{focused_frames} exceed {MAX_FRAMES_PER_POSITION}"
        )
    if not spec.targets:
        warnings.append("no targets declared; the focused round has no positions")
    if overscan[0] > 0 or overscan[1] > 0:
        warnings.append(
            "survey overscan past the field: "
            f"x {overscan[0]:.3f} um, y {overscan[1]:.3f} um"
        )
    envelope = item.envelope
    if envelope is not None:
        points = (
            ("survey configured", report.survey_acquisition),
            ("survey equivalent", report.survey_equivalent),
            ("focused", report.focused_point),
        )
        for label, point in points:
            between = [
                axis.value
                for axis in AcquisitionAxis
                if envelope.band(axis, _axis_value(point, axis)) is AxisBand.BETWEEN
            ]
            if between:
                warnings.append(
                    f"{label} operating point falls between envelope bands on "
                    + ", ".join(between)
                )
    if item.limits is not None:
        verdicts = (
            ("survey configured", report.survey_acquisition_verdict),
            ("focused", report.focused_verdict),
        )
        for label, verdict in verdicts:
            if verdict is not None and not verdict.feasible:
                reasons = ", ".join(reason.value for reason in verdict.reasons)
                issues.append(f"{label} operating point fails limits: {reasons}")
    return _ValidationOutcome(
        report=report,
        issues=tuple(issues),
        warnings=tuple(warnings),
        tile_rows=rows,
        tile_columns=columns,
        overscan_um=overscan,
    )


def _metrics(report: SimulationReport) -> dict[str, float | int]:
    """Return the reported metrics compared by ``diff``."""
    return {
        "tile_count": report.survey.positions,
        "target_count": report.targets,
        "survey_time_s": report.survey.total_s,
        "focused_time_s": report.focused.total_s,
        "total_time_s": report.total_s,
        "survey_throughput_um2_s": report.survey_throughput_um2_s,
        "focused_throughput_um2_s": report.focused_scan.throughput_um2_s,
        "mean_sub_scan_throughput_um2_s": report.averaged.mean_throughput_um2_s,
        "combined_throughput_um2_s": report.averaged.combined_throughput_um2_s,
        "survey_seconds_per_position": report.survey_scan.seconds_per_position,
        "focused_seconds_per_position": report.focused_scan.seconds_per_position,
        "end_to_end_throughput_um2_s": report.end_to_end_throughput_um2_s,
        "targets_per_hour": report.targets_per_hour,
    }


def _corners(report: SimulationReport) -> dict[str, str | None]:
    """Return the three operating corners as codes or ``None``."""
    return {
        "survey_configured": (
            None
            if report.survey_acquisition_corner is None
            else report.survey_acquisition_corner.value
        ),
        "survey_equivalent": (
            None
            if report.survey_equivalent_corner is None
            else report.survey_equivalent_corner.value
        ),
        "focused": (
            None if report.focused_corner is None else report.focused_corner.value
        ),
    }


def _scan_metrics_line(label: str, scan: ScanMetrics) -> str:
    """Render one sub-scan's area, time and average metrics."""
    return (
        f"  {label:<8} area {scan.area_um2 / 1_000_000.0:.3f} mm2"
        f"  {scan.positions} pos  {scan.duration_s:.3f} s"
        f"  {scan.throughput_um2_s / 1_000_000.0:.3f} mm2/s"
        f"  {scan.seconds_per_position:.3f} s/pos"
        f"  travel {scan.travel_fraction:.0%}  exposure {scan.exposure_fraction:.0%}"
    )


def _human_report(spec: SearchImagingSpec, report: SimulationReport) -> str:
    """Render the compact human simulation summary."""
    width, height = spec.field_size_um
    tiles = spec.survey_tiles()
    rows = max(tile.row for tile in tiles) + 1
    columns = max(tile.column for tile in tiles) + 1
    overscan = spec.overscan_um()
    lines = [
        (
            f"{'field':<16}{width:.1f} x {height:.1f} um = "
            f"{width * height / 1_000_000.0:.3f} mm2"
        ),
        f"{'targets':<16}{report.targets}",
        (
            f"{'survey':<16}{report.survey.positions} tiles "
            f"({rows} rows x {columns} columns), tile "
            f"{spec.survey.tile_size_um[0]:.1f} x "
            f"{spec.survey.tile_size_um[1]:.1f} um, overlap "
            f"{spec.survey.overlap:.3f}, overscan {overscan[0]:.3f} x "
            f"{overscan[1]:.3f} um"
        ),
        (
            f"{'focused ROI':<16}{spec.focused.roi_size_um[0]:.1f} x "
            f"{spec.focused.roi_size_um[1]:.1f} um"
        ),
        _time_line("survey time", report.survey.total_s, report.survey),
        f"{'review':<16}{report.review_s:.3f} s",
        _time_line("focused time", report.focused.total_s, report.focused),
        f"{'total':<16}{report.total_s:.3f} s",
        (
            f"{'throughput':<16}survey "
            f"{report.survey_throughput_um2_s:.1f} um2/s = "
            f"{report.survey_throughput_um2_s / 1_000_000.0:.3f} mm2/s; "
            f"end-to-end {report.end_to_end_throughput_um2_s:.1f} um2/s = "
            f"{report.end_to_end_throughput_um2_s / 1_000_000.0:.3f} mm2/s"
        ),
        f"{'targets/hour':<16}{report.targets_per_hour:.1f}",
        "photobleaching",
        (
            f"  preview {report.photobleaching.preview_exposure_s * 1000.0:.1f} ms"
            f" x {report.photobleaching.mean_cover_count:.2f} tiles; prior"
            f" {report.photobleaching.prior_exposure_s_min * 1000.0:.1f}-"
            f"{report.photobleaching.prior_exposure_s_max * 1000.0:.1f} ms"
            f" (mean {report.photobleaching.prior_exposure_s_mean * 1000.0:.1f})"
        ),
        (
            f"  cost {report.photobleaching.cost_min:.4f}-"
            f"{report.photobleaching.cost_max:.4f}"
            f" (mean {report.photobleaching.cost_mean:.4f}) relative to 1;"
            + (
                " retained"
                f" {report.photobleaching.retained_min:.4f}-"
                f"{report.photobleaching.retained_max:.4f}"
                f" (mean {report.photobleaching.retained_mean:.4f}) at"
                f" tau {report.photobleaching.bleach_time_constant_s:g} s"
                if report.photobleaching.retained_mean is not None
                else " retained not computed (declare bleach_time_constant_s)"
            )
        ),
        "sub-scan metrics",
        _scan_metrics_line("survey", report.survey_scan),
        _scan_metrics_line("focused", report.focused_scan),
        "averaged metrics",
        (
            f"  mean    throughput "
            f"{report.averaged.mean_throughput_um2_s / 1_000_000.0:.3f} mm2/s"
            f"  {report.averaged.mean_seconds_per_position:.3f} s/pos"
            f"  travel {report.averaged.mean_travel_fraction:.0%}"
            f"  exposure {report.averaged.mean_exposure_fraction:.0%}"
        ),
        (
            f"  combined acquired "
            f"{report.averaged.combined_acquired_area_um2 / 1_000_000.0:.3f} mm2"
            f" in {report.survey.total_s + report.focused.total_s:.3f} s"
            f" = {report.averaged.combined_throughput_um2_s / 1_000_000.0:.3f}"
            f" mm2/s; total with review {report.averaged.total_duration_s:.3f} s"
        ),
        "operating points",
    ]
    points = (
        (
            "survey configured",
            report.survey_acquisition,
            report.survey_acquisition_corner,
            report.survey_acquisition_verdict,
        ),
        (
            "survey equivalent",
            report.survey_equivalent,
            report.survey_equivalent_corner,
            None,
        ),
        (
            "focused",
            report.focused_point,
            report.focused_corner,
            report.focused_verdict,
        ),
    )
    for label, point, corner, verdict in points:
        corner_text = "<none>" if corner is None else corner.value
        if verdict is None:
            verdict_text = "no verdict"
        elif verdict.feasible:
            verdict_text = "feasible"
        else:
            verdict_text = "infeasible: " + ", ".join(
                reason.value for reason in verdict.reasons
            )
        lines.append(
            f"  {label:<19}v={point.velocity_um_s:.3f} um/s  "
            f"Te={point.exposure_s:.3f} s  W={point.width_um:.3f} um  "
            f"corner={corner_text}  {verdict_text}"
        )
    return "\n".join(lines)


def _time_line(label: str, total_s: float, round_report: RoundReport) -> str:
    """Render one round's total and component breakdown."""
    return (
        f"{label:<16}{total_s:.3f} s (travel {round_report.travel_s:.3f}, "
        f"settle {round_report.settle_s:.3f}, "
        f"exposure {round_report.exposure_s:.3f}, "
        f"readout {round_report.readout_s:.3f}, "
        f"overhead {round_report.overhead_s:.3f})"
    )


def _human_validation(outcome: _ValidationOutcome, spec: SearchImagingSpec) -> str:
    """Render the compact human validation summary."""
    tiles = spec.survey_tiles()
    lines = ["invalid" if outcome.issues else "ok"]
    for issue in outcome.issues:
        lines.append(f"  issue: {issue}")
    for warning in outcome.warnings:
        lines.append(f"  warning: {warning}")
    lines.append(
        f"  tiles {len(tiles)} ({outcome.tile_rows} rows x "
        f"{outcome.tile_columns} columns)"
    )
    lines.append(f"  targets {len(spec.targets)}")
    lines.append(
        f"  overscan {outcome.overscan_um[0]:.3f} x {outcome.overscan_um[1]:.3f} um"
    )
    return "\n".join(lines)


def _changes(left: Any, right: Any, prefix: str = "") -> list[tuple[str, Any, Any]]:
    """Return differing leaves between two canonical specification values."""
    changes: list[tuple[str, Any, Any]] = []
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            left_value = left.get(key, _ABSENT)
            right_value = right.get(key, _ABSENT)
            if left_value is _ABSENT or right_value is _ABSENT:
                changes.append((path, left_value, right_value))
            else:
                changes.extend(_changes(left_value, right_value, path))
        return changes
    if left != right:
        changes.append((prefix, left, right))
    return changes


def _format_value(value: Any) -> str:
    """Render one diff value compactly, including absent keys."""
    if value is _ABSENT:
        return "<absent>"
    return json.dumps(value, sort_keys=True)


def _format_metric(value: float, decimals: int) -> str:
    """Render a metric with its declared precision."""
    if decimals == 0:
        return f"{value:.0f}"
    return f"{value:.{decimals}f}"


def _format_delta(value: float, decimals: int) -> str:
    """Render a signed metric delta with its declared precision."""
    if decimals == 0:
        return f"{value:+.0f}"
    return f"{value:+.{decimals}f}"


def _human_diff(
    changes: Sequence[tuple[str, Any, Any]],
    left_metrics: Mapping[str, float | int],
    right_metrics: Mapping[str, float | int],
    left_corners: Mapping[str, str | None],
    right_corners: Mapping[str, str | None],
) -> str:
    """Render the compact human specification diff."""
    lines = ["changed fields"]
    if changes:
        for path, left_value, right_value in changes:
            lines.append(
                f"  {path}: {_format_value(left_value)} -> {_format_value(right_value)}"
            )
    else:
        lines.append("  none")
    lines.append("metrics")
    for key, label, decimals in _METRIC_KEYS:
        left_value = left_metrics[key]
        right_value = right_metrics[key]
        lines.append(
            f"  {label}: {_format_metric(left_value, decimals)} -> "
            f"{_format_metric(right_value, decimals)} "
            f"({_format_delta(right_value - left_value, decimals)})"
        )
    lines.append("corners")
    for key, label in _CORNER_KEYS:
        left_corner = left_corners[key]
        right_corner = right_corners[key]
        lines.append(
            f"  {label}: "
            f"{'<none>' if left_corner is None else left_corner} -> "
            f"{'<none>' if right_corner is None else right_corner}"
        )
    return "\n".join(lines)


def _diff_payload(
    changes: Sequence[tuple[str, Any, Any]],
    left_metrics: Mapping[str, float | int],
    right_metrics: Mapping[str, float | int],
    left_corners: Mapping[str, str | None],
    right_corners: Mapping[str, str | None],
) -> dict[str, Any]:
    """Return the machine-readable specification diff."""
    return {
        "ok": True,
        "equal": not changes,
        "changed_fields": [
            {
                "path": path,
                "a": None if left_value is _ABSENT else left_value,
                "b": None if right_value is _ABSENT else right_value,
            }
            for path, left_value, right_value in changes
        ],
        "metrics": {
            key: {
                "a": left_metrics[key],
                "b": right_metrics[key],
                "delta": right_metrics[key] - left_metrics[key],
            }
            for key, _label, _decimals in _METRIC_KEYS
        },
        "corners": {
            key: {
                "a": left_corners[key],
                "b": right_corners[key],
                "changed": left_corners[key] != right_corners[key],
            }
            for key, _label in _CORNER_KEYS
        },
    }


def _print_json(payload: Any) -> None:
    """Print a JSON payload without non-finite values."""
    print(json.dumps(payload, indent=2, allow_nan=False))


def _write_output(path: str, payload: Any) -> None:
    """Write a JSON report to ``path``."""
    try:
        Path(path).write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise SpecValidationError(f"cannot write {path}: {error}") from error


def _cmd_simulate(args: argparse.Namespace) -> int:
    """Run ``planner simulate``."""
    item = _load_spec(Path(args.spec))
    report = _simulate(item)
    payload = report.to_dict()
    if args.output:
        _write_output(args.output, payload)
    if args.html:
        write_simulation_html(args.html, report, spec=item.spec, envelope=item.envelope)
    if args.json:
        _print_json(payload)
    else:
        print(_human_report(item.spec, report))
    return _EXIT_OK


def _cmd_validate(args: argparse.Namespace) -> int:
    """Run ``planner validate``."""
    try:
        item = _load_spec(Path(args.spec))
    except SpecValidationError as error:
        issues = [str(error)]
        if args.json:
            _print_json({"ok": False, "issues": issues, "warnings": []})
        else:
            print("invalid")
            print(f"  issue: {error}")
        return _EXIT_INVALID
    outcome = _validate_spec(item)
    if args.json:
        _print_json(
            {
                "ok": not outcome.issues,
                "issues": list(outcome.issues),
                "warnings": list(outcome.warnings),
                "tiles": len(item.spec.survey_tiles()),
                "tile_rows": outcome.tile_rows,
                "tile_columns": outcome.tile_columns,
                "targets": len(item.spec.targets),
                "overscan_um": list(outcome.overscan_um),
                "metrics": _metrics(outcome.report),
                "corners": _corners(outcome.report),
            }
        )
    else:
        print(_human_validation(outcome, item.spec))
    return _EXIT_OK if not outcome.issues else _EXIT_INVALID


def _cmd_diff(args: argparse.Namespace) -> int:
    """Run ``planner diff``."""
    issues: dict[str, list[str]] = {"a": [], "b": []}
    items: dict[str, _SpecInput] = {}
    outcomes: dict[str, _ValidationOutcome] = {}
    for label, path in (("a", args.a), ("b", args.b)):
        try:
            item = _load_spec(Path(path))
            outcome = _validate_spec(item)
        except SpecValidationError as error:
            issues[label].append(str(error))
            continue
        items[label] = item
        outcomes[label] = outcome
        issues[label].extend(outcome.issues)
    if issues["a"] or issues["b"]:
        payload = {"ok": False, "issues": issues}
        if args.json:
            _print_json(payload)
        else:
            print("invalid")
            for label in ("a", "b"):
                for issue in issues[label]:
                    print(f"  {label}: {issue}")
        return _EXIT_INVALID
    left = items["a"]
    right = items["b"]
    left_outcome = outcomes["a"]
    right_outcome = outcomes["b"]
    changes = _changes(left.canonical, right.canonical)
    left_metrics = _metrics(left_outcome.report)
    right_metrics = _metrics(right_outcome.report)
    left_corners = _corners(left_outcome.report)
    right_corners = _corners(right_outcome.report)
    if args.json:
        _print_json(
            _diff_payload(
                changes,
                left_metrics,
                right_metrics,
                left_corners,
                right_corners,
            )
        )
    else:
        print(
            _human_diff(
                changes,
                left_metrics,
                right_metrics,
                left_corners,
                right_corners,
            )
        )
    return _EXIT_OK


_COMPARE_FIELD_UM = 2000.0
_COMPARE_FOV_UM = 133.12
_COMPARE_DWELL_S = 2.0
_COMPARE_OVERVIEW_OVERLAP = 0.1
_COMPARE_SEED = 7
_COMPARE_SCENARIOS: dict[str, tuple[float, float, float, str]] = {
    "measured": (103.7, 92.7, 0.0, "measured readback, no command delay"),
    "blend-2500": (200.0, 2500.0, 0.02, "speed-blend replay, 20 ms delay, 2500 um/s2"),
    "blend-5000": (200.0, 5000.0, 0.02, "speed-blend replay, 20 ms delay, 5000 um/s2"),
}


def _compare_snap_spec(
    targets: tuple[MockTarget, ...],
    v_max: float,
    accel: float,
    *,
    bleach_time_constant_s: float | None = None,
) -> SearchImagingSpec:
    """Build the dual-scan specification used beside the broom in ``compare``."""
    return SearchImagingSpec(
        field_size_um=(_COMPARE_FIELD_UM, _COMPARE_FIELD_UM),
        survey=SurveySpec(
            (_COMPARE_FOV_UM, _COMPARE_FOV_UM),
            _COMPARE_OVERVIEW_OVERLAP,
            0.020,
            1,
            0.060,
            0.050,
            0.13,
            1.0,
        ),
        focused=FocusedSpec(
            2.0,
            1,
            1,
            1,
            0.060,
            0.100,
            0.13,
            1.0,
            (_COMPARE_FOV_UM, _COMPARE_FOV_UM),
        ),
        stage=StageProfile(v_max, accel, 0.010),
        targets=targets,
        review_s=0.0,
        bleach_time_constant_s=bleach_time_constant_s,
    )


def _cmd_compare(args: argparse.Namespace) -> int:
    """Compare an adaptive broom with the dual-scan routine over a 2x2 mm field."""
    selected = (
        list(_COMPARE_SCENARIOS) if args.scenario == "all" else [str(args.scenario)]
    )
    rows_count = ceil(_COMPARE_FIELD_UM / _COMPARE_FOV_UM)
    row_pitch = _COMPARE_FIELD_UM / rows_count
    field_area = (_COMPARE_FIELD_UM / 1000.0) ** 2
    lines = [
        (
            f"dual snap scan vs adaptive broom: {_COMPARE_FIELD_UM:.0f} x "
            f"{_COMPARE_FIELD_UM:.0f} um, FOV {_COMPARE_FOV_UM:.2f} um, "
            f"dwell {_COMPARE_DWELL_S:.1f} s, {rows_count} rows at "
            f"{row_pitch:.2f} um"
        )
    ]
    payload_rows: list[dict[str, Any]] = []
    for scenario in selected:
        v_max, accel, delay, note = _COMPARE_SCENARIOS[scenario]
        lines.append(
            f"\n== {scenario}: v_max {v_max:g} um/s, a {accel:g} um/s2, "
            f"delay {delay:g} s ({note})"
        )
        lines.append(
            "     n  broom-fast  broom-2s  adaptive   added   ramp  delay"
            "    dual-snap  dual-minus-adaptive  vs-broom  speedup  dwell-min  bleach-cost  prior-ms  retained  broom-mm2/min  dual-mm2/min"
        )
        lines.append(
            "broom-fast has no two-second dwell guarantee; broom-2s is the"
            " equivalent regular broom at the dwell speed."
        )
        lines.append(
            "times in s; dual-minus-adaptive = dual-snap - adaptive;"
            " vs-broom = broom-2s / adaptive; speedup = dual-snap / adaptive."
        )
        lines.append(
            "bleach-cost = prior overview exposure / focused exposure;"
            f" retained = exp(-prior / {args.bleach_tau:g} s)."
        )
        for count in args.samples:
            targets = mock_targets(
                (_COMPARE_FIELD_UM, _COMPARE_FIELD_UM),
                count,
                seed=_COMPARE_SEED,
            )
            broom = simulate_broom_scan(
                BroomSpec(
                    field_size_um=(_COMPARE_FIELD_UM, _COMPARE_FIELD_UM),
                    fov_um=_COMPARE_FOV_UM,
                    row_pitch_um=row_pitch,
                    dwell_s=_COMPARE_DWELL_S,
                    stage=StageProfile(v_max, accel),
                    exposure_s=0.020,
                    readout_s=0.060,
                    samples=targets,
                    command_delay_s=delay,
                )
            )
            dual = simulate_search_imaging(
                _compare_snap_spec(
                    targets,
                    v_max,
                    accel,
                    bleach_time_constant_s=args.bleach_tau,
                )
            )
            row = {
                "scenario": scenario,
                "samples": count,
                "broom_cruise_s": broom.regular_cruise_s,
                "broom_dwell_s": broom.regular_dwell_s,
                "adaptive_broom_s": broom.adaptive_s,
                "added_vs_cruise_s": broom.added_vs_cruise_s,
                "saved_vs_dwell_s": broom.saved_vs_dwell_s,
                "transition_s": broom.transition_s,
                "delay_s": broom.delay_s,
                "slow_s": broom.slow_s,
                "min_dwell_s": broom.min_dwell_s,
                "dual_snap_s": dual.total_s,
                "dual_minus_adaptive_s": dual.total_s - broom.adaptive_s,
                "speedup": dual.total_s / broom.adaptive_s,
                "speedup_vs_broom": broom.regular_dwell_s / broom.adaptive_s,
                "broom_mm2_per_min": field_area / broom.adaptive_s * 60.0,
                "dual_mm2_per_min": dual.end_to_end_throughput_um2_s * 60.0 / 1e6,
                "broom_samples_per_hour": broom.samples_per_hour,
                "dual_targets_per_hour": dual.targets_per_hour,
                "bleach_cost_mean": dual.photobleaching.cost_mean,
                "bleach_prior_ms_mean": dual.photobleaching.prior_exposure_s_mean
                * 1000.0,
                "bleach_retained_mean": dual.photobleaching.retained_mean,
            }
            payload_rows.append(row)
            lines.append(
                f"  {count:4d}  {broom.regular_cruise_s:6.1f}  "
                f"{broom.regular_dwell_s:6.1f}  {broom.adaptive_s:7.1f}  "
                f"{broom.added_vs_cruise_s:6.1f}  {broom.transition_s:5.1f}  "
                f"{broom.delay_s:5.1f}  {dual.total_s:7.1f}  "
                f"{dual.total_s - broom.adaptive_s:8.1f}  "
                f"{broom.regular_dwell_s / broom.adaptive_s:5.2f}x  "
                f"{dual.total_s / broom.adaptive_s:5.2f}x  "
                f"{broom.min_dwell_s:8.2f}  "
                f"{dual.photobleaching.cost_mean:10.4f}  "
                f"{dual.photobleaching.prior_exposure_s_mean * 1000.0:8.2f}  "
                f"{(dual.photobleaching.retained_mean or 1.0):8.4f}  "
                f"{field_area / broom.adaptive_s * 60.0:12.2f}  "
                f"{dual.end_to_end_throughput_um2_s * 60.0 / 1e6:12.2f}"
            )
    payload = {
        "field_um": [_COMPARE_FIELD_UM, _COMPARE_FIELD_UM],
        "fov_um": _COMPARE_FOV_UM,
        "dwell_s": _COMPARE_DWELL_S,
        "overview_overlap": _COMPARE_OVERVIEW_OVERLAP,
        "rows": rows_count,
        "row_pitch_um": row_pitch,
        "seed": _COMPARE_SEED,
        "rows_data": payload_rows,
    }
    geometry_count = 50 if 50 in args.samples else min(args.samples)
    geometry_targets = mock_targets(
        (_COMPARE_FIELD_UM, _COMPARE_FIELD_UM),
        geometry_count,
        seed=_COMPARE_SEED,
    )
    geometry_spec = _compare_snap_spec(geometry_targets, 103.7, 92.7)
    payload["geometry"] = {
        "sample_count": geometry_count,
        "field_um": [_COMPARE_FIELD_UM, _COMPARE_FIELD_UM],
        "survey_tile_size_um": list(geometry_spec.survey.tile_size_um),
        "survey_tiles": [
            list(tile.center_xy_um) for tile in geometry_spec.survey_tiles()
        ],
        "samples": [list(target.center_xy_um) for target in geometry_targets],
        "focused_roi_um": list(geometry_spec.focused.roi_size_um),
        "broom_rows_um": [row_pitch * (index + 0.5) for index in range(rows_count)],
        "broom_fov_um": _COMPARE_FOV_UM,
        "slow_zones_um": [
            [
                max(0.0, target.center_xy_um[0] - _COMPARE_FOV_UM / 2.0),
                min(
                    _COMPARE_FIELD_UM,
                    target.center_xy_um[0] + _COMPARE_FOV_UM / 2.0,
                ),
                row_pitch
                * (
                    min(
                        int(target.center_xy_um[1] / row_pitch),
                        rows_count - 1,
                    )
                    + 0.5
                ),
            ]
            for target in geometry_targets
        ],
    }
    geometry_report = simulate_search_imaging(geometry_spec)
    envelope = (
        _load_envelope(Path(args.envelope)) if args.envelope is not None else None
    )
    operating_broom = broom_trace(
        StageProfile(103.7, 92.7),
        _COMPARE_FOV_UM,
        _COMPARE_DWELL_S,
        0.020,
        envelope,
    )
    payload["operating"] = build_operating_payload(
        snap_traces(geometry_report, envelope) + (operating_broom,), envelope
    )
    if args.json:
        _print_json(payload)
    else:
        print("\n".join(lines))
    if args.output:
        _write_output(args.output, payload)
    if args.html:
        write_comparison_html(args.html, payload)
    return _EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    """Return the ``planner`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="planner",
        description=(
            "Hardware-free search-and-image specification tools: simulate, "
            "validate and compare JSON specs."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    simulate = commands.add_parser(
        "simulate", help="run the deterministic search-and-image simulation"
    )
    simulate.add_argument("spec", help="JSON specification path")
    simulate.add_argument(
        "--json", action="store_true", help="print the report JSON instead"
    )
    simulate.add_argument(
        "--output", default=None, help="write the report JSON to this path"
    )
    simulate.add_argument(
        "--html", default=None, help="write a self-contained HTML throughput page"
    )
    simulate.set_defaults(handler=_cmd_simulate)
    validate = commands.add_parser(
        "validate", help="validate a specification and report issues"
    )
    validate.add_argument("spec", help="JSON specification path")
    validate.add_argument(
        "--json", action="store_true", help="print machine-readable findings"
    )
    validate.set_defaults(handler=_cmd_validate)
    diff = commands.add_parser(
        "diff", help="compare two specifications and their metrics"
    )
    diff.add_argument("a", help="first JSON specification path")
    diff.add_argument("b", help="second JSON specification path")
    diff.add_argument(
        "--json", action="store_true", help="print machine-readable differences"
    )
    diff.set_defaults(handler=_cmd_diff)
    compare = commands.add_parser(
        "compare",
        help="compare an adaptive broom with the dual-scan routine",
    )
    compare.add_argument(
        "--samples",
        type=int,
        nargs="+",
        default=[10, 25, 50, 100],
        help="mock sample counts to compare",
    )
    compare.add_argument(
        "--scenario",
        choices=("all", "measured", "blend-2500", "blend-5000"),
        default="all",
        help="stage scenario from the measured readback or the speed-blend replay",
    )
    compare.add_argument(
        "--bleach-tau",
        type=float,
        default=10.0,
        help="declared log-falloff time constant for retained brightness, in s",
    )
    compare.add_argument(
        "--envelope",
        default=None,
        help="JSON envelope with velocity_um_s, exposure_s and width_um pairs",
    )
    compare.add_argument(
        "--json", action="store_true", help="print the comparison payload"
    )
    compare.add_argument(
        "--output", default=None, help="write the comparison JSON to this path"
    )
    compare.add_argument(
        "--html", default=None, help="write a self-contained HTML comparison page"
    )
    compare.set_defaults(handler=_cmd_compare)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the planner command line and return its exit code."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    json_mode = "--json" in arguments
    parser = build_parser()
    args = parser.parse_args(arguments)
    handler: Callable[[argparse.Namespace], int] = args.handler
    try:
        return handler(args)
    except SpecParseError as error:
        if json_mode:
            _print_json({"ok": False, "error": str(error)})
        else:
            print(f"error: {error}", file=sys.stderr)
        return _EXIT_PARSE
    except SpecValidationError as error:
        if json_mode:
            _print_json({"ok": False, "error": str(error)})
        else:
            print(f"error: {error}", file=sys.stderr)
        return _EXIT_INVALID


__all__ = ["build_parser", "main"]
