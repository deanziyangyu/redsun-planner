"""Operating-space traces for the HTML reports.

The traces are display data over declared points and simple speed ramps. They
show where each plan sits relative to the caller-declared low/high envelope and
how a broom moves between its dwell and travel regimes; they do not assert that
the stage can execute the ramp.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .acquisition_space import AcquisitionAxis, AcquisitionEnvelope, AxisBand

if TYPE_CHECKING:
    from .simulation import SimulationReport, StageProfile


@dataclass(frozen=True)
class OperatingPoint:
    """One labelled point in ``(velocity, exposure, width)`` space."""

    label: str
    velocity_um_s: float
    exposure_s: float
    width_um: float
    corner: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary."""
        return {
            "label": self.label,
            "v": self.velocity_um_s,
            "te": self.exposure_s,
            "w": self.width_um,
            "corner": self.corner,
        }


@dataclass(frozen=True)
class OperatingTrace:
    """One ordered path through operating space."""

    name: str
    color: str
    points: tuple[OperatingPoint, ...]
    dashed: bool = False
    markers: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary."""
        return {
            "name": self.name,
            "color": self.color,
            "dashed": self.dashed,
            "markers": self.markers,
            "points": [point.to_dict() for point in self.points],
        }


def _corner(
    envelope: AcquisitionEnvelope | None,
    velocity_um_s: float,
    exposure_s: float,
    width_um: float,
) -> str | None:
    """Return the low/high corner label, or ``None`` without a full corner."""
    if envelope is None:
        return None
    bands = (
        envelope.band(AcquisitionAxis.VELOCITY, velocity_um_s),
        envelope.band(AcquisitionAxis.EXPOSURE, exposure_s),
        envelope.band(AcquisitionAxis.WIDTH, width_um),
    )
    if AxisBand.BETWEEN in bands:
        return None
    return "".join("1" if band is AxisBand.HIGH else "0" for band in bands)


def snap_traces(
    report: SimulationReport,
    envelope: AcquisitionEnvelope | None = None,
) -> tuple[OperatingTrace, ...]:
    """Return the survey and focused operating points for a dual snap scan."""
    survey = report.survey_acquisition
    equivalent = report.survey_equivalent
    focused = report.focused_point
    return (
        OperatingTrace(
            "dual snap survey",
            "#3572a5",
            (
                OperatingPoint(
                    "survey tile",
                    survey.velocity_um_s,
                    survey.exposure_s,
                    survey.width_um,
                    _corner(
                        envelope,
                        survey.velocity_um_s,
                        survey.exposure_s,
                        survey.width_um,
                    ),
                ),
                OperatingPoint(
                    "survey equivalent",
                    equivalent.velocity_um_s,
                    equivalent.exposure_s,
                    equivalent.width_um,
                    _corner(
                        envelope,
                        equivalent.velocity_um_s,
                        equivalent.exposure_s,
                        equivalent.width_um,
                    ),
                ),
            ),
            dashed=True,
        ),
        OperatingTrace(
            "dual snap focused",
            "#2f9e6f",
            (
                OperatingPoint(
                    "focused revisit",
                    focused.velocity_um_s,
                    focused.exposure_s,
                    focused.width_um,
                    _corner(
                        envelope,
                        focused.velocity_um_s,
                        focused.exposure_s,
                        focused.width_um,
                    ),
                ),
            ),
        ),
        OperatingTrace(
            "dual snap mode switch",
            "#94a3b8",
            (
                OperatingPoint(
                    "survey → focused",
                    equivalent.velocity_um_s,
                    equivalent.exposure_s,
                    equivalent.width_um,
                    None,
                ),
                OperatingPoint(
                    "focused",
                    focused.velocity_um_s,
                    focused.exposure_s,
                    focused.width_um,
                    None,
                ),
            ),
            dashed=True,
            markers=False,
        ),
    )


def broom_trace(
    stage: StageProfile,
    fov_um: float,
    dwell_s: float,
    exposure_s: float,
    envelope: AcquisitionEnvelope | None = None,
    *,
    label: str = "adaptive broom",
    color: str = "#d99a2b",
) -> OperatingTrace:
    """Return a dwell-ramp-cruise-ramp-dwell trace for one broom row."""
    dwell_velocity = fov_um / dwell_s if dwell_s > 0 else 0.0
    travel_velocity = stage.max_velocity_um_s
    steps = 9
    velocities = [dwell_velocity]
    for index in range(1, steps + 1):
        velocities.append(
            dwell_velocity + (travel_velocity - dwell_velocity) * index / steps
        )
    for index in range(steps - 1, -1, -1):
        velocities.append(
            dwell_velocity + (travel_velocity - dwell_velocity) * index / steps
        )
    points: list[OperatingPoint] = []
    for index, velocity in enumerate(velocities):
        if index == 0:
            point_label = "dwell at sample"
        elif index == steps:
            point_label = "cruise between samples"
        elif index == len(velocities) - 1:
            point_label = "dwell at next sample"
        elif index < steps:
            point_label = f"accelerate {index}/{steps}"
        else:
            point_label = f"decelerate {len(velocities) - 1 - index}/{steps}"
        points.append(
            OperatingPoint(
                point_label,
                velocity,
                exposure_s,
                fov_um,
                _corner(envelope, velocity, exposure_s, fov_um),
            )
        )
    return OperatingTrace(label, color, tuple(points))


def build_operating_payload(
    traces: tuple[OperatingTrace, ...],
    envelope: AcquisitionEnvelope | None = None,
) -> dict[str, Any]:
    """Build the JSON payload consumed by the HTML operating-space map."""
    points = [point for trace in traces for point in trace.points]
    if not points:
        raise ValueError("at least one operating point is required")
    velocity_max = max(point.velocity_um_s for point in points)
    exposure_max = max(point.exposure_s for point in points)
    width_max = max(point.width_um for point in points)
    thresholds: dict[str, list[float]] | None = None
    if envelope is not None:
        thresholds = {
            "velocity": [
                envelope.velocity_um_s.low_max,
                envelope.velocity_um_s.high_min,
            ],
            "exposure": [
                envelope.exposure_s.low_max,
                envelope.exposure_s.high_min,
            ],
            "width": [envelope.width_um.low_max, envelope.width_um.high_min],
        }
        velocity_max = max(velocity_max, envelope.velocity_um_s.high_min)
        exposure_max = max(exposure_max, envelope.exposure_s.high_min)
        width_max = max(width_max, envelope.width_um.high_min)
    return {
        "has_envelope": envelope is not None,
        "thresholds": thresholds,
        "ranges": {
            "velocity": [0.0, velocity_max * 1.30],
            "exposure": [0.0, exposure_max * 1.30],
            "width": [0.0, width_max * 1.30],
        },
        "traces": [trace.to_dict() for trace in traces],
    }


def simulation_operating_payload(
    report: SimulationReport,
    envelope: AcquisitionEnvelope | None = None,
) -> dict[str, Any]:
    """Return the dual-scan operating map for a simulated specification."""
    return build_operating_payload(snap_traces(report, envelope), envelope)


__all__ = [
    "OperatingPoint",
    "OperatingTrace",
    "broom_trace",
    "build_operating_payload",
    "simulation_operating_payload",
    "snap_traces",
]
