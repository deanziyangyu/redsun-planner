"""Pure scan recipe geometry; importing it opens no devices or execution host."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

from .contracts import AcquisitionSettings, Phase, ResourceRequirement


@dataclass(frozen=True)
class ScanPosition:
    """One named stage position in a scan recipe."""

    position_id: str
    x_um: float
    y_um: float
    row: int | None = None
    column: int | None = None


def grid_positions(
    origin_xy_um: tuple[float, float],
    *,
    rows: int,
    columns: int,
    step_xy_um: tuple[float, float],
    x_sign: int = 1,
    y_sign: int = 1,
    swap_xy: bool = False,
    serpentine: bool = False,
) -> tuple[ScanPosition, ...]:
    """Column-major grid like the donor; optional alternating column traversal."""
    if any(
        isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in (rows, columns)
    ):
        raise ValueError("rows and columns must be positive integers")
    if rows * columns > 10000:
        raise ValueError("scan exceeds 10000 positions")
    if x_sign not in (-1, 1) or y_sign not in (-1, 1):
        raise ValueError("axis signs must be -1 or 1")
    if (
        not all(isfinite(v) for v in (*origin_xy_um, *step_xy_um))
        or min(step_xy_um) <= 0
    ):
        raise ValueError("finite origin and positive micrometre steps required")
    positions = []
    for column in range(columns):
        indices = range(rows - 1, -1, -1) if serpentine and column % 2 else range(rows)
        for row in indices:
            dx, dy = column * step_xy_um[0] * x_sign, row * step_xy_um[1] * y_sign
            if swap_xy:
                dx, dy = dy, dx
            positions.append(
                ScanPosition(
                    f"tile-c{column:03d}-r{row:03d}",
                    origin_xy_um[0] + dx,
                    origin_xy_um[1] + dy,
                    row,
                    column,
                )
            )
    return tuple(positions)


@dataclass(frozen=True)
class StageScanRecipe:
    """Declarative stage-grid/target acquisition, interpreted by an acquisition host.

    Positions are absolute stage micrometres, not image-pixel detections.
    Hardware limits, deadlines, cleanup and transitions are host policy.
    """

    recipe_id: str
    specimen_id: str
    positions: tuple[ScanPosition, ...]
    settings: AcquisitionSettings = field(default_factory=AcquisitionSettings)
    version: str = "0.1-local"
    phase: Phase = Phase.SURVEY
    coordinate_frame: str = "stage-xy"
    calibration_id: str = "uncalibrated-stage-scan"
    resources: tuple[ResourceRequirement, ...] = (
        ResourceRequirement("stage"),
        ResourceRequirement("camera"),
        ResourceRequirement("storage"),
    )

    def __post_init__(self) -> None:
        """Check the declared fields for consistency."""
        if not all(
            isinstance(v, str) and v
            for v in (
                self.recipe_id,
                self.version,
                self.specimen_id,
                self.coordinate_frame,
                self.calibration_id,
            )
        ):
            raise ValueError("recipe identity and coordinate context must be nonempty")
        if self.phase not in (Phase.SURVEY, Phase.REVISIT):
            raise ValueError("stage scan recipe must describe survey or revisit")
        if not isinstance(self.settings, AcquisitionSettings):
            raise TypeError("typed acquisition settings required")
        if (
            not self.resources
            or any(not isinstance(r, ResourceRequirement) for r in self.resources)
            or len({r.name for r in self.resources}) != len(self.resources)
        ):
            raise ValueError("scan resources must be unique typed requirements")
        if (
            not isinstance(self.positions, tuple)
            or not 1 <= len(self.positions) <= 10000
        ):
            raise ValueError("immutable tuple of 1–10000 positions required")
        if any(
            not isinstance(p, ScanPosition)
            or not p.position_id
            or not all(isfinite(v) for v in (p.x_um, p.y_um))
            for p in self.positions
        ):
            raise ValueError("positions must be identified and finite")
        if len({p.position_id for p in self.positions}) != len(self.positions):
            raise ValueError("position IDs must be unique")

    @property
    def required_capabilities(self) -> frozenset[str]:
        """Return the device capabilities a host must provide for this recipe."""
        capabilities = {"stage.xy", "camera.snap", "storage.raw"}
        if self.settings.frames > 1:
            capabilities.add("camera.sequence")
        return frozenset(capabilities)

    @property
    def required_settings(self) -> frozenset[str]:
        """Return the acquisition settings a host must support for this recipe."""
        settings = {"exposure_ms"}
        if self.settings.channel_id is not None:
            settings.add("channel")
        if self.settings.roi_origin is not None:
            settings.add("roi")
        if self.settings.binning is not None:
            settings.add("binning")
        return frozenset(settings)


def build_grid_recipe(
    origin_xy_um: tuple[float, float],
    *,
    specimen_id: str,
    rows: int,
    columns: int,
    step_xy_um: tuple[float, float],
    settings: AcquisitionSettings | None = None,
    x_sign: int = 1,
    y_sign: int = 1,
    swap_xy: bool = False,
    serpentine: bool = False,
) -> StageScanRecipe:
    """Publish a fully resolved grid recipe without connecting or running it."""
    return StageScanRecipe(
        "stage-grid",
        specimen_id,
        grid_positions(
            origin_xy_um,
            rows=rows,
            columns=columns,
            step_xy_um=step_xy_um,
            x_sign=x_sign,
            y_sign=y_sign,
            swap_xy=swap_xy,
            serpentine=serpentine,
        ),
        settings=settings or AcquisitionSettings(),
    )
