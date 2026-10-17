"""Coverage-aware snap-grid geometry for registered, image-derived FoV support.

This module plans in the registered support-mask pixel frame and maps tile
centres/footprints into a stage readback frame through an explicit affine. It
does not identify sample boundaries or authorize hardware motion.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import ceil, isfinite
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

MAX_OBSERVED_COVERAGE_POSITIONS = 10000
AffineXY = tuple[tuple[float, float, float], tuple[float, float, float]]
EnvelopeXY = tuple[float, float, float, float]  # xmin, xmax, ymin, ymax in µm.


def _finite_pair(value: tuple[float, float], name: str) -> tuple[float, float]:
    if len(value) != 2 or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v)
        for v in value
    ):
        raise ValueError(f"{name} must be a finite XY pair")
    return float(value[0]), float(value[1])


@dataclass(frozen=True)
class ObservedCoverage:
    """Registered support mask with its provenance and pixel-to-stage affine.

    ``support_mask`` is a 2-D nonzero-is-supported array in registered mosaic
    pixels. ``mask_reference`` identifies the durable mask artifact; the mask
    itself remains a runtime array and is not duplicated into the run document.
    """

    coverage_id: str
    source_run_id: str
    registration_id: str
    coordinate_frame: str
    stage_coordinate_frame: str
    calibration_id: str
    mask_reference: str
    support_mask_sha256: str
    support_mask: NDArray[Any]
    pixel_to_stage_affine_xy_um: AffineXY
    uncertainty_px: float
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        """Check the declared fields for consistency."""
        for name in (
            "coverage_id",
            "source_run_id",
            "registration_id",
            "coordinate_frame",
            "stage_coordinate_frame",
            "calibration_id",
            "mask_reference",
            "support_mask_sha256",
            "schema_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if (
            isinstance(self.uncertainty_px, bool)
            or not isinstance(self.uncertainty_px, (int, float))
            or not isfinite(self.uncertainty_px)
            or self.uncertainty_px < 0
        ):
            raise ValueError("uncertainty_px must be finite and nonnegative")
        mask = np.asarray(self.support_mask)
        if mask.ndim != 2 or mask.size == 0 or not np.any(mask):
            raise ValueError(
                "support_mask must be a nonempty 2-D mask with supported pixels"
            )
        frozen_mask = np.ascontiguousarray(mask != 0)
        digest = sha256(frozen_mask.tobytes(order="C")).hexdigest()
        if self.support_mask_sha256.lower() != digest:
            raise ValueError(
                "support_mask_sha256 does not match normalized support-mask pixels"
            )
        frozen_mask.setflags(write=False)
        object.__setattr__(self, "support_mask", frozen_mask)
        object.__setattr__(self, "support_mask_sha256", digest)
        matrix = np.asarray(self.pixel_to_stage_affine_xy_um, dtype=np.float64)
        if matrix.shape != (2, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("pixel_to_stage_affine_xy_um must be a finite 2x3 affine")
        if abs(float(np.linalg.det(matrix[:, :2]))) < 1e-12:
            raise ValueError("pixel_to_stage_affine_xy_um must be invertible")
        object.__setattr__(
            self,
            "pixel_to_stage_affine_xy_um",
            tuple(tuple(float(v) for v in row) for row in matrix),
        )
        object.__setattr__(self, "uncertainty_px", float(self.uncertainty_px))

    def to_record(self) -> dict[str, Any]:
        """Return provenance suitable for run metadata, excluding mask bytes."""
        return {
            "schema": "redsun-planner.observed-coverage",
            "schema_version": self.schema_version,
            "coverage_id": self.coverage_id,
            "source_run_id": self.source_run_id,
            "registration_id": self.registration_id,
            "coordinate_frame": self.coordinate_frame,
            "stage_coordinate_frame": self.stage_coordinate_frame,
            "calibration_id": self.calibration_id,
            "mask_reference": self.mask_reference,
            "support_mask_sha256": self.support_mask_sha256,
            "mask_shape_yx": list(self.support_mask.shape),
            "mask_encoding": "nonzero means registered support",
            "pixel_to_stage_affine_xy_um": [
                list(row) for row in self.pixel_to_stage_affine_xy_um
            ],
            "uncertainty_px": self.uncertainty_px,
        }


@dataclass(frozen=True)
class ObservedSnapTile:
    """One proposed snap center in mask pixels and stage readback micrometres."""

    position_id: str
    row: int
    column: int
    center_mosaic_xy_px: tuple[float, float]
    center_stage_xy_um: tuple[float, float]
    footprint_stage_corners_xy_um: tuple[tuple[float, float], ...]
    inside_envelope: bool


@dataclass(frozen=True)
class ObservedSnapGridFit:
    """A deterministic proposed tile grid and support/overscan accounting."""

    coverage: ObservedCoverage
    fov_calibration_id: str
    footprint_xy_px: tuple[int, int]
    overlap: float
    positions: tuple[ObservedSnapTile, ...]
    target_support_pixels: int
    covered_support_pixels: int
    missed_support_pixels: int
    overscan_pixels: int
    outside_envelope_positions: tuple[str, ...]
    within_envelope: bool

    @property
    def coverage_fraction(self) -> float:
        """Return the fraction of target support pixels this plan covers."""
        return self.covered_support_pixels / self.target_support_pixels

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-compatible record of this fit."""
        return {
            "schema": "redsun-planner.observed-snap-grid-fit",
            "schema_version": "1.0",
            "coverage": self.coverage.to_record(),
            "fov_calibration_id": self.fov_calibration_id,
            "footprint_xy_px": list(self.footprint_xy_px),
            "overlap_fraction": self.overlap,
            "position_count": len(self.positions),
            "target_support_pixels": self.target_support_pixels,
            "covered_support_pixels": self.covered_support_pixels,
            "missed_support_pixels": self.missed_support_pixels,
            "coverage_fraction": self.coverage_fraction,
            "overscan_pixels": self.overscan_pixels,
            "within_envelope": self.within_envelope,
            "outside_envelope_positions": list(self.outside_envelope_positions),
            "positions": [
                {
                    "position_id": p.position_id,
                    "row": p.row,
                    "column": p.column,
                    "center_mosaic_xy_px": list(p.center_mosaic_xy_px),
                    "center_stage_xy_um": list(p.center_stage_xy_um),
                    "footprint_stage_corners_xy_um": [
                        list(c) for c in p.footprint_stage_corners_xy_um
                    ],
                    "inside_envelope": p.inside_envelope,
                }
                for p in self.positions
            ],
            "physical_effect": "proposed stage positions; execution requires host preflight and operator policy",
        }


def _interval_centers(
    low: int, high: int, footprint: int, spacing: float
) -> list[float]:
    first, last = low + footprint / 2, high - footprint / 2
    if last <= first:
        return [(low + high) / 2]
    count = max(2, ceil((last - first) / spacing) + 1)
    if count > MAX_OBSERVED_COVERAGE_POSITIONS:
        raise ValueError("observed-coverage grid exceeds position limit")
    return [float(value) for value in np.linspace(first, last, count)]


def fit_snap_grid_to_coverage(
    coverage: ObservedCoverage,
    footprint_xy_px: tuple[int, int],
    overlap: float,
    *,
    fov_calibration_id: str,
    expected_source_run_id: str,
    expected_registration_id: str,
    expected_coordinate_frame: str,
    expected_stage_coordinate_frame: str,
    stage_envelope_xy_um: EnvelopeXY,
    max_positions: int = MAX_OBSERVED_COVERAGE_POSITIONS,
) -> ObservedSnapGridFit:
    """Fit a deterministic snake grid to registered support, preserving holes.

    Positions are planned in mask-pixel space so camera rotation is handled by
    the explicit affine when mapped to stage coordinates. All full tile corners
    are checked against the caller-supplied stage envelope. Out-of-envelope
    tiles are returned and flagged; device execution must reject such a fit.
    """
    if not isinstance(coverage, ObservedCoverage):
        raise TypeError("coverage must be an ObservedCoverage")
    for value, name in (
        (expected_source_run_id, "source run"),
        (expected_registration_id, "registration"),
        (expected_coordinate_frame, "coordinate frame"),
        (expected_stage_coordinate_frame, "stage coordinate frame"),
        (fov_calibration_id, "FoV calibration"),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"expected {name} identity must be nonempty")
    if coverage.source_run_id != expected_source_run_id:
        raise ValueError(
            "observed coverage source run does not match current plan context"
        )
    if coverage.registration_id != expected_registration_id:
        raise ValueError(
            "observed coverage registration does not match current plan context"
        )
    if coverage.coordinate_frame != expected_coordinate_frame:
        raise ValueError(
            "observed coverage pixel frame does not match current plan context"
        )
    if coverage.stage_coordinate_frame != expected_stage_coordinate_frame:
        raise ValueError(
            "observed coverage stage frame does not match current plan context"
        )
    if coverage.calibration_id != fov_calibration_id:
        raise ValueError(
            "observed coverage calibration does not match the requested FoV calibration"
        )
    if (
        not isinstance(footprint_xy_px, tuple)
        or len(footprint_xy_px) != 2
        or any(
            isinstance(v, bool) or not isinstance(v, int) or v <= 0
            for v in footprint_xy_px
        )
    ):
        raise ValueError("footprint_xy_px must be a pair of positive integers")
    if (
        isinstance(overlap, bool)
        or not isinstance(overlap, (int, float))
        or not isfinite(overlap)
        or not 0 <= overlap < 1
    ):
        raise ValueError("overlap must be finite and in [0, 1)")
    if (
        isinstance(max_positions, bool)
        or not isinstance(max_positions, int)
        or max_positions <= 0
    ):
        raise ValueError("max_positions must be a positive integer")
    if (
        len(stage_envelope_xy_um) != 4
        or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v)
            for v in stage_envelope_xy_um
        )
        or stage_envelope_xy_um[0] >= stage_envelope_xy_um[1]
        or stage_envelope_xy_um[2] >= stage_envelope_xy_um[3]
    ):
        raise ValueError(
            "stage_envelope_xy_um must be finite ordered (xmin,xmax,ymin,ymax)"
        )

    mask = coverage.support_mask
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    fov_x, fov_y = footprint_xy_px
    centers_x = _interval_centers(
        int(xs.min()), int(xs.max()) + 1, fov_x, fov_x * (1 - overlap)
    )
    centers_y = _interval_centers(
        int(ys.min()), int(ys.max()) + 1, fov_y, fov_y * (1 - overlap)
    )
    if len(centers_x) * len(centers_y) > min(
        max_positions, MAX_OBSERVED_COVERAGE_POSITIONS
    ):
        raise ValueError("observed-coverage grid exceeds position limit")

    pad_x, pad_y = fov_x, fov_y
    union = np.zeros((h + 2 * pad_y, w + 2 * pad_x), dtype=bool)
    affine = np.asarray(coverage.pixel_to_stage_affine_xy_um, dtype=np.float64)
    envelope = tuple(float(v) for v in stage_envelope_xy_um)
    positions: list[ObservedSnapTile] = []
    outside: list[str] = []
    for row, cy in enumerate(centers_y):
        row_centers = centers_x if row % 2 == 0 else list(reversed(centers_x))
        for column, cx in enumerate(row_centers):
            px, py = round(cx), round(cy)
            x0, x1 = px - fov_x // 2 + pad_x, px + (fov_x - fov_x // 2) + pad_x
            y0, y1 = py - fov_y // 2 + pad_y, py + (fov_y - fov_y // 2) + pad_y
            union[y0:y1, x0:x1] = True
            center_stage = affine @ np.asarray([cx, cy, 1.0])
            pixel_corners = np.asarray(
                [
                    [cx - fov_x / 2, cy - fov_y / 2, 1],
                    [cx + fov_x / 2, cy - fov_y / 2, 1],
                    [cx + fov_x / 2, cy + fov_y / 2, 1],
                    [cx - fov_x / 2, cy + fov_y / 2, 1],
                ]
            )
            stage_corners = pixel_corners @ affine.T
            within = bool(
                np.all(stage_corners[:, 0] >= envelope[0])
                and np.all(stage_corners[:, 0] <= envelope[1])
                and np.all(stage_corners[:, 1] >= envelope[2])
                and np.all(stage_corners[:, 1] <= envelope[3])
            )
            position_id = f"observed-snap-r{row:03d}-c{column:03d}"
            if not within:
                outside.append(position_id)
            positions.append(
                ObservedSnapTile(
                    position_id,
                    row,
                    column,
                    (float(cx), float(cy)),
                    (float(center_stage[0]), float(center_stage[1])),
                    tuple((float(p[0]), float(p[1])) for p in stage_corners),
                    within,
                )
            )

    union_canvas = union[pad_y : pad_y + h, pad_x : pad_x + w]
    total = int(mask.sum())
    covered_pixels = int(np.count_nonzero(mask & union_canvas))
    return ObservedSnapGridFit(
        coverage=coverage,
        fov_calibration_id=fov_calibration_id,
        footprint_xy_px=footprint_xy_px,
        overlap=float(overlap),
        positions=tuple(positions),
        target_support_pixels=total,
        covered_support_pixels=covered_pixels,
        missed_support_pixels=total - covered_pixels,
        overscan_pixels=int(union.sum() - covered_pixels),
        outside_envelope_positions=tuple(outside),
        within_envelope=not outside,
    )
