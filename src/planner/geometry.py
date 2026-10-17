"""Calibrated image-to-stage geometry and deterministic survey coverage."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, hypot, isfinite
from typing import TYPE_CHECKING
from uuid import NAMESPACE_URL, uuid5

from .contracts import SearchObservation, Target

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True)
class Calibration:
    """Affine calibration from native sensor pixels to stage micrometres.

    ``pixel_to_stage`` acts on native (unbinned, full-sensor) pixel offsets.
    Image pixel coordinates denote pixel centers. ``reference_pixel`` is in
    full-sensor coordinates, and ``camera_stage_xy_um`` is the stage position
    at that reference pixel. ROI and binning describe each delivered image.
    """

    calibration_id: str
    coordinate_frame: str
    pixel_to_stage: tuple[tuple[float, float], tuple[float, float]]
    reference_pixel: tuple[float, float]
    camera_stage_xy_um: tuple[float, float]
    roi_origin: tuple[float, float] = (0.0, 0.0)
    binning: tuple[float, float] = (1.0, 1.0)

    def __post_init__(self) -> None:
        """Check the declared fields for consistency."""
        a = self.pixel_to_stage
        if len(a) != 2 or any(len(row) != 2 for row in a):
            raise ValueError("pixel_to_stage must be a 2x2 matrix")
        values = (
            [v for row in a for v in row]
            + list(self.reference_pixel)
            + list(self.camera_stage_xy_um)
            + list(self.roi_origin)
            + list(self.binning)
        )
        if not all(map(_finite, values)):
            raise ValueError("calibration values must be finite")
        if abs(a[0][0] * a[1][1] - a[0][1] * a[1][0]) < 1e-15:
            raise ValueError("pixel_to_stage matrix must be invertible")
        if min(self.binning) <= 0:
            raise ValueError("binning values must be positive")

    def image_to_stage(
        self,
        x_pixel: float,
        y_pixel: float,
        stage_xy_um: tuple[float, float] | None = None,
    ) -> tuple[float, float]:
        """Map an image pixel coordinate to stage micrometres."""
        # Delivered-pixel centers map to the center of their native-pixel bin.
        sensor_x = (
            self.roi_origin[0] + x_pixel * self.binning[0] + (self.binning[0] - 1) / 2
        )
        sensor_y = (
            self.roi_origin[1] + y_pixel * self.binning[1] + (self.binning[1] - 1) / 2
        )
        dx, dy = sensor_x - self.reference_pixel[0], sensor_y - self.reference_pixel[1]
        origin = stage_xy_um or self.camera_stage_xy_um
        return (
            origin[0] + self.pixel_to_stage[0][0] * dx + self.pixel_to_stage[0][1] * dy,
            origin[1] + self.pixel_to_stage[1][0] * dx + self.pixel_to_stage[1][1] * dy,
        )

    def image_extent_um(self, shape_yx: tuple[int, int]) -> tuple[float, float]:
        """Return the stage-space width and height covered by an image shape."""
        h, w = shape_yx
        if h <= 0 or w <= 0:
            raise ValueError("image dimensions must be positive")
        # Tile planning uses an axis-aligned rectangle. Shrink the transformed
        # field to the largest similar rectangle guaranteed to fit inside it.
        m = (
            (
                self.pixel_to_stage[0][0] * self.binning[0],
                self.pixel_to_stage[0][1] * self.binning[1],
            ),
            (
                self.pixel_to_stage[1][0] * self.binning[0],
                self.pixel_to_stage[1][1] * self.binning[1],
            ),
        )
        det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
        inv = ((m[1][1] / det, -m[0][1] / det), (-m[1][0] / det, m[0][0] / det))
        bx, by = _axis_lengths(self, w, h)
        need_x = abs(inv[0][0]) * bx / 2 + abs(inv[0][1]) * by / 2
        need_y = abs(inv[1][0]) * bx / 2 + abs(inv[1][1]) * by / 2
        scale = min(1.0, (w / 2) / need_x, (h / 2) / need_y)
        return bx * scale, by * scale


def _axis_lengths(
    calibration: Calibration, width: int, height: int
) -> tuple[float, float]:
    m = calibration.pixel_to_stage
    bx, by = calibration.binning
    return (
        abs(m[0][0]) * bx * width + abs(m[0][1]) * by * height,
        abs(m[1][0]) * bx * width + abs(m[1][1]) * by * height,
    )


def _finite(value: float) -> bool:
    return isfinite(value)


@dataclass(frozen=True)
class Region:
    """Axis-aligned stage-space rectangle bounding the searched area."""

    x_min_um: float
    y_min_um: float
    x_max_um: float
    y_max_um: float

    def __post_init__(self) -> None:
        """Check the declared fields for consistency."""
        if not all(
            map(_finite, (self.x_min_um, self.y_min_um, self.x_max_um, self.y_max_um))
        ):
            raise ValueError("region bounds must be finite")
        if self.x_max_um <= self.x_min_um or self.y_max_um <= self.y_min_um:
            raise ValueError("region must have positive width and height")

    def contains(self, x_um: float, y_um: float) -> bool:
        """Return whether a stage position falls inside this region."""
        return (
            self.x_min_um <= x_um <= self.x_max_um
            and self.y_min_um <= y_um <= self.y_max_um
        )


@dataclass(frozen=True)
class Tile:
    """One image tile with its grid position and stage position."""

    tile_id: str
    row: int
    column: int
    x_um: float
    y_um: float


def plan_tiles(
    region: Region, field_xy_um: tuple[float, float], overlap: float
) -> tuple[Tile, ...]:
    """Cover a rectangle with overlapping fields, in deterministic serpentine order."""
    fw, fh = field_xy_um
    if not all(map(_finite, (fw, fh, overlap))) or fw <= 0 or fh <= 0:
        raise ValueError("field dimensions must be finite and positive")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")

    def centers(low: float, high: float, field: float) -> list[float]:
        span = high - low
        if span <= field:
            return [(low + high) / 2]
        n = ceil((span - field) / (field * (1 - overlap))) + 1
        first, last = low + field / 2, high - field / 2
        return [first + (last - first) * i / (n - 1) for i in range(n)]

    xs, ys = (
        centers(region.x_min_um, region.x_max_um, fw),
        centers(region.y_min_um, region.y_max_um, fh),
    )
    tiles: list[Tile] = []
    for row, y in enumerate(ys):
        cols = list(enumerate(xs))
        if row % 2:
            cols.reverse()
        for col, x in cols:
            tiles.append(Tile(f"tile-r{row:02d}-c{col:02d}", row, col, x, y))
    return tuple(tiles)


def deduplicate_targets(
    observations: Iterable[SearchObservation],
    calibration: Calibration,
    *,
    distance_um: float,
    region: Region | None = None,
) -> tuple[Target, ...]:
    """Greedy nearest-centroid clustering that avoids transitive chain merging."""
    if distance_um < 0 or not _finite(distance_um):
        raise ValueError("distance_um must be finite and nonnegative")
    points: list[tuple[float, float, SearchObservation, object]] = []
    expected_context: tuple[str, str, str] | None = None
    for obs in observations:
        context = (obs.specimen_id, obs.run_id, obs.coordinate_frame)
        if expected_context is None:
            expected_context = context
        if context != expected_context or obs.phase.value != "detection":
            raise ValueError(
                "observations must share specimen/run/frame and be detection-phase records"
            )
        if obs.missing or obs.value is None:
            continue
        if (
            obs.calibration_id != calibration.calibration_id
            or obs.coordinate_frame != calibration.coordinate_frame
        ):
            raise ValueError("observation calibration/frame does not match")
        objects = obs.value if isinstance(obs.value, (tuple, list)) else (obs.value,)
        for obj in objects:
            if not hasattr(obj, "x_pixel") or not hasattr(obj, "y_pixel"):
                raise TypeError(
                    "observation values must contain objects with x_pixel and y_pixel"
                )
            if not _finite(obj.x_pixel) or not _finite(obj.y_pixel):
                raise ValueError("detection pixel coordinates must be finite")
            if obs.image_shape_yx is not None:
                h, w = obs.image_shape_yx
                if not (0 <= obj.x_pixel < w and 0 <= obj.y_pixel < h):
                    continue
            x, y = calibration.image_to_stage(obj.x_pixel, obj.y_pixel, obs.stage_xy_um)
            if region is not None and not region.contains(x, y):
                continue
            bounds = getattr(obj, "bounds_xyxy_pixel", None)
            touches_edge = bool(getattr(obj, "touches_edge", False))
            if region is not None and bounds is not None:
                x0, y0, x1, y1 = bounds
                # Component bounds are half-open indices; Calibration accepts
                # pixel centres, so the enclosing pixel edges are offset by 1/2.
                corners = [
                    calibration.image_to_stage(px, py, obs.stage_xy_um)
                    for px, py in (
                        (x0 - 0.5, y0 - 0.5),
                        (x1 - 0.5, y0 - 0.5),
                        (x0 - 0.5, y1 - 0.5),
                        (x1 - 0.5, y1 - 0.5),
                    )
                ]
                if not all(region.contains(px, py) for px, py in corners):
                    continue
            if region is not None and touches_edge:
                edge_margin = distance_um
                if (
                    min(
                        x - region.x_min_um,
                        region.x_max_um - x,
                        y - region.y_min_um,
                        region.y_max_um - y,
                    )
                    <= edge_margin
                ):
                    continue
            points.append((x, y, obs, obj))
    clusters: list[list[tuple[float, float, SearchObservation, object]]] = []
    for point in points:
        candidates = []
        for i, cluster in enumerate(clusters):
            if any(p[2].tile_id == point[2].tile_id for p in cluster):
                continue
            cx = sum(p[0] for p in cluster) / len(cluster)
            cy = sum(p[1] for p in cluster) / len(cluster)
            distance = hypot(point[0] - cx, point[1] - cy)
            if distance <= distance_um and all(
                hypot(point[0] - p[0], point[1] - p[1]) <= distance_um for p in cluster
            ):
                candidates.append((distance, i))
        if candidates:
            clusters[min(candidates)[1]].append(point)
        else:
            clusters.append([point])
    targets = []
    for cluster in clusters:
        x = sum(p[0] for p in cluster) / len(cluster)
        y = sum(p[1] for p in cluster) / len(cluster)
        evidence_list = []
        seen = set()
        for p in cluster:
            if p[2].request_id not in seen:
                seen.add(p[2].request_id)
                evidence_list.append(p[2])
        evidence = tuple(evidence_list)
        scores = [getattr(p[3], "score", None) for p in cluster]
        score_values = [s for s in scores if s is not None]
        target_id = str(
            uuid5(
                NAMESPACE_URL,
                f"planner:{cluster[0][2].specimen_id}:{calibration.calibration_id}:{x:.6f}:{y:.6f}:"
                + ",".join(sorted(seen)),
            )
        )
        targets.append(
            Target(
                target_id,
                cluster[0][2].specimen_id,
                calibration.coordinate_frame,
                calibration.calibration_id,
                x,
                y,
                evidence,
                max(score_values) if score_values else None,
            )
        )
    return tuple(targets)
