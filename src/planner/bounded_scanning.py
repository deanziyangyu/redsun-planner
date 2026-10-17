"""Pure, operator-confirmed corner-bound scan geometry in stage micrometres.

Plans constrain the complete axis-aligned image footprint to the confirmed
convex boundary. They describe geometry only and never issue device commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite

Point = tuple[float, float]
MAX_PLAN_POSITIONS = 10000


def _point(value: Point, label: str) -> Point:
    if len(value) != 2 or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v)
        for v in value
    ):
        raise ValueError(f"{label} must be a finite XY pair in micrometres")
    return float(value[0]), float(value[1])


@dataclass(frozen=True)
class CornerObservation:
    """An operator-confirmed corner and its measured stage readback."""

    corner_xy_um: Point
    confirmed: bool
    confirmation_id: str
    readback_xy_um: Point

    def __post_init__(self) -> None:
        """Check the declared fields for consistency."""
        object.__setattr__(
            self, "corner_xy_um", _point(self.corner_xy_um, "corner_xy_um")
        )
        object.__setattr__(
            self, "readback_xy_um", _point(self.readback_xy_um, "readback_xy_um")
        )
        if (
            not self.confirmed
            or not isinstance(self.confirmation_id, str)
            or not self.confirmation_id.strip()
        ):
            raise ValueError(
                "corner observation requires explicit confirmation identity"
            )


def _area(poly: tuple[Point, ...]) -> float:
    return (
        sum(
            poly[i][0] * poly[(i + 1) % len(poly)][1]
            - poly[(i + 1) % len(poly)][0] * poly[i][1]
            for i in range(len(poly))
        )
        / 2
    )


def _cross(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])


@dataclass(frozen=True)
class ConfirmedBoundary:
    """Confirmed axis-aligned rectangle or four-corner convex polygon."""

    corners: tuple[CornerObservation, ...]
    vertices_xy_um: tuple[Point, ...]
    coordinate_frame: str
    boundary_id: str
    kind: str

    def __post_init__(self) -> None:
        """Check the declared fields for consistency."""
        if not self.coordinate_frame or not self.boundary_id:
            raise ValueError("coordinate_frame and boundary_id are required")
        if self.kind not in ("axis_aligned_rectangle", "convex_polygon"):
            raise ValueError("kind must be axis_aligned_rectangle or convex_polygon")
        expected_count = 2 if self.kind == "axis_aligned_rectangle" else 4
        if len(self.vertices_xy_um) != 4 or len(self.corners) != expected_count:
            raise ValueError(
                "boundary requires two opposite rectangle corners or four polygon corners"
            )
        vertices = tuple(_point(p, "boundary vertex") for p in self.vertices_xy_um)
        if any(not isinstance(c, CornerObservation) for c in self.corners):
            raise TypeError("boundary corners must be CornerObservation values")
        if self.kind == "axis_aligned_rectangle":
            c0, c1 = self.corners
            x0, y0 = c0.readback_xy_um
            x1, y1 = c1.readback_xy_um
            if abs(x1 - x0) <= 1e-12 or abs(y1 - y0) <= 1e-12:
                raise ValueError(
                    "opposite readback corners must define nonzero width and height"
                )
            expected = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
            if vertices != expected:
                raise ValueError(
                    "rectangle vertices must be derived from the confirmed readbacks"
                )
        else:
            if len(set(vertices)) != 4 or abs(_area(vertices)) <= 1e-12:
                raise ValueError(
                    "boundary corners must be distinct and have nonzero area"
                )
            if vertices != tuple(c.readback_xy_um for c in self.corners):
                raise ValueError(
                    "polygon vertices must match the ordered confirmed readbacks"
                )
        turns = [
            _cross(vertices[i], vertices[(i + 1) % 4], vertices[(i + 2) % 4])
            for i in range(4)
        ]
        if (
            len(set(vertices)) != 4
            or any(abs(t) <= 1e-12 for t in turns)
            or not (all(t > 0 for t in turns) or all(t < 0 for t in turns))
        ):
            raise ValueError(
                "four polygon corners must be ordered, strictly convex, and non-self-intersecting"
            )
        object.__setattr__(self, "vertices_xy_um", vertices)

    @classmethod
    def from_opposite_corners(
        cls,
        first: CornerObservation,
        second: CornerObservation,
        *,
        coordinate_frame: str,
        boundary_id: str,
    ) -> ConfirmedBoundary:
        """Build a boundary from two confirmed opposite rectangle corners."""
        if not isinstance(first, CornerObservation) or not isinstance(
            second, CornerObservation
        ):
            raise TypeError("confirmed corner observations are required")
        x0, y0 = first.readback_xy_um
        x1, y1 = second.readback_xy_um
        if abs(x1 - x0) <= 1e-12 or abs(y1 - y0) <= 1e-12:
            raise ValueError(
                "opposite rectangle corners must define nonzero width and height"
            )
        vertices = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
        return cls(
            (first, second),
            vertices,
            coordinate_frame,
            boundary_id,
            "axis_aligned_rectangle",
        )

    @classmethod
    def from_polygon(
        cls,
        corners: tuple[CornerObservation, ...],
        *,
        coordinate_frame: str,
        boundary_id: str,
    ) -> ConfirmedBoundary:
        """Build a boundary from four confirmed, ordered convex corners."""
        if not isinstance(corners, tuple) or len(corners) != 4:
            raise ValueError(
                "polygon requires exactly four ordered corner observations"
            )
        return cls(
            corners,
            tuple(c.readback_xy_um for c in corners),
            coordinate_frame,
            boundary_id,
            "convex_polygon",
        )


def _clip(
    poly: tuple[Point, ...], a: Point, b: Point, offset: float
) -> tuple[Point, ...]:
    """Clip to the left of directed edge, shifted inward by offset."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = (dx * dx + dy * dy) ** 0.5
    nx, ny = -dy / length, dx / length
    threshold = nx * a[0] + ny * a[1] + offset
    result: list[Point] = []
    for p, q in zip(poly, poly[1:] + poly[:1]):
        vp = nx * p[0] + ny * p[1] - threshold
        vq = nx * q[0] + ny * q[1] - threshold
        pin, qin = vp >= -1e-10, vq >= -1e-10
        if pin:
            result.append(p)
        if pin != qin:
            t = vp / (vp - vq)
            result.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return tuple(result)


def _feasible(
    boundary: ConfirmedBoundary, footprint: tuple[float, float]
) -> tuple[Point, ...]:
    poly = boundary.vertices_xy_um
    if boundary.kind == "axis_aligned_rectangle":
        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
        xmin, xmax = min(xs) + footprint[0] / 2, max(xs) - footprint[0] / 2
        ymin, ymax = min(ys) + footprint[1] / 2, max(ys) - footprint[1] / 2
        if xmin > xmax + 1e-10 or ymin > ymax + 1e-10:
            raise ValueError(
                "camera footprint does not fit wholly inside confirmed boundary"
            )
        if abs(xmin - xmax) <= 1e-10:
            xmin = xmax = (xmin + xmax) / 2
        if abs(ymin - ymax) <= 1e-10:
            ymin = ymax = (ymin + ymax) / 2
        return ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax))
    area = _area(poly)
    if area < 0:
        poly = tuple(reversed(poly))
    hx, hy = footprint[0] / 2, footprint[1] / 2
    for a, b in zip(poly, poly[1:] + poly[:1]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        norm = (dx * dx + dy * dy) ** 0.5
        # The support function of an axis-aligned rectangular footprint.
        offset = abs(dy / norm) * hx + abs(dx / norm) * hy
        poly = _clip(poly, a, b, offset)
        if len(poly) < 3 or abs(_area(poly)) <= 1e-10:
            raise ValueError(
                "camera footprint does not fit wholly inside confirmed boundary"
            )
    return poly


def _horizontal_interval(
    poly: tuple[Point, ...], y: float
) -> tuple[float, float] | None:
    xs: list[float] = []
    for a, b in zip(poly, poly[1:] + poly[:1]):
        if min(a[1], b[1]) - 1e-10 <= y <= max(a[1], b[1]) + 1e-10:
            if abs(b[1] - a[1]) <= 1e-12:
                xs.extend((a[0], b[0]))
            else:
                t = (y - a[1]) / (b[1] - a[1])
                if -1e-10 <= t <= 1 + 1e-10:
                    xs.append(a[0] + t * (b[0] - a[0]))
    return (min(xs), max(xs)) if xs else None


def _rows(
    poly: tuple[Point, ...], spacing: float
) -> tuple[tuple[float, float, float], ...]:
    ymin, ymax = min(p[1] for p in poly), max(p[1] for p in poly)
    count = max(1, ceil((ymax - ymin) / spacing) + 1)
    if count > MAX_PLAN_POSITIONS:
        raise ValueError(f"scan exceeds {MAX_PLAN_POSITIONS} row/position limit")
    ys = (
        [ymin + (ymax - ymin) * i / (count - 1) for i in range(count)]
        if count > 1
        else [(ymin + ymax) / 2]
    )
    rows: list[tuple[float, float, float]] = []
    for y in ys:
        interval = _horizontal_interval(poly, y)
        if interval is not None:
            rows.append((y, interval[0], interval[1]))
    return tuple(rows)


@dataclass(frozen=True)
class SnapPoint:
    """One snap-scan center with its row and column in the grid."""

    position_id: str
    x_um: float
    y_um: float
    row: int
    column: int


@dataclass(frozen=True)
class SnapScanPlan:
    """Planned serpentine snap centers inside a confirmed operator boundary."""

    boundary_id: str
    coordinate_frame: str
    footprint_xy_um: tuple[float, float]
    overlap: float
    positions: tuple[SnapPoint, ...]
    travel_xy_um: tuple[Point, ...]
    center_boundary_xy_um: tuple[Point, ...]
    whole_footprint_inside: bool = True
    coverage_claim: str = "overlapping snap footprints lie inside the confirmed boundary; polygon edge fill is not claimed"

    @property
    def required_capabilities(self) -> frozenset[str]:
        """Return the device capabilities a host must provide for this plan."""
        return frozenset(("stage.xy", "camera.snap"))


@dataclass(frozen=True)
class BroomStrip:
    """One broom strip: its row plus start and end stage positions."""

    row: int
    start_xy_um: Point
    end_xy_um: Point


@dataclass(frozen=True)
class BroomScanPlan:
    """Planned parallel broom strips inside a confirmed operator boundary."""

    boundary_id: str
    coordinate_frame: str
    footprint_xy_um: tuple[float, float]
    overlap: float
    strips: tuple[BroomStrip, ...]
    travel_xy_um: tuple[Point, ...]
    center_boundary_xy_um: tuple[Point, ...]
    whole_footprint_inside: bool = True
    coverage_claim: str = "continuous strip centerlines remain inside the eroded boundary; polygon edge fill is not claimed"

    @property
    def required_capabilities(self) -> frozenset[str]:
        """Return the device capabilities a host must provide for this plan."""
        return frozenset(("stage.xy", "camera.continuous"))


def _inputs(
    boundary: ConfirmedBoundary, footprint_xy_um: tuple[float, float], overlap: float
) -> tuple[tuple[float, float], tuple[Point, ...]]:
    if not isinstance(boundary, ConfirmedBoundary):
        raise TypeError("a ConfirmedBoundary is required")
    if len(footprint_xy_um) != 2 or any(
        isinstance(v, bool)
        or not isinstance(v, (int, float))
        or not isfinite(v)
        or v <= 0
        for v in footprint_xy_um
    ):
        raise ValueError("footprint dimensions must be finite positive micrometres")
    if not isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError("overlap must be finite and in [0, 1)")
    footprint = float(footprint_xy_um[0]), float(footprint_xy_um[1])
    return footprint, _feasible(boundary, footprint)


def plan_snap_scan(
    boundary: ConfirmedBoundary, footprint_xy_um: tuple[float, float], overlap: float
) -> SnapScanPlan:
    """Plan deterministic snake snaps; every full image footprint stays inside."""
    footprint, poly = _inputs(boundary, footprint_xy_um, overlap)
    positions: list[SnapPoint] = []
    travel: list[Point] = []
    for row, (y, left, right) in enumerate(_rows(poly, footprint[1] * (1 - overlap))):
        n = max(1, ceil((right - left) / (footprint[0] * (1 - overlap))) + 1)
        if len(positions) + n > MAX_PLAN_POSITIONS:
            raise ValueError(f"scan exceeds {MAX_PLAN_POSITIONS} position limit")
        xs = (
            [left + (right - left) * i / (n - 1) for i in range(n)]
            if n > 1
            else [(left + right) / 2]
        )
        if row % 2:
            xs.reverse()
        for col, x in enumerate(xs):
            p = SnapPoint(f"snap-r{row:03d}-c{col:03d}", x, y, row, col)
            positions.append(p)
            travel.append((x, y))
    return SnapScanPlan(
        boundary.boundary_id,
        boundary.coordinate_frame,
        footprint,
        overlap,
        tuple(positions),
        tuple(travel),
        poly,
    )


def plan_broom_scan(
    boundary: ConfirmedBoundary,
    footprint_xy_um: tuple[float, float],
    overlap: float,
    *,
    serpentine: bool = True,
) -> BroomScanPlan:
    """Plan parallel strips; serpentine alternates direction, false uses same-direction passes."""
    if not isinstance(serpentine, bool):
        raise TypeError("serpentine must be boolean")
    footprint, poly = _inputs(boundary, footprint_xy_um, overlap)
    strips: list[BroomStrip] = []
    travel: list[Point] = []
    rows = _rows(poly, footprint[1] * (1 - overlap))
    if len(rows) > MAX_PLAN_POSITIONS:
        raise ValueError(f"scan exceeds {MAX_PLAN_POSITIONS} strip limit")
    for row, (y, left, right) in enumerate(rows):
        reverse = bool(row % 2) if serpentine else False
        start, end = ((right, y), (left, y)) if reverse else ((left, y), (right, y))
        strips.append(BroomStrip(row, start, end))
        if not travel:
            travel.append(start)
        else:
            # Convexity guarantees the straight connector between adjacent strips stays inside.
            travel.append(start)
        travel.append(end)
    return BroomScanPlan(
        boundary.boundary_id,
        boundary.coordinate_frame,
        footprint,
        overlap,
        tuple(strips),
        tuple(travel),
        poly,
    )
