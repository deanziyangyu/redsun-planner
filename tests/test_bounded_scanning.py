from __future__ import annotations

import pytest

from planner.bounded_scanning import (
    ConfirmedBoundary,
    CornerObservation,
    plan_broom_scan,
    plan_snap_scan,
)


def corner(x: float, y: float, identity: str | None = None) -> CornerObservation:
    point = (x, y)
    return CornerObservation(point, True, identity or f"confirmed-{x}-{y}", point)


def inside(
    point: tuple[float, float], polygon: tuple[tuple[float, float], ...]
) -> bool:
    signs = []
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        signs.append(
            (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
        )
    return all(v >= -1e-8 for v in signs) or all(v <= 1e-8 for v in signs)


def test_opposite_corners_make_axis_aligned_boundary_independent_of_axis_sign() -> None:
    boundary = ConfirmedBoundary.from_opposite_corners(
        corner(20, 10),
        corner(-20, -10),
        coordinate_frame="stage-xy",
        boundary_id="well-1",
    )
    plan = plan_snap_scan(boundary, (8, 4), 0.25)
    assert plan.positions
    assert plan.boundary_id == "well-1"
    assert plan.whole_footprint_inside
    assert all(-16 - 1e-8 <= p.x_um <= 16 + 1e-8 for p in plan.positions)
    assert all(-8 - 1e-8 <= p.y_um <= 8 + 1e-8 for p in plan.positions)
    assert any(p.x_um < 0 for p in plan.positions) and any(
        p.x_um > 0 for p in plan.positions
    )
    assert "not claimed" in plan.coverage_claim


def test_convex_polygon_broom_clips_strips_and_turnaround_inside_center_region() -> (
    None
):
    corners = tuple(
        corner(*point) for point in ((-20, -10), (20, -10), (10, 10), (-10, 10))
    )
    boundary = ConfirmedBoundary.from_polygon(
        corners, coordinate_frame="stage-xy", boundary_id="poly"
    )
    plan = plan_broom_scan(boundary, (4, 4), 0.5)
    assert len(plan.strips) > 1
    assert plan.travel_xy_um
    assert all(inside(p, boundary.vertices_xy_um) for p in plan.travel_xy_um)
    assert all(inside(p, plan.center_boundary_xy_um) for p in plan.travel_xy_um)
    assert all(strip.start_xy_um[1] == strip.end_xy_um[1] for strip in plan.strips)


def test_polygon_corner_order_must_be_strictly_convex() -> None:
    invalid = tuple(corner(*point) for point in ((0, 0), (2, 2), (0, 2), (2, 0)))
    with pytest.raises(ValueError, match="nonzero area|ordered, strictly convex"):
        ConfirmedBoundary.from_polygon(
            invalid, coordinate_frame="stage-xy", boundary_id="bad"
        )


def test_unconfirmed_or_mismatched_readback_observation_is_rejected() -> None:
    with pytest.raises(ValueError, match="explicit confirmation"):
        CornerObservation((0, 0), False, "", (0, 0))
    # Requested and measured positions remain distinct provenance fields.
    observed = CornerObservation((3, 4), True, "operator-17", (3.2, 4.1))
    assert observed.corner_xy_um == (3.0, 4.0)
    assert observed.readback_xy_um == (3.2, 4.1)
    boundary = ConfirmedBoundary.from_opposite_corners(
        observed, corner(13, 14), coordinate_frame="stage-xy", boundary_id="measured"
    )
    assert boundary.vertices_xy_um[0] == (3.2, 4.1)
    assert boundary.vertices_xy_um[2] == (13.0, 14.0)


def test_direct_boundary_construction_cannot_bypass_readback_or_kind_validation() -> (
    None
):
    first, second = corner(0, 0), corner(10, 10)
    with pytest.raises(ValueError, match="derived from the confirmed readbacks"):
        ConfirmedBoundary(
            (first, second),
            ((-1, 0), (10, 0), (10, 10), (-1, 10)),
            "stage-xy",
            "fake",
            "axis_aligned_rectangle",
        )
    with pytest.raises(ValueError, match="kind must be"):
        ConfirmedBoundary(
            (first, second),
            ((0, 0), (10, 0), (10, 10), (0, 10)),
            "stage-xy",
            "fake",
            "made_up",
        )


def test_footprint_that_cannot_fit_is_rejected() -> None:
    boundary = ConfirmedBoundary.from_opposite_corners(
        corner(0, 0), corner(10, 10), coordinate_frame="stage-xy", boundary_id="small"
    )
    with pytest.raises(ValueError, match="does not fit"):
        plan_snap_scan(boundary, (11, 2), 0.2)


def test_exact_fit_rectangle_produces_single_center_snap_and_broom_strip() -> None:
    boundary = ConfirmedBoundary.from_opposite_corners(
        corner(0, 0), corner(8, 4), coordinate_frame="stage-xy", boundary_id="exact"
    )
    snap = plan_snap_scan(boundary, (8, 4), 0.5)
    broom = plan_broom_scan(boundary, (8, 4), 0.5)
    assert len(snap.positions) == 1
    assert (snap.positions[0].x_um, snap.positions[0].y_um) == (4, 2)
    assert len(broom.strips) == 1
    assert broom.strips[0].start_xy_um == broom.strips[0].end_xy_um == (4, 2)


def test_plan_size_is_bounded_before_allocating_positions() -> None:
    boundary = ConfirmedBoundary.from_opposite_corners(
        corner(0, 0), corner(100, 100), coordinate_frame="stage-xy", boundary_id="large"
    )
    with pytest.raises(ValueError, match="10000"):
        plan_snap_scan(boundary, (1, 1), 0.9999)


def test_broom_supports_same_direction_passes_with_safe_flyback() -> None:
    boundary = ConfirmedBoundary.from_opposite_corners(
        corner(0, 0), corner(20, 12), coordinate_frame="stage-xy", boundary_id="broom"
    )
    plan = plan_broom_scan(boundary, (4, 4), 0.5, serpentine=False)
    assert len(plan.strips) > 1
    assert all(
        a.start_xy_um[0] == b.start_xy_um[0]
        for a, b in zip(plan.strips, plan.strips[1:])
    )
    assert all(inside(point, plan.center_boundary_xy_um) for point in plan.travel_xy_um)


def test_snap_scan_is_deterministic_and_rows_reverse_for_snake_traversal() -> None:
    boundary = ConfirmedBoundary.from_opposite_corners(
        corner(-10, -8), corner(10, 8), coordinate_frame="stage-xy", boundary_id="r"
    )
    first = plan_snap_scan(boundary, (4, 4), 0.5)
    second = plan_snap_scan(boundary, (4, 4), 0.5)
    assert first == second
    rows: dict[int, list[float]] = {}
    for p in first.positions:
        rows.setdefault(p.row, []).append(p.x_um)
    assert rows[0] == sorted(rows[0])
    assert rows[1] == sorted(rows[1], reverse=True)
