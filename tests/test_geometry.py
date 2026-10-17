from datetime import datetime, timezone
from itertools import pairwise

import pytest

from planner import (
    Calibration,
    DetectedObject,
    Phase,
    Region,
    SearchObservation,
    deduplicate_targets,
    plan_tiles,
)


def test_affine_mapping_handles_rotation_reflection_roi_and_binning() -> None:
    calibration = Calibration(
        "cal-1",
        "stage-xy",
        ((0.0, -0.5), (-0.5, 0.0)),
        (100.0, 200.0),
        (10.0, 20.0),
        (80.0, 180.0),
        (2.0, 2.0),
    )
    assert calibration.image_to_stage(10, 5) == (14.75, 19.75)
    assert calibration.image_extent_um((10, 20)) == (10.0, 20.0)
    with pytest.raises(ValueError, match="invertible"):
        Calibration("bad", "stage-xy", ((1.0, 2.0), (2.0, 4.0)), (0, 0), (0, 0))


def test_tile_plan_has_edge_coverage_overlap_and_serpentine_order() -> None:
    tiles = plan_tiles(Region(0, 0, 39, 25), (12, 10), 0.25)
    assert len(tiles) == 4 * 3
    assert [t.x_um for t in tiles[:4]] == sorted(t.x_um for t in tiles[:4])
    assert [t.x_um for t in tiles[4:8]] == sorted(
        (t.x_um for t in tiles[4:8]), reverse=True
    )
    xs = sorted({t.x_um for t in tiles})
    ys = sorted({t.y_um for t in tiles})
    assert xs[0] - 6 <= 0 and xs[-1] + 6 >= 39
    assert ys[0] - 5 <= 0 and ys[-1] + 5 >= 25
    assert max(b - a for a, b in pairwise(xs)) <= 9
    assert max(b - a for a, b in pairwise(ys)) <= 7.5


def test_dedup_merges_overlap_observations_without_transitive_chain_merge() -> None:
    calibration = Calibration("c", "f", ((1.0, 0.0), (0.0, 1.0)), (0, 0), (0, 0))
    now = datetime.now(timezone.utc)

    def obs(tile: str, x: float) -> SearchObservation:
        return SearchObservation(
            "specimen",
            Phase.DETECTION,
            "f",
            "c",
            now,
            now,
            (0, 0),
            tile,
            tile,
            value=(DetectedObject(x, 0, 10),),
        )

    targets = deduplicate_targets(
        [obs("a", 0), obs("b", 1), obs("c", 3)], calibration, distance_um=2
    )
    assert len(targets) == 2
    assert targets[0].x_um == 0.5
    assert len(targets[0].observations) == 2


def test_each_tile_stage_readback_anchors_its_image_transform() -> None:
    calibration = Calibration("c", "f", ((1.0, 0.0), (0.0, 1.0)), (5, 5), (100, 200))
    now = datetime.now(timezone.utc)
    first = SearchObservation(
        "s",
        Phase.DETECTION,
        "f",
        "c",
        now,
        now,
        (10, 20),
        "a",
        "a",
        value=(DetectedObject(5, 5),),
    )
    second = SearchObservation(
        "s",
        Phase.DETECTION,
        "f",
        "c",
        now,
        now,
        (20, 30),
        "b",
        "b",
        value=(DetectedObject(5, 5),),
    )
    targets = deduplicate_targets([first, second], calibration, distance_um=0)
    assert len(targets) == 2
    assert [(t.x_um, t.y_um) for t in targets] == [(10, 20), (20, 30)]


def test_close_distinct_same_tile_objects_keep_independent_scores() -> None:
    calibration = Calibration("c", "f", ((1.0, 0.0), (0.0, 1.0)), (0, 0), (0, 0))
    now = datetime.now(timezone.utc)
    obs = SearchObservation(
        "s",
        Phase.DETECTION,
        "f",
        "c",
        now,
        now,
        (0, 0),
        "tile",
        "r",
        value=(DetectedObject(0, 0, 100), DetectedObject(1, 0, 1)),
    )
    targets = deduplicate_targets([obs], calibration, distance_um=5)
    assert len(targets) == 2
    assert [target.score for target in targets] == [100, 1]


@pytest.mark.parametrize(
    "matrix", [((1, 0), (0, 1)), ((-1, 0), (0, 1)), ((0, -1), (1, 0))]
)
def test_half_open_component_bounds_map_pixel_edges_under_affine(
    matrix: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    calibration = Calibration("c", "f", matrix, (0, 0), (0, 0))
    now = datetime.now(timezone.utc)
    obs = SearchObservation(
        "s",
        Phase.DETECTION,
        "f",
        "c",
        now,
        now,
        (0, 0),
        "tile",
        "r",
        value=(DetectedObject(0, 0, bounds_xyxy_pixel=(0, 0, 1, 1)),),
    )
    targets = deduplicate_targets(
        [obs], calibration, distance_um=0, region=Region(-0.5, -0.5, 0.5, 0.5)
    )
    assert len(targets) == 1
    assert (targets[0].x_um, targets[0].y_um) == (0, 0)
