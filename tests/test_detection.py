from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from planner.detection import DetectionConfig, detect_bright_objects, make_detector

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def test_returns_weighted_xy_centroid_bbox_area_and_edges() -> None:
    image = np.zeros((8, 10), dtype=np.uint8)
    image[2:4, 3:5] = [[10, 20], [30, 40]]
    image[0, 8] = 100

    detections = detect_bright_objects(
        image, DetectionConfig(threshold=0, min_area_px=2)
    )

    assert len(detections) == 1
    item = detections[0]
    assert item.area_px == 4
    assert item.bounds_xyxy_pixel == (3, 2, 5, 4)
    assert item.x_pixel == pytest.approx(3.6)
    assert item.y_pixel == pytest.approx(2.7)
    assert item.mean_intensity == pytest.approx(25)
    assert item.max_intensity == 40
    assert not item.touches_edge


def test_otsu_finds_bright_object_and_edge_flag() -> None:
    image = np.full((12, 12), 4, dtype=np.uint16)
    image[0:3, 8:10] = 1000

    detections = detect_bright_objects(image)

    assert len(detections) == 1
    assert detections[0].area_px == 6
    assert detections[0].touches_edge
    assert detections[0].x_pixel == pytest.approx(8.5)
    assert detections[0].y_pixel == pytest.approx(1)


def test_connectivity_and_size_filter() -> None:
    image = np.zeros((5, 5), dtype=np.uint8)
    image[1, 1] = image[2, 2] = 7
    assert (
        len(detect_bright_objects(image, DetectionConfig(threshold=0, connectivity=4)))
        == 2
    )
    found = detect_bright_objects(image, DetectionConfig(threshold=0, connectivity=8))
    assert len(found) == 1
    assert found[0].area_px == 2
    assert not detect_bright_objects(image, DetectionConfig(threshold=0, min_area_px=3))


@pytest.mark.parametrize(
    "image", [np.empty((0, 4)), np.ones((2, 2, 2, 2)), np.array([[np.nan]])]
)
def test_rejects_invalid_images(image: ArrayLike) -> None:
    with pytest.raises(ValueError):
        detect_bright_objects(image)


def test_validates_configuration() -> None:
    with pytest.raises(ValueError):
        DetectionConfig(min_area_px=0)
    with pytest.raises(ValueError):
        DetectionConfig(min_area_px=4, max_area_px=3)
    with pytest.raises(ValueError):
        DetectionConfig(threshold=float("nan"))


def test_workflow_adapter_maps_xy_score_and_bounds() -> None:
    image = np.zeros((5, 6), dtype=np.uint8)
    image[1:3, 2:4] = 10

    found = make_detector(DetectionConfig(threshold=0))(image, object())

    assert len(found) == 1
    assert (found[0].x_pixel, found[0].y_pixel) == pytest.approx((2.5, 1.5))
    assert found[0].score == 10
    assert found[0].bounds_xyxy_pixel == (2.0, 1.0, 4.0, 3.0)
