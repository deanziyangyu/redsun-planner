"""Small, dependency-light bright-object detector for survey images.

This is an original implementation of a common threshold-and-components method.
It does not execute or translate AutoScanJ macros.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING

import numpy as np

from .contracts import DetectedObject

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class DetectionConfig:
    """Threshold and component-size settings, expressed in input pixel units."""

    threshold: float | None = None
    min_area_px: int = 1
    max_area_px: int | None = None
    connectivity: int = 8
    channel: int = 0

    def __post_init__(self) -> None:
        """Check the declared fields for consistency."""
        if self.threshold is not None and not isfinite(self.threshold):
            raise ValueError("threshold must be finite when supplied")
        if self.min_area_px < 1:
            raise ValueError("min_area_px must be at least 1")
        if self.max_area_px is not None and self.max_area_px < self.min_area_px:
            raise ValueError("max_area_px must be greater than or equal to min_area_px")
        if self.connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")
        if self.channel < 0:
            raise ValueError("channel must be non-negative")


@dataclass(frozen=True)
class Detection:
    """One connected bright region; coordinates use image XY pixel convention."""

    x_pixel: float
    y_pixel: float
    area_px: int
    mean_intensity: float
    max_intensity: float
    bounds_xyxy_pixel: tuple[int, int, int, int]
    touches_edge: bool


def _gray_plane(image: ArrayLike, channel: int) -> NDArray[np.float64]:
    data = np.asarray(image)
    if data.ndim == 3:
        if channel >= data.shape[-1]:
            raise ValueError(
                f"channel {channel} is outside image channel count {data.shape[-1]}"
            )
        data = data[..., channel]
    if data.ndim != 2:
        raise ValueError("image must be a 2-D plane or a YXC image")
    if data.size == 0 or not np.isfinite(data).all():
        raise ValueError("image must be non-empty and contain only finite values")
    return np.asarray(data, dtype=np.float64)


def _otsu_threshold(image: NDArray[np.float64]) -> float:
    """Compute Otsu's between-class variance threshold from at most 4096 bins."""
    low, high = float(image.min()), float(image.max())
    if low == high:
        return low
    hist, edges = np.histogram(
        image, bins=min(4096, max(256, int(np.sqrt(image.size)))), range=(low, high)
    )
    prob = hist.astype(np.float64)
    prob /= prob.sum()
    centers = (edges[:-1] + edges[1:]) * 0.5
    weight = np.cumsum(prob)
    mean = np.cumsum(prob * centers)
    total_mean = mean[-1]
    denom = weight * (1.0 - weight)
    variance = np.zeros_like(denom)
    valid = denom > 0
    variance[valid] = (total_mean * weight[valid] - mean[valid]) ** 2 / denom[valid]
    return float(centers[int(np.argmax(variance))])


def detect_bright_objects(
    image: ArrayLike, config: DetectionConfig | None = None
) -> list[Detection]:
    """Threshold an image and return 8- or 4-connected bright components.

    By default Otsu's global threshold is used. A supplied threshold selects pixels
    strictly greater than that value. Centroids are intensity-weighted; bounding
    boxes are half-open ``(x0, y0, x1, y1)`` pixel coordinates.
    """
    settings = config or DetectionConfig()
    plane = _gray_plane(image, settings.channel)
    threshold = (
        _otsu_threshold(plane) if settings.threshold is None else settings.threshold
    )
    mask = plane > threshold
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=np.bool_)
    offsets: tuple[tuple[int, int], ...]
    if settings.connectivity == 8:
        offsets = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    else:
        offsets = ((-1, 0), (0, -1), (0, 1), (1, 0))

    found: list[Detection] = []
    for row, col in zip(*np.nonzero(mask & ~visited), strict=True):
        if visited[row, col]:
            continue
        visited[row, col] = True
        queue = deque([(int(row), int(col))])
        pixels: list[tuple[int, int]] = []
        while queue:
            y, x = queue.popleft()
            pixels.append((y, x))
            for dy, dx in offsets:
                ny, nx = y + dy, x + dx
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and mask[ny, nx]
                    and not visited[ny, nx]
                ):
                    visited[ny, nx] = True
                    queue.append((ny, nx))
        area = len(pixels)
        if area < settings.min_area_px or (
            settings.max_area_px is not None and area > settings.max_area_px
        ):
            continue
        ys = np.fromiter((p[0] for p in pixels), dtype=np.intp)
        xs = np.fromiter((p[1] for p in pixels), dtype=np.intp)
        values = plane[ys, xs]
        weights = values - threshold
        if float(weights.sum()) <= 0:
            weights = np.ones(area, dtype=np.float64)
        found.append(
            Detection(
                x_pixel=float(np.average(xs, weights=weights)),
                y_pixel=float(np.average(ys, weights=weights)),
                area_px=area,
                mean_intensity=float(values.mean()),
                max_intensity=float(values.max()),
                bounds_xyxy_pixel=(
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max()) + 1,
                    int(ys.max()) + 1,
                ),
                touches_edge=bool(
                    (xs == 0).any()
                    or (ys == 0).any()
                    or (xs == width - 1).any()
                    or (ys == height - 1).any()
                ),
            )
        )
    return found


def make_detector(
    config: DetectionConfig | None = None,
) -> Callable[[ArrayLike, object], list[DetectedObject]]:
    """Return a workflow-compatible ``(image, observation)`` detector callable.

    Observation metadata is intentionally not interpreted by the image detector;
    the workflow keeps it alongside each detection. Core objects retain XY pixel
    coordinates and half-open bounds. The intensity peak is used as the score.
    """

    def detector(image: ArrayLike, observation: object) -> list[DetectedObject]:
        del observation
        return [
            DetectedObject(
                x_pixel=item.x_pixel,
                y_pixel=item.y_pixel,
                score=item.max_intensity,
                bounds_xyxy_pixel=(
                    float(item.bounds_xyxy_pixel[0]),
                    float(item.bounds_xyxy_pixel[1]),
                    float(item.bounds_xyxy_pixel[2]),
                    float(item.bounds_xyxy_pixel[3]),
                ),
                touches_edge=item.touches_edge,
            )
            for item in detect_bright_objects(image, config)
        ]

    return detector
