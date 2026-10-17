"""Detect bright objects in a synthetic image and place them on the stage.

Runs on `numpy` alone. A host supplies the pixels, from any camera API it likes,
and keeps the device calls; this module only computes.
"""

from __future__ import annotations

import numpy as np

from planner import Calibration
from planner.detection import DetectionConfig, detect_bright_objects

# A synthetic frame: two bright discs on a dim, slightly noisy background.
rng = np.random.default_rng(20261017)
image = rng.normal(200.0, 10.0, size=(96, 128))
for center_y, center_x, amplitude in ((30, 40, 4000.0), (70, 96, 2500.0)):
    yy, xx = np.ogrid[:96, :128]
    disc = (yy - center_y) ** 2 + (xx - center_x) ** 2 <= 16
    image[disc] += amplitude

config = DetectionConfig(threshold=None, min_area_px=4)  # None selects Otsu
detections = detect_bright_objects(image, config)
print("detections", len(detections))

calibration = Calibration(
    calibration_id="demo-calibration",
    coordinate_frame="stage_xy_um",
    pixel_to_stage=((1.0, 0.0), (0.0, 1.0)),
    reference_pixel=(0.0, 0.0),
    camera_stage_xy_um=(1000.0, 500.0),  # where this frame was taken
)
for detection in detections:
    x_um, y_um = calibration.image_to_stage(detection.x_pixel, detection.y_pixel)
    print(
        f"  pixel ({detection.x_pixel:.1f}, {detection.y_pixel:.1f}) "
        f"area {detection.area_px} -> stage ({x_um:.1f}, {y_um:.1f}) um"
    )
