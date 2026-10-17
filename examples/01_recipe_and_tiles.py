"""Build a grid recipe, tile a region, and map image pixels to stage micrometres.

Runs on `numpy` alone: nothing here imports a Bluesky or vendor device package.
"""

from __future__ import annotations

from planner import (
    AcquisitionSettings,
    Calibration,
    Region,
    build_grid_recipe,
    plan_tiles,
)

# A recipe is inert data: no device is contacted, no plan is submitted.
recipe = build_grid_recipe(
    origin_xy_um=(0.0, 0.0),
    specimen_id="specimen-1",
    rows=2,
    columns=3,
    step_xy_um=(25.0, 25.0),
    settings=AcquisitionSettings(exposure_ms=50, frames=1),
)
print("recipe", recipe.recipe_id, "positions", len(recipe.positions))
for position in recipe.positions:
    print(f"  {position.position_id}: ({position.x_um}, {position.y_um}) um")

# Tile a searched region with a given field of view and overlap.
region = Region(0.0, 0.0, 100.0, 80.0)
tiles = plan_tiles(region, field_xy_um=(60.0, 40.0), overlap=0.1)
print("tiles", len(tiles), "serpentine order")
for tile in tiles:
    print(
        f"  {tile.tile_id}: row {tile.row} col {tile.column} at ({tile.x_um}, {tile.y_um}) um"
    )

# A calibration turns an image pixel into a stage position. Supply it from your
# own measured hand-eye fit; this one is an identity mapping for illustration.
calibration = Calibration(
    calibration_id="demo-calibration",
    coordinate_frame="stage_xy_um",
    pixel_to_stage=((1.0, 0.0), (0.0, 1.0)),
    reference_pixel=(0.0, 0.0),
    camera_stage_xy_um=(0.0, 0.0),
)
x_um, y_um = calibration.image_to_stage(128.0, 96.0)
print(f"pixel (128, 96) -> stage ({x_um:.1f}, {y_um:.1f}) um")
